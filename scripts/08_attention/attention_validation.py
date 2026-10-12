import subprocess, sys
subprocess.run(['apt-get', '-qq', 'update'], check=True)
subprocess.run(['apt-get', '-qq', 'install', '-y', 'libopenslide0', 'openslide-tools'], check=True)
subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'openslide-python', 'timm'], check=True)
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import gc, json, math, subprocess, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import wilcoxon
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F
import timm
import openslide
from torchvision import transforms
SEED = 2026
rng = np.random.default_rng(SEED)
assert torch.cuda.is_available(), 'Use an A100 GPU runtime.'
ROOT = Path('/content/drive/MyDrive/AACR_GBM_Project')
SELECTED_PATH = ROOT / 'data/metadata/tcga_gbm_notebook02_selected_slides.csv'
MANIFEST_PATH = ROOT / 'outputs/notebook03_cells/full_cohort_200roi/manifests/full_cohort_200roi_manifest.csv.gz'
CELL_DIR = ROOT / 'outputs/notebook03_cells/full_cohort_200roi/tables'
ENCODER_PATH = ROOT / 'outputs/notebook05_ssl_dino/export/gbm_dino_vit_small_encoder.pt'
OUT = ROOT / 'outputs/notebook08_attention_validation'
TABLES = OUT / 'tables'
FIGS = OUT / 'figures'
STATUS = OUT / 'status'
for p in [OUT, TABLES, FIGS, STATUS]:
    p.mkdir(parents=True, exist_ok=True)
TMP = Path('/content/n8_tmp')
TMP.mkdir(exist_ok=True)
PATCH_SIZE = 224
N_TEST_PATIENTS = 20
ROIS_PER_PATIENT = 10
MIN_TYPE_PROB = 0.5
N_SPATIAL_NULLS = 100
INTERFACE_RADIUS_224 = 32.0
GDC_DATA = 'https://api.gdc.cancer.gov/data'
MAX_DOWNLOAD_ATTEMPTS = 4
selected = pd.read_csv(SELECTED_PATH)
selected['patient_id'] = selected['patient_id'].astype(str)
selected['file_id'] = selected['file_id'].astype(str)
selected['split'] = selected['split'].astype(str).str.lower().replace({'validation': 'val'})
samples = pd.read_csv(MANIFEST_PATH)
samples['patient_id'] = samples['patient_id'].astype(str)
samples['file_id'] = samples['file_id'].astype(str)

def choose(df, names, label):
    for c in names:
        if c in df.columns:
            return c
    raise KeyError(f'Missing {label}; tried {names}')
XCOL = choose(samples, ['roi_x_level0', 'x_level0', 'x', 'tile_x', 'x0'], 'level-0 x')
YCOL = choose(samples, ['roi_y_level0', 'y_level0', 'y', 'tile_y', 'y0'], 'level-0 y')
ROICOL = choose(samples, ['roi', 'roi_id', 'master_order', 'tile_index'], 'ROI identity')
DSCOL = next((c for c in ['source_downsample', 'downsample', 'level_downsample'] if c in samples.columns), None)
test = selected[selected['split'] == 'test'].sort_values('patient_id').copy()
assert len(test) == 58
assert (samples[samples['patient_id'].isin(test.patient_id)].groupby('patient_id').size() == 200).all()
chosen_patients = test['patient_id'].tolist()
rng_select = np.random.default_rng(SEED)
chosen_patients = sorted(rng_select.choice(chosen_patients, size=N_TEST_PATIENTS, replace=False).tolist())
selection = []
for pid in chosen_patients:
    rp = CELL_DIR / f'{pid}__roi_features.csv'
    q = pd.read_csv(rp)
    required = ['roi', 'tile_index', 'n_inflammatory', 'total_nuclei']
    missing = [c for c in required if c not in q.columns]
    if missing:
        raise KeyError(f'{rp} missing required columns: {missing}')
    q['inflam_frac_raw'] = (pd.to_numeric(q['n_inflammatory'], errors='coerce') / pd.to_numeric(q['total_nuclei'], errors='coerce').replace(0, np.nan)).fillna(0)
    q['tile_index_key'] = pd.to_numeric(q['tile_index'], errors='raise').astype('int64')
    q = q.sort_values(['inflam_frac_raw', 'tile_index_key']).reset_index(drop=True)
    idx = np.unique(np.linspace(0, len(q) - 1, ROIS_PER_PATIENT).round().astype(int))
    pick = q.iloc[idx].copy()
    if len(pick) < ROIS_PER_PATIENT:
        extra = q[~q['tile_index_key'].isin(pick['tile_index_key'])].head(ROIS_PER_PATIENT - len(pick))
        pick = pd.concat([pick, extra], ignore_index=True)
    for _, r in pick.head(ROIS_PER_PATIENT).iterrows():
        selection.append({'patient_id': pid, 'cell_roi': str(r['roi']), 'tile_index': int(r['tile_index_key']), 'master_order': int(r['master_order']) if 'master_order' in r and pd.notna(r['master_order']) else np.nan, 'inflam_frac_raw': float(r['inflam_frac_raw'])})
sel = pd.DataFrame(selection)
assert len(sel) == N_TEST_PATIENTS * ROIS_PER_PATIENT
assert not sel.duplicated(['patient_id', 'tile_index']).any()
sel.to_csv(TABLES / 'attention_roi_selection.csv', index=False)
bad = []
for pid, g in sel.groupby('patient_id'):
    ps = samples[samples['patient_id'] == pid].copy()
    if 'tile_index' not in ps.columns:
        raise KeyError('Frozen manifest has no tile_index column; cannot safely crosswalk ROI identities.')
    ps['tile_index_key'] = pd.to_numeric(ps['tile_index'], errors='raise').astype('int64')
    hit = ps[ps['tile_index_key'].isin(g['tile_index'])]
    if len(hit) != ROIS_PER_PATIENT or hit['tile_index_key'].nunique() != ROIS_PER_PATIENT:
        bad.append((pid, len(hit), hit['tile_index_key'].nunique()))
assert not bad, f'ROI crosswalk failed: {bad[:5]}'
device = torch.device('cuda')
ck = torch.load(ENCODER_PATH, map_location='cpu')
MODEL_NAME = ck['model_name']
model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0)
model.load_state_dict(ck['encoder_state_dict'], strict=True)
model = model.to(device).eval()
for p in model.parameters():
    p.requires_grad = False
for b in model.blocks:
    if hasattr(b.attn, 'fused_attn'):
        b.attn.fused_attn = False
capture = {}

def qkv_hook(module, inputs, output):
    capture['qkv'] = output.detach()
hook = model.blocks[-1].attn.qkv.register_forward_hook(qkv_hook)
infer_transform = transforms.Compose([transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC), transforms.ToTensor(), transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

def forward_attention(img):
    capture.clear()
    x = infer_transform(img).unsqueeze(0).to(device)
    with torch.inference_mode(), torch.amp.autocast('cuda'):
        _ = model(x)
    qkv = capture['qkv']
    B, N, C3 = qkv.shape
    H = model.blocks[-1].attn.num_heads
    D = C3 // 3 // H
    qkv = qkv.reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
    q, k = (qkv[0], qkv[1])
    attn = q @ k.transpose(-2, -1) * D ** (-0.5)
    attn = attn.softmax(dim=-1)
    cls = attn[0, :, 0, 1:].mean(dim=0)
    side = int(round(math.sqrt(cls.numel())))
    assert side * side == cls.numel()
    amap = cls.reshape(1, 1, side, side)
    amap = F.interpolate(amap, size=(224, 224), mode='bilinear', align_corners=False)[0, 0]
    a = amap.float().cpu().numpy()
    a = (a - a.min()) / (a.max() - a.min() + 1e-12)
    return a

def download_gdc(file_id, dest):
    dest = Path(dest)
    if dest.exists():
        dest.unlink()
    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        cmd = ['curl', '-L', '--fail', '--show-error', '--connect-timeout', '30', '--max-time', '900', '--retry', '2', '--retry-delay', '5', '--retry-all-errors', '-o', str(dest), f'{GDC_DATA}/{file_id}']
        if subprocess.run(cmd).returncode == 0 and dest.exists() and (dest.stat().st_size > 1000000):
            return dest
        if dest.exists():
            dest.unlink()
        time.sleep(5 * attempt)
    raise RuntimeError('GDC download failed: ' + str(file_id))

def read_roi(slide, row):
    x0 = int(float(row[XCOL]))
    y0 = int(float(row[YCOL]))
    if DSCOL is not None and pd.notna(row[DSCOL]):
        ds = max(1.0, float(row[DSCOL]))
        source_w0 = max(PATCH_SIZE, int(round(256 * ds)))
    else:
        source_w0 = PATCH_SIZE
    img = slide.read_region((x0, y0), 0, (source_w0, source_w0)).convert('RGB')
    if img.size != (PATCH_SIZE, PATCH_SIZE):
        img = img.resize((PATCH_SIZE, PATCH_SIZE), Image.Resampling.LANCZOS)
    return img

def bilinear_values(amap, xy):
    if len(xy) == 0:
        return np.array([], float)
    x = np.clip(xy[:, 0], 0, 223)
    y = np.clip(xy[:, 1], 0, 223)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.clip(x0 + 1, 0, 223)
    y1 = np.clip(y0 + 1, 0, 223)
    wx = x - x0
    wy = y - y0
    return (1 - wx) * (1 - wy) * amap[y0, x0] + wx * (1 - wy) * amap[y0, x1] + (1 - wx) * wy * amap[y1, x0] + wx * wy * amap[y1, x1]

def roi_metrics(amap, cells):
    c = cells[cells['high_confidence'].astype(str).str.lower().isin(['true', '1']) | (pd.to_numeric(cells['type_prob'], errors='coerce') >= MIN_TYPE_PROB)].copy()
    c['type'] = pd.to_numeric(c['type'], errors='coerce')
    xy = np.c_[pd.to_numeric(c['x_roi_40xeq'], errors='coerce'), pd.to_numeric(c['y_roi_40xeq'], errors='coerce')] * (224.0 / 512.0)
    ok = np.isfinite(xy).all(axis=1)
    c = c.iloc[np.where(ok)[0]].copy()
    xy = xy[ok]
    vals = bilinear_values(amap, xy)
    types = c['type'].to_numpy()
    inf = vals[types == 2]
    tum = vals[types == 1]
    inf_xy = xy[types == 2]
    tum_xy = xy[types == 1]
    out = {'n_cells': len(vals), 'n_inflammatory': len(inf), 'n_neoplastic': len(tum), 'mean_attention_inflammatory': np.mean(inf) if len(inf) else np.nan, 'mean_attention_neoplastic': np.mean(tum) if len(tum) else np.nan, 'inflammatory_minus_neoplastic_attention': np.mean(inf) - np.mean(tum) if len(inf) and len(tum) else np.nan}
    if len(inf_xy) and len(tum_xy):
        dist, _ = cKDTree(tum_xy).query(inf_xy, k=1)
        interface = dist <= INTERFACE_RADIUS_224
        out['n_interface_inflammatory'] = int(interface.sum())
        out['mean_attention_interface_inflammatory'] = np.mean(inf[interface]) if interface.any() else np.nan
        out['mean_attention_noninterface_inflammatory'] = np.mean(inf[~interface]) if (~interface).any() else np.nan
        out['interface_minus_noninterface_attention'] = out['mean_attention_interface_inflammatory'] - out['mean_attention_noninterface_inflammatory'] if interface.any() and (~interface).any() else np.nan
    else:
        out.update({'n_interface_inflammatory': 0, 'mean_attention_interface_inflammatory': np.nan, 'mean_attention_noninterface_inflammatory': np.nan, 'interface_minus_noninterface_attention': np.nan})
    return (out, xy, types)
all_rows = []
for pi, pid in enumerate(chosen_patients, 1):
    outp = TABLES / f'{pid}__attention_metrics.csv'
    if outp.exists():
        all_rows.append(pd.read_csv(outp))
        continue
    pr = test[test['patient_id'] == pid].iloc[0]
    fid = str(pr['file_id'])
    svs = TMP / f'{pid}.svs'
    wanted = sel[sel['patient_id'] == pid][['cell_roi', 'tile_index', 'master_order', 'inflam_frac_raw']].copy()
    wanted['tile_index_key'] = pd.to_numeric(wanted['tile_index'], errors='raise').astype('int64')
    ps = samples[samples['patient_id'] == pid].copy()
    if 'tile_index' not in ps.columns:
        raise KeyError('Frozen sampling manifest is missing tile_index.')
    ps['tile_index_key'] = pd.to_numeric(ps['tile_index'], errors='raise').astype('int64')
    ps = ps.merge(wanted[['cell_roi', 'tile_index_key', 'inflam_frac_raw']], on='tile_index_key', how='inner', validate='one_to_one')
    assert len(ps) == ROIS_PER_PATIENT, (pid, len(ps), wanted[['cell_roi', 'tile_index']].to_dict('records'))
    cell_path = CELL_DIR / f'{pid}__cells.csv.gz'
    try:
        cells = pd.read_csv(cell_path)
    except pd.errors.EmptyDataError:
        cells = pd.DataFrame(columns=['roi', 'type', 'type_prob', 'high_confidence', 'x_roi_40xeq', 'y_roi_40xeq'])
    if len(cells):
        cells['_cell_roi_str'] = cells['roi'].astype(str)
    download_gdc(fid, svs)
    slide = openslide.OpenSlide(str(svs))
    rows = []
    try:
        for _, r in ps.iterrows():
            img = read_roi(slide, r)
            amap = forward_attention(img)
            cell_roi = str(r['cell_roi'])
            rcells = cells[cells['_cell_roi_str'] == cell_roi].copy() if len(cells) else cells.copy()
            met, xy, types = roi_metrics(amap, rcells)
            observed = met['inflammatory_minus_neoplastic_attention']
            null = []
            if np.isfinite(observed) and len(xy):
                for _ in range(N_SPATIAL_NULLS):
                    sx = int(rng.integers(1, 224))
                    sy = int(rng.integers(1, 224))
                    rolled = np.roll(np.roll(amap, sx, axis=1), sy, axis=0)
                    vals = bilinear_values(rolled, xy)
                    a = vals[types == 2]
                    b = vals[types == 1]
                    if len(a) and len(b):
                        null.append(float(a.mean() - b.mean()))
            met.update({'patient_id': pid, 'file_id': fid, 'split': 'test', 'roi': cell_roi, 'tile_index': int(r['tile_index_key']), 'null_mean_inflammatory_minus_neoplastic': np.mean(null) if null else np.nan, 'null_sd_inflammatory_minus_neoplastic': np.std(null) if null else np.nan, 'spatial_null_p': (1 + np.sum(np.asarray(null) >= observed)) / (1 + len(null)) if null and np.isfinite(observed) else np.nan})
            rows.append(met)
    finally:
        slide.close()
        if svs.exists():
            svs.unlink()
    q = pd.DataFrame(rows)
    q.to_csv(outp, index=False)
    all_rows.append(q)
    print(f'[{pi}/{len(chosen_patients)}] OK {pid}: {len(q)} ROIs')
    gc.collect()
    torch.cuda.empty_cache()
metrics = pd.concat(all_rows, ignore_index=True)
metrics.to_csv(TABLES / 'attention_metrics_all_test_rois.csv', index=False)
patient = metrics.groupby('patient_id', as_index=False).agg(rois=('roi', 'count'), inflammatory_minus_neoplastic_attention=('inflammatory_minus_neoplastic_attention', 'mean'), interface_minus_noninterface_attention=('interface_minus_noninterface_attention', 'mean'), median_spatial_null_p=('spatial_null_p', 'median'), inflammatory_nuclei=('n_inflammatory', 'sum'), neoplastic_nuclei=('n_neoplastic', 'sum'))
patient.to_csv(TABLES / 'patient_attention_validation.csv', index=False)
tests = []
for metric in ['inflammatory_minus_neoplastic_attention', 'interface_minus_noninterface_attention']:
    x = patient[metric].dropna().to_numpy()
    if len(x) >= 5:
        stat, p = wilcoxon(x, alternative='greater')
        tests.append({'metric': metric, 'patients': len(x), 'mean': float(np.mean(x)), 'median': float(np.median(x)), 'wilcoxon_greater_p': float(p)})
tests = pd.DataFrame(tests)
tests.to_csv(TABLES / 'attention_primary_tests.csv', index=False)
for col, title in [('inflammatory_minus_neoplastic_attention', 'Attention at predicted inflammatory vs neoplastic nuclei'), ('interface_minus_noninterface_attention', 'Attention at tumor-proximal vs non-proximal predicted inflammatory nuclei')]:
    x = patient[col].dropna()
    if len(x):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(np.arange(len(x)), x.to_numpy())
        ax.axhline(0, linewidth=1)
        ax.set_ylabel('Mean attention difference')
        ax.set_xlabel('Held-out test patient')
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(FIGS / f'{col}.png', dpi=300, bbox_inches='tight')
        plt.show()
record = {'notebook': '08_Quantitative_ViT_Attention_Validation', 'model': MODEL_NAME, 'patients_sampled': int(patient.patient_id.nunique()), 'rois_analyzed': int(len(metrics)), 'split': 'held-out test only', 'attention': 'last-block mean-head CLS-to-patch attention', 'cell_reference': 'HoVer-Net PanNuke predicted nuclei, type_prob >= 0.50', 'inflammatory_terminology': 'predicted inflammatory nuclei; not confirmed TILs/T cells', 'interface_proxy_radius_224px': INTERFACE_RADIUS_224, 'spatial_nulls_per_roi': N_SPATIAL_NULLS, 'weights_changed': False}
with open(OUT / 'notebook08_results_record.json', 'w') as f:
    json.dump(record, f, indent=2)
