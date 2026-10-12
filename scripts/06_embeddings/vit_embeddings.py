import subprocess, sys
from google.colab import drive
drive.mount('/content/drive')
subprocess.run(['apt-get', '-qq', 'update'], check=True)
subprocess.run(['apt-get', '-qq', 'install', '-y', 'openslide-tools'], check=True)
subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'openslide-python', 'timm'], check=True)
from pathlib import Path
import gc, json, time, subprocess
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
import timm
import openslide
ROOT = Path('/content/drive/MyDrive/AACR_GBM_Project')
SELECTED_PATH = ROOT / 'data/metadata/tcga_gbm_notebook02_selected_slides.csv'
N3_MANIFEST = ROOT / 'outputs/notebook03_cells/full_cohort_200roi/manifests/full_cohort_200roi_manifest.csv.gz'
ENCODER_PATH = ROOT / 'outputs/notebook05_ssl_dino/export/gbm_dino_vit_small_encoder.pt'
OUT = ROOT / 'outputs/notebook06_embeddings'
ROI_DIR = OUT / 'roi_embeddings'
PATIENT_DIR = OUT / 'patient_embeddings'
STATUS_DIR = OUT / 'status'
LOG_DIR = OUT / 'logs'
for p in [OUT, ROI_DIR, PATIENT_DIR, STATUS_DIR, LOG_DIR]:
    p.mkdir(parents=True, exist_ok=True)
TMP = Path('/content/n6_tmp')
TMP.mkdir(parents=True, exist_ok=True)
EXPECTED_PATIENTS = 377
EXPECTED_ROIS_PER_PATIENT = 200
PATCH_SIZE = 224
BATCH_SIZE = 128
GDC_DATA = 'https://api.gdc.cancer.gov/data'
CURL_CONNECT_TIMEOUT = 30
CURL_MAX_TIME = 900
MAX_DOWNLOAD_ATTEMPTS = 4
selected = pd.read_csv(SELECTED_PATH)
samples = pd.read_csv(N3_MANIFEST)
for df in [selected, samples]:
    df['patient_id'] = df['patient_id'].astype(str)
    df['file_id'] = df['file_id'].astype(str)
assert 'split' in selected.columns
assert selected['patient_id'].nunique() == EXPECTED_PATIENTS, f"Expected {EXPECTED_PATIENTS} selected patients; found {selected['patient_id'].nunique()}"
split_map = selected.set_index('patient_id')['split'].to_dict()
samples['split'] = samples['patient_id'].map(split_map)
counts = samples.groupby('patient_id').size()
assert len(counts) == EXPECTED_PATIENTS
assert (counts == EXPECTED_ROIS_PER_PATIENT).all()

def choose_col(df, candidates, label):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f'Could not find {label}. Tried {candidates}')
XCOL = choose_col(samples, ['roi_x_level0', 'x_level0', 'x', 'tile_x', 'x0'], 'level-0 x')
YCOL = choose_col(samples, ['roi_y_level0', 'y_level0', 'y', 'tile_y', 'y0'], 'level-0 y')
DOWNSAMPLE_COL = None
for c in ['source_downsample', 'downsample', 'level_downsample']:
    if c in samples.columns:
        DOWNSAMPLE_COL = c
        break
ROI_ID_COL = None
for c in ['roi', 'roi_id', 'master_order', 'tile_index']:
    if c in samples.columns:
        ROI_ID_COL = c
        break
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert device.type == 'cuda', 'vit_embeddings.py requires a GPU runtime.'
ck = torch.load(ENCODER_PATH, map_location='cpu')
MODEL_NAME = ck['model_name']
EMBED_DIM = int(ck['num_features'])
encoder = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0)
encoder.load_state_dict(ck['encoder_state_dict'], strict=True)
encoder = encoder.to(device).eval()
for p in encoder.parameters():
    p.requires_grad = False
infer_transform = transforms.Compose([transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC), transforms.ToTensor(), transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

def download_gdc(file_id, dest):
    dest = Path(dest)
    if dest.exists():
        dest.unlink()
    url = f'{GDC_DATA}/{file_id}'
    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        cmd = ['curl', '-L', '--fail', '--show-error', '--connect-timeout', str(CURL_CONNECT_TIMEOUT), '--max-time', str(CURL_MAX_TIME), '--retry', '2', '--retry-delay', '5', '--retry-all-errors', '-o', str(dest), url]
        rc = subprocess.run(cmd).returncode
        if rc == 0 and dest.exists() and (dest.stat().st_size > 1000000):
            return dest
        if dest.exists():
            dest.unlink()
        time.sleep(5 * attempt)
    raise RuntimeError(f'GDC download failed: {file_id}')

def read_roi(slide, row):
    x0 = int(float(row[XCOL]))
    y0 = int(float(row[YCOL]))
    if DOWNSAMPLE_COL is not None and pd.notna(row[DOWNSAMPLE_COL]):
        ds = max(1.0, float(row[DOWNSAMPLE_COL]))
        source_w0 = max(PATCH_SIZE, int(round(256 * ds)))
    else:
        source_w0 = PATCH_SIZE
    img = slide.read_region((x0, y0), 0, (source_w0, source_w0)).convert('RGB')
    if img.size != (PATCH_SIZE, PATCH_SIZE):
        img = img.resize((PATCH_SIZE, PATCH_SIZE), Image.Resampling.LANCZOS)
    return img
patient_rows = selected.sort_values(['split', 'patient_id']).reset_index(drop=True)
run_log = []
for i, pr in patient_rows.iterrows():
    pid = str(pr['patient_id'])
    fid = str(pr['file_id'])
    split = str(pr['split'])
    roi_out = ROI_DIR / f'{pid}__roi_embeddings.npz'
    pat_out = PATIENT_DIR / f'{pid}__patient_embedding.npz'
    done_out = STATUS_DIR / f'{pid}.done.json'
    if roi_out.exists() and pat_out.exists() and done_out.exists():
        try:
            rec = json.load(open(done_out))
            if rec.get('n_expected_rois') == EXPECTED_ROIS_PER_PATIENT:
                run_log.append(rec)
                continue
        except Exception:
            pass
    ps = samples[samples['patient_id'] == pid].copy().reset_index(drop=True)
    assert len(ps) == EXPECTED_ROIS_PER_PATIENT, f'{pid}: {len(ps)} ROIs'
    svs = TMP / f'{pid}.svs'
    t0 = time.time()
    try:
        print(f'[{i + 1}/{len(patient_rows)}] {pid} ({split})')
        download_gdc(fid, svs)
        slide = openslide.OpenSlide(str(svs))
        tensors = []
        keep_idx = []
        blank_idx = []
        for j, row in ps.iterrows():
            img = read_roi(slide, row)
            arr = np.asarray(img)
            if arr.mean() > 252 and arr.std() < 3:
                blank_idx.append(j)
                continue
            tensors.append(infer_transform(img))
            keep_idx.append(j)
        slide.close()
        if len(tensors) == 0:
            raise RuntimeError(f'{pid}: all 200 ROIs were blank extraction failures')
        embs = []
        with torch.inference_mode():
            for b0 in range(0, len(tensors), BATCH_SIZE):
                batch = torch.stack(tensors[b0:b0 + BATCH_SIZE]).to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    z = encoder(batch)
                embs.append(z.float().cpu().numpy())
        E = np.concatenate(embs, axis=0).astype(np.float32)
        assert E.shape == (len(keep_idx), EMBED_DIM)
        assert np.isfinite(E).all()
        meta = ps.iloc[keep_idx].copy()
        x = meta[XCOL].astype(float).to_numpy()
        y = meta[YCOL].astype(float).to_numpy()
        if ROI_ID_COL is not None:
            roi_ids = meta[ROI_ID_COL].astype(str).to_numpy()
        else:
            roi_ids = np.array([str(x) for x in keep_idx])
        np.savez_compressed(roi_out, embeddings=E, roi_id=roi_ids, manifest_row=np.asarray(keep_idx, dtype=np.int32), x_level0=x, y_level0=y, patient_id=np.array([pid]), file_id=np.array([fid]), split=np.array([split]))
        mean_emb = E.mean(axis=0).astype(np.float32)
        std_emb = E.std(axis=0).astype(np.float32)
        np.savez_compressed(pat_out, mean_embedding=mean_emb, std_embedding=std_emb, patient_id=np.array([pid]), file_id=np.array([fid]), split=np.array([split]), n_embedded_rois=np.array([len(E)], dtype=np.int32))
        rec = {'patient_id': pid, 'file_id': fid, 'split': split, 'n_expected_rois': EXPECTED_ROIS_PER_PATIENT, 'n_embedded_rois': int(len(E)), 'n_blank_skipped': int(len(blank_idx)), 'embedding_dim': EMBED_DIM, 'minutes': (time.time() - t0) / 60}
        with open(done_out, 'w') as f:
            json.dump(rec, f, indent=2)
        run_log.append(rec)
    except Exception as e:
        print(f'  ERROR {pid}: {repr(e)}')
        raise
    finally:
        if svs.exists():
            svs.unlink()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
done_files = sorted(STATUS_DIR.glob('*.done.json'))
records = [json.load(open(p)) for p in done_files]
audit = pd.DataFrame(records)
assert audit['patient_id'].nunique() == EXPECTED_PATIENTS, f"Only {audit['patient_id'].nunique()} / {EXPECTED_PATIENTS} patients complete"
assert (audit['embedding_dim'] == EMBED_DIM).all()
expected_total = EXPECTED_PATIENTS * EXPECTED_ROIS_PER_PATIENT
embedded_total = int(audit['n_embedded_rois'].sum())
blank_total = int(audit['n_blank_skipped'].sum())
if embedded_total < expected_total * 0.95:
    raise RuntimeError('Embedding coverage below 95%; inspect before proceeding.')
audit.to_csv(LOG_DIR / 'full_cohort_embedding_audit.csv', index=False)
rows = []
mean_matrix = []
std_matrix = []
for pr in selected.sort_values('patient_id').itertuples(index=False):
    pid = str(pr.patient_id)
    p = PATIENT_DIR / f'{pid}__patient_embedding.npz'
    d = np.load(p, allow_pickle=False)
    mean_matrix.append(d['mean_embedding'])
    std_matrix.append(d['std_embedding'])
    rows.append({'patient_id': pid, 'file_id': str(pr.file_id), 'split': str(pr.split), 'n_embedded_rois': int(d['n_embedded_rois'][0])})
M = np.vstack(mean_matrix).astype(np.float32)
S = np.vstack(std_matrix).astype(np.float32)
meta = pd.DataFrame(rows)
assert M.shape == (EXPECTED_PATIENTS, EMBED_DIM)
assert S.shape == (EXPECTED_PATIENTS, EMBED_DIM)
assert np.isfinite(M).all() and np.isfinite(S).all()
np.savez_compressed(OUT / 'patient_embedding_matrix.npz', mean_embeddings=M, std_embeddings=S, patient_id=meta['patient_id'].to_numpy(), file_id=meta['file_id'].to_numpy(), split=meta['split'].to_numpy(), n_embedded_rois=meta['n_embedded_rois'].to_numpy(dtype=np.int32))
meta.to_csv(OUT / 'patient_embedding_metadata.csv', index=False)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
train_mask = meta['split'].astype(str).str.lower().eq('train').to_numpy()
scaler = StandardScaler()
Xtr = scaler.fit_transform(M[train_mask])
Xall = scaler.transform(M)
pca = PCA(n_components=min(50, Xtr.shape[0] - 1, Xtr.shape[1]), random_state=2026)
pca.fit(Xtr)
Z = pca.transform(Xall)
pca_df = meta.copy()
pca_df['PC1'] = Z[:, 0]
pca_df['PC2'] = Z[:, 1]
pca_df.to_csv(OUT / 'patient_embedding_pca.csv', index=False)
fig, ax = plt.subplots(figsize=(7, 6))
for split in ['train', 'validation', 'test']:
    q = pca_df[pca_df['split'].astype(str).str.lower() == split]
    ax.scatter(q['PC1'], q['PC2'], s=22, alpha=0.7, label=split)
ax.set_xlabel('Embedding PC1')
ax.set_ylabel('Embedding PC2')
ax.set_title('Frozen GBM SSL ViT — patient embedding space')
ax.legend()
fig.tight_layout()
fig.savefig(OUT / 'patient_embedding_pca.png', dpi=300)
plt.show()
record = {'notebook': '06_Frozen_ViT_Embedding_Extraction', 'encoder': MODEL_NAME, 'embedding_dim': EMBED_DIM, 'patients': int(meta['patient_id'].nunique()), 'expected_rois': int(EXPECTED_PATIENTS * EXPECTED_ROIS_PER_PATIENT), 'embedded_rois': int(audit['n_embedded_rois'].sum()), 'blank_skipped': int(audit['n_blank_skipped'].sum()), 'train_patients': int((meta['split'].astype(str).str.lower() == 'train').sum()), 'validation_patients': int((meta['split'].astype(str).str.lower() == 'validation').sum()), 'test_patients': int((meta['split'].astype(str).str.lower() == 'test').sum()), 'encoder_training_rule': 'Notebook 05 optimized encoder on training patients only', 'notebook06_rule': 'frozen inference only; no topology/outcome/demographic labels used', 'patient_matrix': str(OUT / 'patient_embedding_matrix.npz')}
with open(OUT / 'notebook06_results_record.json', 'w') as f:
    json.dump(record, f, indent=2)
