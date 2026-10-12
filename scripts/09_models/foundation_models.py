import subprocess, sys
subprocess.run(['apt-get', '-qq', 'update'], check=True)
subprocess.run(['apt-get', '-qq', 'install', '-y', 'libopenslide0', 'openslide-tools'], check=True)
subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', '-U', 'openslide-python', 'timm>=1.0.20', 'huggingface_hub>=0.25', 'scikit-learn'], check=True)
subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'git+https://github.com/mahmoodlab/CONCH.git'], check=True)
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import gc, json, subprocess, time
import numpy as np
import pandas as pd
from PIL import Image
import torch, timm, openslide
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login, get_token
from conch.open_clip_custom import create_model_from_pretrained
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
SEED = 2026
np.random.seed(SEED)
assert torch.cuda.is_available(), 'Use a GPU runtime; A100 is recommended.'
device = torch.device('cuda')
login()
HF_TOKEN = get_token()
assert HF_TOKEN, 'Hugging Face authentication succeeded without a retrievable token. Re-run login() using your READ token.'
models = {}
transforms_by_model = {}
dims = {}
uni = timm.create_model('hf_hub:MahmoodLab/UNI', pretrained=True, init_values=1e-05, dynamic_img_size=True).to(device).eval()
uni_cfg = resolve_data_config(uni.pretrained_cfg, model=uni)
uni_preprocess = create_transform(**uni_cfg)
dummy = Image.new('RGB', (224, 224), (180, 120, 150))
x = uni_preprocess(dummy).unsqueeze(0).to(device)
with torch.inference_mode(), torch.amp.autocast('cuda'):
    z = uni(x)
assert torch.is_tensor(z) and z.ndim == 2 and (z.shape[0] == 1)
models['UNI'] = uni
transforms_by_model['UNI'] = uni_preprocess
dims['UNI'] = int(z.shape[1])
del x, z
torch.cuda.empty_cache()
conch_model, conch_preprocess = create_model_from_pretrained('conch_ViT-B-16', 'hf_hub:MahmoodLab/conch', hf_auth_token=HF_TOKEN)
conch_model = conch_model.to(device).eval()
x = conch_preprocess(dummy).unsqueeze(0).to(device)
with torch.inference_mode(), torch.amp.autocast('cuda'):
    z = conch_model.encode_image(x, proj_contrast=False, normalize=False)
assert torch.is_tensor(z) and z.ndim == 2 and (z.shape[0] == 1)
models['CONCH'] = conch_model
transforms_by_model['CONCH'] = conch_preprocess
dims['CONCH'] = int(z.shape[1])
del x, z, dummy
torch.cuda.empty_cache()
MODEL_SPECS = {'UNI': 'MahmoodLab/UNI', 'CONCH': 'MahmoodLab/CONCH'}
ROOT = Path('/content/drive/MyDrive/AACR_GBM_Project')
assert ROOT.exists(), ROOT
SELECTED_PATH = ROOT / 'data/metadata/tcga_gbm_notebook02_selected_slides.csv'
MANIFEST_PATH = ROOT / 'outputs/notebook03_cells/full_cohort_200roi/manifests/full_cohort_200roi_manifest.csv.gz'
N4_PATH = ROOT / 'outputs/notebook04_spatial_topology/tcga_gbm_patient_spatial_topology.csv'
GENERIC_RESULTS = ROOT / 'outputs/notebook06B_generic_dino_baseline/tables/generic_dino_test_results.csv'
GENERIC_PREDS = ROOT / 'outputs/notebook06B_generic_dino_baseline/tables/generic_dino_test_predictions.csv'

def find_one(label, candidates, basename=None):
    for p in candidates:
        p = Path(p)
        if p.exists():
            return p
    if basename:
        hits = list(ROOT.rglob(basename))
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise RuntimeError(f'Ambiguous {label}; matches: {hits}')
    raise FileNotFoundError(label)
ADAPTED_RESULTS = find_one('Notebook07B results', [ROOT / 'outputs/notebook07B_validation/tables/corrected_embedding_probe_results.csv'], 'corrected_embedding_probe_results.csv')
ADAPTED_PREDS = find_one('Notebook07B predictions', [ROOT / 'outputs/notebook07B_validation/tables/corrected_embedding_probe_predictions.csv'], 'corrected_embedding_probe_predictions.csv')
for p in [SELECTED_PATH, MANIFEST_PATH, N4_PATH, GENERIC_RESULTS, GENERIC_PREDS]:
    assert p.exists(), f'Missing required prior output: {p}'
OUT = ROOT / 'outputs/notebook09_uni_conch_comparison'
EMB = OUT / 'patient_embeddings'
STATUS = OUT / 'status'
TABLES = OUT / 'tables'
for p in [OUT, EMB, STATUS, TABLES]:
    p.mkdir(parents=True, exist_ok=True)
TMP = Path('/content/n9_tmp')
TMP.mkdir(exist_ok=True)
N_SHARDS = 1
SHARD_ID = 0
assert N_SHARDS >= 1 and 0 <= SHARD_ID < N_SHARDS
PATCH_SIZE = 224
BATCH_SIZE = 32
GDC_DATA = 'https://api.gdc.cancer.gov/data'
ALPHAS = np.logspace(-4, 4, 17)
CV_FOLDS = 5
selected = pd.read_csv(SELECTED_PATH)
selected['patient_id'] = selected['patient_id'].astype(str)
selected['file_id'] = selected['file_id'].astype(str)
selected['split'] = selected['split'].astype(str).str.lower().replace({'validation': 'val', 'valid': 'val'})
assert tuple(selected['split'].value_counts().reindex(['train', 'val', 'test']).tolist()) == (262, 57, 58)
samples = pd.read_csv(MANIFEST_PATH)
samples['patient_id'] = samples['patient_id'].astype(str)
samples['file_id'] = samples['file_id'].astype(str)
counts = samples.groupby('patient_id').size()
assert len(counts) == 377 and (counts == 200).all()

def choose(df, names, label):
    for c in names:
        if c in df.columns:
            return c
    raise KeyError(f'Missing {label}: {names}')
XCOL = choose(samples, ['roi_x_level0', 'x_level0', 'x', 'tile_x', 'x0'], 'level-0 x')
YCOL = choose(samples, ['roi_y_level0', 'y_level0', 'y', 'tile_y', 'y0'], 'level-0 y')
ROICOL = choose(samples, ['tile_index', 'roi', 'roi_id', 'master_order'], 'ROI identity')
DSCOL = next((c for c in ['source_downsample', 'downsample', 'level_downsample'] if c in samples.columns), None)
work_all = selected[selected['split'].isin(['train', 'test'])].sort_values(['split', 'patient_id']).reset_index(drop=True).copy()
assert len(work_all) == 320
work_all['global_work_index'] = np.arange(len(work_all))
work = work_all[work_all['global_work_index'] % N_SHARDS == SHARD_ID].reset_index(drop=True)
DOWNLOAD_CONNECT_TIMEOUT = 30
DOWNLOAD_MAX_TIME = 1800
DOWNLOAD_ATTEMPTS = 3

def download_gdc(fid, dest):
    dest = Path(dest)
    part = Path(str(dest) + '.part')
    if dest.exists() and dest.stat().st_size > 1000000:
        return dest
    for a in range(1, DOWNLOAD_ATTEMPTS + 1):
        t0 = time.time()
        cmd = ['curl', '-L', '--fail', '--show-error', '-C', '-', '--connect-timeout', str(DOWNLOAD_CONNECT_TIMEOUT), '--max-time', str(DOWNLOAD_MAX_TIME), '-o', str(part), f'{GDC_DATA}/{fid}']
        rc = subprocess.run(cmd).returncode
        mb = part.stat().st_size / 1000000.0 if part.exists() else 0
        if rc == 0 and part.exists() and (part.stat().st_size > 1000000):
            part.replace(dest)
            return dest
        time.sleep(3)
    raise RuntimeError('GDC download failed after resumable attempts: ' + str(fid))

def read_roi(slide, row):
    x0 = int(float(row[XCOL]))
    y0 = int(float(row[YCOL]))
    if DSCOL is not None and pd.notna(row[DSCOL]):
        ds = max(1.0, float(row[DSCOL]))
        source_w0 = max(PATCH_SIZE, int(round(256 * ds)))
    else:
        source_w0 = PATCH_SIZE
    return slide.read_region((x0, y0), 0, (source_w0, source_w0)).convert('RGB')

def model_embed(name, pil_images):
    model = models[name]
    tfm = transforms_by_model[name]
    chunks = []
    with torch.inference_mode():
        for b in range(0, len(pil_images), BATCH_SIZE):
            x = torch.stack([tfm(im) for im in pil_images[b:b + BATCH_SIZE]]).to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                if name == 'CONCH':
                    z = model.encode_image(x, proj_contrast=False, normalize=False)
                elif name == 'UNI':
                    z = model(x)
                else:
                    raise KeyError(name)
            assert torch.is_tensor(z) and z.ndim == 2 and (z.shape[0] == x.shape[0]), (name, type(z), getattr(z, 'shape', None))
            chunks.append(z.float().cpu().numpy())
            del x, z
    E = np.concatenate(chunks).astype(np.float32)
    assert E.shape == (len(pil_images), dims[name]), (name, E.shape, dims[name])
    assert np.isfinite(E).all()
    return E
test_imgs = [Image.new('RGB', (224, 224), (170, 120, 145)) for _ in range(2)]
for _name in ['UNI', 'CONCH']:
    _E = model_embed(_name, test_imgs)
    del _E
torch.cuda.empty_cache()

def patient_complete(pid):
    for name in MODEL_SPECS:
        p = EMB / f'{pid}__{name.lower()}__patient_embedding.npz'
        if not p.exists():
            return False
        try:
            z = np.load(p, allow_pickle=True)
            if int(np.asarray(z['n_embedded_rois']).ravel()[0]) != 200:
                return False
            e = np.asarray(z['mean_embedding'])
            if e.shape != (dims[name],) or not np.isfinite(e).all():
                return False
        except Exception:
            return False
    s = STATUS / f'{pid}.done.json'
    return s.exists()
failed = []
for i, pr in work.iterrows():
    pid = str(pr.patient_id)
    fid = str(pr.file_id)
    split = str(pr.split)
    print(f'[{i + 1}/{len(work)}] {pid} ({split})', flush=True)
    if patient_complete(pid):
        continue
    ps = samples[samples.patient_id == pid].copy().reset_index(drop=True)
    assert len(ps) == 200
    svs = TMP / f'{pid}.svs'
    tpatient = time.time()
    try:
        download_gdc(fid, svs)
        slide = openslide.OpenSlide(str(svs))
        imgs = []
        try:
            for _, r in ps.iterrows():
                im = read_roi(slide, r)
                arr = np.asarray(im)
                if arr.mean() > 252 and arr.std() < 3:
                    raise RuntimeError('Unexpected blank ROI in frozen 200-ROI frame')
                imgs.append(im)
        finally:
            slide.close()
        assert len(imgs) == 200
        for name in ['UNI', 'CONCH']:
            t0 = time.time()
            E = model_embed(name, imgs)
            out = EMB / f'{pid}__{name.lower()}__patient_embedding.npz'
            np.savez_compressed(out, mean_embedding=E.mean(0).astype(np.float32), std_embedding=E.std(0).astype(np.float32), patient_id=np.array([pid]), file_id=np.array([fid]), split=np.array([split]), n_embedded_rois=np.array([200]), embedding_dim=np.array([E.shape[1]]))
            del E
            torch.cuda.empty_cache()
        rec = {'patient_id': pid, 'file_id': fid, 'split': split, 'n_embedded_rois': 200, 'models': {k: {'embedding_dim': dims[k]} for k in MODEL_SPECS}}
        with open(STATUS / f'{pid}.done.json', 'w') as f:
            json.dump(rec, f, indent=2)
        print(f'{pid}: complete in {time.time() - tpatient:.1f}s', flush=True)
    except Exception as e:
        failed.append({'patient_id': pid, 'file_id': fid, 'split': split, 'error': repr(e)})
        print('    WARNING: FAILED:', repr(e), flush=True)
    finally:
        if svs.exists():
            svs.unlink()
        gc.collect()
        torch.cuda.empty_cache()
pd.DataFrame(failed).to_csv(TABLES / f'shard_{SHARD_ID}_failures.csv', index=False)
missing = []
for _, r in work_all.iterrows():
    pid = str(r.patient_id)
    if not patient_complete(pid):
        missing.append(pid)
missing = [str(r.patient_id) for _, r in work_all.iterrows() if not patient_complete(str(r.patient_id))]
assert not missing, f'Still missing {len(missing)} patients. Finish extraction first.'
meta = work_all[['patient_id', 'file_id', 'split']].copy().reset_index(drop=True)
matrices = {}
for name in ['UNI', 'CONCH']:
    M = []
    for pid in meta.patient_id:
        z = np.load(EMB / f'{pid}__{name.lower()}__patient_embedding.npz', allow_pickle=True)
        M.append(np.asarray(z['mean_embedding'], dtype=np.float32))
    M = np.stack(M)
    assert M.shape == (320, dims[name]) and np.isfinite(M).all()
    matrices[name] = M
    np.savez_compressed(OUT / f'{name.lower()}_patient_embedding_matrix_train_test.npz', mean_embeddings=M, patient_id=meta.patient_id.to_numpy(), file_id=meta.file_id.to_numpy(), split=meta.split.to_numpy())
meta.to_csv(TABLES / 'uni_conch_patient_metadata_train_test.csv', index=False)
topo = pd.read_csv(N4_PATH)
topo['patient_id'] = topo['patient_id'].astype(str)
assert topo['patient_id'].is_unique
TARGETS = {'global_inflammatory_fraction': 'global_inflammatory_fraction', 'zero_inflammatory_roi_fraction': 'zero_inflammatory_roi_fraction', 'tumor_to_inflammatory_nn': 'tumor_to_inflam_nn_median_px_patient_median', 'inflammatory_to_inflammatory_nn': 'inflam_to_inflam_nn_median_px_patient_median', 'inflammatory_spatial_entropy': 'inflammatory_spatial_entropy_normalized', 'roi_inflammatory_iqr': 'roi_inflammatory_iqr', 'roi_inflammatory_sd': 'roi_inflammatory_sd', 'immune_rich_fraction': 'immune_rich_roi_fraction_ge_10pct'}
for c in TARGETS.values():
    assert c in topo.columns, c
ana = meta.copy()
ana['embed_row'] = np.arange(len(meta))
ana = ana.merge(topo.drop(columns=['split'], errors='ignore'), on='patient_id', how='left', validate='one_to_one')
assert len(ana) == 320 and ana['split'].isin(['train', 'test']).all()

def rho(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    if len(y) < 3 or np.std(y) == 0 or np.std(p) == 0:
        return np.nan
    return float(spearmanr(y, p).statistic)

def fit_probe(X, y):
    pipe = Pipeline([('scale', StandardScaler()), ('ridge', Ridge())])
    cv = KFold(CV_FOLDS, shuffle=True, random_state=SEED)
    g = GridSearchCV(pipe, {'ridge__alpha': ALPHAS}, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1).fit(X, y)
    return (g.best_estimator_, float(g.best_params_['ridge__alpha']))
rows = []
preds = []
for model_name, M in matrices.items():
    for target, col in TARGETS.items():
        q = ana[['patient_id', 'split', 'embed_row', col]].dropna()
        tr = q[q['split'] == 'train']
        te = q[q['split'] == 'test']
        model, alpha = fit_probe(M[tr.embed_row.astype(int)], tr[col].to_numpy(float))
        pp = model.predict(M[te.embed_row.astype(int)])
        rows.append({'model': model_name, 'target': target, 'n': len(te), 'alpha': alpha, 'spearman_rho': rho(te[col], pp), 'mae': float(mean_absolute_error(te[col], pp)), 'r2': float(r2_score(te[col], pp))})
        for pid, y, p in zip(te.patient_id, te[col], pp):
            preds.append({'model': model_name, 'patient_id': pid, 'split': 'test', 'target': target, 'y_true': float(y), 'y_pred': float(p)})
fm_results = pd.DataFrame(rows)
fm_preds = pd.DataFrame(preds)
assert len(fm_results) == 16
fm_results.to_csv(TABLES / 'UNI_CONCH_test_results.csv', index=False)
fm_preds.to_csv(TABLES / 'UNI_CONCH_test_predictions.csv', index=False)
generic = pd.read_csv(GENERIC_RESULTS)
adapted = pd.read_csv(ADAPTED_RESULTS)
if 'split' in adapted.columns:
    adapted = adapted[adapted['split'].astype(str).str.lower() == 'test'].copy()

def norm_results(df, model_name):
    x = df.copy()
    x['model'] = model_name
    return x[['model', 'target', 'n', 'spearman_rho', 'mae', 'r2']]
all_results = pd.concat([norm_results(generic, 'Generic DINO'), norm_results(adapted, 'GBM-adapted DINO'), fm_results[['model', 'target', 'n', 'spearman_rho', 'mae', 'r2']]], ignore_index=True)
assert set(all_results['model']) == {'Generic DINO', 'GBM-adapted DINO', 'UNI', 'CONCH'}
assert all_results.groupby('model').size().eq(8).all()
all_results.to_csv(TABLES / 'FOUR_MODEL_HEAD_TO_HEAD.csv', index=False)
pivot = all_results.pivot(index='target', columns='model', values='spearman_rho')
pivot.to_csv(TABLES / 'FOUR_MODEL_SPEARMAN_TABLE.csv')
generic_p = pd.read_csv(GENERIC_PREDS)
adapted_p = pd.read_csv(ADAPTED_PREDS)
if 'split' in adapted_p.columns:
    adapted_p = adapted_p[adapted_p['split'].astype(str).str.lower() == 'test'].copy()
pred_frames = [generic_p.assign(model='Generic DINO')[['model', 'patient_id', 'target', 'y_true', 'y_pred']], adapted_p.assign(model='GBM-adapted DINO')[['model', 'patient_id', 'target', 'y_true', 'y_pred']], fm_preds[['model', 'patient_id', 'target', 'y_true', 'y_pred']]]
P = pd.concat(pred_frames, ignore_index=True)
P['patient_id'] = P['patient_id'].astype(str)
comparisons = [('UNI', 'Generic DINO'), ('CONCH', 'Generic DINO'), ('UNI', 'GBM-adapted DINO'), ('CONCH', 'GBM-adapted DINO'), ('UNI', 'CONCH')]
rng = np.random.default_rng(SEED)
B = 5000
bout = []
for A, Bmodel in comparisons:
    for target in TARGETS:
        a = P[(P.model == A) & (P.target == target)][['patient_id', 'y_true', 'y_pred']].rename(columns={'y_true': 'yA', 'y_pred': 'pA'})
        b = P[(P.model == Bmodel) & (P.target == target)][['patient_id', 'y_true', 'y_pred']].rename(columns={'y_true': 'yB', 'y_pred': 'pB'})
        q = a.merge(b, on='patient_id', how='inner', validate='one_to_one')
        assert len(q) > 0 and np.allclose(q.yA, q.yB, equal_nan=True)
        y = q.yA.to_numpy(float)
        pa = q.pA.to_numpy(float)
        pb = q.pB.to_numpy(float)
        obs = rho(y, pa) - rho(y, pb)
        vals = []
        for _ in range(5000):
            ix = rng.integers(0, len(q), len(q))
            d = rho(y[ix], pa[ix]) - rho(y[ix], pb[ix])
            if np.isfinite(d):
                vals.append(d)
        vals = np.asarray(vals)
        bout.append({'model_A': A, 'model_B': Bmodel, 'target': target, 'n': len(q), 'delta_rho_A_minus_B': obs, 'ci_low': float(np.percentile(vals, 2.5)), 'ci_high': float(np.percentile(vals, 97.5)), 'bootstrap_prob_delta_gt0': float(np.mean(vals > 0)), 'valid_bootstrap_replicates': len(vals)})
boot = pd.DataFrame(bout)
boot.to_csv(TABLES / 'FOUR_MODEL_PAIRED_BOOTSTRAP.csv', index=False)
required = [TABLES / 'UNI_CONCH_test_results.csv', TABLES / 'UNI_CONCH_test_predictions.csv', TABLES / 'FOUR_MODEL_HEAD_TO_HEAD.csv', TABLES / 'FOUR_MODEL_SPEARMAN_TABLE.csv', TABLES / 'FOUR_MODEL_PAIRED_BOOTSTRAP.csv']
assert all((p.exists() for p in required))
record = {'notebook': '09_Published_Pathology_Foundation_Model_Comparison', 'models': ['Generic DINO', 'GBM-adapted DINO', 'UNI', 'CONCH'], 'new_models': ['UNI', 'CONCH'], 'patients_train_test': 320, 'train_patients': 262, 'test_patients': 58, 'rois_per_patient': 200, 'frozen_sampling_frame': True, 'topology_endpoints': 8, 'probe': 'StandardScaler + Ridge; alpha selected by 5-fold CV on training patients', 'paired_bootstrap_replicates': 5000, 'notes': ['UNI and CONCH were used as frozen published pathology foundation-model encoders.', 'All four representations were evaluated on the same held-out topology endpoints.', 'No outcome or demographic labels were used to train UNI/CONCH in this notebook.', 'Model comparisons should be interpreted endpoint-by-endpoint; no single overall winner is assumed.']}
with open(OUT / 'notebook09_results_record.json', 'w') as f:
    json.dump(record, f, indent=2)
