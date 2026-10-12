from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
SEED = 2026
ALPHAS = np.logspace(-4, 4, 17)
CV_FOLDS = 5
ROOT = Path('/content/drive/MyDrive/AACR_GBM_Project')
assert ROOT.exists(), f'Project folder not found: {ROOT}'
OUT = ROOT / 'outputs/notebook06B_generic_dino_baseline'
PAT_DIR = OUT / 'patient_embeddings'
STATUS = OUT / 'status'
TABLES = OUT / 'tables'
for p in [OUT, TABLES]:
    p.mkdir(parents=True, exist_ok=True)

def resolve_file(label, candidates, basename=None):
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
    raise FileNotFoundError(f'Could not locate {label}')
SELECTED_PATH = resolve_file('Notebook02 selected-slide cohort', [ROOT / 'data/metadata/tcga_gbm_notebook02_selected_slides.csv'], 'tcga_gbm_notebook02_selected_slides.csv')
N4_PATH = resolve_file('Notebook04 patient topology', [ROOT / 'outputs/notebook04_spatial_topology/tcga_gbm_patient_spatial_topology.csv'], 'tcga_gbm_patient_spatial_topology.csv')
ADAPTED_RESULTS = resolve_file('Notebook07B corrected probe results', [ROOT / 'outputs/notebook07B_validation/tables/corrected_embedding_probe_results.csv'], 'corrected_embedding_probe_results.csv')
ADAPTED_PREDS = resolve_file('Notebook07B corrected probe predictions', [ROOT / 'outputs/notebook07B_validation/tables/corrected_embedding_probe_predictions.csv'], 'corrected_embedding_probe_predictions.csv')
assert PAT_DIR.exists(), f'Generic DINO patient embedding directory missing: {PAT_DIR}'
assert STATUS.exists(), f'Generic DINO status directory missing: {STATUS}'
selected = pd.read_csv(SELECTED_PATH)
selected['patient_id'] = selected['patient_id'].astype(str)
selected['file_id'] = selected['file_id'].astype(str)
selected['split'] = selected['split'].astype(str).str.lower().replace({'validation': 'val', 'valid': 'val'})
split_counts = selected['split'].value_counts().to_dict()
assert split_counts.get('train') == 262
assert split_counts.get('val') == 57
assert split_counts.get('test') == 58
needed = selected[selected['split'].isin(['train', 'test'])].copy()
needed = needed.sort_values(['split', 'patient_id']).reset_index(drop=True)
assert len(needed) == 320
missing = []
bad = []
for _, r in needed.iterrows():
    pid = str(r.patient_id)
    po = PAT_DIR / f'{pid}__patient_embedding.npz'
    so = STATUS / f'{pid}.done.json'
    if not po.exists() or not so.exists():
        missing.append(pid)
        continue
    try:
        z = np.load(po, allow_pickle=True)
        e = np.asarray(z['mean_embedding'], dtype=np.float32)
        n = int(np.asarray(z['n_embedded_rois']).ravel()[0])
        if e.shape != (384,) or not np.isfinite(e).all() or n != 200:
            bad.append((pid, e.shape, n))
    except Exception as exc:
        bad.append((pid, repr(exc)))
assert not missing and (not bad), 'Embedding audit failed; do not continue.'
rows = []
M = []
for _, r in needed.iterrows():
    pid = str(r.patient_id)
    z = np.load(PAT_DIR / f'{pid}__patient_embedding.npz', allow_pickle=True)
    e = np.asarray(z['mean_embedding'], dtype=np.float32)
    n = int(np.asarray(z['n_embedded_rois']).ravel()[0])
    M.append(e)
    rows.append({'patient_id': pid, 'file_id': str(r.file_id), 'split': str(r.split), 'n_embedded_rois': n})
M = np.stack(M)
meta = pd.DataFrame(rows)
assert M.shape == (320, 384)
assert np.isfinite(M).all()
assert (meta['n_embedded_rois'] == 200).all()
assert meta['patient_id'].is_unique
assert meta['split'].value_counts().to_dict() == {'train': 262, 'test': 58}
np.savez_compressed(OUT / 'generic_dino_patient_embedding_matrix_train_test.npz', mean_embeddings=M, patient_id=meta.patient_id.to_numpy(), file_id=meta.file_id.to_numpy(), split=meta.split.to_numpy(), n_embedded_rois=meta.n_embedded_rois.to_numpy())
meta.to_csv(TABLES / 'generic_dino_patient_embedding_metadata_train_test.csv', index=False)
topo = pd.read_csv(N4_PATH)
topo['patient_id'] = topo['patient_id'].astype(str)
assert topo['patient_id'].is_unique
TARGETS = {'global_inflammatory_fraction': 'global_inflammatory_fraction', 'zero_inflammatory_roi_fraction': 'zero_inflammatory_roi_fraction', 'tumor_to_inflammatory_nn': 'tumor_to_inflam_nn_median_px_patient_median', 'inflammatory_to_inflammatory_nn': 'inflam_to_inflam_nn_median_px_patient_median', 'inflammatory_spatial_entropy': 'inflammatory_spatial_entropy_normalized', 'roi_inflammatory_iqr': 'roi_inflammatory_iqr', 'roi_inflammatory_sd': 'roi_inflammatory_sd', 'immune_rich_fraction': 'immune_rich_roi_fraction_ge_10pct'}
missing_cols = [c for c in TARGETS.values() if c not in topo.columns]
assert not missing_cols, f'Notebook04 endpoint columns missing: {missing_cols}'
ana = meta.copy()
ana['embed_row'] = np.arange(len(meta))
topo_for_merge = topo.drop(columns=['split'], errors='ignore')
ana = ana.merge(topo_for_merge, on='patient_id', how='left', validate='one_to_one')
assert len(ana) == 320
assert 'split' in ana.columns
assert ana['split'].isin(['train', 'test']).all()

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
for name, col in TARGETS.items():
    q = ana[['patient_id', 'split', 'embed_row', col]].dropna().copy()
    tr = q[q['split'] == 'train']
    te = q[q['split'] == 'test']
    assert len(tr) > 0 and len(te) > 0
    model, alpha = fit_probe(M[tr.embed_row.astype(int)], tr[col].to_numpy(float))
    pp = model.predict(M[te.embed_row.astype(int)])
    rows.append({'target': name, 'model': 'generic_DINO', 'n': len(te), 'alpha': alpha, 'spearman_rho': rho(te[col], pp), 'mae': float(mean_absolute_error(te[col], pp)), 'r2': float(r2_score(te[col], pp))})
    for pid, y, p in zip(te.patient_id, te[col], pp):
        preds.append({'patient_id': pid, 'split': 'test', 'target': name, 'y_true': float(y), 'y_pred': float(p)})
base = pd.DataFrame(rows)
base_preds = pd.DataFrame(preds)
assert len(base) == 8
base.to_csv(TABLES / 'generic_dino_test_results.csv', index=False)
base_preds.to_csv(TABLES / 'generic_dino_test_predictions.csv', index=False)
adapt = pd.read_csv(ADAPTED_RESULTS)
required = {'target', 'split', 'n', 'spearman_rho', 'mae', 'r2'}
assert required.issubset(adapt.columns), f'Missing Notebook07B columns: {required - set(adapt.columns)}'
adapt = adapt[adapt['split'].astype(str).str.lower() == 'test'][['target', 'n', 'spearman_rho', 'mae', 'r2']].copy()
adapt = adapt.rename(columns={'n': 'adapted_n', 'spearman_rho': 'adapted_rho', 'mae': 'adapted_mae', 'r2': 'adapted_r2'})
cmp = base.merge(adapt, on='target', how='inner', validate='one_to_one')
assert len(cmp) == 8, f'Expected 8 matched endpoints, got {len(cmp)}'
cmp['delta_rho_adapted_minus_generic'] = cmp['adapted_rho'] - cmp['spearman_rho']
cmp['delta_r2_adapted_minus_generic'] = cmp['adapted_r2'] - cmp['r2']
cmp['delta_mae_generic_minus_adapted'] = cmp['mae'] - cmp['adapted_mae']
cmp.to_csv(TABLES / 'HEAD_TO_HEAD_GBM_ADAPTED_VS_GENERIC_DINO.csv', index=False)
ap = pd.read_csv(ADAPTED_PREDS)
required = {'patient_id', 'target', 'y_true', 'y_pred', 'split'}
assert required.issubset(ap.columns), f'Missing Notebook07B prediction columns: {required - set(ap.columns)}'
ap = ap[ap['split'].astype(str).str.lower() == 'test'][['patient_id', 'target', 'y_true', 'y_pred']].copy()
ap['patient_id'] = ap['patient_id'].astype(str)
ap = ap.rename(columns={'y_pred': 'adapted_pred', 'y_true': 'adapted_y_true'})
bp = base_preds[['patient_id', 'target', 'y_true', 'y_pred']].copy()
bp['patient_id'] = bp['patient_id'].astype(str)
bp = bp.rename(columns={'y_pred': 'generic_pred', 'y_true': 'generic_y_true'})
pair = ap.merge(bp, on=['patient_id', 'target'], how='inner', validate='one_to_one')
assert len(pair) > 0
assert np.allclose(pair['adapted_y_true'], pair['generic_y_true'], equal_nan=True), 'Ground-truth mismatch between adapted and generic predictions.'
rng = np.random.default_rng(SEED)
B = 5000
boots = []
for target, g in pair.groupby('target'):
    g = g.reset_index(drop=True)
    y = g['adapted_y_true'].to_numpy(float)
    a = g['adapted_pred'].to_numpy(float)
    b = g['generic_pred'].to_numpy(float)
    obs = rho(y, a) - rho(y, b)
    vals = []
    for _ in range(B):
        ix = rng.integers(0, len(g), len(g))
        d = rho(y[ix], a[ix]) - rho(y[ix], b[ix])
        if np.isfinite(d):
            vals.append(d)
    vals = np.asarray(vals, float)
    assert len(vals) > 0
    boots.append({'target': target, 'n': len(g), 'delta_rho': obs, 'delta_rho_ci_low': float(np.percentile(vals, 2.5)), 'delta_rho_ci_high': float(np.percentile(vals, 97.5)), 'bootstrap_prob_delta_gt0': float(np.mean(vals > 0)), 'valid_bootstrap_replicates': len(vals)})
boot = pd.DataFrame(boots)
assert len(boot) == 8
boot.to_csv(TABLES / 'paired_bootstrap_adapted_vs_generic.csv', index=False)
required_outputs = [TABLES / 'generic_dino_patient_embedding_metadata_train_test.csv', TABLES / 'generic_dino_test_results.csv', TABLES / 'generic_dino_test_predictions.csv', TABLES / 'HEAD_TO_HEAD_GBM_ADAPTED_VS_GENERIC_DINO.csv', TABLES / 'paired_bootstrap_adapted_vs_generic.csv']
missing_outputs = [str(p) for p in required_outputs if not p.exists()]
assert not missing_outputs, f'Missing final outputs: {missing_outputs}'
record = {'notebook': '06B_07C_Generic_DINO_Finalizer_Only', 'generic_dino_required_patients': 320, 'generic_dino_train_patients': 262, 'generic_dino_test_patients': 58, 'rois_per_patient': 200, 'generic_dino_required_rois': 64000, 'embedding_dim': 384, 'comparison': 'same architecture, same frozen ROI sampling frame, same train/test split, matched downstream Ridge probes', 'n_endpoints': 8, 'paired_bootstrap_replicates': 5000, 'primary_output': str(TABLES / 'HEAD_TO_HEAD_GBM_ADAPTED_VS_GENERIC_DINO.csv')}
with open(OUT / 'baseline_results_record.json', 'w') as f:
    json.dump(record, f, indent=2)
