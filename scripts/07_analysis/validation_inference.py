from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
SEED = 2026
rng = np.random.default_rng(SEED)
ROOT = Path('/content/drive/MyDrive/AACR_GBM_Project')
N6 = ROOT / 'outputs/notebook06_embeddings'
EMBED_PATH = N6 / 'patient_embedding_matrix.npz'
EMBED_META_PATH = N6 / 'patient_embedding_metadata.csv'
N4 = ROOT / 'outputs/notebook04_spatial_topology'
TOPO_PATH = N4 / 'tcga_gbm_patient_spatial_topology.csv'
CLINICAL_PATH = ROOT / 'data/metadata/tcga_gbm_clinical_indexed.csv'
OUT = ROOT / 'outputs/notebook07B_validation'
TABLES = OUT / 'tables'
FIGS = OUT / 'figures'
for p in [OUT, TABLES, FIGS]:
    p.mkdir(parents=True, exist_ok=True)
ALPHAS = np.logspace(-4, 4, 17)
CV_FOLDS = 5
BOOTSTRAPS = 5000
PERMUTATIONS = 5000
MIN_GROUP_N_TOTAL = 15
MIN_GROUP_N_TEST = 5
d = np.load(EMBED_PATH, allow_pickle=False)
M = d['mean_embeddings'].astype(np.float32)
S = d['std_embeddings'].astype(np.float32)
meta = pd.read_csv(EMBED_META_PATH)
meta['patient_id'] = meta['patient_id'].astype(str)
meta['split'] = meta['split'].astype(str).str.lower().replace({'validation': 'val', 'valid': 'val'})
meta['embed_row'] = np.arange(len(meta))
assert M.shape == (377, 384)
assert S.shape == (377, 384)
assert meta['patient_id'].is_unique
vc = meta['split'].value_counts().to_dict()
assert (vc.get('train', 0), vc.get('val', 0), vc.get('test', 0)) == (262, 57, 58)
topo = pd.read_csv(TOPO_PATH)
topo['patient_id'] = topo['patient_id'].astype(str)
EXPECTED = {'global_inflammatory_fraction': 'global_inflammatory_fraction', 'zero_inflammatory_roi_fraction': 'zero_inflammatory_roi_fraction', 'tumor_to_inflammatory_nn': 'tumor_to_inflam_nn_median_px_patient_median', 'inflammatory_to_inflammatory_nn': 'inflam_to_inflam_nn_median_px_patient_median', 'inflammatory_spatial_entropy': 'inflammatory_spatial_entropy_normalized', 'roi_inflammatory_iqr': 'roi_inflammatory_iqr', 'roi_inflammatory_sd': 'roi_inflammatory_sd', 'immune_rich_fraction': 'immune_rich_roi_fraction_ge_10pct'}
missing = {k: v for k, v in EXPECTED.items() if v not in topo.columns}
if missing:
    raise KeyError('Missing spatial topology endpoint columns: ' + json.dumps(missing, indent=2) + '\nAvailable columns: ' + ', '.join(topo.columns))
TARGETS = EXPECTED.copy()
ABUND = TARGETS['global_inflammatory_fraction']
assert TARGETS['tumor_to_inflammatory_nn'] != TARGETS['inflammatory_to_inflammatory_nn']
ana = meta.merge(topo, on='patient_id', how='left', suffixes=('', '_n4'))
assert ana['patient_id'].is_unique
rows = []
for name, col in TARGETS.items():
    rows.append({'target': name, 'column': col, 'overall_n': int(ana[col].notna().sum()), 'train_n': int(ana.loc[ana['split'] == 'train', col].notna().sum()), 'val_n': int(ana.loc[ana['split'] == 'val', col].notna().sum()), 'test_n': int(ana.loc[ana['split'] == 'test', col].notna().sum())})
coverage = pd.DataFrame(rows)
coverage.to_csv(TABLES / 'endpoint_coverage.csv', index=False)

def spearman_safe(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    if len(y) < 3 or np.std(y) == 0 or np.std(p) == 0:
        return np.nan
    return float(spearmanr(y, p).statistic)

def point_metrics(y, p):
    return {'n': int(len(y)), 'spearman_rho': spearman_safe(y, p), 'mae': float(mean_absolute_error(y, p)), 'r2': float(r2_score(y, p)) if len(y) >= 2 else np.nan}

def fit_ridge_train_only(X, y):
    pipe = Pipeline([('scale', StandardScaler()), ('ridge', Ridge())])
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    gs = GridSearchCV(pipe, {'ridge__alpha': ALPHAS}, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
    gs.fit(X, y)
    return (gs.best_estimator_, float(gs.best_params_['ridge__alpha']))

def bootstrap_metrics(y, p, B=BOOTSTRAPS):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, len(y), len(y))
        yy, pp = (y[idx], p[idx])
        vals.append([spearman_safe(yy, pp), mean_absolute_error(yy, pp), r2_score(yy, pp) if len(np.unique(yy)) > 1 else np.nan])
    a = np.asarray(vals, float)
    out = {}
    for j, name in enumerate(['rho', 'mae', 'r2']):
        z = a[:, j]
        z = z[np.isfinite(z)]
        out[f'{name}_ci_low'] = float(np.percentile(z, 2.5))
        out[f'{name}_ci_high'] = float(np.percentile(z, 97.5))
    return out

def permutation_p_spearman(y, p, B=PERMUTATIONS):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    obs = abs(spearman_safe(y, p))
    null = np.empty(B, float)
    for i in range(B):
        null[i] = abs(spearman_safe(rng.permutation(y), p))
    return float((1 + np.sum(null >= obs)) / (B + 1))
results = []
pred_rows = []
for target, col in TARGETS.items():
    q = ana[['patient_id', 'split', 'embed_row', col]].dropna().copy()
    tr = q[q['split'] == 'train']
    va = q[q['split'] == 'val']
    te = q[q['split'] == 'test']
    model, alpha = fit_ridge_train_only(M[tr['embed_row'].astype(int)], tr[col].to_numpy())
    for split_name, part in [('train', tr), ('val', va), ('test', te)]:
        if len(part) == 0:
            continue
        pred = model.predict(M[part['embed_row'].astype(int)])
        met = point_metrics(part[col], pred)
        row = {'target': target, 'target_col': col, 'split': split_name, 'alpha': alpha, **met}
        if split_name == 'test':
            row.update(bootstrap_metrics(part[col], pred))
            row['spearman_permutation_p'] = permutation_p_spearman(part[col], pred)
        results.append(row)
        for pid, yhat, y in zip(part['patient_id'], pred, part[col]):
            pred_rows.append({'patient_id': pid, 'split': split_name, 'target': target, 'target_col': col, 'y_true': float(y), 'y_pred': float(yhat)})
res = pd.DataFrame(results)
preds = pd.DataFrame(pred_rows)
res.to_csv(TABLES / 'corrected_embedding_probe_results.csv', index=False)
preds.to_csv(TABLES / 'corrected_embedding_probe_predictions.csv', index=False)
abundance_rows = []
residual_rows = []
residual_preds = []
for target, col in TARGETS.items():
    if target == 'global_inflammatory_fraction':
        continue
    q = ana[['patient_id', 'split', 'embed_row', ABUND, col]].dropna().copy()
    tr = q[q['split'] == 'train'].copy()
    va = q[q['split'] == 'val'].copy()
    te = q[q['split'] == 'test'].copy()
    ab_model, ab_alpha = fit_ridge_train_only(tr[[ABUND]].to_numpy(), tr[col].to_numpy())
    for split_name, part in [('val', va), ('test', te)]:
        if len(part) == 0:
            continue
        pp = ab_model.predict(part[[ABUND]].to_numpy())
        row = {'target': target, 'split': split_name, 'alpha': ab_alpha, **point_metrics(part[col], pp)}
        if split_name == 'test':
            row.update(bootstrap_metrics(part[col], pp))
            row['spearman_permutation_p'] = permutation_p_spearman(part[col], pp)
        abundance_rows.append(row)
    adjust = Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=1.0))])
    adjust.fit(tr[[ABUND]], tr[col])
    for part in [tr, va, te]:
        part['residual_target'] = part[col] - adjust.predict(part[[ABUND]])
    emb_model, emb_alpha = fit_ridge_train_only(M[tr['embed_row'].astype(int)], tr['residual_target'])
    for split_name, part in [('train', tr), ('val', va), ('test', te)]:
        if len(part) == 0:
            continue
        pp = emb_model.predict(M[part['embed_row'].astype(int)])
        row = {'target': target, 'split': split_name, 'alpha': emb_alpha, **point_metrics(part['residual_target'], pp)}
        if split_name == 'test':
            row.update(bootstrap_metrics(part['residual_target'], pp))
            row['spearman_permutation_p'] = permutation_p_spearman(part['residual_target'], pp)
        residual_rows.append(row)
        for pid, y, p in zip(part['patient_id'], part['residual_target'], pp):
            residual_preds.append({'patient_id': pid, 'split': split_name, 'target': target, 'y_true': float(y), 'y_pred': float(p)})
abres = pd.DataFrame(abundance_rows)
rr = pd.DataFrame(residual_rows)
rpred = pd.DataFrame(residual_preds)
abres.to_csv(TABLES / 'corrected_abundance_only_results.csv', index=False)
rr.to_csv(TABLES / 'corrected_residual_topology_results.csv', index=False)
rpred.to_csv(TABLES / 'corrected_residual_topology_predictions.csv', index=False)
check = res[(res['split'] == 'test') & res['target'].isin(['tumor_to_inflammatory_nn', 'inflammatory_to_inflammatory_nn'])][['target', 'target_col', 'n', 'spearman_rho', 'mae', 'r2', 'rho_ci_low', 'rho_ci_high', 'spearman_permutation_p']]
assert check['target_col'].nunique() == 2
clinical = pd.read_csv(CLINICAL_PATH)
clinical['patient_id'] = clinical['patient_id'].astype(str)
MISSING_TOKENS = {'', 'nan', 'none', 'unknown', 'not reported', 'not_reported', 'not allowed to collect', 'not allowed to collect (not reported)'}

def norm_demo(v):
    if pd.isna(v):
        return None
    s = str(v).strip().lower()
    if s in MISSING_TOKENS:
        return None
    return s

def collapse_one(series):
    vals = [norm_demo(x) for x in series]
    vals = [x for x in vals if x is not None]
    uniq = sorted(set(vals))
    if len(uniq) == 0:
        return pd.Series({'value': 'Missing/Not reported', 'conflict': False})
    if len(uniq) == 1:
        return pd.Series({'value': uniq[0], 'conflict': False})
    return pd.Series({'value': 'Conflicting clinical records', 'conflict': True})
demo_parts = []
for source, label in [('race', 'race_clean'), ('ethnicity', 'ethnicity_clean'), ('gender', 'sex_clean')]:
    if source in clinical.columns:
        z = clinical.groupby('patient_id')[source].apply(collapse_one).unstack()
        z = z.rename(columns={'value': label, 'conflict': label + '_conflict'})
        demo_parts.append(z)
demo = demo_parts[0]
for z in demo_parts[1:]:
    demo = demo.join(z, how='outer')
demo = demo.reset_index()
disp = meta[['patient_id', 'split']].merge(demo, on='patient_id', how='left')
for c in ['race_clean', 'ethnicity_clean', 'sex_clean']:
    if c in disp:
        disp[c] = disp[c].fillna('Missing/Not reported')
conf_cols = [c for c in disp.columns if c.endswith('_conflict')]
representation = []
for label, c in [('Race', 'race_clean'), ('Ethnicity', 'ethnicity_clean'), ('Sex', 'sex_clean')]:
    if c not in disp.columns:
        continue
    for group, g in disp.groupby(c, dropna=False):
        representation.append({'variable': label, 'group': str(group), 'total_n': int(len(g)), 'train_n': int((g['split'] == 'train').sum()), 'val_n': int((g['split'] == 'val').sum()), 'test_n': int((g['split'] == 'test').sum()), 'percent_of_cohort': 100 * len(g) / len(disp)})
representation = pd.DataFrame(representation)
representation.to_csv(TABLES / 'corrected_demographic_representation.csv', index=False)
testpred = preds[(preds['split'] == 'test') & (preds['target'] == 'global_inflammatory_fraction')].merge(disp.drop(columns=['split']), on='patient_id', how='left')
testpred['abs_error'] = (testpred['y_true'] - testpred['y_pred']).abs()

def bootstrap_mae(x, B=BOOTSTRAPS):
    x = np.asarray(x, float)
    vals = np.empty(B)
    for i in range(B):
        vals[i] = np.mean(rng.choice(x, size=len(x), replace=True))
    return (float(np.mean(x)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
fair = []
for variable, c in [('Race', 'race_clean'), ('Ethnicity', 'ethnicity_clean'), ('Sex', 'sex_clean')]:
    if c not in testpred.columns:
        continue
    for group, g in testpred.groupby(c):
        cohort_n = int((disp[c] == group).sum())
        test_n = len(g)
        eligible = cohort_n >= MIN_GROUP_N_TOTAL and test_n >= MIN_GROUP_N_TEST
        if eligible:
            mae, lo, hi = bootstrap_mae(g['abs_error'])
        else:
            mae = lo = hi = np.nan
        fair.append({'variable': variable, 'group': group, 'cohort_n': cohort_n, 'test_n': test_n, 'performance_estimable': eligible, 'test_mae': mae, 'mae_ci_low': lo, 'mae_ci_high': hi})
fair = pd.DataFrame(fair)
fair.to_csv(TABLES / 'corrected_subgroup_test_error.csv', index=False)
q = res[res['split'] == 'test'].sort_values('spearman_rho')
y = np.arange(len(q))
rho = q['spearman_rho'].to_numpy()
lo = rho - q['rho_ci_low'].to_numpy()
hi = q['rho_ci_high'].to_numpy() - rho
fig, ax = plt.subplots(figsize=(9, max(4, 0.55 * len(q))))
ax.errorbar(rho, y, xerr=np.vstack([lo, hi]), fmt='o', capsize=3)
ax.axvline(0, linewidth=1)
ax.set_yticks(y)
ax.set_yticklabels(q['target'])
ax.set_xlabel('Held-out test Spearman ρ (95% bootstrap CI)')
ax.set_title('Frozen GBM SSL representation predicts tumor–immune phenotypes')
fig.tight_layout()
fig.savefig(FIGS / 'corrected_test_prediction_with_ci.png', dpi=300, bbox_inches='tight')
plt.show()
q = rr[rr['split'] == 'test'].sort_values('spearman_rho')
y = np.arange(len(q))
rho = q['spearman_rho'].to_numpy()
lo = rho - q['rho_ci_low'].to_numpy()
hi = q['rho_ci_high'].to_numpy() - rho
fig, ax = plt.subplots(figsize=(9, max(4, 0.55 * len(q))))
ax.errorbar(rho, y, xerr=np.vstack([lo, hi]), fmt='o', capsize=3)
ax.axvline(0, linewidth=1)
ax.set_yticks(y)
ax.set_yticklabels(q['target'])
ax.set_xlabel('Held-out residual-target Spearman ρ (95% bootstrap CI)')
ax.set_title('Spatial information encoded beyond inflammatory abundance')
fig.tight_layout()
fig.savefig(FIGS / 'corrected_residual_topology_with_ci.png', dpi=300, bbox_inches='tight')
plt.show()
raw = res[res['split'] == 'test'][['target', 'target_col', 'n', 'spearman_rho', 'rho_ci_low', 'rho_ci_high', 'spearman_permutation_p', 'mae', 'r2']].rename(columns={'n': 'raw_test_n', 'spearman_rho': 'raw_test_rho', 'rho_ci_low': 'raw_rho_ci_low', 'rho_ci_high': 'raw_rho_ci_high', 'spearman_permutation_p': 'raw_rho_perm_p', 'mae': 'raw_test_mae', 'r2': 'raw_test_r2'})
resid = rr[rr['split'] == 'test'][['target', 'n', 'spearman_rho', 'rho_ci_low', 'rho_ci_high', 'spearman_permutation_p', 'mae', 'r2']].rename(columns={'n': 'residual_test_n', 'spearman_rho': 'residual_test_rho', 'rho_ci_low': 'residual_rho_ci_low', 'rho_ci_high': 'residual_rho_ci_high', 'spearman_permutation_p': 'residual_rho_perm_p', 'mae': 'residual_test_mae', 'r2': 'residual_test_r2'})
final = raw.merge(resid, on='target', how='left')
final.to_csv(TABLES / 'FINAL_CORRECTED_NOTEBOOK07_EVIDENCE.csv', index=False)
record = {'notebook': '07B_Validation_Fixes_and_Inference', 'patients_total': 377, 'split': {'train': 262, 'val': 57, 'test': 58}, 'endpoint_fix': {'problem': 'Notebook 07 fuzzy resolver duplicated tumor-to-inflammatory NN for the inflammatory-to-inflammatory endpoint', 'corrected_inflammatory_to_inflammatory_column': TARGETS['inflammatory_to_inflammatory_nn'], 'tumor_to_inflammatory_column': TARGETS['tumor_to_inflammatory_nn'], 'distinct_columns': TARGETS['inflammatory_to_inflammatory_nn'] != TARGETS['tumor_to_inflammatory_nn']}, 'clinical_metadata_repair': {'source': str(CLINICAL_PATH), 'strategy': 'collapse consistent non-missing patient-level demographic values; flag conflicts', 'sex_source_field': 'gender'}, 'inference': {'bootstrap_replicates': BOOTSTRAPS, 'permutation_replicates': PERMUTATIONS, 'test_set_used_for_fitting': False}, 'disparities_guardrails': {'race_ethnicity_sex_used_as_predictors': False, 'minimum_total_group_n': MIN_GROUP_N_TOTAL, 'minimum_test_group_n': MIN_GROUP_N_TEST, 'interpretation': 'underrepresentation and subgroup performance uncertainty; no causal demographic inference'}}
with open(OUT / 'notebook07B_results_record.json', 'w') as f:
    json.dump(record, f, indent=2)
