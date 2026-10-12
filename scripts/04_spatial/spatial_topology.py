from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import json, math, os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 100)
ROOT = Path('/content/drive/MyDrive/AACR_GBM_Project')
N3 = ROOT / 'outputs/notebook03_cells/full_cohort_200roi'
TABLE_DIR = N3 / 'tables'
STATUS_DIR = N3 / 'status'
MANIFEST_DIR = N3 / 'manifests'
OUT = ROOT / 'outputs/notebook04_spatial_topology'
ROI_OUT = OUT / 'roi_features'
PATIENT_OUT = OUT / 'patient_features'
FIG_OUT = OUT / 'figures'
STATUS_OUT = OUT / 'status'
for p in [OUT, ROI_OUT, PATIENT_OUT, FIG_OUT, STATUS_OUT]:
    p.mkdir(parents=True, exist_ok=True)
selected_path = ROOT / 'data/metadata/tcga_gbm_notebook02_selected_slides.csv'
selected = pd.read_csv(selected_path)
selected['patient_id'] = selected['patient_id'].astype(str)

def n3_complete(pid):
    return (TABLE_DIR / f'{pid}__cells.csv.gz').exists() and (TABLE_DIR / f'{pid}__roi_features.csv').exists() and (STATUS_DIR / f'{pid}.done.json').exists()
selected['notebook03_complete'] = selected['patient_id'].map(n3_complete)
if not selected['notebook03_complete'].all():
    raise RuntimeError('Cell profiling is incomplete. Finish the missing patients before running spatial_topology.py.')
assert selected['patient_id'].nunique() == len(selected)
MIN_TYPE_PROB = 0.5
TUMOR_TYPE = 1
INFLAM_TYPE = 2
CONNECTIVE_TYPE = 3
DEAD_TYPE = 4
EPITHELIAL_TYPE = 5
CONTACT_RADII_LEVEL0 = [50, 100, 200]
ABS_IMMUNE_FRAC_THRESHOLDS = [0.05, 0.1, 0.2]
EPS = 1e-08

def safe_div(a, b):
    return np.nan if b == 0 else a / b

def shannon_from_counts(counts):
    x = np.asarray(counts, dtype=float)
    x = x[x > 0]
    if len(x) == 0:
        return np.nan
    p = x / x.sum()
    return float(-(p * np.log2(p)).sum())

def nearest_neighbor_summary(A, B, exclude_self=False):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if len(A) == 0 or len(B) == 0:
        return {'median': np.nan, 'mean': np.nan, 'p25': np.nan, 'p75': np.nan}
    tree = cKDTree(B)
    k = 2 if exclude_self and len(B) > 1 else 1
    d, _ = tree.query(A, k=k)
    if k == 2:
        d = d[:, 1]
    elif exclude_self and len(B) == 1:
        return {'median': np.nan, 'mean': np.nan, 'p25': np.nan, 'p75': np.nan}
    return {'median': float(np.median(d)), 'mean': float(np.mean(d)), 'p25': float(np.quantile(d, 0.25)), 'p75': float(np.quantile(d, 0.75))}

def fraction_with_neighbor(A, B, radius):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if len(A) == 0 or len(B) == 0:
        return np.nan
    tree = cKDTree(B)
    d, _ = tree.query(A, k=1)
    return float(np.mean(d <= radius))

def infer_mpp_from_roi_table(roi_df):
    candidates = [('mpp_x', 'mpp_y'), ('mpp_x_level0', 'mpp_y_level0'), ('level0_mpp_x', 'level0_mpp_y')]
    for cx, cy in candidates:
        if cx in roi_df.columns and cy in roi_df.columns:
            x = pd.to_numeric(roi_df[cx], errors='coerce').dropna()
            y = pd.to_numeric(roi_df[cy], errors='coerce').dropna()
            if len(x) and len(y):
                return (float(x.median()), float(y.median()))
    if 'mpp' in roi_df.columns:
        x = pd.to_numeric(roi_df['mpp'], errors='coerce').dropna()
        if len(x):
            v = float(x.median())
            return (v, v)
    return (None, None)

def convert_distance_px_to_um(d, mpp_x, mpp_y):
    if pd.isna(d) or mpp_x is None or mpp_y is None:
        return np.nan
    return float(d * math.sqrt(mpp_x * mpp_y))

def process_patient(pid):
    cell_path = TABLE_DIR / f'{pid}__cells.csv.gz'
    roi_path = TABLE_DIR / f'{pid}__roi_features.csv'
    cells = pd.read_csv(cell_path)
    rois = pd.read_csv(roi_path)
    if 'high_confidence' in cells.columns:
        hc = cells[cells['high_confidence'].astype(bool)].copy()
    else:
        hc = cells[pd.to_numeric(cells['type_prob'], errors='coerce') >= MIN_TYPE_PROB].copy()
    hc['type'] = pd.to_numeric(hc['type'], errors='coerce')
    hc['x_level0'] = pd.to_numeric(hc['x_level0'], errors='coerce')
    hc['y_level0'] = pd.to_numeric(hc['y_level0'], errors='coerce')
    hc = hc.dropna(subset=['type', 'x_level0', 'y_level0'])
    mpp_x, mpp_y = infer_mpp_from_roi_table(rois)
    roi_rows = []
    for roi_id, rr in rois.groupby('roi', sort=True):
        cc = hc[hc['roi'] == roi_id]
        tumor = cc.loc[cc['type'] == TUMOR_TYPE, ['x_level0', 'y_level0']].to_numpy()
        infl = cc.loc[cc['type'] == INFLAM_TYPE, ['x_level0', 'y_level0']].to_numpy()
        counts = cc['type'].value_counts().to_dict()
        n_total = len(cc)
        n_tumor = int(counts.get(TUMOR_TYPE, 0))
        n_infl = int(counts.get(INFLAM_TYPE, 0))
        n_conn = int(counts.get(CONNECTIVE_TYPE, 0))
        n_dead = int(counts.get(DEAD_TYPE, 0))
        n_epi = int(counts.get(EPITHELIAL_TYPE, 0))
        ti = nearest_neighbor_summary(tumor, infl)
        it = nearest_neighbor_summary(infl, tumor)
        ii = nearest_neighbor_summary(infl, infl, exclude_self=True)
        row = {'patient_id': pid, 'roi': roi_id, 'n_high_conf': n_total, 'n_neoplastic': n_tumor, 'n_inflammatory': n_infl, 'n_connective': n_conn, 'n_dead': n_dead, 'n_epithelial': n_epi, 'inflammatory_fraction': safe_div(n_infl, n_total), 'neoplastic_fraction': safe_div(n_tumor, n_total), 'connective_fraction': safe_div(n_conn, n_total), 'immune_to_tumor_ratio': safe_div(n_infl, n_tumor), 'composition_entropy': shannon_from_counts([n_tumor, n_infl, n_conn, n_dead, n_epi]), 'tumor_to_inflam_nn_median_px': ti['median'], 'inflam_to_tumor_nn_median_px': it['median'], 'inflam_to_inflam_nn_median_px': ii['median']}
        for rad in CONTACT_RADII_LEVEL0:
            row[f'tumor_with_inflam_within_{rad}px'] = fraction_with_neighbor(tumor, infl, rad)
            row[f'inflam_with_tumor_within_{rad}px'] = fraction_with_neighbor(infl, tumor, rad)
        if mpp_x is not None:
            row['tumor_to_inflam_nn_median_um'] = convert_distance_px_to_um(ti['median'], mpp_x, mpp_y)
            row['inflam_to_tumor_nn_median_um'] = convert_distance_px_to_um(it['median'], mpp_x, mpp_y)
            row['inflam_to_inflam_nn_median_um'] = convert_distance_px_to_um(ii['median'], mpp_x, mpp_y)
        roi_rows.append(row)
    roi_feat = pd.DataFrame(roi_rows)
    p = {'patient_id': pid, 'n_rois': len(roi_feat), 'n_high_conf_nuclei': int(roi_feat['n_high_conf'].sum()), 'n_neoplastic': int(roi_feat['n_neoplastic'].sum()), 'n_inflammatory': int(roi_feat['n_inflammatory'].sum()), 'n_connective': int(roi_feat['n_connective'].sum()), 'n_dead': int(roi_feat['n_dead'].sum()), 'n_epithelial': int(roi_feat['n_epithelial'].sum())}
    p['global_inflammatory_fraction'] = safe_div(p['n_inflammatory'], p['n_high_conf_nuclei'])
    p['global_neoplastic_fraction'] = safe_div(p['n_neoplastic'], p['n_high_conf_nuclei'])
    p['global_connective_fraction'] = safe_div(p['n_connective'], p['n_high_conf_nuclei'])
    p['global_immune_to_tumor_ratio'] = safe_div(p['n_inflammatory'], p['n_neoplastic'])
    inf = roi_feat['inflammatory_fraction'].fillna(0).to_numpy(float)
    p['roi_inflammatory_mean'] = float(np.mean(inf))
    p['roi_inflammatory_median'] = float(np.median(inf))
    p['roi_inflammatory_sd'] = float(np.std(inf, ddof=1))
    p['roi_inflammatory_iqr'] = float(np.quantile(inf, 0.75) - np.quantile(inf, 0.25))
    p['roi_inflammatory_cv'] = safe_div(p['roi_inflammatory_sd'], p['roi_inflammatory_mean'])
    p['zero_inflammatory_roi_fraction'] = float(np.mean(roi_feat['n_inflammatory'].to_numpy() == 0))
    p['inflammatory_positive_roi_fraction'] = 1 - p['zero_inflammatory_roi_fraction']
    inf_counts = roi_feat['n_inflammatory'].to_numpy(float)
    p['inflammatory_spatial_entropy'] = shannon_from_counts(inf_counts)
    if np.isfinite(p['inflammatory_spatial_entropy']) and len(roi_feat) > 1:
        p['inflammatory_spatial_entropy_normalized'] = p['inflammatory_spatial_entropy'] / np.log2(len(roi_feat))
    else:
        p['inflammatory_spatial_entropy_normalized'] = np.nan
    p['roi_composition_entropy_mean'] = float(roi_feat['composition_entropy'].mean())
    p['roi_composition_entropy_sd'] = float(roi_feat['composition_entropy'].std())
    for col in ['tumor_to_inflam_nn_median_px', 'inflam_to_tumor_nn_median_px', 'inflam_to_inflam_nn_median_px', 'tumor_to_inflam_nn_median_um', 'inflam_to_tumor_nn_median_um', 'inflam_to_inflam_nn_median_um']:
        if col in roi_feat.columns:
            p[f'{col}_patient_median'] = float(roi_feat[col].median())
    for rad in CONTACT_RADII_LEVEL0:
        for prefix in ['tumor_with_inflam', 'inflam_with_tumor']:
            col = f'{prefix}_within_{rad}px'
            p[f'{col}_mean'] = float(roi_feat[col].mean())
    for thr in ABS_IMMUNE_FRAC_THRESHOLDS:
        p[f'immune_rich_roi_fraction_ge_{int(thr * 100)}pct'] = float(np.mean(inf >= thr))
    q75 = float(np.quantile(inf, 0.75))
    p['patient_inflammatory_q75'] = q75
    p['relative_hotspot_roi_fraction'] = float(np.mean(inf >= q75))
    return (roi_feat, p)
patient_rows = []
errors = []
for _, s in tqdm(selected.sort_values('patient_id').iterrows(), total=len(selected)):
    pid = str(s.patient_id)
    roi_save = ROI_OUT / f'{pid}__spatial_roi_features.csv.gz'
    p_save = PATIENT_OUT / f'{pid}__spatial_patient_features.json'
    done = STATUS_OUT / f'{pid}.done.json'
    if roi_save.exists() and p_save.exists() and done.exists():
        with open(p_save) as f:
            patient_rows.append(json.load(f))
        continue
    try:
        roi_feat, p = process_patient(pid)
        p['file_id'] = str(s.file_id)
        p['split'] = str(s['split'])
        assert len(roi_feat) == 200, f'{pid}: expected 200 ROIs, got {len(roi_feat)}'
        tmp_roi = roi_save.with_suffix(roi_save.suffix + '.tmp')
        roi_feat.to_csv(tmp_roi, index=False, compression='gzip')
        os.replace(tmp_roi, roi_save)
        tmp_p = p_save.with_suffix('.tmp')
        with open(tmp_p, 'w') as f:
            json.dump(p, f, indent=2, allow_nan=True)
        os.replace(tmp_p, p_save)
        with open(done, 'w') as f:
            json.dump({'patient_id': pid, 'n_rois': len(roi_feat), 'complete': True}, f)
        patient_rows.append(p)
    except Exception as e:
        errors.append({'patient_id': pid, 'error': repr(e)})
        print('ERROR', pid, repr(e))
patient_features = pd.DataFrame(patient_rows)
patient_features.to_csv(OUT / 'tcga_gbm_patient_spatial_topology.csv', index=False)
pf = pd.read_csv(OUT / 'tcga_gbm_patient_spatial_topology.csv')
assert len(pf) == 377
assert pf['patient_id'].nunique() == 377
assert (pf['n_rois'] == 200).all()
assert int(pf['n_rois'].sum()) == 75400
summary_cols = ['n_high_conf_nuclei', 'global_inflammatory_fraction', 'global_neoplastic_fraction', 'global_connective_fraction', 'zero_inflammatory_roi_fraction', 'inflammatory_spatial_entropy_normalized', 'roi_inflammatory_sd', 'roi_inflammatory_iqr', 'roi_composition_entropy_mean']
for c in ['tumor_to_inflam_nn_median_um_patient_median', 'inflam_to_tumor_nn_median_um_patient_median', 'inflam_to_inflam_nn_median_um_patient_median', 'tumor_to_inflam_nn_median_px_patient_median', 'inflam_to_tumor_nn_median_px_patient_median', 'inflam_to_inflam_nn_median_px_patient_median']:
    if c in pf.columns:
        summary_cols.append(c)
summary = pf[summary_cols].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).T
summary.to_csv(OUT / 'cohort_spatial_topology_summary.csv')
candidate_spatial = ['zero_inflammatory_roi_fraction', 'inflammatory_spatial_entropy_normalized', 'roi_inflammatory_sd', 'roi_inflammatory_iqr', 'roi_composition_entropy_mean']
for c in ['tumor_to_inflam_nn_median_um_patient_median', 'inflam_to_tumor_nn_median_um_patient_median', 'inflam_to_inflam_nn_median_um_patient_median', 'tumor_to_inflam_nn_median_px_patient_median', 'inflam_to_tumor_nn_median_px_patient_median', 'inflam_to_inflam_nn_median_px_patient_median']:
    if c in pf.columns:
        candidate_spatial.append(c)
rows = []
for c in candidate_spatial:
    d = pf[['global_inflammatory_fraction', c]].dropna()
    if len(d) >= 10:
        rho, pval = spearmanr(d['global_inflammatory_fraction'], d[c])
        rows.append({'metric': c, 'n': len(d), 'spearman_rho_vs_inflammatory_fraction': rho, 'p_value': pval})
corr = pd.DataFrame(rows).sort_values('spearman_rho_vs_inflammatory_fraction')
corr.to_csv(OUT / 'spatial_metrics_vs_inflammatory_abundance.csv', index=False)
feature_candidates = ['global_inflammatory_fraction', 'global_neoplastic_fraction', 'global_connective_fraction', 'zero_inflammatory_roi_fraction', 'inflammatory_spatial_entropy_normalized', 'roi_inflammatory_sd', 'roi_inflammatory_iqr', 'roi_composition_entropy_mean', 'immune_rich_roi_fraction_ge_5pct', 'immune_rich_roi_fraction_ge_10pct', 'immune_rich_roi_fraction_ge_20pct']
for c in ['tumor_to_inflam_nn_median_um_patient_median', 'inflam_to_tumor_nn_median_um_patient_median', 'inflam_to_inflam_nn_median_um_patient_median', 'tumor_to_inflam_nn_median_px_patient_median', 'inflam_to_tumor_nn_median_px_patient_median', 'inflam_to_inflam_nn_median_px_patient_median']:
    if c in pf.columns:
        feature_candidates.append(c)
X = pf[feature_candidates].replace([np.inf, -np.inf], np.nan)
keep_cols = [c for c in X.columns if X[c].notna().mean() >= 0.7 and X[c].nunique(dropna=True) > 1]
X = X[keep_cols].copy()
X = X.fillna(X.median(numeric_only=True))
Xs = StandardScaler().fit_transform(X)
pca = PCA(n_components=2, random_state=2026)
Z = pca.fit_transform(Xs)
pca_df = pf[['patient_id', 'split']].copy()
pca_df['PC1'] = Z[:, 0]
pca_df['PC2'] = Z[:, 1]
pca_df['global_inflammatory_fraction'] = pf['global_inflammatory_fraction']
pca_df.to_csv(OUT / 'tumor_immune_architecture_pca.csv', index=False)
loadings = pd.DataFrame(pca.components_.T, index=keep_cols, columns=['PC1_loading', 'PC2_loading'])
loadings.to_csv(OUT / 'tumor_immune_architecture_pca_loadings.csv')
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(pf['global_inflammatory_fraction'].dropna(), bins=30)
ax.set_xlabel('Predicted inflammatory nuclei fraction')
ax.set_ylabel('Patients')
ax.set_title('Inter-patient heterogeneity of inflammatory burden in TCGA-GBM')
fig.tight_layout()
fig.savefig(FIG_OUT / 'Fig1_inflammatory_fraction_distribution.png', dpi=300)
plt.show()
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(pf['global_inflammatory_fraction'], pf['roi_inflammatory_iqr'], alpha=0.7)
ax.set_xlabel('Global predicted inflammatory fraction')
ax.set_ylabel('Across-ROI inflammatory-fraction IQR')
ax.set_title('Inflammatory abundance and spatial heterogeneity are distinct axes')
fig.tight_layout()
fig.savefig(FIG_OUT / 'Fig2_abundance_vs_spatial_heterogeneity.png', dpi=300)
plt.show()
fig, ax = plt.subplots(figsize=(7, 5))
sc = ax.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['global_inflammatory_fraction'], alpha=0.8)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)')
ax.set_title('Unsupervised map of GBM tumor–immune architecture')
fig.colorbar(sc, ax=ax, label='Predicted inflammatory fraction')
fig.tight_layout()
fig.savefig(FIG_OUT / 'Fig3_tumor_immune_architecture_PCA.png', dpi=300)
plt.show()
tmp = pf.copy()
tmp['abundance_rank'] = tmp['global_inflammatory_fraction'].rank(pct=True)
tmp['heterogeneity_rank'] = tmp['roi_inflammatory_iqr'].rank(pct=True)
examples = pd.concat([tmp.nsmallest(3, 'global_inflammatory_fraction').assign(example_group='low inflammatory burden'), tmp.nlargest(3, 'global_inflammatory_fraction').assign(example_group='high inflammatory burden'), tmp.nlargest(3, 'roi_inflammatory_iqr').assign(example_group='high spatial heterogeneity')]).drop_duplicates('patient_id')
cols = ['patient_id', 'split', 'example_group', 'global_inflammatory_fraction', 'roi_inflammatory_iqr', 'zero_inflammatory_roi_fraction', 'inflammatory_spatial_entropy_normalized']
examples[cols].to_csv(OUT / 'illustrative_patient_candidates.csv', index=False)
record = {'notebook': '04_spatial_tumor_immune_topology', 'patients': int(len(pf)), 'total_rois': int(pf['n_rois'].sum()), 'total_high_confidence_nuclei': int(pf['n_high_conf_nuclei'].sum()), 'total_predicted_inflammatory_nuclei': int(pf['n_inflammatory'].sum()), 'total_predicted_neoplastic_nuclei': int(pf['n_neoplastic'].sum()), 'median_patient_inflammatory_fraction': float(pf['global_inflammatory_fraction'].median()), 'median_zero_inflammatory_roi_fraction': float(pf['zero_inflammatory_roi_fraction'].median()), 'pca_pc1_variance': float(pca.explained_variance_ratio_[0]), 'pca_pc2_variance': float(pca.explained_variance_ratio_[1]), 'terminology': 'HoVer-Net–predicted inflammatory nuclei; not confirmed T cells/TILs'}
with open(OUT / 'notebook04_results_record.json', 'w') as f:
    json.dump(record, f, indent=2)
