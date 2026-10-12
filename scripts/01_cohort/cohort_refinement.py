from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import pandas as pd
import numpy as np
import requests, json, re
ROOT = Path('/content/drive/MyDrive/AACR_GBM_Project')
META = ROOT / 'data' / 'metadata'
OUT = ROOT / 'outputs' / 'notebook01A_cohort_refinement'
META.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
SLIDE_MANIFEST = META / 'tcga_gbm_slide_manifest.csv'
assert SLIDE_MANIFEST.exists(), f'Missing {SLIDE_MANIFEST}. Run cohort.py first.'
slides = pd.read_csv(SLIDE_MANIFEST)
primary = slides.loc[slides['sample_type'].eq('Primary Tumor')].copy()

def parse_slide_class(row):
    text = ' '.join([str(row.get('slide_id', '')), str(row.get('file_name', ''))]).upper()
    m = re.search('(?:^|[-_.])(DX|BS)\\d*(?:[-_.]|$)', text)
    return m.group(1) if m else 'OTHER/UNKNOWN'
primary['slide_class'] = primary.apply(parse_slide_class, axis=1)
primary['is_diagnostic'] = primary['slide_class'].eq('DX')
primary['is_biospecimen'] = primary['slide_class'].eq('BS')
audit = primary.groupby('slide_class').agg(slides=('file_id', 'size'), patients=('patient_id', 'nunique'), source_GB=('file_size_gb', 'sum')).sort_values('slides', ascending=False)
unknown = primary.loc[primary.slide_class.eq('OTHER/UNKNOWN'), ['patient_id', 'slide_id', 'file_name', 'sample_id']]
GDC_CASES = 'https://api.gdc.cancer.gov/cases'
fields = ['case_id', 'submitter_id', 'project.project_id', 'demographic.age_at_index', 'demographic.days_to_birth', 'demographic.days_to_death', 'demographic.gender', 'demographic.race', 'demographic.ethnicity', 'demographic.vital_status', 'diagnoses.primary_diagnosis', 'diagnoses.age_at_diagnosis', 'diagnoses.days_to_last_follow_up', 'diagnoses.days_to_last_known_disease_status', 'diagnoses.tumor_grade', 'diagnoses.morphology', 'diagnoses.site_of_resection_or_biopsy', 'diagnoses.tissue_or_organ_of_origin', 'diagnoses.prior_malignancy', 'diagnoses.synchronous_malignancy']
filters = {'op': 'in', 'content': {'field': 'project.project_id', 'value': ['TCGA-GBM']}}
params = {'filters': json.dumps(filters), 'fields': ','.join(fields), 'format': 'JSON', 'size': '2000'}
r = requests.get(GDC_CASES, params=params, timeout=120)
r.raise_for_status()
case_hits = r.json()['data']['hits']

def first_or_none(x):
    return x[0] if isinstance(x, list) and len(x) else {}
rows = []
for h in case_hits:
    d = h.get('demographic') or {}
    dx = first_or_none(h.get('diagnoses') or [])
    age_days = dx.get('age_at_diagnosis')
    age_years = age_days / 365.25 if isinstance(age_days, (int, float)) else None
    if age_years is None:
        age_idx = d.get('age_at_index')
        if isinstance(age_idx, (int, float)):
            age_years = age_idx
    rows.append({'patient_id': h.get('submitter_id'), 'case_uuid': h.get('case_id'), 'age_at_diagnosis_years': age_years, 'gender': d.get('gender'), 'race': d.get('race'), 'ethnicity': d.get('ethnicity'), 'vital_status': d.get('vital_status'), 'days_to_death': d.get('days_to_death'), 'days_to_last_follow_up': dx.get('days_to_last_follow_up'), 'primary_diagnosis': dx.get('primary_diagnosis'), 'tumor_grade': dx.get('tumor_grade'), 'morphology': dx.get('morphology'), 'site_of_resection_or_biopsy': dx.get('site_of_resection_or_biopsy'), 'tissue_or_organ_of_origin': dx.get('tissue_or_organ_of_origin'), 'prior_malignancy': dx.get('prior_malignancy'), 'synchronous_malignancy': dx.get('synchronous_malignancy')})
clinical = pd.DataFrame(rows).drop_duplicates('patient_id')
clinical.to_csv(META / 'tcga_gbm_clinical_indexed.csv', index=False)
cols = ['age_at_diagnosis_years', 'gender', 'race', 'ethnicity', 'vital_status', 'days_to_death', 'days_to_last_follow_up', 'primary_diagnosis', 'tumor_grade', 'site_of_resection_or_biopsy']
completeness = pd.DataFrame({'field': cols, 'available_n': [clinical[c].notna().sum() for c in cols], 'available_pct': [100 * clinical[c].notna().mean() for c in cols]}).sort_values('available_pct', ascending=False)
completeness.to_csv(OUT / 'clinical_completeness.csv', index=False)
dx = primary.loc[primary['is_diagnostic']].copy()
dx = dx.merge(clinical, on='patient_id', how='left', suffixes=('', '_clinical'))
for c in ['percent_tumor_cells', 'percent_tumor_nuclei', 'percent_normal_cells', 'percent_necrosis']:
    dx[c] = pd.to_numeric(dx[c], errors='coerce')
per_patient = dx.groupby('patient_id').agg(n_dx_slides=('file_id', 'size'), dx_source_GB=('file_size_gb', 'sum')).reset_index()
dx.to_csv(META / 'tcga_gbm_primary_dx_all.csv', index=False)
dx['has_tumor_cells'] = dx['percent_tumor_cells'].notna().astype(int)
dx['has_necrosis'] = dx['percent_necrosis'].notna().astype(int)
dx_ranked = dx.sort_values(by=['patient_id', 'has_tumor_cells', 'percent_tumor_cells', 'has_necrosis', 'percent_necrosis', 'file_id'], ascending=[True, False, False, False, True, True], na_position='last').copy()
dx_ranked['pre_qc_rank'] = dx_ranked.groupby('patient_id').cumcount() + 1
preferred = dx_ranked.loc[dx_ranked['pre_qc_rank'].eq(1)].copy()
assert preferred['patient_id'].is_unique
dx_ranked.to_csv(META / 'tcga_gbm_primary_dx_ranked.csv', index=False)
preferred.to_csv(META / 'tcga_gbm_primary_dx_preferred.csv', index=False)
from sklearn.model_selection import train_test_split
SEED = 2026
patients = np.array(sorted(dx['patient_id'].dropna().unique()))
train_ids, temp_ids = train_test_split(patients, test_size=0.3, random_state=SEED)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=SEED)
split_map = {p: 'train' for p in train_ids}
split_map.update({p: 'val' for p in val_ids})
split_map.update({p: 'test' for p in test_ids})
dx_ranked['split'] = dx_ranked['patient_id'].map(split_map)
preferred['split'] = preferred['patient_id'].map(split_map)
sets = {k: set(preferred.loc[preferred['split'].eq(k), 'patient_id']) for k in ['train', 'val', 'test']}
assert sets['train'].isdisjoint(sets['val'])
assert sets['train'].isdisjoint(sets['test'])
assert sets['val'].isdisjoint(sets['test'])
preferred.to_csv(META / 'tcga_gbm_primary_dx_preferred_with_splits.csv', index=False)
dx_ranked.to_csv(META / 'tcga_gbm_primary_dx_ranked_with_splits.csv', index=False)

def gdc_download_manifest(df, path):
    out = df[['file_id', 'file_name', 'md5sum', 'file_size_bytes']].copy()
    out.columns = ['id', 'filename', 'md5', 'size']
    out.to_csv(path, sep='\t', index=False)
    return out
all_dx_manifest = gdc_download_manifest(dx_ranked, META / 'gdc_manifest_primary_dx_all.tsv')
preferred_manifest = gdc_download_manifest(preferred, META / 'gdc_manifest_primary_dx_preferred.tsv')
summary = pd.DataFrame([['Full Notebook 01 WSI inventory', len(slides), slides.patient_id.nunique(), slides.file_size_gb.sum()], ['Primary-tumor slides', len(primary), primary.patient_id.nunique(), primary.file_size_gb.sum()], ['Primary diagnostic (DX) slides', len(dx_ranked), dx_ranked.patient_id.nunique(), dx_ranked.file_size_gb.sum()], ['Preferred pre-QC DX cohort', len(preferred), preferred.patient_id.nunique(), preferred.file_size_gb.sum()]], columns=['cohort', 'slides', 'patients', 'source_GB'])
summary.to_csv(OUT / 'cohort_refinement_summary.csv', index=False)
path_qc = pd.DataFrame({'field': ['percent_tumor_cells', 'percent_tumor_nuclei', 'percent_normal_cells', 'percent_necrosis'], 'available_n': [preferred[c].notna().sum() for c in ['percent_tumor_cells', 'percent_tumor_nuclei', 'percent_normal_cells', 'percent_necrosis']]})
path_qc['available_pct'] = 100 * path_qc['available_n'] / len(preferred)
