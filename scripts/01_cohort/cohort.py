PROJECT_ID = 'TCGA-GBM'
DOWNLOAD_ALL = False
USE_GOOGLE_DRIVE = True
MAX_DOWNLOAD_GB = 500.0
from pathlib import Path
import json, hashlib
import pandas as pd
import requests
from tqdm.auto import tqdm
if USE_GOOGLE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    ROOT = Path('/content/drive/MyDrive/AACR_GBM_Project')
else:
    ROOT = Path('/content/AACR_GBM_Project')
RAW = ROOT / 'data' / 'raw' / 'tcga_gbm_svs'
META = ROOT / 'data' / 'metadata'
QC = ROOT / 'outputs' / 'notebook01_qc'
for p in [RAW, META, QC]:
    p.mkdir(parents=True, exist_ok=True)
GDC_FILES = 'https://api.gdc.cancer.gov/files'
filters = {'op': 'and', 'content': [{'op': 'in', 'content': {'field': 'cases.project.project_id', 'value': [PROJECT_ID]}}, {'op': 'in', 'content': {'field': 'data_type', 'value': ['Slide Image']}}, {'op': 'in', 'content': {'field': 'data_format', 'value': ['SVS']}}, {'op': 'in', 'content': {'field': 'access', 'value': ['open']}}]}
fields = ','.join(['file_id', 'file_name', 'file_size', 'md5sum', 'data_type', 'data_format', 'access', 'cases.case_id', 'cases.submitter_id', 'cases.samples.sample_id', 'cases.samples.submitter_id', 'cases.samples.sample_type', 'cases.samples.portions.slides.slide_id', 'cases.samples.portions.slides.submitter_id', 'cases.samples.portions.slides.percent_tumor_cells', 'cases.samples.portions.slides.percent_tumor_nuclei', 'cases.samples.portions.slides.percent_normal_cells', 'cases.samples.portions.slides.percent_necrosis'])
params = {'filters': json.dumps(filters), 'fields': fields, 'format': 'JSON', 'size': '10000'}
r = requests.get(GDC_FILES, params=params, timeout=120)
r.raise_for_status()
hits = r.json()['data']['hits']
assert len(hits) > 0, 'No slides returned — inspect GDC query/API changes.'
rows = []
for h in hits:
    cases = h.get('cases') or [{}]
    for case in cases:
        samples = case.get('samples') or [{}]
        for sample in samples:
            slides = []
            for portion in sample.get('portions') or []:
                slides.extend(portion.get('slides') or [])
            slide = slides[0] if slides else {}
            rows.append({'project_id': PROJECT_ID, 'patient_id': case.get('submitter_id'), 'case_uuid': case.get('case_id'), 'sample_id': sample.get('submitter_id'), 'sample_uuid': sample.get('sample_id'), 'sample_type': sample.get('sample_type'), 'slide_id': slide.get('submitter_id'), 'slide_uuid': slide.get('slide_id'), 'file_id': h.get('file_id'), 'file_name': h.get('file_name'), 'file_size_bytes': h.get('file_size'), 'file_size_gb': (h.get('file_size') or 0) / 1000000000.0, 'md5sum': h.get('md5sum'), 'access': h.get('access'), 'percent_tumor_cells': slide.get('percent_tumor_cells'), 'percent_tumor_nuclei': slide.get('percent_tumor_nuclei'), 'percent_normal_cells': slide.get('percent_normal_cells'), 'percent_necrosis': slide.get('percent_necrosis')})
manifest = pd.DataFrame(rows).drop_duplicates(subset=['file_id']).sort_values(['patient_id', 'file_name']).reset_index(drop=True)
manifest['local_path'] = manifest['file_name'].map(lambda x: str(RAW / x))
manifest.to_csv(META / 'tcga_gbm_slide_manifest.csv', index=False)
summary = pd.DataFrame({'metric': ['slides', 'unique patients', 'unique samples', 'total size GB', 'median slide size GB', 'slides with tumor-cell metadata'], 'value': [len(manifest), manifest.patient_id.nunique(), manifest.sample_id.nunique(), round(manifest.file_size_gb.sum(), 2), round(manifest.file_size_gb.median(), 3), manifest.percent_tumor_cells.notna().sum()]})
summary.to_csv(QC / 'cohort_summary.csv', index=False)
dup_files = manifest.file_id.duplicated().sum()
missing_patient = manifest.patient_id.isna().sum()
assert dup_files == 0
assert missing_patient == 0, 'Patient IDs are needed for patient-grouped CV.'
patient_manifest = manifest.groupby('patient_id', as_index=False).agg(n_slides=('file_id', 'nunique'), total_size_gb=('file_size_gb', 'sum'), sample_types=('sample_type', lambda x: '; '.join(sorted(set(map(str, x.dropna()))))))
patient_manifest.to_csv(META / 'tcga_gbm_patient_manifest.csv', index=False)
gdc_manifest = manifest[['file_id', 'file_name', 'md5sum', 'file_size_bytes']].copy()
gdc_manifest.columns = ['id', 'filename', 'md5', 'size']
gdc_manifest.to_csv(META / 'gdc_manifest.tsv', sep='\t', index=False)

def md5_file(path, chunk=8 * 1024 * 1024):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()

def download_gdc_file(file_id, dest, expected_size=None, expected_md5=None):
    dest = Path(dest)
    part = dest.with_suffix(dest.suffix + '.part')
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        if expected_md5 is None or md5_file(dest) == expected_md5:
            return 'verified_existing'
    url = f'https://api.gdc.cancer.gov/data/{file_id}'
    headers = {}
    mode = 'wb'
    existing = 0
    if part.exists():
        existing = part.stat().st_size
        headers['Range'] = f'bytes={existing}-'
        mode = 'ab'
    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        if existing and r.status_code == 200:
            mode = 'wb'
            existing = 0
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0)) + existing
        with open(part, mode) as f, tqdm(total=total, initial=existing, unit='B', unit_scale=True, desc=dest.name) as bar:
            for chunk in r.iter_content(8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    part.replace(dest)
    if expected_size and dest.stat().st_size != expected_size:
        raise IOError(f'Size mismatch: {dest}')
    if expected_md5 and md5_file(dest) != expected_md5:
        raise IOError(f'MD5 mismatch: {dest}')
    return 'downloaded'
if DOWNLOAD_ALL:
    total_gb = manifest.file_size_gb.sum()
    if total_gb > MAX_DOWNLOAD_GB:
        raise RuntimeError(f'Cohort is {total_gb:.1f} GB, above MAX_DOWNLOAD_GB={MAX_DOWNLOAD_GB}. Increase guard deliberately if Drive has room.')
    statuses = []
    for row in manifest.itertuples():
        status = download_gdc_file(row.file_id, RAW / row.file_name, int(row.file_size_bytes), row.md5sum)
        statuses.append({'file_id': row.file_id, 'file_name': row.file_name, 'status': status})
    pd.DataFrame(statuses).to_csv(QC / 'download_status.csv', index=False)
records = []
for row in manifest.itertuples():
    p = RAW / row.file_name
    records.append({'file_id': row.file_id, 'file_name': row.file_name, 'exists': p.exists(), 'size_ok': p.exists() and p.stat().st_size == int(row.file_size_bytes), 'local_size_gb': p.stat().st_size / 1000000000.0 if p.exists() else 0})
verify = pd.DataFrame(records)
verify.to_csv(QC / 'local_file_audit.csv', index=False)
