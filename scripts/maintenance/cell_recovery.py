import sys, subprocess, json
import torch
if not torch.cuda.is_available():
    raise RuntimeError('Use a GPU runtime for HoVer-Net inference.')
subprocess.run(['apt-get', '-qq', 'update'], check=True)
subprocess.run(['apt-get', '-qq', 'install', '-y', 'openslide-tools'], check=True, stdout=subprocess.DEVNULL)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'openslide-python', 'opencv-python-headless', 'pandas', 'pyarrow', 'tqdm', 'scipy', 'scikit-image', 'docopt', 'imgaug'], check=True)
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import pandas as pd
import numpy as np
import shutil, time, gc, json, subprocess
import openslide
from PIL import Image
ROOT = Path('/content/drive/MyDrive/AACR_GBM_Project')
COORD_DIR = ROOT / 'outputs/notebook02_wsi_qc/coordinates'
META = ROOT / 'data' / 'metadata'
N2 = ROOT / 'outputs' / 'notebook02_wsi_qc'
COORDS = N2 / 'coordinates'
OUT = ROOT / 'outputs' / 'notebook03_cells'
PROD = OUT / 'full_cohort_200roi'
TABLE_OUT = PROD / 'tables'
STATUS_OUT = PROD / 'status'
MANIFEST_OUT = PROD / 'manifests'
for p in [OUT, PROD, TABLE_OUT, STATUS_OUT, MANIFEST_OUT]:
    p.mkdir(parents=True, exist_ok=True)
SELECTED = META / 'tcga_gbm_notebook02_selected_slides.csv'
assert SELECTED.exists(), f'Missing: {SELECTED}'
selected = pd.read_csv(SELECTED)
assert selected['patient_id'].nunique() == len(selected)
HOVERNET = Path('/content/hover_net')
if HOVERNET.exists():
    shutil.rmtree(HOVERNET)
subprocess.run(['git', 'clone', '-q', 'https://github.com/vqdang/hover_net.git', str(HOVERNET)], check=True)
import importlib.util
imgaug_spec = importlib.util.find_spec('imgaug')
assert imgaug_spec is not None, 'imgaug is not installed.'
imgaug_file = Path(imgaug_spec.origin).parent / 'imgaug.py'
txt = imgaug_file.read_text()
replacements = {'NP_FLOAT_TYPES = set(np.sctypes["float"])': 'NP_FLOAT_TYPES = {np.float16, np.float32, np.float64}', 'NP_INT_TYPES = set(np.sctypes["int"])': 'NP_INT_TYPES = {np.int8, np.int16, np.int32, np.int64}', 'NP_UINT_TYPES = set(np.sctypes["uint"])': 'NP_UINT_TYPES = {np.uint8, np.uint16, np.uint32, np.uint64}'}
for old, new in replacements.items():
    txt = txt.replace(old, new)
imgaug_file.write_text(txt)
tile_file = HOVERNET / 'infer' / 'tile.py'
txt = tile_file.read_text()
txt = txt.replace('np.lib.pad', 'np.pad')
tile_file.write_text(txt)
verify = subprocess.run([sys.executable, '-c', 'import imgaug, numpy as np'], capture_output=True, text=True)
if verify.returncode != 0:
    raise RuntimeError('imgaug compatibility verification failed.')
assert 'np.lib.pad' not in tile_file.read_text()
MODEL_DIR = ROOT / 'models' / 'hovernet'
MODEL_PATH = MODEL_DIR / 'hovernet_fast_pannuke_type_tf2pytorch.tar'
assert MODEL_PATH.exists(), f'PanNuke HoVer-Net checkpoint not found at:\n{MODEL_PATH}'
SOURCE_TILE_PX = 256
REGION_PX_40X = 512
PRODUCTION_ROIS = 200
REFERENCE_MASTER_ROIS = 500
RANDOM_SEED = 2026
MIN_TYPE_PROB = 0.5
TARGET_PATIENT = 'TCGA-02-0099'
TYPE_NAMES = {0: 'unclassified', 1: 'neoplastic', 2: 'inflammatory', 3: 'connective', 4: 'dead', 5: 'epithelial'}
GDC_DATA = 'https://api.gdc.cancer.gov/data'
TMP = Path('/content/tcga_n3c_tmp')
ROI_TMP = Path('/content/hovernet_rois')
HOVER_TMP = Path('/content/hovernet_output')
TMP.mkdir(exist_ok=True)
assert COORD_DIR.exists(), f'Coordinate directory not found: {COORD_DIR}'

def coord_file_for(patient_id, file_id):
    candidates = [COORD_DIR / f'{patient_id}__{file_id}__coords.csv.gz', COORD_DIR / f'{patient_id}__{file_id}__coordinates.csv.gz', COORD_DIR / f'{file_id}__coords.csv.gz', COORD_DIR / f'{patient_id}__coords.csv.gz']
    for p in candidates:
        if p.exists():
            return p
    matches = sorted(COORD_DIR.glob(f'*{file_id}*.csv.gz'))
    if len(matches) == 1:
        return matches[0]
    matches = sorted(COORD_DIR.glob(f'*{patient_id}*.csv.gz'))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f'Could not uniquely locate coordinate file for {patient_id} / {file_id}')

def coordinate_count(patient_id, file_id):
    p = coord_file_for(patient_id, file_id)
    return len(pd.read_csv(p, usecols=[0]))
CURL_CONNECT_TIMEOUT = 30
CURL_MAX_TIME = 600
MAX_RETRIES = 4

def download_gdc_file(file_id, dest, expected_bytes=None):
    url = f'{GDC_DATA}/{file_id}'
    last = None
    for attempt in range(1, MAX_RETRIES + 1):
        if dest.exists():
            dest.unlink()
        cmd = ['curl', '-L', '--fail', '--show-error', '--connect-timeout', str(CURL_CONNECT_TIMEOUT), '--max-time', str(CURL_MAX_TIME), '--retry', '2', '--retry-delay', '5', '--retry-all-errors', '-o', str(dest), url]
        try:
            result = subprocess.run(cmd, text=True, capture_output=True)
            if result.returncode != 0:
                raise RuntimeError(f'curl exit {result.returncode}: {result.stderr[-1200:]}')
            if not dest.exists() or dest.stat().st_size == 0:
                raise IOError('download produced no file')
            if expected_bytes is not None and dest.stat().st_size != int(expected_bytes):
                raise IOError(f'file-size mismatch: got {dest.stat().st_size}, expected {int(expected_bytes)}')
            return dest
        except Exception as e:
            last = e
            print(f'Attempt {attempt} failed:', e)
            if dest.exists():
                dest.unlink()
            if attempt < MAX_RETRIES:
                time.sleep(min(30, 5 * attempt))
    raise RuntimeError(f'GDC download failed after {MAX_RETRIES} attempts: {last}')

def extract_region_40x(slide, x0, y0, source_downsample):
    physical_w0 = int(round(SOURCE_TILE_PX * float(source_downsample)))
    region = slide.read_region((int(x0), int(y0)), 0, (physical_w0, physical_w0)).convert('RGB')
    return region.resize((REGION_PX_40X, REGION_PX_40X), Image.Resampling.LANCZOS)

def production_sample(pid, fid, seed=RANDOM_SEED):
    df = pd.read_csv(coord_file_for(pid, fid))
    if len(df) < PRODUCTION_ROIS:
        raise ValueError(f'{pid}: only {len(df)} coordinates; need {PRODUCTION_ROIS}')
    n_master = min(REFERENCE_MASTER_ROIS, len(df))
    d = df.sort_values(['y_level0', 'x_level0']).reset_index(drop=True)
    rng = np.random.default_rng(seed + sum(map(ord, str(pid))))
    bins = np.array_split(np.arange(len(d)), n_master)
    picks = np.array([rng.choice(b) for b in bins if len(b)], dtype=int)
    rng.shuffle(picks)
    master = d.iloc[picks].copy().reset_index(drop=True)
    master['master_order'] = np.arange(1, len(master) + 1)
    out = master.iloc[:PRODUCTION_ROIS].copy().reset_index(drop=True)
    out['patient_id'] = pid
    out['file_id'] = fid
    out['sampling_exact_03b_design'] = len(df) >= REFERENCE_MASTER_ROIS
    return out
rows = []
for _, s in selected.iterrows():
    pid, fid = (str(s.patient_id), str(s.file_id))
    n = coordinate_count(pid, fid)
    rows.append({'patient_id': pid, 'file_id': fid, 'split': s.get('split', np.nan), 'coord_count': n, 'can_sample_200': n >= PRODUCTION_ROIS, 'can_match_500_master_design': n >= REFERENCE_MASTER_ROIS})
preflight = pd.DataFrame(rows)
PREFLIGHT_PATH = MANIFEST_OUT / 'full_cohort_preflight.csv'
preflight.to_csv(PREFLIGHT_PATH, index=False)
SAMPLE_MANIFEST = MANIFEST_OUT / 'full_cohort_200roi_manifest.csv.gz'
if SAMPLE_MANIFEST.exists():
    samples = pd.read_csv(SAMPLE_MANIFEST)
else:
    parts, skipped = ([], [])
    for _, s in selected.iterrows():
        pid, fid = (str(s.patient_id), str(s.file_id))
        try:
            q = production_sample(pid, fid)
            q['split'] = s.get('split', np.nan)
            parts.append(q)
        except Exception as e:
            skipped.append({'patient_id': pid, 'file_id': fid, 'reason': str(e)})
    samples = pd.concat(parts, ignore_index=True)
    samples.to_csv(SAMPLE_MANIFEST, index=False, compression='gzip')
    if skipped:
        pd.DataFrame(skipped).to_csv(MANIFEST_OUT / 'sampling_skipped_patients.csv', index=False)
per_patient = samples.groupby('patient_id').size()
assert (per_patient == PRODUCTION_ROIS).all()
assert samples[['patient_id', 'tile_index']].duplicated().sum() == 0
target_row = selected[selected['patient_id'].astype(str) == TARGET_PATIENT].copy()
assert len(target_row) == 1, f'Expected exactly one row for {TARGET_PATIENT}, found {len(target_row)}'
target_samples = samples[samples['patient_id'].astype(str) == TARGET_PATIENT].copy()
assert len(target_samples) == PRODUCTION_ROIS, f'Expected {PRODUCTION_ROIS} frozen ROIs for {TARGET_PATIENT}, found {len(target_samples)}'

def patient_complete(pid):
    return (TABLE_OUT / f'{pid}__cells.csv.gz').exists() and (TABLE_OUT / f'{pid}__roi_features.csv').exists() and (STATUS_OUT / f'{pid}.done.json').exists()
run_cohort = selected[selected['patient_id'].astype(str) == TARGET_PATIENT].copy().reset_index(drop=True)
assert len(run_cohort) == 1, f'Expected one target patient, found {len(run_cohort)}'
for run_i, (_, s) in enumerate(run_cohort.iterrows(), start=1):
    pid, fid = (str(s.patient_id), str(s.file_id))
    cell_path = TABLE_OUT / f'{pid}__cells.csv.gz'
    roi_path = TABLE_OUT / f'{pid}__roi_features.csv'
    done_path = STATUS_OUT / f'{pid}.done.json'
    error_path = STATUS_OUT / f'{pid}.error.txt'
    if patient_complete(pid):
        print(f'[{run_i}/{len(run_cohort)}] OK {pid}: complete — skipping')
        continue
    print(f"[{run_i}/{len(run_cohort)}] Processing {pid} | {s.get('split', '')}")
    for p in [cell_path, roi_path, done_path, error_path]:
        if p.exists():
            p.unlink()
    for p in [ROI_TMP, HOVER_TMP]:
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True)
    sub = samples[samples['patient_id'].astype(str) == pid].copy()
    sub = sub.sort_values('master_order').reset_index(drop=True)
    assert len(sub) == PRODUCTION_ROIS
    local = TMP / f'{fid}.svs'
    try:
        download_gdc_file(fid, local)
        slide = openslide.OpenSlide(str(local))
        try:
            for _, r in sub.iterrows():
                im = extract_region_40x(slide, r.x_level0, r.y_level0, r.downsample)
                im.save(ROI_TMP / f'{pid}__{int(r.tile_index):07d}.png')
        finally:
            slide.close()
        assert len(list(ROI_TMP.glob('*.png'))) == PRODUCTION_ROIS
        cmd = [sys.executable, 'run_infer.py', '--gpu=0', '--nr_types=6', '--type_info_path=type_info.json', f'--model_path={MODEL_PATH}', '--model_mode=fast', 'tile', f'--input_dir={ROI_TMP}', f'--output_dir={HOVER_TMP}', '--mem_usage=0.1']
        result = subprocess.run(cmd, cwd=str(HOVERNET))
        if result.returncode != 0:
            raise RuntimeError(f'HoVer-Net return code {result.returncode}')
        json_files = sorted(HOVER_TMP.rglob('*.json'))
        assert len(json_files) == PRODUCTION_ROIS
        lookup = sub.copy()
        lookup['roi'] = lookup.apply(lambda r: f'{pid}__{int(r.tile_index):07d}', axis=1)
        lookup = lookup.set_index('roi')
        cell_rows, roi_rows = ([], [])
        for jf in json_files:
            roi = jf.stem
            r = lookup.loc[roi]
            pred = json.load(open(jf))
            nuclei = pred.get('nuc', {})
            counts = {t: 0 for t in TYPE_NAMES}
            probs = []
            physical_w0 = SOURCE_TILE_PX * float(r.downsample)
            scale_to_level0 = physical_w0 / REGION_PX_40X
            for nucleus_id, nuc in nuclei.items():
                t = int(nuc.get('type', 0))
                p = nuc.get('type_prob', np.nan)
                xy = nuc.get('centroid', [np.nan, np.nan])
                x_roi, y_roi = (float(xy[0]), float(xy[1]))
                counts[t] = counts.get(t, 0) + 1
                if p is not None:
                    probs.append(float(p))
                cell_rows.append({'patient_id': pid, 'file_id': fid, 'split': s.get('split', np.nan), 'roi': roi, 'master_order': int(r.master_order), 'tile_index': int(r.tile_index), 'type': t, 'type_name': TYPE_NAMES.get(t, f'type_{t}'), 'type_prob': float(p) if p is not None else np.nan, 'high_confidence': bool(float(p) >= MIN_TYPE_PROB) if p is not None else False, 'x_roi_40xeq': x_roi, 'y_roi_40xeq': y_roi, 'x_level0': float(r.x_level0) + x_roi * scale_to_level0, 'y_level0': float(r.y_level0) + y_roi * scale_to_level0})
            total = sum(counts.values())
            roi_rows.append({'patient_id': pid, 'file_id': fid, 'split': s.get('split', np.nan), 'roi': roi, 'master_order': int(r.master_order), 'tile_index': int(r.tile_index), 'roi_x_level0': float(r.x_level0), 'roi_y_level0': float(r.y_level0), 'downsample': float(r.downsample), 'sampling_exact_03b_design': bool(r.sampling_exact_03b_design), 'total_nuclei': total, 'median_type_prob': np.median(probs) if probs else np.nan, 'high_conf_fraction': np.mean(np.array(probs) >= MIN_TYPE_PROB) if probs else np.nan, **{f'n_{TYPE_NAMES[t]}': counts.get(t, 0) for t in TYPE_NAMES}})
        cell_df, roi_df = (pd.DataFrame(cell_rows), pd.DataFrame(roi_rows))
        assert len(roi_df) == PRODUCTION_ROIS
        assert roi_df['roi'].nunique() == PRODUCTION_ROIS
        cell_tmp = TABLE_OUT / f'{pid}__cells.tmp.csv.gz'
        roi_tmp = TABLE_OUT / f'{pid}__roi_features.tmp.csv'
        cell_df.to_csv(cell_tmp, index=False, compression='gzip')
        roi_df.to_csv(roi_tmp, index=False)
        cell_tmp.replace(cell_path)
        roi_tmp.replace(roi_path)
        status = {'patient_id': pid, 'file_id': fid, 'split': str(s.get('split', '')), 'n_rois': int(len(roi_df)), 'n_nuclei': int(len(cell_df)), 'n_high_confidence_nuclei': int(cell_df['high_confidence'].sum()) if len(cell_df) else 0, 'sampling_exact_03b_design': bool(roi_df['sampling_exact_03b_design'].all()), 'raw_svs_retained': False, 'raw_hovernet_json_retained': False}
        done_path.write_text(json.dumps(status, indent=2))
        print(f'OK {pid}: {len(roi_df)} ROIs | {len(cell_df):,} nuclei')
    except Exception as e:
        error_path.write_text(repr(e))
        print(f'ERROR {pid}: {e}')
    finally:
        if local.exists():
            local.unlink()
        for p in [ROI_TMP, HOVER_TMP]:
            if p.exists():
                shutil.rmtree(p)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
rows = []
for _, s in selected.iterrows():
    pid = str(s.patient_id)
    done = STATUS_OUT / f'{pid}.done.json'
    err = STATUS_OUT / f'{pid}.error.txt'
    row = {'patient_id': pid, 'file_id': str(s.file_id), 'split': s.get('split', np.nan), 'complete': patient_complete(pid), 'error_marker': err.exists()}
    if done.exists():
        row.update(json.loads(done.read_text()))
    rows.append(row)
audit = pd.DataFrame(rows)
AUDIT_PATH = MANIFEST_OUT / 'full_cohort_completion_audit.csv'
audit.to_csv(AUDIT_PATH, index=False)
feature_rows = []
for _, a in audit[audit['complete']].iterrows():
    pid = str(a.patient_id)
    roi = pd.read_csv(TABLE_OUT / f'{pid}__roi_features.csv')
    total = int(roi['total_nuclei'].sum())
    denom = max(total, 1)
    feature_rows.append({'patient_id': pid, 'file_id': str(a.file_id), 'split': a['split'], 'n_rois': len(roi), 'total_nuclei': total, 'mean_nuclei_per_roi': roi['total_nuclei'].mean(), 'median_nuclei_per_roi': roi['total_nuclei'].median(), 'median_roi_type_confidence': roi['median_type_prob'].median(), 'mean_high_conf_fraction': roi['high_conf_fraction'].mean(), 'predicted_inflammatory_fraction': roi['n_inflammatory'].sum() / denom, 'predicted_neoplastic_fraction': roi['n_neoplastic'].sum() / denom, 'predicted_connective_fraction': roi['n_connective'].sum() / denom, 'predicted_dead_fraction': roi['n_dead'].sum() / denom, 'predicted_epithelial_fraction': roi['n_epithelial'].sum() / denom, 'inflammatory_negative_roi_fraction': (roi['n_inflammatory'] == 0).mean(), 'roi_inflammatory_fraction_sd': (roi['n_inflammatory'] / roi['total_nuclei'].clip(lower=1)).std(ddof=1), 'roi_neoplastic_fraction_sd': (roi['n_neoplastic'] / roi['total_nuclei'].clip(lower=1)).std(ddof=1)})
patient_features = pd.DataFrame(feature_rows)
FEATURE_PATH = TABLE_OUT / 'full_cohort_patient_cellular_features.csv'
patient_features.to_csv(FEATURE_PATH, index=False)
processable = samples['patient_id'].astype(str).nunique()
completed = int(audit[audit['patient_id'].astype(str).isin(samples['patient_id'].astype(str))]['complete'].sum())
if not completed != processable:
    bad, total_rois, total_cells = ([], 0, 0)
    for pid in samples['patient_id'].astype(str).unique():
        r = pd.read_csv(TABLE_OUT / f'{pid}__roi_features.csv')
        c = pd.read_csv(TABLE_OUT / f'{pid}__cells.csv.gz', usecols=['patient_id'])
        total_rois += len(r)
        total_cells += len(c)
        if len(r) != PRODUCTION_ROIS or r['roi'].nunique() != PRODUCTION_ROIS:
            bad.append(pid)
target_done = STATUS_OUT / f'{TARGET_PATIENT}.done.json'
target_cells = TABLE_OUT / f'{TARGET_PATIENT}__cells.csv.gz'
target_rois = TABLE_OUT / f'{TARGET_PATIENT}__roi_features.csv'
if target_done.exists() and target_cells.exists() and target_rois.exists():
    rr = pd.read_csv(target_rois)
    cc = pd.read_csv(target_cells, usecols=['patient_id'])
    assert len(rr) == PRODUCTION_ROIS
    assert rr['roi'].nunique() == PRODUCTION_ROIS
