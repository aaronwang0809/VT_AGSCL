import subprocess, sys
subprocess.run(['apt-get', '-qq', 'update'], check=True)
subprocess.run(['apt-get', '-qq', 'install', '-y', 'openslide-tools'], check=True, stdout=subprocess.DEVNULL)
subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'openslide-python', 'opencv-python-headless', 'tqdm'], check=True)
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import pandas as pd, numpy as np, requests, hashlib, json, os, gc, shutil, time
import cv2, openslide
from PIL import Image
ROOT = Path('/content/drive/MyDrive/AACR_GBM_Project')
META = ROOT / 'data' / 'metadata'
OUT = ROOT / 'outputs' / 'notebook02_wsi_qc'
THUMBS = OUT / 'thumbnails'
MASKS = OUT / 'masks'
COORDS = OUT / 'coordinates'
for p in [META, OUT, THUMBS, MASKS, COORDS]:
    p.mkdir(parents=True, exist_ok=True)
RANKED = META / 'tcga_gbm_primary_dx_ranked_with_splits.csv'
assert RANKED.exists(), f'Missing {RANKED}; run cohort_refinement.py first.'
slides = pd.read_csv(RANKED)
TARGET_MAG = 20.0
TILE_PX = 256
MIN_TILE_TISSUE = 0.7
THUMB_MAX_DIM = 2048
MIN_WSI_TISSUE_FRAC = 0.03
MIN_FOCUS_VAR = 8.0
MIN_MEAN_SATURATION = 12.0
MAX_MEAN_BRIGHTNESS = 245.0
MORPH_KERNEL = 5
DOWNLOAD_CHUNK_MB = 8
REQUEST_TIMEOUT = 180
MAX_RETRIES = 4
KEEP_RAW_SVS = False
MAX_PATIENTS = int(os.environ.get('AACR_GBM_MAX_PATIENTS', '0')) or None
GDC_DATA = 'https://api.gdc.cancer.gov/data'
TMP = Path('/content/tcga_wsi_tmp')
TMP.mkdir(exist_ok=True)

def free_gb(path='/content'):
    return shutil.disk_usage(path).free / 1024 ** 3

def md5_file(path, chunk=8 * 1024 * 1024):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def download_gdc_file(file_id, dest, expected_bytes=None):
    last = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if dest.exists():
                dest.unlink()
            with requests.get(f'{GDC_DATA}/{file_id}', stream=True, timeout=REQUEST_TIMEOUT) as r:
                r.raise_for_status()
                with open(dest, 'wb') as f:
                    for chunk in r.iter_content(DOWNLOAD_CHUNK_MB * 1024 * 1024):
                        if chunk:
                            f.write(chunk)
            if expected_bytes is not None and dest.stat().st_size != int(expected_bytes):
                raise IOError('Downloaded file-size mismatch')
            return dest
        except Exception as e:
            last = e
            if dest.exists():
                dest.unlink()
            print(f'retry {attempt}/{MAX_RETRIES}: {e}')
            time.sleep(min(30, 3 * attempt))
    raise RuntimeError(f'Download failed: {last}')

def safe_float(x):
    try:
        return float(x)
    except:
        return np.nan

def read_wsi_metadata(path):
    slide = openslide.OpenSlide(str(path))
    props = slide.properties
    mpp_x = safe_float(props.get(openslide.PROPERTY_NAME_MPP_X))
    mpp_y = safe_float(props.get(openslide.PROPERTY_NAME_MPP_Y))
    objective = safe_float(props.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER))
    estimated = False
    if not np.isfinite(objective) and np.isfinite(mpp_x) and (mpp_x > 0):
        objective = 10.0 / mpp_x
        estimated = True
    ds = np.array(slide.level_downsamples, dtype=float)
    if np.isfinite(objective) and objective > 0:
        desired = max(1.0, objective / TARGET_MAG)
        level = int(np.argmin(np.abs(np.log2(ds) - np.log2(desired))))
    elif np.isfinite(mpp_x) and mpp_x > 0:
        desired = max(1.0, 10.0 / TARGET_MAG / mpp_x)
        level = int(np.argmin(np.abs(np.log2(ds) - np.log2(desired))))
    else:
        level = 0
    sel = float(slide.level_downsamples[level])
    return (slide, {'width_level0': slide.dimensions[0], 'height_level0': slide.dimensions[1], 'level_count': slide.level_count, 'objective_power': objective, 'objective_estimated': estimated, 'mpp_x_level0': mpp_x, 'mpp_y_level0': mpp_y, 'target_level': level, 'target_downsample': sel, 'target_mpp_x': mpp_x * sel if np.isfinite(mpp_x) else np.nan, 'target_mpp_y': mpp_y * sel if np.isfinite(mpp_y) else np.nan, 'physical_scale_known': bool(np.isfinite(mpp_x) or np.isfinite(objective))})

def make_thumbnail(slide, max_dim=THUMB_MAX_DIM):
    w, h = slide.dimensions
    scale = min(max_dim / w, max_dim / h)
    tw, th = (max(1, int(w * scale)), max(1, int(h * scale)))
    return (np.array(slide.get_thumbnail((tw, th)).convert('RGB')), scale)

def tissue_mask_hsv_otsu(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat, val = (hsv[:, :, 1], hsv[:, :, 2])
    _, sm = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    nonwhite = (val < 245).astype(np.uint8) * 255
    mask = cv2.bitwise_and(sm, nonwhite)
    k = np.ones((MORPH_KERNEL, MORPH_KERNEL), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    clean = np.zeros_like(mask)
    min_area = max(50, int(mask.size * 0.0002))
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[lab == i] = 255
    return clean

def thumbnail_qc(rgb, mask):
    tissue = mask > 0
    frac = float(tissue.mean())
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    if tissue.any():
        vals = gray[tissue]
        sats = hsv[:, :, 1][tissue]
        brightness = float(vals.mean())
        saturation = float(sats.mean())
        g = gray.copy()
        g[~tissue] = int(np.median(vals))
        focus = float(cv2.Laplacian(g, cv2.CV_64F).var())
    else:
        brightness = np.nan
        saturation = np.nan
        focus = 0.0
    flags = []
    if frac < MIN_WSI_TISSUE_FRAC:
        flags.append('low_tissue')
    if focus < MIN_FOCUS_VAR:
        flags.append('low_focus_proxy')
    if np.isfinite(saturation) and saturation < MIN_MEAN_SATURATION:
        flags.append('low_saturation')
    if np.isfinite(brightness) and brightness > MAX_MEAN_BRIGHTNESS:
        flags.append('too_bright')
    return {'tissue_fraction_thumb': frac, 'focus_laplacian_var': focus, 'mean_tissue_brightness': brightness, 'mean_tissue_saturation': saturation, 'technical_qc_pass': len(flags) == 0, 'qc_flags': '|'.join(flags) if flags else 'PASS'}

def integral_image(mask):
    return cv2.integral(mask.astype(np.uint8))

def rect_sum(ii, x0, y0, x1, y1):
    return ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]

def generate_tile_coordinates(slide, meta, thumb_mask, thumb_scale):
    level = meta['target_level']
    ds = float(meta['target_downsample'])
    lw, lh = slide.level_dimensions[level]
    ii = integral_image(thumb_mask > 0)
    mh, mw = thumb_mask.shape
    rows = []
    idx = 0
    for yL in range(0, lh - TILE_PX + 1, TILE_PX):
        for xL in range(0, lw - TILE_PX + 1, TILE_PX):
            x0 = int(round(xL * ds))
            y0 = int(round(yL * ds))
            x1 = int(round((xL + TILE_PX) * ds))
            y1 = int(round((yL + TILE_PX) * ds))
            tx0 = max(0, min(mw - 1, int(np.floor(x0 * thumb_scale))))
            ty0 = max(0, min(mh - 1, int(np.floor(y0 * thumb_scale))))
            tx1 = max(tx0 + 1, min(mw, int(np.ceil(x1 * thumb_scale))))
            ty1 = max(ty0 + 1, min(mh, int(np.ceil(y1 * thumb_scale))))
            area = (tx1 - tx0) * (ty1 - ty0)
            frac = rect_sum(ii, tx0, ty0, tx1, ty1) / max(area, 1)
            if frac >= MIN_TILE_TISSUE:
                rows.append({'tile_index': idx, 'x_level0': x0, 'y_level0': y0, 'level': level, 'tile_px': TILE_PX, 'downsample': ds, 'estimated_tissue_fraction': float(frac), 'mpp_x': meta['target_mpp_x'], 'mpp_y': meta['target_mpp_y']})
                idx += 1
    return pd.DataFrame(rows)
CHECKPOINT = OUT / 'slide_attempts_checkpoint.csv'
SELECTED = META / 'tcga_gbm_notebook02_selected_slides.csv'

def load_checkpoint():
    return pd.read_csv(CHECKPOINT) if CHECKPOINT.exists() else pd.DataFrame()

def save_attempt(row):
    old = load_checkpoint()
    new = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
    new = new.drop_duplicates('file_id', keep='last')
    new.to_csv(CHECKPOINT, index=False)
cp = load_checkpoint()
ranked = slides.sort_values(['patient_id', 'pre_qc_rank']).copy()
patient_ids = sorted(ranked.patient_id.unique())
if MAX_PATIENTS is not None:
    patient_ids = patient_ids[:MAX_PATIENTS]

def previously_selected(pid):
    cp = load_checkpoint()
    return len(cp[(cp.patient_id == pid) & (cp.status == 'SELECTED')]) > 0 if len(cp) else False
for pnum, pid in enumerate(patient_ids, 1):
    if previously_selected(pid):
        print(f'[{pnum}/{len(patient_ids)}] {pid}: already selected')
        continue
    cand = ranked[ranked.patient_id == pid].sort_values('pre_qc_rank')
    print(f'\n[{pnum}/{len(patient_ids)}] {pid}: {len(cand)} candidate(s)')
    chosen = False
    for _, row in cand.iterrows():
        fid = str(row.file_id)
        old = load_checkpoint()
        prior = old[old.file_id.astype(str) == fid] if len(old) else pd.DataFrame()
        if len(prior):
            continue
        expected_gb = float(row.file_size_gb)
        if free_gb() < expected_gb + 5:
            raise RuntimeError(f'Need >{expected_gb + 5:.1f} GB free; only {free_gb():.1f} GB available')
        local = TMP / f'{fid}.svs'
        attempt = {'patient_id': pid, 'file_id': fid, 'file_name': row.file_name, 'pre_qc_rank': row.pre_qc_rank, 'split': row.split, 'status': 'ERROR', 'error': ''}
        slide = None
        try:
            download_gdc_file(fid, local, int(row.file_size_bytes))
            expected_md5 = str(row.md5sum)
            if expected_md5.lower() != 'nan' and md5_file(local).lower() != expected_md5.lower():
                raise IOError('MD5 mismatch')
            slide, meta = read_wsi_metadata(local)
            rgb, scale = make_thumbnail(slide)
            mask = tissue_mask_hsv_otsu(rgb)
            qc = thumbnail_qc(rgb, mask)
            attempt.update(meta)
            attempt.update(qc)
            attempt['source_size_gb'] = expected_gb
            if not meta['physical_scale_known']:
                attempt['status'] = 'REJECTED'
                attempt['qc_flags'] = (attempt['qc_flags'] + '|missing_physical_scale').strip('|')
            elif not qc['technical_qc_pass']:
                attempt['status'] = 'REJECTED'
            else:
                coords = generate_tile_coordinates(slide, meta, mask, scale)
                if len(coords) == 0:
                    attempt['status'] = 'REJECTED'
                    attempt['qc_flags'] = (attempt['qc_flags'] + '|no_usable_tiles').strip('|')
                else:
                    stem = f'{pid}__{fid}'
                    Image.fromarray(rgb).save(THUMBS / f'{stem}.jpg', quality=85)
                    Image.fromarray(mask).save(MASKS / f'{stem}.png')
                    coords.insert(0, 'patient_id', pid)
                    coords.insert(1, 'file_id', fid)
                    coords.insert(2, 'slide_id', row.slide_id)
                    coords.insert(3, 'split', row.split)
                    coords.to_csv(COORDS / f'{stem}_coords.csv.gz', index=False, compression='gzip')
                    attempt['n_usable_tiles'] = len(coords)
                    attempt['status'] = 'SELECTED'
                    chosen = True
            save_attempt(attempt)
            if chosen:
                break
        except Exception as e:
            attempt['status'] = 'ERROR'
            attempt['error'] = repr(e)[:1000]
            save_attempt(attempt)
            print('   ERROR:', e)
        finally:
            try:
                if slide is not None:
                    slide.close()
            except:
                pass
            if local.exists() and (not KEEP_RAW_SVS):
                local.unlink()
            gc.collect()
cp = load_checkpoint()
sel = cp[cp.status == 'SELECTED'].sort_values(['patient_id', 'pre_qc_rank']).drop_duplicates('patient_id').copy()
base = ranked[['patient_id', 'file_id', 'slide_id', 'file_name', 'sample_id', 'file_size_gb', 'pre_qc_rank', 'split']].drop_duplicates('file_id')
sel = sel.drop(columns=[c for c in ['slide_id', 'file_name', 'split'] if c in sel], errors='ignore')
sel = sel.merge(base, on=['patient_id', 'file_id', 'pre_qc_rank'], how='left')
sel.to_csv(SELECTED, index=False)
sel = pd.read_csv(SELECTED) if SELECTED.exists() else pd.DataFrame()
if len(sel):
    total = int(pd.to_numeric(sel.n_usable_tiles, errors='coerce').fillna(0).sum())
cp = load_checkpoint()
sel = cp[cp.status == 'SELECTED'].drop_duplicates('patient_id') if len(cp) else pd.DataFrame()
report = {'patients_in_ranked_manifest': int(ranked.patient_id.nunique()), 'patients_targeted_this_run': int(len(patient_ids)), 'slides_attempted_total_checkpoint': int(len(cp)), 'patients_selected_total_checkpoint': int(sel.patient_id.nunique()) if len(sel) else 0, 'rejected_attempts': int((cp.status == 'REJECTED').sum()) if len(cp) else 0, 'error_attempts': int((cp.status == 'ERROR').sum()) if len(cp) else 0, 'target_magnification': TARGET_MAG, 'tile_px': TILE_PX, 'minimum_tile_tissue_fraction': MIN_TILE_TISSUE, 'raw_svs_retained': KEEP_RAW_SVS}
with open(OUT / 'notebook02_run_report.json', 'w') as f:
    json.dump(report, f, indent=2)
