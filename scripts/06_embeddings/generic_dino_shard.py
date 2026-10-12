import os
import subprocess, sys
subprocess.run(['apt-get', '-qq', 'update'], check=True)
subprocess.run(['apt-get', '-qq', 'install', '-y', 'libopenslide0', 'openslide-tools'], check=True)
subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'openslide-python', 'timm'], check=True)
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import gc, json, subprocess, time
import numpy as np
import pandas as pd
from PIL import Image
import torch, timm, openslide
from torchvision import transforms
SEED = 2026
assert torch.cuda.is_available(), 'Use A100 GPU.'
ROOT = Path('/content/drive/MyDrive/AACR_GBM_Project')
SELECTED_PATH = ROOT / 'data/metadata/tcga_gbm_notebook02_selected_slides.csv'
MANIFEST_PATH = ROOT / 'outputs/notebook03_cells/full_cohort_200roi/manifests/full_cohort_200roi_manifest.csv.gz'
ADAPTED_MATRIX = ROOT / 'outputs/notebook06_embeddings/patient_embedding_matrix.npz'
N4_PATH = ROOT / 'outputs/notebook04_spatial_topology/tcga_gbm_patient_spatial_topology.csv'
OUT = ROOT / 'outputs/notebook06B_generic_dino_baseline'
ROI_DIR = OUT / 'roi_embeddings'
PAT_DIR = OUT / 'patient_embeddings'
STATUS = OUT / 'status'
TABLES = OUT / 'tables'
for p in [OUT, ROI_DIR, PAT_DIR, STATUS, TABLES]:
    p.mkdir(parents=True, exist_ok=True)
TMP = Path('/content/n6b_tmp')
TMP.mkdir(exist_ok=True)
MODEL_NAME = 'vit_small_patch16_224.dino'
PATCH_SIZE = 224
BATCH_SIZE = 128
EXPECTED_PATIENTS = 377
EXPECTED_ROIS = 200
GDC_DATA = 'https://api.gdc.cancer.gov/data'
ALPHAS = np.logspace(-4, 4, 17)
CV_FOLDS = 5
N_SHARDS = 8
shard_setting = os.environ.get('AACR_GBM_SHARD_ID', 'all')
SHARD_ID = None if shard_setting == 'all' else int(shard_setting)
assert SHARD_ID is None or 0 <= SHARD_ID < N_SHARDS, 'AACR_GBM_SHARD_ID must be all or 0 through 7'
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
ROICOL = choose(samples, ['roi', 'roi_id', 'master_order', 'tile_index'], 'ROI identity')
DSCOL = next((c for c in ['source_downsample', 'downsample', 'level_downsample'] if c in samples.columns), None)
device = torch.device('cuda')
baseline = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0).to(device).eval()
for p in baseline.parameters():
    p.requires_grad = False
EMBED_DIM = baseline.num_features
assert EMBED_DIM == 384
ad = np.load(ADAPTED_MATRIX, allow_pickle=False)
assert ad['mean_embeddings'].shape == (377, 384)
infer_transform = transforms.Compose([transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC), transforms.ToTensor(), transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
DOWNLOAD_CONNECT_TIMEOUT = 30
DOWNLOAD_MAX_TIME = 1800
DOWNLOAD_ATTEMPTS = 3

def download_gdc(fid, dest):
    dest = Path(dest)
    part = Path(str(dest) + '.part')
    if dest.exists() and dest.stat().st_size > 1000000:
        return dest
    if dest.exists():
        dest.unlink()
    for a in range(1, DOWNLOAD_ATTEMPTS + 1):
        cmd = ['curl', '-L', '--fail', '--show-error', '--connect-timeout', str(DOWNLOAD_CONNECT_TIMEOUT), '--max-time', str(DOWNLOAD_MAX_TIME), '-C', '-', '-o', str(part), f'{GDC_DATA}/{fid}']
        rc = subprocess.run(cmd).returncode
        if rc == 0 and part.exists() and part.stat().st_size > 1000000:
            part.replace(dest)
            return dest
        time.sleep(5)
    raise RuntimeError(f'GDC download failed for {fid}; partial file retained for retry')

def read_roi(slide, row):
    x0 = int(float(row[XCOL]))
    y0 = int(float(row[YCOL]))
    if DSCOL is not None and pd.notna(row[DSCOL]):
        ds = max(1.0, float(row[DSCOL]))
        source_w0 = max(PATCH_SIZE, int(round(256 * ds)))
    else:
        source_w0 = PATCH_SIZE
    img = slide.read_region((x0, y0), 0, (source_w0, source_w0)).convert('RGB')
    if img.size != (224, 224):
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
    return img
work = selected[selected['split'].isin(['train', 'test'])].sort_values(['split', 'patient_id']).reset_index(drop=True).copy()
work['global_work_index'] = np.arange(len(work))
if SHARD_ID is not None:
    work = work[work['global_work_index'] % N_SHARDS == SHARD_ID].reset_index(drop=True)

def patient_complete(pid):
    roi_path = ROI_DIR / f'{pid}__roi_embeddings.npz'
    patient_path = PAT_DIR / f'{pid}__patient_embedding.npz'
    status_path = STATUS / f'{pid}.done.json'
    if not (roi_path.exists() and patient_path.exists() and status_path.exists()):
        return False
    try:
        status = json.loads(status_path.read_text())
        with np.load(roi_path, allow_pickle=False) as roi, np.load(patient_path, allow_pickle=False) as patient:
            return status.get('n_embedded_rois') == EXPECTED_ROIS and roi['embeddings'].shape == (EXPECTED_ROIS, EMBED_DIM) and patient['mean_embedding'].shape == (EMBED_DIM,)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False

logs = []
failed = []
for local_i, pr in work.iterrows():
    pid = str(pr.patient_id)
    fid = str(pr.file_id)
    split = str(pr.split)
    ro = ROI_DIR / f'{pid}__roi_embeddings.npz'
    po = PAT_DIR / f'{pid}__patient_embedding.npz'
    so = STATUS / f'{pid}.done.json'
    if patient_complete(pid):
        logs.append(json.loads(so.read_text()))
        print(f'[{local_i + 1}/{len(work)}] OK {pid} already complete — RESUMED', flush=True)
        continue
    ps = samples[samples.patient_id == pid].copy().reset_index(drop=True)
    assert len(ps) == 200
    svs = TMP / f'{pid}.svs'
    tpatient = time.time()
    print(f'[{local_i + 1}/{len(work)}] START {pid} ({split})', flush=True)
    try:
        try:
            download_gdc(fid, svs)
        except Exception as e:
            failed.append({'patient_id': pid, 'file_id': fid, 'split': split, 'stage': 'download', 'error': repr(e)})
            print('    WARNING: SKIPPED download:', repr(e), flush=True)
            continue
        t0 = time.time()
        slide = openslide.OpenSlide(str(svs))
        tensors = []
        keep = []
        try:
            for j, r in ps.iterrows():
                img = read_roi(slide, r)
                arr = np.asarray(img)
                if arr.mean() > 252 and arr.std() < 3:
                    continue
                tensors.append(infer_transform(img))
                keep.append(j)
        finally:
            slide.close()
        if len(tensors) != 200:
            failed.append({'patient_id': pid, 'file_id': fid, 'split': split, 'stage': 'roi', 'error': f'expected 200, got {len(tensors)}'})
            print(f'    WARNING: SKIPPED: expected 200 nonblank ROIs, got {len(tensors)}', flush=True)
            continue
        t0 = time.time()
        ee = []
        with torch.inference_mode():
            for b in range(0, 200, BATCH_SIZE):
                x = torch.stack(tensors[b:b + BATCH_SIZE]).to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    z = baseline(x)
                ee.append(z.float().cpu().numpy())
        E = np.concatenate(ee).astype(np.float32)
        assert E.shape == (200, 384) and np.isfinite(E).all()
        m = ps.iloc[keep]
        np.savez_compressed(ro, embeddings=E, roi_id=m[ROICOL].astype(str).to_numpy(), x_level0=m[XCOL].astype(float).to_numpy(), y_level0=m[YCOL].astype(float).to_numpy(), patient_id=np.array([pid]), file_id=np.array([fid]), split=np.array([split]))
        np.savez_compressed(po, mean_embedding=E.mean(0).astype(np.float32), std_embedding=E.std(0).astype(np.float32), patient_id=np.array([pid]), file_id=np.array([fid]), split=np.array([split]), n_embedded_rois=np.array([200]))
        rec = {'patient_id': pid, 'file_id': fid, 'split': split, 'n_embedded_rois': 200}
        json.dump(rec, open(so, 'w'), indent=2)
        logs.append(rec)
        print(f'{pid}: complete in {time.time() - tpatient:.1f}s', flush=True)
    except Exception as e:
        failed.append({'patient_id': pid, 'file_id': fid, 'split': split, 'stage': 'processing', 'error': repr(e)})
        print('    WARNING: Processing FAILURE:', repr(e), flush=True)
    finally:
        if svs.exists():
            svs.unlink()
        gc.collect()
        torch.cuda.empty_cache()
fail_path = TABLES / f"shard_{SHARD_ID if SHARD_ID is not None else 'all'}_failed.csv"
pd.DataFrame(failed).to_csv(fail_path, index=False)
