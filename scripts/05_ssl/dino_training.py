import subprocess, sys
from google.colab import drive
drive.mount('/content/drive')
subprocess.run(['apt-get', '-qq', 'update'], check=True)
subprocess.run(['apt-get', '-qq', 'install', '-y', 'openslide-tools'], check=True)
subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'openslide-python', 'timm'], check=True)
from pathlib import Path
import io, gc, json, math, time, random, tarfile, subprocess
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
import openslide
ROOT = Path('/content/drive/MyDrive/AACR_GBM_Project')
SELECTED_PATH = ROOT / 'data/metadata/tcga_gbm_notebook02_selected_slides.csv'
N3 = ROOT / 'outputs/notebook03_cells/full_cohort_200roi'
N3_MANIFEST = N3 / 'manifests/full_cohort_200roi_manifest.csv.gz'
OUT = ROOT / 'outputs/notebook05_ssl_dino'
SHARD_DIR = OUT / 'train_patch_shards'
CKPT_DIR = OUT / 'checkpoints'
LOG_DIR = OUT / 'logs'
EXPORT_DIR = OUT / 'export'
for p in [OUT, SHARD_DIR, CKPT_DIR, LOG_DIR, EXPORT_DIR]:
    p.mkdir(parents=True, exist_ok=True)
TMP = Path('/content/n5_tmp')
TMP.mkdir(parents=True, exist_ok=True)
SEED = 2026
PATCHES_PER_TRAIN_PATIENT = 200
PATCH_SIZE = 224
JPEG_QUALITY = 90
PATIENTS_PER_TAR_SHARD = 8
MODEL_NAME = 'vit_small_patch16_224.dino'
OUT_DIM = 8192
BATCH_SIZE = 128
EPOCHS = 20
BASE_LR = 5e-05
WEIGHT_DECAY = 0.04
TEACHER_MOMENTUM = 0.996
STUDENT_TEMP = 0.1
TEACHER_TEMP = 0.04
CENTER_MOMENTUM = 0.9
NUM_WORKERS = 4
GDC_DATA = 'https://api.gdc.cancer.gov/data'
CURL_CONNECT_TIMEOUT = 30
CURL_MAX_TIME = 900
MAX_DOWNLOAD_ATTEMPTS = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
selected = pd.read_csv(SELECTED_PATH)
selected['patient_id'] = selected['patient_id'].astype(str)
selected['file_id'] = selected['file_id'].astype(str)
assert 'split' in selected.columns
train = selected[selected['split'].astype(str).str.lower() == 'train'].copy()
train = train.sort_values('patient_id').reset_index(drop=True)
samples = pd.read_csv(N3_MANIFEST)
samples['patient_id'] = samples['patient_id'].astype(str)
samples['file_id'] = samples['file_id'].astype(str)
train_ids = set(train['patient_id'])
train_samples = samples[samples['patient_id'].isin(train_ids)].copy()
counts = train_samples.groupby('patient_id').size()
assert len(train) == 262, f'Expected 262 train patients, found {len(train)}'
assert len(counts) == 262
assert (counts == PATCHES_PER_TRAIN_PATIENT).all()
nontrain = set(selected.loc[selected['split'].astype(str).str.lower() != 'train', 'patient_id'])
assert not set(train_samples['patient_id']) & nontrain

def choose_col(df, candidates, label):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f'Could not find {label}. Tried: {candidates}')
XCOL = choose_col(train_samples, ['roi_x_level0', 'x_level0', 'x', 'tile_x', 'x0'], 'level-0 ROI x')
YCOL = choose_col(train_samples, ['roi_y_level0', 'y_level0', 'y', 'tile_y', 'y0'], 'level-0 ROI y')
DOWNSAMPLE_COL = None
for c in ['source_downsample', 'downsample', 'level_downsample']:
    if c in train_samples.columns:
        DOWNSAMPLE_COL = c
        break

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
    raise RuntimeError(f'GDC download failed after {MAX_DOWNLOAD_ATTEMPTS} attempts: {file_id}')

def read_patch_around_sample(slide, row):
    x0 = int(float(row[XCOL]))
    y0 = int(float(row[YCOL]))
    if DOWNSAMPLE_COL is not None and pd.notna(row[DOWNSAMPLE_COL]):
        ds = max(1.0, float(row[DOWNSAMPLE_COL]))
        source_w0 = max(PATCH_SIZE, int(round(256 * ds)))
    else:
        source_w0 = PATCH_SIZE
    region = slide.read_region((x0, y0), 0, (source_w0, source_w0)).convert('RGB')
    if region.size != (PATCH_SIZE, PATCH_SIZE):
        region = region.resize((PATCH_SIZE, PATCH_SIZE), Image.Resampling.LANCZOS)
    return region
train_order = train.sort_values('patient_id').reset_index(drop=True)
n_shards = math.ceil(len(train_order) / PATIENTS_PER_TAR_SHARD)
build_log = []
for shard_id in range(n_shards):
    lo = shard_id * PATIENTS_PER_TAR_SHARD
    hi = min(len(train_order), (shard_id + 1) * PATIENTS_PER_TAR_SHARD)
    shard_patients = train_order.iloc[lo:hi].copy()
    tar_path = SHARD_DIR / f'gbm_train_{shard_id:03d}.tar'
    done_path = SHARD_DIR / f'gbm_train_{shard_id:03d}.done.json'
    if tar_path.exists() and done_path.exists():
        with open(done_path) as f:
            build_log.append(json.load(f))
        continue
    tmp_tar = Path('/content') / f'gbm_train_{shard_id:03d}.tar'
    if tmp_tar.exists():
        tmp_tar.unlink()
    shard_count = 0
    patient_counts = {}
    with tarfile.open(tmp_tar, 'w') as tf:
        for _, pr in shard_patients.iterrows():
            pid = str(pr.patient_id)
            fid = str(pr.file_id)
            svs = TMP / f'{pid}.svs'
            try:
                download_gdc(fid, svs)
                slide = openslide.OpenSlide(str(svs))
                ss = train_samples[train_samples['patient_id'] == pid].copy()
                assert len(ss) == PATCHES_PER_TRAIN_PATIENT
                n_ok = 0
                for j, (_, row) in enumerate(ss.iterrows()):
                    img = read_patch_around_sample(slide, row)
                    arr = np.asarray(img)
                    if arr.mean() > 252 and arr.std() < 3:
                        continue
                    bio = io.BytesIO()
                    img.save(bio, format='JPEG', quality=JPEG_QUALITY, optimize=True)
                    data = bio.getvalue()
                    name = f'{pid}__{j:03d}.jpg'
                    info = tarfile.TarInfo(name=name)
                    info.size = len(data)
                    tf.addfile(info, io.BytesIO(data))
                    n_ok += 1
                    shard_count += 1
                slide.close()
                patient_counts[pid] = n_ok
            finally:
                if svs.exists():
                    svs.unlink()
                gc.collect()
    if tar_path.exists():
        tar_path.unlink()
    tmp_tar.replace(tar_path)
    rec = {'shard_id': shard_id, 'patient_start': lo, 'patient_end_exclusive': hi, 'n_patients': len(shard_patients), 'n_patches': shard_count, 'patient_counts': patient_counts}
    with open(done_path, 'w') as f:
        json.dump(rec, f, indent=2)
    build_log.append(rec)
    print(f'OK shard {shard_id:03d}: {shard_count:,} patches')
tar_files = sorted(SHARD_DIR.glob('gbm_train_*.tar'))
done_files = sorted(SHARD_DIR.glob('gbm_train_*.done.json'))
assert len(tar_files) == n_shards
assert len(done_files) == n_shards
records = [json.load(open(p)) for p in done_files]
stored = sum((r['n_patches'] for r in records))
expected = len(train) * PATCHES_PER_TRAIN_PATIENT
audit = pd.DataFrame([{'shard_id': r['shard_id'], 'n_patients': r['n_patients'], 'n_patches': r['n_patches']} for r in records])
audit.to_csv(LOG_DIR / 'patch_shard_audit.csv', index=False)
if stored < expected * 0.95:
    raise RuntimeError('More than 5% of frozen training patches were lost/rejected; inspect before SSL.')

class TarPatchDataset(Dataset):

    def __init__(self, tar_paths, transform):
        self.tar_paths = [Path(p) for p in tar_paths]
        self.transform = transform
        self.index = []
        for ti, tp in enumerate(self.tar_paths):
            with tarfile.open(tp, 'r') as tf:
                names = [m.name for m in tf.getmembers() if m.isfile() and m.name.endswith('.jpg')]
            self.index.extend(((ti, n) for n in names))
        self._handles = {}

    def __len__(self):
        return len(self.index)

    def _tar(self, ti):
        if ti not in self._handles:
            self._handles[ti] = tarfile.open(self.tar_paths[ti], 'r')
        return self._handles[ti]

    def __getitem__(self, idx):
        ti, name = self.index[idx]
        tf = self._tar(ti)
        f = tf.extractfile(name)
        img = Image.open(f).convert('RGB')
        return (self.transform(img), name)
normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

class TwoViewTransform:

    def __init__(self):
        self.view1 = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.5, 1.0), interpolation=transforms.InterpolationMode.BICUBIC), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomApply([transforms.ColorJitter(0.25, 0.25, 0.2, 0.05)], p=0.8), transforms.RandomGrayscale(p=0.1), transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)), transforms.ToTensor(), normalize])
        self.view2 = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.5, 1.0), interpolation=transforms.InterpolationMode.BICUBIC), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomApply([transforms.ColorJitter(0.25, 0.25, 0.2, 0.05)], p=0.8), transforms.RandomGrayscale(p=0.1), transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5), transforms.ToTensor(), normalize])

    def __call__(self, img):
        return (self.view1(img), self.view2(img))
dataset = TarPatchDataset(tar_files, TwoViewTransform())
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS > 0, drop_last=True)

class DINOHead(nn.Module):

    def __init__(self, in_dim, out_dim=8192, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, bottleneck_dim))
        self.last = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last.weight_g.data.fill_(1)
        self.last.weight_g.requires_grad = False

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        return self.last(x)

class SSLModel(nn.Module):

    def __init__(self, model_name, out_dim):
        super().__init__()
        self.encoder = timm.create_model(model_name, pretrained=True, num_classes=0)
        dim = self.encoder.num_features
        self.head = DINOHead(dim, out_dim=out_dim)

    def forward(self, x):
        return self.head(self.encoder(x))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert device.type == 'cuda', 'dino_training.py requires a GPU runtime.'
student = SSLModel(MODEL_NAME, OUT_DIM).to(device)
teacher = SSLModel(MODEL_NAME, OUT_DIM).to(device)
teacher.load_state_dict(student.state_dict())
for p in teacher.parameters():
    p.requires_grad = False
center = torch.zeros(1, OUT_DIM, device=device)

def dino_loss(student_outs, teacher_outs, center):
    s = [x / STUDENT_TEMP for x in student_outs]
    t = [F.softmax((x - center) / TEACHER_TEMP, dim=-1).detach() for x in teacher_outs]
    total = 0.0
    n = 0
    for iq, q in enumerate(t):
        for iv, v in enumerate(s):
            if iq == iv:
                continue
            total += torch.sum(-q * F.log_softmax(v, dim=-1), dim=-1).mean()
            n += 1
    return total / n
optimizer = torch.optim.AdamW(student.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
scaler = torch.amp.GradScaler('cuda')
latest = CKPT_DIR / 'latest.pt'
start_epoch = 0
if latest.exists():
    ck = torch.load(latest, map_location=device)
    student.load_state_dict(ck['student'])
    teacher.load_state_dict(ck['teacher'])
    optimizer.load_state_dict(ck['optimizer'])
    center = ck['center'].to(device)
    start_epoch = int(ck['epoch']) + 1
history = []
hist_path = LOG_DIR / 'ssl_training_history.csv'
if hist_path.exists():
    history = pd.read_csv(hist_path).to_dict('records')
for epoch in range(start_epoch, EPOCHS):
    student.train()
    teacher.eval()
    losses = []
    t0 = time.time()
    lr = BASE_LR * 0.5 * (1 + math.cos(math.pi * epoch / max(1, EPOCHS)))
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    pbar = tqdm(loader, desc=f'Epoch {epoch + 1}/{EPOCHS}')
    for (v1, v2), names in pbar:
        v1 = v1.to(device, non_blocking=True)
        v2 = v2.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            s1 = student(v1)
            s2 = student(v2)
            with torch.no_grad():
                t1 = teacher(v1)
                t2 = teacher(v2)
            loss = dino_loss([s1, s2], [t1, t2], center)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 3.0)
        scaler.step(optimizer)
        scaler.update()
        with torch.no_grad():
            for ps, pt in zip(student.parameters(), teacher.parameters()):
                pt.data.mul_(TEACHER_MOMENTUM).add_(ps.data, alpha=1 - TEACHER_MOMENTUM)
            batch_center = torch.cat([t1, t2], dim=0).mean(dim=0, keepdim=True)
            center.mul_(CENTER_MOMENTUM).add_(batch_center, alpha=1 - CENTER_MOMENTUM)
        losses.append(float(loss.detach().cpu()))
        pbar.set_postfix(loss=f'{np.mean(losses[-20:]):.4f}', lr=f'{lr:.2e}')
    rec = {'epoch': epoch, 'mean_loss': float(np.mean(losses)), 'lr': lr, 'minutes': (time.time() - t0) / 60}
    history.append(rec)
    pd.DataFrame(history).to_csv(hist_path, index=False)
    ck = {'epoch': epoch, 'student': student.state_dict(), 'teacher': teacher.state_dict(), 'optimizer': optimizer.state_dict(), 'center': center.detach().cpu(), 'config': {'model_name': MODEL_NAME, 'out_dim': OUT_DIM, 'seed': SEED, 'train_patients': len(train), 'training_patches': len(dataset)}}
    epoch_path = CKPT_DIR / f'epoch_{epoch:03d}.pt'
    torch.save(ck, epoch_path)
    torch.save(ck, latest)
hist = pd.read_csv(LOG_DIR / 'ssl_training_history.csv')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(hist['epoch'] + 1, hist['mean_loss'], marker='o')
ax.set_xlabel('Epoch')
ax.set_ylabel('DINO-style self-distillation loss')
ax.set_title('GBM self-supervised ViT adaptation')
fig.tight_layout()
fig.savefig(LOG_DIR / 'ssl_training_curve.png', dpi=300)
plt.show()
student.eval()
encoder_path = EXPORT_DIR / 'gbm_dino_vit_small_encoder.pt'
torch.save({'model_name': MODEL_NAME, 'encoder_state_dict': student.encoder.state_dict(), 'num_features': student.encoder.num_features, 'patch_size': PATCH_SIZE, 'seed': SEED, 'train_patients': len(train), 'training_patches': len(dataset), 'source_split': 'train only'}, encoder_path)
record = {'notebook': '05_GBM_SSL_DINO_ViT', 'model_name': MODEL_NAME, 'train_patients': int(len(train)), 'frozen_training_locations': int(len(train_samples)), 'stored_training_patches': int(len(dataset)), 'epochs': int(EPOCHS), 'final_loss': float(hist.iloc[-1]['mean_loss']), 'encoder_path': str(encoder_path), 'leakage_audit': 'training split only; no validation/test patients or outcome labels used'}
with open(OUT / 'notebook05_results_record.json', 'w') as f:
    json.dump(record, f, indent=2)
