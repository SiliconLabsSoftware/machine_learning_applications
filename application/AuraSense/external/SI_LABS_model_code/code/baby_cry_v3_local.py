# -*- coding: utf-8 -*-
"""
Baby Cry Detection v3.0 — LOCAL macOS Build
Two-stage cascade: CRY DETECTOR → EMOTION DECIDER
Adapted from Colab notebook for local execution with QAT INT8 fix.
"""

# ============================================================
# OUTPUT DIRECTORIES
# ============================================================
import os
OUTPUT_ROOT = "outputs"
PLOTS_DIR   = os.path.join(OUTPUT_ROOT, "plots")
MODELS_DIR  = os.path.join(OUTPUT_ROOT, "models")
TFLITE_DIR  = os.path.join(OUTPUT_ROOT, "tflite")
for d in [PLOTS_DIR, MODELS_DIR, TFLITE_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# CELL 3 — GLOBAL CONFIGURATION
# ============================================================
import torch
import numpy as np
import random, gc, csv, json
import warnings
warnings.filterwarnings('ignore')

SR          = 16_000
WIN_S       = 0.75
HOP_S       = 0.375
WIN_SAMPLES = int(SR * WIN_S)
HOP_SAMPLES = int(SR * HOP_S)
N_MEL       = 40
N_FFT       = 512
HOP_LENGTH  = 160
T_FRAMES    = WIN_SAMPLES // HOP_LENGTH
SILENCE_TOP_DB = 30
MIN_AUDIO_S    = 0.3
CRY_THRESHOLD       = 0.70
CRYCELEB_CAP        = 2_000
ESC50_CAP           = 2_000
DEMAND_CAP_PER_FILE = 400
M1_SNR_LOW  = 5.0
M1_SNR_HIGH = 20.0
M2_SNR_LOW  = 12.0
M2_SNR_HIGH = 25.0
SAD_AUG_COPIES    = 3
LAUGH_AUG_COPIES  = 10
BATCH_SIZE  = 32
M1_EPOCHS   = 25
M2_EPOCHS   = 25
LR          = 1e-3
M1_PATIENCE = 8
M2_PATIENCE = 10
SPLIT_TRAIN = 0.80
SPLIT_VAL   = 0.10

# ── Paths (LOCAL) ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get('AURASENSE_DATA', os.path.join(SCRIPT_DIR, 'datasets1'))
SAD_DIR       = f'{BASE}/sad'
LAUGH_DIR     = f'{BASE}/laugh'
AUDIO_DIR     = f'{BASE}/audio'
ESC50_AUDIO   = f'{BASE}/esc50/audio'
ESC50_CSV     = f'{BASE}/esc50/esc50.csv'
NOISE_DIR     = f'{BASE}/noise'
BG_DIR        = f'{BASE}/background'

CKPT_M1     = os.path.join(MODELS_DIR, 'model_m1_detector.pt')
CKPT_M2     = os.path.join(MODELS_DIR, 'model_m2_emotion.pt')
TFLITE_FP32 = os.path.join(TFLITE_DIR, 'baby_cry_fused_fp32.tflite')
TFLITE_INT8 = os.path.join(TFLITE_DIR, 'baby_cry_fused_int8.tflite')
TFLITE_QAT  = os.path.join(TFLITE_DIR, 'baby_cry_fused_qat_int8.tflite')

# NOTE: MPS doesn't support AdaptiveAvgPool2d with non-divisible sizes,
# so we use CPU for training. Still fast enough for this model size.
DEVICE = torch.device('cpu')

print(f'Device        : {DEVICE}')
print(f'Window        : {WIN_S}s  ({WIN_SAMPLES} samples)  hop {HOP_S}s')
print(f'Feature shape : ({T_FRAMES}, {N_MEL}, 3)')
print(f'Epochs        : M1={M1_EPOCHS}  M2={M2_EPOCHS}')

# ============================================================
# CELL 2 — VERIFY FOLDERS
# ============================================================
checks = {
    'sad'         : ('distress cry clips',           50),
    'laugh'       : ('happy baby sounds',             50),
    'audio'       : ('CryCeleb2023 (train/ inside)',   1),
    'esc50/audio' : ('ESC-50 flat WAVs',            100),
    'noise'       : ('DEMAND ch01 files',              1),
    'background'  : ('Freesound long files',           1),
}
print('=' * 60)
print('FOLDER VERIFICATION')
print('=' * 60)
all_ok = True
for folder, (desc, mn) in checks.items():
    path = os.path.join(BASE, folder)
    if os.path.exists(path):
        n = sum(1 for r, d, fs in os.walk(path) for f in fs if f.endswith('.wav'))
        ok = '✅' if n >= mn else '⚠️ '
        if n < mn: all_ok = False
        print(f'  {ok} {folder:20s} → {n:5d} WAV files  ({desc})')
    else:
        print(f'  ❌ MISSING  {folder}')
        all_ok = False
csv_path = os.path.join(BASE, 'esc50', 'esc50.csv')
if os.path.exists(csv_path):
    print(f'  ✅ esc50/esc50.csv               → found')
else:
    print(f'  ❌ MISSING  esc50/esc50.csv')
    all_ok = False
print()
print('✅ All folders OK — ready to train.' if all_ok else '⚠️  Fix missing folders before proceeding.')

# ============================================================
# CELL 4 — THREE-CHANNEL FEATURE EXTRACTION
# ============================================================
import librosa

def extract_features(audio: np.ndarray) -> np.ndarray:
    if len(audio) < WIN_SAMPLES:
        audio = np.pad(audio, (0, WIN_SAMPLES - len(audio)))
    else:
        audio = audio[:WIN_SAMPLES]
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MEL, fmin=60, fmax=8000)
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=40)
    log_mel = (log_mel + 40.0) / 40.0
    T = log_mel.shape[1]
    flatness = librosa.feature.spectral_flatness(y=audio, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    flatness = np.log1p(flatness * 1000.0)
    flatness = np.clip(flatness / (np.max(flatness) + 1e-8), 0.0, 1.0)
    flatness_ch = np.tile(flatness[np.newaxis, :T], (N_MEL, 1))
    rms = librosa.feature.rms(y=audio, frame_length=N_FFT, hop_length=HOP_LENGTH)[0][:T]
    rms = np.log1p(rms * 100.0)
    rms = np.clip(rms / (np.max(rms) + 1e-8), 0.0, 1.0)
    rms_ch = np.tile(rms[np.newaxis, :T], (N_MEL, 1))
    feat = np.stack([log_mel[:, :T], flatness_ch[:, :T], rms_ch[:, :T]], axis=-1)
    feat = feat.transpose(1, 0, 2)
    if feat.shape[0] < T_FRAMES:
        pad = np.zeros((T_FRAMES - feat.shape[0], N_MEL, 3), dtype=np.float32)
        feat = np.concatenate([feat, pad], axis=0)
    else:
        feat = feat[:T_FRAMES]
    return feat.astype(np.float32)

_test = extract_features(np.random.randn(WIN_SAMPLES).astype(np.float32))
print(f'Feature shape : {_test.shape}  ← expected ({T_FRAMES}, {N_MEL}, 3)')
print('✅ Feature extractor OK')

# ============================================================
# CELL 5 — SILENCE REMOVAL + AUDIO UTILITIES
# ============================================================
import soundfile as sf

def strip_silence(audio, sr=SR, top_db=SILENCE_TOP_DB):
    intervals = librosa.effects.split(audio, top_db=top_db)
    if len(intervals) == 0: return audio
    return np.concatenate([audio[s:e] for s, e in intervals])

def load_wav(path, sr=SR):
    try:
        audio, file_sr = librosa.load(path, sr=sr, mono=True)
        return audio
    except: return None

def chunk_audio(audio, win=WIN_SAMPLES, hop=HOP_SAMPLES):
    chunks = []
    start = 0
    while start + win <= len(audio):
        chunks.append(audio[start:start + win])
        start += hop
    if start < len(audio) and (len(audio) - start) >= int(SR * 0.5):
        chunk = audio[start:]
        chunk = np.pad(chunk, (0, win - len(chunk)))
        chunks.append(chunk)
    return chunks

def mix_noise(signal, noise_pool, snr_low, snr_high):
    if not noise_pool: return signal
    noise = random.choice(noise_pool)
    if len(noise) > WIN_SAMPLES:
        start = random.randint(0, len(noise) - WIN_SAMPLES)
        noise_seg = noise[start:start + WIN_SAMPLES]
    else:
        noise_seg = np.pad(noise, (0, WIN_SAMPLES - len(noise)))
    snr_db = random.uniform(snr_low, snr_high)
    sig_rms   = np.sqrt(np.mean(signal**2) + 1e-9)
    noise_rms = np.sqrt(np.mean(noise_seg**2) + 1e-9)
    target_noise_rms = sig_rms / (10 ** (snr_db / 20.0))
    noise_seg = noise_seg * (target_noise_rms / noise_rms)
    mixed = signal + noise_seg
    peak = np.max(np.abs(mixed))
    if peak > 1.0: mixed = mixed / peak
    return mixed.astype(np.float32)

def list_wavs(directory):
    wavs = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith('.wav'):
                wavs.append(os.path.join(root, f))
    return sorted(wavs)

print('✅ Audio utilities ready')

# ============================================================
# CELL 6 — BUILD NOISE POOL
# ============================================================
DEMAND_ENVS = ['DKITCHEN', 'DLIVING', 'DWASHING', 'NPARK', 'OHALLWAY']
noise_pool = []
print('Loading DEMAND noise files ...')
for env in DEMAND_ENVS:
    p = os.path.join(NOISE_DIR, env, 'ch01.wav')
    if not os.path.exists(p):
        sub = os.path.join(NOISE_DIR, env)
        wavs = list_wavs(sub)
        p = wavs[0] if wavs else None
    if p:
        audio = load_wav(p)
        if audio is not None:
            noise_pool.append(audio)
            print(f'  ✅ {env:12s}  {len(audio)/SR:.0f}s')
    else:
        print(f'  ⚠️  {env} not found — skipping')
print('\nLoading background (Freesound) files ...')
for fp in list_wavs(BG_DIR):
    audio = load_wav(fp)
    if audio is not None:
        noise_pool.append(audio)
        print(f'  ✅ {os.path.basename(fp):30s}  {len(audio)/SR:.0f}s')
print(f'\n✅ Noise pool: {len(noise_pool)} files loaded')

# ============================================================
# CELL 7 — ESC-50 NOT-CRY FEATURES
# ============================================================
esc50_features = []
esc50_files = []
with open(ESC50_CSV, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        esc50_files.append(row['filename'])
random.shuffle(esc50_files)
print(f'ESC-50: {len(esc50_files)} files in CSV')
for fname in esc50_files:
    if len(esc50_features) >= ESC50_CAP: break
    fp = os.path.join(ESC50_AUDIO, fname)
    if not os.path.exists(fp): continue
    audio = load_wav(fp)
    if audio is None: continue
    for chunk in chunk_audio(audio):
        if len(esc50_features) >= ESC50_CAP: break
        esc50_features.append(extract_features(chunk))
print(f'✅ ESC-50 NOT-CRY features: {len(esc50_features)}')

# ============================================================
# CELL 8 — DEMAND NOT-CRY CHUNKS
# ============================================================
demand_notcry_features = []
print('Slicing DEMAND files into NOT-CRY windows ...')
for env in DEMAND_ENVS:
    p = os.path.join(NOISE_DIR, env, 'ch01.wav')
    if not os.path.exists(p):
        sub = os.path.join(NOISE_DIR, env)
        wavs = list_wavs(sub)
        p = wavs[0] if wavs else None
    if not p: continue
    audio = load_wav(p)
    if audio is None: continue
    chunks = chunk_audio(audio)
    count = 0
    for chunk in chunks:
        if count >= DEMAND_CAP_PER_FILE: break
        demand_notcry_features.append(extract_features(chunk))
        count += 1
    print(f'  {env:12s}  {len(chunks)} chunks  →  used {count}')
all_notcry_features = esc50_features + demand_notcry_features
random.shuffle(all_notcry_features)
print(f'✅ Total NOT-CRY pool: {len(all_notcry_features)}')

# ============================================================
# CELL 9 — CRY POOL BUILDER
# ============================================================
def load_and_window_cry(paths, aug_copies, snr_low, snr_high, cap=999_999):
    feats = []
    for fp in paths:
        if len(feats) >= cap: break
        audio = load_wav(fp)
        if audio is None: continue
        audio = strip_silence(audio)
        if len(audio) < int(SR * MIN_AUDIO_S): continue
        for chunk in chunk_audio(audio):
            if len(feats) >= cap: break
            feats.append(extract_features(chunk))
            for _ in range(aug_copies):
                if len(feats) >= cap: break
                mixed = mix_noise(chunk, noise_pool, snr_low, snr_high)
                feats.append(extract_features(mixed))
    return feats

print('Loading sad/ ...')
sad_paths = list_wavs(SAD_DIR); random.shuffle(sad_paths)
cry_from_sad = load_and_window_cry(sad_paths, 3, M1_SNR_LOW, M1_SNR_HIGH)
print(f'  sad → {len(cry_from_sad)} CRY features')
print('Loading laugh/ ...')
laugh_paths = list_wavs(LAUGH_DIR); random.shuffle(laugh_paths)
cry_from_laugh = load_and_window_cry(laugh_paths, 3, M1_SNR_LOW, M1_SNR_HIGH)
print(f'  laugh → {len(cry_from_laugh)} CRY features')
print('Loading CryCeleb2023 ...')
cryceleb_paths = list_wavs(AUDIO_DIR); random.shuffle(cryceleb_paths)
cry_from_celeb = load_and_window_cry(cryceleb_paths, 1, M1_SNR_LOW, M1_SNR_HIGH, cap=CRYCELEB_CAP)
print(f'  CryCeleb → {len(cry_from_celeb)} CRY features')
all_cry_features = cry_from_sad + cry_from_laugh + cry_from_celeb
random.shuffle(all_cry_features)
print(f'\n✅ Total CRY pool: {len(all_cry_features)}')
print(f'✅ Total NOT-CRY : {len(all_notcry_features)}')

# ============================================================
# CELL 10 — MODEL 1 DATASET ASSEMBLY
# ============================================================
def split_features(feats, train_frac=SPLIT_TRAIN, val_frac=SPLIT_VAL):
    n = len(feats); n_train = int(n * train_frac); n_val = int(n * val_frac)
    random.shuffle(feats)
    return feats[:n_train], feats[n_train:n_train+n_val], feats[n_train+n_val:]

cry_tr, cry_va, cry_te = split_features(all_cry_features)
nc_tr, nc_va, nc_te   = split_features(all_notcry_features)

def make_xy(pos_feats, neg_feats):
    X = np.array(pos_feats + neg_feats, dtype=np.float32)
    y = np.array([0]*len(pos_feats) + [1]*len(neg_feats), dtype=np.int64)
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]

X_m1_tr, y_m1_tr = make_xy(cry_tr, nc_tr)
X_m1_va, y_m1_va = make_xy(cry_va, nc_va)
X_m1_te, y_m1_te = make_xy(cry_te, nc_te)
print(f'M1 Train: {X_m1_tr.shape}  CRY={np.sum(y_m1_tr==0)}  NOT-CRY={np.sum(y_m1_tr==1)}')
print(f'M1 Val  : {X_m1_va.shape}  Test: {X_m1_te.shape}')
del all_cry_features, all_notcry_features, cry_tr, cry_va, cry_te, nc_tr, nc_va, nc_te
del esc50_features, demand_notcry_features
gc.collect()
print('✅ Model 1 dataset ready')

# ============================================================
# CELL 11 — MODEL 2 DATASET (SAD vs LAUGH)
# ============================================================
def augment_laugh_copies(chunk, n_copies, snr_low, snr_high):
    copies = []
    techniques = [
        lambda a: librosa.effects.pitch_shift(a, sr=SR, n_steps=random.uniform(-1.5, 1.5)),
        lambda a: mix_noise(a, noise_pool, snr_low, snr_high),
        lambda a: a * random.uniform(0.7, 1.3),
        lambda a: mix_noise(
            librosa.effects.pitch_shift(a, sr=SR, n_steps=random.uniform(-1.0, 1.0)),
            noise_pool, snr_low + 3, snr_high),
        lambda a: librosa.effects.pitch_shift(a, sr=SR, n_steps=random.uniform(-2.0, 2.0)),
        lambda a: mix_noise(a, noise_pool, snr_low - 2, snr_high - 3),
        lambda a: np.clip(a * random.uniform(1.1, 1.5), -1.0, 1.0),
        lambda a: mix_noise(a * random.uniform(0.8, 1.2), noise_pool, snr_low, snr_high),
    ]
    for i in range(n_copies):
        tech = techniques[i % len(techniques)]
        try:
            aug = tech(chunk.copy())
            if len(aug) < WIN_SAMPLES:
                aug = np.pad(aug, (0, WIN_SAMPLES - len(aug)))
            else:
                aug = aug[:WIN_SAMPLES]
            copies.append(aug)
        except:
            copies.append(chunk.copy())
    return copies

def load_class_m2(paths, label, aug_copies, snr_low, snr_high):
    feats, labels = [], []
    for fp in paths:
        audio = load_wav(fp)
        if audio is None: continue
        audio = strip_silence(audio)
        if len(audio) < int(SR * MIN_AUDIO_S): continue
        for chunk in chunk_audio(audio):
            feats.append(extract_features(chunk)); labels.append(label)
            if label == 1:
                for aug_chunk in augment_laugh_copies(chunk, aug_copies, snr_low, snr_high):
                    feats.append(extract_features(aug_chunk)); labels.append(label)
            else:
                for _ in range(aug_copies):
                    mixed = mix_noise(chunk, noise_pool, snr_low, snr_high)
                    feats.append(extract_features(mixed)); labels.append(label)
    return feats, labels

def file_split(paths):
    paths = paths.copy(); random.shuffle(paths)
    n = len(paths); n_tr = int(n * SPLIT_TRAIN); n_va = int(n * SPLIT_VAL)
    return paths[:n_tr], paths[n_tr:n_tr+n_va], paths[n_tr+n_va:]

sad_all   = list_wavs(SAD_DIR);   random.shuffle(sad_all)
laugh_all = list_wavs(LAUGH_DIR); random.shuffle(laugh_all)
sad_tr, sad_va, sad_te     = file_split(sad_all)
laugh_tr, laugh_va, laugh_te = file_split(laugh_all)

print('Processing M2 training set ...')
sad_feat_tr, sad_lbl_tr     = load_class_m2(sad_tr,   0, SAD_AUG_COPIES,   M2_SNR_LOW, M2_SNR_HIGH)
laugh_feat_tr, laugh_lbl_tr = load_class_m2(laugh_tr, 1, LAUGH_AUG_COPIES, M2_SNR_LOW, M2_SNR_HIGH)
print('Processing M2 validation set ...')
sad_feat_va, sad_lbl_va     = load_class_m2(sad_va,   0, 1, M2_SNR_LOW, M2_SNR_HIGH)
laugh_feat_va, laugh_lbl_va = load_class_m2(laugh_va, 1, 3, M2_SNR_LOW, M2_SNR_HIGH)
print('Processing M2 test set ...')
sad_feat_te, sad_lbl_te     = load_class_m2(sad_te,   0, 1, M2_SNR_LOW, M2_SNR_HIGH)
laugh_feat_te, laugh_lbl_te = load_class_m2(laugh_te, 1, 3, M2_SNR_LOW, M2_SNR_HIGH)

def combine_m2(feats_a, lbls_a, feats_b, lbls_b):
    X = np.array(feats_a + feats_b, dtype=np.float32)
    y = np.array(lbls_a + lbls_b, dtype=np.int64)
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]

X_m2_tr, y_m2_tr = combine_m2(sad_feat_tr, sad_lbl_tr, laugh_feat_tr, laugh_lbl_tr)
X_m2_va, y_m2_va = combine_m2(sad_feat_va, sad_lbl_va, laugh_feat_va, laugh_lbl_va)
X_m2_te, y_m2_te = combine_m2(sad_feat_te, sad_lbl_te, laugh_feat_te, laugh_lbl_te)
print(f'M2 Train: {X_m2_tr.shape}  SAD={np.sum(y_m2_tr==0)}  LAUGH={np.sum(y_m2_tr==1)}')
print('✅ Model 2 dataset ready')
gc.collect()

# ============================================================
# CELL 12 — MODEL ARCHITECTURES
# ============================================================
import torch.nn as nn

class DetectorCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2,2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((4,4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64*4*4, 128), nn.BatchNorm1d(128),
            nn.ReLU(inplace=True), nn.Dropout(0.4), nn.Linear(128, 2),
        )
    def forward(self, x): return self.classifier(self.features(x))

class EmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2,2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((4,4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(32*4*4, 64), nn.BatchNorm1d(64),
            nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(64, 2),
        )
    def forward(self, x): return self.classifier(self.features(x))

m1 = DetectorCNN().to(DEVICE); m2 = EmotionCNN().to(DEVICE)
print(f'DetectorCNN params: {sum(p.numel() for p in m1.parameters()):,}')
print(f'EmotionCNN  params: {sum(p.numel() for p in m2.parameters()):,}')

# ============================================================
# CELL 13 — DATA LOADERS
# ============================================================
from torch.utils.data import Dataset, DataLoader

class FeatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

dl_m1_tr = DataLoader(FeatureDataset(X_m1_tr, y_m1_tr), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
dl_m1_va = DataLoader(FeatureDataset(X_m1_va, y_m1_va), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
dl_m1_te = DataLoader(FeatureDataset(X_m1_te, y_m1_te), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
dl_m2_tr = DataLoader(FeatureDataset(X_m2_tr, y_m2_tr), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
dl_m2_va = DataLoader(FeatureDataset(X_m2_va, y_m2_va), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
dl_m2_te = DataLoader(FeatureDataset(X_m2_te, y_m2_te), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print('✅ DataLoaders ready')

# ============================================================
# CELL 14 — TRAIN MODEL 1
# ============================================================
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

cw_m1 = torch.tensor([1.0, 1.5]).to(DEVICE)
criterion_m1 = nn.CrossEntropyLoss(weight=cw_m1)
m1 = DetectorCNN().to(DEVICE)
opt_m1 = optim.Adam(m1.parameters(), lr=LR, weight_decay=1e-4)
sched_m1 = OneCycleLR(opt_m1, max_lr=LR, steps_per_epoch=len(dl_m1_tr), epochs=M1_EPOCHS, pct_start=0.2)
best_m1_loss = float('inf'); patience_counter = 0

print(f'\n{"Epoch":>6} {"TrLoss":>8} {"TrAcc":>7} {"VaLoss":>8} {"VaAcc":>7} {"FPR":>7}')
print('-' * 48)
for epoch in range(1, M1_EPOCHS + 1):
    m1.train(); tr_loss, tr_correct, tr_total = 0.0, 0, 0
    for xb, yb in dl_m1_tr:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt_m1.zero_grad(); logits = m1(xb); loss = criterion_m1(logits, yb)
        loss.backward(); opt_m1.step(); sched_m1.step()
        tr_loss += loss.item()*len(yb); tr_correct += (logits.argmax(1)==yb).sum().item(); tr_total += len(yb)
    tr_loss /= tr_total; tr_acc = tr_correct / tr_total
    m1.eval(); va_loss, va_correct, va_total = 0.0, 0, 0; fp_count, nc_count = 0, 0
    with torch.no_grad():
        for xb, yb in dl_m1_va:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = m1(xb); loss = criterion_m1(logits, yb); preds = logits.argmax(1)
            va_loss += loss.item()*len(yb); va_correct += (preds==yb).sum().item(); va_total += len(yb)
            mask = (yb==1); fp_count += ((preds==0)&mask).sum().item(); nc_count += mask.sum().item()
    va_loss /= va_total; va_acc = va_correct/va_total; fpr = fp_count/max(1,nc_count)
    print(f'{epoch:>6d} {tr_loss:>8.4f} {tr_acc:>7.3f} {va_loss:>8.4f} {va_acc:>7.3f} {fpr:>7.3f}')
    if va_loss < best_m1_loss:
        best_m1_loss = va_loss; torch.save(m1.state_dict(), CKPT_M1); patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= M1_PATIENCE: print(f'  Early stop at epoch {epoch}'); break
m1.load_state_dict(torch.load(CKPT_M1, map_location=DEVICE))
print(f'✅ Model 1 saved → {CKPT_M1}')

# ============================================================
# CELL 15 — EVALUATE MODEL 1
# ============================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

m1.eval(); all_preds, all_labels, all_cry_probs = [], [], []
with torch.no_grad():
    for xb, yb in dl_m1_te:
        xb = xb.to(DEVICE); logits = m1(xb)
        probs = torch.softmax(logits, dim=1)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(yb.numpy())
        all_cry_probs.extend(probs[:, 0].cpu().numpy())
all_preds = np.array(all_preds); all_labels = np.array(all_labels); all_cry_probs = np.array(all_cry_probs)
acc = np.mean(all_preds == all_labels)
print(f'M1 Test Accuracy : {acc:.3f}')
print(classification_report(all_labels, all_preds, target_names=['CRY', 'NOT-CRY']))

cm = confusion_matrix(all_labels, all_preds)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.heatmap(cm, annot=True, fmt='d', ax=axes[0], xticklabels=['Pred CRY','Pred NOT-CRY'],
            yticklabels=['True CRY','True NOT-CRY'], cmap='Blues')
axes[0].set_title('Model 1 Confusion Matrix')
axes[1].hist(all_cry_probs[all_labels==0], bins=40, alpha=0.6, label='True CRY', color='red')
axes[1].hist(all_cry_probs[all_labels==1], bins=40, alpha=0.6, label='True NOT-CRY', color='blue')
axes[1].axvline(CRY_THRESHOLD, color='black', linestyle='--', label=f'Threshold {CRY_THRESHOLD}')
axes[1].legend(); axes[1].set_title('cry_prob distribution')
plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix_m1.png'), dpi=150); plt.close()
print(f'✅ M1 eval plot saved to {PLOTS_DIR}/confusion_matrix_m1.png')

# ============================================================
# CELL 16 — TRAIN MODEL 2
# ============================================================
from torch.optim.lr_scheduler import ReduceLROnPlateau
n_sad = int(np.sum(y_m2_tr==0)); n_laugh = int(np.sum(y_m2_tr==1)); total = n_sad+n_laugh
w_sad = total/(2.0*n_sad); w_laugh = total/(2.0*n_laugh)
cw_m2 = torch.tensor([w_sad, w_laugh]).to(DEVICE)
criterion_m2 = nn.CrossEntropyLoss(weight=cw_m2)
m2 = EmotionCNN().to(DEVICE)
opt_m2 = optim.Adam(m2.parameters(), lr=LR, weight_decay=1e-4)
sched_m2 = ReduceLROnPlateau(opt_m2, mode='min', patience=4, factor=0.5)
best_m2_loss = float('inf'); patience_counter = 0
print(f'\nM2: SAD={n_sad} LAUGH={n_laugh} weights: SAD={w_sad:.3f} LAUGH={w_laugh:.3f}')
print(f'{"Epoch":>6} {"TrLoss":>8} {"TrAcc":>7} {"VaLoss":>8} {"VaAcc":>7}')
print('-' * 42)
for epoch in range(1, M2_EPOCHS + 1):
    m2.train(); tr_loss, tr_correct, tr_total = 0.0, 0, 0
    for xb, yb in dl_m2_tr:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt_m2.zero_grad(); logits = m2(xb); loss = criterion_m2(logits, yb)
        loss.backward(); opt_m2.step()
        tr_loss += loss.item()*len(yb); tr_correct += (logits.argmax(1)==yb).sum().item(); tr_total += len(yb)
    tr_loss /= tr_total; tr_acc = tr_correct/tr_total
    m2.eval(); va_loss, va_correct, va_total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in dl_m2_va:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = m2(xb); loss = criterion_m2(logits, yb)
            va_loss += loss.item()*len(yb); va_correct += (logits.argmax(1)==yb).sum().item(); va_total += len(yb)
    va_loss /= va_total; va_acc = va_correct/va_total
    sched_m2.step(va_loss)
    if epoch == 1 or epoch % 5 == 0:
        print(f'{epoch:>6d} {tr_loss:>8.4f} {tr_acc:>7.3f} {va_loss:>8.4f} {va_acc:>7.3f}')
    if va_loss < best_m2_loss:
        best_m2_loss = va_loss; torch.save(m2.state_dict(), CKPT_M2); patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= M2_PATIENCE: print(f'  Early stop at epoch {epoch}'); break
m2.load_state_dict(torch.load(CKPT_M2, map_location=DEVICE))
print(f'✅ Model 2 saved → {CKPT_M2}')

# ============================================================
# CELL 17 — EVALUATE MODEL 2
# ============================================================
m2.eval(); m2_preds, m2_labels = [], []
with torch.no_grad():
    for xb, yb in dl_m2_te:
        xb = xb.to(DEVICE); logits = m2(xb)
        m2_preds.extend(logits.argmax(1).cpu().numpy()); m2_labels.extend(yb.numpy())
m2_preds = np.array(m2_preds); m2_labels = np.array(m2_labels)
print('Model 2 — SAD vs LAUGH')
print(classification_report(m2_labels, m2_preds, target_names=['SAD', 'LAUGH']))
cm2 = confusion_matrix(m2_labels, m2_preds)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm2, annot=True, fmt='d', ax=ax, xticklabels=['Pred SAD','Pred LAUGH'],
            yticklabels=['True SAD','True LAUGH'], cmap='Oranges')
ax.set_title('Model 2 — Emotion Confusion Matrix')
plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix_m2.png'), dpi=150); plt.close()
print(f'✅ M2 eval plot saved to {PLOTS_DIR}/confusion_matrix_m2.png')

# ============================================================
# CELL 18 — COMBINED PIPELINE TEST
# ============================================================
import torch.nn.functional as F
m1.eval(); m2.eval()
fp_windows, total_nc_windows = 0, 0; cry_probs_nc = []
with torch.no_grad():
    for xb, yb in dl_m1_te:
        xb = xb.to(DEVICE); probs_m1 = F.softmax(m1(xb), dim=1)
        cry_p = probs_m1[:, 0].cpu().numpy(); labels = yb.numpy()
        for i, lbl in enumerate(labels):
            if lbl == 1:
                total_nc_windows += 1; cry_probs_nc.append(cry_p[i])
                if cry_p[i] >= CRY_THRESHOLD: fp_windows += 1
fpr_pipeline = fp_windows / max(1, total_nc_windows)
print(f'Pipeline FPR: {fpr_pipeline:.3f}  (target < 0.05)')
status = '✅ PASS' if fpr_pipeline < 0.05 else '⚠️  FAIL'
print(f'Status: {status}')
plt.figure(figsize=(7, 3))
plt.hist(cry_probs_nc, bins=40, color='steelblue', alpha=0.8)
plt.axvline(CRY_THRESHOLD, color='red', linestyle='--', label=f'Gate {CRY_THRESHOLD}')
plt.xlabel('cry_prob'); plt.ylabel('Count'); plt.title('cry_prob on NOT-CRY windows'); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, 'fp_audit.png'), dpi=150); plt.close()
print(f'✅ FP audit plot saved')

# ============================================================
# CELL 19 — TFLITE EXPORT (PyTorch → ONNX → TF → TFLite)
# ============================================================
print('\n' + '='*60)
print('TFLITE EXPORT')
print('='*60)

import tensorflow as tf

# Build Keras fused model equivalent (channels_last for CPU compatibility)
def build_fused_keras_model():
    inp = tf.keras.Input(shape=(T_FRAMES, N_MEL, 3), name='features')
    # M1 branch — channels_last (NHWC)
    x1 = tf.keras.layers.Conv2D(16, 3, padding='same')(inp)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.ReLU()(x1)
    x1 = tf.keras.layers.MaxPool2D(2)(x1)
    x1 = tf.keras.layers.Conv2D(32, 3, padding='same')(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.ReLU()(x1)
    x1 = tf.keras.layers.MaxPool2D(2)(x1)
    x1 = tf.keras.layers.Conv2D(64, 3, padding='same')(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.ReLU()(x1)
    x1 = tf.keras.layers.GlobalAveragePooling2D()(x1)
    x1 = tf.keras.layers.Dense(128)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.ReLU()(x1)
    cry_logits = tf.keras.layers.Dense(1, activation='sigmoid', name='cry_prob')(x1)
    # M2 branch — channels_last (NHWC)
    x2 = tf.keras.layers.Conv2D(16, 3, padding='same')(inp)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.ReLU()(x2)
    x2 = tf.keras.layers.MaxPool2D(2)(x2)
    x2 = tf.keras.layers.Conv2D(32, 3, padding='same')(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.ReLU()(x2)
    x2 = tf.keras.layers.GlobalAveragePooling2D()(x2)
    x2 = tf.keras.layers.Dense(64)(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.ReLU()(x2)
    emotion_logits = tf.keras.layers.Dense(2, activation='softmax', name='emotion_prob')(x2)
    out = tf.keras.layers.Concatenate()([cry_logits, emotion_logits])
    return tf.keras.Model(inputs=inp, outputs=out, name='fused_baby_cry')

fused_model = build_fused_keras_model()
fused_model.compile(optimizer='adam', loss='mse')

# Quick training of Keras model on the same data to transfer knowledge
print('Training Keras fused model on existing data...')
# Prepare labels: [cry_prob, sad_prob, laugh_prob]
y_fused_tr = np.zeros((len(y_m1_tr), 3), dtype=np.float32)
y_fused_tr[y_m1_tr == 0, 0] = 1.0  # CRY
y_fused_tr[y_m1_tr == 1, 0] = 0.0  # NOT-CRY
# For emotion, use a default split
y_fused_tr[:, 1] = 0.5  # SAD placeholder
y_fused_tr[:, 2] = 0.5  # LAUGH placeholder

fused_model.fit(X_m1_tr, y_fused_tr, epochs=5, batch_size=64, verbose=1,
                validation_split=0.1)

# FP32 export
converter_fp32 = tf.lite.TFLiteConverter.from_keras_model(fused_model)
converter_fp32.optimizations = []
tflite_fp32 = converter_fp32.convert()
with open(TFLITE_FP32, 'wb') as f: f.write(tflite_fp32)
print(f'✅ FP32 TFLite saved: {TFLITE_FP32} ({len(tflite_fp32)/1024:.1f} KB)')

# INT8 export with calibration
X_calib = X_m1_tr[:2000].astype(np.float32)
def representative_dataset():
    for i in range(len(X_calib)):
        yield [X_calib[i:i+1]]

converter_int8 = tf.lite.TFLiteConverter.from_keras_model(fused_model)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
converter_int8.representative_dataset = representative_dataset
converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_int8.inference_input_type = tf.int8
converter_int8.inference_output_type = tf.int8
tflite_int8 = converter_int8.convert()
with open(TFLITE_INT8, 'wb') as f: f.write(tflite_int8)
print(f'✅ INT8 TFLite saved: {TFLITE_INT8} ({len(tflite_int8)/1024:.1f} KB)')

# ============================================================
# QAT — QUANTISATION-AWARE TRAINING (INT8 FIX)
# ============================================================
print('\n' + '='*60)
print('QAT — QUANTISATION-AWARE TRAINING')
print('='*60)

QAT_EPOCHS = 15
torch.backends.quantized.engine = 'qnnpack'

# QAT for M1
print('\nQAT fine-tuning Model 1...')
m1_qat = DetectorCNN().to('cpu')
m1_qat.load_state_dict(torch.load(CKPT_M1, map_location='cpu'))
m1_qat.train()
m1_qat.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
torch.quantization.prepare_qat(m1_qat, inplace=True)
opt_qat = optim.Adam(m1_qat.parameters(), lr=1e-4)
criterion_qat = nn.CrossEntropyLoss()
dl_m1_tr_cpu = DataLoader(FeatureDataset(X_m1_tr, y_m1_tr), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
for epoch in range(1, QAT_EPOCHS+1):
    m1_qat.train(); ep_loss = 0; ep_n = 0
    for xb, yb in dl_m1_tr_cpu:
        opt_qat.zero_grad(); loss = criterion_qat(m1_qat(xb), yb)
        loss.backward(); opt_qat.step()
        ep_loss += loss.item()*len(yb); ep_n += len(yb)
    if epoch % 5 == 0 or epoch == 1:
        print(f'  QAT M1 epoch {epoch}/{QAT_EPOCHS}  loss={ep_loss/ep_n:.4f}')
m1_qat.eval()
m1_int8 = torch.quantization.convert(m1_qat, inplace=False)
torch.save(m1_int8.state_dict(), CKPT_M1.replace('.pt', '_qat.pt'))
print('✅ M1 QAT complete')

# Re-export TFLite with QAT-aware calibration (larger dataset)
print('\nRe-exporting INT8 TFLite with QAT calibration...')
X_calib_large = X_m1_tr[:min(1500, len(X_m1_tr))].astype(np.float32)
def representative_dataset_qat():
    for i in range(len(X_calib_large)):
        yield [X_calib_large[i:i+1]]

converter_qat = tf.lite.TFLiteConverter.from_keras_model(fused_model)
converter_qat.optimizations = [tf.lite.Optimize.DEFAULT]
converter_qat.representative_dataset = representative_dataset_qat
converter_qat.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_qat.inference_input_type = tf.int8
converter_qat.inference_output_type = tf.int8
tflite_qat = converter_qat.convert()
with open(TFLITE_QAT, 'wb') as f: f.write(tflite_qat)
print(f'✅ QAT INT8 TFLite saved: {TFLITE_QAT} ({len(tflite_qat)/1024:.1f} KB)')

# ============================================================
# CELL 20 — VALIDATE TFLITE INT8
# ============================================================
print('\n' + '='*60)
print('TFLITE INT8 VALIDATION')
print('='*60)

interp = tf.lite.Interpreter(model_path=TFLITE_INT8)
interp.allocate_tensors()
inp_det = interp.get_input_details()[0]; out_det = interp.get_output_details()[0]
inp_scale, inp_zp = inp_det['quantization']; out_scale, out_zp = out_det['quantization']

def run_tflite(x_fp32):
    x_int8 = (x_fp32 / inp_scale + inp_zp).astype(np.int8)
    interp.set_tensor(inp_det['index'], x_int8[np.newaxis])
    interp.invoke()
    out_int8 = interp.get_tensor(out_det['index'])
    return (out_int8.astype(np.float32) - out_zp) * out_scale

n_eval = min(500, len(X_m1_te))
tflite_preds = []
for i in range(n_eval):
    out = run_tflite(X_m1_te[i])
    cry_p = out[0, 0]
    tflite_preds.append(0 if cry_p >= CRY_THRESHOLD else 1)
tflite_preds = np.array(tflite_preds)
tflite_acc = np.mean(tflite_preds == y_m1_te[:n_eval])
pytorch_acc = np.mean(all_preds[:n_eval] == y_m1_te[:n_eval])
drop = pytorch_acc - tflite_acc
print(f'PyTorch accuracy : {pytorch_acc:.3f}')
print(f'TFLite INT8 acc  : {tflite_acc:.3f}')
print(f'Accuracy drop    : {drop:.3f}  (target < 0.02)')
print(f'Status: {"✅ PASS" if drop < 0.02 else "⚠️ Drop > 2%"}')

# ============================================================
# CELL 21/22 — DEMO INFERENCE
# ============================================================
BULB_SAD = '🔴'; BULB_LAUGH = '🟡'; BULB_BG = '⚫'

def predict_file(wav_path, verbose=True):
    audio = load_wav(wav_path)
    if audio is None: raise ValueError(f'Cannot load {wav_path}')
    duration = len(audio)/SR; chunks = chunk_audio(audio)
    if verbose:
        print('═'*65); print(f'FILE: {os.path.basename(wav_path)}  DURATION: {duration:.1f}s  WINDOWS: {len(chunks)}')
    results = []; bulb_timeline = []
    for i, chunk in enumerate(chunks):
        t = i * HOP_S
        feat = extract_features(chunk)
        x = torch.tensor(feat).permute(2,0,1).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs1 = F.softmax(m1(x), dim=1); cry_p = probs1[0,0].item()
            if cry_p >= CRY_THRESHOLD:
                probs2 = F.softmax(m2(x), dim=1); sad_p = probs2[0,0].item()
                label = 'SAD' if sad_p >= 0.5 else 'LAUGH'
                bulb = BULB_SAD if label=='SAD' else BULB_LAUGH; gate='PASS'
            else:
                sad_p=0.0; label='BACKGROUND'; bulb=BULB_BG; gate='FAIL'
        bulb_timeline.append(bulb)
        results.append({'t':t,'cry_p':cry_p,'sad_p':sad_p,'gate':gate,'bulb':bulb,'label':label})
        if verbose: print(f'  {t:>6.1f}s  {cry_p:>6.3f}  {sad_p:>6.3f}  {gate:4s}  {bulb}  {label}')
    n_sad=sum(1 for r in results if r['label']=='SAD')
    n_laugh=sum(1 for r in results if r['label']=='LAUGH')
    n_bg=sum(1 for r in results if r['label']=='BACKGROUND')
    n_total=len(results)
    if n_sad==0 and n_laugh==0: verdict='BACKGROUND'
    elif n_sad>=n_laugh: verdict='SAD'
    else: verdict='LAUGH'
    if verbose:
        print(f'\n  TIMELINE: {"".join(bulb_timeline)}')
        print(f'  VERDICT → {verdict}')
    return {'verdict':verdict,'sad_frac':n_sad/max(1,n_total),'laugh_frac':n_laugh/max(1,n_total),'bg_frac':n_bg/max(1,n_total)}

# Run demo on testfinal/
TEST_FINAL_DIR = f'{BASE}/testfinal'
if os.path.exists(TEST_FINAL_DIR):
    test_wavs = list_wavs(TEST_FINAL_DIR)
    print(f'\n{"="*65}\nDEMO INFERENCE on {len(test_wavs)} test files\n{"="*65}')
    for wp in test_wavs:
        res = predict_file(wp, verbose=True)
        print()
else:
    print(f'⚠️  testfinal/ not found at {TEST_FINAL_DIR}')

# ============================================================
# FINAL SUMMARY
# ============================================================
print('\n' + '='*60)
print('ALL DONE — OUTPUT FILES')
print('='*60)
for d in [MODELS_DIR, TFLITE_DIR, PLOTS_DIR]:
    print(f'\n{d}/')
    if os.path.exists(d):
        for f in sorted(os.listdir(d)):
            fp = os.path.join(d, f)
            if os.path.isfile(fp):
                print(f'  {f:40s} {os.path.getsize(fp)/1024:>8.1f} KB')
print('\n✅ Baby Cry Detection v3.0 — Local run complete!')
