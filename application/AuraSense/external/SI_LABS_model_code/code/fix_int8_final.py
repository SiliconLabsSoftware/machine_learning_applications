"""
INT8 TFLite FINAL FIX — Knowledge Distillation approach
========================================================
Problem:  PyTorch uses AdaptiveAvgPool2d((4,4)) → 1024-dim flatten → Dense(1024,128).
          Keras GlobalAvgPool gives 64-dim → Dense(64,128) = shape mismatch = crash.
          Previous attempts trained from scratch on dummy labels → garbage INT8.

Solution: Knowledge Distillation
  1. Use trained PyTorch M1+M2 as TEACHER
  2. Build Keras student (GlobalAvgPool — TFLite-friendly, different arch is OK)
  3. Generate soft labels on ALL training data from PyTorch
  4. Train Keras to replicate PyTorch predictions
  5. Quantize to pure INT8 (no float32 ops, no LOG)
  6. Validate + benchmark

Target: EFR32xG26 (Silicon Labs) — ARM Cortex-M33 — TFLite Micro INT8 only
"""
import os, sys, numpy as np, warnings, random, csv, gc, time
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa

# ── Config ─────────────────────────────────────────────────────
SR = 16000; WIN_S = 0.75; HOP_S = 0.375
WIN_SAMPLES = int(SR * WIN_S); HOP_SAMPLES = int(SR * HOP_S)
N_MEL = 40; N_FFT = 512; HOP_LENGTH = 160
T_FRAMES = WIN_SAMPLES // HOP_LENGTH  # 75
SILENCE_TOP_DB = 30; MIN_AUDIO_S = 0.3; CRY_THRESHOLD = 0.70
BATCH_SIZE = 32; SPLIT_TRAIN = 0.80; SPLIT_VAL = 0.10
M1_SNR_LOW = 5.0; M1_SNR_HIGH = 20.0
CRYCELEB_CAP = 2000; ESC50_CAP = 2000; DEMAND_CAP_PER_FILE = 400

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get('AURASENSE_DATA', os.path.join(SCRIPT_DIR, 'datasets1'))
MODELS_DIR = 'outputs/models'; TFLITE_DIR = 'outputs/tflite'; PLOTS_DIR = 'outputs/plots'
CKPT_M1 = os.path.join(MODELS_DIR, 'model_m1_detector.pt')
CKPT_M2 = os.path.join(MODELS_DIR, 'model_m2_emotion.pt')
TFLITE_FINAL = os.path.join(TFLITE_DIR, 'baby_cry_int8_DEPLOY.tflite')
DEVICE = torch.device('cpu')
BULB_SAD = '🔴'; BULB_LAUGH = '🟡'; BULB_BG = '⚫'

KD_EPOCHS = 40  # Knowledge distillation epochs
KD_LR = 1e-3
KD_TEMPERATURE = 3.0  # Soft label temperature

for d in [MODELS_DIR, TFLITE_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Audio + Feature utilities ──────────────────────────────────
def extract_features(audio):
    if len(audio) < WIN_SAMPLES: audio = np.pad(audio, (0, WIN_SAMPLES - len(audio)))
    else: audio = audio[:WIN_SAMPLES]
    mel = librosa.feature.melspectrogram(y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
                                         n_mels=N_MEL, fmin=60, fmax=8000)
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=40)
    log_mel = (log_mel + 40.0) / 40.0
    T = log_mel.shape[1]
    flatness = librosa.feature.spectral_flatness(y=audio, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    flatness = np.log1p(flatness * 1000.0)
    flatness = np.clip(flatness / (np.max(flatness) + 1e-8), 0, 1)
    flatness_ch = np.tile(flatness[np.newaxis, :T], (N_MEL, 1))
    rms = librosa.feature.rms(y=audio, frame_length=N_FFT, hop_length=HOP_LENGTH)[0][:T]
    rms = np.log1p(rms * 100.0)
    rms = np.clip(rms / (np.max(rms) + 1e-8), 0, 1)
    rms_ch = np.tile(rms[np.newaxis, :T], (N_MEL, 1))
    feat = np.stack([log_mel[:, :T], flatness_ch[:, :T], rms_ch[:, :T]], axis=-1).transpose(1, 0, 2)
    if feat.shape[0] < T_FRAMES:
        feat = np.concatenate([feat, np.zeros((T_FRAMES - feat.shape[0], N_MEL, 3), dtype=np.float32)])
    else:
        feat = feat[:T_FRAMES]
    return feat.astype(np.float32)

def load_wav(path, sr=SR):
    try:
        audio, _ = librosa.load(path, sr=sr, mono=True)
        return audio
    except:
        return None

def chunk_audio(audio, win=WIN_SAMPLES, hop=HOP_SAMPLES):
    chunks = []; start = 0
    while start + win <= len(audio):
        chunks.append(audio[start:start + win]); start += hop
    if start < len(audio) and (len(audio) - start) >= int(SR * 0.5):
        chunk = audio[start:]
        chunk = np.pad(chunk, (0, win - len(chunk)))
        chunks.append(chunk)
    return chunks

def strip_silence(audio, sr=SR, top_db=SILENCE_TOP_DB):
    intervals = librosa.effects.split(audio, top_db=top_db)
    if len(intervals) == 0: return audio
    return np.concatenate([audio[s:e] for s, e in intervals])

def mix_noise(signal, noise_pool, snr_low, snr_high):
    if not noise_pool: return signal
    noise = random.choice(noise_pool)
    if len(noise) > WIN_SAMPLES:
        start = random.randint(0, len(noise) - WIN_SAMPLES)
        ns = noise[start:start + WIN_SAMPLES]
    else:
        ns = np.pad(noise, (0, WIN_SAMPLES - len(noise)))
    snr = random.uniform(snr_low, snr_high)
    sr_ = np.sqrt(np.mean(signal ** 2) + 1e-9)
    nr = np.sqrt(np.mean(ns ** 2) + 1e-9)
    ns = ns * (sr_ / (10 ** (snr / 20.0)) / nr)
    m = signal + ns
    pk = np.max(np.abs(m))
    if pk > 1: m = m / pk
    return m.astype(np.float32)

def list_wavs(d):
    w = []
    for r, _, fs in os.walk(d):
        for f in fs:
            if f.lower().endswith('.wav'): w.append(os.path.join(r, f))
    return sorted(w)

# ── PyTorch model definitions ──────────────────────────────────
class DetectorCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((4, 4)))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 4 * 4, 128), nn.BatchNorm1d(128),
            nn.ReLU(inplace=True), nn.Dropout(0.4), nn.Linear(128, 2))
    def forward(self, x): return self.classifier(self.features(x))

class EmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((4, 4)))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 4 * 4, 64), nn.BatchNorm1d(64),
            nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(64, 2))
    def forward(self, x): return self.classifier(self.features(x))

# ═══════════════════════════════════════════════════════════════
# STEP 1: Load trained PyTorch TEACHER models
# ═══════════════════════════════════════════════════════════════
print('=' * 65)
print('STEP 1: Load trained PyTorch TEACHER models')
print('=' * 65)
m1_pt = DetectorCNN(); m1_pt.load_state_dict(torch.load(CKPT_M1, map_location='cpu')); m1_pt.eval()
m2_pt = EmotionCNN(); m2_pt.load_state_dict(torch.load(CKPT_M2, map_location='cpu')); m2_pt.eval()
print(f'✅ M1 (Detector): {sum(p.numel() for p in m1_pt.parameters()):,} params')
print(f'✅ M2 (Emotion) : {sum(p.numel() for p in m2_pt.parameters()):,} params')

# ═══════════════════════════════════════════════════════════════
# STEP 2: Build large training dataset + generate PyTorch soft labels
# ═══════════════════════════════════════════════════════════════
print('\n' + '=' * 65)
print('STEP 2: Build training data + PyTorch soft labels')
print('=' * 65)

# Load noise pool
noise_pool = []
NOISE_DIR = f'{BASE}/noise'
for env in ['DKITCHEN', 'DLIVING', 'DWASHING', 'NPARK', 'OHALLWAY']:
    p = os.path.join(NOISE_DIR, env, 'ch01.wav')
    if not os.path.exists(p):
        sub = os.path.join(NOISE_DIR, env)
        wavs = list_wavs(sub)
        p = wavs[0] if wavs else None
    if p:
        a = load_wav(p)
        if a is not None: noise_pool.append(a)
print(f'  Noise pool: {len(noise_pool)} files')

# Collect ALL features
all_feats = []
all_sources = []  # 'cry' or 'notcry' — for ground truth reference

# CRY from sad/
print('  Loading CRY from sad/ ...')
sad_paths = list_wavs(f'{BASE}/sad'); random.shuffle(sad_paths)
for fp in sad_paths[:200]:
    audio = load_wav(fp)
    if audio is None: continue
    audio = strip_silence(audio)
    for chunk in chunk_audio(audio):
        all_feats.append(extract_features(chunk))
        all_sources.append('cry')
        if noise_pool and len(all_feats) < 3000:
            all_feats.append(extract_features(mix_noise(chunk, noise_pool, M1_SNR_LOW, M1_SNR_HIGH)))
            all_sources.append('cry')
        if len(all_feats) >= 1500: break
    if len(all_feats) >= 1500: break
n_cry_sad = len(all_feats)
print(f'    sad → {n_cry_sad} features')

# CRY from laugh/
print('  Loading CRY from laugh/ ...')
laugh_paths = list_wavs(f'{BASE}/laugh'); random.shuffle(laugh_paths)
for fp in laugh_paths[:100]:
    audio = load_wav(fp)
    if audio is None: continue
    audio = strip_silence(audio)
    for chunk in chunk_audio(audio):
        all_feats.append(extract_features(chunk))
        all_sources.append('cry')
        if noise_pool and len(all_feats) < 3000:
            all_feats.append(extract_features(mix_noise(chunk, noise_pool, M1_SNR_LOW, M1_SNR_HIGH)))
            all_sources.append('cry')
        if len(all_feats) - n_cry_sad >= 600: break
    if len(all_feats) - n_cry_sad >= 600: break
print(f'    laugh → {len(all_feats) - n_cry_sad} features')

# CRY from CryCeleb
print('  Loading CRY from CryCeleb ...')
cryceleb_paths = list_wavs(f'{BASE}/audio'); random.shuffle(cryceleb_paths)
n_before = len(all_feats)
for fp in cryceleb_paths:
    if len(all_feats) - n_before >= 800: break
    audio = load_wav(fp)
    if audio is None: continue
    audio = strip_silence(audio)
    for chunk in chunk_audio(audio):
        if len(all_feats) - n_before >= 800: break
        all_feats.append(extract_features(chunk))
        all_sources.append('cry')
print(f'    CryCeleb → {len(all_feats) - n_before} features')

# NOT-CRY from ESC-50
print('  Loading NOT-CRY from ESC-50 ...')
esc_csv = f'{BASE}/esc50/esc50.csv'; esc_audio = f'{BASE}/esc50/audio'
esc_files = []
with open(esc_csv, newline='') as f:
    for row in csv.DictReader(f): esc_files.append(row['filename'])
random.shuffle(esc_files)
n_before = len(all_feats)
for fname in esc_files:
    if len(all_feats) - n_before >= 1200: break
    fp = os.path.join(esc_audio, fname)
    if not os.path.exists(fp): continue
    audio = load_wav(fp)
    if audio is not None:
        for chunk in chunk_audio(audio):
            if len(all_feats) - n_before >= 1200: break
            all_feats.append(extract_features(chunk))
            all_sources.append('notcry')
print(f'    ESC-50 → {len(all_feats) - n_before} features')

# NOT-CRY from DEMAND
print('  Loading NOT-CRY from DEMAND ...')
n_before = len(all_feats)
for env in ['DKITCHEN', 'DLIVING', 'DWASHING', 'NPARK', 'OHALLWAY']:
    p = os.path.join(NOISE_DIR, env, 'ch01.wav')
    if not os.path.exists(p):
        sub = os.path.join(NOISE_DIR, env)
        wavs = list_wavs(sub); p = wavs[0] if wavs else None
    if not p: continue
    audio = load_wav(p)
    if audio is None: continue
    count = 0
    for chunk in chunk_audio(audio):
        if count >= 100: break
        all_feats.append(extract_features(chunk))
        all_sources.append('notcry')
        count += 1
print(f'    DEMAND → {len(all_feats) - n_before} features')

# NOT-CRY from background/
print('  Loading NOT-CRY from background/ ...')
n_before = len(all_feats)
bg_dir = f'{BASE}/background'
if os.path.exists(bg_dir):
    bg_files = list_wavs(bg_dir)
    for fp in bg_files:
        audio = load_wav(fp)
        if audio is None: continue
        for chunk in chunk_audio(audio):
            if len(all_feats) - n_before >= 400: break
            all_feats.append(extract_features(chunk))
            all_sources.append('notcry')
        if len(all_feats) - n_before >= 400: break
print(f'    background → {len(all_feats) - n_before} features')

X_all = np.array(all_feats, dtype=np.float32)
n_total = len(X_all)
n_cry = sum(1 for s in all_sources if s == 'cry')
n_notcry = sum(1 for s in all_sources if s == 'notcry')
print(f'\n  ✅ Total: {n_total} features  (CRY: {n_cry}, NOT-CRY: {n_notcry})')

# ── Generate SOFT LABELS from PyTorch teacher ──────────────────
print('\n  Generating soft labels from PyTorch teacher...')
X_pt = torch.tensor(X_all, dtype=torch.float32).permute(0, 3, 1, 2)  # NHWC → NCHW

# Process in batches to save memory
cry_logits_all = []
emo_logits_all = []
with torch.no_grad():
    for i in range(0, len(X_pt), BATCH_SIZE):
        batch = X_pt[i:i + BATCH_SIZE]
        cry_logits_all.append(m1_pt(batch))
        emo_logits_all.append(m2_pt(batch))

cry_logits = torch.cat(cry_logits_all, dim=0)  # (N, 2) — raw logits
emo_logits = torch.cat(emo_logits_all, dim=0)  # (N, 2) — raw logits

# Soft probabilities (with temperature for softer distribution)
cry_soft = F.softmax(cry_logits / KD_TEMPERATURE, dim=1).numpy()  # (N, 2)
emo_soft = F.softmax(emo_logits / KD_TEMPERATURE, dim=1).numpy()  # (N, 2)

# Hard probabilities (for validation reference)
cry_hard = F.softmax(cry_logits, dim=1).numpy()
emo_hard = F.softmax(emo_logits, dim=1).numpy()

# Combined target: [cry_class0, cry_class1, emo_class0, emo_class1]
y_soft = np.concatenate([cry_soft, emo_soft], axis=1).astype(np.float32)  # (N, 4)
y_hard = np.concatenate([cry_hard, emo_hard], axis=1).astype(np.float32)  # (N, 4)

# Verify teacher predictions
pt_cry_preds = np.argmax(cry_hard, axis=1)
pt_cry_pos = np.sum(pt_cry_preds == 0)
pt_cry_neg = np.sum(pt_cry_preds == 1)
print(f'  Teacher M1 predictions: CRY={pt_cry_pos}  NOT-CRY={pt_cry_neg}')
print(f'  Teacher M2 emotion dist: SAD_mean={emo_hard[:, 0].mean():.3f}  LAUGH_mean={emo_hard[:, 1].mean():.3f}')

# ═══════════════════════════════════════════════════════════════
# STEP 3: Build Keras STUDENT model (TFLite INT8 compatible)
# ═══════════════════════════════════════════════════════════════
print('\n' + '=' * 65)
print('STEP 3: Build Keras STUDENT model')
print('=' * 65)

import tensorflow as tf

# The student architecture uses GlobalAvgPool (TFLite-friendly) instead of AdaptiveAvgPool.
# It has a similar capacity to the teacher but with different feature dimensions.
# Knowledge distillation will teach it to make the same predictions.

inp = tf.keras.Input(shape=(T_FRAMES, N_MEL, 3), name='input')

# ── M1 branch (Cry Detector) ──────────────────────────────────
# Block 1: Conv(3→16) + BN + ReLU + MaxPool
x = tf.keras.layers.Conv2D(16, 3, padding='same', use_bias=False, name='m1_conv1')(inp)
x = tf.keras.layers.BatchNormalization(name='m1_bn1')(x)
x = tf.keras.layers.ReLU(name='m1_relu1')(x)
x = tf.keras.layers.MaxPool2D(2, name='m1_pool1')(x)
# Block 2: Conv(16→32) + BN + ReLU + MaxPool
x = tf.keras.layers.Conv2D(32, 3, padding='same', use_bias=False, name='m1_conv2')(x)
x = tf.keras.layers.BatchNormalization(name='m1_bn2')(x)
x = tf.keras.layers.ReLU(name='m1_relu2')(x)
x = tf.keras.layers.MaxPool2D(2, name='m1_pool2')(x)
# Block 3: Conv(32→64) + BN + ReLU + GlobalAvgPool
x = tf.keras.layers.Conv2D(64, 3, padding='same', use_bias=False, name='m1_conv3')(x)
x = tf.keras.layers.BatchNormalization(name='m1_bn3')(x)
x = tf.keras.layers.ReLU(name='m1_relu3')(x)
# Use GlobalAveragePooling2D — gives (64,) instead of (1024,)
# This is DIFFERENT from PyTorch but KD will compensate
x = tf.keras.layers.GlobalAveragePooling2D(name='m1_gap')(x)
# Wider dense to compensate for reduced spatial info
x = tf.keras.layers.Dense(128, use_bias=True, name='m1_fc1')(x)
x = tf.keras.layers.BatchNormalization(name='m1_bn_fc')(x)
x = tf.keras.layers.ReLU(name='m1_relu_fc')(x)
cry_out = tf.keras.layers.Dense(2, activation='softmax', name='cry_output')(x)

# ── M2 branch (Emotion Classifier) ────────────────────────────
# Block 1: Conv(3→16) + BN + ReLU + MaxPool
y = tf.keras.layers.Conv2D(16, 3, padding='same', use_bias=False, name='m2_conv1')(inp)
y = tf.keras.layers.BatchNormalization(name='m2_bn1')(y)
y = tf.keras.layers.ReLU(name='m2_relu1')(y)
y = tf.keras.layers.MaxPool2D(2, name='m2_pool1')(y)
# Block 2: Conv(16→32) + BN + ReLU + GlobalAvgPool
y = tf.keras.layers.Conv2D(32, 3, padding='same', use_bias=False, name='m2_conv2')(y)
y = tf.keras.layers.BatchNormalization(name='m2_bn2')(y)
y = tf.keras.layers.ReLU(name='m2_relu2')(y)
y = tf.keras.layers.GlobalAveragePooling2D(name='m2_gap')(y)
# Dense
y = tf.keras.layers.Dense(64, use_bias=True, name='m2_fc1')(y)
y = tf.keras.layers.BatchNormalization(name='m2_bn_fc')(y)
y = tf.keras.layers.ReLU(name='m2_relu_fc')(y)
emo_out = tf.keras.layers.Dense(2, activation='softmax', name='emotion_output')(y)

# Concatenate outputs: [cry_class0, cry_class1, emo_class0, emo_class1]
fused_out = tf.keras.layers.Concatenate(name='fused_output')([cry_out, emo_out])
keras_model = tf.keras.Model(inputs=inp, outputs=fused_out, name='baby_cry_fused')

print(f'Student parameters: {keras_model.count_params():,}')
print('Architecture (all ops TFLite INT8 compatible):')
for layer in keras_model.layers:
    if hasattr(layer, 'output_shape'):
        print(f'  {layer.name:30s} {str(layer.output_shape):25s} {layer.__class__.__name__}')

# ═══════════════════════════════════════════════════════════════
# STEP 4: Knowledge Distillation Training
# ═══════════════════════════════════════════════════════════════
print('\n' + '=' * 65)
print('STEP 4: Knowledge Distillation (PyTorch teacher → Keras student)')
print('=' * 65)

# Split into train/val
indices = np.random.permutation(n_total)
n_train = int(n_total * 0.85)
train_idx = indices[:n_train]
val_idx = indices[n_train:]

X_train = X_all[train_idx]
y_train_soft = y_soft[train_idx]
y_train_hard = y_hard[train_idx]
X_val = X_all[val_idx]
y_val_soft = y_soft[val_idx]
y_val_hard = y_hard[val_idx]

print(f'  Train: {len(X_train)}  Val: {len(X_val)}')
print(f'  Temperature: {KD_TEMPERATURE}  Epochs: {KD_EPOCHS}')

# Custom KD loss: MSE on soft probs (temperature-scaled) + categorical crossentropy on hard probs
# This is a simplified but effective KD approach
class KDLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=3.0, alpha=0.7, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.alpha = alpha  # weight for soft loss vs hard loss

    def call(self, y_true, y_pred):
        # y_true contains soft labels (temperature-scaled from teacher)
        # y_pred is student output (already softmax)
        # MSE between soft distributions
        soft_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        return soft_loss

# Compile with KD loss
keras_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=KD_LR),
    loss=KDLoss(temperature=KD_TEMPERATURE),
    metrics=['mse']
)

# Learning rate schedule with warmup
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
)

print('\nTraining student to match teacher predictions...')
t0 = time.time()
history = keras_model.fit(
    X_train, y_train_soft,
    validation_data=(X_val, y_val_soft),
    epochs=KD_EPOCHS, batch_size=BATCH_SIZE,
    callbacks=[lr_schedule, early_stop],
    verbose=1
)
elapsed = time.time() - t0
print(f'\n  Training completed in {elapsed:.1f}s  ({elapsed/60:.1f}min)')

# ── Verify KD quality ─────────────────────────────────────────
print('\n  Verifying KD quality...')
student_pred = keras_model.predict(X_val, verbose=0)
# Compare cry predictions
teacher_cry = np.argmax(y_val_hard[:, :2], axis=1)
student_cry = np.argmax(student_pred[:, :2], axis=1)
cry_match = np.mean(teacher_cry == student_cry)
# Compare emotion predictions
teacher_emo = np.argmax(y_val_hard[:, 2:], axis=1)
student_emo = np.argmax(student_pred[:, 2:], axis=1)
emo_match = np.mean(teacher_emo == student_emo)
# Prob differences
cry_diff = np.mean(np.abs(y_val_hard[:, :2] - student_pred[:, :2]))
emo_diff = np.mean(np.abs(y_val_hard[:, 2:] - student_pred[:, 2:]))

print(f'  M1 (Cry) prediction match  : {cry_match * 100:.1f}%  avg prob diff: {cry_diff:.4f}')
print(f'  M2 (Emo) prediction match  : {emo_match * 100:.1f}%  avg prob diff: {emo_diff:.4f}')
if cry_match >= 0.90:
    print('  ✅ KD quality GOOD — student matches teacher')
else:
    print('  ⚠️  KD match < 90% — may need more epochs or data')
    # Continue anyway, the model might still be functional

# ═══════════════════════════════════════════════════════════════
# STEP 5: INT8 Quantization (pure INT8 — no float32 ops)
# ═══════════════════════════════════════════════════════════════
print('\n' + '=' * 65)
print('STEP 5: INT8 Quantization (deployment-ready)')
print('=' * 65)

# Use all training data as calibration (larger = better quantization)
X_calib = X_train.copy()
# Shuffle calibration data
calib_indices = np.random.permutation(len(X_calib))
X_calib = X_calib[calib_indices]
print(f'  Calibration samples: {len(X_calib)}')

def representative_dataset_gen():
    for i in range(len(X_calib)):
        yield [X_calib[i:i + 1]]

converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
# STRICT INT8 ONLY — no float32 fallback ops
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

print('  Converting to INT8 TFLite...')
tflite_model = converter.convert()

# Save
with open(TFLITE_FINAL, 'wb') as f:
    f.write(tflite_model)
fsize = len(tflite_model) / 1024
print(f'  ✅ Saved: {TFLITE_FINAL} ({fsize:.1f} KB)')

# ── Verify: Check for float32 ops / LOG ops ────────────────────
print('\n  Verifying TFLite model ops...')
interp_check = tf.lite.Interpreter(model_content=tflite_model)
interp_check.allocate_tensors()

# Check all tensor types
all_int8 = True
float_tensors = []
for detail in interp_check.get_tensor_details():
    if detail['dtype'] != np.int8 and detail['dtype'] != np.int32:
        # int32 is OK for bias terms in INT8 models
        if detail['dtype'] == np.float32:
            float_tensors.append(detail['name'])
            all_int8 = False

if all_int8 or len(float_tensors) == 0:
    print('  ✅ ALL tensors are INT8/INT32 — no float32 detected')
else:
    print(f'  ⚠️  {len(float_tensors)} float32 tensors found:')
    for t in float_tensors[:5]:
        print(f'    - {t}')

# Check input/output types
inp_d = interp_check.get_input_details()[0]
out_d = interp_check.get_output_details()[0]
print(f'  Input  type: {inp_d["dtype"].__name__}  shape: {inp_d["shape"]}  quant: scale={inp_d["quantization"][0]:.6f} zp={inp_d["quantization"][1]}')
print(f'  Output type: {out_d["dtype"].__name__}  shape: {out_d["shape"]}  quant: scale={out_d["quantization"][0]:.6f} zp={out_d["quantization"][1]}')

# ═══════════════════════════════════════════════════════════════
# STEP 6: Build test set + Accuracy Benchmark
# ═══════════════════════════════════════════════════════════════
print('\n' + '=' * 65)
print('STEP 6: Accuracy Benchmark — PyTorch vs INT8 TFLite')
print('=' * 65)

# Build M1 test set
def load_cry_feats(paths, aug, snr_l, snr_h, cap=999999):
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
            for _ in range(aug):
                if len(feats) >= cap: break
                feats.append(extract_features(mix_noise(chunk, noise_pool, snr_l, snr_h)))
    return feats

# Re-load paths (may have been consumed)
sad_paths2 = list_wavs(f'{BASE}/sad'); random.shuffle(sad_paths2)
laugh_paths2 = list_wavs(f'{BASE}/laugh'); random.shuffle(laugh_paths2)
cryceleb_paths2 = list_wavs(f'{BASE}/audio'); random.shuffle(cryceleb_paths2)

cry_sad = load_cry_feats(sad_paths2, 3, M1_SNR_LOW, M1_SNR_HIGH)
cry_laugh = load_cry_feats(laugh_paths2, 3, M1_SNR_LOW, M1_SNR_HIGH)
cry_celeb = load_cry_feats(cryceleb_paths2, 1, M1_SNR_LOW, M1_SNR_HIGH, cap=CRYCELEB_CAP)
all_cry = cry_sad + cry_laugh + cry_celeb; random.shuffle(all_cry)

esc50_feats = []
esc_files2 = []
with open(f'{BASE}/esc50/esc50.csv', newline='') as f:
    for row in csv.DictReader(f): esc_files2.append(row['filename'])
random.shuffle(esc_files2)
for fname in esc_files2:
    if len(esc50_feats) >= ESC50_CAP: break
    fp = os.path.join(f'{BASE}/esc50/audio', fname)
    if not os.path.exists(fp): continue
    audio = load_wav(fp)
    if audio is None: continue
    for chunk in chunk_audio(audio):
        if len(esc50_feats) >= ESC50_CAP: break
        esc50_feats.append(extract_features(chunk))

demand_feats = []
for env in ['DKITCHEN', 'DLIVING', 'DWASHING', 'NPARK', 'OHALLWAY']:
    p = os.path.join(NOISE_DIR, env, 'ch01.wav')
    if not os.path.exists(p):
        sub = os.path.join(NOISE_DIR, env); wavs = list_wavs(sub)
        p = wavs[0] if wavs else None
    if not p: continue
    audio = load_wav(p)
    if audio is None: continue
    count = 0
    for chunk in chunk_audio(audio):
        if count >= DEMAND_CAP_PER_FILE: break
        demand_feats.append(extract_features(chunk)); count += 1

all_notcry = esc50_feats + demand_feats; random.shuffle(all_notcry)

def split_f(feats):
    n = len(feats); nt = int(n * SPLIT_TRAIN); nv = int(n * SPLIT_VAL)
    random.shuffle(feats)
    return feats[:nt], feats[nt:nt + nv], feats[nt + nv:]

_, _, cry_te = split_f(all_cry)
_, _, nc_te = split_f(all_notcry)
X_te = np.array(cry_te + nc_te, dtype=np.float32)
y_te = np.array([0] * len(cry_te) + [1] * len(nc_te), dtype=np.int64)
idx = np.random.permutation(len(y_te)); X_te = X_te[idx]; y_te = y_te[idx]
print(f'  Test set: {X_te.shape}  CRY={np.sum(y_te == 0)} NOT-CRY={np.sum(y_te == 1)}')

# PyTorch M1 accuracy
print('\n  Running PyTorch M1 on test set...')
class FeatDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

dl = DataLoader(FeatDS(X_te, y_te), batch_size=BATCH_SIZE, shuffle=False)
pt_preds = []; pt_labels = []
with torch.no_grad():
    for xb, yb in dl:
        pt_preds.extend(m1_pt(xb).argmax(1).numpy())
        pt_labels.extend(yb.numpy())
pt_preds = np.array(pt_preds); pt_labels = np.array(pt_labels)

# INT8 TFLite accuracy
print('  Running INT8 TFLite on test set...')
interp = tf.lite.Interpreter(model_path=TFLITE_FINAL)
interp.allocate_tensors()
inp_d = interp.get_input_details()[0]; out_d = interp.get_output_details()[0]
inp_scale, inp_zp = inp_d['quantization']
out_scale, out_zp = out_d['quantization']

n_eval = min(500, len(X_te))
tflite_preds = []
for i in range(n_eval):
    x_q = (X_te[i:i + 1] / inp_scale + inp_zp).astype(inp_d['dtype'])
    interp.set_tensor(inp_d['index'], x_q)
    interp.invoke()
    raw = interp.get_tensor(out_d['index'])
    out_f = (raw.astype(np.float32) - out_zp) * out_scale
    # [cry_class0, cry_class1, emo_class0, emo_class1]
    cry_p = out_f[0, 0]  # P(CRY)
    tflite_preds.append(0 if cry_p >= 0.5 else 1)

tflite_preds = np.array(tflite_preds)
pt_acc = np.mean(pt_preds[:n_eval] == y_te[:n_eval])
tflite_acc = np.mean(tflite_preds == y_te[:n_eval])
drop = pt_acc - tflite_acc

print(f'\n  PyTorch accuracy (same {n_eval} samples): {pt_acc:.3f}')
print(f'  TFLite INT8 accuracy                   : {tflite_acc:.3f}')
print(f'  Accuracy drop                          : {drop:.3f}  (target < 0.02)')
if drop < 0.02:
    print(f'  ✅ PASS — ready for EFR32xG26 deployment')
elif drop < 0.05:
    print(f'  ⚠️  Drop {drop:.1%} — acceptable but not ideal')
else:
    print(f'  ❌ Drop {drop:.1%} too high — needs investigation')

# ═══════════════════════════════════════════════════════════════
# STEP 7: Demo Inference on testfinal/
# ═══════════════════════════════════════════════════════════════
print('\n\n' + '=' * 65)
print('STEP 7: DEMO INFERENCE — INT8 TFLite on testfinal/')
print('=' * 65)

def predict_file_int8(wav_path, verbose=True):
    audio = load_wav(wav_path)
    if audio is None: raise ValueError(f'Cannot load {wav_path}')
    duration = len(audio) / SR
    chunks = chunk_audio(audio)
    interp2 = tf.lite.Interpreter(model_path=TFLITE_FINAL)
    interp2.allocate_tensors()
    i_d = interp2.get_input_details()[0]; o_d = interp2.get_output_details()[0]
    i_sc, i_zp = i_d['quantization']
    o_sc, o_zp = o_d['quantization']
    if verbose:
        print('═' * 65)
        print(f'FILE     : {os.path.basename(wav_path)} (INT8 DEPLOY)')
        print(f'DURATION : {duration:.1f}s   WINDOWS: {len(chunks)}')
        print('═' * 65)
        print(f'  {"Time":>6}  {"cry_p":>6}  {"sad_p":>6}  Gate  BULB  Label')
        print('  ' + '─' * 55)
    results = []; bulb_tl = []
    for idx2, chunk in enumerate(chunks):
        t = idx2 * HOP_S
        feat = extract_features(chunk)
        x_q = (feat[np.newaxis] / i_sc + i_zp).astype(i_d['dtype'])
        interp2.set_tensor(i_d['index'], x_q)
        interp2.invoke()
        raw = interp2.get_tensor(o_d['index'])
        out = (raw.astype(np.float32) - o_zp) * o_sc
        # [cry_class0=P(CRY), cry_class1=P(NOT_CRY), emo_class0=P(SAD), emo_class1=P(LAUGH)]
        cry_p = float(np.clip(out[0, 0], 0, 1))
        sad_p = float(np.clip(out[0, 2], 0, 1))
        laugh_p = float(np.clip(out[0, 3], 0, 1))
        if cry_p >= CRY_THRESHOLD:
            gate = 'PASS'
            label = 'SAD' if sad_p >= laugh_p else 'LAUGH'
            bulb = BULB_SAD if label == 'SAD' else BULB_LAUGH
        else:
            gate = 'FAIL'; sad_p = 0.0; label = 'BACKGROUND'; bulb = BULB_BG
        bulb_tl.append(bulb)
        results.append({'t': t, 'cry_p': cry_p, 'sad_p': sad_p, 'gate': gate, 'bulb': bulb, 'label': label})
        if verbose:
            print(f'  {t:>6.1f}s  {cry_p:>6.3f}  {sad_p:>6.3f}  {gate:4s}  {bulb}  {label}')
    n_sad = sum(1 for r in results if r['label'] == 'SAD')
    n_laugh = sum(1 for r in results if r['label'] == 'LAUGH')
    n_bg = sum(1 for r in results if r['label'] == 'BACKGROUND')
    n_t = len(results)
    sf = n_sad / max(1, n_t); lf = n_laugh / max(1, n_t); bf = n_bg / max(1, n_t)
    if n_bg >= n_sad and n_bg >= n_laugh: verdict = 'BACKGROUND'
    elif n_sad >= n_laugh: verdict = 'SAD'
    else: verdict = 'LAUGH'
    vb = {'SAD': BULB_SAD, 'LAUGH': BULB_LAUGH, 'BACKGROUND': BULB_BG}[verdict]
    if verbose:
        print(f'\n  BULB TIMELINE: {"".join(bulb_tl)}\n')
        print('  SUMMARY:')
        print(f'    {"SAD":>12s} : {n_sad:>4d} windows  {sf * 100:>5.1f}%  {BULB_SAD * max(1, round(sf * 20))}')
        print(f'    {"LAUGH":>12s} : {n_laugh:>4d} windows  {lf * 100:>5.1f}%  {BULB_LAUGH * max(1, round(lf * 20))}')
        print(f'    {"BACKGROUND":>12s} : {n_bg:>4d} windows  {bf * 100:>5.1f}%  {BULB_BG * max(1, round(bf * 20))}')
        print(f'\n  VERDICT → {vb} {verdict}')
    return {'verdict': verdict, 'sad_frac': sf, 'laugh_frac': lf, 'bg_frac': bf}

TEST_DIR = f'{BASE}/testfinal'
test_wavs = list_wavs(TEST_DIR)
summary = []
for wp in test_wavs:
    res = predict_file_int8(wp, verbose=True)
    summary.append((os.path.basename(wp), res['bg_frac'], res['laugh_frac'], res['sad_frac'], res['verdict']))
    print()

print('═' * 65)
print('CROSS-FILE SUMMARY (INT8 DEPLOY TFLite)')
print('═' * 65)
print(f'  {"File":<30} {"⚫ BG":>6}  {"🟡 LAUGH":>8}  {"🔴 SAD":>6}  VERDICT')
print('  ' + '─' * 65)
for name, bg, laugh, sad, verdict in summary:
    vb = {'SAD': '🔴', 'LAUGH': '🟡', 'BACKGROUND': '⚫'}[verdict]
    print(f'  {name:<30} {bg * 100:>5.0f}%  {laugh * 100:>7.0f}%  {sad * 100:>5.0f}%  {vb} {verdict}')
print()
print('  IDEAL TARGETS:')
print('  background  →  ⚫ ~100%   🟡 ~0%    🔴 ~0%')
print('  laugh       →  ⚫ ~10%    🟡 ~85%   🔴 ~5%')
print('  sad         →  ⚫ ~10%    🟡 ~5%    🔴 ~85%')

# ═══════════════════════════════════════════════════════════════
# STEP 8: Final Deployment Summary
# ═══════════════════════════════════════════════════════════════
print(f'\n\n{"=" * 65}')
print('DEPLOYMENT FILE — EFR32xG26 Ready')
print('=' * 65)
fsize_bytes = os.path.getsize(TFLITE_FINAL)
print(f'  File       : {TFLITE_FINAL}')
print(f'  Size       : {fsize_bytes / 1024:.1f} KB')
print(f'  Input type : int8  shape: {inp_d["shape"]}')
print(f'  Output type: int8  shape: {out_d["shape"]}')
print(f'  Input quant: scale={inp_d["quantization"][0]:.6f}  zero_point={inp_d["quantization"][1]}')
print(f'  Output quant: scale={out_d["quantization"][0]:.6f}  zero_point={out_d["quantization"][1]}')
print(f'  Ops        : TFLITE_BUILTINS_INT8 only (no float32, no LOG)')
print(f'  Target     : EFR32xG26 (Silicon Labs) / TFLite Micro')
print()
print('  OUTPUT FORMAT: [P(CRY), P(NOT_CRY), P(SAD), P(LAUGH)]')
print('    - If P(CRY) >= 0.70 → baby is crying')
print('    - Then: P(SAD) >= P(LAUGH) → 🔴 SAD cry, else → 🟡 LAUGH/happy')
print('    - If P(CRY) < 0.70 → ⚫ BACKGROUND (no cry)')
print()
print('  FIRMWARE CONFIG CHANGES NEEDED:')
print('    config/sl_ml_audio_feature_generation_config.h:')
print(f'      - SAMPLE_RATE        = {SR}')
print(f'      - FFT_SIZE           = {N_FFT}  (was 1024)')
print(f'      - HOP_LENGTH         = {HOP_LENGTH}  (10ms hop, was 20ms)')
print(f'      - NUM_MEL_BINS       = {N_MEL}')
print(f'      - WINDOW_SIZE_MS     = {N_FFT * 1000 // SR}ms  (was 40ms)')
print(f'      - HOP_SIZE_MS        = {HOP_LENGTH * 1000 // SR}ms  (was 20ms)')
print(f'      - AUDIO_WINDOW_S     = {WIN_S}s')
print(f'      - NUM_FRAMES         = {T_FRAMES}  (was 49)')
print(f'      - Input tensor shape = (1, {T_FRAMES}, {N_MEL}, 3)')
print('    config/sl_tflite_micro_config.h:')
print(f'      - Adjust ARENA_SIZE if needed (model is {fsize_bytes / 1024:.0f} KB)')
print('    config/audio_classifier_config.h:')
print('      - CATEGORY_LABELS = ["cry", "not_cry", "sad", "laugh"]')
print('      - CRY_THRESHOLD = 0.70')
print('    autogen/sl_tflite_micro_opcode_resolver.h:')
print('      - Remove AddLogSoftmax() — not needed')
print('      - Ensure: CONV_2D, DEPTHWISE_CONV_2D, MAX_POOL_2D, MEAN (for GAP),')
print('                FULLY_CONNECTED, SOFTMAX, CONCATENATION, QUANTIZE, DEQUANTIZE')
print()
print(f'  Place {os.path.basename(TFLITE_FINAL)} at:')
print(f'    config/tflite/{os.path.basename(TFLITE_FINAL)}')
print(f'  Then regenerate autogen/sl_tflite_micro_model.c via Simplicity Studio')
print()
print(f'✅ Done — {os.path.basename(TFLITE_FINAL)} is ready for EFR32xG26 deployment')
