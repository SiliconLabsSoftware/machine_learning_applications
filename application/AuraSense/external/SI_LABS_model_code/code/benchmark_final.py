"""
FINAL BENCHMARK — baby_cry_int8_DEPLOY.tflite
Runs both:
  1. testfinal/ demo (background.wav, laugh.wav, sad.wav)
  2. PyTorch vs INT8 accuracy on 500 M1 test samples
"""
import os, sys, numpy as np, warnings, random, csv, gc
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa
import tensorflow as tf

# ── Config ─────────────────────────────────────────────────────
SR = 16000; WIN_S = 0.75; HOP_S = 0.375
WIN_SAMPLES = int(SR * WIN_S); HOP_SAMPLES = int(SR * HOP_S)
N_MEL = 40; N_FFT = 512; HOP_LENGTH = 160
T_FRAMES = WIN_SAMPLES // HOP_LENGTH
SILENCE_TOP_DB = 30; MIN_AUDIO_S = 0.3; CRY_THRESHOLD = 0.70
BATCH_SIZE = 32; SPLIT_TRAIN = 0.80; SPLIT_VAL = 0.10
M1_SNR_LOW = 5.0; M1_SNR_HIGH = 20.0
CRYCELEB_CAP = 2000; ESC50_CAP = 2000; DEMAND_CAP_PER_FILE = 400

BASE = '/Users/rishabhsahay/Desktop/hola/datasets1'
MODELS_DIR = 'outputs/models'; TFLITE_DIR = 'outputs/tflite'
CKPT_M1 = os.path.join(MODELS_DIR, 'model_m1_detector.pt')
CKPT_M2 = os.path.join(MODELS_DIR, 'model_m2_emotion.pt')
TFLITE_DEPLOY = os.path.join(TFLITE_DIR, 'baby_cry_int8_DEPLOY.tflite')
DEVICE = torch.device('cpu')
BULB_SAD = '🔴'; BULB_LAUGH = '🟡'; BULB_BG = '⚫'

# ── Audio utilities ────────────────────────────────────────────
def extract_features(audio):
    if len(audio) < WIN_SAMPLES: audio = np.pad(audio, (0, WIN_SAMPLES - len(audio)))
    else: audio = audio[:WIN_SAMPLES]
    mel = librosa.feature.melspectrogram(y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
                                         n_mels=N_MEL, fmin=60, fmax=8000)
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=40)
    log_mel = (log_mel + 40.0) / 40.0; T = log_mel.shape[1]
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
    else: feat = feat[:T_FRAMES]
    return feat.astype(np.float32)

def load_wav(path, sr=SR):
    try: audio, _ = librosa.load(path, sr=sr, mono=True); return audio
    except: return None

def chunk_audio(audio, win=WIN_SAMPLES, hop=HOP_SAMPLES):
    chunks = []; start = 0
    while start + win <= len(audio):
        chunks.append(audio[start:start + win]); start += hop
    if start < len(audio) and (len(audio) - start) >= int(SR * 0.5):
        chunk = audio[start:]; chunk = np.pad(chunk, (0, win - len(chunk))); chunks.append(chunk)
    return chunks

def strip_silence(audio, sr=SR, top_db=SILENCE_TOP_DB):
    intervals = librosa.effects.split(audio, top_db=top_db)
    if len(intervals) == 0: return audio
    return np.concatenate([audio[s:e] for s, e in intervals])

def mix_noise(signal, noise_pool, snr_low, snr_high):
    if not noise_pool: return signal
    noise = random.choice(noise_pool)
    if len(noise) > WIN_SAMPLES:
        start = random.randint(0, len(noise) - WIN_SAMPLES); ns = noise[start:start + WIN_SAMPLES]
    else: ns = np.pad(noise, (0, WIN_SAMPLES - len(noise)))
    snr = random.uniform(snr_low, snr_high)
    sr_ = np.sqrt(np.mean(signal ** 2) + 1e-9); nr = np.sqrt(np.mean(ns ** 2) + 1e-9)
    ns = ns * (sr_ / (10 ** (snr / 20.0)) / nr); m = signal + ns
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
            nn.Flatten(), nn.Linear(64*4*4, 128), nn.BatchNorm1d(128),
            nn.ReLU(inplace=True), nn.Dropout(0.4), nn.Linear(128, 2))
    def forward(self, x): return self.classifier(self.features(x))

class EmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((4, 4)))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(32*4*4, 64), nn.BatchNorm1d(64),
            nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(64, 2))
    def forward(self, x): return self.classifier(self.features(x))

class FeatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

# ── Load models ────────────────────────────────────────────────
print('Loading trained PyTorch models...')
m1 = DetectorCNN(); m1.load_state_dict(torch.load(CKPT_M1, map_location='cpu')); m1.eval()
m2 = EmotionCNN(); m2.load_state_dict(torch.load(CKPT_M2, map_location='cpu')); m2.eval()
print(f'✅ M1 loaded from {CKPT_M1}')
print(f'✅ M2 loaded from {CKPT_M2}')

# ════════════════════════════════════════════════════════════════
# PART 1: PyTorch Demo on testfinal/
# ════════════════════════════════════════════════════════════════
def predict_file_pytorch(wav_path, verbose=True):
    audio = load_wav(wav_path)
    if audio is None: raise ValueError(f'Cannot load {wav_path}')
    duration = len(audio) / SR; chunks = chunk_audio(audio)
    if verbose:
        print('═' * 65)
        print(f'FILE     : {os.path.basename(wav_path)}')
        print(f'DURATION : {duration:.1f}s   WINDOWS: {len(chunks)}')
        print('═' * 65)
        print(f'  {"Time":>6}  {"cry_p":>6}  {"sad_p":>6}  Gate  BULB  Label')
        print('  ' + '─' * 55)
    results = []; bulb_tl = []
    for i, chunk in enumerate(chunks):
        t = i * HOP_S; feat = extract_features(chunk)
        x = torch.tensor(feat).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            probs1 = F.softmax(m1(x), dim=1); cry_p = probs1[0, 0].item()
            if cry_p >= CRY_THRESHOLD:
                probs2 = F.softmax(m2(x), dim=1); sad_p = probs2[0, 0].item()
                label = 'SAD' if sad_p >= 0.5 else 'LAUGH'
                bulb = BULB_SAD if label == 'SAD' else BULB_LAUGH; gate = 'PASS'
            else:
                sad_p = 0.0; label = 'BACKGROUND'; bulb = BULB_BG; gate = 'FAIL'
        bulb_tl.append(bulb)
        results.append({'t': t, 'cry_p': cry_p, 'sad_p': sad_p, 'gate': gate, 'bulb': bulb, 'label': label})
        if verbose: print(f'  {t:>6.1f}s  {cry_p:>6.3f}  {sad_p:>6.3f}  {gate:4s}  {bulb}  {label}')
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

# ════════════════════════════════════════════════════════════════
# PART 2: TFLite INT8 Demo on testfinal/
# ════════════════════════════════════════════════════════════════
def predict_file_tflite(wav_path, verbose=True):
    audio = load_wav(wav_path)
    if audio is None: raise ValueError(f'Cannot load {wav_path}')
    duration = len(audio) / SR; chunks = chunk_audio(audio)
    interp = tf.lite.Interpreter(model_path=TFLITE_DEPLOY)
    interp.allocate_tensors()
    i_d = interp.get_input_details()[0]; o_d = interp.get_output_details()[0]
    i_sc, i_zp = i_d['quantization']; o_sc, o_zp = o_d['quantization']
    if verbose:
        print('═' * 65)
        print(f'FILE     : {os.path.basename(wav_path)} (TFLite INT8 DEPLOY)')
        print(f'DURATION : {duration:.1f}s   WINDOWS: {len(chunks)}')
        print('═' * 65)
        print(f'  {"Time":>6}  {"cry_p":>6}  {"sad_p":>6}  Gate  BULB  Label')
        print('  ' + '─' * 55)
    results = []; bulb_tl = []
    for idx, chunk in enumerate(chunks):
        t = idx * HOP_S; feat = extract_features(chunk)
        x_q = (feat[np.newaxis] / i_sc + i_zp).astype(i_d['dtype'])
        interp.set_tensor(i_d['index'], x_q); interp.invoke()
        raw = interp.get_tensor(o_d['index'])
        out = (raw.astype(np.float32) - o_zp) * o_sc
        cry_p = float(np.clip(out[0, 0], 0, 1))
        sad_p = float(np.clip(out[0, 2], 0, 1))
        laugh_p = float(np.clip(out[0, 3], 0, 1))
        if cry_p >= CRY_THRESHOLD:
            gate = 'PASS'; label = 'SAD' if sad_p >= laugh_p else 'LAUGH'
            bulb = BULB_SAD if label == 'SAD' else BULB_LAUGH
        else:
            gate = 'FAIL'; sad_p = 0.0; label = 'BACKGROUND'; bulb = BULB_BG
        bulb_tl.append(bulb)
        results.append({'t': t, 'cry_p': cry_p, 'sad_p': sad_p, 'gate': gate, 'bulb': bulb, 'label': label})
        if verbose: print(f'  {t:>6.1f}s  {cry_p:>6.3f}  {sad_p:>6.3f}  {gate:4s}  {bulb}  {label}')
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

# ════════════════════════════════════════════════════════════════
# RUN PART 1: PyTorch testfinal/
# ════════════════════════════════════════════════════════════════
TEST_DIR = f'{BASE}/testfinal'
test_wavs = list_wavs(TEST_DIR)

print(f'\n{"=" * 65}')
print(f'PYTORCH DEMO INFERENCE on {len(test_wavs)} test files')
print(f'{"=" * 65}')
pt_summary = []
for wp in test_wavs:
    res = predict_file_pytorch(wp, verbose=True)
    pt_summary.append((os.path.basename(wp), res['bg_frac'], res['laugh_frac'], res['sad_frac'], res['verdict']))
    print()

print('═' * 65)
print('CROSS-FILE SUMMARY (PyTorch)')
print('═' * 65)
print(f'  {"File":<35} {"⚫ BG":>6}  {"🟡 LAUGH":>8}  {"🔴 SAD":>6}  VERDICT')
print('  ' + '─' * 65)
for name, bg, laugh, sad, verdict in pt_summary:
    vb = {'SAD': '🔴', 'LAUGH': '🟡', 'BACKGROUND': '⚫'}[verdict]
    print(f'  {name:<35} {bg*100:>5.0f}%  {laugh*100:>7.0f}%  {sad*100:>5.0f}%  {vb} {verdict}')
print()
print('  IDEAL TARGETS:')
print('  background  →  ⚫ ~100%   🟡 ~0%    🔴 ~0%')
print('  laugh       →  ⚫ ~10%    🟡 ~85%   🔴 ~5%')
print('  sad         →  ⚫ ~10%    🟡 ~5%    🔴 ~85%')

# ════════════════════════════════════════════════════════════════
# RUN PART 2: TFLite INT8 testfinal/
# ════════════════════════════════════════════════════════════════
print(f'\n\n{"=" * 65}')
print(f'TFLITE INT8 DEPLOY — DEMO INFERENCE on {len(test_wavs)} test files')
print(f'{"=" * 65}')
tfl_summary = []
for wp in test_wavs:
    res = predict_file_tflite(wp, verbose=True)
    tfl_summary.append((os.path.basename(wp), res['bg_frac'], res['laugh_frac'], res['sad_frac'], res['verdict']))
    print()

print('═' * 65)
print('CROSS-FILE SUMMARY (TFLite INT8 DEPLOY)')
print('═' * 65)
print(f'  {"File":<35} {"⚫ BG":>6}  {"🟡 LAUGH":>8}  {"🔴 SAD":>6}  VERDICT')
print('  ' + '─' * 65)
for name, bg, laugh, sad, verdict in tfl_summary:
    vb = {'SAD': '🔴', 'LAUGH': '🟡', 'BACKGROUND': '⚫'}[verdict]
    print(f'  {name:<35} {bg*100:>5.0f}%  {laugh*100:>7.0f}%  {sad*100:>5.0f}%  {vb} {verdict}')
print()
print('  IDEAL TARGETS:')
print('  background  →  ⚫ ~100%   🟡 ~0%    🔴 ~0%')
print('  laugh       →  ⚫ ~10%    🟡 ~85%   🔴 ~5%')
print('  sad         →  ⚫ ~10%    🟡 ~5%    🔴 ~85%')

# ════════════════════════════════════════════════════════════════
# PART 3: PyTorch vs INT8 Accuracy on M1 Test Set (500 samples)
# ════════════════════════════════════════════════════════════════
print(f'\n\n{"=" * 65}')
print('INT8 vs PYTORCH ACCURACY COMPARISON (M1 Test Set)')
print('=' * 65)

print('\nRebuilding M1 test set for accuracy comparison...')

# Load noise pool
noise_pool = []
NOISE_DIR = f'{BASE}/noise'
for env in ['DKITCHEN', 'DLIVING', 'DWASHING', 'NPARK', 'OHALLWAY']:
    p = os.path.join(NOISE_DIR, env, 'ch01.wav')
    if not os.path.exists(p):
        sub = os.path.join(NOISE_DIR, env); wavs = list_wavs(sub)
        p = wavs[0] if wavs else None
    if p:
        audio = load_wav(p)
        if audio is not None: noise_pool.append(audio)

# ESC-50 NOT-CRY
esc50_feats = []
with open(f'{BASE}/esc50/esc50.csv', newline='') as f:
    esc_files = [row['filename'] for row in csv.DictReader(f)]
random.shuffle(esc_files)
for fname in esc_files:
    if len(esc50_feats) >= ESC50_CAP: break
    fp = os.path.join(f'{BASE}/esc50/audio', fname)
    if not os.path.exists(fp): continue
    audio = load_wav(fp)
    if audio is None: continue
    for chunk in chunk_audio(audio):
        if len(esc50_feats) >= ESC50_CAP: break
        esc50_feats.append(extract_features(chunk))

# DEMAND NOT-CRY
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

# CRY features
def load_cry(paths, aug, snr_l, snr_h, cap=999999):
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

sad_paths = list_wavs(f'{BASE}/sad'); random.shuffle(sad_paths)
laugh_paths = list_wavs(f'{BASE}/laugh'); random.shuffle(laugh_paths)
cryceleb_paths = list_wavs(f'{BASE}/audio'); random.shuffle(cryceleb_paths)

cry_sad = load_cry(sad_paths, 3, M1_SNR_LOW, M1_SNR_HIGH)
cry_laugh = load_cry(laugh_paths, 3, M1_SNR_LOW, M1_SNR_HIGH)
cry_celeb = load_cry(cryceleb_paths, 1, M1_SNR_LOW, M1_SNR_HIGH, cap=CRYCELEB_CAP)
all_cry = cry_sad + cry_laugh + cry_celeb; random.shuffle(all_cry)

print(f'  CRY pool: {len(all_cry)}  NOT-CRY pool: {len(all_notcry)}')

def split_f(feats):
    n = len(feats); nt = int(n * SPLIT_TRAIN); nv = int(n * SPLIT_VAL)
    random.shuffle(feats)
    return feats[:nt], feats[nt:nt+nv], feats[nt+nv:]

_, _, cry_te = split_f(all_cry)
_, _, nc_te = split_f(all_notcry)
X_te = np.array(cry_te + nc_te, dtype=np.float32)
y_te = np.array([0]*len(cry_te) + [1]*len(nc_te), dtype=np.int64)
idx = np.random.permutation(len(y_te)); X_te = X_te[idx]; y_te = y_te[idx]
print(f'  Test set: {X_te.shape}  CRY={np.sum(y_te==0)}  NOT-CRY={np.sum(y_te==1)}')

# PyTorch accuracy
print('\nRunning PyTorch on M1 test set...')
dl = DataLoader(FeatureDataset(X_te, y_te), batch_size=BATCH_SIZE, shuffle=False)
pt_preds = []; pt_labels = []
with torch.no_grad():
    for xb, yb in dl:
        pt_preds.extend(m1(xb).argmax(1).numpy()); pt_labels.extend(yb.numpy())
pt_preds = np.array(pt_preds); pt_labels = np.array(pt_labels)

# TFLite INT8 accuracy
print('Running INT8 TFLite on M1 test set ...')
interp = tf.lite.Interpreter(model_path=TFLITE_DEPLOY)
interp.allocate_tensors()
inp_d = interp.get_input_details()[0]; out_d = interp.get_output_details()[0]
inp_scale, inp_zp = inp_d['quantization']; out_scale, out_zp = out_d['quantization']

n_eval = min(500, len(X_te))
tflite_preds = []
for i in range(n_eval):
    x_q = (X_te[i:i+1] / inp_scale + inp_zp).astype(inp_d['dtype'])
    interp.set_tensor(inp_d['index'], x_q); interp.invoke()
    raw = interp.get_tensor(out_d['index'])
    out_f = (raw.astype(np.float32) - out_zp) * out_scale
    cry_p = out_f[0, 0]
    tflite_preds.append(0 if cry_p >= 0.5 else 1)

tflite_preds = np.array(tflite_preds)
pytorch_acc = np.mean(pt_preds[:n_eval] == y_te[:n_eval])
tflite_acc = np.mean(tflite_preds == y_te[:n_eval])
drop = pytorch_acc - tflite_acc

print(f'PyTorch accuracy (same {n_eval} samples): {pytorch_acc:.3f}')
print(f'TFLite INT8 accuracy                   : {tflite_acc:.3f}')
print(f'Accuracy drop                          : {drop:.3f}  (target < 0.02)')
if drop < 0.02:
    status = '✅ PASS — drop < 2%, ready for deployment'
elif drop < 0.05:
    status = '⚠️  Drop < 5% — acceptable for IoT deployment'
else:
    status = '❌ Drop > 5% — check quantisation'
print(f'Status: {status}')
print(f'\n✅ TFLite validation complete')
print(f'\n{"=" * 65}')
print('DEPLOYMENT MODEL INFO')
print('=' * 65)
fsize = os.path.getsize(TFLITE_DEPLOY)
print(f'  File  : {TFLITE_DEPLOY}')
print(f'  Size  : {fsize/1024:.1f} KB')
print(f'  Input : {inp_d["dtype"].__name__}  shape={inp_d["shape"]}  scale={inp_scale:.6f}  zp={inp_zp}')
print(f'  Output: {out_d["dtype"].__name__}  shape={out_d["shape"]}  scale={out_scale:.6f}  zp={out_zp}')
print(f'  Ops   : TFLITE_BUILTINS_INT8 only')
print(f'  Target: EFR32xG26 / TFLite Micro')
