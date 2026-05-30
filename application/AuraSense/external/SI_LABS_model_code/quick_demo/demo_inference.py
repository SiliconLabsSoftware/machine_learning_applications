import os
import sys
import argparse
import subprocess

import numpy as np
import warnings
import librosa
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

Interpreter = None

warnings.filterwarnings('ignore')

# ── Config ─────────────────────────────────────────────────────
SR = 16000
WIN_S = 0.75
HOP_S = 0.375
WIN_SAMPLES = int(SR * WIN_S)
HOP_SAMPLES = int(SR * HOP_S)

N_MEL = 40
N_FFT = 512
HOP_LENGTH = 160
T_FRAMES = WIN_SAMPLES // HOP_LENGTH

CRY_THRESHOLD = 0.70

# Emojis requested by user
EMOJI_SAD = '🔵'
EMOJI_BG = '🔴'
EMOJI_HAPPY = '🟡'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'model', 'baby_cry_int8_DEPLOY.tflite')
AUDIO_DIR = os.path.join(SCRIPT_DIR, 'test_audio')


def _tensorflow_safe_to_import():
    probe_code = 'import tensorflow as tf; print(tf.__version__)'
    try:
        result = subprocess.run(
            [sys.executable, '-c', probe_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
            env={**os.environ, 'TF_CPP_MIN_LOG_LEVEL': '3'}
        )
        return result.returncode == 0
    except Exception:
        return False


def _resolve_interpreter(allow_tensorflow_fallback=True):
    try:
        from tflite_runtime.interpreter import Interpreter as RtInterpreter
        return RtInterpreter, 'tflite_runtime'
    except Exception:
        pass

    if not allow_tensorflow_fallback:
        return None, None

    if not _tensorflow_safe_to_import():
        return None, None

    try:
        import tensorflow as tf
        return tf.lite.Interpreter, 'tensorflow'
    except Exception:
        return None, None


def _quantize_input(feat, inp_d):
    dtype = inp_d['dtype']
    scale, zp = inp_d.get('quantization', (0.0, 0))
    x = feat[np.newaxis].astype(np.float32)
    if dtype in (np.int8, np.uint8) and scale not in (0, 0.0, None):
        x = np.round(x / scale + zp)
        if dtype == np.int8:
            x = np.clip(x, -128, 127)
        else:
            x = np.clip(x, 0, 255)
        return x.astype(dtype)
    return x.astype(dtype)


def _dequantize_output(raw, out_d):
    scale, zp = out_d.get('quantization', (0.0, 0))
    if raw.dtype in (np.int8, np.uint8) and scale not in (0, 0.0, None):
        return (raw.astype(np.float32) - zp) * scale
    return raw.astype(np.float32)


def _decode_scores(out):
    vals = out.reshape(-1)
    if vals.size >= 4:
        cry_p = float(np.clip(vals[0], 0.0, 1.0))
        sad_p = float(np.clip(vals[2], 0.0, 1.0))
        laugh_p = float(np.clip(vals[3], 0.0, 1.0))
        return cry_p, sad_p, laugh_p
    if vals.size == 3:
        cry_p = float(np.clip(vals[0], 0.0, 1.0))
        sad_p = float(np.clip(vals[1], 0.0, 1.0))
        laugh_p = float(np.clip(vals[2], 0.0, 1.0))
        return cry_p, sad_p, laugh_p
    if vals.size == 2:
        cry_p = float(np.clip(vals[0], 0.0, 1.0))
        laugh_p = float(np.clip(vals[1], 0.0, 1.0))
        sad_p = float(np.clip(1.0 - laugh_p, 0.0, 1.0))
        return cry_p, sad_p, laugh_p
    raise ValueError(f'Unexpected model output shape: {out.shape}')

# ── Audio Features ────────────────────────────────────────────
def extract_features(audio):
    """Matches the exact feature extraction logic of the firmware."""
    if len(audio) < WIN_SAMPLES:
        audio = np.pad(audio, (0, WIN_SAMPLES - len(audio)))
    else:
        audio = audio[:WIN_SAMPLES]
        
    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MEL, fmin=60, fmax=8000)
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=40)
    log_mel = (log_mel + 40.0) / 40.0
    T = log_mel.shape[1]
    
    # Spectral Flatness
    flatness = librosa.feature.spectral_flatness(y=audio, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    flatness = np.log1p(flatness * 1000.0)
    flatness = np.clip(flatness / (np.max(flatness) + 1e-8), 0, 1)
    flatness_ch = np.tile(flatness[np.newaxis, :T], (N_MEL, 1))
    
    # RMS Energy
    rms = librosa.feature.rms(y=audio, frame_length=N_FFT, hop_length=HOP_LENGTH)[0][:T]
    rms = np.log1p(rms * 100.0)
    rms = np.clip(rms / (np.max(rms) + 1e-8), 0, 1)
    rms_ch = np.tile(rms[np.newaxis, :T], (N_MEL, 1))
    
    # Stack 3 Channels
    feat = np.stack([log_mel[:, :T], flatness_ch[:, :T], rms_ch[:, :T]], axis=-1).transpose(1, 0, 2)
    
    # Pad to T_FRAMES if needed
    if feat.shape[0] < T_FRAMES:
        feat = np.concatenate([feat, np.zeros((T_FRAMES - feat.shape[0], N_MEL, 3), dtype=np.float32)])
    else:
        feat = feat[:T_FRAMES]
        
    return feat.astype(np.float32)

def chunk_audio(audio, win=WIN_SAMPLES, hop=HOP_SAMPLES):
    """Breaks audio into overlapping inference windows."""
    chunks = []
    start = 0
    while start + win <= len(audio):
        chunks.append(audio[start:start + win])
        start += hop
    # Get remaining tail if it's long enough
    if start < len(audio) and (len(audio) - start) >= int(SR * 0.5):
        chunk = audio[start:]
        chunk = np.pad(chunk, (0, win - len(chunk)))
        chunks.append(chunk)
    # If the whole file is extremely short
    if not chunks and len(audio) > 0:
        chunk = np.pad(audio, (0, win - len(audio)))
        chunks.append(chunk)
    return chunks

# ── Inference ──────────────────────────────────────────────────
def evaluate_audio_file(file_path, interpreter, inp_d, out_d):
    try:
        audio, _ = librosa.load(file_path, sr=SR, mono=True)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return
        
    chunks = chunk_audio(audio)
    filename = os.path.basename(file_path)
    print(f"\n==================================================")
    print(f"File: {filename} ({len(audio)/SR:.2f}s) - {len(chunks)} windows")
    print(f"==================================================")
    
    for idx, chunk in enumerate(chunks):
        t = idx * HOP_S
        feat = extract_features(chunk)
        
        # Quantize Feature Matrix
        x_q = _quantize_input(feat, inp_d)
        
        # Run Inference
        interpreter.set_tensor(inp_d['index'], x_q)
        interpreter.invoke()
        
        # De-quantize Output
        raw = interpreter.get_tensor(out_d['index'])
        out = _dequantize_output(raw, out_d)
        cry_p, sad_p, laugh_p = _decode_scores(out)
        
        # Classification Logic
        if cry_p >= CRY_THRESHOLD:
            if sad_p >= laugh_p:
                label = 'SAD'
                emoji = EMOJI_SAD
            else:
                label = 'HAPPY'
                emoji = EMOJI_HAPPY
        else:
            label = 'BACKGROUND'
            emoji = EMOJI_BG
            
        print(f"  [{t:>5.2f}s]  CryP={cry_p:.2f}  SadP={sad_p:.2f}  LaughP={laugh_p:.2f}  →  {emoji} {label}")

def main():
    parser = argparse.ArgumentParser(description='AuraSense quick demo inference')
    parser.add_argument(
        '--no-tensorflow-fallback',
        action='store_true',
        help='Disable TensorFlow fallback and require tflite_runtime only.'
    )
    args = parser.parse_args()

    interpreter_cls, backend = _resolve_interpreter(
        allow_tensorflow_fallback=not args.no_tensorflow_fallback
    )

    if interpreter_cls is None:
        print('Error: No safe TFLite interpreter available.')
        print('Recommended fix (stable on this demo):')
        print('  /usr/bin/python3 -m pip install tensorflow-macos')
        print('  or')
        print('  /usr/bin/python3 -m pip install tflite-runtime')
        print('Tip: fallback probe prevents hard crash if TensorFlow is unsafe.')
        sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        sys.exit(1)
        
    print(f"Loading TFLite model from: {MODEL_PATH}")
    print(f"Interpreter backend: {backend}")
    interpreter = interpreter_cls(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    
    inp_d = interpreter.get_input_details()[0]
    out_d = interpreter.get_output_details()[0]
    
    if not os.path.exists(AUDIO_DIR):
        print(f"Error: Audio directory not found at {AUDIO_DIR}")
        sys.exit(1)
        
    audio_files = [
        f for f in os.listdir(AUDIO_DIR)
        if f.lower().endswith(('.wav', '.mp3', '.m4a'))
    ]
    if not audio_files:
        print(f"No supported audio files found in {AUDIO_DIR}")
        return
        
    print(f"Found {len(audio_files)} test files. Starting inference...\n")
    for f in sorted(audio_files):
        evaluate_audio_file(os.path.join(AUDIO_DIR, f), interpreter, inp_d, out_d)

if __name__ == '__main__':
    main()
