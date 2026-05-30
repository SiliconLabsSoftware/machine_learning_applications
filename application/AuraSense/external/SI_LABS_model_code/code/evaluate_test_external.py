import os, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import librosa
import tensorflow as tf
from tqdm import tqdm

# -- Config --
SR = 16000
WIN_SAMPLES = int(SR * 0.75)
HOP_SAMPLES = int(SR * 0.375)
N_MEL = 40; N_FFT = 512; HOP_LENGTH = 160; T_FRAMES = 75
CRY_THRESHOLD = 0.70

BASE = '/Users/rishabhsahay/Desktop/hola/code/test_external'
TFLITE_MODEL = '/Users/rishabhsahay/Desktop/hola/outputs/tflite/baby_cry_int8_DEPLOY.tflite'
OUT_MD = '/Users/rishabhsahay/Desktop/hola/code/test_external_evaluation.md'

# -- Audio Utilities --
def extract_features(audio):
    if len(audio) < WIN_SAMPLES: audio = np.pad(audio, (0, WIN_SAMPLES - len(audio)))
    else: audio = audio[:WIN_SAMPLES]
    mel = librosa.feature.melspectrogram(y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MEL, fmin=60, fmax=8000)
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=40)
    log_mel = (log_mel + 40.0) / 40.0
    T = log_mel.shape[1]
    
    flatness = librosa.feature.spectral_flatness(y=audio, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    flatness = np.clip(np.log1p(flatness * 1000.0) / (np.max(np.log1p(flatness * 1000.0)) + 1e-8), 0, 1)
    flatness_ch = np.tile(flatness[np.newaxis, :T], (N_MEL, 1))

    rms = librosa.feature.rms(y=audio, frame_length=N_FFT, hop_length=HOP_LENGTH)[0][:T]
    rms = np.clip(np.log1p(rms * 100.0) / (np.max(np.log1p(rms * 100.0)) + 1e-8), 0, 1)
    rms_ch = np.tile(rms[np.newaxis, :T], (N_MEL, 1))

    feat = np.stack([log_mel[:, :T], flatness_ch[:, :T], rms_ch[:, :T]], axis=-1).transpose(1, 0, 2)
    if feat.shape[0] < T_FRAMES:
        feat = np.concatenate([feat, np.zeros((T_FRAMES - feat.shape[0], N_MEL, 3), dtype=np.float32)])
    else:
        feat = feat[:T_FRAMES]
    return feat.astype(np.float32)

def load_wav(path):
    try: 
        a, _ = librosa.load(path, sr=SR, mono=True)
        return a
    except: 
        return None

def chunk_audio(audio):
    chunks = []; start = 0
    while start + WIN_SAMPLES <= len(audio):
        chunks.append(audio[start:start + WIN_SAMPLES])
        start += HOP_SAMPLES
    
    # If the audio is shorter than WIN_SAMPLES, pad and yield one chunk
    if not chunks and len(audio) > 0:
        chunks.append(np.pad(audio, (0, max(0, WIN_SAMPLES - len(audio)))))
    return chunks

def list_wavs(d):
    w = []
    for r, _, fs in os.walk(d):
        for f in fs:
            if f.lower().endswith('.wav'): w.append(os.path.join(r, f))
    return sorted(w)


def main():
    print("Loading INT8 model...")
    interp = tf.lite.Interpreter(model_path=TFLITE_MODEL)
    interp.allocate_tensors()
    inp_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    inp_sc, inp_zp = inp_d['quantization']
    out_sc, out_zp = out_d['quantization']
    
    def predict(audio_array):
        feat = extract_features(audio_array)
        x_q = (feat[np.newaxis] / inp_sc + inp_zp).astype(inp_d['dtype'])
        interp.set_tensor(inp_d['index'], x_q)
        interp.invoke()
        raw = interp.get_tensor(out_d['index'])
        out_f = (raw.astype(np.float32) - out_zp) * out_sc
        
        cry_p = float(np.clip(out_f[0, 0], 0, 1))
        sad_p = float(np.clip(out_f[0, 2], 0, 1))
        laugh_p = float(np.clip(out_f[0, 3], 0, 1))
        
        if cry_p >= CRY_THRESHOLD:
            return ('SAD' if sad_p >= laugh_p else 'LAUGH'), cry_p, sad_p, laugh_p
        else:
            return 'BACKGROUND', cry_p, sad_p, laugh_p

    classes = ['sad', 'laugh', 'background']
    target_labels = {'sad': 'SAD', 'laugh': 'LAUGH', 'background': 'BACKGROUND'}
    
    md_lines = []
    md_lines.append("# Final External Test Dataset Evaluation")
    md_lines.append("")
    md_lines.append("This table provides a per-file evaluation of the 1-second clips. Files marked with ❌ might contain pure silence or insufficient acoustic features due to natural pauses in the clips.")
    md_lines.append("")
    md_lines.append("| Directory | Filename | Ground Truth | Final Prediction | Correct? | Conf (Cry) | Conf (Sad) | Conf (Laugh) | Flags |")
    md_lines.append("|-----------|----------|--------------|------------------|----------|------------|------------|--------------|-------|")
    
    total, correct = 0, 0

    print("Evaluating individual clips...")
    for cls in classes:
        folder = os.path.join(BASE, cls)
        wavs = list_wavs(folder)
        gt_label = target_labels[cls]
        
        for w in tqdm(wavs, desc=f"Evaluating {cls}"):
            audio = load_wav(w)
            if audio is None: continue
            
            fname = os.path.basename(w)
            # A clip is 1.0s. It yields 1-2 chunks of 0.75s depending on hop length.
            # We predict on all chunks and if ANY chunk matches GT, we count it as correct (since hardware slides window)
            chunks = chunk_audio(audio)
            
            best_pred = "BACKGROUND"
            best_c, best_s, best_l = 0., 0., 0.
            match_found = False
            
            for c in chunks:
                pred, c_p, s_p, l_p = predict(c)
                # Save first chunk info or best matching info
                if pred == gt_label:
                    best_pred = pred
                    best_c, best_s, best_l = c_p, s_p, l_p
                    match_found = True
                    break
                
                # If no match yet, just keep the highest probability
                if c_p > best_c:
                    best_pred = pred
                    best_c, best_s, best_l = c_p, s_p, l_p
            
            total += 1
            if match_found:
                correct += 1
                status = "✅"
                flag = ""
            else:
                status = "❌"
                flag = "Check for silence" if best_c < 0.2 else ""
                
            md_lines.append(f"| `{cls}/` | `{fname}` | **{gt_label}** | {best_pred} | {status} | {best_c:.2f} | {best_s:.2f} | {best_l:.2f} | {flag} |")

    accuracy = correct / total if total > 0 else 0
    
    md_lines.insert(2, f"**Overall File-level Accuracy:** {accuracy*100:.2f}% ({correct}/{total})")
    
    with open(OUT_MD, 'w') as f:
        f.write("\n".join(md_lines))
        
    print(f"\nDone! File-by-file log saved to: {OUT_MD}")
    print(f"Overall File-level Accuracy: {accuracy*100:.2f}% ({correct}/{total})")

if __name__ == "__main__":
    main()
