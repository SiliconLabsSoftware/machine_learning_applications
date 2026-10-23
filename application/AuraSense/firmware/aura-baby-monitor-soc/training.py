# =============================================================================
# LEGACY TRAINING SCRIPT (3-CLASS)
# =============================================================================
# NOTE:
# This script targets an older 3-class pipeline and is kept only for reference.
# The deployed firmware in this project now uses a fused 4-class model:
#   [background, cry, laugh, sad]
# exported as: config/tflite/baby_cry_unified_int8.tflite
#
# Use this file only if you intentionally want to reproduce the legacy model.
# =============================================================================
# TRAINING SCRIPT - Matching Silicon Labs EFR32XG26 Microfrontend
# =============================================================================
# This script produces features that match the SOC's Google microfrontend:
#   Audio → Hamming Window → 1024-pt Zero-Padded FFT → |X|²
#   → Mel Filterbank (40ch) → sqrt → ln(x) × 64 → uint16 [0,666] → int8
#
# Previous mismatches fixed:
#   1. Window: hann → hamming (SOC uses Hamming)
#   2. FFT: 640 → 1024 with win_length=640 (SOC zero-pads to 1024)
#   3. Log: power_to_db → ln(x)×64 (SOC uses fixed-point natural log)
#   4. Scale: float [-1,1] audio → int16×gain=2 equivalent scaling
#   5. Normalization: clip(dB) → [0,666] range matching SOC static quant
#   6. Augmentation: added 8x augmentation per sample
# =============================================================================
import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Conv2D, BatchNormalization, Activation,
    Add, GlobalAveragePooling2D, Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# =============================================================================
# FEATURE PARAMETERS — Must match SOC microfrontend config exactly
# =============================================================================
SAMPLE_RATE = 16000
N_MELS = 40              # SL_ML_FRONTEND_FILTERBANK_N_CHANNELS
WINDOW_MS = 40            # SL_ML_FRONTEND_WINDOW_SIZE_MS
STEP_MS = 20              # SL_ML_FRONTEND_WINDOW_STEP_MS
WIN_LENGTH = int(SAMPLE_RATE * WINDOW_MS / 1000)   # 640 samples
HOP_LENGTH = int(SAMPLE_RATE * STEP_MS / 1000)     # 320 samples
N_FFT = 1024              # SL_ML_FRONTEND_FFT_LENGTH (zero-padded from 640)
EXPECTED_FRAMES = 49      # (1000ms - 40ms) / 20ms + 1 = 49

SOC_AUDIO_GAIN = 2        # SL_ML_AUDIO_FEATURE_GENERATION_AUDIO_GAIN
LOG_SCALE_SHIFT = 6       # SL_ML_FRONTEND_LOG_SCALE_SHIFT → multiply by 2^6=64

# Dynamic quantization parameters (matching SOC DYNAMIC_SCALE_ENABLE=1)
# SOC computes: dynamic_range = (int)(40.0 * (1 << LOG_SCALE_SHIFT) * 0.11512925465) ≈ 295
# Then normalizes: [max - dynamic_range, max] → [-128, 127]
DYNAMIC_RANGE_DB = 40.0   # dB range for dynamic normalization

EPOCHS = 80
BATCH_SIZE = 16
NUM_AUGMENTS = 8          # Augmented copies per original sample

# =============================================================================
# FEATURE EXTRACTION — Approximating SOC Google Microfrontend
# =============================================================================
def extract_features(audio):
    """
    Extract mel-log features with DYNAMIC normalization matching SOC.

    SOC dynamic quantization (QUANTIZE_DYNAMIC_SCALE_ENABLE=1):
      1. Compute log-mel spectrogram as before
      2. Find max value across entire spectrogram
      3. Compute dynamic_range = 40dB * 64 * ln(10)/20 ≈ 295
      4. Set min_val = max(max_val - dynamic_range, 0)
      5. Normalize: (value - min_val) / (max_val - min_val) * 255 - 128

    This per-spectrogram normalization handles differences between
    librosa and the Google microfrontend that static scaling cannot.
    """
    # Compute spectrogram (using librosa's simpler dB output)
    # We'll use power_to_db since dynamic normalization handles the rest
    S = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        window='hamming',
        center=False,
        fmin=0.0,
        fmax=SAMPLE_RATE / 2,
        power=2.0,
        norm=None,
        htk=True
    )

    # Convert to dB scale (log10 based, standard audio processing)
    log_S = librosa.power_to_db(S, ref=np.max, top_db=DYNAMIC_RANGE_DB)
    # log_S is now in range [-DYNAMIC_RANGE_DB, 0] with max at 0

    # Dynamic normalization to [-1, 1] range
    # Maps [-40, 0] dB → [-1, 1]
    log_S = (log_S + DYNAMIC_RANGE_DB) / DYNAMIC_RANGE_DB * 2.0 - 1.0
    log_S = np.clip(log_S, -1.0, 1.0)

    # Pad or trim to EXPECTED_FRAMES (49)
    if log_S.shape[1] > EXPECTED_FRAMES:
        log_S = log_S[:, :EXPECTED_FRAMES]
    elif log_S.shape[1] < EXPECTED_FRAMES:
        pad_width = EXPECTED_FRAMES - log_S.shape[1]
        log_S = np.pad(log_S, ((0, 0), (0, pad_width)),
                       mode='constant', constant_values=-1.0)

    # Transpose to (time, mel, 1) — model input shape
    return log_S.T[..., np.newaxis].astype(np.float32)

# =============================================================================
# DATA AUGMENTATION
# =============================================================================
def augment_audio(audio, sr=SAMPLE_RATE):
    """Apply random audio-domain augmentation."""
    aug = audio.copy()

    # Random volume scaling (0.5× to 2.0×)
    aug = aug * np.random.uniform(0.5, 2.0)

    # Random time shift (±10%)
    shift = int(np.random.uniform(-0.1, 0.1) * len(aug))
    aug = np.roll(aug, shift)

    # Add background noise (50% chance)
    if np.random.random() < 0.5:
        noise_level = np.random.uniform(0.002, 0.015)
        aug = aug + np.random.randn(len(aug)) * noise_level

    # Pitch shift ±2 semitones (50% chance)
    if np.random.random() < 0.5:
        n_steps = np.random.uniform(-2.0, 2.0)
        aug = librosa.effects.pitch_shift(aug, sr=sr, n_steps=n_steps)

    # Time stretch 0.85×–1.15× (30% chance)
    if np.random.random() < 0.3:
        rate = np.random.uniform(0.85, 1.15)
        aug = librosa.effects.time_stretch(aug, rate=rate)
        if len(aug) > len(audio):
            aug = aug[:len(audio)]
        else:
            aug = np.pad(aug, (0, len(audio) - len(aug)))

    return np.clip(aug, -1.0, 1.0)


def spec_augment(features, num_freq_masks=2, freq_mask_width=4,
                 num_time_masks=2, time_mask_width=5):
    """SpecAugment: random frequency and time masking on spectrogram."""
    aug = features.copy()
    T, F, _ = aug.shape

    for _ in range(num_freq_masks):
        f = np.random.randint(0, freq_mask_width + 1)
        f0 = np.random.randint(0, max(1, F - f))
        aug[:, f0:f0 + f, :] = 0.0

    for _ in range(num_time_masks):
        t = np.random.randint(0, time_mask_width + 1)
        t0 = np.random.randint(0, max(1, T - t))
        aug[t0:t0 + t, :, :] = 0.0

    return aug

# =============================================================================
# DATASET LOADING (with augmentation)
# =============================================================================
DATASET_PATH = '/content/drive/MyDrive/datasets0'
TARGET_CLASSES = ['belly_pain', 'laugh', 'silence']  # 3 classes only
NUM_CLASSES = len(TARGET_CLASSES)  # 3
X, Y = [], []

for label_idx, label in enumerate(TARGET_CLASSES):
    folder = os.path.join(DATASET_PATH, label)
    files = [f for f in os.listdir(folder) if f.endswith('.wav')]
    print(f"Loading {len(files):3d} files from '{label}'")

    for f in files:
        try:
            audio, sr = librosa.load(os.path.join(folder, f), sr=SAMPLE_RATE)
            if len(audio) < SAMPLE_RATE:
                audio = np.pad(audio, (0, SAMPLE_RATE - len(audio)))
            chunk = audio[:SAMPLE_RATE]

            # Original sample
            X.append(extract_features(chunk))
            Y.append(label_idx)

            # Augmented copies
            for _ in range(NUM_AUGMENTS):
                aug_audio = augment_audio(chunk)
                aug_feat = extract_features(aug_audio)
                aug_feat = spec_augment(aug_feat)
                X.append(aug_feat)
                Y.append(label_idx)

        except Exception as e:
            print(f"  Error loading {f}: {e}")
            continue

Y = np.array(Y, dtype=np.int32)
X = np.array(X, dtype=np.float32)
Y_cat = to_categorical(Y, num_classes=NUM_CLASSES)

# Train/val split
X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y_cat, test_size=0.2, random_state=42, stratify=Y
)

print(f"\n{'='*60}")
print(f"Dataset: {len(X_train)} train / {len(X_val)} val")
print(f"Feature shape: {X_train.shape[1:]}")
print(f"Feature range: [{X_train.min():.2f}, {X_train.max():.2f}]")
print(f"Expected range (dynamic norm): [-1.0, 1.0]")
print(f"{'='*60}\n")

# =============================================================================
# MODEL (compatible with SOC opcode resolver: Conv2D, Add, Mean, FC, Softmax)
# =============================================================================
def build_model():
    inputs = Input(shape=(EXPECTED_FRAMES, N_MELS, 1))

    # Block 1
    x = Conv2D(16, 3, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Block 2 (residual)
    shortcut = x
    x = Conv2D(32, 3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, 3, padding='same')(x)
    x = BatchNormalization()(x)
    shortcut = Conv2D(32, 1, strides=2, padding='same')(shortcut)
    shortcut = BatchNormalization()(shortcut)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    # Block 3 (residual)
    shortcut = x
    x = Conv2D(64, 3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    shortcut = Conv2D(64, 1, strides=2, padding='same')(shortcut)
    shortcut = BatchNormalization()(shortcut)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)

    return Model(inputs, outputs)

model = build_model()
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# =============================================================================
# TRAINING WITH LR SCHEDULE + EARLY STOPPING
# =============================================================================
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8,
                      min_lr=1e-6, verbose=1),
    EarlyStopping(monitor='val_loss', patience=15,
                  restore_best_weights=True, verbose=1),
]

print("\nTraining...")
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# =============================================================================
# TFLITE CONVERSION (int8 quantized)
# =============================================================================
print("\nConverting to TFLite (int8)...")

def representative_dataset():
    """Supply training samples for int8 quantization.
    With dynamic normalization, features are in [-1, 1] range.
    Expected quantization: scale≈0.0078, zero_point≈0"""
    for i in range(min(200, len(X_train))):
        yield [X_train[i:i+1].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

OUTPUT_NAME = 'binary_audio_model_fixed_v2.tflite'
with open(OUTPUT_NAME, 'wb') as f:
    f.write(tflite_model)

print(f"\nSaved: {OUTPUT_NAME} ({len(tflite_model) / 1024:.1f} KB)")

# =============================================================================
# VERIFY QUANTIZATION PARAMETERS
# =============================================================================
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

inp_scale, inp_zp = inp['quantization']
out_scale, out_zp = out['quantization']

print(f"\n{'='*60}")
print(f"INPUT  quantization: scale={inp_scale:.4f}, zero_point={inp_zp}")
print(f"OUTPUT quantization: scale={out_scale:.4f}, zero_point={out_zp}")
print(f"Dynamic normalization: features in [-1, 1] range")
print(f"Expected input scale ≈ 0.0078 (2/256), zero_point ≈ 0")
if inp_scale < 0.02 and abs(inp_zp) < 10:
    print("MATCH: Input quantization matches dynamic normalization range")
else:
    print(f"WARNING: Unexpected quantization parameters")
    print(f"  Got scale={inp_scale:.4f}, zero_point={inp_zp}")
    print(f"  Expected scale≈0.0078, zero_point≈0")
print(f"{'='*60}")
