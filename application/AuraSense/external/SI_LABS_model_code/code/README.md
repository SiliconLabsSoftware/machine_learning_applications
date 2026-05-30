# Code README (`code/`)

## Purpose
This directory contains the complete execution pipeline for training, distilling, quantifying, and validating the AuraSense neural network. 

## Environment Setup
Run these commands in your virtual environment:

```bash
# PyTorch (CPU version is sufficient for these dataset sizes)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Core ML and Audio Processing
pip install tensorflow librosa soundfile numpy
```

## Easiest Copy-Paste Run

### macOS / Linux

If your dataset is in `/Users/rishabhsahay/Desktop/hola/dataset`, this is the exact command flow:

```bash
cd /Users/rishabhsahay/Desktop/hola
source .venv_train/bin/activate
export AURASENSE_DATA=/Users/rishabhsahay/Desktop/hola/dataset
python code/baby_cry_v3_local.py
```

### Windows (PowerShell)

If your dataset is in `C:\Users\<YourUser>\Desktop\hola\dataset`, this is the exact command flow:

```powershell
cd C:\Users\<YourUser>\Desktop\hola
.\.venv_train\Scripts\Activate.ps1
$env:AURASENSE_DATA="C:\Users\<YourUser>\Desktop\hola\dataset"
python code\baby_cry_v3_local.py
```

Quick startup check (shows first logs only):

```bash
cd /Users/rishabhsahay/Desktop/hola
source .venv_train/bin/activate
AURASENSE_DATA=/Users/rishabhsahay/Desktop/hola/dataset python code/baby_cry_v3_local.py | head -n 35
```

Windows PowerShell equivalent:

```powershell
cd C:\Users\<YourUser>\Desktop\hola
.\.venv_train\Scripts\Activate.ps1
$env:AURASENSE_DATA="C:\Users\<YourUser>\Desktop\hola\dataset"
python code\baby_cry_v3_local.py | Select-Object -First 35
```

Important:
- `| head -n 35` is only for startup verification.
- Full training is expected to take long on CPU; this is normal.

## Execution Order
The pipeline must be executed in this exact sequence:

### 1. Train Main Models
```bash
AURASENSE_DATA=/absolute/path/to/dataset python code/baby_cry_v3_local.py
```
```powershell
$env:AURASENSE_DATA="C:\absolute\path\to\dataset"
python code\baby_cry_v3_local.py
```
- **What it does:** Extracts 3-channel audio features (Mel Spectrogram, Spectral Flatness, RMS Energy) and trains two Float32 PyTorch models (M1 and M2).
- **Expected Output:** `artifacts/models/model_m1_detector.pt` and `artifacts/models/model_m2_emotion.pt`

### 2. Run Quantization & Export
```bash
python code/fix_int8_final.py
```
- **What it does:** Uses Knowledge Distillation (KD) to transfer PyTorch weights to a TFLite-compatible Keras student. Performs Post-Training Quantization (PTQ [weight and activation conversion to 8-bit integers]).
- **Expected Output:** `artifacts/tflite/baby_cry_int8_DEPLOY.tflite` (Target: ~55.7 KB)

### 3. Run Benchmark
```bash
python code/benchmark_final.py
```
- **What it does:** Benchmarks the INT8 TFLite model against the Float32 PyTorch baseline. Validates that the quantization penalty is within acceptable bounds (<5% drop).
- **Expected Output:** Confusion matrices saved to `verification/plots/`

### 4. Run External Evaluation (Optional)
```bash
python code/evaluate_test_external.py
```
- **What it does:** Runs the final pipeline on completely isolated test samples to prove independent generalization zero data leakage.

## Firmware Deployment
For the EFR32xG26 Silicon Labs deployment:
1. Copy the final `artifacts/tflite/baby_cry_int8_DEPLOY.tflite` to your firmware configuration directory.
2. Use **Simplicity Studio v6** to regenerate the `autogen/sl_tflite_micro_model.c` array.
3. Flash the unit. Ensure the firmware uses a **375ms inference interval** to match the pipeline training frequency.
