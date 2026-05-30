# AuraSense Training From Scratch (Full Rerun)

This guide is for a complete rerun of training from scratch (new train + new model artifacts).

## What this does
- Trains Model 1 (CRY vs NOT-CRY) from zero.
- Trains Model 2 (SAD vs LAUGH) from zero.
- Exports TFLite outputs and evaluation plots.

## Why this guide exists
- Single copy-paste flow.
- No ambiguity about command order.
- Includes preflight checks before long training starts.

---

## 0) Requirements
- Python 3.9+
- Dataset present at: `/Users/rishabhsahay/Desktop/hola/dataset`
- Windows example dataset path: `C:\Users\<YourUser>\Desktop\hola\dataset`

Expected dataset folders:
- `sad/`
- `laugh/`
- `audio/`
- `esc50/audio/`
- `esc50/esc50.csv`
- `noise/`
- `background/`

---

## 1) Create and activate clean virtual environment

macOS / Linux:

```bash
cd /Users/rishabhsahay/Desktop/hola
/usr/bin/python3 -m venv .venv_train
source .venv_train/bin/activate
python -m pip install --upgrade pip
```

Windows (PowerShell):

```powershell
cd C:\Users\<YourUser>\Desktop\hola
py -m venv .venv_train
.\.venv_train\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

---

## 2) Install dependencies

macOS (Apple Silicon):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install tensorflow-macos librosa soundfile numpy matplotlib seaborn scikit-learn
```

Windows / Linux:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install tensorflow librosa soundfile numpy matplotlib seaborn scikit-learn
```

---

## 3) Preflight verification (must pass)

### 3.1 Python file sanity

```bash
python -m py_compile code/baby_cry_v3_local.py
python -m py_compile code/fix_int8_final.py
python -m py_compile code/benchmark_final.py
python -m py_compile code/evaluate_test_external.py
```

### 3.2 Dataset path check

macOS / Linux:

```bash
python - << 'PY'
from pathlib import Path
base = Path('/Users/rishabhsahay/Desktop/hola/dataset')
required = [
    'sad', 'laugh', 'audio', 'esc50/audio', 'esc50/esc50.csv', 'noise', 'background'
]
missing = [r for r in required if not (base / r).exists()]
print('MISSING:', missing if missing else 'None')
PY
```

Windows (PowerShell):

```powershell
python -c "from pathlib import Path; base = Path(r'C:\Users\<YourUser>\Desktop\hola\dataset'); required = ['sad','laugh','audio','esc50/audio','esc50/esc50.csv','noise','background']; missing = [r for r in required if not (base / r).exists()]; print('MISSING:', missing if missing else 'None')"
```

If anything is missing, fix dataset before training.

---

## 4) Full from-scratch training run

Important: this is long-running.

macOS / Linux:

```bash
cd /Users/rishabhsahay/Desktop/hola
source .venv_train/bin/activate
AURASENSE_DATA=/Users/rishabhsahay/Desktop/hola/dataset python code/baby_cry_v3_local.py
```

Windows (PowerShell):

```powershell
cd C:\Users\<YourUser>\Desktop\hola
.\.venv_train\Scripts\Activate.ps1
$env:AURASENSE_DATA="C:\Users\<YourUser>\Desktop\hola\dataset"
python code\baby_cry_v3_local.py
```

This command:
- builds features,
- trains both models,
- saves model checkpoints,
- exports tflite outputs,
- saves plots.

---

## 5) Optional post-training steps

### 5.1 Quantization fix/extra export

```bash
AURASENSE_DATA=/Users/rishabhsahay/Desktop/hola/dataset python code/fix_int8_final.py
```

```powershell
$env:AURASENSE_DATA="C:\Users\<YourUser>\Desktop\hola\dataset"
python code\fix_int8_final.py
```

### 5.2 Benchmark

```bash
AURASENSE_DATA=/Users/rishabhsahay/Desktop/hola/dataset python code/benchmark_final.py
```

```powershell
$env:AURASENSE_DATA="C:\Users\<YourUser>\Desktop\hola\dataset"
python code\benchmark_final.py
```

### 5.3 External evaluation

```bash
AURASENSE_DATA=/Users/rishabhsahay/Desktop/hola/dataset python code/evaluate_test_external.py
```

```powershell
$env:AURASENSE_DATA="C:\Users\<YourUser>\Desktop\hola\dataset"
python code\evaluate_test_external.py
```

---

## 6) Expected outputs

- Models: `outputs/models/`
- TFLite: `outputs/tflite/`
- Plots: `outputs/plots/`

Key files expected:
- `outputs/models/model_m1_detector.pt`
- `outputs/models/model_m2_emotion.pt`
- `outputs/tflite/baby_cry_fused_int8.tflite`
- `outputs/plots/confusion_matrix_m1.png`
- `outputs/plots/confusion_matrix_m2.png`

---

## 7) Common issues

### Wrong Python interpreter
Use the venv Python (`source .venv_train/bin/activate`) before running scripts.

On Windows PowerShell use `\.venv_train\Scripts\Activate.ps1`.

### Dataset not found
Set env var exactly:

```bash
AURASENSE_DATA=/Users/rishabhsahay/Desktop/hola/dataset
```

Windows PowerShell:

```powershell
$env:AURASENSE_DATA="C:\Users\<YourUser>\Desktop\hola\dataset"
```

### Long runtime
This is expected for full scratch training on CPU.
