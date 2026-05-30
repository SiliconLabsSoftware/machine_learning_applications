# 🍼 AuraSense — Edge AI Baby Cry Detection & Emotion Classification

> **Fully on-device, real-time baby cry detection and emotion classification on the Silicon Labs EFR32xG26 (MG26) microcontroller.**
> Zero cloud. Zero audio leaves the chip. Sub-second response. ₹875–₹1,505 BOM.

**Team:** Rishabh Sahay · Siddharth Nautiyal · Naitik Gupta

---

## What AuraSense Does

AuraSense is a **55.7 KB pure INT8 neural network** deployed on an ARM Cortex-M33 microcontroller that:

1. **Listens** — captures audio via I2S MEMS microphone at 16 kHz
2. **Detects** — binary cry gate: is a baby vocalising? (Stage 1)
3. **Classifies** — emotion: is the baby **sad/distressed** 🔵 or **happy/laughing** 🟡? (Stage 2)
4. **Signals** — RGB LEDs on the IoT device + BLE 5.0 notification to phone app

| State | LED | App Alert | BLE Class ID |
|---|---|---|---|
| 🔵 SAD / Distressed | Blue | Sound alert triggered | 1 |
| 🟡 HAPPY / Laughing | Yellow | Silent display | 0 |
| 🔴 BACKGROUND / Quiet | Red (idle) | No alert | 2 |

---

## Key Metrics

| Metric | Value |
|---|---|
| Model size | **55.7 KB** (pure INT8, zero float32 ops) |
| Inference time | **~30 ms** on Cortex-M33 @ 78 MHz |
| Full cycle | **375 ms** (53 ms active, 322 ms EM1 sleep) |
| End-to-end alert | **< 1.5 seconds** (6-frame majority vote) |
| Deployed accuracy | **95.0%** INT8 TFLite |
| Stage 1 recall (cry detection) | **99.8%** (2,983 / 2,990) |
| Stage 2 recall (sad) | **99.7%** (1,248 / 1,252) |
| Stage 2 recall (laugh) | **100%** (376 / 376) |
| Flash usage | 286 KB / 1,024 KB (28%) |
| SRAM usage | 193 KB / 256 KB (75%) |
| Active power | < 50 mW |
| BLE payload | **4 bytes** — zero audio transmitted |
| Dataset | 5,000+ strictly independent samples from 6 sources |

---

## Why Edge Intelligence (Not Cloud)

| Factor | Cloud AI | AuraSense (Edge) |
|---|---|---|
| **Latency** | 1–5 s round-trip | ~30 ms local inference |
| **Privacy** | Baby audio uploaded to servers | 100% on-chip, zero audio leaves device |
| **Reliability** | Requires stable WiFi | Works standalone — no Internet needed |
| **Cost** | Recurring API fees | Zero recurring cost, ₹875–₹1,505 BOM |
| **Power** | Always-on WiFi radio | 86% sleep duty cycle, < 50 mW active |

---

## Repository Structure

```
SI_LABS/
├── README.md                    ← You are here
├── .gitignore
├── requirements.txt             ← Python dependencies
│
├── dataset/                     ← Audio corpus (all training/eval data)
│   ├── README.md                ← Dataset documentation
│   ├── sad/                     ← Distressed baby cry WAV files
│   ├── laugh/                   ← Happy baby audio WAV files
│   ├── audio/                   ← CryCeleb2023 cry corpus
│   ├── esc50/                   ← ESC-50 environmental sounds
│   │   ├── esc50.csv
│   │   └── audio/
│   ├── noise/                   ← DEMAND real-world noise scenes
│   │   ├── DKITCHEN/
│   │   ├── DLIVING/
│   │   ├── DWASHING/
│   │   ├── NPARK/
│   │   └── OHALLWAY/
│   └── background/              ← Ambient room tones
│
├── code/                        ← ML pipeline scripts
│   ├── README.md                ← Setup + run commands
│   ├── baby_cry_v3_local.py     ← Step 1: Train PyTorch teacher models
│   ├── fix_int8_final.py        ← Step 2: Knowledge Distillation → INT8 TFLite
│   ├── benchmark_final.py       ← Step 3: Benchmark INT8 vs PyTorch
│   ├── split_datasets.py        ← Utility: Dataset splitting
│   └── evaluate_test_external.py← Step 4: External dataset evaluation
│
├── artifacts/                   ← Trained model files
│   ├── README.md                ← What each artifact proves
│   ├── models/
│   │   ├── model_m1_detector.pt
│   │   ├── model_m2_emotion.pt
│   │   └── model_m1_detector_qat.pt
│   └── tflite/
│       ├── baby_cry_fused_fp32.tflite
│       ├── baby_cry_fused_int8.tflite
│       ├── baby_cry_fused_qat_int8.tflite
│       └── baby_cry_int8_DEPLOY.tflite   ★ Final deployment model
│
└── verification/                ← Evidence & proof package
    ├── README.md                ← Auditor reading order
    ├── plots/
    │   ├── confusion_matrix_m1.png
    │   ├── confusion_matrix_m2.png
    │   ├── fp_audit.png
    │   ├── accuracy_vs_noise_IoT.png
    │   └── confusion_matrix_15db_IoT.png
    └── proofs/
        ├── DOC1_MODEL1_COMPLETE_LABELED_DATA.txt
        ├── DOC2_MODEL2_COMPLETE_LABELED_DATA.txt
        ├── DOC3_LINE_BY_LINE_PROOF_NO_LEAKAGE.txt
        ├── DATA_LEAKAGE_ANALYSIS.txt
        ├── TRAIN_TEST_VAL_SPLIT_EXPLANATION.txt
        ├── MODEL1_COMPLETE_FILENAMES.txt
        └── MODEL2_COMPLETE_FILENAMES.txt
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/SiddharthN123/SI_LABS.git
cd SI_LABS
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell equivalent:

```powershell
git clone https://github.com/SiddharthN123/SI_LABS.git
cd SI_LABS
py -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
# Step 1: Train PyTorch teacher models (M1: Cry Detector, M2: Emotion Classifier)
AURASENSE_DATA=/absolute/path/to/dataset python code/baby_cry_v3_local.py

# Step 2: Knowledge Distillation + INT8 Quantization → TFLite
python code/fix_int8_final.py

# Step 3: Benchmark INT8 model against PyTorch baseline
python code/benchmark_final.py

# Step 4: (Optional) Evaluate on external test set
python code/evaluate_test_external.py
```

Windows PowerShell equivalent:

```powershell
# Step 1: Train PyTorch teacher models (M1: Cry Detector, M2: Emotion Classifier)
$env:AURASENSE_DATA="C:\absolute\path\to\dataset"
python code\baby_cry_v3_local.py

# Step 2: Knowledge Distillation + INT8 Quantization → TFLite
python code\fix_int8_final.py

# Step 3: Benchmark INT8 model against PyTorch baseline
python code\benchmark_final.py

# Step 4: (Optional) Evaluate on external test set
python code\evaluate_test_external.py
```

Quick startup-check only (first 35 lines):

```bash
AURASENSE_DATA=/absolute/path/to/dataset python code/baby_cry_v3_local.py | head -n 35
```

Windows PowerShell equivalent:

```powershell
$env:AURASENSE_DATA="C:\absolute\path\to\dataset"
python code\baby_cry_v3_local.py | Select-Object -First 35
```

Note: full Step 1 training is CPU-heavy and can take a long time; that is expected.

### 3. Expected Outputs

After running the pipeline:
- `artifacts/models/` — PyTorch `.pt` checkpoints (teacher models)
- `artifacts/tflite/baby_cry_int8_DEPLOY.tflite` — **55.7 KB deployment model** (pure INT8)
- `verification/plots/` — Confusion matrices + FP audit + noise robustness plots

---

## Hardware Architecture

```
[MEMS Microphone]
    │ I2S/PDM — 16-bit PCM @ 16 kHz — DMA Ring Buffer
    ▼
[EFR32xG26 (MG26) MCU — ARM Cortex-M33]
    ├── HW ML Audio Frontend (512-pt FFT + 40-bin Mel Filterbank)
    ├── 3-Channel Feature Assembly:
    │     Ch0: Mel Spectrogram (voice signature)
    │     Ch1: Spectral Flatness (tonal vs noise discrimination)
    │     Ch2: RMS Energy (loudness envelope)
    ├── TFLite Micro INT8 Inference (55.7 KB flatbuffer, ~30 ms)
    ├── 6-Frame Majority Vote Decision Logic
    │
    ├──[GPIO]──▶ [3× RGB LEDs]  (🔵 Blue / 🟡 Yellow / 🔴 Red)
    └──[BLE 5.0]──▶ [Si Connect App]  (4-byte GATT payload, ZERO audio)
```

**BLE Payload (4 bytes — what leaves the device):**

| Byte | Content | Description |
|---|---|---|
| 0 | Class ID | 0 = Happy, 1 = Sad, 2 = Background |
| 1 | Cry Confidence % | Stage 1 model probability × 100 |
| 2 | Emotion Confidence % | Stage 2 model probability × 100 |
| 3 | Reserved | App toggle flags (future) |

---

## ML Pipeline — Two-Stage Binary Cascade

### Why Not 5-Class?

Spectrogram analysis revealed that **distress sub-classes overlap heavily** (hunger, pain, discomfort share acoustic features), while **laugh is morphologically distinct**. Academic 5-class models reporting 90–95% accuracy suffer from **data leakage** (augmenting 8 burp clips into 300+ training samples). Under honest evaluation with file-level splitting, 5-class accuracy drops to ~60%.

### Our Approach

```
Audio Input (0.75s window, 16 kHz)
        │
        ▼
┌─────────────────────────┐
│  Stage 1: CRY vs NOT-CRY │  ← Binary gate, P(CRY) ≥ 0.70
│  (99.8% recall)           │
└────────┬────────┬─────────┘
     PASS │        │ FAIL
         ▼        ▼
┌────────────┐  🔴 BACKGROUND
│ Stage 2:    │
│ SAD vs LAUGH│  ← Emotion classifier
│ (99.7%/100%)│
└──┬──────┬──┘
   ▼      ▼
🔵 SAD  🟡 HAPPY
```

### Knowledge Distillation (KD)

The PyTorch teacher model uses `AdaptiveAvgPool2d` (not supported in INT8 TFLite). A Keras student model with `GlobalAveragePooling2D` was trained via KD (Temperature=3.0) to replicate teacher predictions → pure INT8 quantization → **55.7 KB deployment file with only 3.4% accuracy penalty**.

---

## Confusion Matrix Results (Final Deployed Model)

**Model 1 — Cry Detection:**
```
                  Pred CRY    Pred NOT-CRY
True CRY            2,983           7         (99.8% recall)
True NOT-CRY           12         388
```

**Model 2 — Emotion Classification:**
```
                  Pred SAD    Pred LAUGH
True SAD            1,248           4         (99.7% recall)
True LAUGH              0         376         (100% recall)
```

**INT8 Demo on Real Audio (IoT Hardware):**

| Test File | 🔴 Background | 🟡 Happy | 🔵 Sad | Verdict |
|---|:---:|:---:|:---:|---|
| background.wav | **81%** | 19% | 0% | 🔴 BACKGROUND ✅ |
| laugh.wav | 15% | **74%** | 11% | 🟡 HAPPY ✅ |
| sad.wav | 12% | 14% | **74%** | 🔵 SAD ✅ |

---

## Development Journey

| Phase | Period | What Happened | Outcome |
|---|---|---|---|
| 1 | January | Concept: breath analysis + 5-class monitoring | Breath analysis abandoned — embedded constraints |
| 2 | Mid-Jan | Basic raw CNN deployed | Failed against background noise |
| 3 | Late-Jan | Wav2Vec, YAMNet + CNN hybrids | Too large for IoT. Domain transfer failed |
| 4 | February | Deep literature review | Found academic data-leakage flaw |
| 5 | Early-Mar | Spectrogram analysis → Binary Cascade | Architecture breakthrough |
| 6 | Mid-Mar | Knowledge Distillation + INT8 quantization | Deployed on EFR32xG26 |
| 7 | Late-Mar | 3-channel pipeline, confidence scoring, app | System fully functional |

---

## Firmware Deployment (EFR32xG26)

1. Copy `artifacts/tflite/baby_cry_int8_DEPLOY.tflite` → firmware `config/tflite/` directory
2. Open project in **Simplicity Studio v5** → **Generate** → **Build**
3. Flash via J-Link debugger
4. Open serial console (115200 baud) to view live inference logs

See `code/README.md` for detailed deployment steps.

---

## License

This project uses portions of the Silicon Labs Gecko SDK under the Silicon Labs MSLA. Audio datasets are subject to their respective licenses:
- **ESC-50:** CC-BY-NC
- **DEMAND:** CC-BY-SA
- **CryCeleb2023:** Research use
