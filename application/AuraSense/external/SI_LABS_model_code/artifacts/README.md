# Artifacts README (`artifacts/`)

## Purpose
Stores the generated, compiled, and serialized model files from the training and export pipeline. These files represent the deliverable intelligence of the project.

## Subfolders

### `models/`
Stores the PyTorch training checkpoints (.pt files) containing serialized parameter weights.

- `model_m1_detector.pt`: 
  - PyTorch Float32 baseline for Stage 1 (Cry vs Background).
  - Proves the theoretical capability (accuracy ceiling) of the architecture before compression.
- `model_m2_emotion.pt`: 
  - PyTorch Float32 baseline for Stage 2 (Sad vs Laugh).
- `model_m1_detector_qat.pt`: 
  - Checkpoint generated during Quantization-Aware Training (QAT).

### `tflite/`
Stores the final TensorFlow Lite models exported for edge inference.

- `baby_cry_fused_fp32.tflite`: 
  - The unfused Float32 graph. Baseline for hardware latency comparisons.
- `baby_cry_fused_int8.tflite`: 
  - Intermediate 8-bit quantization result.
- `baby_cry_fused_qat_int8.tflite`: 
  - The QAT (Quantization Aware Training) INT8 branch result.
- `baby_cry_int8_DEPLOY.tflite`: 
  - **The final, golden deployment model (55.7 KB).**
  - Uses Post-Training Quantization (PTQ [8-bit fixed-point conversion]).
  - Proves the model can fit entirely within the 256 KB SRAM constraint of the EFR32xG26 MCU without invoking Float32 fallback operations.
