# Model Artifacts

This folder contains deployable machine-learning files.

## Included Artifact

- `baby_cry_int8_DEPLOY.tflite`

## What It Is

This is the TensorFlow Lite model prepared for deployment into the AuraSense embedded firmware.

## Where It Is Used

The firmware project keeps its board-side model copy here:

- `../../firmware/aura-baby-monitor-soc/config/tflite/`

If you train a newer model, update that firmware-side location and rebuild the firmware.
