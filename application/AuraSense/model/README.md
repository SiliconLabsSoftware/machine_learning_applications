# Model

This folder contains the machine-learning side of AuraSense.

## What Is Here

- `training/` - training script and training notes
- `artifacts/` - deployable TFLite model used by firmware
- `datasets/` - dataset reference and dataset access notes

## What The Model Is Used For

The model helps the EFR32xG26 firmware classify nursery audio into classes such as background, cry, laugh, and sad/distress.

## Typical Workflow

1. Prepare or expand the dataset
2. Train or fine-tune the model
3. Export a deployable `.tflite` file
4. Place the deployment model into the firmware project
5. Rebuild and flash the firmware
