# Training

This folder contains the training-side code for the AuraSense audio classifier.

## Main File

- `training.py`

## Purpose

Use this script as the starting point for retraining or improving the baby-audio classifier before exporting a new TensorFlow Lite deployment model.

## Output Of This Step

The important result of training is a deployable `.tflite` model that can be copied into:

- `../artifacts/`
- `../../firmware/aura-baby-monitor-soc/config/tflite/`

## Reminder

After changing the model used by the firmware, regenerate the embedded model source expected by the Simplicity Studio build flow.
