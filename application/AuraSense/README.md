# AuraSense EIC 2026

AuraSense is a smart baby-monitoring project built for the EIC 2026 context. It combines an EFR32xG26 firmware application, a custom Android caregiver app, and an edge AI audio model so a nursery device can listen for baby distress sounds and notify a caregiver over Bluetooth Low Energy (BLE).

## What This Project Does

In simple terms:

1. The EFR32xG26 board listens to nearby baby audio.
2. The firmware runs an on-device TinyML model.
3. The board classifies the sound as background, cry, laugh, or sad/distress.
4. The result is sent to the Android app over BLE.
5. The app shows live status, confidence, room temperature, and alerts the caregiver when distress is detected.

## Who This Repo Is For

- Students or judges who want to understand the project quickly
- Developers who want to build the firmware or app
- Anyone working with Silicon Labs EFR32xG26 hardware

## Hardware Target

This firmware project is configured for:

- Silicon Labs EFR32xG26 family
- Device: `EFR32MG26B510F3200IM68`
- Board component in project metadata: `brd2608a`

If you use a different EFR32xG26 board, you may need to adjust microphone, sensor, or pin configuration before building.

## Repository Layout

```text
AuraSense_EIC_2026/
  firmware/
    aura-baby-monitor-soc/     # Main EFR32xG26 firmware project
    releases/                  # Ready-to-flash .s37 firmware image
  mobile/
    aurasense-android-app/     # Android caregiver application
  model/
    training/                  # Training script and training notes
    artifacts/                 # Deployable TFLite model
    datasets/                  # Dataset pointer / dataset notes
```

## Quick Start

If you only want to try the project as fast as possible:

1. Flash `firmware/releases/aura_baby_monitor.s37` to your EFR32xG26 board.
2. Open `mobile/aurasense-android-app` in Android Studio.
3. Build and install the Android app on an Android phone.
4. Power the board, pair/connect over BLE, and open the Baby Cry Monitor screen.

## Build Requirements

### Firmware

- Simplicity Studio 6
- Silicon Labs Simplicity SDK `2025.12.1`
- AI/ML extension `2.2.0`
- Supported EFR32xG26 hardware
- J-Link / Simplicity Commander for flashing

### Android App

- Android Studio
- JDK 17
- Android SDK for `compileSdk 36`
- Android device with BLE support

## Main Files You Will Care About

### Firmware

- `firmware/aura-baby-monitor-soc/ml_ble_classifier.slcp`
- `firmware/aura-baby-monitor-soc/app.c`
- `firmware/aura-baby-monitor-soc/audio_classifier.cc`
- `firmware/aura-baby-monitor-soc/config/tflite/baby_cry_int8_DEPLOY.tflite`

### Android App

- `mobile/aurasense-android-app/mobile/build.gradle.kts`
- `mobile/aurasense-android-app/mobile/src/main/java/com/siliconlabs/bledemo/features/demo/babycry/BabyCryMonitorActivity.kt`
- `mobile/aurasense-android-app/mobile/src/main/res/values/strings_baby_cry.xml`

### Model

- `model/training/training.py`
- `model/artifacts/baby_cry_int8_DEPLOY.tflite`
- `model/datasets/datasets.txt`

## How The System Works

- The firmware performs audio feature extraction and TensorFlow Lite Micro inference on the device.
- BLE notifications carry the inference result to the phone.
- The Android app turns those predictions into a caregiver-friendly monitoring screen and alert flow.
- Temperature data is also surfaced when available through the firmware configuration.

## Notes For Public Use

- This repo intentionally excludes local IDE files, build output, and generated machine-specific folders.
- The Android app is a customized fork of Silicon Labs' mobile app shell, adapted here for AuraSense monitoring.

## Folder Guides

Each main folder contains its own `README.md` with detailed instructions:

- `firmware/README.md`
- `firmware/aura-baby-monitor-soc/README.md`
- `firmware/releases/README.md`
- `mobile/README.md`
- `mobile/aurasense-android-app/README.md`
- `model/README.md`
- `model/training/README.md`
- `model/artifacts/README.md`
- `model/datasets/README.md`
