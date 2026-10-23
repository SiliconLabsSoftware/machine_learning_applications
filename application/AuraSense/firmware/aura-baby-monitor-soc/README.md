# AuraSense Firmware Project

This is the main embedded project for the AuraSense nursery monitor. It runs on Silicon Labs EFR32xG26 hardware and performs on-device audio classification plus BLE communication with the Android app.

## Target Hardware

- Family: `EFR32xG26`
- Device in project metadata: `EFR32MG26B510F3200IM68`
- Board reference in project metadata: `brd2608a`

## What The Firmware Does

- Captures audio features on the board
- Runs a TensorFlow Lite Micro model locally
- Classifies audio into these classes:
  - `0` = `background`
  - `1` = `cry`
  - `2` = `laugh`
  - `3` = `sad`
- Sends the result to the mobile app using BLE notifications
- Accepts basic control commands from the app
- Exposes room-temperature data where supported by the board configuration

## Important Files

- `ml_ble_classifier.slcp` - Simplicity Studio project entry point
- `app.c` - BLE app logic and event handling
- `app.h` - shared definitions and BLE state
- `audio_classifier.cc` - audio inference pipeline and decision logic
- `recognize_commands.cc` - classification helper logic
- `config/audio_classifier_config.h` - classifier configuration
- `config/pin_config.h` - board and pin-related setup
- `config/tflite/baby_cry_int8_DEPLOY.tflite` - deployable model file
- `autogen/sl_tflite_micro_model.c` - generated embedded C array form of the model
- `autogen/gatt_db.c` - generated BLE GATT database

## BLE Behavior

### Notification Payload

Characteristic: `gattdb_inference_result`

Notification payload format:

- Byte `0`: class id
- Byte `1`: confidence score encoded as unsigned 8-bit value

### Control Characteristic

Service UUID:

- `a3c87500-8ed3-4bdf-8a39-a01bebede295`

Control Characteristic UUID:

- `a3c87502-8ed3-4bdf-8a39-a01bebede295`

Command format:

- Byte `0`: command id
- Byte `1`: command value

Supported command ids:

- `1` = monitoring enable (`0/1`)
- `2` = alerts enable (`0/1`)
- `3` = confidence threshold (`0..100`)
- `4` = debounce count (`1..10`)
- `5` = deep sleep enable (`0/1`)
- `6` = device enable (`0/1`)

## Build From Source

### Simplicity Studio 6

1. Open Simplicity Studio 6.
2. Import `ml_ble_classifier.slcp`.
3. Let Studio resolve the required SDK and AI/ML components.
4. Build the project.
5. Flash it to the board.

## Model Notes

- The `.tflite` file lives in `config/tflite/`.
- The generated C model array lives in `autogen/sl_tflite_micro_model.c`.
- If you replace the model, regenerate the embedded model output before rebuilding.

## Practical Advice

- If you only want to demo the project, use `../releases/aura_baby_monitor.s37`.
- If you move to another EFR32xG26 board, check microphone, sensor, and pin mappings first.
