# Firmware

This folder contains everything needed on the embedded side of AuraSense.

## What Is Here

- `aura-baby-monitor-soc/` - the main Simplicity Studio firmware project
- `releases/` - prebuilt firmware image for fast flashing

## What The Firmware Does

- Records audio from the on-board microphone path
- Runs TensorFlow Lite Micro inference on the EFR32xG26
- Sends classification results over BLE
- Exposes basic control commands for the phone app
- Can publish room-temperature data when the board configuration supports it

## If You Are New

Start here:

1. Read `aura-baby-monitor-soc/README.md`
2. If you want the fastest demo path, use `releases/README.md`

## Hardware Focus

The included project metadata targets the EFR32xG26 family and is configured around the Silicon Labs board setup referenced in the `.slcp` project file.
