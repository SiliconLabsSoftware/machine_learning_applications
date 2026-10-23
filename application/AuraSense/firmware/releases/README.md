# Firmware Releases

This folder contains prebuilt firmware images for AuraSense.

## Included File

- `aura_baby_monitor.s37` - ready-to-flash firmware image for the prepared EFR32xG26 setup

## When To Use This

Use this file if you want to demo the project quickly without rebuilding the firmware from source.

## Flashing

With Simplicity Commander:

```bash
commander flash firmware/releases/aura_baby_monitor.s37 --device EFR32MG26 --serialno <JLINK_SERIAL>
commander device reset --serialno <JLINK_SERIAL>
```

If you only have one connected board/debug adapter, the `--serialno` argument can usually be omitted.

## After Flashing

1. Power or reset the board.
2. Open the Android app.
3. Connect to the device over BLE.
4. Open the Baby Cry Monitor screen and wait for live data.
