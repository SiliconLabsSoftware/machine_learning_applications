# AuraSense Android App

This Android Studio project is the phone-side companion for the AuraSense embedded monitor. It is based on the Silicon Labs mobile application shell and customized here for baby cry monitoring.

## What The App Shows

- Live connection status to the EFR32xG26 device
- Current classification result from the firmware
- Confidence score
- Room temperature when present
- Alert and escalation state
- Controls for monitoring, alerts, thresholds, and debounce timing

## Main AuraSense Customization

The custom monitoring screen is implemented here:

- `mobile/src/main/java/com/siliconlabs/bledemo/features/demo/babycry/BabyCryMonitorActivity.kt`

Related custom resources:

- `mobile/src/main/res/values/strings_baby_cry.xml`

## Build Requirements

- Android Studio
- JDK 17
- Android SDK for `compileSdk 36`
- Android phone with BLE support

## Open And Run

1. Open `mobile/aurasense-android-app` in Android Studio.
2. Let Gradle sync finish.
3. Connect an Android phone or start a supported emulator for non-BLE UI work.
4. Run the `mobile` app configuration.

## Command Line Build

From this folder:

```bash
./gradlew assembleDebug
```

On Windows:

```powershell
.\gradlew.bat assembleDebug
```

## Key Project Files

- `settings.gradle.kts` - Android project entry
- `build.gradle.kts` - top-level build file
- `mobile/build.gradle.kts` - app module config
- `mobile/src/main/AndroidManifest.xml` - manifest
- `mobile/src/main/java/.../BabyCryMonitorActivity.kt` - AuraSense monitor logic

## Notes

- The internal package name still follows the Silicon Labs app base (`com.siliconlabs.bledemo`) to keep the customized project stable.
- This repo excludes local Gradle caches, IDE state, and generated build output.
