# AuraSense Quick Demo (Easy Run)

## What this is
This folder lets anyone run baby-cry inference quickly using the already exported TFLite model.

## Why this exists
- No training needed.
- No dataset setup needed.
- Just run and see predictions on sample/test audio.

## Important (for your Mac)
Use **system Python** to avoid the crash you saw with `python3` from another environment.

For Windows, use `py` or your active virtual environment Python consistently.

---

## 1) One-time setup (copy-paste)

Run from repository root (`/Users/rishabhsahay/Desktop/hola`):

```bash
/usr/bin/python3 -m pip install --user numpy librosa soundfile tensorflow-macos
```

Windows PowerShell equivalent:

```powershell
cd C:\Users\<YourUser>\Desktop\hola
py -m pip install --user numpy librosa soundfile tensorflow
```

---

## 2) Run the demo (copy-paste)

```bash
cd /Users/rishabhsahay/Desktop/hola
/usr/bin/python3 quick_demo/demo_inference.py
```

Windows PowerShell equivalent:

```powershell
cd C:\Users\<YourUser>\Desktop\hola
py quick_demo\demo_inference.py
```

---

## 3) Add your own audio
Put files into `quick_demo/test_audio/`.

Supported formats:
- `.wav`
- `.mp3`
- `.m4a`

Example:

```bash
cp /path/to/your_audio.wav /Users/rishabhsahay/Desktop/hola/quick_demo/test_audio/
/usr/bin/python3 /Users/rishabhsahay/Desktop/hola/quick_demo/demo_inference.py
```

Windows PowerShell equivalent:

```powershell
Copy-Item C:\path\to\your_audio.wav C:\Users\<YourUser>\Desktop\hola\quick_demo\test_audio\
py C:\Users\<YourUser>\Desktop\hola\quick_demo\demo_inference.py
```

---

## Output meaning
Each line prints one audio window result:
- `CryP` = cry probability
- `SadP` = sad probability
- `LaughP` = laugh probability

Labels:
- 🔵 `SAD`
- 🟡 `HAPPY` (laugh)
- 🔴 `BACKGROUND`

---

## Replace with newer model

```bash
cp /path/to/new_model.tflite /Users/rishabhsahay/Desktop/hola/quick_demo/model/baby_cry_int8_DEPLOY.tflite
/usr/bin/python3 /Users/rishabhsahay/Desktop/hola/quick_demo/demo_inference.py
```

Windows PowerShell equivalent:

```powershell
Copy-Item C:\path\to\new_model.tflite C:\Users\<YourUser>\Desktop\hola\quick_demo\model\baby_cry_int8_DEPLOY.tflite
py C:\Users\<YourUser>\Desktop\hola\quick_demo\demo_inference.py
```

---

## Troubleshooting

If command crashes, always run this exact command:

```bash
/usr/bin/python3 /Users/rishabhsahay/Desktop/hola/quick_demo/demo_inference.py
```

Windows PowerShell equivalent:

```powershell
py C:\Users\<YourUser>\Desktop\hola\quick_demo\demo_inference.py
```

If still unstable, disable TensorFlow fallback:

```bash
/usr/bin/python3 /Users/rishabhsahay/Desktop/hola/quick_demo/demo_inference.py --no-tensorflow-fallback
```

Windows PowerShell equivalent:

```powershell
py C:\Users\<YourUser>\Desktop\hola\quick_demo\demo_inference.py --no-tensorflow-fallback
```
