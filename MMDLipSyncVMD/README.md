# MMDLipSyncVMD

Python tools for generating MMD lip-sync VMD motion data.

## Requirements

- Windows 11
- Python 3.12, 3.13, or 3.14

## Setup

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e . pytest
```

## Run

```powershell
mmd-lip-sync-vmd --version
```

Detect vowels from a WAV vocal file and write VMD morph keyframes:

```powershell
mmd-lip-sync-vmd .\input\vocal.wav .\output\lip_sync.vmd
```

The VMD output maps vowels to MMD morphs:

```text
A -> あ
I -> い
U -> う
E -> え
O -> お
```

Each keyframe uses `time_sec * 30` as the frame number and `confidence` as
the morph weight.

The processing pipeline is:

```text
detect_vowels() -> smooth_detections() -> insert_silence_frames() -> write_vmd() / write_csv()
```

Silent gaps between detected vowels generate mouth-close morph frames at 30 FPS.
The default mouth-close morph is `口閉じ`, and generated close frames use weight
`1.0`.

Detect vowels from a WAV vocal file and write CSV instead:

```powershell
mmd-lip-sync-vmd .\input\vocal.wav .\output\vowels.csv
```

Smoothing is enabled by default with a centered window of `3` detections:

```powershell
mmd-lip-sync-vmd .\input\vocal.wav .\output\lip_sync.vmd --smooth-window 3
```

Disable smoothing:

```powershell
mmd-lip-sync-vmd .\input\vocal.wav .\output\lip_sync.vmd --smooth-window 1
```

Configure silence gap handling:

```powershell
mmd-lip-sync-vmd .\input\vocal.wav .\output\lip_sync.vmd --silence-threshold 0.12 --close-morph 口閉じ
```

Use `--silence-threshold` to set the minimum gap, in seconds, before
mouth-close frames are inserted. Use `--close-morph` when your model uses a
different close-mouth morph name.

The CSV contains:

```csv
time_sec,vowel,confidence
0.0000,A,0.8123
0.0333,口閉じ,1.0
0.0400,A,0.7865
```

Optional analysis timing controls:

```powershell
mmd-lip-sync-vmd .\input\vocal.wav .\output\lip_sync.vmd --frame-duration 0.08 --hop-duration 0.04 --smooth-window 3 --silence-threshold 0.12
```

## Test

```powershell
pytest
```
