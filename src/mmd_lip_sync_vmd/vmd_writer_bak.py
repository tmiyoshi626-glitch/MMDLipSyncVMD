from __future__ import annotations

import struct
from pathlib import Path

from mmd_lip_sync_vmd.detector import DEFAULT_CLOSE_MORPH, VowelDetection

VMD_HEADER = "Vocaloid Motion Data 0002"
VMD_FPS = 30

VOWEL_TO_MORPH = {
"A": "あ",
"I": "い",
"U": "う",
"E": "え",
"O": "お",
}

def _fixed_shift_jis(value: str, size: int) -> bytes:
encoded = value.encode("shift_jis")
if len(encoded) > size:
encoded = encoded[:size]
return encoded + (b"\x00" * (size - len(encoded)))

def _frame_number(time_sec: float) -> int:
return max(0, round(time_sec * VMD_FPS))

def _morph_name(
detection: VowelDetection,
close_morph: str,
) -> str | None:
if detection.vowel in VOWEL_TO_MORPH:
return VOWEL_TO_MORPH[detection.vowel]

```
if detection.vowel == close_morph:
    return close_morph

return None
```

def build_vmd_bytes(
detections: list[VowelDetection],
*,
model_name: str = "MMDLipSyncVMD",
close_morph: str = DEFAULT_CLOSE_MORPH,
) -> bytes:

```
data = bytearray()

data.extend(_fixed_shift_jis(VMD_HEADER, 30))
data.extend(_fixed_shift_jis(model_name, 20))

# bone frame count
data.extend(struct.pack("<I", 0))

source_keyframes = [
    (
        morph_name,
        _frame_number(detection.time_sec),
        min(float(detection.confidence), 0.6),
    )
    for detection in detections
    if (morph_name := _morph_name(detection, close_morph)) is not None
]

morph_keyframes = []

for index, (morph_name, frame_number, weight) in enumerate(source_keyframes):

    if index + 1 < len(source_keyframes):
        next_frame_number = source_keyframes[index + 1][1]
        reset_frame_number = max(
            frame_number + 3,
            next_frame_number - 1,
        )
    else:
        reset_frame_number = frame_number + 3

    morph_keyframes.extend(
        [
            (morph_name, frame_number, weight),
            (morph_name, reset_frame_number, 0.0),
        ]
    )

data.extend(struct.pack("<I", len(morph_keyframes)))

for morph_name, frame_number, weight in morph_keyframes:
    data.extend(_fixed_shift_jis(morph_name, 15))
    data.extend(struct.pack("<I", frame_number))
    data.extend(struct.pack("<f", weight))

data.extend(struct.pack("<I", 0))
data.extend(struct.pack("<I", 0))
data.extend(struct.pack("<I", 0))
data.extend(struct.pack("<I", 0))

return bytes(data)
```

def write_vmd(
detections: list[VowelDetection],
vmd_path: str | Path,
*,
model_name: str = "MMDLipSyncVMD",
close_morph: str = DEFAULT_CLOSE_MORPH,
) -> None:

```
Path(vmd_path).write_bytes(
    build_vmd_bytes(
        detections,
        model_name=model_name,
        close_morph=close_morph,
    )
)
```
