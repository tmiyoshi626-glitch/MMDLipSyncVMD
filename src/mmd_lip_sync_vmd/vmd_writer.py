from __future__ import annotations

import struct
from pathlib import Path

from mmd_lip_sync_vmd.detector import VowelDetection

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


def _morph_name(detection: VowelDetection) -> str | None:
    if detection.vowel in VOWEL_TO_MORPH:
        return VOWEL_TO_MORPH[detection.vowel]

    return None


def _select_one_morph_per_frame(
    detections: list[VowelDetection],
) -> list[tuple[str, int, float]]:
    """Keep the highest-confidence mouth morph candidate for each VMD frame."""

    selected: list[tuple[str, int, float]] = []
    positions_by_frame: dict[int, int] = {}
    for detection in detections:
        morph_name = _morph_name(detection)
        if morph_name is None:
            continue

        frame_number = _frame_number(detection.time_sec)
        candidate = (morph_name, frame_number, float(detection.confidence))
        existing_position = positions_by_frame.get(frame_number)
        if existing_position is None:
            positions_by_frame[frame_number] = len(selected)
            selected.append(candidate)
        elif candidate[2] > selected[existing_position][2]:
            selected[existing_position] = candidate

    return selected


def build_vmd_bytes(
    detections: list[VowelDetection],
    *,
    model_name: str = "MMDLipSyncVMD",
) -> bytes:

    data = bytearray()

    # VMDヘッダ
    data.extend(_fixed_shift_jis(VMD_HEADER, 30))
    data.extend(_fixed_shift_jis(model_name, 20))

    # ボーンキーフレーム数
    data.extend(struct.pack("<I", 0))

    source_keyframes = _select_one_morph_per_frame(detections)

    morph_keyframes: list[tuple[str, int, float]] = []

    for index, (morph_name, frame_number, weight) in enumerate(source_keyframes):

        if index + 1 < len(source_keyframes):
            next_frame_number = source_keyframes[index + 1][1]

            reset_frame_number = max(
                frame_number + 1,
                next_frame_number - 1,
            )
        else:
            reset_frame_number = frame_number + 1

        morph_keyframes.extend(
            [
                (morph_name, frame_number, weight),
                (morph_name, reset_frame_number, 0.0),
            ]
        )    # モーフキーフレーム数
    data.extend(struct.pack("<I", len(morph_keyframes)))

    for morph_name, frame_number, weight in morph_keyframes:
        data.extend(_fixed_shift_jis(morph_name, 15))
        data.extend(struct.pack("<I", frame_number))
        data.extend(struct.pack("<f", weight))

    # カメラ・ライト・セルフシャドウ・IK
    data.extend(struct.pack("<I", 0))
    data.extend(struct.pack("<I", 0))
    data.extend(struct.pack("<I", 0))
    data.extend(struct.pack("<I", 0))

    return bytes(data)


def write_vmd(
    detections: list[VowelDetection],
    vmd_path: str | Path,
    *,
    model_name: str = "MMDLipSyncVMD",
) -> None:

    Path(vmd_path).write_bytes(
        build_vmd_bytes(
            detections,
            model_name=model_name,
        )
    )
