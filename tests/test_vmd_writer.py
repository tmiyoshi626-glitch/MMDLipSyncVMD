import struct

from mmd_lip_sync_vmd.detector import VowelDetection
from mmd_lip_sync_vmd.vmd_writer import (
    VMD_HEADER,
    VOWEL_TO_MORPH,
    build_vmd_bytes,
    write_vmd,
)


def _strip_nulls(value: bytes) -> bytes:
    return value.split(b"\x00", 1)[0]


def _morph_keyframes(data: bytes) -> list[tuple[str, int, float]]:
    keyframe_count = struct.unpack_from("<I", data, 54)[0]
    keyframes: list[tuple[str, int, float]] = []
    offset = 58
    for _ in range(keyframe_count):
        name = _strip_nulls(data[offset:offset + 15]).decode("shift_jis")
        frame = struct.unpack_from("<I", data, offset + 15)[0]
        weight = struct.unpack_from("<f", data, offset + 19)[0]
        keyframes.append((name, frame, weight))
        offset += 23
    return keyframes


def test_build_vmd_bytes_writes_vmd_header_and_counts() -> None:
    data = build_vmd_bytes(
        [
            VowelDetection(time_sec=0.0, vowel="A", confidence=0.75),
            VowelDetection(time_sec=0.5, vowel="I", confidence=0.5),
        ]
    )

    assert _strip_nulls(data[:30]).decode("shift_jis") == VMD_HEADER
    assert _strip_nulls(data[30:50]).decode("shift_jis") == "MMDLipSyncVMD"
    assert struct.unpack_from("<I", data, 50)[0] == 0
    assert struct.unpack_from("<I", data, 54)[0] == 4


def test_build_vmd_bytes_converts_vowels_to_morph_keyframes() -> None:
    data = build_vmd_bytes(
        [
            VowelDetection(time_sec=0.0, vowel="A", confidence=0.75),
            VowelDetection(time_sec=0.5, vowel="O", confidence=0.5),
        ]
    )

    assert _morph_keyframes(data) == [
        (VOWEL_TO_MORPH["A"], 0, 0.75),
        (VOWEL_TO_MORPH["A"], 14, 0.0),
        (VOWEL_TO_MORPH["O"], 15, 0.5),
        (VOWEL_TO_MORPH["O"], 16, 0.0),
    ]


def test_build_vmd_bytes_resets_vowel_morph_after_keyframe() -> None:
    data = build_vmd_bytes(
        [VowelDetection(time_sec=0.0, vowel="A", confidence=1.0)]
    )

    assert _morph_keyframes(data) == [
        (VOWEL_TO_MORPH["A"], 0, 1.0),
        (VOWEL_TO_MORPH["A"], 1, 0.0),
    ]


def test_build_vmd_bytes_never_resets_before_next_frame() -> None:
    data = build_vmd_bytes(
        [
            VowelDetection(time_sec=0.0, vowel="A", confidence=1.0),
            VowelDetection(time_sec=1 / 30, vowel="I", confidence=1.0),
        ]
    )

    assert _morph_keyframes(data) == [
        (VOWEL_TO_MORPH["A"], 0, 1.0),
        (VOWEL_TO_MORPH["A"], 1, 0.0),
        (VOWEL_TO_MORPH["I"], 1, 1.0),
        (VOWEL_TO_MORPH["I"], 2, 0.0),
    ]


def test_build_vmd_bytes_uses_highest_confidence_vowel_per_frame() -> None:
    data = build_vmd_bytes(
        [
            VowelDetection(time_sec=0.0, vowel="A", confidence=0.25),
            VowelDetection(time_sec=0.01, vowel="I", confidence=0.75),
        ]
    )

    assert _morph_keyframes(data) == [
        (VOWEL_TO_MORPH["I"], 0, 0.75),
        (VOWEL_TO_MORPH["I"], 1, 0.0),
    ]


def test_build_vmd_bytes_keeps_first_vowel_when_confidence_ties() -> None:
    data = build_vmd_bytes(
        [
            VowelDetection(time_sec=0.0, vowel="A", confidence=0.75),
            VowelDetection(time_sec=0.01, vowel="I", confidence=0.75),
        ]
    )

    assert _morph_keyframes(data) == [
        (VOWEL_TO_MORPH["A"], 0, 0.75),
        (VOWEL_TO_MORPH["A"], 1, 0.0),
    ]


def test_write_vmd_creates_binary_file(tmp_path) -> None:
    output_path = tmp_path / "lip_sync.vmd"

    write_vmd(
        [VowelDetection(time_sec=1.0, vowel="E", confidence=0.25)],
        output_path,
    )

    assert output_path.read_bytes() == build_vmd_bytes(
        [VowelDetection(time_sec=1.0, vowel="E", confidence=0.25)]
    )
