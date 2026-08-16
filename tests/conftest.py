from __future__ import annotations

import struct
from pathlib import Path

import pytest


def _text(value: str, encoding: str) -> bytes:
    data = value.encode(encoding)
    return struct.pack("<i", len(data)) + data


def _morph_offset(morph_type: int) -> bytes:
    if morph_type == 0:  # group
        return b"\x00" + struct.pack("<f", 1.0)
    if morph_type == 1:  # vertex
        return b"\x00" + (b"\x00" * 12)
    if morph_type == 2:  # bone
        return b"\x00" + (b"\x00" * 28)
    if morph_type == 3:  # UV
        return b"\x00" + (b"\x00" * 16)
    if morph_type == 8:  # material
        return b"\xff\x00" + (b"\x00" * 112)
    raise ValueError(f"Unsupported test morph type: {morph_type}")


def _build_minimal_pmx(encoding: str = "utf-8") -> bytes:
    """Build a self-contained PMX 2.0 file for parser tests."""
    data = bytearray(b"PMX ")
    data.extend(struct.pack("<f", 2.0))
    encoding_flag = 0 if encoding == "utf-16-le" else 1
    data.extend(bytes((8, encoding_flag, 0, 1, 2, 1, 1, 1, 1)))

    for value in ("テストモデル", "Test model", "", ""):
        data.extend(_text(value, encoding))

    data.extend(struct.pack("<I", 1))
    data.extend(b"\x00" * 32)  # position, normal, and UV
    data.extend(b"\x00")  # BDEF1
    data.extend(b"\xff")  # no bone index
    data.extend(struct.pack("<f", 1.0))  # edge scale

    data.extend(struct.pack("<I", 0))  # face index count
    data.extend(struct.pack("<I", 2))
    data.extend(_text("body.png", encoding))
    data.extend(_text("face.png", encoding))

    data.extend(struct.pack("<I", 1))
    data.extend(_text("Body", encoding))
    data.extend(_text("Body", encoding))
    data.extend(b"\x00" * 16)  # diffuse
    data.extend(b"\x00" * 12)  # specular
    data.extend(b"\x00" * 4)  # specular strength
    data.extend(b"\x00" * 12)  # ambient
    data.extend(b"\x00")  # draw flags
    data.extend(b"\x00" * 16)  # edge color
    data.extend(b"\x00" * 4)  # edge size
    data.extend(b"\xff\xff")  # texture index
    data.extend(b"\xff\xff")  # sphere texture index
    data.extend(b"\x00")  # sphere mode
    data.extend(b"\x01")  # shared toon texture
    data.extend(b"\x00")  # shared toon texture index
    data.extend(_text("", encoding))  # memo
    data.extend(struct.pack("<I", 0))  # material face count

    data.extend(struct.pack("<I", 1))  # bone count
    data.extend(_text("センター", encoding))
    data.extend(_text("Center", encoding))
    data.extend(b"\x00" * 12)  # position
    data.extend(b"\xff")  # parent bone index
    data.extend(struct.pack("<i", 0))  # transform level
    data.extend(struct.pack("<H", 0))  # bone flags
    data.extend(b"\x00" * 12)  # tail position

    morphs = [
        ("あ", "A", 3, 0, 1),
        ("い", "I", 3, 1, 1),
        ("う", "U", 3, 2, 1),
        ("え", "E", 3, 3, 1),
        ("お", "O", 4, 8, 1),
    ]
    data.extend(struct.pack("<I", len(morphs)))
    for name_jp, name_en, panel, morph_type, offset_count in morphs:
        data.extend(_text(name_jp, encoding))
        data.extend(_text(name_en, encoding))
        data.extend(bytes((panel,)))
        data.extend(bytes((morph_type,)))
        data.extend(struct.pack("<I", offset_count))
        for _ in range(offset_count):
            data.extend(_morph_offset(morph_type))

    return bytes(data)


@pytest.fixture
def pmx_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "minimal.pmx"
    path.write_bytes(_build_minimal_pmx())
    return path


@pytest.fixture
def pmx_utf16_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "minimal-utf16.pmx"
    path.write_bytes(_build_minimal_pmx("utf-16-le"))
    return path
