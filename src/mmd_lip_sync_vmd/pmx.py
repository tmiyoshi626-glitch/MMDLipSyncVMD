"""PMX model utilities."""

from __future__ import annotations

import struct

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PMXHeader:
    """PMX header information."""

    version: float
    encoding: str
    additional_uvs: int
    vertex_index_size: int
    texture_index_size: int
    material_index_size: int
    bone_index_size: int
    morph_index_size: int
    rigid_body_index_size: int

def _read_uint32(f):
    """4バイトの符号なし整数をリトルエンディアンで読み込む。"""
    return struct.unpack("<I", f.read(4))[0]

def _read_pmx_text(f, encoding: str) -> str:
    """Read one PMX text field."""

    length = struct.unpack("<i", f.read(4))[0]

    if length == 0:
        return ""

    data = f.read(length)

    if encoding == "UTF-16":
        return data.decode("utf-16-le")

    return data.decode("utf-8")

def _read_index(f, size: int, signed: bool = True) -> int:
    """Read a PMX index."""

    if size == 1:
        fmt = "<b" if signed else "<B"
    elif size == 2:
        fmt = "<h" if signed else "<H"
    elif size == 4:
        fmt = "<i" if signed else "<I"
    else:
        raise ValueError(f"Unsupported index size: {size}")

    return struct.unpack(fmt, f.read(size))[0]

def _skip_vertex_weight(f, weight_type: int, bone_index_size: int) -> None:
    """Skip one PMX vertex weight block."""

    if weight_type == 0:  # BDEF1
        f.seek(bone_index_size, 1)

    elif weight_type == 1:  # BDEF2
        f.seek(bone_index_size * 2 + 4, 1)

    elif weight_type == 2:  # BDEF4
        f.seek(bone_index_size * 4 + 16, 1)

    elif weight_type == 3:  # SDEF
        f.seek(bone_index_size * 2 + 40, 1)

    elif weight_type == 4:  # QDEF
        f.seek(bone_index_size * 4 + 16, 1)

    else:
        raise ValueError(f"Unknown vertex weight type: {weight_type}")

# TODO(Version2):
# Read and skip all vertex records according to the PMX 2.0 specification.
# The current implementation only reads the vertex count.


def _skip_vertices(f, header: PMXHeader) -> int:
    """Read and skip all PMX vertex records."""

    vertex_count = _read_uint32(f)

    return vertex_count


def _skip_faces(f, header: PMXHeader) -> int:
    """Read and skip face indices."""

    face_index_count = _read_uint32(f)

    f.seek(
        face_index_count * header.vertex_index_size,
        1,
    )

    return face_index_count


def read_pmx_header(pmx_path: Path) -> PMXHeader:
    ...


def read_pmx_header(pmx_path: Path) -> PMXHeader:
    """Read the PMX header."""

    with pmx_path.open("rb") as f:
        magic = f.read(4)

        if magic != b"PMX ":
            raise ValueError("Not a PMX file.")

        version = struct.unpack("<f", f.read(4))[0]

        globals_size = f.read(1)[0]
        globals_data = f.read(globals_size)

        encoding = "UTF-16" if globals_data[0] == 0 else "UTF-8"

        additional_uvs = globals_data[1]

        vertex_index_size = globals_data[2]
        texture_index_size = globals_data[3]
        material_index_size = globals_data[4]
        bone_index_size = globals_data[5]
        morph_index_size = globals_data[6]
        rigid_body_index_size = globals_data[7]

        return PMXHeader(
            version=version,
            encoding=encoding,
            additional_uvs=additional_uvs,
            vertex_index_size=vertex_index_size,
            texture_index_size=texture_index_size,
            material_index_size=material_index_size,
            bone_index_size=bone_index_size,
            morph_index_size=morph_index_size,
            rigid_body_index_size=rigid_body_index_size,
        )
    
@dataclass
class PMXModel:
    """PMX model information."""

    header: PMXHeader
    model_name_jp: str
    model_name_en: str
    vertex_count: int

def read_pmx_model(pmx_path: Path) -> PMXModel:
    """Read PMX header and model names."""

    with pmx_path.open("rb") as f:
        magic = f.read(4)
        if magic != b"PMX ":
            raise ValueError("Not a PMX file.")

        version = struct.unpack("<f", f.read(4))[0]

        globals_size = f.read(1)[0]
        globals_data = f.read(globals_size)

        encoding = "UTF-16" if globals_data[0] == 0 else "UTF-8"

        header = PMXHeader(
            version=version,
            encoding=encoding,
            additional_uvs=globals_data[1],
            vertex_index_size=globals_data[2],
            texture_index_size=globals_data[3],
            material_index_size=globals_data[4],
            bone_index_size=globals_data[5],
            morph_index_size=globals_data[6],
            rigid_body_index_size=globals_data[7],
        )

        model_name_jp = _read_pmx_text(f, encoding)
        model_name_en = _read_pmx_text(f, encoding)

        comment_jp = _read_pmx_text(f, encoding)
        comment_en = _read_pmx_text(f, encoding)

        vertex_count = _skip_vertices(f, header)



        return PMXModel(
            header=header,
            model_name_jp=model_name_jp,
            model_name_en=model_name_en,
            vertex_count=vertex_count,
        )

