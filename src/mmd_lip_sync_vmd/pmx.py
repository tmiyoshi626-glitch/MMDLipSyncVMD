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

def _skip_vertices(f, header: PMXHeader) -> int:
    """Read the vertex count (temporary implementation)."""

    vertex_count = struct.unpack("<i", f.read(4))[0]

    print(f"Vertex count = {vertex_count}")

    return vertex_count

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

        return PMXModel(
            header=header,
            model_name_jp=model_name_jp,
            model_name_en=model_name_en,
        )