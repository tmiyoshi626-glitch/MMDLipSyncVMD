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