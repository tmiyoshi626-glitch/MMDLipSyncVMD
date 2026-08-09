"""PMX model utilities."""

from __future__ import annotations

import struct

from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class PMXMorph:
    """PMX morph metadata used to identify model expressions."""

    name_jp: str
    name_en: str
    panel: int
    morph_type: int
    offset_count: int

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

def _skip_bytes(f, size: int) -> None:
    """Skip a fixed number of bytes."""
    f.seek(size, 1)

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

    for _ in range(vertex_count):
        # position (3 floats)
        f.seek(12, 1)

        # normal (3 floats)
        f.seek(12, 1)

        # UV (2 floats)
        f.seek(8, 1)

        # additional UVs
        f.seek(16 * header.additional_uvs, 1)

        # weight type
        weight_type = struct.unpack("<B", f.read(1))[0]

        # weight data
        _skip_vertex_weight(
            f,
            weight_type,
            header.bone_index_size,
        )

        # edge scale
        f.seek(4, 1)

    return vertex_count


def _skip_faces(f, header: PMXHeader) -> int:
    """Read and skip face indices."""

    face_index_count = _read_uint32(f)

    f.seek(
        face_index_count * header.vertex_index_size,
        1,
    )

    return face_index_count
def _skip_material(f, header: PMXHeader, encoding: str) -> None:
    """Skip remaining PMX material data after names."""
    
    # diffuse color
    f.seek(16, 1)

    # specular color
    f.seek(12, 1)

    # specular strength
    f.seek(4, 1)

    # ambient color
    f.seek(12, 1)

    # draw flags
    f.seek(1, 1)

    # edge color
    f.seek(16, 1)

    # edge size
    f.seek(4, 1)

    # texture index
    _read_index(f, header.texture_index_size)

    # sphere texture index
    _read_index(f, header.texture_index_size)

    # sphere mode
    f.seek(1, 1)

    toon_flag = struct.unpack("<B", f.read(1))[0]
    if toon_flag == 0:
        _read_index(f, header.texture_index_size)
    elif toon_flag == 1:
        _skip_bytes(f, 1)
    else:
        raise ValueError(f"Unknown PMX toon flag: {toon_flag}")

    # memo
    _read_pmx_text(f, encoding)

    # face count
    f.seek(4, 1)
def _read_textures(f, encoding: str) -> list[str]:
    """Read PMX texture names."""

    texture_count = _read_uint32(f)

    textures = []

    for _ in range(texture_count):
        textures.append(_read_pmx_text(f, encoding))

    return textures

def _read_materials(f, header: PMXHeader, encoding: str) -> list[str]:
    """Read PMX material names."""

    material_count = _read_uint32(f)
    materials: list[str] = []

    for _ in range(material_count):
        material_name_jp = _read_pmx_text(f, encoding)
        _read_pmx_text(f, encoding)  # English name

        materials.append(material_name_jp)

        _skip_material(f, header, encoding)

    return materials

def _skip_bones(f, header: PMXHeader, encoding: str) -> int:
    """Read and skip all PMX bone records."""

    bone_count = _read_uint32(f)
    for _ in range(bone_count):
        _read_pmx_text(f, encoding)
        _read_pmx_text(f, encoding)
        _skip_bytes(f, 12)  # position
        _read_index(f, header.bone_index_size)  # parent bone index
        _skip_bytes(f, 4)  # transform level
        flags = struct.unpack("<H", f.read(2))[0]

        if flags & 0x0001:  # connection index
            _read_index(f, header.bone_index_size)
        else:  # connection position
            _skip_bytes(f, 12)

        if flags & (0x0100 | 0x0200):  # additional rotation or movement
            _read_index(f, header.bone_index_size)
            _skip_bytes(f, 4)

        if flags & 0x0400:  # fixed axis
            _skip_bytes(f, 12)

        if flags & 0x0800:  # local axis
            _skip_bytes(f, 24)

        if flags & 0x2000:  # external parent transform
            _skip_bytes(f, 4)

        if flags & 0x0020:  # inverse kinematics
            _read_index(f, header.bone_index_size)
            _skip_bytes(f, 8)  # iterations and angle limit
            link_count = _read_uint32(f)
            for _ in range(link_count):
                _read_index(f, header.bone_index_size)
                has_limits = struct.unpack("<B", f.read(1))[0]
                if has_limits:
                    _skip_bytes(f, 24)

    return bone_count


def _skip_morph_offsets(
    f,
    header: PMXHeader,
    morph_type: int,
    offset_count: int,
) -> None:
    """Skip PMX 2.0 morph offsets for one morph."""

    if morph_type == 0:  # group
        offset_size = header.morph_index_size + 4
    elif morph_type == 1:  # vertex
        offset_size = header.vertex_index_size + 12
    elif morph_type == 2:  # bone
        offset_size = header.bone_index_size + 28
    elif morph_type in (3, 4, 5, 6, 7):  # UV and additional UVs
        offset_size = header.vertex_index_size + 16
    elif morph_type == 8:  # material
        offset_size = header.material_index_size + 113
    else:
        raise ValueError(f"Unsupported PMX 2.0 morph type: {morph_type}")

    _skip_bytes(f, offset_size * offset_count)


def _read_morphs(
    f,
    header: PMXHeader,
    encoding: str,
) -> list[PMXMorph]:
    """Read PMX morph metadata while skipping morph offsets."""

    morph_count = _read_uint32(f)
    morphs: list[PMXMorph] = []
    for _ in range(morph_count):
        name_jp = _read_pmx_text(f, encoding)
        name_en = _read_pmx_text(f, encoding)
        panel = struct.unpack("<B", f.read(1))[0]
        morph_type = struct.unpack("<B", f.read(1))[0]
        offset_count = _read_uint32(f)
        morphs.append(
            PMXMorph(
                name_jp=name_jp,
                name_en=name_en,
                panel=panel,
                morph_type=morph_type,
                offset_count=offset_count,
            )
        )
        _skip_morph_offsets(f, header, morph_type, offset_count)

    return morphs

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
    face_index_count: int
    textures: list[str]
    materials: list[str]
    bone_count: int
    morphs: list[str]
    morph_details: list[PMXMorph] = field(default_factory=list)

def read_pmx_model(pmx_path: Path) -> PMXModel:
    """Read PMX header and model names."""
    bone_count: int
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
        face_index_count = _skip_faces(f, header)
        textures = _read_textures(f, encoding)
        materials = _read_materials(f, header, encoding)
        bone_count = _skip_bones(f, header, encoding)
        morph_details = _read_morphs(f, header, encoding)
        morphs = [morph.name_jp for morph in morph_details]
          
        return PMXModel(
            header=header,
            model_name_jp=model_name_jp,
            model_name_en=model_name_en,
            vertex_count=vertex_count,
            face_index_count=face_index_count,
            textures=textures,
            materials=materials,
            bone_count=bone_count,
            morphs=morphs,
            morph_details=morph_details,
        )

