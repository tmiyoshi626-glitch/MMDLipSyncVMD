from pathlib import Path

from mmd_lip_sync_vmd.pmx import read_pmx_header, read_pmx_model


def test_read_pmx_header():
    header = read_pmx_header(Path("sample/Teto Red(NCHL).pmx"))

    assert header.version == 2.0


def test_read_pmx_model():
    model = read_pmx_model(Path("sample/Teto Red(NCHL).pmx"))

    assert model.header.version == 2.0
    assert model.model_name_jp != ""
    assert model.model_name_en is not None
    assert model.vertex_count > 0
    assert model.face_index_count > 0

# TODO:
# Add tests for faces, textures, materials, bones and morphs.