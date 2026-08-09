from mmd_lip_sync_vmd.pmx import (
    PMXHeader,
    PMXModel,
    PMXMorph,
    read_pmx_header,
    read_pmx_model,
)


def test_read_pmx_header(pmx_fixture):
    header = read_pmx_header(pmx_fixture)

    assert header.version == 2.0


def test_pmx_model_keeps_constructor_compatibility() -> None:
    header = PMXHeader(2.0, "UTF-8", 0, 1, 1, 1, 1, 1, 1)

    model = PMXModel(header, "モデル", "Model", 0, 0, [], [], 0, [])

    assert model.morph_details == []


def test_read_pmx_model(pmx_fixture):
    model = read_pmx_model(pmx_fixture)

    assert model.header.version == 2.0
    assert model.model_name_jp == "テストモデル"
    assert model.model_name_en == "Test model"
    assert model.vertex_count == 1
    assert model.face_index_count == 0
    assert model.textures == ["body.png", "face.png"]
    assert model.materials == ["Body"]
    assert model.bone_count == 1
    assert model.morphs == ["あ", "い", "う", "え", "お", "口閉じ"]
    assert model.morph_details == [
        PMXMorph("あ", "A", 3, 0, 1),
        PMXMorph("い", "I", 3, 1, 1),
        PMXMorph("う", "U", 3, 2, 1),
        PMXMorph("え", "E", 3, 3, 1),
        PMXMorph("お", "O", 4, 8, 1),
        PMXMorph("口閉じ", "Close", 0, 0, 0),
    ]


def test_read_pmx_model_reads_utf16_morph_names(pmx_utf16_fixture):
    model = read_pmx_model(pmx_utf16_fixture)

    assert model.header.encoding == "UTF-16"
    assert model.morphs == ["あ", "い", "う", "え", "お", "口閉じ"]
    assert model.morph_details[0] == PMXMorph("あ", "A", 3, 0, 1)
    assert model.morph_details[-1] == PMXMorph("口閉じ", "Close", 0, 0, 0)
