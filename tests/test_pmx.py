from mmd_lip_sync_vmd.pmx import read_pmx_header, read_pmx_model


def test_read_pmx_header(pmx_fixture):
    header = read_pmx_header(pmx_fixture)

    assert header.version == 2.0


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


def test_read_pmx_model_reads_utf16_morph_names(pmx_utf16_fixture):
    model = read_pmx_model(pmx_utf16_fixture)

    assert model.header.encoding == "UTF-16"
    assert model.morphs == ["あ", "い", "う", "え", "お", "口閉じ"]
