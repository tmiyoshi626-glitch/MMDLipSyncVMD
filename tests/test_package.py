from mmd_lip_sync_vmd import __version__
from types import SimpleNamespace

import pytest

import mmd_lip_sync_vmd.detector as detector
from mmd_lip_sync_vmd.detector import (
    VOWELS,
    VowelDetection,
    detect_vowels,
    hiragana_to_vowels,
    merge_consecutive_vowels,
    smooth_detections,
    text_to_hiragana,
    text_to_vowels,
    write_csv,
)


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_text_to_hiragana_converts_japanese_text() -> None:
    hiragana = text_to_hiragana("歌う")

    assert hiragana == "うたう"


def test_text_to_vowels_handles_kanji_with_context() -> None:
    assert text_to_hiragana("過ぎる") == "すぎる"
    assert text_to_vowels("過ぎる") == ["U", "I", "U"]


def test_hiragana_to_vowels_maps_kana_to_supported_vowels() -> None:
    vowels = hiragana_to_vowels("あいうえおんっきゃー")

    assert vowels == ["A", "I", "U", "E", "O", "I", "A", "A"]
    assert set(vowels).issubset(VOWELS)


def test_text_to_vowels_converts_recognized_text_to_vowels(monkeypatch) -> None:
    monkeypatch.setattr(detector, "text_to_hiragana", lambda text: "てすと")

    assert text_to_vowels("テスト") == ["E", "U", "O"]


def test_segment_word_vowels_uses_full_segment_context(monkeypatch) -> None:
    segment = SimpleNamespace(
        start=5.42,
        end=6.12,
        text="過ぎる",
        words=[
            SimpleNamespace(start=5.42, end=5.70, word="過", probability=0.99),
            SimpleNamespace(start=5.70, end=5.88, word="ぎ", probability=0.99),
            SimpleNamespace(start=5.88, end=6.12, word="る", probability=0.99),
        ],
    )

    assert detector._segment_word_vowels(segment) == [
        VowelDetection(time_sec=5.42, vowel="U", confidence=0.99, end_sec=5.70),
        VowelDetection(time_sec=5.70, vowel="I", confidence=0.99, end_sec=5.88),
        VowelDetection(time_sec=5.88, vowel="U", confidence=0.99, end_sec=6.12),
    ]


def test_detect_vowels_uses_segment_context_for_split_words(monkeypatch, tmp_path) -> None:
    input_wav = tmp_path / "vocal.wav"
    input_wav.write_bytes(b"")

    class FakeWhisperModel:
        def __init__(self, model_size_or_path, *, device, compute_type):
            pass

        def transcribe(self, wav_path, **kwargs):
            return (
                [
                    SimpleNamespace(
                        start=5.42,
                        end=6.12,
                        text="過ぎる",
                        words=[
                            SimpleNamespace(
                                start=5.42,
                                end=5.70,
                                word="過",
                                probability=0.99,
                            ),
                            SimpleNamespace(
                                start=5.70,
                                end=5.88,
                                word="ぎ",
                                probability=0.99,
                            ),
                            SimpleNamespace(
                                start=5.88,
                                end=6.12,
                                word="る",
                                probability=0.99,
                            ),
                        ],
                    )
                ],
                SimpleNamespace(),
            )

    monkeypatch.setitem(
        __import__("sys").modules,
        "faster_whisper",
        SimpleNamespace(WhisperModel=FakeWhisperModel),
    )
    monkeypatch.setattr(
        detector,
        "_is_audio_silent",
        lambda wav_path, start, end: False,
    )

    assert detect_vowels(
        input_wav,
        model_size_or_path="tiny",
        device="cpu",
        compute_type="int8",
    ) == [
        VowelDetection(time_sec=5.42, vowel="U", confidence=0.99, end_sec=5.70),
        VowelDetection(time_sec=5.70, vowel="I", confidence=0.99, end_sec=5.88),
        VowelDetection(time_sec=5.88, vowel="U", confidence=0.99, end_sec=6.12),
    ]


def test_detect_vowels_ignores_zero_duration_whisper_words(monkeypatch, tmp_path) -> None:
    input_wav = tmp_path / "vocal.wav"
    input_wav.write_bytes(b"")

    class FakeWhisperModel:
        def __init__(self, model_size_or_path, *, device, compute_type):
            pass

        def transcribe(self, wav_path, **kwargs):
            return ([SimpleNamespace(start=1.0, end=1.0, text="あ", words=[SimpleNamespace(start=1.0, end=1.0, word="あ", probability=0.9)])], SimpleNamespace())

    monkeypatch.setitem(__import__("sys").modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeWhisperModel))

    monkeypatch.setattr(detector, "_is_audio_silent", lambda wav_path, start, end: False)
    assert detect_vowels(input_wav, model_size_or_path="tiny", device="cpu", compute_type="int8") == []


def test_detect_vowels_ignores_whisper_results_in_silence(monkeypatch, tmp_path) -> None:
    input_wav = tmp_path / "vocal.wav"
    input_wav.write_bytes(b"")

    class FakeWhisperModel:
        def __init__(self, model_size_or_path, *, device, compute_type):
            pass

        def transcribe(self, wav_path, **kwargs):
            return ([SimpleNamespace(start=2.0, end=3.0, text="あ", words=[SimpleNamespace(start=2.0, end=3.0, word="あ", probability=0.9)])], SimpleNamespace())

    monkeypatch.setitem(__import__("sys").modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeWhisperModel))
    monkeypatch.setattr(detector, "_is_audio_silent", lambda wav_path, start, end: True)

    assert detect_vowels(input_wav, model_size_or_path="tiny", device="cpu", compute_type="int8") == []




def test_detect_vowels_uses_whisper_word_timestamps(monkeypatch, tmp_path) -> None:
    input_wav = tmp_path / "vocal.wav"
    input_wav.write_bytes(b"")
    transcribe_calls = []

    class FakeWhisperModel:
        def __init__(self, model_size_or_path, *, device, compute_type):
            assert model_size_or_path == "tiny"
            assert device == "cpu"
            assert compute_type == "int8"

        def transcribe(self, wav_path, **kwargs):
            transcribe_calls.append((wav_path, kwargs))
            return (
                [
                    SimpleNamespace(
                        start=0.0,
                        end=1.0,
                        text="unused",
                        words=[
                            SimpleNamespace(
                                start=0.2,
                                end=0.8,
                                word="あい",
                                probability=0.75,
                            )
                        ],
                    )
                ],
                SimpleNamespace(),
            )

    monkeypatch.setitem(
        __import__("sys").modules,
        "faster_whisper",
        SimpleNamespace(WhisperModel=FakeWhisperModel),
    )

    monkeypatch.setattr(detector, "_is_audio_silent", lambda wav_path, start, end: False)
    assert detect_vowels(
        input_wav,
        model_size_or_path="tiny",
        device="cpu",
        compute_type="int8",
    ) == [
        VowelDetection(time_sec=0.2, vowel="A", confidence=0.75, end_sec=0.5),
        VowelDetection(time_sec=0.5, vowel="I", confidence=0.75, end_sec=0.8),
    ]
    assert transcribe_calls == [
        (
            str(input_wav),
            {
                "language": "ja",
                "task": "transcribe",
                "word_timestamps": True,
            },
        )
    ]


def test_merge_consecutive_vowels_keeps_highest_confidence_in_run() -> None:
    detections = [
        VowelDetection(time_sec=0.0, vowel="A", confidence=0.4),
        VowelDetection(time_sec=0.1, vowel="A", confidence=0.9),
        VowelDetection(time_sec=0.2, vowel="I", confidence=0.5),
    ]

    assert merge_consecutive_vowels(detections) == [
        VowelDetection(time_sec=0.1, vowel="A", confidence=0.9),
        VowelDetection(time_sec=0.2, vowel="I", confidence=0.5),
    ]


def test_merge_consecutive_vowels_preserves_separate_runs() -> None:
    detections = [
        VowelDetection(time_sec=0.0, vowel="A", confidence=0.4),
        VowelDetection(time_sec=0.1, vowel="I", confidence=0.5),
        VowelDetection(time_sec=0.2, vowel="A", confidence=0.9),
    ]

    assert merge_consecutive_vowels(detections) == detections


def test_write_csv_uses_required_columns(tmp_path) -> None:
    output_path = tmp_path / "vowels.csv"
    write_csv(
        [
            VowelDetection(time_sec=0.0, vowel="A", confidence=0.9),
            VowelDetection(time_sec=0.04, vowel="I", confidence=0.8),
        ],
        output_path,
    )

    assert output_path.read_text().splitlines() == [
        "time_sec,vowel,confidence",
        "0.0,A,0.9",
        "0.04,I,0.8",
    ]


def test_smooth_detections_uses_majority_vowel_and_average_confidence() -> None:
    detections = [
        VowelDetection(time_sec=0.0, vowel="A", confidence=0.6),
        VowelDetection(time_sec=0.1, vowel="I", confidence=0.9),
        VowelDetection(time_sec=0.2, vowel="A", confidence=0.3),
    ]

    smoothed = smooth_detections(detections, window_size=3)

    assert smoothed[1] == VowelDetection(
        time_sec=0.1,
        vowel="A",
        confidence=0.6,
    )


def test_smooth_detections_rejects_invalid_window_size() -> None:
    with pytest.raises(ValueError, match="window_size"):
        smooth_detections([], window_size=0)


def test_segment_word_vowels_distributes_multiple_vowels_within_one_word(monkeypatch) -> None:
    segment = SimpleNamespace(
        start=4.34,
        end=4.90,
        text="ように",
        words=[
            SimpleNamespace(
                start=4.34,
                end=4.90,
                word="ように",
                probability=0.99,
            ),
        ],
    )

    assert detector._segment_word_vowels(segment) == [
        VowelDetection(time_sec=4.34, vowel="O", confidence=0.99, end_sec=4.5267),
        VowelDetection(time_sec=4.5267, vowel="U", confidence=0.99, end_sec=4.7133),
        VowelDetection(time_sec=4.7133, vowel="I", confidence=0.99, end_sec=4.9),
    ]


def test_segment_word_vowels_keeps_vowels_inside_each_word(monkeypatch) -> None:
    segment = SimpleNamespace(
        start=4.34,
        end=5.42,
        text="ようにも",
        words=[
            SimpleNamespace(
                start=4.34,
                end=4.90,
                word="ように",
                probability=0.99,
            ),
            SimpleNamespace(
                start=4.90,
                end=5.42,
                word="も",
                probability=0.99,
            ),
        ],
    )

    assert detector._segment_word_vowels(segment) == [
        VowelDetection(time_sec=4.34, vowel="O", confidence=0.99, end_sec=4.5267),
        VowelDetection(time_sec=4.5267, vowel="U", confidence=0.99, end_sec=4.7133),
        VowelDetection(time_sec=4.7133, vowel="I", confidence=0.99, end_sec=4.9),
        VowelDetection(time_sec=4.90, vowel="O", confidence=0.99, end_sec=5.42),
    ]
