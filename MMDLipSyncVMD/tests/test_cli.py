from pathlib import Path

import pytest

from mmd_lip_sync_vmd import cli
from mmd_lip_sync_vmd.detector import VowelDetection


def test_parser_defaults_smooth_window_to_three() -> None:
    args = cli.build_parser().parse_args(["vocal.wav", "output.vmd"])

    assert args.input_wav == Path("vocal.wav")
    assert args.output_path == Path("output.vmd")
    assert args.whisper_model == "small"
    assert args.whisper_device == "auto"
    assert args.whisper_compute_type == "default"
    assert args.smooth_window == 3
    assert args.silence_threshold == 0.12
    assert args.close_morph == "口閉じ"


def test_parser_allows_smooth_window_one_to_disable_smoothing() -> None:
    args = cli.build_parser().parse_args(
        ["vocal.wav", "output.csv", "--smooth-window", "1"]
    )

    assert args.smooth_window == 1


def test_parser_rejects_smooth_window_less_than_one() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            ["vocal.wav", "output.vmd", "--smooth-window", "0"]
        )


def test_main_inserts_silence_after_smoothing_before_writing_csv(
    monkeypatch,
    tmp_path,
) -> None:
    input_wav = tmp_path / "vocal.wav"
    output_csv = tmp_path / "output.csv"
    detected = [VowelDetection(time_sec=0.0, vowel="I", confidence=0.5)]
    smoothed = [VowelDetection(time_sec=0.0, vowel="A", confidence=0.75)]
    with_silence = [
        VowelDetection(time_sec=0.0, vowel="A", confidence=0.75),
        VowelDetection(time_sec=0.1, vowel="close", confidence=1.0),
    ]
    merged = [VowelDetection(time_sec=0.0, vowel="A", confidence=0.75)]
    calls: list[str] = []

    def fake_detect_vowels(*args, **kwargs):
        calls.append("detect")
        assert args == (input_wav,)
        assert kwargs == {
            "model_size_or_path": "tiny",
            "device": "cpu",
            "compute_type": "int8",
        }
        return detected

    def fake_smooth_detections(detections, *, window_size):
        calls.append("smooth")
        assert detections == detected
        assert window_size == 5
        return smoothed

    def fake_insert_silence_frames(detections, *, silence_threshold_sec, close_morph):
        calls.append("silence")
        assert detections == smoothed
        assert silence_threshold_sec == 0.2
        assert close_morph == "close"
        return with_silence

    def fake_merge_consecutive_vowels(detections):
        calls.append("merge")
        assert detections == with_silence
        return merged

    def fake_write_csv(detections, csv_path):
        calls.append("write_csv")
        assert detections == merged
        assert csv_path == output_csv

    monkeypatch.setattr(cli, "detect_vowels", fake_detect_vowels)
    monkeypatch.setattr(cli, "smooth_detections", fake_smooth_detections)
    monkeypatch.setattr(cli, "insert_silence_frames", fake_insert_silence_frames)
    monkeypatch.setattr(cli, "merge_consecutive_vowels", fake_merge_consecutive_vowels)
    monkeypatch.setattr(cli, "write_csv", fake_write_csv)
    monkeypatch.setattr(
        "sys.argv",
        [
            "mmd-lip-sync-vmd",
            str(input_wav),
            str(output_csv),
            "--whisper-model",
            "tiny",
            "--whisper-device",
            "cpu",
            "--whisper-compute-type",
            "int8",
            "--smooth-window",
            "5",
            "--silence-threshold",
            "0.2",
            "--close-morph",
            "close",
        ],
    )

    assert cli.main() == 0
    assert calls == ["detect", "smooth", "silence", "merge", "write_csv"]


def test_main_passes_close_morph_to_vmd_writer(monkeypatch, tmp_path) -> None:
    input_wav = tmp_path / "vocal.wav"
    output_vmd = tmp_path / "output.vmd"
    with_silence = [VowelDetection(time_sec=0.0, vowel="close", confidence=1.0)]
    merged = [VowelDetection(time_sec=0.0, vowel="close", confidence=1.0)]

    monkeypatch.setattr(
        cli,
        "detect_vowels",
        lambda *args, **kwargs: [
            VowelDetection(time_sec=0.0, vowel="A", confidence=0.75)
        ],
    )
    monkeypatch.setattr(
        cli,
        "smooth_detections",
        lambda detections, *, window_size: detections,
    )
    monkeypatch.setattr(
        cli,
        "insert_silence_frames",
        lambda detections, *, silence_threshold_sec, close_morph: with_silence,
    )
    monkeypatch.setattr(
        cli,
        "merge_consecutive_vowels",
        lambda detections: merged,
    )

    def fake_write_vmd(detections, vmd_path, *, close_morph):
        assert detections == merged
        assert vmd_path == output_vmd
        assert close_morph == "close"

    monkeypatch.setattr(cli, "write_vmd", fake_write_vmd)
    monkeypatch.setattr(
        "sys.argv",
        [
            "mmd-lip-sync-vmd",
            str(input_wav),
            str(output_vmd),
            "--close-morph",
            "close",
        ],
    )

    assert cli.main() == 0
