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
