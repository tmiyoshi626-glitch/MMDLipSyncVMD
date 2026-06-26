from __future__ import annotations

import argparse
from pathlib import Path

from mmd_lip_sync_vmd import __version__
from mmd_lip_sync_vmd.detector import (
    DEFAULT_CLOSE_MORPH,
    DEFAULT_WHISPER_MODEL,
    detect_vowels,
    insert_silence_frames,
    merge_consecutive_vowels,
    smooth_detections,
    write_csv,
)
from mmd_lip_sync_vmd.vmd_writer import write_vmd


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be greater than or equal to 1")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to 0")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmd-lip-sync-vmd",
        description="Generate MMD lip-sync VMD motion data.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "input_wav",
        nargs="?",
        type=Path,
        help="Path to the input WAV vocal file.",
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        type=Path,
        help="Path for the output CSV or VMD file.",
    )
    parser.add_argument(
        "--whisper-model",
        default=DEFAULT_WHISPER_MODEL,
        help=f"faster-whisper model size or path. Default: {DEFAULT_WHISPER_MODEL}",
    )
    parser.add_argument(
        "--whisper-device",
        default="auto",
        help="Device passed to faster-whisper. Default: auto",
    )
    parser.add_argument(
        "--whisper-compute-type",
        default="default",
        help="Compute type passed to faster-whisper. Default: default",
    )
    parser.add_argument(
        "--smooth-window",
        type=positive_int,
        default=3,
        help="Centered smoothing window size. Use 1 to disable smoothing. Default: 3",
    )
    parser.add_argument(
        "--silence-threshold",
        type=non_negative_float,
        default=0.12,
        help="Minimum gap in seconds before inserting mouth-close frames. Default: 0.12",
    )
    parser.add_argument(
        "--close-morph",
        default=DEFAULT_CLOSE_MORPH,
        help=f"MMD morph name for generated mouth-close frames. Default: {DEFAULT_CLOSE_MORPH}",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.input_wav is None or args.output_path is None:
        parser.error("input_wav and output_path are required unless using --version")

    detections = detect_vowels(
        args.input_wav,
        model_size_or_path=args.whisper_model,
        device=args.whisper_device,
        compute_type=args.whisper_compute_type,
    )
    detections = smooth_detections(detections, window_size=args.smooth_window)
    print("close_morph =", repr(args.close_morph))
    detections = insert_silence_frames(
        detections,
        silence_threshold_sec=args.silence_threshold,
        close_morph=args.close_morph,
    )
    detections = merge_consecutive_vowels(detections)

    output_suffix = args.output_path.suffix.lower()
    if output_suffix == ".vmd":
        write_vmd(detections, args.output_path, close_morph=args.close_morph)
    elif output_suffix == ".csv":
        for d in detections[:20]:
            print(d)

        write_csv(detections, args.output_path)
    else:
        parser.error("output_path must end with .vmd or .csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
