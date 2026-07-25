from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


VOWELS = ("A", "I", "U", "E", "O")
DEFAULT_CLOSE_MORPH = "口閉じ"
SILENCE_FRAME_RATE = 30
DEFAULT_WHISPER_MODEL = "small"


@dataclass(frozen=True)
class VowelDetection:
    time_sec: float
    vowel: str
    confidence: float


HIRAGANA_VOWELS = {
    **dict.fromkeys("あかがさざただなはばぱまやゃらわゎ", "A"),
    **dict.fromkeys("いきぎしじちぢにひびぴみりゐぃ", "I"),
    **dict.fromkeys("うくぐすずつづぬふぶぷむゆゅるゔぅ", "U"),
    **dict.fromkeys("えけげせぜてでねへべぺめれゑぇ", "E"),
    **dict.fromkeys("おこごそぞとどのほぼぽもよょろをぉ", "O"),
}


def _clamp_confidence(value: float | None) -> float:
    if value is None:
        return 1.0
    return round(max(0.0, min(1.0, float(value))), 4)


def _katakana_to_hiragana(text: str) -> str:
    chars: list[str] = []
    for char in text:
        codepoint = ord(char)
        if 0x30A1 <= codepoint <= 0x30F6:
            chars.append(chr(codepoint - 0x60))
        else:
            chars.append(char)
    return "".join(chars)


def text_to_hiragana(text: str) -> str:
    """Convert Japanese text recognized by Whisper to hiragana."""
    if not text.strip():
        return ""

    try:
        import pykakasi
    except ImportError:
        return _katakana_to_hiragana(text)

    kakasi = pykakasi.kakasi()
    return "".join(item["hira"] for item in kakasi.convert(text))


def hiragana_to_vowels(text: str) -> list[str]:
    vowels: list[str] = []
    previous_vowel: str | None = None
    for char in _katakana_to_hiragana(text):
        if char == "ー" and previous_vowel is not None:
            vowels.append(previous_vowel)
            continue

        vowel = HIRAGANA_VOWELS.get(char)
        if vowel is None:
            continue

        vowels.append(vowel)
        previous_vowel = vowel

    return vowels


def text_to_vowels(text: str) -> list[str]:
    return hiragana_to_vowels(text_to_hiragana(text))


def _time_aligned_vowels(
    *,
    text: str,
    start: float,
    end: float,
    confidence: float | None,
) -> list[VowelDetection]:
    vowels = text_to_vowels(text)
    if not vowels:
        return []

    duration = max(0.0, end - start)
    step = duration / len(vowels) if duration > 0 else 0.0
    return [
        VowelDetection(
            time_sec=round(start + (index * step), 4),
            vowel=vowel,
            confidence=_clamp_confidence(confidence),
        )
        for index, vowel in enumerate(vowels)
    ]


def _iter_whisper_items(segment: Any) -> Iterable[tuple[str, float, float, float | None]]:
    words = getattr(segment, "words", None)
    if words:
        for word in words:
            yield (
                getattr(word, "word", ""),
                float(getattr(word, "start", getattr(segment, "start", 0.0))),
                float(getattr(word, "end", getattr(segment, "end", 0.0))),
                getattr(word, "probability", None),
            )
        return

    yield (
        getattr(segment, "text", ""),
        float(getattr(segment, "start", 0.0)),
        float(getattr(segment, "end", 0.0)),
        None,
    )


def detect_vowels(
    wav_path: str | Path,
    *,
    model_size_or_path: str = DEFAULT_WHISPER_MODEL,
    device: str = "auto",
    compute_type: str = "default",
) -> list[VowelDetection]:
    from faster_whisper import WhisperModel

    model = WhisperModel(
        model_size_or_path,
        device=device,
        compute_type=compute_type,
    )
    segments, _info = model.transcribe(
        str(Path(wav_path)),
        language="ja",
        task="transcribe",
        word_timestamps=True,
    )
    detections: list[VowelDetection] = []
    for segment in segments:
        for text, start, end, confidence in _iter_whisper_items(segment):
            print(f"{start:.2f} {end:.2f} {text} {confidence}")

            
            detections.extend(
                _time_aligned_vowels(
                    text=text,
                    start=start,
                    end=end,
                    confidence=confidence,
                )
            )

    return detections


def smooth_detections(
    detections: list[VowelDetection],
    window_size: int = 3,
) -> list[VowelDetection]:
    if window_size < 1:
        raise ValueError("window_size must be at least 1")

    if window_size == 1 or len(detections) < 2:
        return list(detections)

    half_window = window_size // 2
    smoothed: list[VowelDetection] = []
    for index, detection in enumerate(detections):
        start = max(0, index - half_window)
        stop = min(len(detections), index + half_window + 1)
        window = detections[start:stop]

        vowel_counts = Counter(item.vowel for item in window)
        vowel = max(
            vowel_counts,
            key=lambda candidate: (
                vowel_counts[candidate],
                -abs(index - next(
                    window_index
                    for window_index, item in enumerate(detections[start:stop], start)
                    if item.vowel == candidate
                )),
            ),
        )
        confidence = sum(item.confidence for item in window) / len(window)
        smoothed.append(
            VowelDetection(
                time_sec=detection.time_sec,
                vowel=vowel,
                confidence=round(confidence, 4),
            )
        )

    return smoothed


def _frame_number(time_sec: float, frame_rate: int = SILENCE_FRAME_RATE) -> int:
    return max(0, round(time_sec * frame_rate))


def insert_silence_frames(
    detections: list[VowelDetection],
    *,
    silence_threshold_sec: float = 0.12,
    close_morph: str = DEFAULT_CLOSE_MORPH,
    frame_rate: int = SILENCE_FRAME_RATE,
) -> list[VowelDetection]:
    if silence_threshold_sec < 0:
        raise ValueError("silence_threshold_sec must be non-negative")
    if frame_rate < 1:
        raise ValueError("frame_rate must be at least 1")

    if len(detections) < 2:
        return list(detections)

    ordered_detections = sorted(detections, key=lambda detection: detection.time_sec)
    frames: list[VowelDetection] = [ordered_detections[0]]
    for previous, current in zip(
        ordered_detections,
        ordered_detections[1:],
    ):
        if current.time_sec - previous.time_sec > silence_threshold_sec:
            previous_frame = _frame_number(previous.time_sec, frame_rate)
            current_frame = _frame_number(current.time_sec, frame_rate)
            for frame_number in range(previous_frame + 1, current_frame):
                frames.append(
                    VowelDetection(
                        time_sec=round(frame_number / frame_rate, 4),
                        vowel=close_morph,
                        confidence=1.0,
                    )
                )

        frames.append(current)

    return frames


def merge_consecutive_vowels(
    detections: list[VowelDetection],
) -> list[VowelDetection]:
    if not detections:
        return []

    merged: list[VowelDetection] = []
    run: list[VowelDetection] = [detections[0]]
    for detection in detections[1:]:
        if detection.vowel == run[-1].vowel:
            run.append(detection)
            continue

        merged.append(max(run, key=lambda item: item.confidence))
        run = [detection]

    merged.append(max(run, key=lambda item: item.confidence))
    return merged


def write_csv(detections: list[VowelDetection], csv_path: str | Path) -> None:
    import pandas as pd

    rows = [
        {
            "time_sec": detection.time_sec,
            "vowel": detection.vowel,
            "confidence": detection.confidence,
        }
        for detection in detections
    ]
    dataframe = pd.DataFrame(rows, columns=["time_sec", "vowel", "confidence"])
    dataframe.to_csv(Path(csv_path), index=False, encoding="utf-8")