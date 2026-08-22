from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


VOWELS = ("A", "I", "U", "E", "O")
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
    if not vowels or end <= start:
        return []

    duration = end - start
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


def _segment_word_vowels(
    segment: Any,
) -> list[VowelDetection]:
    """Convert a Whisper segment using the full text for reading context."""
    words = getattr(segment, "words", None)

    if not words:
        text = getattr(segment, "text", "")
        start = float(getattr(segment, "start", 0.0))
        end = float(getattr(segment, "end", 0.0))
        return _time_aligned_vowels(
            text=text,
            start=start,
            end=end,
            confidence=None,
        )

    valid_words = []
    for word in words:
        text = getattr(word, "word", "")
        start = float(getattr(word, "start", getattr(segment, "start", 0.0)))
        end = float(getattr(word, "end", getattr(segment, "end", 0.0)))
        confidence = getattr(word, "probability", None)

        if not text.strip() or end <= start:
            continue

        valid_words.append((text, start, end, confidence))

    if not valid_words:
        return []

    full_text = "".join(item[0] for item in valid_words)
    full_vowels = text_to_vowels(full_text)

    if not full_vowels:
        return []

    detections: list[VowelDetection] = []
    vowel_index = 0

    for text, start, end, confidence in valid_words:
        # Use the word itself only to determine how many vowel slots it
        # contributes. The actual vowel sequence comes from the full text
        # so Japanese reading context is preserved.
        word_vowel_count = len(text_to_vowels(text))

        if word_vowel_count <= 0:
            continue

        word_vowels = full_vowels[vowel_index : vowel_index + word_vowel_count]
        vowel_index += len(word_vowels)

        if not word_vowels:
            break

        duration = end - start
        step = duration / len(word_vowels)

        for index, vowel in enumerate(word_vowels):
            detections.append(
                VowelDetection(
                    time_sec=round(start + (index * step), 4),
                    vowel=vowel,
                    confidence=_clamp_confidence(confidence),
                )
            )

    return detections


def _is_audio_silent(wav_path: str | Path, start: float, end: float, threshold: float = 0.001) -> bool:
    import librosa
    import numpy as np

    if end <= start:
        return True
    try:
        y, _sr = librosa.load(str(wav_path), sr=None, mono=True, offset=start, duration=end - start)
    except (OSError, EOFError, ValueError):
        return False
    if len(y) == 0:
        return True
    rms = float(np.sqrt(np.mean(y ** 2)))
    return rms < threshold


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
        words = getattr(segment, "words", None)

        if words:
            for word in words:
                text = getattr(word, "word", "")
                start = float(getattr(word, "start", getattr(segment, "start", 0.0)))
                end = float(getattr(word, "end", getattr(segment, "end", 0.0)))
                confidence = getattr(word, "probability", None)
                print(f"{start:.2f} {end:.2f} {text} {confidence}")

        segment_words = [
            word
            for word in (words or [])
            if getattr(word, "word", "").strip()
            and float(getattr(word, "end", getattr(segment, "end", 0.0)))
            > float(getattr(word, "start", getattr(segment, "start", 0.0)))
        ]

        if segment_words:
            if all(
                _is_audio_silent(
                    wav_path,
                    float(getattr(word, "start", getattr(segment, "start", 0.0))),
                    float(getattr(word, "end", getattr(segment, "end", 0.0))),
                )
                for word in segment_words
            ):
                continue

            detections.extend(_segment_word_vowels(segment))
            continue

        text = getattr(segment, "text", "")
        start = float(getattr(segment, "start", 0.0))
        end = float(getattr(segment, "end", 0.0))
        print(f"{start:.2f} {end:.2f} {text} None")

        if _is_audio_silent(wav_path, start, end):
            continue

        detections.extend(
            _time_aligned_vowels(
                text=text,
                start=start,
                end=end,
                confidence=None,
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