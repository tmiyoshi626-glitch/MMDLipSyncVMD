from mmd_lip_sync_vmd.detector import VowelDetection, merge_consecutive_vowels


def test_merge_consecutive_vowels_does_not_merge_across_time_gap() -> None:
    detections = [
        VowelDetection(time_sec=65.44, vowel="A", confidence=1.0, end_sec=66.12),
        VowelDetection(time_sec=77.32, vowel="A", confidence=1.0, end_sec=77.96),
    ]

    merged = merge_consecutive_vowels(detections)

    assert merged == detections
