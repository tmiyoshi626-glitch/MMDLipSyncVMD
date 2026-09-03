from mmd_lip_sync_vmd.detector import VowelDetection, smooth_detections


def test_smooth_detections_preserves_end_sec() -> None:
    detections = [
        VowelDetection(time_sec=0.0, vowel="A", confidence=0.6, end_sec=0.05),
        VowelDetection(time_sec=0.1, vowel="I", confidence=0.9, end_sec=0.15),
        VowelDetection(time_sec=0.2, vowel="A", confidence=0.3, end_sec=0.25),
    ]

    smoothed = smooth_detections(detections, window_size=3)

    assert smoothed[1].end_sec == 0.15
