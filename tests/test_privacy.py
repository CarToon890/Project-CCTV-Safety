"""Unit checks for the local face-blur primitive."""

import cv2
import numpy as np

from cctv_safety.privacy import YuNetFaceAnonymizer, anonymize


class FixedFaceDetector:
    def detect(self, frame):
        assert frame.shape == (64, 64, 3)
        return [[16, 16, 40, 40]]

    @staticmethod
    def blur(frame, boxes):
        from cctv_safety.privacy import YuNetFaceAnonymizer

        return YuNetFaceAnonymizer.blur(frame, boxes)


def test_anonymize_blurs_copy_without_changing_model_input():
    source = np.zeros((64, 64, 3), dtype=np.uint8)
    source[22:35, 22:35] = 255
    before = source.copy()

    safe, boxes = anonymize(FixedFaceDetector(), source)

    assert boxes == [[16, 16, 40, 40]]
    assert np.array_equal(source, before)
    assert safe is not source
    assert not np.array_equal(safe[16:40, 16:40], source[16:40, 16:40])
    assert np.array_equal(safe[:8, :8], source[:8, :8])


def test_yunet_only_resets_input_size_when_frame_shape_changes(tmp_path, monkeypatch):
    class Detector:
        def __init__(self):
            self.sizes = []

        def setInputSize(self, size):
            self.sizes.append(size)

        def detect(self, _frame):
            return None, None

    detector = Detector()
    monkeypatch.setattr(cv2.FaceDetectorYN, "create", lambda *_args: detector)
    (tmp_path / "face_detection_yunet_2023mar.onnx").write_bytes(b"stub")
    anonymizer = YuNetFaceAnonymizer(tmp_path)
    anonymizer.detect(np.zeros((120, 160, 3), dtype=np.uint8))
    anonymizer.detect(np.zeros((120, 160, 3), dtype=np.uint8))
    anonymizer.detect(np.zeros((240, 320, 3), dtype=np.uint8))
    assert detector.sizes == [(160, 120), (320, 240)]
