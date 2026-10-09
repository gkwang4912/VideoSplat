import json
import math
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import (
    camera_signature,
    opencv_c2w_to_opengl,
    rotation_matrix_to_qvec,
    scaled_camera_for_image,
    write_json,
)


class CameraConversionTests(unittest.TestCase):
    def test_identity_rotation(self):
        qvec = rotation_matrix_to_qvec([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertEqual(qvec, (1.0, 0.0, 0.0, 0.0))

    def test_z_rotation_90_degrees(self):
        qvec = rotation_matrix_to_qvec([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertAlmostEqual(qvec[0], math.sqrt(0.5), places=12)
        self.assertAlmostEqual(qvec[3], math.sqrt(0.5), places=12)

    def test_intrinsic_signature(self):
        camera = {"width": 704, "height": 1344, "fx": 10, "fy": 11, "cx": 352, "cy": 672}
        self.assertEqual(camera_signature(camera), (704, 1344, 10.0, 11.0, 352.0, 672.0))

    def test_opencv_to_opengl_c2w(self):
        matrix = [
            [1.0, 0.0, 0.0, 4.0],
            [0.0, 1.0, 0.0, 5.0],
            [0.0, 0.0, 1.0, 6.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        converted = opencv_c2w_to_opengl(matrix)
        self.assertEqual(converted[0], [1.0, -0.0, -0.0, 4.0])
        self.assertEqual(converted[1], [0.0, -1.0, -0.0, 5.0])
        self.assertEqual(converted[2], [0.0, -0.0, -1.0, 6.0])
        self.assertEqual(converted[3], [0.0, -0.0, -0.0, 1.0])

    def test_camera_intrinsics_scale_with_video_resolution(self):
        camera = {
            "width": 704,
            "height": 1344,
            "fx": 100.0,
            "fy": 110.0,
            "cx": 352.0,
            "cy": 672.0,
            "intrinsic_matrix": [[100.0, 0.0, 352.0], [0.0, 110.0, 672.0], [0.0, 0.0, 1.0]],
        }
        scaled = scaled_camera_for_image(camera, 1408, 2688)
        self.assertEqual((scaled["fx"], scaled["fy"], scaled["cx"], scaled["cy"]), (200.0, 220.0, 704.0, 1344.0))
        self.assertEqual(scaled["intrinsic_matrix"][0], [200.0, 0.0, 704.0])

    def test_json_is_utf8_and_atomic(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "camera.json"
            write_json(path, {"label": "相機", "value": 1})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["label"], "相機")
            self.assertFalse(path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
