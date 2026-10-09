from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.request import Request, urlopen

import app


ROOT = Path(__file__).resolve().parents[1]


def make_fixture(root: Path) -> Path:
    input_dir = root / "input"
    path_dir = input_dir / "paths" / "01_front_to_left"
    images_dir = path_dir / "images"
    images_dir.mkdir(parents=True)
    frames = []
    for index in range(3):
        name = f"{index:03d}.png"
        (images_dir / name).write_bytes(b"fixture")
        frames.append(
            {
                "image": name,
                "local_path_angle_degrees": index * 45.0,
                "camera_position": [float(index), 0.0, 3.0],
                "camera_forward": [0.0, 0.0, 1.0],
                "camera_up": [0.0, -1.0, 0.0],
                "subject_center": [0.0, 0.0, 0.0],
                "sphere_radius": 3.0,
                "width": 64,
                "height": 64,
            }
        )
    (path_dir / "cameras.json").write_text(
        json.dumps({"frames": frames}), encoding="utf-8"
    )
    graph = {
        "nodes": [],
        "edges": [
            {
                "id": "01_front_to_left",
                "start_anchor": "front",
                "end_anchor": "left",
                "degrees_per_frame": 45.0,
                "frame_count": 3,
                "directory": "paths/01_front_to_left",
            }
        ],
    }
    (input_dir / "path_graph.json").write_text(
        json.dumps(graph), encoding="utf-8"
    )
    header = (
        b"ply\nformat binary_little_endian 1.0\n"
        b"comment sh degree: 3\nelement vertex 6\n"
        b"property float x\nproperty float y\nproperty float z\nend_header\n"
    )
    (input_dir / "fixture.ply").write_bytes(header + bytes(256))
    return input_dir


class DatasetIndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.input_dir = make_fixture(Path(cls.temp.name))
        cls.dataset = app.DatasetIndex.load(cls.input_dir)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_discovers_paths_and_frames(self) -> None:
        stats = self.dataset.manifest["stats"]
        self.assertEqual(stats["path_count"], 1)
        self.assertEqual(stats["frame_counts"], [3])
        self.assertEqual(stats["total_camera_samples"], 3)

    def test_all_asset_urls_resolve(self) -> None:
        for path in self.dataset.manifest["paths"]:
            for frame in path["frames"]:
                relative = frame["image_url"].removeprefix("/input/")
                self.assertTrue((self.input_dir / relative).is_file())

    def test_gaussian_metadata(self) -> None:
        model = self.dataset.manifest["model"]
        self.assertEqual(model["format"], "binary_little_endian")
        self.assertEqual(model["vertex_count"], 6)
        self.assertEqual(model["sh_degree"], 3)

    def test_partial_config_merges_with_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "config.json"
            config_path.write_text(
                json.dumps({"ply_mode": {"model_scale": 1.25}}),
                encoding="utf-8",
            )
            dataset = app.DatasetIndex.load(self.input_dir, config_path)
        self.assertEqual(dataset.manifest["config"]["ply_mode"]["model_scale"], 1.25)
        self.assertEqual(
            dataset.manifest["config"]["ply_mode"]["camera_point_size_px"], 3.0
        )


class HttpServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.input_dir = make_fixture(Path(cls.temp.name))
        cls.server = app.create_server(
            port=0,
            input_dir=cls.input_dir,
            dist_dir=ROOT / "frontend" / "dist",
        )
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)
        cls.temp.cleanup()

    def test_health_endpoint(self) -> None:
        with urlopen(f"{self.base_url}/api/health", timeout=5) as response:
            payload = json.load(response)
        self.assertEqual(payload, {"status": "ok", "paths": 1, "samples": 3})

    def test_ply_supports_byte_ranges(self) -> None:
        model_url = self.server.dataset.manifest["model"]["url"]
        request = Request(
            f"{self.base_url}{model_url}", headers={"Range": "bytes=0-127"}
        )
        with urlopen(request, timeout=5) as response:
            body = response.read()
            self.assertEqual(response.status, 206)
            self.assertEqual(response.headers["Accept-Ranges"], "bytes")
            self.assertEqual(len(body), 128)
            self.assertTrue(body.startswith(b"ply\n"))


if __name__ == "__main__":
    unittest.main()
