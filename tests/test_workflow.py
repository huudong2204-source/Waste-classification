"""Kiểm tra chia nhóm ảnh trùng và luồng giao diện mà không mở webcam vật lý."""
import sys
import tempfile
import time
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import app
from material import BEST, LABELS
from train import split_data


class WorkflowTests(unittest.TestCase):
    def test_duplicates_stay_together(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for i, label in enumerate(LABELS):
                (root / label).mkdir()
                for j in range(10):
                    Image.new("RGB", (10, 10), (20*i, 10*j, 0)).save(root / label / f"{j}.png")
            Image.open(root / "plastic/0.png").save(root / "plastic/copy.bmp")
            rows = split_data(root, 42)
            self.assertEqual(rows, split_data(root, 42))
            location = {r["path"]: key for key, part in rows.items() for r in part}
            self.assertEqual(location["plastic/0.png"], location["plastic/copy.bmp"])
            for part in rows.values():
                self.assertEqual({r["target"] for r in part}, set(range(len(LABELS))))

    def test_station_controls(self):
        root = tk.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        with tempfile.TemporaryDirectory() as tmp:
            station = app.Station(root, Path(tmp) / "missing.pt")
            self.assertIn("Thiếu checkpoint", station.status.cget("text"))

            def until(predicate):
                deadline = time.monotonic() + 15
                while not predicate() and time.monotonic() < deadline:
                    root.update()
                    time.sleep(.01)
                self.assertTrue(predicate(), station.status.cget("text"))

            station.toggle.invoke()
            until(lambda: station.worker is None)
            self.assertIn("best.pt", station.status.cget("text"))
            station.checkpoint = BEST
            with patch.object(app, "open_camera", side_effect=OSError("Không mở được camera thử")):
                station.toggle.invoke()
                until(lambda: station.worker is None)
                self.assertIn("Không mở được camera", station.status.cget("text"))

            class FakeCamera:
                released = False

                def read(self):
                    time.sleep(.025)
                    return True, np.full((240, 320, 3), 236, np.uint8)

                def release(self):
                    self.released = True

            camera = FakeCamera()
            with patch.object(app, "open_camera", return_value=(camera, camera.read()[1])), patch.object(app, "PROJECT", Path(tmp)):
                station.toggle.invoke()
                until(lambda: str(station.learn.cget("state")) == "normal")
                station.learn.invoke()
                until(lambda: station.status.cget("text") == "Đưa một vật vào khung")
                station.toggle.invoke()
                until(lambda: station.worker is None)
                self.assertTrue(camera.released)
                self.assertTrue((Path(tmp) / "outputs/webcam/sessions.csv").is_file())
                self.assertEqual(station.timing.cget("text"), "")


if __name__ == "__main__":
    unittest.main()
