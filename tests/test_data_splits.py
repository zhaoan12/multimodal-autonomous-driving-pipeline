from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmdrive_pipeline.data import create_dataset_split, load_dataset_manifest


class DataSplitTests(unittest.TestCase):
    def test_split_generation_is_deterministic(self) -> None:
        scene_paths = load_dataset_manifest("data/sample/scene_manifest.json")

        split = create_dataset_split(scene_paths, train_ratio=0.0, val_ratio=0.0)

        self.assertEqual(split.train, [])
        self.assertEqual(split.val, [])
        self.assertEqual(len(split.test), 1)
        self.assertTrue(split.test[0].endswith("scene_0001.json"))


if __name__ == "__main__":
    unittest.main()
