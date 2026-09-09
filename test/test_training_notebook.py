"""Notebook Colab : syntax and data preparation, without pip installs or GPU."""
import ast
from collections import Counter
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
import zipfile

import cv2
import numpy as np
from PIL import Image
import yaml

try:
    import pandas as pd
except ImportError:
    pd = None


ROOT = Path(__file__).resolve().parents[1]


class TrainingNotebookTests(unittest.TestCase):
    def setUp(self):
        self.notebook = json.loads((ROOT / "ENTRAINEMENT_YOLO.ipynb").read_text())
        self.code = {cell["id"]: "".join(cell["source"]) for cell in self.notebook["cells"]
                     if cell["cell_type"] == "code"}
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workdir = Path(self.temporary.name)

    def run_cell(self, name, env):
        with redirect_stdout(io.StringIO()):
            exec(compile(self.code[name], name, "exec"), env)

    def environment(self, archive):
        return {
            "Path": Path, "tempfile": tempfile, "zipfile": zipfile, "stat": stat,
            "hashlib": hashlib, "json": json, "yaml": yaml, "Counter": Counter,
            "np": np, "cv2": cv2, "Image": Image, "pd": pd,
            "display": lambda value: None,
            "WORKDIR": self.workdir, "EXPECTED_CLASSES": {0: "connector", 1: "clip", 2: "cable"},
            "files": SimpleNamespace(upload=lambda: {str(archive): archive.read_bytes()}),
        }

    def archive(self):
        output = self.workdir / "dataset.zip"
        image = io.BytesIO()
        Image.new("RGB", (64, 64), "white").save(image, format="JPEG")
        with zipfile.ZipFile(output, "w") as handle:
            handle.writestr("dataset/data.yaml", "path: /obsolete\nnames: [connector, clip, cable]\n")
            for split in ("train", "val"):
                handle.writestr(f"dataset/images/{split}/sample.jpg", image.getvalue())
                handle.writestr(f"dataset/labels/{split}/sample.txt", "\n".join(
                    f"{key} 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2" for key in range(3)))
        return output

    def prepare(self):
        if pd is None:
            self.skipTest("pandas is required for Colab table rendering")
        env = self.environment(self.archive())
        for name in ("chargement", "configuration-dataset", "controle-labels"):
            self.run_cell(name, env)
        return env

    def test_all_code_cells_compile_and_have_no_fabricated_outputs(self):
        ids = []
        for cell in self.notebook["cells"]:
            ids.append(cell["id"])
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]), filename=cell["id"])
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertIn("BASE_MODEL = 'yolov8n-seg.pt'", self.code["parametres"])
        self.assertIn('"business_rule_evaluated": False', self.code["validation"])

    def test_portable_zip_prepares_absolute_paths_and_three_segment_classes(self):
        env = self.prepare()
        cfg = yaml.safe_load(env["dataset_yaml"].read_text())
        self.assertEqual(Path(cfg["path"]), env["dataset_dir"].resolve())
        self.assertEqual(cfg["names"], env["EXPECTED_CLASSES"])
        self.assertNotIn("/obsolete", env["dataset_yaml"].read_text())
        self.assertEqual([row["Images"] for row in env["summary"]], [1, 1])
        self.assertEqual(env["summary"][0]["cable"], 1)

    def test_second_upload_keeps_previous_extraction(self):
        env = self.prepare()
        previous = env["dataset_dir"]
        self.run_cell("chargement", env)
        self.assertNotEqual(previous, env["dataset_dir"])
        self.assertTrue(previous.is_dir())

    def test_boxes_are_rejected_before_training(self):
        env = self.prepare()
        (env["dataset_dir"] / "labels/train/sample.txt").write_text("0 0.5 0.5 0.1 0.1\n")
        with self.assertRaisesRegex(ValueError, "polygone YOLO-Seg"):
            self.run_cell("controle-labels", env)

    def test_missing_labels_are_not_silently_used_as_background(self):
        env = self.prepare()
        (env["dataset_dir"] / "labels/val/sample.txt").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Annotation manquante"):
            self.run_cell("controle-labels", env)

    def test_wrong_class_order_is_rejected(self):
        env = self.environment(self.archive())
        self.run_cell("chargement", env)
        env["yaml_candidates"][0].write_text("names: [clip, connector, cable]\n")
        with self.assertRaisesRegex(ValueError, "Classes incompatibles"):
            self.run_cell("configuration-dataset", env)

    def test_zip_traversal_is_rejected(self):
        archive = self.workdir / "unsafe.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("../escape.txt", "not allowed")
        with self.assertRaisesRegex(ValueError, "Chemin non autoris"):
            self.run_cell("chargement", self.environment(archive))
        self.assertFalse((self.workdir / "escape.txt").exists())

    def test_parameter_cell_selects_segmentation_and_sensible_device_batch(self):
        if pd is None:
            self.skipTest("pandas is required for Colab table rendering")
        for device, batch in ((0, 8), ("cpu", 1)):
            env = {"device": device, "pd": pd, "display": lambda value: None}
            self.run_cell("parametres", env)
            self.assertEqual(env["BASE_MODEL"], "yolov8n-seg.pt")
            self.assertEqual(env["BATCH"], batch)


if __name__ == "__main__":
    unittest.main()
