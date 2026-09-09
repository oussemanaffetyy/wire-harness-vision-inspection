"""Tests sans GPU : export, retour Colab, tracabilite et contrat du predicteur."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import cv2
import numpy as np

from scripts import annotate_sam2 as annotation


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def job(self, name, source=None):
        job = self.root / name
        (job / "frames").mkdir(parents=True)
        height, width = 80, 100
        frames = []
        for index in range(3):
            path = annotation.image_path(job, index)
            cv2.imwrite(str(path), np.full((height, width, 3), 200, np.uint8))
            frames.append({"source_index": index * 5, "time_s": index / 6, "sha256": annotation.file_hash(path)})
        meta = {"schema": 1, "classes": {str(k): v for k, v in annotation.CLASSES.items()},
                "source_name": name + ".MOV", "source_sha256": source or name * 20,
                "width": width, "height": height, "frames": frames}
        prompts = {"objects": [
            {"id": 10 + class_id, "class_id": class_id,
             "prompts": {"1": {"points": [[15, 15]], "labels": [1]}}}
            for class_id in range(3)
        ]}
        annotation.write_json(job / "sequence.json", meta)
        annotation.write_json(job / "prompts.json", prompts)
        run = job / "predictions" / "test_run"
        (run / "masks").mkdir(parents=True)
        masks = {str(obj["id"]): np.zeros((height, width), np.uint8) for obj in prompts["objects"]}
        cv2.rectangle(masks["10"], (4, 5), (15, 18), 1, -1)
        cv2.rectangle(masks["11"], (40, 5), (49, 14), 1, -1)
        cv2.polylines(masks["12"], [np.array([[20, 50], [30, 36], [52, 38], [70, 59], [85, 53]])], False, 1, 4)
        hashes = {}
        for index in range(3):
            path = run / "masks" / f"{index:05d}.npz"
            np.savez_compressed(path, **masks)
            hashes[str(index)] = annotation.file_hash(path)
        annotation.write_json(run / "run.json", {"fingerprint": annotation.fingerprint(meta, prompts), "mask_hashes": hashes})
        annotation.write_json(run / "review.json", {"1": {"status": "rejected"}})
        annotation.write_json(job / "active_run.json", {"path": "predictions/test_run"})
        self.approve_for_test(job, run)
        return job, run, masks

    def approve_for_test(self, job, run, empty=False, index=0):
        # Only synthetic fixtures are approved automatically by the tests.
        meta, prompts = annotation.load_job(job)
        info = annotation.read_json(run / "run.json")
        settings = annotation.review_settings(job, prompts["objects"], len(meta["frames"]))
        masks = annotation.checked_masks(run, info, index)
        visibility = annotation.load_visibility(job, meta, prompts)
        converted = annotation.convert_frame(masks, prompts["objects"], settings, index, visibility)
        self.assertFalse(converted["blocked"], converted["messages"])
        decisions = annotation.read_json(run / "review.json")
        decisions[str(index)] = {"status": "approved", "empty_confirmed": empty,
                                "conversion_token": annotation.conversion_token(meta, prompts, info, index, converted, settings)}
        annotation.write_json(run / "review.json", decisions)

    def test_export_only_approved_and_roundtrip_curved_cable(self):
        train, _, originals = self.job("train")
        val, _, _ = self.job("val")
        output = self.root / "dataset"
        archive = annotation.export_dataset([train], [val], output)
        self.assertTrue(archive.is_file())
        self.assertEqual(len(list((output / "images/train").glob("*.jpg"))), 1)
        self.assertEqual(len(list((output / "images/val").glob("*.jpg"))), 1)
        lines = next((output / "labels/train").glob("*.txt")).read_text().splitlines()
        self.assertEqual([int(line.split()[0]) for line in lines], [0, 1, 2])
        for line in lines:
            fields = line.split()
            coordinates = np.asarray(fields[1:], dtype=np.float32).reshape(-1, 2)
            self.assertTrue(np.all((coordinates >= 0) & (coordinates < 1)))
            polygon = (coordinates * np.array([100, 80], dtype=np.float32)).astype(np.int32)
            raster = np.zeros((80, 100), np.uint8)
            cv2.fillPoly(raster, [polygon], 1)
            np.testing.assert_array_equal(raster, originals[str(10 + int(fields[0]))])
        self.assertNotIn("path: .", (output / "data.yaml").read_text())
        with zipfile.ZipFile(archive) as handle:
            self.assertIn("dataset/data.yaml", handle.namelist())
        sample = annotation.read_json(output / "manifest.json")["samples"][0]
        np.testing.assert_array_equal(annotation.read_masks(output / sample["original_masks"])["12"], originals["12"])
        self.assertEqual(annotation.file_hash(output / sample["original_masks"]), sample["mask_sha256"])

    def test_same_source_cannot_leak_into_validation(self):
        train, _, _ = self.job("train", source="same_video")
        val, _, _ = self.job("val", source="same_video")
        with self.assertRaisesRegex(ValueError, "Meme video"):
            annotation.export_dataset([train], [val], self.root / "dataset")
        self.assertFalse((self.root / "dataset").exists())

    def test_draft_exports_unreviewed_and_rejected_without_changing_sources(self):
        train, run, _ = self.job("train")
        val, _, _ = self.job("val")
        annotation.write_json(run / "review.json", {"0": {"status": "rejected"}})
        before = {path: annotation.file_hash(path) for job in (train, val)
                  for path in job.rglob("*") if path.is_file()}
        with self.assertRaisesRegex(ValueError, "Aucune image approuvee"):
            annotation.export_dataset([train], [val], self.root / "strict")
        output = self.root / "draft"
        annotation.export_dataset([train], [val], output, draft=True)
        manifest = annotation.read_json(output / "manifest.json")
        self.assertEqual(manifest["export_mode"], "draft")
        self.assertEqual(len(manifest["samples"]), 6)
        self.assertTrue(all(not sample["reviewed"] for sample in manifest["samples"]))
        self.assertEqual(manifest["samples"][0]["review_decision"], "rejected")
        self.assertTrue(any(sample["quality_blocked"] for sample in manifest["samples"]))
        self.assertTrue((output / "LIRE_AVANT_ENTRAINEMENT.txt").is_file())
        self.assertEqual(before, {path: annotation.file_hash(path) for path in before})

    def test_draft_accepts_imprecise_candidate_and_preserves_raw_mask(self):
        train, run, masks = self.job("train")
        val, _, _ = self.job("val")
        masks["12"][:] = 0
        masks["12"][3:8, 3:8] = 1
        masks["12"][60:65, 80:85] = 1
        path = run / "masks/00000.npz"
        np.savez_compressed(path, **masks)
        info = annotation.read_json(run / "run.json")
        info["mask_hashes"]["0"] = annotation.file_hash(path)
        annotation.write_json(run / "run.json", info)
        output = self.root / "draft"
        annotation.export_dataset([train], [val], output, draft=True)
        sample = annotation.read_json(output / "manifest.json")["samples"][0]
        self.assertTrue(sample["quality_blocked"])
        self.assertLess(sample["conversion"]["12"]["iou"], .98)
        np.testing.assert_array_equal(annotation.read_masks(output / sample["original_masks"])["12"], masks["12"])
        lines = next((output / "labels/train").glob("*00000.txt")).read_text().splitlines()
        self.assertEqual([int(line.split()[0]) for line in lines], [0, 1, 2])

    def test_draft_keeps_human_absence_and_excludes_uncertain_image(self):
        train, _, _ = self.job("train")
        val, _, _ = self.job("val")
        self.mark(train, 10, 0, 0, "absent")
        self.mark(train, 10, 1, 1, "ignore")
        output = self.root / "draft"
        annotation.export_dataset([train], [val], output, draft=True)
        manifest = annotation.read_json(output / "manifest.json")
        self.assertEqual(len(manifest["samples"]), 5)
        self.assertEqual(manifest["skipped"][0]["frame_index"], 1)
        lines = next((output / "labels/train").glob("*00000.txt")).read_text().splitlines()
        self.assertEqual([int(line.split()[0]) for line in lines], [1, 2])

    def test_draft_skips_whole_image_instead_of_losing_degenerate_object(self):
        train, run, masks = self.job("train")
        val, _, _ = self.job("val")
        masks["12"][:] = 0
        masks["12"][20, 5:30] = 1
        path = run / "masks/00000.npz"
        np.savez_compressed(path, **masks)
        info = annotation.read_json(run / "run.json")
        info["mask_hashes"]["0"] = annotation.file_hash(path)
        annotation.write_json(run / "run.json", info)
        output = self.root / "draft"
        annotation.export_dataset([train], [val], output, draft=True)
        manifest = annotation.read_json(output / "manifest.json")
        self.assertEqual(len(manifest["samples"]), 5)
        self.assertIn("non vide sans polygone", manifest["skipped"][0]["reason"])
        self.assertFalse(list((output / "labels/train").glob("*00000.txt")))

    def test_draft_still_refuses_stale_prompts_corrupt_masks_and_source_leakage(self):
        train, run, _ = self.job("train")
        val, _, _ = self.job("val", source="train" * 20)
        with self.assertRaisesRegex(ValueError, "Meme video"):
            annotation.export_dataset([train], [val], self.root / "dataset", draft=True)
        prompts = annotation.read_json(train / "prompts.json")
        prompts["objects"][0]["prompts"]["1"]["points"][0] = [16, 16]
        annotation.write_json(train / "prompts.json", prompts)
        with self.assertRaisesRegex(ValueError, "reperes ont change"):
            annotation.draft_frames(train)
        prompts["objects"][0]["prompts"]["1"]["points"][0] = [15, 15]
        annotation.write_json(train / "prompts.json", prompts)
        (run / "masks/00000.npz").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "Masque modifie"):
            annotation.draft_frames(train)

    def test_changed_prompt_invalidates_approval(self):
        job, _, _ = self.job("test")
        prompts = annotation.read_json(job / "prompts.json")
        prompts["objects"][0]["prompts"]["1"]["points"][0] = [16, 16]
        annotation.write_json(job / "prompts.json", prompts)
        with self.assertRaisesRegex(ValueError, "reperes ont change"):
            annotation.approved_frames(job)

    def test_modified_masks_or_images_are_rejected(self):
        job, run, masks = self.job("test")
        masks["10"][7, 7] = 0
        np.savez_compressed(run / "masks/00000.npz", **masks)
        with self.assertRaisesRegex(ValueError, "Masque modifie"):
            annotation.approved_frames(job)
        annotation.image_path(job, 0).write_bytes(b"not an image")
        with self.assertRaisesRegex(ValueError, "Image absente ou modifiee"):
            annotation.load_job(job)

    def test_thin_mask_still_blocks_export(self):
        masks = []
        fragmented = np.zeros((40, 40), np.uint8)
        fragmented[2:10, 2:10] = 1
        fragmented[20:30, 20:30] = 1
        hollow = np.zeros_like(fragmented)
        hollow[3:30, 3:30] = 1
        hollow[10:20, 10:20] = 0
        thin = np.zeros_like(fragmented)
        thin[20, 5:30] = 1
        masks.append(thin)
        for mask in masks:
            with self.subTest(mask=mask.sum()):
                polygon, _, blocked = annotation.polygon_from_mask(mask)
                self.assertIsNone(polygon)
                self.assertTrue(blocked)
        _, _, blocked = annotation.polygon_from_mask(np.zeros((40, 40), bool))
        self.assertFalse(blocked)

    def test_holes_are_encoded_without_filling_them(self):
        mask = np.zeros((100, 100), bool)
        mask[10:90, 10:90] = True
        mask[30:70, 30:70] = False
        polygon, _, blocked = annotation.polygon_from_mask(mask)
        self.assertFalse(blocked)
        raster = annotation.polygon_raster(polygon, mask.shape)
        np.testing.assert_array_equal(raster, mask)
        self.assertFalse(raster[30:70, 30:70].any())

    def test_large_bridge_is_shown_but_blocked(self):
        mask = np.zeros((40, 40), np.uint8)
        mask[2:10, 2:10] = 1
        mask[20:30, 20:30] = 1
        polygon, messages, blocked = annotation.polygon_from_mask(mask)
        self.assertIsNotNone(polygon)
        self.assertTrue(blocked)
        self.assertTrue(any("imprecise" in text for text in messages))

    def test_empty_image_requires_explicit_confirmation(self):
        job, run, masks = self.job("test")
        masks = {key: np.zeros_like(mask) for key, mask in masks.items()}
        path = run / "masks/00000.npz"
        np.savez_compressed(path, **masks)
        info = annotation.read_json(run / "run.json")
        info["mask_hashes"]["0"] = annotation.file_hash(path)
        annotation.write_json(run / "run.json", info)
        with self.assertRaisesRegex(ValueError, "vide non confirmee"):
            annotation.approved_frames(job)
        self.approve_for_test(job, run, empty=True)
        self.assertEqual(len(annotation.approved_frames(job)[3]), 1)

    def multipart(self, clip_in_bridge=False):
        cable = np.zeros((110, 120), bool)
        cable[10:80, 5:53] = True
        cable[10:80, 58:106] = True
        clip = np.zeros_like(cable)
        if clip_in_bridge:
            clip[8:13, 54:57] = True
        else:
            clip[93:102, 54:57] = True
        objects = [{"id": 2, "class_id": 2}, {"id": 3, "class_id": 1}]
        return {"2": cable, "3": clip}, objects

    def test_multi_segments_keep_one_instance_without_business_constraints(self):
        masks, objects = self.multipart()
        original = {key: mask.copy() for key, mask in masks.items()}
        result = annotation.convert_frame(masks, objects, {"empty_clip_ids": [3]})
        self.assertFalse(result["blocked"], result["messages"])
        self.assertIs(type(result["blocked"]), bool)
        self.assertEqual(set(result["polygons"]), {"2", "3"})
        self.assertTrue(np.all(result["rasters"]["2"][masks["2"]]))
        self.assertGreater(result["added"]["2"].sum(), 0)
        self.assertFalse(np.any(result["rasters"]["2"] & masks["3"]))
        for key in masks:
            np.testing.assert_array_equal(masks[key], original[key])

    def test_thin_format_bridge_near_clip_is_not_blocked_by_legacy_rule(self):
        masks, objects = self.multipart(clip_in_bridge=True)
        self.assertFalse(np.any(masks["2"] & masks["3"]))
        result = annotation.convert_frame(masks, objects, {"empty_clip_ids": [3]})
        self.assertFalse(result["blocked"], result["messages"])
        self.assertFalse(any("zone protegee" in text for text in result["messages"]))
        self.assertTrue(np.any(result["added"]["2"] & masks["3"]))
        self.assertGreater(result["statistics"]["2"]["iou"], .98)

    def test_review_does_not_draw_a_protected_clip_box(self):
        masks, objects = self.multipart(clip_in_bridge=True)
        result = annotation.convert_frame(masks, objects)
        frame = np.full((*masks["2"].shape, 3), 200, np.uint8)
        for mode, selected in ((0, masks), (1, result["rasters"])):
            np.testing.assert_array_equal(annotation.review_canvas(frame, masks, objects, result, mode),
                                          annotation.overlay(frame, selected, objects))

    def test_clip_on_cable_and_occluded_clip_do_not_impose_empty_state(self):
        cable = np.zeros((80, 100), bool)
        cable[20:60, 30:40] = True
        clip = np.zeros_like(cable)
        clip[35:45, 28:42] = True
        masks = {"2": cable, "4": clip}
        objects = [{"id": 2, "class_id": 2}, {"id": 4, "class_id": 1}]
        legacy = {"empty_clip_ids": [4]}
        result = annotation.convert_frame(masks, objects, legacy)
        self.assertFalse(result["blocked"], result["messages"])
        self.assertTrue(np.any(result["rasters"]["2"] & result["rasters"]["4"]))
        masks["4"][:] = False
        result = annotation.convert_frame(masks, objects, legacy)
        self.assertFalse(result["blocked"], result["messages"])
        self.assertTrue(any("masque vide : verifier l'absence" in text for text in result["messages"]))

    def test_noise_removal_is_bounded_and_never_erases_a_main_fragment(self):
        masks, objects = self.multipart()
        masks["2"][108, 118] = True
        original = masks["2"].copy()
        result = annotation.convert_frame(masks, objects)
        self.assertEqual(result["statistics"]["2"]["removed_pixels"], 1)
        self.assertTrue(np.all(result["rasters"]["2"][10:80, 5:53]))
        self.assertTrue(np.all(result["rasters"]["2"][10:80, 58:106]))
        np.testing.assert_array_equal(masks["2"], original)

    def test_three_fragments_do_not_fill_between_them(self):
        mask = np.zeros((300, 300), bool)
        mask[10:110, 10:110] = True
        mask[10:110, 120:220] = True
        mask[120:220, 120:220] = True
        polygon, _, blocked = annotation.polygon_from_mask(mask)
        self.assertFalse(blocked)
        raster = annotation.polygon_raster(polygon, mask.shape)
        self.assertTrue(np.all(raster[mask]))
        self.assertLess(np.count_nonzero(raster & ~mask), 25)
        self.assertFalse(raster[160, 60])

    def test_serialized_vertices_match_preview_at_factory_resolution(self):
        mask = np.zeros((848, 464), bool)
        mask[50:300, 230:240] = True
        mask[320:800, 230:240] = True
        polygon, _, blocked = annotation.polygon_from_mask(mask)
        self.assertFalse(blocked)
        values = np.fromstring(annotation.polygon_text(polygon, mask.shape), sep=" ", dtype=np.float32).reshape(-1, 2)
        raster = np.zeros_like(mask, dtype=np.uint8)
        cv2.fillPoly(raster, [(values * np.array([464, 848], np.float32)).astype(np.int32)], 1)
        np.testing.assert_array_equal(raster.astype(bool), annotation.polygon_raster(polygon, mask.shape))

    def test_old_approval_and_policy_changes_require_review_but_legacy_rules_do_not(self):
        job, run, _ = self.job("test")
        with patch.dict(annotation.CONVERSION_POLICY, {"min_iou": .97}):
            with self.assertRaisesRegex(ValueError, "approbation ancienne"):
                annotation.approved_frames(job)
        annotation.write_json(job / "review_settings.json", {"empty_clip_ids": [11]})
        self.assertEqual(len(annotation.approved_frames(job)[3]), 1)
        annotation.write_json(run / "review.json", {"0": {"status": "approved"}})
        with self.assertRaisesRegex(ValueError, "approbation ancienne"):
            annotation.approved_frames(job)

    def test_difference_view_marks_added_pixels(self):
        masks, objects = self.multipart()
        objects = objects[:1]
        result = annotation.convert_frame(masks, objects)
        frame = np.full((*masks["2"].shape, 3), 200, np.uint8)
        rendered = annotation.review_canvas(frame, masks, objects, result, 2)
        self.assertTrue(np.all(rendered[result["added"]["2"]] == (220, 40, 220)))

    def test_multi_component_export_is_one_row_and_keeps_raw_masks(self):
        train, run, masks = self.job("train")
        val, _, _ = self.job("val")
        masks["12"][:] = 0
        masks["12"][30:70, 15:48] = 1
        masks["12"][30:70, 52:85] = 1
        path = run / "masks/00000.npz"
        np.savez_compressed(path, **masks)
        info = annotation.read_json(run / "run.json")
        info["mask_hashes"]["0"] = annotation.file_hash(path)
        annotation.write_json(run / "run.json", info)
        annotation.write_json(train / "review_settings.json", {"empty_clip_ids": [11]})
        self.approve_for_test(train, run)
        output = self.root / "multi_dataset"
        annotation.export_dataset([train], [val], output)
        lines = next((output / "labels/train").glob("*.txt")).read_text().splitlines()
        cable_lines = [line for line in lines if line.split()[0] == "2"]
        self.assertEqual(len(cable_lines), 1)
        sample = annotation.read_json(output / "manifest.json")["samples"][0]
        np.testing.assert_array_equal(annotation.read_masks(output / sample["original_masks"])["12"], masks["12"])
        values = np.array(cable_lines[0].split()[1:], np.float32).reshape(-1, 2)
        raster = np.zeros_like(masks["12"])
        cv2.fillPoly(raster, [(values * np.array([100, 80], np.float32)).astype(np.int32)], 1)
        expected = annotation.convert_frame(masks, annotation.load_job(train)[1]["objects"], {"empty_clip_ids": [11]})
        np.testing.assert_array_equal(raster, expected["rasters"]["12"])
        self.assertFalse(np.any(raster & masks["11"]))

    def test_legacy_settings_are_ignored_even_with_obsolete_object_ids(self):
        job, _, _ = self.job("test")
        objects = annotation.load_job(job)[1]["objects"]
        for value in ({"empty_clip_ids": [12]}, {"empty_clip_ids": [11, 11]},
                      {"empty_clip_ids": [True]}, {"empty_clip_ids": [], "ignore_guard": True}):
            annotation.write_json(job / "review_settings.json", value)
            self.assertEqual(annotation.review_settings(job, objects), {})

    def test_frame_scoped_legacy_rules_never_block_overlap(self):
        mask = np.zeros((80, 100), bool)
        mask[50:60, 30:40] = True
        masks = {"4": mask, "2": mask.copy()}
        objects = [{"id": 4, "class_id": 1}, {"id": 2, "class_id": 2}]
        settings = {"empty_clip_ids": [], "empty_clip_frames": {"4": [13, 25]}}
        for index in (None, 12, 13, 14, 25):
            with self.subTest(index=index):
                result = annotation.convert_frame(masks, objects, settings, index)
                self.assertFalse(result["blocked"], result["messages"])
                self.assertNotIn("empty_clip_ids", result)
                self.assertNotIn("inspection_status", result)

    def test_wrong_clip_target_and_negative_click_are_blocked_on_seed_frame(self):
        clip = np.zeros((100, 100), bool)
        clip[10:20, 30:40] = True  # The old, upper clip.
        obj = {"id": 4, "class_id": 1, "prompts": {
            "13": {"points": [[35, 65], [45, 65]], "labels": [1, 0]}}}
        masks = {"4": clip.copy()}
        result = annotation.convert_frame(masks, [obj], frame_index=13)
        self.assertTrue(result["blocked"])
        self.assertTrue(any("positif hors du masque" in m for m in result["messages"]))
        np.testing.assert_array_equal(masks["4"], clip)
        self.assertFalse(annotation.convert_frame(masks, [obj], frame_index=12)["blocked"])
        masks["4"][:] = False
        masks["4"][60:70, 30:40] = True  # The new, lower clip.
        self.assertFalse(annotation.convert_frame(masks, [obj], frame_index=13)["blocked"])
        masks["4"][60:70, 40:50] = True
        result = annotation.convert_frame(masks, [obj], frame_index=13)
        self.assertTrue(result["blocked"])
        self.assertTrue(any("negatif inclus" in m for m in result["messages"]))

    def test_legacy_frame_settings_do_not_need_repair(self):
        job, _, _ = self.job("test")
        objects = annotation.load_job(job)[1]["objects"]
        for per_frame in ({"12": [0]}, {"011": [0]}, {"11": []}, {"11": [0, 0]},
                          {"11": [True]}, {"11": [-1]}, {"11": [3]}, {"11": "0"}, []):
            with self.subTest(per_frame=per_frame):
                annotation.write_json(job / "review_settings.json", {
                    "empty_clip_ids": [], "empty_clip_frames": per_frame})
                self.assertEqual(annotation.review_settings(job, objects, 3), {})
        annotation.write_json(job / "review_settings.json", {
            "empty_clip_ids": [11], "empty_clip_frames": {"11": [0]}})
        self.assertEqual(annotation.review_settings(job, objects, 3), {})

    def test_legacy_rules_are_not_repacked_or_used_by_export_or_restore(self):
        train, run, _ = self.job("train")
        val, _, _ = self.job("val")
        settings = {"empty_clip_ids": [], "empty_clip_frames": {"11": [0, 2]}}
        annotation.write_json(train / "review_settings.json", settings)
        self.approve_for_test(train, run)
        annotation.export_dataset([train], [val], self.root / "scoped_dataset")
        manifest = annotation.read_json(self.root / "scoped_dataset/manifest.json")
        self.assertEqual(manifest["samples"][0]["review_settings"], {})
        annotation.pack_jobs([train], self.root / "scoped_upload.zip")
        with zipfile.ZipFile(self.root / "scoped_upload.zip") as archive:
            self.assertNotIn("jobs/train/review_settings.json", archive.namelist())
        with zipfile.ZipFile(self.root / "scoped_results.zip", "w") as archive:
            for path in train.rglob("*"):
                if path.is_file():
                    archive.write(path, "jobs/train/" + path.relative_to(train).as_posix())
        restored = annotation.restore_results(self.root / "scoped_results.zip", self.root / "scoped_return")
        # Historical files are preserved when restoring an old ZIP, but inert.
        self.assertEqual(annotation.read_json(restored / "train/review_settings.json"), settings)
        restored_meta, restored_prompts = annotation.load_job(restored / "train")
        self.assertEqual(annotation.review_settings(restored / "train", restored_prompts["objects"],
                                                    len(restored_meta["frames"])), {})
        self.assertEqual(annotation.read_json(restored / "train/predictions/test_run/review.json"), {})

    def test_mobile_clip_on_connector_and_cable_both_export(self):
        jobs = []
        for name, clip_box in (("test1", (14, 8, 19, 14)), ("test2", (50, 45, 60, 55))):
            job, run, masks = self.job(name)
            masks["11"][:] = 0
            x1, y1, x2, y2 = clip_box
            masks["11"][y1:y2, x1:x2] = 1
            masks["12"][:] = 0
            masks["12"][40:60, 30:80] = 1
            path = run / "masks/00000.npz"
            np.savez_compressed(path, **masks)
            info = annotation.read_json(run / "run.json")
            info["mask_hashes"]["0"] = annotation.file_hash(path)
            annotation.write_json(run / "run.json", info)
            annotation.write_json(job / "review_settings.json", {"empty_clip_ids": [11]})
            self.approve_for_test(job, run)
            jobs.append(job)
        output = self.root / "mobile_clip_dataset"
        annotation.export_dataset([jobs[0]], [jobs[1]], output)
        manifest = annotation.read_json(output / "manifest.json")
        self.assertEqual(len(manifest["samples"]), 2)
        for sample in manifest["samples"]:
            masks = annotation.read_masks(output / sample["original_masks"])
            partner = "10" if sample["source_name"] == "test1.MOV" else "12"
            self.assertTrue(np.any(masks["11"] & masks[partner]))
            self.assertNotIn("inspection_status", sample)
            self.assertEqual(sample["review_settings"], {})

    def test_review_blocks_seed_mismatch_and_tokens_include_frame_index(self):
        job, run, _ = self.job("test")
        meta, prompts = annotation.load_job(job)
        # Frame 1 has deliberately incorrect fixture prompts for clip and cable.
        keys = iter([ord('a'), ord('q')])
        with patch.object(cv2, "namedWindow"), patch.object(cv2, "createTrackbar"), \
             patch.object(cv2, "setMouseCallback"), patch.object(cv2, "getTrackbarPos", return_value=1), \
             patch.object(cv2, "imshow"), patch.object(cv2, "waitKey", side_effect=lambda _: next(keys)), \
             patch.object(cv2, "getWindowProperty", return_value=1), patch.object(cv2, "destroyAllWindows"):
            annotation.viewer(job, review=True)
        self.assertEqual(annotation.read_json(run / "review.json")["1"]["status"], "rejected")
        info = annotation.read_json(run / "run.json")
        masks = annotation.checked_masks(run, info, 0)
        converted = annotation.convert_frame(masks, prompts["objects"], frame_index=0)
        settings = {"empty_clip_ids": []}
        self.assertNotEqual(annotation.conversion_token(meta, prompts, info, 0, converted, settings),
                            annotation.conversion_token(meta, prompts, info, 2, converted, settings))

    def test_review_requires_export_view_and_stores_conversion_token(self):
        job, run, _ = self.job("test")
        for keys_to_send, approved in (([ord('v'), ord('v'), ord('a'), ord('q')], False),
                                       ([ord('a'), ord('q')], True)):
            annotation.write_json(run / "review.json", {})
            keys = iter(keys_to_send)
            with patch.object(cv2, "namedWindow"), patch.object(cv2, "createTrackbar"), \
                 patch.object(cv2, "setMouseCallback"), patch.object(cv2, "getTrackbarPos", return_value=0), \
                 patch.object(cv2, "imshow"), patch.object(cv2, "waitKey", side_effect=lambda _: next(keys)), \
                 patch.object(cv2, "getWindowProperty", return_value=1), patch.object(cv2, "destroyAllWindows"):
                annotation.viewer(job, review=True)
            decisions = annotation.read_json(run / "review.json")
            self.assertEqual(decisions.get("0", {}).get("status") == "approved", approved)
            if approved:
                self.assertIn("conversion_token", decisions["0"])
                self.assertNotIn("inspection_status", decisions["0"])
                self.assertEqual(len(annotation.approved_frames(job)[3]), 1)

    def test_negative_clicks_preserve_object_id_and_video_directions(self):
        class Predictor:
            def reset_state(self, state):
                self.inputs, self.directions = [], []

            def add_new_points_or_box(self, **kwargs):
                self.inputs.append(kwargs)

            def propagate_in_video(self, state, start_frame_idx, reverse):
                self.directions.append(reverse)
                indices = range(start_frame_idx, -1, -1) if reverse else range(start_frame_idx, 5)
                for index in indices:
                    values = np.zeros((2, 1, 4, 6), np.float32)
                    values[0] = -1
                    values[1] = 1
                    yield index, [99, 7], values

        predictor = Predictor()
        obj = {"id": 7, "class_id": 2, "prompts": {"2": {"points": [[2, 1], [5, 3]], "labels": [1, 0]}}}
        results = dict(annotation.track_object(predictor, {}, obj, 5))
        self.assertEqual(set(results), set(range(5)))
        self.assertTrue(all(mask.all() for mask in results.values()))
        self.assertEqual(predictor.directions, [False, True])
        np.testing.assert_array_equal(predictor.inputs[0]["labels"], [1, 0])
        self.assertEqual(predictor.inputs[0]["obj_id"], 7)

    def test_missing_sam_frames_fail(self):
        class Predictor:
            def reset_state(self, state):
                pass

            def add_new_points_or_box(self, **kwargs):
                pass

            def propagate_in_video(self, *args, **kwargs):
                yield 0, [1], np.ones((1, 1, 5, 5), np.float32)

        obj = {"id": 1, "prompts": {"0": {"points": [[1, 1]], "labels": [1]}}}
        with self.assertRaisesRegex(ValueError, "toutes les images"):
            list(annotation.track_object(Predictor(), {}, obj, 2))

    def test_no_need_to_invent_absent_classes_in_a_sequence(self):
        job, _, _ = self.job("test")
        meta, prompts = annotation.load_job(job)
        prompts["objects"] = prompts["objects"][:1]
        annotation.validate_prompts(meta, prompts)
        prompts["objects"][0]["prompts"]["1"]["points"] = [[100, 20]]
        with self.assertRaisesRegex(ValueError, "sort de l'image"):
            annotation.validate_prompts(meta, prompts)

    def test_pack_portable_job_and_restore_result_without_overwrite(self):
        job, _, _ = self.job("test")
        annotation.write_json(job / "review_settings.json", {"empty_clip_ids": [11]})
        archive = self.root / "upload.zip"
        annotation.pack_jobs([job], archive)
        with zipfile.ZipFile(archive) as handle:
            self.assertIn("scripts/annotate_sam2.py", handle.namelist())
            self.assertIn("requirements-sam2.txt", handle.namelist())
            self.assertNotIn("jobs/test/review_settings.json", handle.namelist())
            self.assertFalse(any("predictions" in name for name in handle.namelist()))
        results = self.root / "results.zip"
        with zipfile.ZipFile(results, "w") as handle:
            for path in job.rglob("*"):
                if path.is_file():
                    handle.write(path, "jobs/test/" + path.relative_to(job).as_posix())
        restored = annotation.restore_results(results, self.root / "restored")
        self.assertEqual(annotation.read_json(restored / "test/predictions/test_run/review.json"), {})
        self.assertEqual(annotation.read_json(restored / "test/review_settings.json"), {"empty_clip_ids": [11]})
        self.assertTrue(annotation.read_json(job / "predictions/test_run/review.json"))
        with self.assertRaises(FileExistsError):
            annotation.restore_results(results, self.root / "restored")

    def test_restore_rejects_path_traversal(self):
        archive = self.root / "bad.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("jobs/../../escaped.txt", "bad")
        with self.assertRaisesRegex(ValueError, "Chemin interdit"):
            annotation.restore_results(archive, self.root / "restored")
        self.assertFalse((self.root / "escaped.txt").exists())
        self.assertFalse((self.root / "restored").exists())

    def test_click_coordinates_use_original_image_resolution(self):
        job, _, _ = self.job("test")
        callback = {}
        keys = iter([ord("0"), None, ord("q")])

        def wait(delay):
            key = next(keys)
            if key is None:
                callback["mouse"](cv2.EVENT_LBUTTONDOWN, 12, 15, 0, None)
                return -1
            return key

        with patch.object(cv2, "namedWindow"), patch.object(cv2, "createTrackbar"), \
             patch.object(cv2, "setMouseCallback", side_effect=lambda title, cb: callback.update(mouse=cb)), \
             patch.object(cv2, "getTrackbarPos", return_value=0), patch.object(cv2, "imshow"), \
             patch.object(cv2, "waitKey", side_effect=wait), \
             patch.object(cv2, "getWindowProperty", return_value=1), patch.object(cv2, "destroyAllWindows"):
            annotation.viewer(job)
        obj = annotation.read_json(job / "prompts.json")["objects"][-1]
        self.assertEqual(obj["class_id"], 0)
        self.assertEqual(obj["prompts"]["0"]["points"], [[8.0, 10.0]])

    def mark(self, job, object_id, start, end, state):
        meta, prompts = annotation.load_job(job)
        visibility = annotation.load_visibility(job, meta, prompts)
        obj = next(obj for obj in prompts["objects"] if obj["id"] == object_id)
        annotation.set_visibility(visibility, obj, start, end, state, len(meta["frames"]))
        annotation.write_json(job / "visibility.json", visibility)
        return visibility

    def test_absence_is_per_object_and_range_without_modifying_raw_masks(self):
        job, run, masks = self.job("test")
        meta, prompts = annotation.load_job(job)
        before = annotation.file_hash(run / "masks/00000.npz")
        visibility = self.mark(job, 10, 1, 0, "absent")  # Reversed endpoints also work.
        for index in (0, 1):
            result = annotation.convert_frame(masks, prompts["objects"], frame_index=index, visibility=visibility)
            self.assertIsNone(result["polygons"]["10"])
            self.assertFalse(result["effective_masks"]["10"].any())
            self.assertEqual(result["statistics"]["10"]["human_suppressed_pixels"], int(masks["10"].sum()))
            for key in ("11", "12"):
                np.testing.assert_array_equal(result["effective_masks"][key], masks[key])
        result = annotation.convert_frame(masks, prompts["objects"], frame_index=2, visibility=visibility)
        self.assertIsNotNone(result["polygons"]["10"])
        self.assertEqual(before, annotation.file_hash(run / "masks/00000.npz"))
        annotation.active_run(job, meta, prompts)  # Absence does not require recalculating SAM.
        visibility = self.mark(job, 10, 1, 1, "clear")
        self.assertEqual(annotation.visibility_at(visibility, 1), {})
        self.assertEqual(annotation.visibility_at(visibility, 0), {"10": "absent"})

    def test_absence_invalidates_approval_and_export_keeps_raw_and_visible_masks(self):
        train, run, _ = self.job("train")
        val, _, _ = self.job("val")
        self.mark(train, 10, 0, 0, "absent")
        with self.assertRaisesRegex(ValueError, "approbation ancienne"):
            annotation.approved_frames(train)
        self.approve_for_test(train, run)
        self.approve_for_test(train, run, index=2)  # Keep an actually visible connector in train.
        output = self.root / "visibility_dataset"
        annotation.export_dataset([train], [val], output)
        manifest = annotation.read_json(output / "manifest.json")
        samples = [s for s in manifest["samples"] if s["source_name"] == "train.MOV"]
        self.assertEqual(len(samples), 2)
        sample = next(s for s in samples if s["source_index"] == 0)
        label = output / sample["image"].replace("images/", "labels/").replace(".jpg", ".txt")
        self.assertEqual([int(line.split()[0]) for line in label.read_text().splitlines()], [1, 2])
        self.assertTrue(annotation.read_masks(output / sample["original_masks"])["10"].any())
        self.assertFalse(annotation.read_masks(output / sample["visible_masks"])["10"].any())
        self.assertEqual(sample["visibility"], {"10": "absent"})
        self.assertEqual(annotation.file_hash(output / sample["visible_masks"]), sample["visible_mask_sha256"])
        visible_sample = next(s for s in samples if s["source_index"] == 10)
        self.assertTrue(annotation.read_masks(output / visible_sample["visible_masks"])["10"].any())

    def test_uncertain_object_excludes_entire_image_even_if_previously_approved(self):
        train, run, masks = self.job("train")
        val, _, _ = self.job("val")
        visibility = self.mark(train, 10, 0, 0, "ignore")
        meta, prompts = annotation.load_job(train)
        result = annotation.convert_frame(masks, prompts["objects"], frame_index=0, visibility=visibility)
        self.assertTrue(result["blocked"])
        self.assertTrue(result["excluded"])
        self.assertTrue(result["effective_masks"]["10"].any())  # Not silently turned into background.
        self.approve_for_test(train, run, index=2)
        output = self.root / "uncertain_dataset"
        annotation.export_dataset([train], [val], output)
        samples = annotation.read_json(output / "manifest.json")["samples"]
        self.assertEqual([s["source_index"] for s in samples if s["source_name"] == "train.MOV"], [10])

    def test_visibility_is_bound_to_sequence_and_object_class(self):
        job, _, masks = self.job("test")
        meta, prompts = annotation.load_job(job)
        valid = self.mark(job, 10, 0, 0, "absent")
        with self.assertRaisesRegex(ValueError, "indice d'image"):
            annotation.convert_frame(masks, prompts["objects"], visibility=valid)
        for frames in ({"3": "absent"}, {"00": "absent"}, {"-1": "absent"}, {"0": "visible"}):
            value = {**valid, "objects": {"10": {"class_id": 0, "frames": frames}}}
            annotation.write_json(job / "visibility.json", value)
            with self.assertRaises(ValueError):
                annotation.load_visibility(job, meta, prompts)
        annotation.write_json(job / "visibility.json", {**valid, "sequence_fingerprint": "wrong"})
        with self.assertRaises(ValueError):
            annotation.load_visibility(job, meta, prompts)
        annotation.write_json(job / "visibility.json", valid)
        prompts["objects"][0]["class_id"] = 1
        with self.assertRaisesRegex(ValueError, "classe modifiee"):
            annotation.load_visibility(job, meta, prompts)

    def test_visibility_roundtrip_including_older_notebook_missing_sidecar(self):
        job, run, _ = self.job("test")
        visibility = self.mark(job, 10, 0, 1, "absent")
        self.mark(job, 11, 2, 2, "ignore")
        visibility = annotation.load_visibility(job, *annotation.load_job(job))
        annotation.pack_jobs([job], self.root / "with_visibility.zip")
        with zipfile.ZipFile(self.root / "with_visibility.zip") as archive:
            self.assertEqual(archive.read("jobs/test/visibility.json"), (job / "visibility.json").read_bytes())
        info = annotation.read_json(run / "run.json")
        info["visibility"] = visibility  # Snapshot supplied by propagate.
        annotation.write_json(run / "run.json", info)
        for include_sidecar in (True, False):
            archive_path = self.root / f"result_{include_sidecar}.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for path in job.rglob("*"):
                    if path.is_file() and (include_sidecar or path.name != "visibility.json"):
                        archive.write(path, "jobs/test/" + path.relative_to(job).as_posix())
            restored = annotation.restore_results(archive_path, self.root / f"return_{include_sidecar}") / "test"
            self.assertEqual(annotation.load_visibility(restored, *annotation.load_job(restored)), visibility)
            self.assertEqual(annotation.read_json(restored / "predictions/test_run/review.json"), {})

    def test_points_range_shortcuts_save_visibility_without_changing_prompts(self):
        job, run, _ = self.job("test")
        before = annotation.read_json(job / "prompts.json")
        cursor = {"index": 0}
        actions = iter([ord("g"), "advance", ord("h"), ord("q")])

        def wait(delay):
            action = next(actions)
            if action == "advance":
                cursor["index"] = 2
                return -1
            return action

        with patch.object(cv2, "namedWindow"), patch.object(cv2, "createTrackbar"), \
             patch.object(cv2, "setMouseCallback"), \
             patch.object(cv2, "getTrackbarPos", side_effect=lambda *args: cursor["index"]), \
             patch.object(cv2, "imshow"), patch.object(cv2, "waitKey", side_effect=wait), \
             patch.object(cv2, "getWindowProperty", return_value=1), patch.object(cv2, "destroyAllWindows"):
            annotation.viewer(job)
        visibility = annotation.load_visibility(job, *annotation.load_job(job))
        self.assertEqual(visibility["objects"]["10"]["frames"], {"0": "absent", "1": "absent", "2": "absent"})
        self.assertEqual(annotation.read_json(job / "prompts.json"), before)

    def test_review_approves_absence_but_refuses_uncertainty(self):
        job, run, _ = self.job("test")
        for state, expected in (("absent", "approved"), ("ignore", None)):
            self.mark(job, 10, 0, 0, state)
            annotation.write_json(run / "review.json", {})
            actions = iter([ord("a"), ord("q")])
            with patch.object(cv2, "namedWindow"), patch.object(cv2, "createTrackbar"), \
                 patch.object(cv2, "setMouseCallback"), patch.object(cv2, "getTrackbarPos", return_value=0), \
                 patch.object(cv2, "imshow"), patch.object(cv2, "waitKey", side_effect=lambda _: next(actions)), \
                 patch.object(cv2, "getWindowProperty", return_value=1), patch.object(cv2, "destroyAllWindows"):
                annotation.viewer(job, review=True)
            self.assertEqual(annotation.read_json(run / "review.json").get("0", {}).get("status"), expected)


if __name__ == "__main__":
    unittest.main()
