"""Inference contracts and mobile-clip decisions, without GPU or MQTT services."""
from contextlib import ExitStack, chdir
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

import run
from src import app_runner
from src.detector.base import Detection, DetectorResult
from src.detector.yolo_detector import YoloDetector
from src.inspection_logger import InspectionLogger
from src.ui_payload import build_event_payload, build_status_payload
from src.utils.drawing import render_inspection_overlay
from src.validation import ClipAttachmentValidator
from src.video_source import FramePacket


def piece(label, box, confidence=0.9):
    x1, y1, x2, y2 = box
    mask = np.zeros((100, 120), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return Detection(label, confidence, box, ((x1 + x2) // 2, (y1 + y2) // 2), mask=mask)


def scene(state):
    connector = piece("connector", (10, 10, 40, 40))
    cable = piece("cable", (75, 5, 85, 95))
    if state == "NOK":
        clip = piece("clip", (35, 20, 45, 30))
    elif state == "OK":
        clip = piece("clip", (75, 50, 85, 60))
    else:
        clip = piece("clip", (50, 70, 60, 80))
    return DetectorResult([connector, clip, cable], detector_name="yolo")


class ClipAttachmentTests(unittest.TestCase):
    def validator(self, **settings):
        return ClipAttachmentValidator({"connector_near_px": 0, **settings})

    def test_connector_attachment_is_nok_without_requiring_cable(self):
        result = scene("NOK")
        result.detections = result.detections[:2]
        verdict = self.validator().validate(result)
        self.assertEqual(verdict.status, "NOK")
        self.assertEqual(app_runner._derive_error_label(verdict), "clip_attached_to_connector")

    def test_clip_on_cable_away_from_connector_is_ok(self):
        self.assertEqual(self.validator().validate(scene("OK")).status, "OK")

    def test_copresence_without_contact_is_default_ok(self):
        self.assertEqual(self.validator().validate(scene("unattached")).status, "OK")

    def test_overlapping_boxes_now_trigger_nok_despite_partial_masks(self):
        result = scene("unattached")
        for detection in result.detections:
            detection.bbox = (0, 0, 120, 100)
        self.assertEqual(self.validator().validate(result).status, "NOK")

    def test_missing_or_low_confidence_target_is_default_ok(self):
        for index, label in ((0, "connector"), (1, "clip")):
            for kind in ("missing", "low_confidence", "nonfinite_confidence"):
                result = scene("NOK")
                if kind == "missing":
                    result.detections.pop(index)
                else:
                    result.detections[index].confidence = 0.1 if kind == "low_confidence" else float("nan")
                with self.subTest(label=label, kind=kind):
                    verdict = self.validator().validate(result)
                    self.assertEqual(verdict.status, "OK")
                    self.assertIn(label, verdict.missing_classes)
                    self.assertEqual(verdict.confidence, 0.0)
                    self.assertEqual(verdict.relation["decision_basis"], "default_ok")

    def test_missing_or_empty_mask_uses_available_box(self):
        for state in ("NOK", "OK"):
            for empty in (False, True):
                result = scene(state)
                result.detections[1].mask = np.zeros((100, 120), bool) if empty else None
                with self.subTest(state=state, empty=empty):
                    verdict = self.validator().validate(result)
                    self.assertEqual(verdict.status, state)
                    self.assertNotIn("clip", verdict.missing_classes)

    def test_cable_geometry_does_not_change_default_ok(self):
        result = scene("unattached")
        result.detections[2].bbox = (0, 0, 120, 100)
        self.assertEqual(self.validator().validate(result).status, "OK")

    def test_no_connector_is_default_ok(self):
        result = scene("OK")
        result.detections.pop(0)
        self.assertEqual(self.validator().validate(result).status, "OK")

    def test_no_cable_is_ok_when_targets_are_separated(self):
        result = scene("OK")
        result.detections.pop(2)
        verdict = self.validator().validate(result)
        self.assertEqual(verdict.status, "OK")
        self.assertIn("cable", verdict.missing_classes)
        self.assertEqual(verdict.confidence, 0.0)

    def test_empty_detections_are_default_ok_with_diagnostics(self):
        verdict = self.validator().validate(DetectorResult([]))
        self.assertEqual(verdict.status, "OK")
        self.assertEqual(verdict.missing_classes, ["connector", "clip", "cable"])
        self.assertEqual(verdict.relation["pairs"], [])
        self.assertEqual(verdict.confidence, 0.0)
        self.assertIn("OK par defaut", verdict.details[0])

    def test_single_pixel_connector_contact_is_immediately_nok(self):
        result = DetectorResult([
            piece("connector", (10, 10, 40, 40)),
            piece("clip", (39, 39, 49, 49)),
            piece("cable", (39, 45, 60, 60)),
        ])
        self.assertEqual(self.validator().validate(result).status, "NOK")

    def test_mask_contact_can_trigger_nok_even_if_boxes_are_partial(self):
        result = scene("NOK")
        result.detections[1].bbox = (90, 90, 100, 99)
        self.assertEqual(self.validator().validate(result).status, "NOK")

    def test_nok_takes_priority_even_if_clip_also_touches_cable(self):
        result = scene("NOK")
        result.detections[2] = piece("cable", (35, 5, 45, 95))
        self.assertEqual(self.validator().validate(result).status, "NOK")

    def test_decisions_are_immediate_without_stale_ok_or_nok(self):
        validator = self.validator()
        self.assertEqual(validator.validate(scene("OK")).status, "OK")
        self.assertEqual(validator.validate(scene("NOK")).status, "NOK")
        self.assertEqual(validator.validate(DetectorResult([])).status, "OK")
        self.assertEqual(validator.validate(scene("NOK")).status, "NOK")
        self.assertEqual(validator.validate(scene("OK")).status, "OK")

    def test_unattached_extra_clip_does_not_prevent_default_ok(self):
        result = scene("OK")
        result.detections.append(piece("clip", (50, 70, 60, 80)))
        self.assertEqual(self.validator().validate(result).status, "OK")

    def test_near_box_distance_does_not_modify_masks(self):
        result = DetectorResult([piece("connector", (10, 10, 40, 40)), piece("clip", (53, 20, 63, 30))])
        originals = [d.mask.copy() for d in result.detections]
        self.assertEqual(self.validator().validate(result).status, "OK")
        verdict = self.validator(connector_near_px=20).validate(result)
        self.assertEqual(verdict.status, "NOK")
        self.assertEqual(verdict.relation["pairs"][0]["box_gap_px"], 13)
        for original, detection in zip(originals, result.detections):
            np.testing.assert_array_equal(original, detection.mask)

    def test_distance_is_euclidean_between_edges_not_centers(self):
        result = DetectorResult([piece("connector", (10, 10, 40, 40)), piece("clip", (52, 56, 62, 66))])
        self.assertEqual(self.validator(connector_near_px=20).validate(result).status, "NOK")
        self.assertEqual(self.validator(connector_near_px=19.99).validate(result).status, "OK")

    def test_touching_box_edges_are_contact(self):
        result = DetectorResult([piece("connector", (10, 10, 40, 40)), piece("clip", (40, 20, 50, 30))])
        self.assertEqual(self.validator().validate(result).status, "NOK")

    def test_nok_priority_checks_all_clips_before_any_ok(self):
        result = scene("OK")
        result.detections.append(piece("clip", (35, 20, 45, 30)))
        self.assertEqual(self.validator().validate(result).status, "NOK")

    def test_near_connector_overrules_simultaneous_cable_contact(self):
        result = DetectorResult([piece("connector", (10, 10, 40, 40)),
                                 piece("clip", (53, 20, 63, 30)), piece("cable", (53, 5, 63, 95))])
        self.assertEqual(self.validator(connector_near_px=20).validate(result).status, "NOK")
        self.assertEqual(self.validator(connector_near_px=12).validate(result).status, "OK")

    def test_bad_settings_and_misaligned_masks_are_rejected(self):
        for key, value in (("min_confidence", float("nan")), ("connector_near_px", -1),
                           ("connector_near_px", float("inf")), ("connector_near_px", float("nan"))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validator(**{key: value})
        result = scene("OK")
        result.detections[1].mask = np.ones((10, 10), bool)
        with self.assertRaisesRegex(ValueError, "same frame"):
            self.validator().validate(result)
        result = scene("OK")
        result.detections[1].bbox = (1, 1, 1, 10)
        with self.assertRaisesRegex(ValueError, "positive dimensions"):
            self.validator().validate(result)

    def test_binary_payloads_are_serializable_and_explain_decision(self):
        for result, status, basis in ((DetectorResult([]), "OK", "default_ok"),
                                      (scene("NOK"), "NOK", "connector_contact")):
            verdict = self.validator().validate(result)
            for payload in (build_status_payload(verdict, "offline", 0, "video.MOV"),
                            build_event_payload(verdict, "offline", 0, "video.MOV", "status_change")):
                with self.subTest(status=status):
                    self.assertEqual(json.loads(json.dumps(payload))["status"], status)
                    self.assertEqual(payload["relation"]["method"], "box_distance_and_mask_intersection")
                    self.assertEqual(payload["relation"]["decision_basis"], basis)

    def test_masks_and_status_do_not_cover_or_modify_source_image(self):
        frame = np.full((100, 120, 3), 180, np.uint8)
        original = frame.copy()
        result = scene("NOK")
        verdict = self.validator().validate(result)
        painted = render_inspection_overlay(frame, [], result.detections, verdict, "offline", 0,
                                            draw_zones=False, draw_boxes=False, show_labels=False)
        self.assertEqual(painted.shape, (212, 120, 3))
        np.testing.assert_array_equal(frame, original)
        changed = (painted[64:164] != original).any(axis=2)
        union = np.logical_or.reduce([d.mask for d in result.detections])
        self.assertTrue(changed[union].all())
        self.assertFalse(changed[~union].any())

    def test_default_logger_initializes_and_records_indeterminate_on_mac(self):
        with tempfile.TemporaryDirectory() as temp, chdir(temp), patch("src.inspection_logger.sys.platform", "darwin"):
            logger = InspectionLogger()
            logger.log_result("INDETERMINE", "clip non visible")
            content = logger.log_file.read_text()
            self.assertIn("INDETERMINE: clip non visible", content)
            self.assertNotIn("Test NOK", content)

    def test_logger_appends_ok_and_nok_without_erasing_history(self):
        with tempfile.TemporaryDirectory() as temp:
            log_file = Path(temp) / "IACom.txt"
            log_file.write_text("Historique existant\n", encoding="utf-8")
            logger = InspectionLogger(log_dir=temp)
            logger.log_from_validation_result(SimpleNamespace(status="OK", details=[]))
            logger.log_from_validation_result(SimpleNamespace(status="NOK", details=["clip sur connecteur"]))
            lines = log_file.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 3)
            self.assertEqual(lines[0], "Historique existant")
            self.assertTrue(lines[1].endswith(" - Test OK"))
            self.assertTrue(lines[2].endswith(" - Test NOK: clip sur connecteur"))

    def test_default_windows_logger_uses_c_test(self):
        with patch("src.inspection_logger.sys", SimpleNamespace(platform="win32")), \
                patch("src.inspection_logger.Path") as path:
            InspectionLogger()
            path.assert_called_once_with("C:/test")
            path.return_value.mkdir.assert_called_once_with(parents=True, exist_ok=True)
            path.return_value.__truediv__.assert_called_once_with("IACom.txt")


class YoloMaskTests(unittest.TestCase):
    def infer(self, masks, box_count=1):
        with tempfile.TemporaryDirectory() as temp:
            model_file = Path(temp) / "test.pt"
            model_file.touch()
            boxes = [SimpleNamespace(xyxy=np.array([[2, 3, 10, 15]]), cls=np.array([1]), conf=np.array([0.8]))
                     for _ in range(box_count)]
            tensor = Mock()
            tensor.detach.return_value.cpu.return_value.numpy.return_value = masks
            prediction = SimpleNamespace(boxes=boxes, names={1: "clip"},
                                         masks=SimpleNamespace(data=tensor) if masks is not None else None)
            with patch("src.detector.yolo_detector.YOLO") as factory:
                factory.return_value.predict.return_value = [prediction]
                detector = YoloDetector(str(model_file), {})
                result = detector.infer(np.zeros((20, 30, 3), np.uint8))
                self.assertTrue(factory.return_value.predict.call_args.kwargs["retina_masks"])
                return result

    def test_native_masks_are_copied_without_polygon_reconstruction(self):
        mask = np.zeros((1, 20, 30), np.float32)
        mask[0, 2:6, 3:8] = 1
        mask[0, 10:15, 20:25] = 1
        result = self.infer(mask)
        self.assertEqual(result.detections[0].label, "clip")
        np.testing.assert_array_equal(result.detections[0].mask, mask[0].astype(bool))

    def test_no_detections_is_supported(self):
        self.assertEqual(self.infer(None, 0).detections, [])

    def test_masks_must_match_source_resolution(self):
        with self.assertRaisesRegex(ValueError, "not aligned"):
            self.infer(np.ones((1, 10, 10), np.float32))


class ApplicationTests(unittest.TestCase):
    def execute(self, *, display=False, fail=False, task="segment", loop=False):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            video = Path(temp) / "test.MOV"
            video.touch()
            args = run.build_runtime_args("offline", video=str(video), no_display=not display,
                                          no_mqtt=True, max_frames=3)
            config = {"runtime": {"show_window": True, "save_nok_snapshots": True, "log_dir": "data/logs"},
                      "validation": {"mode": "clip_attachment", "clip_attachment": {
                          "connector_near_px": 0}}}
            source = Mock()
            source.read.side_effect = [FramePacket(np.zeros((100, 120, 3), np.uint8), 0 if loop else i, str(video))
                                      for i in range(3)] + [None]
            detector = Mock(name="detector")
            detector.name = "yolo"
            detector.model_path = None
            detector.model = SimpleNamespace(task=task, names={0: "connector", 1: "clip", 2: "cable"})
            detector.infer.side_effect = ([RuntimeError("inference failed")] if fail else
                                         [scene("NOK")] * 3 if loop else [DetectorResult([]), scene("NOK"), scene("OK")])
            publisher = Mock(enabled=False)
            mocks = {}
            replacements = {"setup_logger": Mock(), "_load_yaml": config, "VideoSource": source,
                            "_create_detector": detector, "_create_person_masker": None, "MqttPublisher": publisher,
                            "InspectionLogger": Mock(), "save_snapshot": str(Path(temp) / "nok.jpg"),
                            "encode_frame_to_base64": "image"}
            for name, value in replacements.items():
                mocks[name] = stack.enter_context(patch.object(app_runner, name, return_value=value))
            stack.enter_context(patch.object(app_runner, "_load_json", side_effect=[{"zones": []}, {}]))
            gui = {name: stack.enter_context(patch.object(app_runner.cv2, name)) for name in
                   ("namedWindow", "resizeWindow", "imshow", "waitKey", "getWindowProperty", "destroyAllWindows")}
            gui["waitKey"].return_value = ord("q")
            try:
                app_runner.run_application(args)
            finally:
                if task == "segment":
                    source.release.assert_called_once()
                    publisher.close.assert_called_once()
            return publisher, gui, mocks

    def test_windows_keeps_default_inspection_log_directory(self):
        with patch.object(app_runner, "sys", SimpleNamespace(platform="win32")):
            _, _, mocks = self.execute()
        self.assertIsNone(mocks["InspectionLogger"].call_args.kwargs["log_dir"])
        self.assertEqual(mocks["InspectionLogger"].return_value.log_from_validation_result.call_count, 3)

    def test_default_ok_counted_as_ok_and_only_nok_saved(self):
        publisher, gui, mocks = self.execute()
        messages = publisher.publish.call_args_list
        statuses = [call.args[1]["status"] for call in messages if call.args[0] == "status"]
        self.assertEqual(statuses, ["OK", "NOK", "OK"])
        metrics = [call.args[1] for call in messages if call.args[0] == "metrics"][-1]
        self.assertEqual([metrics[key] for key in ("ok_count", "nok_count", "indeterminate_count", "total_frames")],
                         [2, 1, 0, 3])
        self.assertEqual(metrics["last_error_label"], "clip_attached_to_connector")
        mocks["save_snapshot"].assert_called_once()
        gui["imshow"].assert_not_called()
        self.assertFalse(mocks["MqttPublisher"].call_args.args[0]["enabled"])

    def test_visible_preview_quits_and_cleans_up_on_q(self):
        _, gui, _ = self.execute(display=True)
        gui["namedWindow"].assert_called_once()
        gui["imshow"].assert_called_once()
        gui["destroyAllWindows"].assert_called_once()

    def test_loop_restart_does_not_delay_nok(self):
        publisher, _, _ = self.execute(loop=True)
        statuses = [call.args[1]["status"] for call in publisher.publish.call_args_list if call.args[0] == "status"]
        self.assertEqual(statuses, ["NOK"] * 3)

    def test_failure_releases_resources(self):
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            self.execute(fail=True)

    def test_wrong_model_task_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires a segmentation model"):
            self.execute(task="detect")

    def test_launch_options_reach_runner(self):
        argv = ["run.py", "offline", "--video", "test2.MOV", "--no-mqtt", "--no-display", "--max-frames", "5"]
        with patch("sys.argv", argv), patch.object(run, "load_app_config", return_value={}), patch.object(run, "run_inspection") as start:
            run.main()
        args = start.call_args.args[0]
        self.assertTrue(args.no_mqtt and args.no_display)
        self.assertEqual(args.max_frames, 5)
        self.assertEqual(args.video, "test2.MOV")


if __name__ == "__main__":
    unittest.main()
