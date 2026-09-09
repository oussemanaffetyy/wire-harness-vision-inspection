"""Playlist continuity and MQTT contracts without cameras or network services."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import paho.mqtt.client as mqtt

import run
from src.messaging.mqtt_publisher import MqttPublisher
from src.video_source import VideoSource


class PlaylistTests(unittest.TestCase):
    def capture_factory(self, counts, opened=True):
        captures = []

        def create(path):
            capture = Mock()
            capture.isOpened.return_value = opened
            frame = np.full((8, 12, 3), 10 if path == "a.MOV" else 20, np.uint8)
            capture.read.side_effect = [(True, frame.copy()) for _ in range(counts[path])] + [(False, None)]
            capture.get.return_value = 0
            captures.append(capture)
            return capture

        return create, captures

    def test_no_gap_between_files_and_global_indices(self):
        factory, captures = self.capture_factory({"a.MOV": 2, "b.MOV": 3})
        source = VideoSource("offline", video_paths=["a.MOV", "b.MOV"], loop_video=False)
        with patch("src.video_source.cv2.VideoCapture", side_effect=factory):
            source.open()
            packets = [source.read() for _ in range(5)]
            self.assertIsNone(source.read())
            source.release()
        self.assertEqual([p.frame_index for p in packets], list(range(5)))
        self.assertEqual([p.source_name for p in packets], ["a.MOV"] * 2 + ["b.MOV"] * 3)
        self.assertEqual([int(p.frame[0, 0, 0]) for p in packets], [10, 10, 20, 20, 20])
        for capture in captures:
            capture.release.assert_called_once()

    def test_loop_restarts_playlist_without_resetting_global_index(self):
        factory, _ = self.capture_factory({"a.MOV": 1, "b.MOV": 1})
        source = VideoSource("offline", video_paths=["a.MOV", "b.MOV"], loop_video=True)
        with patch("src.video_source.cv2.VideoCapture", side_effect=factory):
            source.open()
            packets = [source.read() for _ in range(5)]
            source.release()
        self.assertEqual([p.frame_index for p in packets], list(range(5)))
        self.assertEqual([p.source_name for p in packets], ["a.MOV", "b.MOV", "a.MOV", "b.MOV", "a.MOV"])

    def test_empty_video_raises_instead_of_spinning_or_skipping(self):
        factory, _ = self.capture_factory({"a.MOV": 0})
        source = VideoSource("offline", video_path="a.MOV", loop_video=True)
        with patch("src.video_source.cv2.VideoCapture", side_effect=factory):
            source.open()
            with self.assertRaisesRegex(RuntimeError, "no readable frames"):
                source.read()
            source.release()

    def test_unopenable_next_file_releases_both_captures(self):
        factory, captures = self.capture_factory({"a.MOV": 1, "b.MOV": 0})

        def create(path):
            capture = factory(path)
            capture.isOpened.return_value = path == "a.MOV"
            return capture

        source = VideoSource("offline", video_paths=["a.MOV", "b.MOV"], loop_video=False)
        with patch("src.video_source.cv2.VideoCapture", side_effect=create):
            source.open()
            source.read()
            with self.assertRaisesRegex(RuntimeError, "b.MOV"):
                source.read()
            source.release()
        for capture in captures:
            capture.release.assert_called_once()

    def test_demo_arguments_enable_mqtt_and_loop_unless_once(self):
        config = {"source": {"demo_videos": ["a.MOV", "b.MOV"]}}
        for once in (False, True):
            argv = ["run.py", "demo"] + (["--once"] if once else [])
            with patch("sys.argv", argv), patch.object(run, "load_app_config", return_value=config), \
                    patch.object(run, "run_inspection") as start:
                run.main()
                args = start.call_args.args[0]
                self.assertEqual(args.videos, ["a.MOV", "b.MOV"])
                self.assertEqual(args.loop, not once)
                self.assertFalse(args.no_mqtt)
                self.assertEqual(args.mode, "offline")
                start.assert_called_once()


class MqttTests(unittest.TestCase):
    def config(self):
        return {"enabled": True, "topics": {"status": "test/status", "video_stream": "test/video", "events": "test/events"}}

    def test_success_uses_connack_and_flushes_before_disconnect(self):
        publisher = MqttPublisher(self.config(), Mock())
        with patch("src.messaging.mqtt_publisher.mqtt.Client") as factory:
            client = factory.return_value
            client.loop_start.side_effect = lambda: client.on_connect(client, None, None, SimpleNamespace(is_failure=False), None)
            publisher.connect()
            self.assertTrue(publisher.connected)
            self.assertEqual(factory.call_args.args[0], mqtt.CallbackAPIVersion.VERSION2)
            client.publish.return_value.rc = mqtt.MQTT_ERR_SUCCESS
            for key, retain in (("status", True), ("video_stream", True), ("events", False)):
                payload = {"status": "OK", "image_base64": "jpeg"}
                self.assertTrue(publisher.publish(key, payload))
                self.assertEqual(json.loads(client.publish.call_args.args[1]), payload)
                self.assertEqual(client.publish.call_args.kwargs["retain"], retain)
            publisher.close()
            client.publish.return_value.wait_for_publish.assert_called_once_with(timeout=2)
            client.disconnect.assert_called_once()
            client.loop_stop.assert_called_once()
            self.assertFalse(publisher.connected)

    def test_offline_retry_and_reconnect_do_not_publish_stale_frames(self):
        publisher = MqttPublisher(self.config(), Mock())
        with patch("src.messaging.mqtt_publisher.mqtt.Client") as factory, patch.object(publisher._ready, "wait", return_value=False):
            publisher.connect()
            client = factory.return_value
            self.assertFalse(publisher.connected)
            self.assertFalse(publisher.publish("status", {"status": "OK"}))
            client.publish.assert_not_called()
            publisher._on_connect(client, None, None, SimpleNamespace(is_failure=False), None)
            self.assertTrue(publisher.connected)
            publisher._on_disconnect(client, None, None, SimpleNamespace(is_failure=True), None)
            self.assertFalse(publisher.connected)
            publisher._on_connect(client, None, None, SimpleNamespace(is_failure=False), None)
            self.assertTrue(publisher.connected)
            publisher.close()

    def test_broker_rejection_and_disabled_mode_are_not_connected(self):
        publisher = MqttPublisher({**self.config(), "enabled": False}, Mock())
        with patch("src.messaging.mqtt_publisher.mqtt.Client") as factory:
            publisher.connect()
            factory.assert_not_called()
        self.assertFalse(publisher.publish("status", {"status": "OK"}))
        publisher._on_connect(None, None, None, SimpleNamespace(is_failure=True), None)
        self.assertFalse(publisher.connected)

    def test_flow_matches_python_topics_and_has_unique_client_id(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "config/mqtt.json").read_text())
        nodes = json.loads((root / "nodered/wire_harness_dashboard_flow.json").read_text())
        self.assertTrue(config["enabled"])
        self.assertEqual(sum(n["type"] == "ui-base" for n in nodes), 1)
        ids = {n["id"] for n in nodes}
        self.assertEqual(len(ids), len(nodes))
        for node in nodes:
            if node["type"] == "mqtt-broker":
                self.assertNotEqual(node["clientid"], config["broker"]["client_id"])
                self.assertEqual(node["broker"], config["broker"]["host"])
            for wire in node.get("wires", []):
                self.assertTrue(set(wire) <= ids)
        self.assertEqual({n["topic"] for n in nodes if n["type"] == "mqtt in"},
                         {config["topics"][k] for k in ("status", "video_stream")})


if __name__ == "__main__":
    unittest.main()
