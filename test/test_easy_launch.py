"""Launcher checks without YOLO inference, videos, or a running MQTT service."""
from contextlib import ExitStack
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

import run
from scripts import start_broker


class EasyLaunchTests(unittest.TestCase):
    def test_no_arguments_starts_continuous_demo(self):
        config = {"source": {"demo_videos": ["test1.MOV", "test2.MOV"]}}
        with patch("sys.argv", ["run.py"]), patch.object(run, "load_app_config", return_value=config), \
                patch.object(run, "run_inspection") as start:
            run.main()
        args = start.call_args.args[0]
        self.assertEqual(args.videos, config["source"]["demo_videos"])
        self.assertTrue(args.loop)
        self.assertFalse(args.no_display)
        self.assertFalse(args.no_mqtt)

    def test_menu_remains_explicit(self):
        self.assertEqual(run.build_parser().parse_args(["menu"]).preset, "menu")

    def test_skips_automatic_broker_when_disabled_or_custom(self):
        cases = [({"enabled": False}, True), ({}, False),
                 ({"broker": {"host": "192.168.1.10"}}, True),
                 ({"broker": {"port": 1884}}, True)]
        with patch.object(start_broker.subprocess, "Popen") as spawn, \
                patch.object(start_broker, "port_open") as probe:
            for config, enabled in cases:
                with self.subTest(config=config, enabled=enabled):
                    with start_broker.local_broker(config, enabled=enabled):
                        pass
            spawn.assert_not_called()
            probe.assert_not_called()

    def test_existing_service_is_never_started_or_stopped(self):
        with patch.object(start_broker, "port_open", return_value=True), \
                patch.object(start_broker.subprocess, "Popen") as spawn:
            with start_broker.local_broker({}):
                pass
            spawn.assert_not_called()

    def broker_mocks(self, stack, *, ports=(False, True), polls=(None, None)):
        process = Mock()
        process.poll.side_effect = polls
        stack.enter_context(patch.object(start_broker, "port_open", side_effect=ports))
        spawn = stack.enter_context(patch.object(start_broker.subprocess, "Popen", return_value=process))
        return process, spawn

    def test_uses_venv_python_and_stops_owned_broker(self):
        with ExitStack() as stack:
            process, spawn = self.broker_mocks(stack)
            with start_broker.local_broker({}):
                process.terminate.assert_not_called()
            self.assertEqual(spawn.call_args.args[0][0], sys.executable)
            self.assertEqual(spawn.call_args.kwargs["cwd"], start_broker.PROJECT_ROOT)
            process.terminate.assert_called_once()
            process.wait.assert_called_once_with(timeout=5)

    def test_inspection_failure_still_stops_owned_broker(self):
        with ExitStack() as stack:
            process, _ = self.broker_mocks(stack)
            with self.assertRaisesRegex(RuntimeError, "inspection failed"):
                with start_broker.local_broker({}):
                    raise RuntimeError("inspection failed")
            process.terminate.assert_called_once()

    def test_startup_failure_does_not_launch_inspection(self):
        with ExitStack() as stack:
            process, _ = self.broker_mocks(stack, ports=(False,), polls=(1, 1))
            with self.assertRaisesRegex(RuntimeError, "pas demarre"):
                with start_broker.local_broker({}):
                    self.fail("Inspection must not start after broker failure")
            process.terminate.assert_not_called()
            process.wait.assert_called_once_with(timeout=5)

    def test_startup_timeout_cleans_up_child(self):
        with ExitStack() as stack:
            process, _ = self.broker_mocks(stack, ports=(False, False))
            stack.enter_context(patch.object(start_broker.time, "monotonic", side_effect=[0, 16]))
            with self.assertRaisesRegex(RuntimeError, "Delai"):
                with start_broker.local_broker({}):
                    self.fail("Inspection must not start before broker is ready")
            process.terminate.assert_called_once()

    def test_stuck_child_is_killed_and_reaped(self):
        with ExitStack() as stack:
            process, _ = self.broker_mocks(stack)
            process.wait.side_effect = [subprocess.TimeoutExpired("broker", 5), 0]
            with start_broker.local_broker({}):
                pass
            process.kill.assert_called_once()
            self.assertEqual(process.wait.call_count, 2)

    def test_cmd_uses_only_project_node_red(self):
        script = (Path(__file__).resolve().parents[1] / "node-red.cmd").read_text()
        self.assertIn('call npm.cmd --prefix "%~dp0nodered" start -- %*', script)
        self.assertIn("exit /b %errorlevel%", script)


if __name__ == "__main__":
    unittest.main()
