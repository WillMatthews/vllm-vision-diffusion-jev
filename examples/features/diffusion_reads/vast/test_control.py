# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify cleanup scope, retry, and post-delete verification without a rental."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from control import Vast
from select_machine import ScreeningFailed, create_and_use, main


class CleanupTests(unittest.TestCase):
    def test_delete_only_run_instances_and_verify_absence(self):
        api = Vast("test-key")
        calls = []
        instances = [{"id": 1, "label": "this-run"}, {"id": 2, "label": "other"}]

        def call(method, path, body=None):
            calls.append((method, path))
            if method == "GET":
                return {"instances": list(instances)}
            self.assertEqual(path, "/instances/1/")
            instances.pop(0)
            return {"success": True}

        with patch.object(api, "call", side_effect=call), patch("control.time.sleep"):
            api.destroy("this-run")
        self.assertEqual(instances, [{"id": 2, "label": "other"}])
        self.assertEqual(calls[-1], ("GET", "/instances/"))
        self.assertEqual(sum(method == "DELETE" for method, _ in calls), 1)

    def test_network_failure_retries_and_absent_run_is_idempotent(self):
        api = Vast("test-key")
        with (
            patch.object(
                api, "call", side_effect=[OSError(), {"instances": []}]
            ) as call,
            patch("control.time.sleep"),
        ):
            api.destroy("this-run")
        self.assertEqual(call.call_count, 2)

    def test_unconfirmed_cleanup_fails_loudly(self):
        api = Vast("test-key")
        with (
            patch.object(api, "call", side_effect=OSError()),
            patch("control.time.sleep"),
            self.assertRaisesRegex(RuntimeError, "NOT confirmed"),
        ):
            api.destroy("this-run")


class RentalLifecycleTests(unittest.TestCase):
    def test_lost_create_response_destroys_the_rental_without_running_job(self):
        """An API timeout can occur after the provider creates a paid instance."""
        api = Vast("test-key")
        instances = []
        use = Mock()

        def call(method, path, body=None):
            if method == "PUT":
                instances.append({"id": 42, "label": "this-run"})
                raise TimeoutError("Create response lost")
            if method == "GET":
                return {"instances": list(instances)}
            self.assertEqual((method, path), ("DELETE", "/instances/42/"))
            instances.clear()
            return {"success": True}

        with (
            patch.object(api, "call", side_effect=call),
            patch("control.time.sleep"),
            self.assertRaisesRegex(TimeoutError, "response lost"),
        ):
            create_and_use(api, "this-run", {}, 123, use)
        self.assertEqual(instances, [])
        use.assert_not_called()

    def run_fake_selection(self, root, api, screening):
        """Run the public controller entry point with all external IO replaced."""
        credentials = root / "credentials"
        credentials.write_text("unused test credential file")
        argv = [
            "select_machine.py",
            "--rent",
            "--run",
            str(root / "run"),
            "--credential-file",
            str(credentials),
            "--download-url",
            "https://example.com/not-downloaded",
            "--max-hours",
            "1",
        ]
        instance = {
            "id": 42,
            "ssh_host": "unused",
            "ssh_port": 22,
            "dph_total": 0.4,
        }
        with (
            patch("select_machine.sys.argv", argv),
            patch.dict("select_machine.os.environ", {"VAST_API_KEY": "unused"}),
            patch("select_machine.Vast", return_value=api),
            patch("select_machine.subprocess.run"),
            patch("select_machine.arm_timer", return_value="unused"),
            patch("select_machine.wait_ready", return_value=instance),
            patch("select_machine.network_bytes", return_value=1024),
            patch("select_machine.screen", side_effect=screening),
        ):
            main()

    def fake_api(self):
        api = Mock()
        api.instances.return_value = []

        def call(method, path, body=None):
            if path == "/users/current/":
                return {"credit": 100}
            if path == "/bundles":
                return {
                    "offers": [
                        {"id": 1, "machine_id": 10, "dph_total": 0.4},
                        {"id": 2, "machine_id": 20, "dph_total": 0.4},
                    ]
                }
            if method == "PUT":
                return {"success": True}
            self.fail(f"Unexpected API call: {method} {path}")

        api.call.side_effect = call
        return api

    def test_rejected_host_is_destroyed_before_renting_the_next_host(self):
        api = self.fake_api()
        events = []
        original_call = api.call.side_effect

        def call(method, path, body=None):
            if method == "PUT":
                events.append(path)
            return original_call(method, path, body)

        api.call.side_effect = call
        api.destroy.side_effect = lambda label: events.append("destroyed")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.run_fake_selection(
                root, api, [ScreeningFailed("Disk too slow"), {"passed": True}]
            )
            journal = json.loads((root / "run" / "journal.json").read_text())
            self.assertEqual(
                [entry["status"] for entry in journal],
                ["rejected", "complete_destroyed"],
            )
        self.assertEqual(
            events,
            [
                "destroyed",
                "/asks/1/",
                "destroyed",
                "/asks/2/",
                "destroyed",
                "destroyed",
            ],
        )

    def test_unconfirmed_cleanup_stops_selection_before_another_paid_create(self):
        api = self.fake_api()
        api.destroy.side_effect = [
            None,
            RuntimeError("Cleanup NOT confirmed"),
            RuntimeError("Cleanup NOT confirmed"),
        ]
        with (
            tempfile.TemporaryDirectory() as tmp,
            self.assertRaisesRegex(RuntimeError, "Cleanup NOT confirmed"),
        ):
            self.run_fake_selection(Path(tmp), api, [ScreeningFailed("Disk too slow")])
        creates = [call for call in api.call.call_args_list if call.args[0] == "PUT"]
        self.assertEqual(len(creates), 1)

    def test_unexpected_probe_error_aborts_instead_of_churning_rentals(self):
        api = self.fake_api()
        with (
            tempfile.TemporaryDirectory() as tmp,
            self.assertRaisesRegex(ValueError, "Invalid report"),
        ):
            self.run_fake_selection(Path(tmp), api, [ValueError("Invalid report")])
        creates = [call for call in api.call.call_args_list if call.args[0] == "PUT"]
        self.assertEqual(len(creates), 1)
        api.destroy.assert_called()


if __name__ == "__main__":
    unittest.main()
