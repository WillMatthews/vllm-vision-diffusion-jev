# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify cleanup scope, retry, and post-delete verification without a rental."""

import unittest
from unittest.mock import patch

from control import Vast


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


if __name__ == "__main__":
    unittest.main()
