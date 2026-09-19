# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vast API helper. Credentials stay local; cleanup targets one saved run label."""

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


class Vast:
    def __init__(self, token, base="https://console.vast.ai/api/v0"):
        self.token = token
        self.base = base

    def call(self, method, path, body=None):
        req = urllib.request.Request(
            self.base + path,
            method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={
                "Authorization": "Bearer " + self.token,
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)

    def instances(self, label):
        return [
            i
            for i in self.call("GET", "/instances/")["instances"]
            if i.get("label") == label
        ]

    def destroy(self, label):
        """Retry teardown and verify absence, including lost create responses."""
        for attempt in range(12):
            try:
                matches = self.instances(label)
                if not matches:
                    return
                for instance in matches:
                    result = self.call("DELETE", f"/instances/{instance['id']}/", {})
                    if not result.get("success"):
                        raise RuntimeError("Vast did not acknowledge destruction")
                    print(
                        f"Destroy requested for instance {instance['id']}", flush=True
                    )
            except (OSError, ValueError, RuntimeError) as exc:
                print(f"Cleanup retry {attempt + 1}: {type(exc).__name__}", flush=True)
            time.sleep(5)
        raise RuntimeError("Cleanup NOT confirmed; check Vast console immediately")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["status", "destroy", "create"])
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.run / "plan.json").read_text())
    api = Vast(os.environ["VAST_API_KEY"])
    if args.action == "destroy":
        api.destroy(config["label"])
        print("Confirmed: no instances remain for this run")
    elif args.action == "status":
        fields = ("id", "label", "actual_status", "ssh_host", "ssh_port", "dph_total")
        print(
            json.dumps(
                [{k: i.get(k) for k in fields} for i in api.instances(config["label"])],
                indent=2,
            )
        )
    else:
        if (args.run / "create-attempted").exists():
            raise RuntimeError(
                "Create already attempted; inspect status before retrying"
            )
        if api.instances(config["label"]):
            raise RuntimeError("Run already has an instance")
        (args.run / "create-attempted").touch()
        # Never retry a create: a timeout may hide a successful paid rental.
        try:
            result = api.call("PUT", f"/asks/{config['offer_id']}/", config["request"])
        except urllib.error.HTTPError as exc:
            detail = json.loads(exc.read())
            safe = {k: detail.get(k) for k in ("error", "msg", "success")}
            (args.run / "create-error.json").write_text(json.dumps(safe, indent=2))
            raise RuntimeError(f"Vast rejected rental: {safe}") from None
        result = {k: result.get(k) for k in ("success", "new_contract")}
        (args.run / "created.json").write_text(json.dumps(result, indent=2))
        if not result.get("success"):
            raise RuntimeError("Vast rejected the rental; inspect created.json")
        print(json.dumps(result))


if __name__ == "__main__":
    main()
