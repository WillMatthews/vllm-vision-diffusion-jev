# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen bounded Vast rentals, run a local job driver, and always destroy them.

The job driver receives MACHINE_RUN_DIR and must wait for its remote work and
retrieve artifacts before exiting. Without --job, a passing probe is destroyed.
Credentials stay local. The detached timer requires a live local systemd user
manager; it is a safeguard, not a provider-enforced spending cap.
"""

import argparse
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import uuid
from contextlib import suppress
from pathlib import Path

from control import Vast


class ScreeningFailed(RuntimeError):
    pass


class JobFailed(RuntimeError):
    pass


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def offer_query(max_hourly):
    return {
        "limit": 20,
        "type": "on-demand",
        "verified": {"eq": True},
        "rentable": {"eq": True},
        "rented": {"eq": False},
        "num_gpus": {"eq": 1},
        "gpu_name": {"eq": "RTX 4090"},
        "cuda_max_good": {"gte": 13.0},
        "disk_space": {"gte": 100},
        "cpu_ram": {"gte": 64000},
        "cpu_cores_effective": {"gte": 8},
        "inet_down": {"gte": 1000},
        "reliability": {"gte": 0.99},
        "dph_total": {"lte": max_hourly},
        "inet_down_cost": {"lte": 0.005},
        "inet_up_cost": {"lte": 0.005},
        "order": [["dph_total", "asc"]],
        "allocated_storage": 100,
    }


def arm_timer(run, hours, credential_file):
    unit = json.loads((run / "plan.json").read_text())["label"] + "-cleanup"
    helper = run / "destroy.sh"
    control = Path(__file__).with_name("control.py").absolute()
    helper.write_text(
        "#!/bin/zsh\nset -eu\nsource "
        + shlex.quote(str(credential_file))
        + "\nexport VAST_API_KEY\nexec "
        + shlex.join([sys.executable, str(control), "destroy", "--run", str(run)])
        + "\n"
    )
    helper.chmod(0o700)
    subprocess.run(
        [
            "systemd-run",
            "--user",
            "--unit",
            unit,
            f"--on-active={int(hours * 3600)}s",
            "--timer-property=AccuracySec=1s",
            "--property=Restart=on-failure",
            "--property=RestartSec=20s",
            str(helper),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", unit + ".timer"], check=True
    )
    return unit


def ssh_args(run, instance):
    return [
        "ssh",
        "-x",
        "-i",
        str(run / "id_ed25519"),
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "UserKnownHostsFile=" + str(run / "known_hosts"),
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=10",
        "-o",
        "ServerAliveCountMax=2",
        "-p",
        str(instance["ssh_port"]),
        "root@" + instance["ssh_host"],
    ]


def remote(run, instance, command, timeout=30, stdin=None):
    return subprocess.run(
        [*ssh_args(run, instance), command],
        input=stdin,
        capture_output=True,
        timeout=timeout,
        check=True,
    ).stdout


def wait_ready(api, run, label, deadline):
    attached = set()
    while time.monotonic() < deadline:
        instances = api.instances(label)
        if len(instances) > 1:
            raise RuntimeError("Multiple matching instances; refusing to select one")
        if instances:
            instance = instances[0]
            if instance["id"] not in attached:
                response = api.call(
                    "POST",
                    f"/instances/{instance['id']}/ssh/",
                    {"ssh_key": (run / "id_ed25519.pub").read_text()},
                )
                if response.get("success"):
                    attached.add(instance["id"])
            if instance.get("actual_status") == "running":
                try:
                    remote(run, instance, "mkdir -p /workspace/acid-test", timeout=15)
                    return instance
                except (subprocess.SubprocessError, OSError):
                    pass
        time.sleep(5)
    raise ScreeningFailed("Provisioning/SSH deadline expired")


def screen(run, instance, download_url, timeout):
    probe = Path(__file__).with_name("machine_probe.py").read_bytes()
    remote(run, instance, "cat > /workspace/acid-test/machine_probe.py", stdin=probe)
    # Setup is bounded and logs are downloaded even on failure. No API key is sent.
    command = """set -eu
cd /workspace/acid-test
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
apt-get update
apt-get install -y build-essential python3.12-dev
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
uv venv --python 3.12
.venv/bin/python machine_probe.py --phase preflight --output preflight.json \\
    --work-dir /workspace --download-url DOWNLOAD
uv pip install torch==2.13.0 torchvision==0.28.0 \\
    --index-url https://download.pytorch.org/whl/cu130
.venv/bin/python machine_probe.py --phase full --output probe.json \\
    --work-dir /workspace --download-url DOWNLOAD
""".replace("DOWNLOAD", shlex.quote(download_url))
    remote(run, instance, "cat > /workspace/acid-test/run.sh", stdin=command.encode())
    error = None
    try:
        remote(
            run,
            instance,
            f"timeout {int(timeout)}s bash /workspace/acid-test/run.sh "
            "> /workspace/acid-test/setup.log 2>&1",
            timeout=timeout + 30,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        error = exc
    for name in ("preflight.json", "probe.json", "setup.log"):
        with suppress(subprocess.SubprocessError, OSError):
            (run / name).write_bytes(
                remote(run, instance, "cat /workspace/acid-test/" + name)
            )
    if error:
        raise ScreeningFailed("Probe/setup failed; see setup.log") from error
    result = json.loads((run / "probe.json").read_text())
    if not result["passed"]:
        raise ScreeningFailed("Machine performance gate failed; see probe.json")
    return result


def network_bytes(run, instance):
    output = remote(
        run,
        instance,
        "for dev in /sys/class/net/*; do "
        '[ "${dev##*/}" = lo ] && continue; '
        'cat "$dev/statistics/rx_bytes" "$dev/statistics/tx_bytes"; done',
    )
    values = [int(value) for value in output.split()]
    if not values or any(value < 0 for value in values):
        raise RuntimeError("Cannot verify container network usage")
    return sum(values)


def create_and_use(api, label, request, offer_id, use):
    """Never retry an ambiguous create; confirm cleanup before returning control."""
    try:
        try:
            response = api.call("PUT", f"/asks/{offer_id}/", request)
        except urllib.error.HTTPError as exc:
            if exc.code in (400, 404, 409):
                raise ScreeningFailed(f"Offer rejected (HTTP {exc.code})") from exc
            raise
        if not response.get("success"):
            raise ScreeningFailed("Provider declined this offer")
        return use()
    finally:
        # A create timeout may hide a successful rental. Cleanup failure propagates
        # and must stop the loop; otherwise a retry could create parallel charges.
        api.destroy(label)


def run_job(command, run, api, label, check_budget):
    env = dict(os.environ, MACHINE_RUN_DIR=str(run))
    process = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        while process.poll() is None:
            check_budget()
            if not api.instances(label):
                raise RuntimeError("Rental disappeared while job was running")
            time.sleep(15)
        if process.returncode:
            raise JobFailed(f"Job driver failed with exit code {process.returncode}")
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--rent", action="store_true")
    parser.add_argument("--budget-usd", type=float, default=10)
    parser.add_argument("--screening-usd", type=float, default=1.50)
    parser.add_argument("--max-hourly", type=float, default=0.75)
    parser.add_argument("--max-hours", type=float, default=8)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--download-url")
    parser.add_argument("--workload-probe", type=Path)
    parser.add_argument("--credential-file", type=Path, default=Path.home() / ".tokens")
    parser.add_argument("--job", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    for value in (args.budget_usd, args.screening_usd, args.max_hourly, args.max_hours):
        if not math.isfinite(value) or value <= 0:
            parser.error("Budgets, rates and deadlines must be finite and positive")
    if args.max_hourly * args.max_hours + 3 > args.budget_usd:
        parser.error("Reserve at least $3 beyond worst-case GPU/disk runtime")
    if not 1 <= args.max_attempts <= 3 or args.screening_usd > 1.5:
        parser.error("Screening is limited to three hosts and $1.50")
    api = Vast(os.environ["VAST_API_KEY"])
    if not args.rent:
        offers = api.call("POST", "/bundles", offer_query(args.max_hourly))["offers"]
        print(
            json.dumps(
                [
                    {
                        k: offer.get(k)
                        for k in (
                            "id",
                            "machine_id",
                            "gpu_name",
                            "dph_total",
                            "cpu_ram",
                            "disk_bw",
                        )
                    }
                    for offer in offers
                ],
                indent=2,
            )
        )
        return
    if not args.run or not args.download_url or not args.credential_file.is_file():
        parser.error("Rental needs a fresh --run, --download-url and credential file")
    run = args.run.absolute()
    run.mkdir(parents=True, exist_ok=False)
    run.chmod(0o700)
    label = "visjev-pilot-" + uuid.uuid4().hex[:12]
    save(run / "plan.json", {"label": label, "budget_usd": args.budget_usd})
    subprocess.run(
        ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(run / "id_ed25519")],
        check=True,
    )
    start = time.monotonic()
    initial_credit = api.call("GET", "/users/current/")["credit"]
    save(run / "balance-before.json", {"credit": initial_credit})
    api.destroy(label)  # Exercise scoped cleanup against the empty fresh label.
    unit = arm_timer(run, args.max_hours, args.credential_file.absolute())
    journal = []
    used_machines = set()
    transfer_bytes = 0
    uncertain_create = False

    def check_budget(screening=False):
        if transfer_bytes > 100 * 1024**3:
            raise RuntimeError("Cumulative transfer limit reached; no retry allowed")
        elapsed = time.monotonic() - start
        credit_delta = initial_credit - api.call("GET", "/users/current/")["credit"]
        # Elapsed time at the maximum rate bounds all sequential rentals, including
        # provider billing lag. Reserve $1 for transfers and $2 for cleanup/lag.
        conservative = max(
            credit_delta,
            elapsed / 3600 * args.max_hourly + transfer_bytes / 1e9 * 0.005,
        )
        limit = args.screening_usd if screening else args.budget_usd - 3
        if conservative >= limit or elapsed >= args.max_hours * 3600 - 120:
            raise RuntimeError("Budget/deadline reached; no further work authorized")
        return conservative

    try:
        for attempt in range(args.max_attempts):
            spent = check_budget(screening=True)
            remaining_seconds = (
                (args.screening_usd - spent - 0.15) / args.max_hourly * 3600
            )
            if remaining_seconds < 180:
                raise RuntimeError("Insufficient screening reserve for another host")
            screening_deadline = time.monotonic() + min(1500, remaining_seconds)
            offers = api.call("POST", "/bundles", offer_query(args.max_hourly))[
                "offers"
            ]
            offer = next(
                (o for o in offers if o["machine_id"] not in used_machines), None
            )
            if offer is None:
                raise RuntimeError("No untried qualifying host available")
            used_machines.add(offer["machine_id"])
            current = run / f"attempt-{attempt + 1}"
            current.mkdir()
            for name in ("id_ed25519", "id_ed25519.pub", "known_hosts"):
                (current / name).symlink_to(run / name)
            request = {
                "client_id": "me",
                "image": "nvidia/cuda:13.0.2-runtime-ubuntu24.04",
                "disk": 100,
                "label": label,
                "runtype": "ssh",
                "target_state": "running",
                "cancel_unavail": True,
                "onstart": "chown root:root /root/.ssh /root/.ssh/authorized_keys; "
                "chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys",
            }
            record = {"offer": offer, "started_at": time.time(), "status": "creating"}
            save(current / "plan.json", {"label": label, "request": request})
            journal.append(record)
            save(run / "journal.json", journal)

            def use(
                current=current,
                record=record,
                offer=offer,
                screening_deadline=screening_deadline,
            ):
                nonlocal uncertain_create
                uncertain_create = False
                nonlocal transfer_bytes
                instance = wait_ready(
                    api,
                    current,
                    label,
                    min(time.monotonic() + 480, screening_deadline),
                )
                save(
                    current / "connection.json",
                    {
                        k: instance[k]
                        for k in ("id", "ssh_host", "ssh_port", "dph_total")
                    },
                )
                check_budget(screening=True)

                def monitor(screening=False):
                    current_bytes = network_bytes(current, instance)
                    if transfer_bytes + current_bytes > 100 * 1024**3:
                        raise RuntimeError("100 GiB cumulative transfer limit reached")
                    if screening and time.monotonic() >= screening_deadline:
                        raise ScreeningFailed("Screening workload deadline expired")
                    check_budget(screening)

                try:
                    timeout = min(720, screening_deadline - time.monotonic() - 120)
                    if timeout < 60:
                        raise ScreeningFailed("Insufficient setup time remains")
                    screen(current, instance, args.download_url, timeout=timeout)
                    monitor(screening=True)
                    if args.workload_probe:
                        try:
                            run_job(
                                [sys.executable, str(args.workload_probe.absolute())],
                                current,
                                api,
                                label,
                                lambda: monitor(screening=True),
                            )
                        except JobFailed as exc:
                            raise ScreeningFailed(
                                "Training workload probe failed"
                            ) from exc
                    record["status"] = "passed"
                    save(run / "journal.json", journal)
                    print(
                        f"Accepted machine {offer['machine_id']}: {current}", flush=True
                    )
                    if args.job:
                        run_job(args.job, current, api, label, monitor)
                finally:
                    # Failure to read counters stops the run rather than assuming
                    # unmeasured transfer was free. Outer cleanup still executes.
                    transfer_bytes += network_bytes(current, instance)
                    record["cumulative_transfer_bytes"] = transfer_bytes
                return True

            try:
                uncertain_create = True
                create_and_use(api, label, request, offer["id"], use)
            except ScreeningFailed as exc:
                uncertain_create = False
                record["status"] = "rejected"
                record["reason"] = str(exc)
                print(f"Rejected and destroyed: {exc}", flush=True)
            else:
                record["status"] = "complete_destroyed"
                return
            finally:
                record["ended_at"] = time.time()
                save(run / "journal.json", journal)
        raise RuntimeError("All screening attempts failed; no rental retained")
    finally:
        api.destroy(label)
        save(
            run / "ending.json",
            {
                "remaining_scoped_instances": len(api.instances(label)),
                "observed_credit_delta": initial_credit
                - api.call("GET", "/users/current/")["credit"],
                "finished_at": time.time(),
            },
        )
        if not uncertain_create:
            subprocess.run(["systemctl", "--user", "stop", unit + ".timer"], check=True)
        else:
            print(
                "Create response was ambiguous; cleanup timer remains armed", flush=True
            )


if __name__ == "__main__":
    main()
