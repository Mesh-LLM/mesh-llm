"""Controller-owned admission estimates and host guard for family certification.

These are conservative scheduling allowances, not measured memory guarantees.
Core parity releases its full-model oracle before loading partitioned stages;
state handoff releases its source before restore. Non-chat certification keeps
candidate and oracle resident together. The guard observes host availability
throughout execution because allocations and unrelated workloads can exceed an
estimate. Never modify the source-owned canonical policy plan here.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

GIB = 1024 ** 3
TIERS = (128, 256)


def positive_bytes(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"missing or invalid {name}")
    return value


def artifact_bytes(artifact):
    if not isinstance(artifact, dict) or not artifact.get("files"):
        raise ValueError("missing pinned artifact files for memory estimate")
    files = artifact["files"]
    if len(files) != len(set(files)):
        raise ValueError("duplicate pinned artifact files")
    integrity = artifact.get("file_integrity", {})
    return sum(positive_bytes(integrity.get(name, {}).get("size_bytes"),
                              f"artifact size for {name}") for name in files)


def memory_estimate(model):
    weights = max(artifact_bytes(model["artifact"]),
                  positive_bytes(model["resources"]["estimated_model_bytes"], "model bytes"))
    auxiliaries = sum(artifact_bytes(model[key]) for key in ("mmproj_artifact", "draft_artifact")
                      if model.get(key))
    model_class = model["class"]
    if model_class not in {"causal_generation", "embedding", "rerank", "encoder_decoder",
                          "ocr", "speech_synthesis", "speech_recognition"}:
        raise ValueError(f"unknown memory execution class: {model_class}")
    # Causal stages partition one model; reserve 25% for duplicated tensors,
    # KV/recurrent state and scratch plus 2 GiB per process (driver + 2 stages).
    # Non-chat runs two full copies with 2048-context candidate/oracle buffers.
    copies = 1 if model_class == "causal_generation" else 2
    processes = 3 if copies == 1 else 2
    resident = copies * (weights + auxiliaries)
    allowance = (resident + 3) // 4 + processes * 2 * GIB
    peak = resident + allowance
    return {"resident_model_bytes": resident, "runtime_allowance_bytes": allowance,
            "estimated_peak_bytes": peak}


def tier_for(peak):
    positive_bytes(peak, "estimated peak bytes")
    for tier in TIERS:
        if peak <= tier * GIB * 85 // 100:
            return f"accelerator-memory-{tier}plus"
    raise ValueError(f"estimated peak {peak / GIB:.2f} GiB exceeds largest runner budget (217.6 GiB)")


def placement(model):
    estimate = memory_estimate(model)
    try:
        return {**estimate, "memory_tier": tier_for(estimate["estimated_peak_bytes"])}
    except ValueError as error:
        raise ValueError(f"{model['family']}: {error}") from error


def parse_vm_stat(text):
    match = re.search(r"page size of (\d+) bytes", text)
    if not match:
        raise ValueError("cannot read vm_stat page size")
    fields = dict(re.findall(r"^(Pages [^:]+):\s+(\d+)\.", text, re.MULTILINE))
    # Inactive/speculative pages are reclaimable; do not double-count purgeable
    # pages (a subset) or treat compressed/wired memory as available.
    keys = ("Pages free", "Pages inactive", "Pages speculative")
    if any(key not in fields for key in keys):
        raise ValueError("cannot read vm_stat available pages")
    return int(match[1]) * sum(int(fields[key]) for key in keys)


def host_memory():
    if sys.platform != "darwin":
        raise ValueError("family memory guard requires a macOS runner")
    total = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True, timeout=10))
    available = parse_vm_stat(subprocess.check_output(["vm_stat"], text=True, timeout=10))
    positive_bytes(total, "host physical memory")
    if not 0 <= available <= total:
        raise ValueError("invalid available memory observation")
    return total, available


def admission(estimate, total, available):
    reserve = (total * 15 + 99) // 100
    peak = estimate["estimated_peak_bytes"]
    if peak > total - reserve:
        raise ValueError("assigned host is too small for the family with 15% reserved")
    if peak > available - reserve:
        raise ValueError("insufficient available host memory after reserving 15%; other workloads must finish")
    return reserve


def group_running(pgid):
    rows = subprocess.check_output(["ps", "-axo", "pgid=,stat="], text=True, timeout=10)
    return any(int(fields[0]) == pgid and not fields[1].startswith("Z")
               for row in rows.splitlines() if len(fields := row.split()) == 2)


def stop_group(process):
    # The battery's timeout wrappers supervise their own nested process groups.
    # Give those wrappers time to forward TERM and finish their bounded cleanup,
    # even when the outer shell has already exited. Waiting on the shell alone
    # would kill the wrappers too soon and strand model-serving descendants.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait(timeout=5)
        return
    deadline = time.monotonic() + 15
    while group_running(process.pid) and time.monotonic() < deadline:
        process.poll()
        time.sleep(0.05)
    if group_running(process.pid):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            # macOS can report EPERM for a group containing only zombies.
            if group_running(process.pid):
                raise
    process.wait(timeout=5)
    deadline = time.monotonic() + 5
    while group_running(process.pid):
        if time.monotonic() >= deadline:
            raise RuntimeError("family process group did not stop")
        time.sleep(0.05)


def guarded_run(model, expected_tier, command, evidence, *, cwd=None):
    import fcntl
    evidence.mkdir(parents=True, exist_ok=True)
    report = {"family": model["family"], "reserve_percent": 15, "status": "failed"}
    process = None
    # macOS per-user temp root is outside runner workspaces and survives jobs.
    import tempfile
    lock_path = Path(tempfile.gettempdir()) / f"mesh-canary-family-{os.getuid()}.lock"
    try:
        with lock_path.open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("another family certification holds the host lock") from error
            estimate = placement(model)
            report.update(estimate)
            if estimate["memory_tier"] != expected_tier:
                raise ValueError("scheduled memory tier differs from verified family estimate")
            total, available = host_memory()
            reserve = admission(estimate, total, available)
            report.update(physical_bytes=total, initial_available_bytes=available,
                          reserve_bytes=reserve, minimum_available_bytes=available)
            print(json.dumps(report, sort_keys=True), flush=True)
            previous = {}
            received_signal = None
            def interrupted(signum, _frame):
                # Never interrupt Popen ownership transfer or reenter wait locks.
                nonlocal received_signal
                if received_signal is None:
                    received_signal = signum
            try:
                for signum in (signal.SIGTERM, signal.SIGINT):
                    previous[signum] = signal.signal(signum, interrupted)
                process = subprocess.Popen(command, cwd=cwd, start_new_session=True)
                while True:
                    if received_signal is not None:
                        raise InterruptedError(f"family interrupted by signal {received_signal}")
                    if process.poll() is not None:
                        break
                    observed_total, available = host_memory()
                    report["minimum_available_bytes"] = min(report["minimum_available_bytes"], available)
                    if observed_total != total or available < reserve:
                        raise ValueError("host available memory crossed the 15% reserve; stopping family")
                    time.sleep(1)
                report["exit_code"] = process.returncode
                report["status"] = "passed" if process.returncode == 0 else "failed"
                return process.returncode
            finally:
                if process is not None:
                    stop_group(process)
                for signum, handler in previous.items():
                    signal.signal(signum, handler)
    except (ValueError, OSError, subprocess.SubprocessError, RuntimeError) as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        (evidence / "memory-admission.json").write_text(json.dumps(report, indent=2) + "\n")
