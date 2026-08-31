#!/usr/bin/env python3
"""Compare old and new staged native-MTP execution with an identical fixture."""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * quantile + 0.999999) - 1))
    return ordered[index]


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    elapsed = [
        float(row["elapsed_ms"])
        for row in result["per_request"]
        if "error" not in row
    ]
    return {
        "requests": result["requests"],
        "successful": result["successful"],
        "failed": result["failed"],
        "throughput_rps": result["throughput_rps"],
        "latency_p50_ms": statistics.median(elapsed) if elapsed else None,
        "latency_p99_ms": percentile(elapsed, 0.99),
        "drafted": result["drafted"],
        "accepted": result["accepted"],
        "acceptance_rate": result["acceptance_rate"],
    }


def write_config(
    path: Path,
    package: Path,
    port: int,
    lanes: int,
    model_id: str,
    layer_start: int,
    layer_end: int,
) -> None:
    config = {
        "run_id": "mtp-scheduler-benchmark",
        "topology_id": "mtp-scheduler-benchmark-final-stage",
        "model_id": model_id,
        "package_ref": str(package),
        "model_path": str(package),
        "stage_id": "stage-final",
        "stage_index": 1,
        "layer_start": layer_start,
        "layer_end": layer_end,
        "ctx_size": 256 * lanes,
        "lane_count": lanes,
        # Verification executes the target token plus at least one proposal in
        # one native batch, even when the concurrency sweep starts at N=1.
        "n_batch": max(8, lanes),
        "n_ubatch": max(8, lanes),
        "n_gpu_layers": 0,
        "mmap": True,
        "mlock": False,
        "cache_type_k": "f16",
        "cache_type_v": "f16",
        "flash_attn_type": "disabled",
        "filter_tensors_on_load": True,
        "selected_device": {"backend_device": "CPU"},
        "native_mtp_enabled": True,
        "load_mode": "layer-package",
        "bind_addr": f"127.0.0.1:{port}",
        "upstream": {
            "stage_id": "stage-prev",
            "stage_index": 0,
            "endpoint": "tcp://127.0.0.1:19000",
        },
        "downstream": None,
    }
    path.write_text(json.dumps(config, indent=2) + "\n")


def wait_ready(addr: tuple[str, int], proc: subprocess.Popen[str], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}")
        try:
            with socket.create_connection(addr, timeout=1) as sock:
                sock.settimeout(1)
                magic = sock.recv(4)
                if len(magic) == 4:
                    return
        except OSError:
            time.sleep(0.25)
    raise TimeoutError(f"server did not become ready at {addr[0]}:{addr[1]}")


def run_server(
    label: str,
    server_bin: Path,
    client_bin: Path,
    package: Path,
    native_build: Path,
    output_dir: Path,
    sweep: list[int],
    requests: int,
    model_id: str,
    layer_start: int,
    layer_end: int,
    activation_width: int,
) -> dict[str, Any]:
    port = free_port()
    lanes = max(sweep)
    config = output_dir / f"{label}-stage.json"
    log_path = output_dir / f"{label}-server.log"
    write_config(config, package, port, lanes, model_id, layer_start, layer_end)
    command = [
        str(server_bin),
        "serve-binary",
        "--config",
        str(config),
        "--bind-addr",
        f"127.0.0.1:{port}",
        "--activation-width",
        str(activation_width),
        "--activation-wire-dtype",
        "f32",
        "--max-inflight",
        str(lanes),
        "--telemetry-level",
        "debug",
    ]
    env = os.environ.copy()
    env["LLAMA_STAGE_BUILD_DIR"] = str(native_build)
    env["SKIPPY_NATIVE_MTP_GREEDY_SAMPLING_FASTPATH"] = "1"
    env["SKIPPY_TELEMETRY_STDERR"] = "1"
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        try:
            wait_ready(("127.0.0.1", port), proc, 900)
            rows = []
            for concurrency in sweep:
                completed = subprocess.run(
                    [
                        str(client_bin),
                        "--addr",
                        f"127.0.0.1:{port}",
                        "--requests",
                        str(max(requests, concurrency)),
                        "--concurrency",
                        str(concurrency),
                        "--activation-width",
                        str(activation_width),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=1800,
                )
                result = json.loads(completed.stdout)
                rows.append(
                    {
                        "concurrency": concurrency,
                        "metrics": summarize(result),
                        "per_request": result["per_request"],
                    }
                )
            return {
                "label": label,
                "binary": str(server_bin),
                "config": str(config),
                "log": str(log_path),
                "concurrency_sweep": rows,
            }
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


def parity(old: dict[str, Any], new: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for old_row, new_row in zip(old["concurrency_sweep"], new["concurrency_sweep"]):
        old_requests = {
            row["request_id"]: row
            for row in old_row["per_request"]
            if "error" not in row
        }
        new_requests = {
            row["request_id"]: row
            for row in new_row["per_request"]
            if "error" not in row
        }
        comparable = sorted(old_requests.keys() & new_requests.keys())
        fields = ["predicted", "draft", "verified", "accepted"]
        exact = sum(
            old_requests[index].get(key) == new_requests[index].get(key)
            for index in comparable
            for key in fields
        )
        exact_requests = sum(
            all(old_requests[index].get(key) == new_requests[index].get(key) for key in fields)
            for index in comparable
        )
        rows.append(
            {
                "concurrency": old_row["concurrency"],
                "comparable_requests": len(comparable),
                "exact_requests": exact_requests,
                "exact_field_matches": exact,
                "exact_field_total": len(comparable) * len(fields),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-bin", type=Path, required=True)
    parser.add_argument("--new-bin", type=Path, required=True)
    parser.add_argument("--client-bin", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--native-build", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--concurrency", default="1,2,4,8")
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument(
        "--model-id", default="meshllm/GLM-5.2-Q2_K-MTP-Q8-layers"
    )
    parser.add_argument("--layer-start", type=int, default=74)
    parser.add_argument("--layer-end", type=int, default=78)
    parser.add_argument("--activation-width", type=int, default=6144)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sweep = [int(value) for value in args.concurrency.split(",")]
    old = run_server(
        "old",
        args.old_bin,
        args.client_bin,
        args.package,
        args.native_build,
        args.output_dir,
        sweep,
        args.requests,
        args.model_id,
        args.layer_start,
        args.layer_end,
        args.activation_width,
    )
    new = run_server(
        "new",
        args.new_bin,
        args.client_bin,
        args.package,
        args.native_build,
        args.output_dir,
        sweep,
        args.requests,
        args.model_id,
        args.layer_start,
        args.layer_end,
        args.activation_width,
    )
    result = {
        "model_id": args.model_id,
        "package": str(args.package),
        "layer_range": [args.layer_start, args.layer_end],
        "activation_width": args.activation_width,
        "old": old,
        "new": new,
        "parity": parity(old, new),
    }
    output = args.output_dir / "comparison.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
