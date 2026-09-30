#!/usr/bin/env python3
"""Start a composed MeshLLM product and run the Laya golden read battery."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
PARITY = ROOT / "scripts" / "skippy-laya-parity.py"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def get_json(url: str, timeout: float = 2.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        value = json.loads(response.read())
    if not isinstance(value, dict):
        raise ValueError(f"{url} returned a non-object JSON value")
    return value


def wait_for_model(process: subprocess.Popen[str], api_port: int, timeout: float) -> str:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{api_port}/v1/models"
    while time.monotonic() < deadline:
        exit_status = process.poll()
        if exit_status is not None:
            raise RuntimeError(f"mesh-llm exited during startup with status {exit_status}")
        try:
            models = get_json(url)
            rows = models.get("data")
            if isinstance(rows, list) and rows and isinstance(rows[0], dict):
                model_id = rows[0].get("id")
                if isinstance(model_id, str) and model_id:
                    return model_id
        except (OSError, ValueError, urllib.error.URLError):
            pass
        time.sleep(1)
    raise TimeoutError(f"mesh-llm did not publish a model within {timeout:.0f}s")


def stop(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--startup-timeout", type=float, default=300.0)
    parser.add_argument("--read-timeout", type=float, default=300.0)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    binary = args.mesh_binary.resolve()
    model = args.model.resolve()
    if not binary.is_file():
        parser.error(f"mesh-llm binary not found: {binary}")
    if not model.is_file():
        parser.error(f"Laya GGUF not found: {model}")
    if not PARITY.is_file():
        parser.error(f"Laya parity driver not found: {PARITY}")

    api_port = free_port()
    console_port = free_port()
    # A Windows child may retain the inherited log handle briefly after the
    # server exits. Preserve the actual smoke result instead of replacing it
    # with WinError 32 while removing this disposable runner directory.
    with tempfile.TemporaryDirectory(
        prefix="mesh-laya-smoke-", ignore_cleanup_errors=sys.platform == "win32"
    ) as directory:
        state = Path(directory)
        log_path = state / "mesh-llm.log"
        parity_path = args.json_out or state / "laya-parity.json"
        environment = {
            **os.environ,
            "MESH_LLM_CONFIG": str(state / "config.toml"),
            "MESH_LLM_RUNTIME_ROOT": str(state / "runtime"),
            "MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR": str(binary.parent / "native-runtimes"),
            # The smoke must consume only the runtime bundled beside the host.
            "MESH_LLM_NATIVE_RUNTIME_MANIFEST_URL": "http://127.0.0.1:9/native-runtimes.json",
        }
        if args.device == "Vulkan0":
            # The GPU smoke runner may lack vulkaninfo. Admit the explicit
            # Vulkan selection; loading the model and golden reads still
            # require a working Vulkan device and packaged runtime.
            environment["MESH_LLM_VULKAN_AVAILABLE"] = "1"
        command = [
            str(binary),
            "--log-format",
            "json",
            "serve",
            "--gguf",
            str(model),
            "--no-draft",
            "--device",
            args.device,
            "--ctx-size",
            "1024",
            "--port",
            str(api_port),
            "--console",
            str(console_port),
            "--headless",
        ]
        print(f"starting Laya smoke on {args.device}: {' '.join(command)}")
        failure: Exception | None = None
        model_id: str | None = None
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                model_id = wait_for_model(process, api_port, args.startup_timeout)
                parity = subprocess.run(
                    [
                        sys.executable,
                        str(PARITY),
                        "--base-url",
                        f"http://127.0.0.1:{api_port}",
                        "--model",
                        model_id,
                        "--timeout",
                        str(args.read_timeout),
                        "--json-out",
                        str(parity_path),
                    ],
                    cwd=ROOT,
                    env=environment,
                    text=True,
                    check=False,
                )
                if parity.returncode != 0:
                    raise RuntimeError(f"Laya parity battery exited {parity.returncode}")
            except Exception as error:  # Preserve the server log for CI diagnosis.
                failure = error
            finally:
                stop(process)

        if failure is not None:
            print(f"Laya product smoke failed: {failure}", file=sys.stderr)
            print(log_path.read_text(encoding="utf-8", errors="replace")[-12000:], file=sys.stderr)
            return 1

        print(f"Laya product smoke passed: model={model_id} device={args.device}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
