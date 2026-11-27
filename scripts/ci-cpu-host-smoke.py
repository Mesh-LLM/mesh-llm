#!/usr/bin/env python3
"""Smoke a MeshLLM host with the pinned Skippy CPU runtime on any platform."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener


def isolated_environment(work: Path, cache: Path) -> dict[str, str]:
    env = os.environ.copy()
    for name in (
        "MESH_LLM_NATIVE_RUNTIME_DIR",
        "MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR",
        "MESH_LLM_NATIVE_RUNTIME_MANIFEST_URL",
        "MESH_LLM_NO_RUNTIME_DOWNLOAD",
        "SKIPPY_NATIVE_RUNTIME_BUNDLE_DIR",
        "SKIPPY_NATIVE_RUNTIME_CACHE_DIR",
        "SKIPPY_NATIVE_RUNTIME_MANIFEST_URL",
    ):
        env.pop(name, None)
    for name, path in (
        ("HOME", work / "home"),
        ("USERPROFILE", work / "home"),
        ("APPDATA", work / "appdata"),
        ("LOCALAPPDATA", work / "localappdata"),
        ("XDG_CACHE_HOME", work / "cache-home"),
    ):
        path.mkdir(exist_ok=True)
        env[name] = str(path)
    env["MESH_LLM_NATIVE_RUNTIME_CACHE_DIR"] = str(cache)
    return env


def install_cpu_runtime(host: Path, cache: Path, env: dict[str, str]) -> str:
    result = subprocess.run(
        [str(host), "runtime", "install", "cpu", "--cache-dir", str(cache), "--json"],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"CPU runtime install failed: {result.stdout}\n{result.stderr}")
    manifests = list(cache.rglob("manifest.json"))
    cpu_ids = []
    for manifest in manifests:
        runtime = json.loads(manifest.read_text(encoding="utf-8"))["runtime"]
        if runtime["backend"]["kind"] == "cpu":
            cpu_ids.append(runtime["id"])
    if len(cpu_ids) != 1:
        raise RuntimeError(f"expected one installed CPU runtime, found {cpu_ids}")
    return cpu_ids[0]


def available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def get_json(url: str, body: dict | None = None) -> dict:
    payload = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"} if payload else {},
    )
    with build_opener(ProxyHandler({})).open(request, timeout=120 if payload else 10) as reply:
        return json.load(reply)


def wait_for_model(process: subprocess.Popen, port: int) -> str:
    deadline = time.monotonic() + 240
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"host exited during startup with status {process.returncode}")
        try:
            models = get_json(url).get("data", [])
            if models and models[0].get("id"):
                return models[0]["id"]
        except (OSError, URLError):
            pass
        time.sleep(1)
    raise RuntimeError("host did not advertise a model within 240 seconds")


def stop(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def smoke(host: Path, model: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="mesh-cpu-smoke-") as temporary:
        work = Path(temporary).resolve()
        cache = work / "runtime-cache"
        cache.mkdir()
        binary = work / ("mesh-llm.exe" if os.name == "nt" else "mesh-llm")
        shutil.copy2(host, binary)
        env = isolated_environment(work, cache)
        runtime_id = install_cpu_runtime(binary, cache, env)
        env["MESH_LLM_NO_RUNTIME_DOWNLOAD"] = "1"
        port = available_port()
        log_path = work / "host.log"
        try:
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    [
                        str(binary),
                        "serve",
                        "--local-model-only",
                        "--gguf",
                        str(model),
                        "--port",
                        str(port),
                        "--log-format",
                        "json",
                    ],
                    cwd=work,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                try:
                    model_id = wait_for_model(process, port)
                    reply = get_json(
                        f"http://127.0.0.1:{port}/v1/chat/completions",
                        {
                            "model": model_id,
                            "max_tokens": 24,
                            "messages": [{"role": "user", "content": "What is the capital of France?"}],
                        },
                    )
                    if process.poll() is not None:
                        raise RuntimeError("host exited during the completion")
                    content = reply.get("choices", [{}])[0].get("message", {}).get("content")
                    tokens = reply.get("usage", {}).get("completion_tokens", 0)
                    if reply.get("model") != model_id or not content or tokens < 1:
                        raise RuntimeError(f"invalid completion: {reply}")
                    print(f"CPU runtime: {runtime_id}")
                    print(f"model: {model_id}")
                    print(f"reply: {content}")
                    print("CPU host smoke OK")
                finally:
                    stop(process)
        except Exception:
            if log_path.exists():
                print(log_path.read_text(encoding="utf-8", errors="replace")[-5000:], file=sys.stderr)
            raise


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} <mesh-llm-binary> <model.gguf>")
    host, model = (Path(value).resolve() for value in sys.argv[1:])
    if not host.is_file() or not model.is_file():
        raise SystemExit("host binary and GGUF model must exist")
    smoke(host, model)


if __name__ == "__main__":
    main()
