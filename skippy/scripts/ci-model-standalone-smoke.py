#!/usr/bin/env python3
"""Run a pinned model through a composed standalone CPU product."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def completion_evidence(response: dict, model_id: str) -> dict:
    require(response.get("model") == model_id, "completion model differs from selected model")
    choices = response.get("choices")
    require(isinstance(choices, list) and len(choices) == 1, "completion must have one choice")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    require(isinstance(message, dict) and isinstance(message.get("content"), str)
            and bool(message["content"].strip()), "completion has no generated text")
    usage = response.get("usage")
    require(isinstance(usage, dict) and type(usage.get("prompt_tokens")) is int
            and usage["prompt_tokens"] > 0 and type(usage.get("completion_tokens")) is int
            and usage["completion_tokens"] > 0, "completion lacks positive prefill and decode usage")
    return {"prompt_tokens": usage["prompt_tokens"], "completion_tokens": usage["completion_tokens"]}


def request_json(url: str, payload: dict | None = None) -> dict:
    body = None if payload is None else json.dumps(payload).encode()
    request = Request(url, data=body, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=30) as response:
        value = json.load(response)
    require(isinstance(value, dict), "API response must be an object")
    return value


def run(product_dir: Path, model: Path, model_sha256: str, model_id: str, suite: str) -> dict:
    manifest_path = product_dir / "product-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("contract") == "skippy-product-v1" and manifest.get("backend") == "cpu",
            "model pilot requires a composed CPU product")
    binary = product_dir / manifest["cli"]["path"]
    runtime = product_dir / manifest["runtime"]["path"]
    require(binary.is_file() and not binary.is_symlink() and runtime.is_dir() and not runtime.is_symlink(),
            "composed CLI or runtime is missing")
    require(sha256(binary) == manifest["cli"]["sha256"], "composed CLI digest differs")
    require(model.is_file() and sha256(model) == model_sha256, "pinned model digest differs")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base = f"http://127.0.0.1:{port}/v1"
    with tempfile.TemporaryDirectory(prefix="skippy-dense-ci-") as temporary:
        log_path = Path(temporary) / "serve.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [str(binary), "--runtime-bundle", str(runtime), "--runtime-cache", str(Path(temporary) / "runtime-cache"),
                 "--runtime-selection", "cpu", "serve", "--model-path", str(model),
                 "--model-id", model_id, "--ctx-size", "512", "--n-gpu-layers", "0",
                 "--bind-addr", f"127.0.0.1:{port}"],
                stdout=log, stderr=subprocess.STDOUT,
            )
            try:
                deadline = time.monotonic() + 120
                while time.monotonic() < deadline:
                    require(process.poll() is None, f"Skippy exited during startup; {log_path.read_text(errors='replace')[-4000:]}")
                    try:
                        models = request_json(f"{base}/models")
                        require(any(item.get("id") == model_id for item in models.get("data", [])
                                    if isinstance(item, dict)), "selected model absent from standalone API")
                        break
                    except (URLError, TimeoutError):
                        time.sleep(1)
                else:
                    raise ValueError(f"Skippy startup timed out; {log_path.read_text(errors='replace')[-4000:]}")
                response = request_json(f"{base}/chat/completions", {
                    "model": model_id, "messages": [{"role": "user", "content": "Say hello."}],
                    "max_tokens": 8, "temperature": 0,
                })
                usage = completion_evidence(response, model_id)
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    return {"schema_version": 1, "suite": suite, "source_sha": manifest["source_sha"],
            "product_manifest_sha256": sha256(manifest_path), "model_id": model_id,
            "model_sha256": model_sha256, "backend": "cpu", "cases": ["load", "prefill-decode"],
            "usage": usage}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--suite", choices=("dense-pilot", "recurrent-pilot", "moe-pilot"), required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    try:
        evidence = run(args.product_dir.resolve(), args.model.resolve(), args.model_sha256, args.model_id, args.suite)
        args.evidence.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"standalone model smoke failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
