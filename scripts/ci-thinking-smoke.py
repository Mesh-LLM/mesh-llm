#!/usr/bin/env python3
"""Run a focused thinking-enabled smoke suite against one mesh-llm model/backend."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_WAIT_SECONDS = 300
DEFAULT_REQUEST_TIMEOUT = 300


def pick_free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def http_json(url: str, payload: dict[str, Any] | None = None, timeout: int = 60) -> dict[str, Any]:
    if payload is None:
        request = urllib.request.Request(url)
    else:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def case_dir() -> Path | None:
    raw = os.environ.get("VALIDATION_CASE_DIR", "").strip()
    if not raw:
        return None
    return Path(raw)


def sync_runtime_logs(case_directory: Path | None, mesh_log_path: Path) -> None:
    if case_directory is None:
        return
    case_directory.mkdir(parents=True, exist_ok=True)
    if mesh_log_path.exists():
        shutil.copyfile(mesh_log_path, case_directory / "mesh.log")

    temp_dir = Path(tempfile.gettempdir())
    for source_name, target_name in (
        ("mesh-llm-llama-server.log", "llama-server.log"),
        ("mesh-llm-rpc-server.log", "rpc-server.log"),
    ):
        source_path = temp_dir / source_name
        if source_path.exists():
            shutil.copyfile(source_path, case_directory / target_name)


def model_root_for(model_arg: str) -> Path:
    path = Path(model_arg)
    return path if path.is_dir() else path.parent


def ensure_expected_template_source(model_arg: str, expected_template_source: str) -> None:
    model_root = model_root_for(model_arg)
    expected_path = model_root / expected_template_source
    if not expected_path.exists():
        print(
            f"❌ Expected template source file not found in model directory: {expected_template_source}",
            file=sys.stderr,
        )
        print(f"Model directory: {model_root}", file=sys.stderr)
        raise SystemExit(1)


def build_launch_command(args: argparse.Namespace, api_port: int, console_port: int) -> list[str]:
    command = [args.mesh_llm]
    if args.backend == "mlx":
        command.extend(["--mlx-file", args.model])
    else:
        command.extend(["--gguf-file", args.model, "--bin-dir", args.bin_dir])
    command.extend(["--no-draft", "--port", str(api_port), "--console", str(console_port)])
    return command


def wait_until_ready(process: subprocess.Popen[str], console_port: int, log_path: Path, timeout: int) -> None:
    status_url = f"http://127.0.0.1:{console_port}/api/status"
    for second in range(1, timeout + 1):
        sync_runtime_logs(case_dir(), log_path)
        if process.poll() is not None:
            print("❌ mesh-llm exited unexpectedly", file=sys.stderr)
            print(log_path.read_text(encoding="utf-8", errors="replace")[-8000:], file=sys.stderr)
            raise SystemExit(1)
        try:
            status = http_json(status_url, timeout=5)
            if bool(status.get("llama_ready", False)):
                print(f"✅ Model loaded in {second}s", flush=True)
                return
        except Exception:
            pass
        if second % 15 == 0:
            print(f"  Still waiting... ({second}s)", flush=True)
        time.sleep(1)
    sync_runtime_logs(case_dir(), log_path)
    print(f"❌ Model failed to load within {timeout}s", file=sys.stderr)
    print(log_path.read_text(encoding="utf-8", errors="replace")[-8000:], file=sys.stderr)
    raise SystemExit(1)


def write_progress(
    *,
    status: str,
    backend: str,
    model: str,
    check_count: int,
    completed_checks: int,
    failed_checks: int,
    current_label: str = "",
) -> None:
    out_dir = case_dir()
    if out_dir is None:
        return
    payload = {
        "status": status,
        "backend": backend,
        "model": model,
        "check_count": check_count,
        "completed_checks": completed_checks,
        "failed_check_count": failed_checks,
        "current_label": current_label,
    }
    (out_dir / "progress.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_chat(api_port: int, messages: list[dict[str, str]], max_tokens: int) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    payload = {
        "model": "any",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "top_k": 1,
        "seed": 123,
        "enable_thinking": True,
    }
    response = http_json(
        f"http://127.0.0.1:{api_port}/v1/chat/completions",
        payload=payload,
        timeout=DEFAULT_REQUEST_TIMEOUT,
    )
    choice = response["choices"][0]
    content = choice["message"]["content"]
    finish_reason = choice.get("finish_reason", "")
    return payload, response, content, finish_reason


def strip_tagged_reasoning(content: str) -> str:
    stripped = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    stripped = re.sub(r"<\|channel>thought.*?<channel\|>", "", stripped, flags=re.DOTALL)
    return stripped.strip()


def validate_case(
    *,
    api_port: int,
    case_cfg: dict[str, Any],
    thinking_mode: str,
    max_tokens_override: int | None,
) -> dict[str, Any]:
    label = str(case_cfg.get("label", "thinking"))
    messages = list(case_cfg.get("messages", []))
    max_tokens = int(case_cfg.get("max_tokens", 128) or 128)
    if max_tokens_override is not None:
        max_tokens = max_tokens_override
    if not messages:
        return {
            "label": label,
            "passed": False,
            "issues": ["missing messages"],
            "request": {},
            "response": {},
            "content": "",
            "finish_reason": "",
        }

    try:
        request_payload, response_payload, content, finish_reason = run_chat(api_port, messages, max_tokens)
    except Exception as exc:
        return {
            "label": label,
            "passed": False,
            "issues": [f"request failed: {exc}"],
            "request": {},
            "response": {},
            "content": "",
            "finish_reason": "",
        }

    issues: list[str] = []
    normalized = content.strip()
    if not normalized:
        issues.append("empty output")
    if not finish_reason:
        issues.append("missing finish_reason")
    if thinking_mode == "tagged":
        has_marker = "<think>" in content or "<|channel>thought" in content
        if not has_marker:
            issues.append("missing tagged reasoning marker")
        visible_answer = strip_tagged_reasoning(content)
        if not visible_answer:
            issues.append("missing answer outside reasoning tags")
    elif thinking_mode == "multiline":
        if "\n" not in normalized:
            issues.append("expected multiline reasoning output")

    return {
        "label": label,
        "passed": not issues,
        "issues": issues,
        "request": request_payload,
        "response": response_payload,
        "content": content,
        "finish_reason": finish_reason,
    }


def write_models_artifact(api_port: int) -> None:
    models = http_json(f"http://127.0.0.1:{api_port}/v1/models", timeout=DEFAULT_REQUEST_TIMEOUT)
    model_count = len(models.get("data", []))
    if model_count == 0:
        print("❌ No models in /v1/models", file=sys.stderr)
        raise SystemExit(1)
    artifact_root = case_dir()
    if artifact_root is not None:
        models_dir = artifact_root / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        (models_dir / "v1-models.json").write_text(
            json.dumps(models, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"✅ /v1/models returned {model_count} model(s)", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["gguf", "mlx"], required=True)
    parser.add_argument("--mesh-llm", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--bin-dir", default="")
    parser.add_argument("--expected-template-source", default="")
    parser.add_argument("--prompt-suite-json", required=True)
    parser.add_argument("--thinking-mode", default="nonempty")
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--wait-seconds", type=int, default=DEFAULT_WAIT_SECONDS)
    parser.add_argument("--label", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--mesh-log-output", default="")
    args = parser.parse_args()

    if args.backend == "gguf" and not args.bin_dir:
        parser.error("--bin-dir is required for gguf backend")
    if args.backend == "mlx" and args.expected_template_source:
        ensure_expected_template_source(args.model, args.expected_template_source)
    max_tokens_override = args.max_tokens if args.max_tokens > 0 else None

    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    prompt_suite = json.loads(args.prompt_suite_json)
    api_port = pick_free_port()
    console_port = pick_free_port()
    while api_port == console_port:
        console_port = pick_free_port()

    print("=== Thinking Smoke ===", flush=True)
    print(f"  backend: {args.backend}", flush=True)
    print(f"  model:   {args.model}", flush=True)
    print(f"  checks:  {len(prompt_suite)}", flush=True)
    print(f"  mode:    {args.thinking_mode}", flush=True)
    write_progress(
        status="starting",
        backend=args.backend,
        model=args.label or args.model,
        check_count=len(prompt_suite),
        completed_checks=0,
        failed_checks=0,
    )

    with tempfile.TemporaryDirectory(prefix="mesh-llm-thinking-") as temp_dir:
        os.environ["TMPDIR"] = temp_dir
        log_path = Path(temp_dir) / "mesh-llm.log"
        if args.mesh_log_output:
            mesh_log_output = Path(args.mesh_log_output)
        else:
            mesh_log_output = log_path

        launch_cmd = build_launch_command(args, api_port, console_port)
        with open(log_path, "w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                launch_cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                wait_until_ready(process, console_port, log_path, args.wait_seconds)
                write_models_artifact(api_port)

                results: list[dict[str, Any]] = []
                failed_checks = 0
                for index, case_cfg in enumerate(prompt_suite, start=1):
                    label = str(case_cfg.get("label", index))
                    write_progress(
                        status="running",
                        backend=args.backend,
                        model=args.label or args.model,
                        check_count=len(prompt_suite),
                        completed_checks=index - 1,
                        failed_checks=failed_checks,
                        current_label=label,
                    )
                    result = validate_case(
                        api_port=api_port,
                        case_cfg=case_cfg,
                        thinking_mode=args.thinking_mode,
                        max_tokens_override=max_tokens_override,
                    )
                    results.append(result)
                    if result["passed"]:
                        print(f"[{index:02d}/{len(prompt_suite)}] PASS {label}", flush=True)
                    else:
                        failed_checks += 1
                        print(f"[{index:02d}/{len(prompt_suite)}] FAIL {label}", flush=True)

                payload = {
                    "backend": args.backend,
                    "label": args.label,
                    "model": args.model,
                    "thinking_mode": args.thinking_mode,
                    "failed_check_count": failed_checks,
                    "check_count": len(prompt_suite),
                    "results": results,
                }
                Path(args.output_json).write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                if mesh_log_output != log_path:
                    sync_runtime_logs(case_dir(), log_path)
                    if log_path.exists():
                        shutil.copyfile(log_path, mesh_log_output)
                write_progress(
                    status="completed",
                    backend=args.backend,
                    model=args.label or args.model,
                    check_count=len(prompt_suite),
                    completed_checks=len(prompt_suite),
                    failed_checks=failed_checks,
                )
                if failed_checks:
                    print(f"❌ Thinking smoke failed: {failed_checks} check(s) flagged", flush=True)
                    return 1
                print("✅ Thinking smoke passed", flush=True)
                return 0
            finally:
                sync_runtime_logs(case_dir(), log_path)
                try:
                    process.send_signal(signal.SIGINT)
                    process.wait(timeout=20)
                except Exception:
                    process.kill()
                    process.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
