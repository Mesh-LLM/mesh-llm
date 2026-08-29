#!/usr/bin/env python3
"""Serving-level L3 warm-up benchmark against the KV acceptance criteria.

Drives a real `mesh-llm serve` through the OpenAI endpoint and measures
time-to-first-token for the cases the criteria name:

  cold        first request ever (empty L3, cold radix) — full prefill
  warm        same prefix again in the same process — radix (L1) hit
  turn        prefix + previous answer + a new question — longest-prefix
              reuse across a growing multi-turn prompt
  restart     same prefix after a full process restart — L3 fill from disk
  concurrent  two simultaneous same-prefix requests after another restart —
              must both succeed without duplicated disk loads (single-flight)

Usage:
  l3_warmup_bench.py --model <gguf-or-ref> --prefix-file agent-prefix.txt \
      [--port 18080] [--l3-dir /tmp/skippy-l3-bench] [--max-tokens 24] \
      [--serve-arg=--foo ...] [--report-out report.json]

The server is started with SKIPPY_KV_CACHE=on and SKIPPY_L3_DIR set; with
L3 enabled dense models record KV pages automatically. Requires a mesh-llm
binary on PATH or MESH_LLM_BIN.
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request


def wait_ready(port: int, process: subprocess.Popen, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"serve exited early with {process.returncode}")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/v1/models", timeout=2
            ) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("serve did not become ready in time")


def first_model(port: int, timeout_s: float = 600.0) -> str:
    """The served model id, waiting out the asynchronous model load."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/v1/models", timeout=5
            ) as response:
                payload = json.load(response)
            if payload.get("data"):
                return payload["data"][0]["id"]
        except Exception:
            pass
        time.sleep(1.0)
    raise RuntimeError("no model appeared on /v1/models in time")


def ttft_request(port: int, model: str, prompt: str, max_tokens: int) -> tuple[float, str]:
    """Streamed completion; returns (ttft_ms, generated_text)."""
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    ttft_ms = None
    text = []
    with urllib.request.urlopen(request, timeout=600) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            piece = chunk.get("choices", [{}])[0].get("text", "")
            if piece and ttft_ms is None:
                ttft_ms = (time.monotonic() - started) * 1000.0
            text.append(piece)
    if ttft_ms is None:
        raise RuntimeError("stream produced no tokens")
    return ttft_ms, "".join(text)


class Serve:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.process: subprocess.Popen | None = None

    def start(self) -> None:
        binary = os.environ.get("MESH_LLM_BIN", "mesh-llm")
        env = dict(os.environ)
        env.update({
            "SKIPPY_KV_CACHE": "on",
            "SKIPPY_L3_DIR": self.args.l3_dir,
        })
        command = [
            binary, "serve",
            "--model", self.args.model,
            "--port", str(self.args.port),
            *self.args.serve_arg,
        ]
        log = open(self.args.serve_log, "ab")
        self.process = subprocess.Popen(command, env=env, stdout=log, stderr=log)
        wait_ready(self.args.port, self.process, self.args.startup_timeout)

    def stop(self) -> None:
        if self.process is None:
            return
        self.process.send_signal(signal.SIGTERM)
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)
        self.process = None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prefix-file", required=True)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--l3-dir", default="/tmp/skippy-l3-bench")
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    parser.add_argument("--serve-log", default="l3-bench-serve.log")
    parser.add_argument("--serve-arg", action="append", default=[])
    parser.add_argument("--report-out")
    parser.add_argument(
        "--keep-l3", action="store_true",
        help="Do not wipe the L3 directory first (measure against existing state)",
    )
    args = parser.parse_args()

    with open(args.prefix_file) as handle:
        prefix = handle.read()
    question_one = "\n\nQuestion: summarize the scheduler invariants in one sentence.\nAnswer:"
    question_two = "\n\nQuestion: which file owns the OpenAI ingress?\nAnswer:"

    if not args.keep_l3:
        shutil.rmtree(args.l3_dir, ignore_errors=True)
    os.makedirs(args.l3_dir, exist_ok=True)

    results: dict[str, float] = {}
    serve = Serve(args)

    print("== starting serve (empty L3)")
    serve.start()
    try:
        model = first_model(args.port)
        print(f"== model: {model}")

        ttft, answer_one = ttft_request(args.port, model, prefix + question_one, args.max_tokens)
        results["cold_ttft_ms"] = ttft
        print(f"cold      TTFT {ttft:8.0f} ms  (full prefill)")

        ttft, _ = ttft_request(args.port, model, prefix + question_two, args.max_tokens)
        results["warm_ttft_ms"] = ttft
        print(f"warm      TTFT {ttft:8.0f} ms  (radix hit, same process)")

        turn_prompt = prefix + question_one + answer_one + question_two
        ttft, _ = ttft_request(args.port, model, turn_prompt, args.max_tokens)
        results["turn_ttft_ms"] = ttft
        print(f"turn      TTFT {ttft:8.0f} ms  (multi-turn longest-prefix reuse)")
    finally:
        serve.stop()

    print("== restarting serve (warm L3, cold RAM)")
    serve.start()
    try:
        model = first_model(args.port)
        ttft, _ = ttft_request(args.port, model, prefix + question_two, args.max_tokens)
        results["restart_ttft_ms"] = ttft
        print(f"restart   TTFT {ttft:8.0f} ms  (L3 fill from disk)")
    finally:
        serve.stop()

    print("== restarting serve (concurrency: two same-prefix requests at once)")
    serve.start()
    try:
        model = first_model(args.port)
        concurrent: dict[str, float] = {}
        errors: list[str] = []

        def run(tag: str, tail: str) -> None:
            try:
                ttft, _ = ttft_request(args.port, model, prefix + tail, args.max_tokens)
                concurrent[tag] = ttft
            except Exception as error:  # noqa: BLE001 - report, don't crash the bench
                errors.append(f"{tag}: {error}")

        threads = [
            threading.Thread(target=run, args=("a", question_one)),
            threading.Thread(target=run, args=("b", question_two)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise RuntimeError("; ".join(errors))
        results["concurrent_a_ttft_ms"] = concurrent["a"]
        results["concurrent_b_ttft_ms"] = concurrent["b"]
        print(
            f"concurrent TTFT {concurrent['a']:7.0f} / {concurrent['b']:.0f} ms  "
            "(single-flight fill; loser prefills, winner warms radix)"
        )
    finally:
        serve.stop()

    cold = results["cold_ttft_ms"]
    print("\n== summary (speedup vs cold prefill)")
    for name in ("warm", "turn", "restart"):
        ttft = results[f"{name}_ttft_ms"]
        print(f"{name:8} {ttft:8.0f} ms   {cold / max(ttft, 1e-9):6.2f}x")
    if args.report_out:
        with open(args.report_out, "w") as handle:
            json.dump(results, handle, indent=2)
        print(f"report written to {args.report_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
