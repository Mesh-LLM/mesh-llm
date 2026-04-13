#!/usr/bin/env python3
"""
Test how well different injection framings work.

Starts a mock hook server that returns a known-correct answer with
different text framings, then sends the same question to llama-server
and checks if the model incorporates the hint correctly.

Requirements:
  - llama-server built with mesh hooks
  - A GGUF model file

Usage:
  python3 evals/test_injection_framing.py --model /path/to/model.gguf

This will:
  1. Start a mock hook handler on port 3199
  2. Start llama-server with --mesh-port 3199
  3. Run test cases with different framings
  4. Print results
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler

# ==========================================================================
# Test cases: (question, correct_answer, wrong_baseline_hint)
# ==========================================================================
TEST_CASES = [
    {
        "question": "What is the population of Nauru?",
        "correct_hint": "The population of Nauru is approximately 12,500 people as of 2023.",
        "check": lambda r: any(x in r for x in ["12", "13,0", "12,5", "12.5", "13.0"]),
        "category": "factual",
    },
    {
        "question": "Who won the 1953 Pulitzer Prize for Fiction?",
        "correct_hint": "Ernest Hemingway won the 1953 Pulitzer Prize for Fiction for The Old Man and the Sea.",
        "check": lambda r: "hemingway" in r.lower(),
        "category": "factual",
    },
    {
        "question": "What is the capital of Burkina Faso?",
        "correct_hint": "The capital of Burkina Faso is Ouagadougou.",
        "check": lambda r: "ouagadougou" in r.lower(),
        "category": "factual",
    },
    {
        "question": "Translate to Swahili: The meeting has been rescheduled to Thursday",
        "correct_hint": "Mkutano umesogezwa hadi Alhamisi.",
        "check": lambda r: "mkutano" in r.lower() or "alhamisi" in r.lower(),
        "category": "translation",
    },
    {
        "question": "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost?",
        "correct_hint": "The ball costs $0.05. If the ball is $0.05, the bat is $1.05, and $0.05 + $1.05 = $1.10.",
        "check": lambda r: "$0.05" in r or "5 cents" in r.lower() or "0.05" in r,
        "category": "reasoning",
    },
]

# ==========================================================================
# Injection framings to test
# ==========================================================================
FRAMINGS = {
    "current": lambda hint: (
        f"\nHere is relevant information to help answer: {hint}\n\n"
        f"Now answer the user's question directly:\n"
    ),
    "system_note": lambda hint: (
        f"\n\n[System note: A more knowledgeable source provided this information: {hint}]\n\n"
    ),
    "reference": lambda hint: (
        f"\n\nReference answer: {hint}\n\n"
        f"Use the reference above to provide an accurate response.\n"
    ),
    "assistant_draft": lambda hint: (
        f"\n\nHere is a draft answer from another assistant:\n{hint}\n\n"
        f"Incorporate this into your response. Answer concisely:\n"
    ),
    "rag_context": lambda hint: (
        f"\n\nContext from knowledge base:\n{hint}\n\n"
        f"Based on the above context, answer the question:\n"
    ),
}


# ==========================================================================
# Mock hook server
# ==========================================================================
class MockHookHandler(BaseHTTPRequestHandler):
    """Returns controlled inject responses for testing."""

    current_framing = "current"
    current_hint = ""
    hook_fired = threading.Event()

    def log_message(self, format, *args):
        pass  # quiet

    def do_POST(self):
        if self.path != "/mesh/hook":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        hook = body.get("hook", "")

        if hook == "post_prefill":
            # Always inject — we want to test what the model does with it
            framing_fn = FRAMINGS[MockHookHandler.current_framing]
            text = framing_fn(MockHookHandler.current_hint)
            resp = {"action": "inject", "text": text}
            MockHookHandler.hook_fired.set()
        else:
            resp = {"action": "none"}

        resp_bytes = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_bytes)))
        self.end_headers()
        self.wfile.write(resp_bytes)


def start_mock_server(port=3199):
    server = HTTPServer(("127.0.0.1", port), MockHookHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


# ==========================================================================
# llama-server management
# ==========================================================================
def find_llama_server():
    """Find the llama-server binary."""
    candidates = [
        "llama.cpp/build/bin/llama-server",
        "target/release/llama-server",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def start_llama_server(model_path, server_port=8199, hook_port=3199):
    binary = find_llama_server()
    if not binary:
        print("ERROR: llama-server not found")
        sys.exit(1)

    cmd = [
        binary,
        "-m", model_path,
        "--host", "127.0.0.1",
        "--port", str(server_port),
        "--mesh-port", str(hook_port),
        "-ngl", "99",
        "--no-warmup",
    ]
    print(f"Starting: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )

    # Wait for ready
    for _ in range(60):
        time.sleep(1)
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{server_port}/health")
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read())
                if data.get("status") == "ok":
                    print("llama-server ready")
                    return proc
        except Exception:
            if proc.poll() is not None:
                out = proc.stdout.read().decode()
                print(f"llama-server exited early:\n{out[-2000:]}")
                sys.exit(1)

    print("Timeout waiting for llama-server")
    proc.kill()
    sys.exit(1)


def query_llama(port, question, max_tokens=150, temperature=0.1):
    body = {
        "model": "test",
        "messages": [{"role": "user", "content": question}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())
    elapsed = time.time() - t0
    content = result["choices"][0]["message"]["content"]
    return content, elapsed


# ==========================================================================
# Main
# ==========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to GGUF model file")
    parser.add_argument("--server-port", type=int, default=8199)
    parser.add_argument("--hook-port", type=int, default=3199)
    args = parser.parse_args()

    # Start mock hook server
    print(f"Starting mock hook server on port {args.hook_port}...")
    mock = start_mock_server(args.hook_port)

    # Start llama-server
    proc = start_llama_server(args.model, args.server_port, args.hook_port)

    try:
        results = []

        # --- Baseline: no hooks (hooks fire but return "none") ---
        print("\n" + "=" * 70)
        print("BASELINE (no injection)")
        print("=" * 70)
        saved_framing = MockHookHandler.current_framing
        for tc in TEST_CASES:
            MockHookHandler.current_hint = ""
            MockHookHandler.current_framing = "current"
            # Override to return none
            original = FRAMINGS["current"]
            FRAMINGS["_none"] = lambda h: ""  # empty = no injection
            MockHookHandler.current_framing = "_none"

            answer, elapsed = query_llama(args.server_port, tc["question"])
            passed = tc["check"](answer)
            results.append({
                "framing": "baseline",
                "question": tc["question"][:50],
                "category": tc["category"],
                "passed": passed,
                "elapsed": round(elapsed, 2),
                "answer": answer[:120],
            })
            print(f"  {'✅' if passed else '❌'} [{elapsed:.1f}s] {tc['question'][:50]}")
            print(f"     → {answer[:100]}")

        FRAMINGS.pop("_none", None)

        # --- Test each framing ---
        for framing_name, framing_fn in FRAMINGS.items():
            print(f"\n{'=' * 70}")
            print(f"FRAMING: {framing_name}")
            print("=" * 70)

            for tc in TEST_CASES:
                MockHookHandler.current_framing = framing_name
                MockHookHandler.current_hint = tc["correct_hint"]
                MockHookHandler.hook_fired.clear()

                answer, elapsed = query_llama(args.server_port, tc["question"])
                hook_used = MockHookHandler.hook_fired.is_set()
                passed = tc["check"](answer)

                results.append({
                    "framing": framing_name,
                    "question": tc["question"][:50],
                    "category": tc["category"],
                    "passed": passed,
                    "hook_fired": hook_used,
                    "elapsed": round(elapsed, 2),
                    "answer": answer[:120],
                })
                hook_str = "🔗" if hook_used else "⚪"
                pass_str = "✅" if passed else "❌"
                print(f"  {pass_str} {hook_str} [{elapsed:.1f}s] {tc['question'][:50]}")
                print(f"     → {answer[:100]}")

        # --- Summary ---
        print(f"\n{'=' * 70}")
        print("SUMMARY")
        print("=" * 70)
        for framing_name in ["baseline"] + list(FRAMINGS.keys()):
            fr = [r for r in results if r["framing"] == framing_name]
            passed = sum(1 for r in fr if r["passed"])
            total = len(fr)
            avg_time = sum(r["elapsed"] for r in fr) / total if total else 0
            print(f"  {framing_name:20s}: {passed}/{total} correct, avg {avg_time:.1f}s")

        # Save results
        os.makedirs("evals/results", exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        outfile = f"evals/results/{ts}-framing.json"
        with open(outfile, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {outfile}")

    finally:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=10)
        mock.shutdown()


if __name__ == "__main__":
    main()
