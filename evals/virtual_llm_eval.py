#!/usr/bin/env python3
"""
Virtual LLM eval — compare model answers with and without mesh hooks.

Usage:
    # Start mesh-llm with hooks (normal mode):
    mesh-llm --model <path> --auto

    # Run eval:
    python3 evals/virtual_llm_eval.py --port 9337

    # Run with hooks disabled for baseline:
    python3 evals/virtual_llm_eval.py --port 9337 --no-hooks

Results are written to evals/results/<timestamp>.jsonl
"""

import argparse
import json
import time
import os
import urllib.request

QUESTIONS = [
    # Easy — model should be confident, hooks should NOT fire
    {"q": "What is 2+2?", "category": "easy", "expected_hooks": False},
    {"q": "What is the capital of France?", "category": "easy", "expected_hooks": False},

    # Translation — small models often struggle
    {"q": "Translate to Swahili: The meeting has been rescheduled to Thursday", "category": "translation", "expected_hooks": True},
    {"q": "Translate to Japanese: Please send me the quarterly report by Friday", "category": "translation", "expected_hooks": True},

    # Obscure factual — model may not know
    {"q": "What was the GDP of Liechtenstein in 1987 in USD?", "category": "factual", "expected_hooks": True},
    {"q": "Who won the 1953 Pulitzer Prize for Fiction?", "category": "factual", "expected_hooks": True},
    {"q": "What is the population of Nauru?", "category": "factual", "expected_hooks": True},

    # Reasoning — needs coherent multi-step thinking
    {"q": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?", "category": "reasoning", "expected_hooks": True},
    {"q": "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost?", "category": "reasoning", "expected_hooks": True},

    # Creative — many valid answers, model may be uncertain about direction
    {"q": "Write a haiku about compiler errors", "category": "creative", "expected_hooks": False},
    {"q": "Explain quantum entanglement to a 5 year old", "category": "creative", "expected_hooks": False},

    # Technical — small models may hallucinate
    {"q": "What is the time complexity of Dijkstra's algorithm with a binary heap?", "category": "technical", "expected_hooks": True},
    {"q": "Explain the difference between TCP and UDP in exactly 3 sentences", "category": "technical", "expected_hooks": True},
]


def query(port, model, question, max_tokens=200, disable_hooks=False):
    """Send a chat completion request, return response + timing."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    if disable_hooks:
        body["mesh_hooks"] = False

    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
    except Exception as e:
        return {"error": str(e), "elapsed": time.time() - t0}

    elapsed = time.time() - t0
    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    usage = result.get("usage", {})
    timings = result.get("timings", {})

    return {
        "content": content,
        "elapsed": round(elapsed, 3),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "prompt_ms": timings.get("prompt_ms", 0),
        "predicted_ms": timings.get("predicted_ms", 0),
    }


def get_model(port):
    """Get the local model name from /v1/models."""
    req = urllib.request.Request(f"http://localhost:{port}/v1/models")
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
    # Find the local model (shortest name usually, or just first)
    models = [m["id"] for m in data.get("data", [])]
    return models[0] if models else "unknown"


def main():
    parser = argparse.ArgumentParser(description="Virtual LLM eval")
    parser.add_argument("--port", type=int, default=9337)
    parser.add_argument("--no-hooks", action="store_true", help="Disable hooks for baseline run")
    parser.add_argument("--model", type=str, default=None, help="Model name (auto-detected if omitted)")
    parser.add_argument("--max-tokens", type=int, default=200)
    args = parser.parse_args()

    model = args.model or get_model(args.port)
    mode = "baseline" if args.no_hooks else "hooks"
    print(f"Model: {model}")
    print(f"Mode: {mode}")
    print(f"Questions: {len(QUESTIONS)}")
    print()

    os.makedirs("evals/results", exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    outfile = f"evals/results/{ts}-{mode}.jsonl"

    with open(outfile, "w") as f:
        for i, item in enumerate(QUESTIONS):
            q = item["q"]
            print(f"[{i+1}/{len(QUESTIONS)}] {q[:60]}...")

            result = query(args.port, model, q, args.max_tokens, disable_hooks=args.no_hooks)

            record = {
                "question": q,
                "category": item["category"],
                "expected_hooks": item["expected_hooks"],
                "mode": mode,
                "model": model,
                **result,
            }
            f.write(json.dumps(record) + "\n")

            if "error" in result:
                print(f"  ERROR: {result['error']}")
            else:
                print(f"  {result['elapsed']}s | {result['completion_tokens']} tokens | {result['content'][:80]}...")
            print()

    print(f"Results written to {outfile}")


if __name__ == "__main__":
    main()
