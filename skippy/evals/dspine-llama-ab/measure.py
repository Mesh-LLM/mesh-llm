#!/usr/bin/env python3
"""Drives one benchmark arm against a running llama-server and prints one
DSPINE_AB JSON summary line per arm.

Usage: measure.py <port> <arm-name> <prompts.jsonl> [limit]

Sequentially sends each prompt to /completion (greedy, 256 tokens), records
per-request decode tok/s and draft acceptance from the response timings, then
prints the aggregate as a single machine-readable line on stdout.
"""
import json
import sys
import time
import urllib.request


def completion(port, prompt, n_predict):
    body = json.dumps({
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0,
        "cache_prompt": False,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as resp:
        return json.load(resp)


def main():
    port, arm, prompts_path = sys.argv[1], sys.argv[2], sys.argv[3]
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    n_predict = 256

    prompts = []
    with open(prompts_path) as f:
        for line in f:
            row = json.loads(line)
            prompts.append((row["id"], row.get("category", "?"), row["prompt"]))
    if limit:
        prompts = prompts[:limit]

    # warmup
    completion(port, "Hello", 8)

    rows = []
    tot_tokens = 0
    tot_ms = 0.0
    tot_draft = 0
    tot_accepted = 0
    t0 = time.time()
    for i, (pid, cat, prompt) in enumerate(prompts):
        r = completion(port, prompt, n_predict)
        t = r["timings"]
        n = t["predicted_n"]
        ms = t["predicted_ms"]
        draft = t.get("draft_n", 0)
        acc = t.get("draft_n_accepted", 0)
        tot_tokens += n
        tot_ms += ms
        tot_draft += draft
        tot_accepted += acc
        rows.append({
            "id": pid,
            "category": cat,
            "tokens": n,
            "ms": round(ms, 1),
            "tok_s": round(1000.0 * n / ms, 2) if ms else None,
            "draft": draft,
            "accepted": acc,
        })
        print(f"  [{i + 1}/{len(prompts)}] {pid} {1000.0 * n / ms:.1f} tok/s"
              + (f" draft={draft} accepted={acc}" if draft else ""), flush=True)

    summary = {
        "arm": arm,
        "prompts": len(rows),
        "wall_s": round(time.time() - t0, 1),
        "tokens": tot_tokens,
        "tok_s": round(1000.0 * tot_tokens / tot_ms, 2) if tot_ms else None,
        "draft_n": tot_draft,
        "draft_n_accepted": tot_accepted,
        "acceptance_pct": round(100.0 * tot_accepted / tot_draft, 1) if tot_draft else None,
        "per_prompt": rows,
    }
    print("DSPINE_AB " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
