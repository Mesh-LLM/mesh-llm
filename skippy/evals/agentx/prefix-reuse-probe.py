#!/usr/bin/env python3
"""Prefix-reuse probe for skippy serve.

Replays N interleaved multi-turn conversations with append-only histories
against a running skippy serve and reports per-request cached tokens from the
OpenAI usage payload. This is the minimal reproduction for the exact-state
retention regression where concurrent sessions evicted each other down to a
single catalog entry (see kv_integration/config.rs retention floor).

Usage:
  python3 prefix-reuse-probe.py [port] [nconv] [turns] [block_chars]

The probe is intentionally dependency-free (urllib only). Conversations use a
stable per-conversation filler so each turn's prompt is a strict extension of
the previous turn's prompt; any cache miss beyond the first turn therefore
measures retention, not prompt churn.
"""
import concurrent.futures
import json
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 64000
NCONV = int(sys.argv[2]) if len(sys.argv) > 2 else 4
TURNS = int(sys.argv[3]) if len(sys.argv) > 3 else 6
BLOCK = int(sys.argv[4]) if len(sys.argv) > 4 else 6300


def convo(i: int, turn: int):
    filler = f"Session {i} stable context block. " * 220
    history = "\n".join(
        f"User: {filler[:BLOCK * k]} Question {i}-{k}: name a color."
        for k in range(1, turn + 1)
    )
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": history + f" Q{i}-{turn}: color now."},
    ]


def ask(i: int, turn: int):
    body = json.dumps(
        {"model": "probe", "messages": convo(i, turn), "max_tokens": 8, "temperature": 0}
    ).encode()
    t0 = time.time()
    r = urllib.request.urlopen(
        urllib.request.Request(
            f"http://127.0.0.1:{PORT}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        ),
        timeout=1800,
    )
    out = json.load(r)
    usage = out.get("usage", {})
    detail = usage.get("prompt_tokens_details", {}) or {}
    return (
        i,
        turn,
        usage.get("prompt_tokens", 0),
        detail.get("cached_tokens", 0),
        time.time() - t0,
    )


def main() -> int:
    print(f"probe: {NCONV} conversations x {TURNS} turns, concurrent per turn")
    misses = 0
    for turn in range(2, TURNS + 1):
        with concurrent.futures.ThreadPoolExecutor(max_workers=NCONV) as ex:
            for i, t, prompt, cached, dt in ex.map(lambda i: ask(i, turn), range(NCONV)):
                hit = cached > 0
                misses += 0 if hit else 1
                print(
                    f"  turn {t} conv {i}: prompt {prompt} cached {cached} "
                    f"({100 * cached / max(prompt, 1):.0f}%) {dt:.1f}s"
                )
    print(f"misses-after-first-turn: {misses}")
    return 0 if misses == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
