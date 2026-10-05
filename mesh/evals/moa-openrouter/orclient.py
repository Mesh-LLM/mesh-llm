"""Minimal OpenRouter client — stdlib only (no `requests` dependency).

Shared by the MoA probe scripts in this directory.
"""

import json
import os
import time
import urllib.error
import urllib.request

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


def chat(
    model,
    messages,
    tools=None,
    max_tokens=512,
    temperature=0.7,
    timeout=120,
    no_think=False,
):
    """One OpenAI-shaped chat completion. Returns (response_json, elapsed_s).

    Retries on 429/5xx with a small backoff ladder. Unlike Together's
    reference implementation, every failure path returns a structured
    error instead of raising — the caller decides what a dead worker means.

    `no_think=True` mirrors mesh-llm's `effective_enable_thinking_for_moa`
    default: reasoning models get told to skip the think phase, so a small
    worker budget isn't spent producing reasoning tokens and no answer.
    """
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return {"error": "OPENROUTER_API_KEY not set"}, 0.0

    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools:
        body["tools"] = tools
    if no_think:
        body["reasoning_effort"] = "none"
        body["chat_template_kwargs"] = {"enable_thinking": False}

    def _build(b):
        return urllib.request.Request(
            ENDPOINT,
            data=json.dumps(b).encode(),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/mesh-llm",
                "X-Title": "mesh-llm MoA probe",
            },
        )

    started = time.time()
    last_err = None
    for sleep_time in (0, 2, 5):
        if sleep_time:
            time.sleep(sleep_time)
        try:
            with urllib.request.urlopen(_build(body), timeout=timeout) as resp:
                payload = json.loads(resp.read().decode())
            if "error" in payload:
                last_err = str(payload["error"])
                continue
            return payload, time.time() - started
        except urllib.error.HTTPError as e:
            detail = e.read().decode()[:300]
            last_err = f"HTTP {e.code}: {detail}"
            # Some endpoints (e.g. minimax) *require* reasoning and reject
            # our thinking-disable flags with 400. Drop them and retry once
            # rather than losing the worker entirely.
            if e.code == 400 and "reasoning" in detail.lower() and no_think:
                body.pop("reasoning_effort", None)
                body.pop("chat_template_kwargs", None)
                no_think = False
                continue
            if e.code not in (429, 500, 502, 503, 504):
                break
        except Exception as e:  # timeout, connection reset, bad JSON
            last_err = f"{type(e).__name__}: {e}"

    return {"error": last_err or "unknown"}, time.time() - started


def first_choice(resp):
    """Extract (text, tool_calls) from a response. Either may be empty/None."""
    try:
        msg = resp["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return None, None
    return msg.get("content"), msg.get("tool_calls")


# ─── Mesh-realistic worker pool ──────────────────────────────────────
#
# Chosen so tiers line up with how mesh-llm's `is_single_digit_b_name`
# classifies names (single-digit-B => small tier, everything else big),
# and so the mix mirrors a real mesh: a couple of small local-ish models
# plus bigger MoE/frontier-ish open weights.
POOL_SMALL = [
    "qwen/qwen3-8b",
    "qwen/qwen3.5-9b",
    "mistralai/ministral-8b-2512",
    "mistralai/ministral-3b-2512",
]
POOL_BIG = [
    "qwen/qwen3-14b",
    "qwen/qwen3-32b",
    "qwen/qwen3-30b-a3b-instruct-2507",
    "minimax/minimax-m2.5",
    "mistralai/mistral-small-3.2-24b-instruct",
]
POOL = POOL_SMALL + POOL_BIG


def tier(model):
    """Mirror of mesh-llm's name-derived tiering, for reporting only."""
    name = model.lower()
    for i, c in enumerate(name):
        if not c.isdigit() or c == "0":
            continue
        if i > 0 and (name[i - 1].isdigit() or name[i - 1] == "." or name[i - 1].isalpha()):
            continue
        if i + 1 < len(name) and name[i + 1] == "b":
            if i + 2 < len(name) and name[i + 2].isdigit():
                continue
            return "small"
    return "big"
