"""Full-session completeness, long-context eligibility and recurrent evidence."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path


def expected_turns(trajectory):
    return [
        f"{trajectory['session_id']}:{i}"
        for i in range(
            sum(message["role"] == "assistant" for message in trajectory["messages"])
        )
    ]


def complete_sessions(trajectories, requests):
    expected = {t["session_id"]: expected_turns(t) for t in trajectories}
    observed = defaultdict(list)
    problems = []
    for request in requests:
        observed[request.get("session_id")].append(request.get("request_id"))
        if request.get("error"):
            problems.append(f"{request.get('request_id')}: {request['error']}")
    if len(expected) != len(trajectories):
        problems.append("duplicate session IDs")
    for session, ids in expected.items():
        if observed.get(session) != ids:
            problems.append(f"{session}: missing, duplicate, or out-of-order turns")
    if set(observed) - set(expected):
        problems.append("unexpected sessions")
    return {
        "passed": not problems,
        "problems": problems,
        "expected_request_ids": [i for ids in expected.values() for i in ids],
        "expected_turns": sum(map(len, expected.values())),
    }


def session_summaries(trajectories, requests):
    result = []
    for trajectory in trajectories:
        rows = [r for r in requests if r.get("session_id") == trajectory["session_id"]]
        successful = [r for r in rows if not r.get("error")]
        prompt = sum(r["prompt_tokens"] for r in successful)
        cached = sum(r["cached_tokens"] for r in successful)
        result.append(
            {
                "session_id": trajectory["session_id"],
                "expected_request_ids": expected_turns(trajectory),
                "complete": complete_sessions([trajectory], rows)["passed"],
                "prompt_tokens": prompt,
                "cached_tokens": cached,
                "cache_pct": 100 * cached / prompt if prompt else None,
                "max_prompt_tokens": max(
                    (r["prompt_tokens"] for r in successful), default=0
                ),
                "turns": [
                    {
                        "request_id": r["request_id"],
                        "prompt_tokens": r.get("prompt_tokens"),
                        "cached_tokens": r.get("cached_tokens"),
                        "cache_pct": (
                            (100 * r["cached_tokens"] / r["prompt_tokens"])
                            if r.get("prompt_tokens")
                            else None
                        ),
                        "ttft_seconds": r.get("ttft_seconds"),
                        "error": r.get("error"),
                    }
                    for r in rows
                ],
            }
        )
    return result


def runtime_context(document, required):
    stages = document.get("stages", [])
    if not stages or len({s.get("model_id") for s in stages}) != 1:
        raise ValueError(
            "context preflight requires one identified local model with stage metadata"
        )
    values = [s.get("ctx_size") for s in stages]
    if any(type(v) is not int or v < required for v in values):
        raise ValueError(
            f"effective runtime context {values} is below required {required}"
        )
    return min(values)


def context_eligibility(
    trajectories, probes, context, maximum_output, minimum_long, budget
):
    complete = complete_sessions(trajectories, probes)
    problems = list(complete["problems"])
    by_id = {p["request_id"]: p for p in probes}
    rows = []
    for trajectory in trajectories:
        turn = 0
        lengths = []
        for message in trajectory["messages"]:
            if message["role"] != "assistant":
                continue
            request_id = f"{trajectory['session_id']}:{turn}"
            turn += 1
            probe = by_id.get(request_id, {})
            tokens = probe.get("prompt_tokens")
            output = budget(message, maximum_output)
            fits = type(tokens) is int and tokens > 0 and tokens + output <= context
            if not fits:
                problems.append(
                    f"{request_id}: formatted prompt + output does not fit {context}"
                )
            if type(tokens) is int:
                lengths.append(tokens)
            rows.append(
                {
                    "request_id": request_id,
                    "prompt_tokens": tokens,
                    "output_budget": output,
                    "fits": fits,
                }
            )
        if not lengths or max(lengths) < minimum_long:
            problems.append(
                f"{trajectory['session_id']}: no prompt reaches {minimum_long} tokens"
            )
    return {
        "passed": not problems,
        "problems": problems,
        "context_tokens": context,
        "minimum_session_prompt_tokens": minimum_long,
        "turns": rows,
    }


def recurrent_evidence(log_paths, requests, minimum_restored_tokens=1):
    """Join one local lookup decision per sequential turn within a cache namespace.

    Each session has a stable namespace. Extra/missing decisions fail correlation;
    only a successful exact restore of a recurrent payload proves state reuse.
    """
    decisions = defaultdict(list)
    seen = set()
    for path in log_paths:
        for line in Path(path).read_text(errors="replace").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if (
                not isinstance(event, dict)
                or event.get("event") != "stage.openai_kv_lookup_decision"
            ):
                continue
            digest = hashlib.sha256(line.encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            attrs = event.get("attributes", {})
            session = attrs.get("openai.prompt_cache_key")
            if session:
                decisions[session].append(event)
    grouped = defaultdict(list)
    for request in requests:
        grouped[request["session_id"]].append(request)
    problems = []
    restores = 0
    rows = []
    for session, turns in grouped.items():
        events = sorted(
            decisions.get(session, []), key=lambda e: e["start_time_unix_nanos"]
        )
        if len(events) != len(turns):
            problems.append(
                f"{session}: {len(events)} lookup events for {len(turns)} turns"
            )
            continue
        session_restores = 0
        for turn, event in zip(turns, events):
            attrs = event["attributes"]
            tokens = attrs.get("skippy.exact_cache.restored_tokens", 0)
            recurrent = attrs.get("skippy.exact_cache.payload_kind") in (
                "kv-recurrent",
                "recurrent-only",
            )
            restored = (
                attrs.get("skippy.kv.decision") == "exact_hit"
                and recurrent
                and type(tokens) is int
                and minimum_restored_tokens <= tokens <= turn.get("prompt_tokens", 0)
            )
            if restored and turn.get("assistant_turn", 0) > 0:
                restores += 1
                session_restores += 1
            rows.append(
                {
                    "request_id": turn["request_id"],
                    "state_restored": restored,
                    "restored_tokens": tokens if restored else 0,
                    "lookup": attrs,
                }
            )
        if not session_restores:
            problems.append(
                f"{session}: no recurrent restore on a later recorded-history turn"
            )
    return {
        "passed": bool(grouped) and not problems,
        "problems": problems,
        "restores": restores,
        "minimum_restored_tokens": minimum_restored_tokens,
        "turns": rows,
    }
