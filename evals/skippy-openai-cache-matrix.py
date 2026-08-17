#!/usr/bin/env python3
"""Compare native llama-server and Skippy OpenAI prompt-cache behavior.

The runner attaches to already-running endpoints. Skippy cache mode is a stage
configuration choice, so pass separate cold and warm Skippy endpoints.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchRequest:
    shared_prefix: str
    measured_tail: str
    warmup_tail: str
    pattern: str


def http_json(
    url: str,
    payload: dict[str, Any],
    api_key: str | None,
    timeout: float,
) -> dict[str, Any]:
    headers = {"content-type": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def elapsed_request(
    url: str,
    payload: dict[str, Any],
    api_key: str | None,
    timeout: float,
) -> tuple[dict[str, Any], float]:
    started = time.monotonic()
    response = http_json(url, payload, api_key, timeout)
    return response, (time.monotonic() - started) * 1000.0


def llama_payload(prompt: str, max_tokens: int, cache_prompt: bool) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": 0,
        "top_k": 1,
        "cache_prompt": cache_prompt,
    }


def skippy_payload(
    model: str,
    shared_prefix: str,
    tail: str,
    max_tokens: int,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": shared_prefix},
            {"role": "user", "content": tail},
        ],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
    }


def prompt_text(request: BenchRequest, tail: str) -> str:
    return f"System prefix:\n{request.shared_prefix}\n\nUser request:\n{tail}\n"


def cached_tokens_from_skippy(response: dict[str, Any]) -> int:
    usage = response.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    value = details.get("cached_tokens", 0)
    return int(value) if isinstance(value, (int, float)) else 0


def prompt_tokens_from_skippy(response: dict[str, Any]) -> int:
    usage = response.get("usage") or {}
    value = usage.get("prompt_tokens", 0)
    return int(value) if isinstance(value, (int, float)) else 0


def cache_tokens_from_llama(response: dict[str, Any]) -> int:
    timings = response.get("timings") or {}
    for key in ("cache_n", "tokens_cached"):
        value = timings.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def prompt_tokens_from_llama(response: dict[str, Any]) -> int:
    timings = response.get("timings") or {}
    for key in ("prompt_n", "tokens_evaluated"):
        value = timings.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def total_prompt_tokens_from_llama(response: dict[str, Any]) -> int:
    processed = prompt_tokens_from_llama(response)
    cached = cache_tokens_from_llama(response)
    return processed + cached


def cacheable_prefix_tokens(prompt_tokens: int, backend: str) -> int:
    if backend == "skippy-openai":
        return max(prompt_tokens - 1, 0)
    return prompt_tokens


def cache_status(cache_mode: str, cached_tokens: int) -> str:
    if "disabled" in cache_mode or "false" in cache_mode:
        return "disabled"
    if cached_tokens > 0:
        return "hit"
    return "miss"


def hit_kind(backend: str, status: str) -> str:
    if status != "hit":
        return "none"
    if backend == "llama-server":
        return "llama_prompt_cache"
    return "usage_cached_tokens"


def cache_observation(
    backend: str,
    cache_mode: str,
    prompt_tokens: int,
    cached_tokens: int,
) -> dict[str, Any]:
    cacheable_tokens = cacheable_prefix_tokens(prompt_tokens, backend)
    status = cache_status(cache_mode, cached_tokens)
    uncached_tokens = max(cacheable_tokens - cached_tokens, 0)
    efficiency = (cached_tokens / cacheable_tokens) if cacheable_tokens else 0.0
    return {
        "cache_status": status,
        "hit_kind": hit_kind(backend, status),
        "cacheable_prefix_tokens": cacheable_tokens,
        "cached_tokens": cached_tokens,
        "suffix_prefill_tokens": uncached_tokens,
        "cache_efficiency": efficiency,
    }


def run_observation(
    backend: str,
    cache_mode: str,
    elapsed_ms: float,
    prompt_tokens: int,
    cached_tokens: int,
    run: int | None = None,
) -> dict[str, Any]:
    observation = {
        "elapsed_ms": elapsed_ms,
        "prompt_tokens": prompt_tokens,
        **cache_observation(backend, cache_mode, prompt_tokens, cached_tokens),
    }
    if run is not None:
        observation = {"run": run, **observation}
    return observation


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def summarize_row(
    name: str,
    backend: str,
    cache_mode: str,
    runs: list[dict[str, Any]],
    warmup: dict[str, Any] | None = None,
) -> dict[str, Any]:
    elapsed = [
        float(run["elapsed_ms"])
        for run in runs
        if isinstance(run.get("elapsed_ms"), (int, float))
    ]
    cached = [
        int(run["cached_tokens"])
        for run in runs
        if isinstance(run.get("cached_tokens"), (int, float))
    ]
    prompt_tokens = [
        int(run["prompt_tokens"])
        for run in runs
        if isinstance(run.get("prompt_tokens"), (int, float))
    ]
    cacheable_tokens = [
        int(run["cacheable_prefix_tokens"])
        for run in runs
        if isinstance(run.get("cacheable_prefix_tokens"), (int, float))
    ]
    suffix_prefill = [
        int(run["suffix_prefill_tokens"])
        for run in runs
        if isinstance(run.get("suffix_prefill_tokens"), (int, float))
    ]
    efficiencies = [
        float(run["cache_efficiency"])
        for run in runs
        if isinstance(run.get("cache_efficiency"), (int, float))
    ]
    max_prompt_tokens = max(prompt_tokens) if prompt_tokens else 0
    max_cacheable_tokens = max(cacheable_tokens) if cacheable_tokens else 0
    min_cached_tokens = min(cached) if cached else 0
    max_cached_tokens = max(cached) if cached else 0
    max_suffix_prefill_tokens = max(suffix_prefill) if suffix_prefill else 0
    max_prompt_ratio = (max_cached_tokens / max_prompt_tokens) if max_prompt_tokens else 0.0
    min_cache_efficiency = min(efficiencies) if efficiencies else 0.0
    max_cache_efficiency = max(efficiencies) if efficiencies else 0.0
    statuses = sorted({str(run.get("cache_status", "unknown")) for run in runs})
    verdict = cache_verdict(cache_mode, cached)
    return {
        "name": name,
        "backend": backend,
        "cache_mode": cache_mode,
        "warmup": warmup,
        "runs": runs,
        "verdict": verdict,
        "cache_statuses": statuses,
        "median_elapsed_ms": median(elapsed),
        "mean_cached_tokens": (sum(cached) / len(cached)) if cached else 0,
        "min_cached_tokens": min_cached_tokens,
        "max_cached_tokens": max_cached_tokens,
        "max_prompt_tokens": max_prompt_tokens,
        "max_cacheable_prefix_tokens": max_cacheable_tokens,
        "max_suffix_prefill_tokens": max_suffix_prefill_tokens,
        "max_prompt_cached_ratio": max_prompt_ratio,
        "min_cache_efficiency": min_cache_efficiency,
        "max_cache_efficiency": max_cache_efficiency,
    }


def cache_verdict(cache_mode: str, cached_tokens: list[int]) -> str:
    if "disabled" in cache_mode or "false" in cache_mode:
        return (
            "PASS disabled/no-cache"
            if all(cached == 0 for cached in cached_tokens)
            else "FAIL disabled cached"
        )
    if cached_tokens and all(cached > 0 for cached in cached_tokens):
        return "PASS all-hit"
    return "FAIL missed-hit"


def format_range(min_value: int | float, max_value: int | float, suffix: str = "") -> str:
    if min_value == max_value:
        return f"{min_value}{suffix}"
    return f"{min_value}{suffix}-{max_value}{suffix}"


def format_percent_range(min_value: float, max_value: float) -> str:
    if min_value == max_value:
        return f"{min_value:.1%}"
    return f"{min_value:.1%}-{max_value:.1%}"


def run_llama_row(
    base_url: str,
    request: BenchRequest,
    repeats: int,
    max_tokens: int,
    cache_prompt: bool,
    timeout: float,
    warmup: bool,
) -> dict[str, Any]:
    base_url = base_url.rstrip("/")
    url = f"{base_url}/completion"
    backend = "llama-server"
    mode = "warm" if cache_prompt else "cold"
    cache_mode = "cache_prompt=true" if cache_prompt else "cache_prompt=false"
    warmup_result = None
    if warmup:
        response, elapsed_ms = elapsed_request(
            url,
            llama_payload(prompt_text(request, request.warmup_tail), max_tokens, cache_prompt),
            None,
            timeout,
        )
        warmup_result = run_observation(
            backend,
            cache_mode,
            elapsed_ms,
            total_prompt_tokens_from_llama(response),
            cache_tokens_from_llama(response),
        )

    runs = []
    payload = llama_payload(prompt_text(request, request.measured_tail), max_tokens, cache_prompt)
    for index in range(repeats):
        response, elapsed_ms = elapsed_request(url, payload, None, timeout)
        runs.append(
            run_observation(
                backend,
                cache_mode,
                elapsed_ms,
                total_prompt_tokens_from_llama(response),
                cache_tokens_from_llama(response),
                run=index + 1,
            )
        )
    return summarize_row(
        f"{mode} native",
        backend,
        cache_mode,
        runs,
        warmup_result,
    )


def run_skippy_row(
    base_url: str,
    model: str,
    request: BenchRequest,
    repeats: int,
    max_tokens: int,
    api_key: str | None,
    timeout: float,
    warmup: bool,
    name: str,
    cache_mode: str,
) -> dict[str, Any]:
    base_url = base_url.rstrip("/")
    url = f"{base_url}/chat/completions"
    backend = "skippy-openai"
    warmup_result = None
    if warmup:
        response, elapsed_ms = elapsed_request(
            url,
            skippy_payload(model, request.shared_prefix, request.warmup_tail, max_tokens),
            api_key,
            timeout,
        )
        warmup_result = run_observation(
            backend,
            cache_mode,
            elapsed_ms,
            prompt_tokens_from_skippy(response),
            cached_tokens_from_skippy(response),
        )

    runs = []
    payload = skippy_payload(model, request.shared_prefix, request.measured_tail, max_tokens)
    for index in range(repeats):
        response, elapsed_ms = elapsed_request(url, payload, api_key, timeout)
        runs.append(
            run_observation(
                backend,
                cache_mode,
                elapsed_ms,
                prompt_tokens_from_skippy(response),
                cached_tokens_from_skippy(response),
                run=index + 1,
            )
        )
    return summarize_row(name, backend, cache_mode, runs, warmup_result)


def markdown_report(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Mode | Verdict | Warmup | Statuses | Cache mode | Median ms | Prompt | Cacheable | Cached | Suffix/uncached | Prompt ratio | Cache efficiency |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        median_ms = row.get("median_elapsed_ms")
        prompt_ratio = row.get("max_prompt_cached_ratio")
        min_efficiency = row.get("min_cache_efficiency")
        max_efficiency = row.get("max_cache_efficiency")
        warmup = row.get("warmup") or {}
        warmup_status = warmup.get("cache_status", "none")
        lines.append(
            "| {name} | {verdict} | {warmup} | {statuses} | `{cache_mode}` | {median_ms} | {prompt} | {cacheable} | {cached} | {suffix} | {prompt_ratio} | {efficiency} |".format(
                name=row["name"],
                verdict=row["verdict"],
                warmup=warmup_status,
                statuses=", ".join(row["cache_statuses"]),
                cache_mode=row["cache_mode"],
                median_ms=f"{median_ms:.1f}" if isinstance(median_ms, (int, float)) else "n/a",
                prompt=row["max_prompt_tokens"],
                cacheable=row["max_cacheable_prefix_tokens"],
                cached=format_range(row["min_cached_tokens"], row["max_cached_tokens"]),
                suffix=row["max_suffix_prefill_tokens"],
                prompt_ratio=(
                    f"{prompt_ratio:.1%}"
                    if isinstance(prompt_ratio, (int, float))
                    else "n/a"
                ),
                efficiency=(
                    format_percent_range(min_efficiency, max_efficiency)
                    if isinstance(min_efficiency, (int, float))
                    and isinstance(max_efficiency, (int, float))
                    else "n/a"
                ),
            )
        )
    return "\n".join(lines) + "\n"


def build_request(prefix_repetitions: int, pattern: str) -> BenchRequest:
    prefix = "Skippy prompt cache benchmark shared prefix. " * prefix_repetitions
    measured_tail = "Measure the reusable prefix and answer with the word measured."
    warmup_tail = measured_tail
    if pattern == "shared-prefix":
        warmup_tail = "Warm the reusable prefix and answer with the word warmup."
    return BenchRequest(
        shared_prefix=prefix,
        warmup_tail=warmup_tail,
        measured_tail=measured_tail,
        pattern=pattern,
    )


def require_warm_cache(rows: list[dict[str, Any]]) -> None:
    warm_rows = [row for row in rows if row["name"] in {"warm native", "warm Skippy"}]
    failures = [
        row["name"]
        for row in warm_rows
        if not isinstance(row.get("max_cached_tokens"), int)
        or row["max_cached_tokens"] <= 0
    ]
    if failures:
        joined = ", ".join(failures)
        raise SystemExit(f"warm cache proof failed: {joined} reported zero cached tokens")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--llama-base-url",
        help="Use one llama-server endpoint for both native rows; prefer separate cold/warm URLs for strict comparisons",
    )
    parser.add_argument(
        "--llama-cold-base-url",
        help="llama-server endpoint started with prompt cache disabled, for example --cache-ram 0",
    )
    parser.add_argument(
        "--llama-warm-base-url",
        help="llama-server endpoint started with prompt cache enabled, for example --cache-ram N",
    )
    parser.add_argument("--skippy-cold-base-url", required=True)
    parser.add_argument("--skippy-warm-base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key")
    parser.add_argument("--prefix-repetitions", type=int, default=256)
    parser.add_argument(
        "--pattern",
        choices=("exact", "shared-prefix"),
        default="exact",
        help="exact repeats the whole prompt; shared-prefix warms a common prefix with a different tail",
    )
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--allow-missing-warm-cache",
        action="store_true",
        help="write timing results even if warm rows report zero cached tokens",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("target/skippy-openai-cache-matrix"),
    )
    args = parser.parse_args()

    if args.prefix_repetitions <= 0:
        raise SystemExit("--prefix-repetitions must be positive")
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    if args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be positive")
    llama_cold_base_url = args.llama_cold_base_url or args.llama_base_url
    llama_warm_base_url = args.llama_warm_base_url or args.llama_base_url
    if not llama_cold_base_url or not llama_warm_base_url:
        raise SystemExit(
            "pass --llama-cold-base-url and --llama-warm-base-url, or pass --llama-base-url"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    request = build_request(args.prefix_repetitions, args.pattern)
    rows = [
        run_llama_row(
            llama_cold_base_url,
            request,
            args.repeats,
            args.max_tokens,
            cache_prompt=False,
            timeout=args.timeout,
            warmup=False,
        ),
        run_skippy_row(
            args.skippy_cold_base_url,
            args.model,
            request,
            args.repeats,
            args.max_tokens,
            args.api_key,
            args.timeout,
            warmup=False,
            name="cold Skippy",
            cache_mode="prefix-cache-disabled",
        ),
        run_llama_row(
            llama_warm_base_url,
            request,
            args.repeats,
            args.max_tokens,
            cache_prompt=True,
            timeout=args.timeout,
            warmup=True,
        ),
        run_skippy_row(
            args.skippy_warm_base_url,
            args.model,
            request,
            args.repeats,
            args.max_tokens,
            args.api_key,
            args.timeout,
            warmup=True,
            name="warm Skippy",
            cache_mode="prefix-cache-enabled",
        ),
    ]
    result = {
        "model": args.model,
        "pattern": request.pattern,
        "prefix_repetitions": args.prefix_repetitions,
        "max_tokens": args.max_tokens,
        "repeats": args.repeats,
        "rows": rows,
    }
    (args.output_dir / "cache-matrix.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output_dir / "cache-matrix.md").write_text(markdown_report(rows))
    print(markdown_report(rows))
    print(f"Wrote {args.output_dir / 'cache-matrix.json'}")
    print(f"Wrote {args.output_dir / 'cache-matrix.md'}")
    if not args.allow_missing_warm_cache:
        require_warm_cache(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
