#!/usr/bin/env python3
"""Grounded, judge-free rubric for the committee task fixture.

Why this exists
---------------
Every quality number in RESULTS.md came from an LLM judging prose pairwise, and
the powered 9B run (RESEARCH/MOA_9B_CONFIRMATION_2026_08_27.md) showed the
verdict is largely a length effect: r(length, verdict) = +0.387/+0.405, and
100% of wins in both arms lived in the "MoA wrote more" subset. A second judge
did not help — it shares the bias.

So: replace the judge with checkable content. Each task in
crates/mesh-mixture-of-agents/tests/fixtures/committee_tasks.json has a known
correct answer set (the defect planted in the snippet, the failure mode, the
required steps). Each rubric item is a deterministic regex over the answer.

Anti-padding by construction
----------------------------
1. `items` are CREDIT: a real, specific, checkable point. Padding cannot earn
   them because they name mechanisms, not adjectives.
2. `decoys` are DEBIT: confident claims that are wrong for this snippet. A
   model that lists every generic concern to look thorough hits decoys. This is
   what stops "write more" from being a winning strategy — the exact hole in
   the LLM judge.

score = covered_items/len(items) - 0.5 * hit_decoys/max(1,len(decoys))

Usage:
    python3 rubric.py run.jsonl [run2.jsonl ...]

Reads the same JSONL the e2e tests emit (text_a = solo, text_b = MoA).
"""

import json
import math
import re
import sys
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Rubrics. Keyed by task_id from committee_tasks.json.
#   items:  (name, regex) — must be present for credit
#   decoys: (name, regex) — wrong-for-this-snippet claim, costs half an item
# Regexes are case-insensitive, matched against the whole answer.
# ---------------------------------------------------------------------------

R = {
    # ---------------- code_review ----------------
    "review_error_handling": {
        "items": [
            ("unwrap_panics", r"\bunwrap\b.{0,80}(panic|crash|abort)|panic.{0,80}\bunwrap\b"),
            ("return_result", r"Result<|->\s*Result|return\s+a\s+Result|anyhow|thiserror"),
            ("status_not_checked", r"status|non-?2xx|4xx|5xx|error_for_status"),
            ("no_timeout", r"timeout|time\s*out|hang(s|ing)? forever|never returns"),
            ("body_size", r"unbounded|body size|content-?length|memory|large response|stream"),
        ],
        "decoys": [
            ("claims_blocking", r"blocking (call|io)|not async|synchronous call"),
        ],
    },
    "review_lock_scope": {
        "items": [
            ("lock_held_across_await", r"(held|hold|holding).{0,60}await|across.{0,20}await|await.{0,40}while.{0,20}(lock|guard)"),
            ("contention_or_serialized", r"serial(is|iz)ed|contention|blocks other|no concurrency|one at a time|deadlock"),
            ("drop_guard", r"drop\(|release the lock|scope the lock|narrow.{0,20}(the )?(lock|critical)|reacquire|re-?acquire"),
            ("early_return_poisons", r"\?\s*operator|early return|`\?`|error propagat"),
        ],
        "decoys": [
            ("claims_std_mutex_await", r"std::sync::Mutex (cannot|can't|must not) be (held )?across"),
        ],
    },
    "review_unbounded": {
        "items": [
            ("unbounded_spawn", r"unbounded|unlimited|no (back)?pressure|spawn.{0,40}(without|no) (limit|bound)|task explosion|exhaust"),
            ("semaphore_or_jointset", r"semaphore|JoinSet|bounded (channel|queue)|worker pool|concurrency limit|permits?"),
            ("unwrap_on_recv", r"recv\(\)?.{0,40}unwrap|unwrap.{0,40}(closed|None|sender dropped)|panics? when.{0,30}(channel|sender)"),
            ("memory_or_oom", r"OOM|out of memory|memory|file descriptor|fd exhaust"),
        ],
        "decoys": [
            ("claims_needs_mutex", r"needs? a mutex|add a Mutex around"),
        ],
    },
    "review_error_swallow": {
        "items": [
            ("debug_level_filtered", r"debug.{0,60}(level|filter|not (enabled|emitted)|suppress)|RUST_LOG|log level"),
            ("raise_level", r"\bwarn\b|\berror\b.{0,30}level|log at (warn|error)|tracing::(warn|error)"),
            ("metric_or_alert", r"metric|counter|alert|monitor|dashboard|SLO"),
            ("no_aggregation_or_propagate", r"propagat|return.{0,20}(the )?error|collect.{0,20}(the )?(errors|failures)|aggregate|partial failure"),
        ],
        "decoys": [
            ("claims_loop_stops", r"loop (stops|exits|breaks) on (the )?(first )?error|short-?circuits the loop"),
        ],
    },
    "review_timeout_math": {
        "items": [
            ("budget_exceeds_client", r"(30|25).{0,60}(exceed|longer than|more than|blow|over).{0,20}(budget|client|30)|3\s*[x×]\s*10|30s?\s*(of )?retr(y|ies)"),
            ("retries_multiply", r"retr(y|ies).{0,40}(multipl|3\s*[x×]|stack|serial|add up|cumulative)|10s?\s*[x×]\s*3"),
            ("client_gives_up_first", r"client.{0,60}(gives up|times out|abandons|disconnect)|work continues|orphan|wasted work"),
            ("shrinking_budget", r"(deadline|budget) propagat|shrink|decreas.{0,20}budget|pass.{0,20}deadline|remaining time"),
        ],
        "decoys": [
            ("claims_total_75", r"\b75s\b|\b75 seconds\b"),
        ],
    },
    "review_partial_write": {
        "items": [
            ("truncate_on_crash", r"partial|truncat|torn|half-?written|corrupt"),
            ("no_fsync", r"fsync|sync_all|sync_data|flush.{0,20}(to )?disk|durab"),
            ("atomic_rename", r"rename|temp(orary)? file|\.tmp|atomic"),
            ("fsync_dir", r"(sync|fsync).{0,30}(directory|dir|parent)|directory entry"),
        ],
        "decoys": [
            ("claims_write_all_partial", r"write_all.{0,40}(may|can|might) write (only )?part"),
        ],
    },
    "review_retry_storm": {
        "items": [
            ("thundering_or_amplify", r"thundering herd|retry storm|amplif|synchron(is|iz)ed|5\s*[x×]\s*(the )?load|self-?inflicted|congestion collapse"),
            ("backoff", r"backoff|back-?off"),
            ("jitter", r"jitter|random(is|iz)"),
            ("circuit_breaker_or_budget", r"circuit break|retry budget|token bucket|rate limit|adaptive"),
            ("only_retryable", r"idempot|only retry|retryable|429|503|non-?retryable|4xx"),
        ],
        "decoys": [
            ("claims_5_is_fine", r"5 retries is (fine|reasonable|acceptable)"),
        ],
    },
    "review_api_shape": {
        "items": [
            ("boolean_blindness", r"boolean (blind|trap)|unreadable|which (bool|flag)|positional|call site.{0,40}(mean|unclear)|magic"),
            ("struct_or_builder", r"struct|builder|config(uration)? (struct|object)|named (field|argument)"),
            ("newtype_or_enum", r"enum|newtype|typed|Duration"),
            ("duration_not_u64", r"Duration|seconds\?|unit(s)? (are|is) (unclear|ambiguous)|ms or s|millisecond"),
            ("error_type", r"String.{0,40}error|typed error|thiserror|Result<\(\),\s*\w*Error"),
        ],
        "decoys": [],
    },
    "review_test_quality": {
        "items": [
            ("only_asserts_ok", r"only (asserts|checks|proves).{0,40}(ok|not.{0,10}err)|is_ok.{0,60}(nothing|little|weak)|tautolog|vacuous"),
            ("no_value_check", r"assert_eq|check the (value|result|route|target)|unwrap.{0,20}(and )?(compare|assert)|which (route|target)"),
            ("no_negative_case", r"negative|error case|unknown model|failure case|is_err"),
            ("table_or_cases", r"table[- ]driven|parameter(is|iz)ed|multiple cases|edge case"),
        ],
        "decoys": [
            ("claims_async_needed", r"should be (an )?async test|needs tokio::test"),
        ],
    },
    "review_cancellation": {
        "items": [
            ("select_drops_future", r"drop(ped|s)?.{0,60}(future|write_batch|branch)|cancel(led|s|lation).{0,40}(mid|part|future)|not cancel-?safe"),
            ("data_loss", r"lose|lost|drop(ped)? data|silently discard|partial(ly)? (written|drained)|inconsisten"),
            ("buffer_state_unknown", r"buffer.{0,60}(unknown|indeterminate|partially|half)|no way to know how much"),
            ("fix_spawn_or_biased", r"spawn|JoinHandle|complete the write|graceful|drain (first|before)|CancellationToken|biased"),
        ],
        "decoys": [
            ("claims_select_awaits_both", r"select.{0,40}(waits for|awaits) both"),
        ],
    },
    # ---------------- reason_over_output ----------------
    "reason_root_cause": {
        "items": [
            ("dedup_or_identity", r"dedup|duplicate|identity|equality|Eq|Hash|peer id|same peer.{0,30}(twice|counted)"),
            ("self_included", r"self|own (node|peer)|includes itself"),
            ("test_expectation_wrong", r"(test|assertion).{0,40}(wrong|stale|outdated|expectation)|off-?by-?one"),
            ("check_first_print", r"print|dbg!|log the (set|peers)|inspect the (actual )?(set|contents)|which (three|3)"),
        ],
        "decoys": [
            ("claims_flaky_network", r"network (flake|flakiness)|DNS"),
        ],
    },
    "reason_diff_review": {
        "items": [
            ("not_guaranteed", r"\bno\b|not guaranteed|does not|doesn't (guarantee|hold)|fails"),
            ("dedup_keeps_first", r"dedup.{0,60}(keeps|retains|first)|removes (all but )?the first|first (occurrence|element)"),
            ("sort_unstable", r"sort_by_key.{0,40}stable|stable sort|unstable|relative order|sort_unstable"),
            ("fix_sort_by_seen", r"sort.{0,40}(last_seen|timestamp|recenc)|sort by.{0,30}(seen|time)|reverse|HashMap.{0,30}insert|max_by_key"),
        ],
        "decoys": [
            ("claims_dedup_needs_sorted_wrong", r"dedup_by_key does not require|no need to sort"),
        ],
    },
    "reason_log_triage": {
        "items": [
            ("upstream_overload_504", r"504|gateway timeout|upstream|overload|saturat"),
            ("hedge_amplifies", r"hedg(e|ing).{0,60}(amplif|more load|extra|duplicate)|fan-?out.{0,30}load"),
            ("cooldown_cascade", r"cooldown.{0,60}(all|every|cascade|no targets|exhaust)|no healthy|all (targets|candidates) unhealthy|correlated"),
            ("rule_out_single_target", r"same (node|model|target|host)|single (backend|target|instance)|is it one|shared"),
        ],
        "decoys": [
            ("claims_client_bug", r"client (bug|misconfigur)"),
        ],
    },
    "reason_perf": {
        "items": [
            ("vram_or_kv_pressure", r"VRAM|memory pressure|KV cache.{0,40}(shrink|evict|smaller)|paging|offload"),
            ("sm_contention", r"SM|compute contention|time-?slic|scheduler|kernel.{0,30}(queue|serial)|MPS"),
            ("bandwidth", r"memory bandwidth|bandwidth-?bound|HBM"),
            ("batching_lost", r"batch(ing)?.{0,40}(smaller|lost|split|less)|continuous batching|concurrency"),
            ("measure_next", r"nvidia-smi|profil|dmon|utili(s|z)ation|measure"),
        ],
        "decoys": [
            ("claims_oom", r"\bis OOM|out of memory (is|being) the cause"),
        ],
    },
    "reason_data": {
        "items": [
            ("pick_a", r"\b(ship|choose|pick|prefer|go with)\b.{0,30}\bA\b|strategy A"),
            ("p99_tail_matters", r"p99|tail.{0,40}(matters|dominates|user|agent)|2\.?5\s*[x×]|5200|multi-?turn.{0,40}(compound|amplif)"),
            ("error_rate_tradeoff", r"0\.4|0\.2|error rate.{0,40}(double|half|twice|small)|retry.{0,30}(cheap|masks)"),
            ("what_changes_mind", r"change my mind|would change|if (the )?(p99|tail)|sample size|traffic (mix|shape)|SLO"),
        ],
        "decoys": [
            ("picks_b_on_p50", r"(ship|choose|pick).{0,20}B\b.{0,120}p50"),
        ],
    },
    "reason_flaky": {
        "items": [
            ("race_missing_sync", r"race|data race|missing (await|join|synchron)|not awaited|ordering"),
            ("timing_assumption", r"sleep|timing assumption|slower (machine|CI)|fewer (cores|cpus)|scheduler|single-?threaded runtime"),
            ("atomic_ordering_or_shared", r"Ordering::|Relaxed|atomic|Mutex|Arc"),
            ("confirm_how", r"loom|stress|--test-threads|repeat|run in a loop|seed|tokio::test.{0,40}(single|multi)|instrument|logging"),
        ],
        "decoys": [
            ("claims_ci_network", r"network (issue|flake) in CI"),
        ],
    },
    "reason_memory": {
        "items": [
            ("many_small_or_fragmentation", r"fragment|many small|death by a thousand|allocator|arena|jemalloc|glibc"),
            ("unbounded_growth_structure", r"unbounded (growth|map|cache|queue)|cache.{0,30}(no|without) (eviction|TTL)|never (evicted|freed|cleared)|leak.{0,30}(per (request|connection))"),
            ("per_connection_or_task_leak", r"connection|task|session|handle.{0,30}(not|never) (closed|dropped)|retain"),
            ("measure_next", r"RSS vs heap|jemalloc stats|massif|heaptrack|/proc|count.{0,30}(live|objects)|gauge"),
        ],
        "decoys": [
            ("claims_profiler_conclusive", r"profiler (already )?(shows|proves) (the )?leak"),
        ],
    },
    "reason_regression": {
        "items": [
            ("pin_the_one_dep", r"pin|lockfile|Cargo.lock|revert (just|only) (the )?(one|dependency)|cargo tree|which (crate|version)"),
            ("platform_difference", r"Linux|epoll|TLS|openssl|rustls|kernel|musl|glibc|IPv6|localhost.{0,20}(resolution|::1)"),
            ("reproduce_in_container", r"docker|container|same image|reproduce.{0,30}(locally|in CI)|CI shell|ssh into"),
            ("capture_wire", r"tcpdump|pcap|strace|RUST_LOG|verbose|server-?side log|who (closed|reset)"),
        ],
        "decoys": [
            ("claims_bisect_needed", r"bisect the (whole|entire) dependency tree"),
        ],
    },
    "reason_metric_conflict": {
        "items": [
            ("wrong_denominator", r"denominator|averag(e|ing).{0,40}(hides|masks)|aggregat.{0,30}(hides|masks)|per-?(user|tenant|endpoint)|skew"),
            ("server_vs_client", r"server-?side.{0,40}client|client-?side|does not (see|count)|before it reaches|edge|timeout.{0,30}(counted as|success)"),
            ("success_definition", r"HTTP 200.{0,60}(wrong|error|empty|bad)|200 with|counts.{0,20}as success|definition of success|semantic"),
            ("instrument_client", r"(client|browser|RUM|end-?to-?end|synthetic) (metric|instrument|telemetr)|per-?request trace|session success"),
        ],
        "decoys": [
            ("claims_one_is_wrong", r"one of (them|the measurements) (is|must be) (wrong|broken|incorrect)"),
        ],
    },
    "reason_race_output": {
        "items": [
            ("log_ordering_not_causal", r"log.{0,60}(order|buffer|interleav|not causal|async|non-?blocking)|timestamps|writer"),
            ("shutdown_not_awaited", r"(shutdown|complete).{0,60}(before|without) (waiting|awaiting|join)|does not (wait|join)|no drain|premature"),
            ("worker_outlives", r"worker.{0,40}(still|outliv|after|continues)|use after close|handle.{0,30}closed"),
            ("fix_join_or_token", r"join|JoinSet|await (all|the) (task|worker)|CancellationToken|graceful"),
        ],
        "decoys": [
            ("claims_clock_skew", r"clock skew|NTP"),
        ],
    },
    # ---------------- planning ----------------
    "plan_fix": {
        "items": [
            ("distinguish_streaming", r"stream(ing)?.{0,60}(exempt|different|idle|per-?chunk|no total)|SSE|chunked"),
            ("idle_vs_total", r"idle timeout|inactivity|time to first byte|TTFB|total (timeout|deadline)|read timeout"),
            ("propagate_cancel", r"cancel|abort|propagat|deadline|drop the upstream|close the connection"),
            ("rollout_config", r"config|feature flag|default (off|generous)|staged|canary|observe first|metric"),
            ("risks", r"risk|regress|false (positive|timeout)|long-?running (request|job)"),
        ],
        "decoys": [
            ("single_global_timeout", r"a single global timeout (is|will be) (enough|sufficient)"),
        ],
    },
    "plan_migration": {
        "items": [
            ("optional_ignore_unknown", r"optional|unknown field|ignore.{0,30}(unknown|unrecogni)|forward(s)?-?compat|backward(s)?-?compat"),
            ("wire_format_check", r"protobuf|serde|CBOR|JSON|field (number|tag)|deny_unknown|schema"),
            ("write_after_read", r"(read|parse) (support )?(first|before).{0,40}(write|emit|send)|two-?(phase|step)|deploy.{0,30}(readers|first)"),
            ("mixed_version_test", r"mixed[- ]version|old.{0,20}new|both directions|round-?trip|compat(ibility)? test"),
            ("rollback_safe", r"rollback|revert|downgrade"),
        ],
        "decoys": [
            ("flag_day", r"upgrade all nodes at once|flag day"),
        ],
    },
    "plan_debug": {
        "items": [
            ("correlate_ids", r"request id|correlation|trace id|tracing|sample.{0,20}(request|trace)"),
            ("empty_response_source", r"empty.{0,60}(finish_reason|truncat|token|upstream|timeout|cancel)|which layer|where.{0,30}(empty|lost)"),
            ("load_correlation", r"correlate.{0,40}(load|concurrency|queue|utili)|under load|only at (high )?(load|concurrency)|5%"),
            ("nondisruptive", r"shadow|sampl|read-?only|non-?disruptive|no restart|dynamic log level|canary|one (node|instance)"),
            ("hypothesis_order", r"first|then|priority|narrow|bisect|rule out"),
        ],
        "decoys": [
            ("reproduce_locally_only", r"reproduce (it )?locally (first|is the first step)"),
        ],
    },
    "reason_ambiguous_req": {
        "items": [
            ("which_metric", r"which (metric|latency)|TTFT|time to first token|tok(ens)?/s|throughput vs latency|p50|p99|define (fast|faster)"),
            ("workload_shape", r"workload|prompt (length|size)|context (size|length)|batch|concurren|single-?user|agentic"),
            ("baseline_first", r"baseline|measure (first|before)|current numbers|profile"),
            ("levers", r"quant|batching|KV cache|specul|split|hardware|model size|network"),
            ("constraints", r"budget|cost|quality (loss|regress)|accuracy|hardware available"),
        ],
        "decoys": [
            ("just_buy_gpus", r"just (buy|add) (more )?(GPUs|hardware)$"),
        ],
    },
    "plan_test": {
        "items": [
            ("version_matrix", r"matrix|old.{0,20}new|N-?1|both directions|cross-?version"),
            ("two_machines_roles", r"two machines|machine A|different hardware|one (as|runs) (old|new)|heterogen"),
            ("determinism_or_golden", r"golden|fixture|deterministic|seed|record(ed)? (trace|request)|replay"),
            ("routing_assertions", r"assert.{0,40}(route|target|which model)|routing decision|distribution|counter"),
            ("failure_injection", r"fault inject|kill|partition|drop|unhealthy|chaos"),
        ],
        "decoys": [
            ("unit_tests_enough", r"unit tests (are|should be) (enough|sufficient)"),
        ],
    },
    "plan_rollback": {
        "items": [
            ("schema_blocks_rollback", r"(schema|migration).{0,80}(cannot|can't|not).{0,20}(roll ?back|revert)|backward.{0,20}incompat|one-?way|destructive"),
            ("revert_request_path_only", r"revert (just|only) the (request|code) path|separate the two|disable (the )?(new )?code path|feature flag"),
            ("triage_first", r"which (change|part)|is the error from|logs|correlate|3\s*[x×].{0,30}(what|which)|scope the impact"),
            ("comms_and_timebox", r"declare|incident|page|communicate|timebox|30 minutes|deadline to decide"),
            ("forward_fix_option", r"forward fix|roll forward"),
        ],
        "decoys": [
            ("rollback_everything_now", r"(immediately )?roll ?back everything"),
        ],
    },
    "plan_refactor": {
        "items": [
            ("seams_first", r"seam|interface|trait|boundar|extract.{0,30}(module|trait|struct)|facade"),
            ("incremental_moves", r"incremental|one (piece|concern) at a time|small PR|strangler|step by step"),
            ("no_behaviour_change", r"pure (move|refactor)|no behaviou?r change|mechanical|keep.{0,20}public API"),
            ("tests_as_net", r"test.{0,40}(before|first|net|characteri)|characteri(s|z)ation test|snapshot"),
            ("avoid_conflicts", r"conflict|rebase|others (keep )?shipping|coordinate|short-?lived branch|move files (early|late)"),
        ],
        "decoys": [
            ("freeze_the_module", r"freeze (the )?(module|file)|stop other work"),
        ],
    },
    "plan_cache": {
        "items": [
            ("key_includes_params", r"key.{0,80}(model|prompt|temperature|sampling|version|system prompt|tools)|hash of"),
            ("never_cache", r"never cache|do not cache|exclude|non-?deterministic|temperature\s*>\s*0|streaming|per-?user|personal|auth|tool (call|result)"),
            ("invalidation", r"TTL|invalidat|version.{0,20}(bump|key)|evict"),
            ("prove_not_stale", r"shadow|compare.{0,40}(live|fresh)|sample.{0,30}(bypass|miss)|canary|hit rate.{0,40}(correct|audit)|A/?B"),
            ("privacy_isolation", r"tenant|isolat|leak|cross-?user"),
        ],
        "decoys": [
            ("key_on_prompt_only", r"key (it )?on (just )?the prompt (alone|only)"),
        ],
    },
    "plan_capacity": {
        "items": [
            ("find_bottleneck_first", r"find|identif|measure|profil|load test.{0,40}bottleneck|don'?t (guess|optimi)"),
            ("load_test", r"load test|benchmark|synthetic (load|traffic)|replay (production )?traffic|stress"),
            ("headroom_metrics", r"utili(s|z)ation|saturation|queue (depth|time)|USE method|p99|headroom"),
            ("efficiency_levers", r"cach(e|ing)|batch|quant|shed|rate limit|degrade|admission control"),
            ("priority_order", r"in (priority )?order|first|then|highest (leverage|impact)"),
        ],
        "decoys": [
            ("scale_out_blindly", r"just (scale|add) (out|more) (replicas|nodes) (first|immediately)"),
        ],
    },
    "plan_observability": {
        "items": [
            ("start_at_boundaries", r"boundar|edge|entry point|ingress|RPC layer|middleware|start (with|at) the"),
            ("propagate_context", r"propagat|context|trace(-| )?(id|context)|W3C|traceparent|baggage|header"),
            ("sampling", r"sampl(e|ing)|head-?based|tail-?based|1%|cost"),
            ("pick_one_journey", r"one (flow|journey|path|request type)|highest-?value|the (cross-?node) case that hurts|pilot"),
            ("logs_to_traces_bridge", r"correlate.{0,40}(log|existing)|inject.{0,30}trace id.{0,30}log|span (id|links)|structured log"),
        ],
        "decoys": [
            ("instrument_everything", r"instrument everything (at once|up front)"),
        ],
    },
    # ---------------- explain ----------------
    "explain_concept": {
        "items": [
            ("definition", r"\w"),
        ],
        "decoys": [],
    },
}

# Generic fallback for the `explain` tasks: these have no single planted defect,
# so a regex rubric would be measuring vocabulary, not correctness. They are
# EXCLUDED rather than scored badly — see report note.
EXCLUDED_CATEGORIES = {"explain"}


def score(task_id: str, text: str):
    """Return (score, covered, n_items, decoys_hit, n_decoys) or None if no rubric."""
    r = R.get(task_id)
    if not r or not text:
        return None
    items = r["items"]
    decoys = r.get("decoys", [])
    cov = sum(1 for _, rx in items if re.search(rx, text, re.I | re.S))
    dh = sum(1 for _, rx in decoys if re.search(rx, text, re.I | re.S))
    s = cov / len(items) - 0.5 * (dh / max(1, len(decoys)) if decoys else 0.0)
    return s, cov, len(items), dh, len(decoys)


def sign_p(w, l):
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx and dy else float("nan")


def main(paths):
    for p in paths:
        rows = [json.loads(l) for l in open(p)]
        scored = []
        for r in rows:
            if r.get("category") in EXCLUDED_CATEGORIES:
                continue
            sa = score(r["task_id"], r.get("text_a") or "")
            sb = score(r["task_id"], r.get("text_b") or "")
            if sa is None or sb is None:
                continue
            scored.append((r, sa, sb))

        n = len(scored)
        if not n:
            print(f"{p}: no scorable rows")
            continue
        w = sum(1 for _, sa, sb in scored if sb[0] > sa[0])
        t = sum(1 for _, sa, sb in scored if sb[0] == sa[0])
        l = sum(1 for _, sa, sb in scored if sb[0] < sa[0])
        mean_a = sum(sa[0] for _, sa, _ in scored) / n
        mean_b = sum(sb[0] for _, _, sb in scored) / n

        # length control: does the rubric reward writing more?
        dl = [r["len_b"] - r["len_a"] for r, _, _ in scored]
        ds = [sb[0] - sa[0] for _, sa, sb in scored]
        r_len = pearson(dl, ds)
        # and the judge's own verdict vs the rubric's, on the same rows
        agree = sum(
            1
            for r, sa, sb in scored
            if (r.get("b_vs_a") or 0) == (1 if sb[0] > sa[0] else -1 if sb[0] < sa[0] else 0)
        )

        print(f"\n=== {p} ===")
        print(f"scorable rows        {n}  (explain excluded: no planted answer set)")
        print(f"MoA vs solo (rubric) {w}W / {t}T / {l}L   sign p = {sign_p(w, l):.4g}")
        print(f"mean coverage score  solo {mean_a:.3f}   MoA {mean_b:.3f}   delta {mean_b - mean_a:+.3f}")
        print(f"r(len delta, score delta) = {r_len:+.3f}   <- length leverage on THIS metric")
        print(f"rubric agrees with judge verdict on {agree}/{n} rows ({agree / n:.0%})")

        by_cat = defaultdict(lambda: [0, 0, 0])
        for r, sa, sb in scored:
            c = by_cat[r["category"]]
            c[0 if sb[0] > sa[0] else 1 if sb[0] == sa[0] else 2] += 1
        for c, (cw, ct, cl) in sorted(by_cat.items()):
            print(f"  {c:22s} {cw}W/{ct}T/{cl}L  p={sign_p(cw, cl):.3g}")

        # decoy behaviour: does MoA make more confident-wrong claims?
        da = sum(sa[3] for _, sa, _ in scored)
        db = sum(sb[3] for _, _, sb in scored)
        print(f"decoy hits (confident-wrong claims): solo {da}, MoA {db}")

        # hardest/easiest items, to show the rubric discriminates at all
        cov_a = Counter()
        cov_b = Counter()
        for r, sa, sb in scored:
            cov_a[r["task_id"]] += sa[1]
            cov_b[r["task_id"]] += sb[1]
        worst = sorted(cov_b.items(), key=lambda kv: kv[1])[:3]
        print(f"lowest MoA coverage tasks: {worst}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])


def calibrate(paths):
    """Per-item hit rate across ALL answers (solo and MoA).

    An item at ~0% is almost certainly a broken regex, not a hard item; an item
    at ~100% is not discriminating. This is the guard against the rubric
    repeating the judge's failure — measuring phrasing instead of content.
    """
    hits = defaultdict(int)
    tot = defaultdict(int)
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            spec = R.get(r["task_id"])
            if not spec:
                continue
            for text in (r.get("text_a") or "", r.get("text_b") or ""):
                if not text:
                    continue
                for name, rx in spec["items"]:
                    k = (r["task_id"], name)
                    tot[k] += 1
                    if re.search(rx, text, re.I | re.S):
                        hits[k] += 1
    rows = [(h / tot[k], k, hits[k], tot[k]) for k, h in ((k, hits[k]) for k in tot)]
    rows.sort()
    print("rate   task/item                                  hits/total")
    for rate, k, h, t in rows:
        flag = "  <-- suspect" if rate < 0.10 or rate > 0.97 else ""
        print(f"{rate:5.2f}  {k[0]}/{k[1]:32s} {h}/{t}{flag}")
