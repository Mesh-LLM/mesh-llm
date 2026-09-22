#!/usr/bin/env python3
"""Closed-loop load generator for one concurrency level.

Usage: loadgen.py LEVEL OUT_DIR [--warmup S] [--window S]

The trace is deterministic (seed 20260918): mixed prompt lengths, max_tokens
uniform in [200, 800], non-streaming. Every request carries a unique id at the
start of the prompt so no two requests share a cacheable prefix. Request i is
the same text in every arm.
"""
import json, os, random, sys, threading, time, urllib.request

SEED = 20260918
URL = "http://127.0.0.1:9447/v1/chat/completions"
MODEL = os.environ.get("SAT_MODEL", "Qwen/Qwen3-8B-GGUF:Q4_K_M")

TOPICS = [
    "how TCP congestion control evolved from Tahoe to BBR",
    "the design trade-offs of log-structured merge trees",
    "why the Roman Republic collapsed into the Empire",
    "how vaccines train the adaptive immune system",
    "the economics of container shipping since 1956",
    "how a compiler turns source code into machine code",
    "the causes and consequences of the 2008 financial crisis",
    "how plate tectonics shapes mountains and oceans",
    "the history and mechanics of public-key cryptography",
    "how modern CPUs predict branches and execute out of order",
    "the ecology of coral reefs and threats to them",
    "how the transformer architecture processes language",
]
SENTENCES = [
    "The committee reviewed the quarterly figures and noted a steady rise in operating costs.",
    "Field engineers reported intermittent faults on the northern relay after the storm.",
    "A second survey found that most respondents preferred the revised schedule.",
    "The prototype handled sustained load well but degraded under burst traffic.",
    "Several suppliers raised prices, citing freight delays and currency movements.",
    "The audit identified three undocumented dependencies in the build pipeline.",
    "Researchers measured soil moisture at forty sites across two growing seasons.",
    "The migration plan assumed a maintenance window that operations never approved.",
    "Customer tickets about latency doubled in the week after the release.",
    "The archive contained letters, invoices and a partial ledger from 1887.",
    "Test results diverged between the staging cluster and the developer laptops.",
    "The council deferred the vote pending an independent traffic assessment.",
]


def build_trace(n=6000):
    rng = random.Random(SEED)
    trace = []
    for i in range(n):
        r = rng.random()
        if r < 0.3:
            k = 0  # short: instruction only (~40 tokens)
        elif r < 0.7:
            k = rng.randint(15, 30)  # medium: ~300-600 tokens of context
        else:
            k = rng.randint(60, 90)  # long: ~1200-1800 tokens of context
        notes = " ".join(rng.choice(SENTENCES) for _ in range(k))
        topic = rng.choice(TOPICS)
        prompt = f"[request {i:05d}] "
        if notes:
            prompt += f"Background notes:\n{notes}\n\n"
        prompt += (f"Write a long, detailed essay on {topic}. "
                   "Cover history, mechanisms, examples and open problems. Do not stop early.")
        trace.append({"idx": i, "prompt": prompt, "max_tokens": rng.randint(200, 800)})
    return trace


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main():
    level = int(sys.argv[1])
    out = sys.argv[2]
    warmup = float(os.environ.get("SAT_WARMUP", "60"))
    window = float(os.environ.get("SAT_WINDOW", "240"))
    start_idx = int(os.environ.get("SAT_START_IDX", "0"))
    trace = build_trace()
    lock = threading.Lock()
    nxt = [start_idx]
    records = []
    t0 = time.time()
    w_start, w_end = t0 + warmup, t0 + warmup + window
    stop_issuing = [False]

    def worker(wid):
        time.sleep(wid * min(10.0, warmup / 4) / max(level, 1))  # stagger to avoid phase lock
        while True:
            with lock:
                if stop_issuing[0] or time.time() >= w_end:
                    return
                item = trace[nxt[0] % len(trace)]
                nxt[0] += 1
            body = {"model": MODEL, "messages": [{"role": "user", "content": item["prompt"]}],
                    "max_tokens": item["max_tokens"], "temperature": 0, "stream": False}
            req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": "Bearer mesh"})
            s = time.time()
            rec = {"level": level, "worker": wid, "idx": item["idx"], "max_tokens": item["max_tokens"],
                   "start": s}
            try:
                r = json.load(urllib.request.urlopen(req, timeout=1800))
                u = r.get("usage") or {}
                rec.update(status="ok", prompt_tokens=u.get("prompt_tokens"),
                           completion_tokens=u.get("completion_tokens") or 0,
                           finish=(r.get("choices") or [{}])[0].get("finish_reason"))
            except Exception as e:  # noqa: BLE001
                body_txt = ""
                if hasattr(e, "read"):
                    try:
                        body_txt = e.read()[:200].decode(errors="replace")
                    except Exception:  # noqa: BLE001
                        pass
                rec.update(status="error", error=f"{e!r} {body_txt}"[:300], completion_tokens=0)
                time.sleep(1)
            rec["end"] = time.time()
            with lock:
                records.append(rec)

    threads = [threading.Thread(target=worker, args=(w,), daemon=True) for w in range(level)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=max(0.0, w_end - time.time()) + 1800)
    stop_issuing[0] = True

    # Tokens attributed to the window: each request's completion tokens prorated by
    # the fraction of its lifetime that overlaps [w_start, w_end].
    win_tokens = 0.0
    lat, done_in_window, errors = [], 0, 0
    for rec in records:
        s, e = rec["start"], rec["end"]
        ov = max(0.0, min(e, w_end) - max(s, w_start))
        if rec["status"] == "ok" and e > s:
            win_tokens += rec["completion_tokens"] * ov / (e - s)
        if s >= w_start and s < w_end:
            if rec["status"] == "ok":
                lat.append(e - s)
            else:
                errors += 1
        if rec["status"] == "ok" and w_start <= e <= w_end:
            done_in_window += 1
    summary = {
        "level": level,
        "window_s": window,
        "warmup_s": warmup,
        "w_start": w_start,
        "w_end": w_end,
        "output_tok_s": win_tokens / window,
        "requests_completed_in_window": done_in_window,
        "requests_started_in_window": len(lat) + errors,
        "errors_started_in_window": errors,
        "errors_total": sum(1 for r in records if r["status"] != "ok"),
        "e2e_p50_s": pct(lat, 50),
        "e2e_p99_s": pct(lat, 99),
        "mean_completion_tokens": (sum(r["completion_tokens"] for r in records if r["status"] == "ok")
                                   / max(1, sum(1 for r in records if r["status"] == "ok"))),
        "next_idx": nxt[0],
    }
    with open(os.path.join(out, f"requests-c{level}.jsonl"), "w") as f:
        for rec in sorted(records, key=lambda r: r["start"]):
            f.write(json.dumps(rec) + "\n")
    with open(os.path.join(out, f"summary-c{level}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("LEVEL-SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
