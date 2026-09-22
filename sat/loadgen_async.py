#!/usr/bin/env python3
"""Async-workload throughput: tokens produced per window at a held in-flight level.

Usage: loadgen_async.py OUT_DIR LEVELS  (e.g. "4,8,16")
Env: SAT_SETTLE (s, default 15), SAT_WINDOW (s, default 60), SAT_MODEL, SAT_URL.

Workers stream completions and timestamp every generated token, so a window
counts exactly the tokens produced inside it regardless of how long requests
run. Levels only ever add workers (4 -> 8 -> 16): earlier workers keep going,
nothing is cancelled. Per-request latency is recorded but is not the metric.
"""
import json, os, sys, threading, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loadgen import build_trace  # same seeded trace as the saturation sweep

URL = os.environ.get("SAT_URL", "http://127.0.0.1:9447/v1/chat/completions")
MODEL = os.environ.get("SAT_MODEL", "Qwen/Qwen3-8B-GGUF:Q4_K_M")


def main():
    out, levels = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
    settle = float(os.environ.get("SAT_SETTLE", "15"))
    window = float(os.environ.get("SAT_WINDOW", "60"))
    trace = build_trace()
    lock = threading.Lock()
    nxt = [0]
    token_times = []      # (timestamp) of every streamed token
    requests = []         # per-request records
    stop = threading.Event()

    def worker(wid):
        while not stop.is_set():
            with lock:
                item = trace[nxt[0] % len(trace)]
                nxt[0] += 1
            body = {"model": MODEL, "messages": [{"role": "user", "content": item["prompt"]}],
                    "max_tokens": item["max_tokens"], "temperature": 0, "stream": True,
                    "stream_options": {"include_usage": True}}
            req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": "Bearer mesh"})
            rec = {"worker": wid, "idx": item["idx"], "start": time.time(), "chunks": 0}
            local_times = []
            try:
                with urllib.request.urlopen(req, timeout=3600) as resp:
                    for raw in resp:
                        line = raw.decode("utf-8", "replace").strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except ValueError:
                            continue
                        if chunk.get("usage"):
                            rec["completion_tokens"] = chunk["usage"].get("completion_tokens")
                        for choice in chunk.get("choices") or []:
                            delta = choice.get("delta") or {}
                            if delta.get("content") or delta.get("reasoning_content"):
                                local_times.append(time.time())
                rec["status"] = "ok"
            except Exception as e:  # noqa: BLE001
                rec["status"] = "error"
                rec["error"] = repr(e)[:300]
                time.sleep(1)
            rec["end"] = time.time()
            rec["chunks"] = len(local_times)
            with lock:
                token_times.extend(local_times)
                requests.append(rec)

    threads, summaries = [], []
    for level in levels:
        while len(threads) < level:
            t = threading.Thread(target=worker, args=(len(threads),), daemon=True)
            t.start()
            threads.append(t)
        time.sleep(settle)
        w_start = time.time()
        time.sleep(window)
        w_end = time.time()
        with lock:
            # token_times only holds tokens of finished requests; add in-flight
            # requests' tokens at the end, so count at the end of the run instead.
            pass
        summaries.append({"level": level, "w_start": w_start, "w_end": w_end, "window_s": window})
        print(f"LEVEL-WINDOW {level} {w_start:.3f} {w_end:.3f}", flush=True)

    # Let in-flight requests finish so every token in the windows is recorded.
    stop.set()
    for t in threads:
        t.join(timeout=3600)
    with open(os.path.join(out, "async-requests.jsonl"), "w") as f:
        for rec in requests:
            f.write(json.dumps(rec) + "\n")
    ok = [r for r in requests if r["status"] == "ok" and r.get("completion_tokens")]
    calib = (sum(r["chunks"] for r in ok) / max(1, sum(r["completion_tokens"] for r in ok))) if ok else None
    for s in summaries:
        n = sum(1 for t in token_times if s["w_start"] <= t < s["w_end"])
        s["streamed_tokens"] = n
        s["output_tok_s"] = n / s["window_s"]
        s["errors"] = sum(1 for r in requests if r["status"] != "ok" and s["w_start"] <= r["end"] < s["w_end"])
        s["chunks_per_usage_token"] = calib
        print("ASYNC-SUMMARY " + json.dumps(s), flush=True)
    with open(os.path.join(out, "async-summary.json"), "w") as f:
        json.dump(summaries, f, indent=1)


if __name__ == "__main__":
    main()
