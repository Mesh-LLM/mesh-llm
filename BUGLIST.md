# BUGLIST

Ranked inventory of every open **defect** in `Mesh-LLM/mesh-llm`, ordered by the work needed to
test, reproduce, and correct it — cheapest first.

- Snapshot: 2026-10-25, `main` at `8c19d33645`, 141 open issues.
- 53 are defects. The other 88 are features, specs, research/experiments, or tracking issues and are
  listed in [Not ranked](#not-ranked) with the reason.
- Effort is one engineer who knows the code, including repro, fix, and tests, **excluding CI queue
  time and release-pipeline latency**. Hardware-gated items say so; on those, lab access usually
  costs more than the code.
- Ranking is by effort, not severity. See [If you are prioritizing by risk](#if-you-are-prioritizing-by-risk).

Issue bodies were read on the snapshot date. Re-check an issue before starting it: line numbers and
`main` behaviour drift, and several entries cite commits that are no longer tip.

## Ranked

| # | Issue | Defect | Effort |
|---:|---|---|---|
| 1 | [#2119](https://github.com/Mesh-LLM/mesh-llm/issues/2119) | 16 top-level value-taking flags are missing from `VALUE_TAKING_FLAGS`, so their value is parsed as a pseudo-subcommand. Add the global ones plus a test pinning struct ↔ registry. | 1–2 h |
| 2 | [#1955](https://github.com/Mesh-LLM/mesh-llm/issues/1955) | Catalog filters return zero rows (`≤ 8 GB`, Capability, Multi-machine) although matching models exist. Three causes, all in row→attribute mapping, including a double-escaped `data-hardware`. | 2–4 h |
| 3 | [#2339](https://github.com/Mesh-LLM/mesh-llm/issues/2339) | `/mcp` `tools/list`, `resources/list`, `prompts/list` omit `ttlMs` and `cacheScope` for protocol `2026-07-28`, so Claude Code loads none of the node's plugin tools. **IN-PROGRESS** — [PR #2583](https://github.com/Mesh-LLM/mesh-llm/pull/2583) on `fix/3-2339-mcp-cache-metadata`; [repro confirmed](https://github.com/Mesh-LLM/mesh-llm/issues/2339#issuecomment-6404315687). | 2–4 h |
| 4 | [#2210](https://github.com/Mesh-LLM/mesh-llm/issues/2210) | The external-endpoint path rewrites only target and `Host`, forwarding caller `Authorization`, `Proxy-Authorization`, `X-API-Key`, `api-key` over plaintext HTTP. **IN-PROGRESS** — [PR #2584](https://github.com/Mesh-LLM/mesh-llm/pull/2584) on `fix/4-2210-external-endpoint-credentials`; [repro confirmed](https://github.com/Mesh-LLM/mesh-llm/issues/2210#issuecomment-6404684611). | 2–4 h |
| 5 | [#2214](https://github.com/Mesh-LLM/mesh-llm/issues/2214) | Inbound plugin frames compare `source_peer_id` with the QUIC sender but only log the mismatch, then dispatch and may forward the forged identity. | 2–4 h |
| 6 | [#2212](https://github.com/Mesh-LLM/mesh-llm/issues/2212) | KV prune/clear routes are omitted from `requires_trusted_local_access` and accept JSON regardless of `Content-Type`, so a hostile page can evict caches. **IN-PROGRESS** — [PR 2587](https://github.com/Mesh-LLM/mesh-llm/pull/2587) on `fix/6-2212-kv-cache-csrf`; [repro confirmed](https://github.com/Mesh-LLM/mesh-llm/issues/2212#issuecomment-6405182670). | 3–4 h |
| 7 | [#2344](https://github.com/Mesh-LLM/mesh-llm/issues/2344) | Console Chat lists decision-only System One models and shows only "Chat request failed: 422", never the server's reason. | 3–5 h |
| 8 | [#2135](https://github.com/Mesh-LLM/mesh-llm/issues/2135) | `status()` returns `{connected: true, peerCount: 1}` from a boolean before any dial; `peer_count` is `usize::from(self.connected)`, not a measurement. **IN-PROGRESS** — [PR #2589](https://github.com/Mesh-LLM/mesh-llm/pull/2589) on `fix/8-2135-peer-count-measurement`; [repro confirmed](https://github.com/Mesh-LLM/mesh-llm/issues/2135#issuecomment-6405671011). | 4–6 h |
| 9 | [#1954](https://github.com/Mesh-LLM/mesh-llm/issues/1954) | The OpenAI concurrency guard and default still read unclamped `config.lane_count` for encoder-decoder models, so startup admits N while the runtime serves one lane. | 4–6 h |
| 10 | [#2029](https://github.com/Mesh-LLM/mesh-llm/issues/2029) | Unresolvable `*-layers` refs log at `debug`, the resolved set comes up empty, and the coordinator reports only `missing_model_interest` — a diagnostic dead end. | 1 d |
| 11 | [#2385](https://github.com/Mesh-LLM/mesh-llm/issues/2385) | `models show` rejects `hf://` layer-package refs and pinned MXFP4 refs that `serve --model` and `models download` accept. | 1 d |
| 12 | [#2003](https://github.com/Mesh-LLM/mesh-llm/issues/2003) | `platform-checks` declares `cache_mode: none`, so Windows and macOS recompile cold on every `main` push (0.25% sccache hit rate, ~21 min/run). | 1 d + slow CI |
| 13 | [#2384](https://github.com/Mesh-LLM/mesh-llm/issues/2384) | Laya runs on CPU by default but is charged to the GPU fit budget, so it is refused beside a GPU model that it would not compete with. | 1–1.5 d |
| 14 | [#2195](https://github.com/Mesh-LLM/mesh-llm/issues/2195) | Long non-streaming prefill fails at a hardcoded 600 s backend timeout and a hardcoded 10-minute local first-byte timeout, both independent of the client budget. | 1 d |
| 15 | [#2563](https://github.com/Mesh-LLM/mesh-llm/issues/2563) | The llama.cpp upstream canary has failed two consecutive scheduled runs; the earliest phase without progress is unisolated. | 1–2 d |
| 16 | [#2205](https://github.com/Mesh-LLM/mesh-llm/issues/2205) | Native-runtime IDs require only non-emptiness, so absolute or parent components escape the install/cache roots and reach create/truncate and recursive delete. | 1 d |
| 17 | [#2246](https://github.com/Mesh-LLM/mesh-llm/issues/2246) | Payment admission compares the rewritten internal served name against operator-typed pricing keys, so a public-alias request can serve unpaid. | 1–1.5 d |
| 18 | [#2059](https://github.com/Mesh-LLM/mesh-llm/issues/2059) | Paid peers enter the MoA worker pool and a 402 ends that worker with no standby failover, silently shrinking the committee. | 1–1.5 d |
| 19 | [#2216](https://github.com/Mesh-LLM/mesh-llm/issues/2216) | All HTTP MCP sessions share one `ActiveBridge` slot holding the latest client peer, so callbacks can be delivered to the wrong session. | 1.5–2 d |
| 20 | [#2213](https://github.com/Mesh-LLM/mesh-llm/issues/2213) | Transitive gossip replaces direct-peer models, role, capacity, policy, and payment offers while the victim's direct admission stays live. | 1.5–2 d |
| 21 | [#2211](https://github.com/Mesh-LLM/mesh-llm/issues/2211) | Named plugin install passes no expected digest, does not bind the catalog version, and can execute a mutable `latest` release. | 1.5–2 d |
| 22 | [#2340](https://github.com/Mesh-LLM/mesh-llm/issues/2340) | A path naming a `.gguf` is served from the SafeTensors checkpoint in the same directory, while the node still reports the GGUF's context length. | 1.5–2 d |
| 23 | [#2343](https://github.com/Mesh-LLM/mesh-llm/issues/2343) | SafeTensors→GGUF conversion types Qwen's `<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>`, ` thinking`, `</think>` as `NORMAL`, so they split into byte-pair pieces. | 1.5–2 d |
| 24 | [#2338](https://github.com/Mesh-LLM/mesh-llm/issues/2338) | Claude Code gets HTTP 502 on Qwen3.5/3.8: the inline `role: system` turn becomes a second system message and the chat template refuses it. **CONFIRMED** — the Messages adapter now folds inline `role: system` turns into the leading system message, [PR #2612](https://github.com/Mesh-LLM/mesh-llm/pull/2612) on `fix/24-2338-inline-system-fold`; [repro confirmed](https://github.com/Mesh-LLM/mesh-llm/issues/2338#issuecomment-6422277779). | 1.5–2 d |
| 25 | [#2342](https://github.com/Mesh-LLM/mesh-llm/issues/2342) | `config validate` says "Config valid" for a `model_path` that then stops the whole node, and accepts unknown keys and tables without a warning. **CONFIRMED** — the `model_path` half shipped via #2554; unknown keys/tables now warn, [PR #2615](https://github.com/Mesh-LLM/mesh-llm/pull/2615) on `fix/24b-2342-config-unknown-fields`; [repro confirmed](https://github.com/Mesh-LLM/mesh-llm/issues/2342#issuecomment-6422493389). | 2 d |
| 26 | [#2348](https://github.com/Mesh-LLM/mesh-llm/issues/2348) | Named `tool_choice` is passed as `required`; if the model calls a different tool the call is dropped, yielding an empty `stop` reply that still bills the tokens. | 2 d |
| 27 | [#1951](https://github.com/Mesh-LLM/mesh-llm/issues/1951) | `request_workload_class` ends in `_ => None`, so an unclassified inference path gets no workload admission; the only guard is a hand-written enumeration test. **CONFIRMED** — `/v1/messages` is now classified `CausalGeneration` and the workload map is gated against the router's inference surface, [PR #2618](https://github.com/Mesh-LLM/mesh-llm/pull/2618) on `fix/26-1951-workload-route-coverage`; [repro confirmed](https://github.com/Mesh-LLM/mesh-llm/issues/1951#issuecomment-6424609084). | 2 d |
| 28 | [#2352](https://github.com/Mesh-LLM/mesh-llm/issues/2352) | Four papercuts: `stop` skips clients, a node retries its own invite token forever, `--discover <name>` leaves the mesh name unset, no `x-request-id` echo. | 2–2.5 d |
| 29 | [#2508](https://github.com/Mesh-LLM/mesh-llm/issues/2508) | Gemma 4 26B-A4B fails every request with `llama_decode rc=-3` when auto context sizing is uncapped. **Repro on `polars.patio51.com`** (M5 Ultra, 256 GB) — the M5 Max 128 GB in the report is not required, and the larger memory tier should still overshoot. | 2–3 d |
| 30 | [#2217](https://github.com/Mesh-LLM/mesh-llm/issues/2217) | Stage-control `Claim`/`Load`/`Stop`/`Status`/`Inventory` discard the authenticated QUIC requester, so any admitted peer can forge a coordinator identity, fence the real one, or stop a generation. **High.** | 2–3 d |
| 31 | [#2218](https://github.com/Mesh-LLM/mesh-llm/issues/2218) | Stage activation/import lengths allocate up to 512 MiB before the 8 MiB frame limit applies, and accepted connections spawn OS threads before a bounded handshake. **High.** | 2–3 d |
| 32 | [#2206](https://github.com/Mesh-LLM/mesh-llm/issues/2206) | The inference listener spawns work per connection with no global limit, and header/body readers have byte ceilings but no total or idle deadline. | 2–3 d |
| 33 | [#2207](https://github.com/Mesh-LLM/mesh-llm/issues/2207) | Stage tunnels check general mesh admission but never verify the peer is the target's coordinator, predecessor, or assigned topology participant. | 2–3 d |
| 34 | [#2208](https://github.com/Mesh-LLM/mesh-llm/issues/2208) | `SessionTracker::claim` waits only for `Releasing` and then overwrites a live `Owned` session, so a peer can seize, trim, stop, or continue another session. | 2–3 d |
| 35 | [#2193](https://github.com/Mesh-LLM/mesh-llm/issues/2193) | `benchmark tune` launches four lanes despite a one-lane base config, changing the memory and throughput being optimized. **Needs Orin.** | 3 d |
| 36 | [#2192](https://github.com/Mesh-LLM/mesh-llm/issues/2192) | Tune sends `model:auto` with mesh hooks, which suppresses batched native-MTP accept-and-skip, so an MTP sweep measures serial fallback. **Needs Orin.** | 3–4 d |
| 37 | [#2383](https://github.com/Mesh-LLM/mesh-llm/issues/2383) | Plugin host calls re-sort JSON object keys in both directions, a call over 30 s kills and replays the plugin, and every error is flattened to 502. | 3–4 d |
| 38 | [#2387](https://github.com/Mesh-LLM/mesh-llm/issues/2387) | OTLP: failed exports are silent, nodes share one stream identity, `docs/USAGE.md` points at the gRPC port while Mesh sends OTLP/HTTP, histogram buckets overflow, pre-routing rejections are uncounted, and no request timing is exported. | 3–4 d |
| 39 | [#2151](https://github.com/Mesh-LLM/mesh-llm/issues/2151) | Protected workflows pass resolved package lists straight to platform test loops without workspace membership filtering, so removing a crate needs CI repair PRs. | 3–4 d |
| 40 | [#1001](https://github.com/Mesh-LLM/mesh-llm/issues/1001) | The split planner weights per-node layer allocation by `vram_bytes`, a hold-capacity blend, so a RAM-heavy weak node gets a larger pipeline share than a fast GPU. | 3–5 d |
| 41 | [#2182](https://github.com/Mesh-LLM/mesh-llm/issues/2182) | `@mesh-llm/sdk@0.77.0` is not on npm: the packaging run failed fetching a `darwin-x64` addon outside the supported four-target matrix, so assemble and publish were skipped. | 3–5 d release ops |
| 42 | [#2147](https://github.com/Mesh-LLM/mesh-llm/issues/2147) | Embedded startup accepts any successful `/api/status` JSON as readiness, so a foreign local instance on the console port satisfies the handle before the new worker fails to bind. | 4–5 d |
| 43 | [#2055](https://github.com/Mesh-LLM/mesh-llm/issues/2055) | A 2.8 GB model still fails `failed to create llama context` under `CUDA_VISIBLE_DEVICES=1` with correct `vram_bytes`, no CUDA OOM, and no oversized context trace. **Needs 5090 + 3080 host.** | 4–5 d |
| 44 | [#2146](https://github.com/Mesh-LLM/mesh-llm/issues/2146) | The Windows ROCm runtime ships `rocblas.dll` and `hipblaslt.dll` without `rocblas/library` or `hipblaslt/library`, so rocBLAS cannot find `TensileLibrary.dat` and the node exits. **Needs Windows + ROCm.** | 4–6 d |
| 45 | [#2199](https://github.com/Mesh-LLM/mesh-llm/issues/2199) | Successful long requests can lack native decode timing summaries when diagnostics report dropped events; the loss point is explicitly unisolated. | 4–6 d |
| 46 | [#2209](https://github.com/Mesh-LLM/mesh-llm/issues/2209) | Owner-control bootstrap, config, lifecycle, cache, and guardrail routes treat loopback TCP as caller identity and load the owner key on the caller's behalf, so another local account can read secrets or install a plugin command. | 4–6 d |
| 47 | [#2349](https://github.com/Mesh-LLM/mesh-llm/issues/2349) | `mesh-llm load` cannot set context or lanes (v0.78.1 allocates four lanes of the planned context), reports its 30 s timeout as "Can't connect", and drops the load failure reason. | 4–6 d |
| 48 | [#2228](https://github.com/Mesh-LLM/mesh-llm/issues/2228) | The documented Rust public-mesh client accepts the QUIC tunnel stream, writes zero bytes, and closes, so `list_models()` and `openai_stream()` never complete. | 5–8 d |
| 49 | [#2247](https://github.com/Mesh-LLM/mesh-llm/issues/2247) | DeepSeek V4 Flash gets no prefix reuse with the automatic cache payload; `full-state` restores it (20 s → 3 s on a repeated 28.5K prompt). **Needs 4× RTX PRO 6000.** | 5–8 d |
| 50 | [#2158](https://github.com/Mesh-LLM/mesh-llm/issues/2158) | CPU sampling takes ~21% of the decode thread and ~16% of throughput at 16 streams; temperature-0 chains with `ignore_eos`, logit bias, or inert penalties still run the full chain. | 1.5–2 wk |
| 51 | [#2191](https://github.com/Mesh-LLM/mesh-llm/issues/2191) | After an MTP rejection, two serial target-model steps run before drafting resumes, even with a one-token window. **Needs Orin.** | 1–1.5 wk |
| 52 | [#2190](https://github.com/Mesh-LLM/mesh-llm/issues/2190) | Native MTP rejection repair deletes the head's entire historical attention KV and resumes drafting without rebuilding the accepted prefix. **Needs Orin.** | 1–1.5 wk |
| 53 | [#2318](https://github.com/Mesh-LLM/mesh-llm/issues/2318) | DFlash phase 2: Skippy's native speculative verify path has five overheads upstream does not, making DFlash ~2.3× slower than baseline and native MTP with >1 draft ~2.6× slower than no speculation on hybrid models. | 2–3 wk |

## Bands

**1–9, under a day.** Every item names the file and the cause. These are one batch: registry gap,
catalog attribute mapping, MCP schema fields, three small security gaps, two console/reporting
fixes, two one-line correctness reads.

**10–28, one to two and a half days.** Single-crate fixes with deterministic tests. The cost is
usually one design decision rather than code volume:

- #2246 needs a ruling on whether pricing keys are public IDs or internal served names.
- #2344 needs a ruling on filtering versus disabling in the picker.
- #2342 needs a decision on whether one bad model row may stop the node.
- #2348 needs a grammar constraint in llama.cpp, or an explicit reject instead of an empty `stop`.
- #2059 has a small fix (exclude priced peers, treat 402 as retryable) plus an open design question
  about paying one layer below MoA.

**29–47, two to six days.** These need a real topology, a release pipeline, or hardware in the loop.
The `#2205`–`#2218` security cluster is uniform in shape but not in cost: #2210/#2212/#2214 are
hours, while #2209/#2217/#2218 are days, mostly because each needs negative-path tests that must not
disturb a live node. #2182 and #2146 are release/packaging recoveries, not code changes.

**48–53, one to three weeks.** Native llama.cpp patch-queue work and lab-bound KV/speculative
measurement. #2318 is the only entry that is really a multi-fix program rather than an issue.

Aggregate: roughly **120–140 engineer-days**, about a third of it in the last six items.

## Hardware gates

| Gate | Issues | Effect on the estimate |
|---|---|---|
| Jetson AGX Orin 64 GB (two hosts used in the reports) | #2190, #2191, #2192, #2193, #2199 | Without the hosts these are unverifiable, not merely slower |
| Windows + ROCm (gfx1103 observed) | #2146 | Needs a runtime rebuild and a Windows ROCm install to prove the fix |
| RTX 5090 + RTX 3080 in one host | #2055 | The failure only appears with `CUDA_VISIBLE_DEVICES=1` on that pair |
| 4× RTX PRO 6000 96 GB | #2247 | The reported model does not fit on smaller cards |
| Apple M5 Ultra 256 GB — `polars.patio51.com` | #2508 | Auto sizing overshoots at that memory tier; the 128 GB M5 Max in the report is not a requirement, and the larger machine should reproduce the same uncapped auto-ctx failure |
| Public mesh + real invite tokens | #2228, #2135, #2029 | Nondeterministic peers make repro flaky |

If those hosts are already in the lab, each item drops roughly one band. If they are not, the
estimate is a lower bound on calendar time, not on engineering effort.

## If you are prioritizing by risk

Ranking by effort puts two **high-severity** items at 30 and 31, behind a catalog filter and a
`debug!` line. By severity, the order at the top is:

1. #2217 — any admitted peer can forge a coordinator identity and stop a live generation.
2. #2218 — an admitted peer can exhaust memory and threads before admission checks.
3. #2209 — another local OS account can read secrets and install an executable plugin command.
4. #2205 — path-like runtime IDs reach create/truncate and recursive delete.
5. #2208, #2207, #2206 — session takeover, tunnel authorization, connection exhaustion.
6. #2211, #2213, #2214, #2212, #2210 — supply chain, gossip provenance, identity spoofing, CSRF, credential leakage.
7. #2246 — unpaid inference on a paid mesh.

The whole security cluster is ~20 engineer-days and is the only block where "not yet fixed" is
itself the risk, so it is a reasonable candidate to pull ahead of the 1-day papercuts.

## Not ranked

Excluded because they are capabilities, investigations, or process work rather than wrong behavior.
Each is listed with the reason it does not belong in a bug ranking.

### Features and capability gaps

| Issue | Why excluded |
|---|---|
| #165 pipeline routing | Extend a speculative feature to remote models; not a defect |
| #170 TTS + model-loading contract | New subsystem |
| #182 ACP agent pooling | New subsystem |
| #265 behavioral trust scoring | New subsystem |
| #369 parallel slots vs replicas | Specification for config/UX |
| #811 Hugging Face PKCE | New auth flow |
| #943 MoA idle-worker judges | Enhancement |
| #987 MoE-aware elastic batching | Enhancement |
| #1020 RTT matrix for placement | Enhancement |
| #1052 per-model topology TOML | Replaces a flag; not a defect |
| #1128 activation bandwidth in placement | Enhancement |
| #1131 catalog-ID override | Feature request with a bug framing; the bug it works around is #2029 |
| #1185 expert usage stats | Enhancement |
| #1246 macOS Core AI backend | New backend |
| #1264 SWA KV persistence | Documented limitation: export correctly declines rather than corrupting state |
| #1393 specialized model runtimes | Spec |
| #1528 bandwidth-aware context planning | Enhancement |
| #1609 OpenAI client nonce lifecycle | Incomplete feature |
| #1610 automatic GPU runtime discovery | Feature |
| #1630 secure-join approval UX | Feature |
| #1633 API keys for inference | Spec/feature |
| #1644 | Refactor: authoritative native loaded-stage resource descriptor |
| #1645 | Refactor: remove split family policy and post-load GGUF resource mirrors |
| #1646 | Refactor: typed activation planes and zero-copy framing |
| #1647 | KV instrumentation and matched promotion gates |
| #1648 | Packed append-only L3 KV storage |
| #1649 | Streamed local KV export/restore |
| #1650 | Benefit-per-byte KV admission/eviction |
| #1651 | Bounded host-RAM L2 KV tier |
| #1652 | Versioned KV segments |
| #1653 | Selective non-prefix KV reuse |
| #1654 | Backend-native KV compression |
| #1732 | KV roadmap rollup |
| #1864, #2249, #2251 | Certification work |
| #1952 | Collapse three copies of a path list: refactor |
| #1991 | Fixture tracking |
| #2009 | New model class |
| #2020 | System One image support |
| #2061 | New request-constraint header |
| #2136 | SDK feature layering |
| #2159 | New decision-inference mode |
| #2166 | New model family |
| #2194, #2198 | New measurements/fields |
| #2230 | Proposal: sign release manifests |
| #2232 | Protocol-neutral refactor |
| #2264 | Plugin twin-pair spec |
| #2295 | Console live-mesh Q&A |
| #2301 | Gossip signature phase 2 |
| #2309 | Multi-mesh serving |
| #2320 | Electricity cost per token |
| #2334, #2335, #2336 | Payment observability/feature work |
| #2350, #2353, #2354 | Trust-policy and ownership-cert features |
| #2486 | Package-managed start without catalog fetch |
| #2499 | Dependency hygiene |
| #2500 | Endpoint model filtering |
| #2576 | Energy measurement |

### Research, experiments, and tracking

| Issue | Why excluded |
|---|---|
| #656 pi harness MoA matrix | Test task, not a defect |
| #777 peer-baton MoA | Research |
| #803 skipped branch prediction | Research |
| #1233 proof of inference | Research |
| #1236 large MoE across a mesh | Research |
| #1320 Qwen3.8 flagship status | Status tracking |
| #1322 lab-node readiness contract | Spec |
| #1332 capsule receipts | Experiment |
| #1346 HEIR/FHE | Experiment |
| #1461 compose-mtp CI coverage | Test-coverage task |
| #1724, #1729, #1730 | Release-train design and staging |
| #1923 idle LoRA fine-tuning | Exploration |
| #1965, #1966 | Specs |
| #2010 next split architectures | Research |
| #2196, #2197, #2201, #2202, #2222 | Profiling, validation, or evaluation tasks |
| #2503 WASM peer validation | Experiment |
| #2575 local MoA learning | Experiment |

### Borderline, included above

- **#2003** is framed as a question but describes a real cold-cache defect on `main`; ranked at 12.
- **#2199** states its root cause is unisolated; it is still a wrong/incomplete output, ranked at 45.
- **#2387** opens with `## Feature`, but items 1, 2, 4, and 5 are defects; ranked at 38.
- **#1951** is fail-open by construction rather than an observed failure; ranked at 27.
- **#2200** is excluded: the report itself says it does not establish a MeshLLM software defect
  (host NVMe completion-polled timeouts).

## Progress

Statuses and links were read live from GitHub on 2026-10-26 against `fix/megabugs` at `973ba7faf0`.
`IN-REVIEW` means an open PR exists, `IN-PROGRESS` means a pushed branch with no PR yet.

| Rank | Issue | Status | Branch | PR | Evidence |
|---:|---|---|---|---|---|
| 1 | [#2119](https://github.com/Mesh-LLM/mesh-llm/issues/2119) | NOT-REPRODUCIBLE | — | — | [confirmation](https://github.com/Mesh-LLM/mesh-llm/issues/2119#issuecomment-6402598950) |
| 2 | [#1955](https://github.com/Mesh-LLM/mesh-llm/issues/1955) | IN-REVIEW | `fix/2-1955-catalog-filters` | [#2580](https://github.com/Mesh-LLM/mesh-llm/pull/2580) | — |
| 3 | [#2339](https://github.com/Mesh-LLM/mesh-llm/issues/2339) | IN-REVIEW | `fix/3-2339-mcp-cache-metadata` | [#2583](https://github.com/Mesh-LLM/mesh-llm/pull/2583) | — |
| 4 | [#2210](https://github.com/Mesh-LLM/mesh-llm/issues/2210) | IN-REVIEW | `fix/4-2210-external-endpoint-credentials` | [#2584](https://github.com/Mesh-LLM/mesh-llm/pull/2584) | — |
| 5 | [#2214](https://github.com/Mesh-LLM/mesh-llm/issues/2214) | IN-REVIEW | `fix/5-2214-plugin-sender-identity` | [#2585](https://github.com/Mesh-LLM/mesh-llm/pull/2585) | — |
| 6 | [#2212](https://github.com/Mesh-LLM/mesh-llm/issues/2212) | FIXED | `fix/6-2212-kv-cache-csrf` | [#2587](https://github.com/Mesh-LLM/mesh-llm/pull/2587) | — |
| 7 | [#2344](https://github.com/Mesh-LLM/mesh-llm/issues/2344) | IN-REVIEW | `fix/7-2344-console-chat-model-errors` | [#2623](https://github.com/Mesh-LLM/mesh-llm/pull/2623) | — |
| 8 | [#2135](https://github.com/Mesh-LLM/mesh-llm/issues/2135) | IN-REVIEW | `fix/8-2135-peer-count-measurement` | [#2589](https://github.com/Mesh-LLM/mesh-llm/pull/2589) | — |
| 9 | [#1954](https://github.com/Mesh-LLM/mesh-llm/issues/1954) | IN-REVIEW | `fix/9-1954-enc-dec-lane-clamp` | [#2590](https://github.com/Mesh-LLM/mesh-llm/pull/2590) | [confirmation](https://github.com/Mesh-LLM/mesh-llm/issues/1954#issuecomment-6406117780) |
| 10 | [#2029](https://github.com/Mesh-LLM/mesh-llm/issues/2029) | IN-REVIEW | `fix/10-2029-unresolved-layer-diagnostics` | [#2591](https://github.com/Mesh-LLM/mesh-llm/pull/2591) | [triage](https://github.com/Mesh-LLM/mesh-llm/issues/2029#issuecomment-6406444518) |
| 11 | [#2385](https://github.com/Mesh-LLM/mesh-llm/issues/2385) | FIXED | `fix/11-2385-models-show-refs` | [#2592](https://github.com/Mesh-LLM/mesh-llm/pull/2592) | — |
| 12 | [#2003](https://github.com/Mesh-LLM/mesh-llm/issues/2003) | IN-PROGRESS | `fix/12-2003-platform-checks-cache-mode` | — | — |
| 13 | [#2384](https://github.com/Mesh-LLM/mesh-llm/issues/2384) | FIXED | `fix/13-2384-laya-gpu-capacity` | [#2593](https://github.com/Mesh-LLM/mesh-llm/pull/2593) | — |
| 14 | [#2195](https://github.com/Mesh-LLM/mesh-llm/issues/2195) | IN-REVIEW | `fix/14-2195-long-prefill-timeout-budget` | [#2594](https://github.com/Mesh-LLM/mesh-llm/pull/2594) | — |
| 15 | [#2563](https://github.com/Mesh-LLM/mesh-llm/issues/2563) | BLOCKED | `fix/15-2563-llama-canary-infra-blocked` | — | [triage](https://github.com/Mesh-LLM/mesh-llm/issues/2563#issuecomment-6408480561) |
| 16 | [#2205](https://github.com/Mesh-LLM/mesh-llm/issues/2205) | IN-REVIEW | `fix/16-2205-native-runtime-id-validation` | [#2596](https://github.com/Mesh-LLM/mesh-llm/pull/2596) | [reproduction and fix evidence](https://github.com/Mesh-LLM/mesh-llm/issues/2205#issuecomment-6408703644) |
| 17 | [#2246](https://github.com/Mesh-LLM/mesh-llm/issues/2246) | IN-REVIEW | `fix/17-2246-payment-admission-public-model-id` | [#2598](https://github.com/Mesh-LLM/mesh-llm/pull/2598) | [reproduction and fix evidence](https://github.com/Mesh-LLM/mesh-llm/issues/2246#issuecomment-6408776729) |

#2246 reproduced on the unmodified base: `cargo test -p mesh-llm-host-runtime --lib
priced_public_alias_still_requires_payment_after_ingress_alias_rewrite` exits 101. The test composes the real
`rewrite_public_model_alias` in ingress order with the real `priced_model_requires_payment`, and fails on the
payment assertion, not its precondition: ingress rewrites the advertised ID
`tiiuae/Falcon-H1-1.5B-Instruct-GGUF@0d3a6cfe…:Q4_K_M` to the internal served name
`Falcon-H1-1.5B-Instruct-Q4_K_M`, and the exact-key offer lookup then misses a price the operator recorded under
the advertised ID, so a remote caller is served without payment. Admission now also checks the public identity of
the same served model, derived with the `public_model_id` mapping the model listing already publishes, and keeps
the exact-key path for a model with no distinct public alias. `cargo xtool repository affected-crates` limits CI to
`mesh-llm-host-runtime` and `mesh-llm`; the crate suite, `cargo check --all-targets`, Clippy with `-D warnings`,
`just no-console-print`, and `cargo fmt --check` pass locally.

Next eligible rank: 18, [#2059](https://github.com/Mesh-LLM/mesh-llm/issues/2059).

## Regenerating this list

```sh
gh issue list --repo Mesh-LLM/mesh-llm --state open --limit 300 \
  --json number,title,labels,createdAt,body \
  --template '{{range .}}=== #{{.number}} | {{.title}} | labels: {{range .labels}}{{.name}},{{end}}
{{.body}}
{{end}}'
```

Re-read each body before trusting a rank: several entries cite commit SHAs, line numbers, and
release tags that will have moved.

## Progress

Status is `TODO`, `IN-PROGRESS`, `CONFIRMED`, `FIXED`, `NOT-REPRODUCIBLE`, or `BLOCKED`;
`FIXED` requires a merged PR. One row per ranked issue; add the row the first time an issue is
picked up. Branch is the worker branch, PR the open or merged pull request, and Evidence the
confirmation or disposition comment.

| Rank | Issue | Status | Branch | PR | Evidence |
|---:|---|---|---|---|---|
| 1 | [#2119](https://github.com/Mesh-LLM/mesh-llm/issues/2119) | NOT-REPRODUCIBLE | `fix/1-2119-value-taking-flags` | — | [confirmation](https://github.com/Mesh-LLM/mesh-llm/issues/2119#issuecomment-6402598950) |
| 2 | [#1955](https://github.com/Mesh-LLM/mesh-llm/issues/1955) | FIXED | `fix/2-1955-catalog-filters` | [#2580](https://github.com/Mesh-LLM/mesh-llm/pull/2580) | [confirmation](https://github.com/Mesh-LLM/mesh-llm/issues/1955#issuecomment-6403463791) |
| 3 | [#2339](https://github.com/Mesh-LLM/mesh-llm/issues/2339) | FIXED | `fix/3-2339-mcp-cache-metadata` | [#2583](https://github.com/Mesh-LLM/mesh-llm/pull/2583) | [confirmation](https://github.com/Mesh-LLM/mesh-llm/issues/2339#issuecomment-6404315687) |
| 4 | [#2210](https://github.com/Mesh-LLM/mesh-llm/issues/2210) | FIXED | `fix/4-2210-external-endpoint-credentials` | [#2584](https://github.com/Mesh-LLM/mesh-llm/pull/2584) | [confirmation](https://github.com/Mesh-LLM/mesh-llm/issues/2210#issuecomment-6404684611) |
| 5 | [#2214](https://github.com/Mesh-LLM/mesh-llm/issues/2214) | FIXED | `fix/5-2214-plugin-sender-identity` | [#2585](https://github.com/Mesh-LLM/mesh-llm/pull/2585) | [confirmation](https://github.com/Mesh-LLM/mesh-llm/issues/2214#issuecomment-6404991047) |
| 6 | [#2212](https://github.com/Mesh-LLM/mesh-llm/issues/2212) | FIXED | `fix/6-2212-kv-cache-csrf` | [#2587](https://github.com/Mesh-LLM/mesh-llm/pull/2587) | [confirmation](https://github.com/Mesh-LLM/mesh-llm/issues/2212#issuecomment-6405182670) |
| 7 | [#2344](https://github.com/Mesh-LLM/mesh-llm/issues/2344) | CONFIRMED | `fix/7-2344-console-chat-model-errors` | [#2623](https://github.com/Mesh-LLM/mesh-llm/pull/2623) | [confirmation](https://github.com/Mesh-LLM/mesh-llm/issues/2344#issuecomment-6405487960) |
| 8 | [#2135](https://github.com/Mesh-LLM/mesh-llm/issues/2135) | CONFIRMED | `fix/8-2135-peer-count-measurement` | [#2589](https://github.com/Mesh-LLM/mesh-llm/pull/2589) | [confirmation](https://github.com/Mesh-LLM/mesh-llm/issues/2135#issuecomment-6405671011) |
| 9 | [#1954](https://github.com/Mesh-LLM/mesh-llm/issues/1954) | FIXED | `fix/9-1954-enc-dec-lane-clamp` | [#2590](https://github.com/Mesh-LLM/mesh-llm/pull/2590) | [confirmation](https://github.com/Mesh-LLM/mesh-llm/issues/1954#issuecomment-6406117780) |
| 10 | [#2029](https://github.com/Mesh-LLM/mesh-llm/issues/2029) | FIXED | `fix/10-2029-unresolved-layer-diagnostics` | [#2591](https://github.com/Mesh-LLM/mesh-llm/pull/2591) | [confirmation](https://github.com/Mesh-LLM/mesh-llm/issues/2029#issuecomment-6406262501) |
| 11 | [#2385](https://github.com/Mesh-LLM/mesh-llm/issues/2385) | FIXED | `fix/11-2385-models-show-refs` | [#2592](https://github.com/Mesh-LLM/mesh-llm/pull/2592) | [confirmation](https://github.com/Mesh-LLM/mesh-llm/issues/2385#issuecomment-6406651065) |
| 12 | [#2003](https://github.com/Mesh-LLM/mesh-llm/issues/2003) | NOT-REPRODUCIBLE | `fix/12-2003-platform-checks-cache-mode` | — | [disposition](https://github.com/Mesh-LLM/mesh-llm/issues/2003#issuecomment-6406982137) |
| 13 | [#2384](https://github.com/Mesh-LLM/mesh-llm/issues/2384) | FIXED | `fix/13-2384-laya-gpu-capacity` | [#2593](https://github.com/Mesh-LLM/mesh-llm/pull/2593) | [confirmation](https://github.com/Mesh-LLM/mesh-llm/issues/2384#issuecomment-6407177264) |
| 14 | [#2195](https://github.com/Mesh-LLM/mesh-llm/issues/2195) | FIXED | `fix/14-2195-long-prefill-timeout-budget` | [#2594](https://github.com/Mesh-LLM/mesh-llm/pull/2594) | [confirmation](https://github.com/Mesh-LLM/mesh-llm/issues/2195#issuecomment-6407979111) |
| 15 | [#2563](https://github.com/Mesh-LLM/mesh-llm/issues/2563) | BLOCKED | `fix/15-2563-llama-canary-infra-blocked` | — | [triage](https://github.com/Mesh-LLM/mesh-llm/issues/2563#issuecomment-6408480561) |
| 16 | [#2205](https://github.com/Mesh-LLM/mesh-llm/issues/2205) | CONFIRMED | `fix/16-2205-native-runtime-id-validation` | [#2596](https://github.com/Mesh-LLM/mesh-llm/pull/2596) | [confirmation](https://github.com/Mesh-LLM/mesh-llm/issues/2205#issuecomment-6408703644) |
| 17 | [#2246](https://github.com/Mesh-LLM/mesh-llm/issues/2246) | CONFIRMED | `fix/17-2246-payment-admission-public-model-id` | [#2598](https://github.com/Mesh-LLM/mesh-llm/pull/2598) | [confirmation](https://github.com/Mesh-LLM/mesh-llm/issues/2246#issuecomment-6408776729) |
| 18 | [#2059](https://github.com/Mesh-LLM/mesh-llm/issues/2059) | CONFIRMED | `fix/18-2059-moa-paid-peer-402-failover` | [#2599](https://github.com/Mesh-LLM/mesh-llm/pull/2599) | [confirmation](https://github.com/Mesh-LLM/mesh-llm/issues/2059#issuecomment-6409202060) |
| 19 | [#2216](https://github.com/Mesh-LLM/mesh-llm/issues/2216) | FIXED | `fix/19-2216-mcp-session-isolation` | [#2600](https://github.com/Mesh-LLM/mesh-llm/pull/2600) | [confirmation](https://github.com/Mesh-LLM/mesh-llm/issues/2216#issuecomment-6409874506) |
| 20 | [#2213](https://github.com/Mesh-LLM/mesh-llm/issues/2213) | CONFIRMED | `fix/20-2213-direct-peer-gossip-provenance` | [#2602](https://github.com/Mesh-LLM/mesh-llm/pull/2602) | [confirmation](https://github.com/Mesh-LLM/mesh-llm/issues/2213#issuecomment-6412037644) |
| 21 | [#2211](https://github.com/Mesh-LLM/mesh-llm/issues/2211) | CONFIRMED | `fix/21-2211-catalog-pin-plugin-install` | [#2603](https://github.com/Mesh-LLM/mesh-llm/pull/2603) | [confirmation](https://github.com/Mesh-LLM/mesh-llm/issues/2211#issuecomment-6412946859) |
| 22 | [#2340](https://github.com/Mesh-LLM/mesh-llm/issues/2340) | CONFIRMED | `fix/22-2340-gguf-path-served-checkpoint` | [#2606](https://github.com/Mesh-LLM/mesh-llm/pull/2606) | [confirmation](https://github.com/Mesh-LLM/mesh-llm/issues/2340#issuecomment-6413856693) |
| 23 | [#2343](https://github.com/Mesh-LLM/mesh-llm/issues/2343) | FIXED | `fix/23-2343-qwen-special-token-types` | [#2611](https://github.com/Mesh-LLM/mesh-llm/pull/2611) | [confirmation](https://github.com/Mesh-LLM/mesh-llm/issues/2343#issuecomment-6422044400) |
| 24 | [#2338](https://github.com/Mesh-LLM/mesh-llm/issues/2338) | FIXED | `fix/24-2338-inline-system-fold` | [#2612](https://github.com/Mesh-LLM/mesh-llm/pull/2612) | [confirmation](https://github.com/Mesh-LLM/mesh-llm/issues/2338#issuecomment-6422277779) |
| 25 | [#2342](https://github.com/Mesh-LLM/mesh-llm/issues/2342) | FIXED | `fix/24b-2342-config-unknown-fields` | [#2615](https://github.com/Mesh-LLM/mesh-llm/pull/2615) | [confirmation](https://github.com/Mesh-LLM/mesh-llm/issues/2342#issuecomment-6422493389) |
| 26 | [#2348](https://github.com/Mesh-LLM/mesh-llm/issues/2348) | FIXED | `fix/25-2348-named-tool-choice` | [#2616](https://github.com/Mesh-LLM/mesh-llm/pull/2616) | [confirmation](https://github.com/Mesh-LLM/mesh-llm/issues/2348#issuecomment-6424434335) |
| 27 | [#1951](https://github.com/Mesh-LLM/mesh-llm/issues/1951) | FIXED | `fix/26-1951-workload-route-coverage` | [#2618](https://github.com/Mesh-LLM/mesh-llm/pull/2618) | [confirmation](https://github.com/Mesh-LLM/mesh-llm/issues/1951#issuecomment-6424609084) |
| 28 | [#2352](https://github.com/Mesh-LLM/mesh-llm/issues/2352) | FIXED | `fix/27-2352-papercuts` | [#2619](https://github.com/Mesh-LLM/mesh-llm/pull/2619) | [confirmation](https://github.com/Mesh-LLM/mesh-llm/issues/2352#issuecomment-6425300831) |
| 29 | [#2508](https://github.com/Mesh-LLM/mesh-llm/issues/2508) | NOT-REPRODUCIBLE | — | — | [disposition](https://github.com/Mesh-LLM/mesh-llm/issues/2508#issuecomment-6427982348) |
| 30 | [#2217](https://github.com/Mesh-LLM/mesh-llm/issues/2217) | CONFIRMED | `fix/30-2217-stage-control-coordinator-auth` | [#2621](https://github.com/Mesh-LLM/mesh-llm/pull/2621) | [confirmation](https://github.com/Mesh-LLM/mesh-llm/issues/2217#issuecomment-6428331830) |
| 31 | [#2218](https://github.com/Mesh-LLM/mesh-llm/issues/2218) | FIXED | `fix/31-2218-stage-transport-budgets` | [#2622](https://github.com/Mesh-LLM/mesh-llm/pull/2622) | [confirmation](https://github.com/Mesh-LLM/mesh-llm/issues/2218#issuecomment-6428881530) |
| 32 | [#2206](https://github.com/Mesh-LLM/mesh-llm/issues/2206) | CONFIRMED | `fix/32-2206-inference-connection-bounds` | [#2624](https://github.com/Mesh-LLM/mesh-llm/pull/2624) | [confirmation](https://github.com/Mesh-LLM/mesh-llm/issues/2206#issuecomment-6429715273) |
| 33 | [#2207](https://github.com/Mesh-LLM/mesh-llm/issues/2207) | TODO | — | — | — |
| 34 | [#2208](https://github.com/Mesh-LLM/mesh-llm/issues/2208) | TODO | — | — | — |
| 35 | [#2193](https://github.com/Mesh-LLM/mesh-llm/issues/2193) | TODO | — | — | — |
| 36 | [#2192](https://github.com/Mesh-LLM/mesh-llm/issues/2192) | TODO | — | — | — |
| 37 | [#2383](https://github.com/Mesh-LLM/mesh-llm/issues/2383) | TODO | — | — | — |
| 38 | [#2387](https://github.com/Mesh-LLM/mesh-llm/issues/2387) | TODO | — | — | — |
| 39 | [#2151](https://github.com/Mesh-LLM/mesh-llm/issues/2151) | TODO | — | — | — |
| 40 | [#1001](https://github.com/Mesh-LLM/mesh-llm/issues/1001) | TODO | — | — | — |
| 41 | [#2182](https://github.com/Mesh-LLM/mesh-llm/issues/2182) | TODO | — | — | — |
| 42 | [#2147](https://github.com/Mesh-LLM/mesh-llm/issues/2147) | TODO | — | — | — |
| 43 | [#2055](https://github.com/Mesh-LLM/mesh-llm/issues/2055) | TODO | — | — | — |
| 44 | [#2146](https://github.com/Mesh-LLM/mesh-llm/issues/2146) | TODO | — | — | — |
| 45 | [#2199](https://github.com/Mesh-LLM/mesh-llm/issues/2199) | TODO | — | — | — |
| 46 | [#2209](https://github.com/Mesh-LLM/mesh-llm/issues/2209) | TODO | — | — | — |
| 47 | [#2349](https://github.com/Mesh-LLM/mesh-llm/issues/2349) | TODO | — | — | — |
| 48 | [#2228](https://github.com/Mesh-LLM/mesh-llm/issues/2228) | TODO | — | — | — |
| 49 | [#2247](https://github.com/Mesh-LLM/mesh-llm/issues/2247) | TODO | — | — | — |
| 50 | [#2158](https://github.com/Mesh-LLM/mesh-llm/issues/2158) | TODO | — | — | — |
| 51 | [#2191](https://github.com/Mesh-LLM/mesh-llm/issues/2191) | TODO | — | — | — |
| 52 | [#2190](https://github.com/Mesh-LLM/mesh-llm/issues/2190) | TODO | — | — | — |
| 53 | [#2318](https://github.com/Mesh-LLM/mesh-llm/issues/2318) | TODO | — | — | — |
