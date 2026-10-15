<!-- SPDX-License-Identifier: Apache-2.0 -->
# Split-inference stage records — who ran which slice, and did the hand-offs line up

> **Design only.** Nothing in this document changes code. It fixes the record shape, the
> coordinator's citation of stage records, the delivery path and the console surface for
> split ("Skippy") inference, so the build targets a reviewed shape. Seal-path code waits
> for review of this document.
>
> **Read against:**
> - upstream `Mesh-LLM/mesh-llm` `main` @ `6e75272ff7e1bcd29b9271a3067eb5ce9f3c4f3b` for
>   every host citation (paths relative to `crates/`);
> - the capsule-emit-mesh repository, where this plugin was developed, @
>   `55a3656b6b7eb9a296cc4f3e3110d894d2b35a7e`, and its Rust producer @
>   `08c63607d62cef434e6f1b33144a17177b94755b` (now the `capsule-emit` crate and
>   `mesh/crates/mesh-llm-capsules-plugin/` here).

## 0. Summary

Today a split run is one exchange between the requester and the coordinator. The nodes that
ran the other slices leave no participation record, so nobody can say who ran which layers,
or whether what one stage handed on is what the next stage took in.

**Model: nested exchanges.** The requester ⟷ coordinator exchange stays as it is. Each
coordinator ⟷ stage pair becomes an exchange of its own.
- The **stage node** seals one *stage record* per request. It carries the run and request
  ids, its stage index, its layer range, the package id, and a running digest of every frame
  it read and wrote in each direction on each of its hops. It also carries its token counts.
- The **coordinator** seals its own side of each stage exchange, then cites those records
  from its record of the requester's exchange.

**What this establishes, stated narrowly:**
- **Per-hop agreement.** For each hop, both ends commit to what crossed it, in both
  directions. When both ends' records reach a verifier and they differ, the disagreement is
  visible without disclosing any activation.
  - This catches **one side misreporting, or a hand-off altered in transit.**
  - It does **not** catch two neighbouring stages that collude and report matching digests.
  - It does **not** catch a stage that alters its own output and reports that altered output
    consistently.
  - A coordinator that withholds one record turns a disagreement into a gap. §7.2 shows that
    as a gap. It is never shown as agreement.
- **Per-stage attribution.** Each slice is tied to the key that signed for it. This says
  nothing about whether the stages are independent of each other or of the coordinator: a
  coordinator can place every stage on keys it controls.

**What this does not establish:** that any stage computed its slice correctly (§9).

## 1. What exists, and what is missing

| Piece | State | Where |
|---|---|---|
| Coordinator receipt binding stage order to per-stage record digests: three states `present / absent / not_requested`, and two arrays (`topology[]` = "I routed this", `stages[]` = "I hold proof of this") | Built, **Python only**, not on the live path | `mesh_coordinator_receipt_emitter.py`, `mesh_coordinator_bundle_flow.py`, `docs/TRUST-MODEL.md` §4.4 |
| Per-hop lifecycle record carrying `hop_id` / `exchange_id` / `terminal_state` | Built, Python only | `mesh_record_emitter.py` (`x-mesh-lifecycle-v1`) |
| Live exchange records: `role` ∈ `requested / served / conflict / unknown`; the other side cited with `citation_purpose: "counterparty_half"` | Live (Rust) | `mesh/crates/mesh-llm-capsules-plugin/src/producer/capsule.rs` (shape and constant), `mesh/crates/mesh-llm-capsules-plugin/src/capsule_emit.rs` (sealing) |
| Record push to a counterparty (`record-push/1` mesh channel) | Live | `mesh/crates/mesh-llm-capsules-plugin/src/record_push_bridge.rs` |
| Bundle form of that push (record + inclusion proof + checkpoint); coordinator bundle ask | **In progress, not on the reviewed refs** | separate branches |
| **Anything per stage, per request, from the host** | **Missing** | see §2 |

The Python coordinator receipt answered "in what order, and do I hold each stage's record".
It never defined what a stage record says about a split, and it never checked that
neighbouring stages agree with each other. This document adds both, and moves the design
onto the live Rust producer.

## 2. Host facts this design rests on

All citations are to upstream `main` @ `6e75272ff7e1bcd29b9271a3067eb5ce9f3c4f3b`.

1. **The coordinator is stage 0.** `local_split.rs` rejects a topology whose stage 0 is not
   the canonical coordinator (`mesh-llm-host-runtime/src/runtime/local_split.rs`:
   `ensure!(stage0.node_id == canonical_coordinator, …)`). A re-plan that would move stage 0
   is skipped. So an N-stage split has the coordinator running one slice itself, plus N−1
   other stage nodes.
2. **`run_id` identifies a topology generation, not a request.**
   - The first run is `format!("mesh-split-{}", now_unix_nanos())`.
   - A re-plan is `format!("mesh-split-{}-g{}", now_unix_nanos(), generation)`
     (`local_split/coordinator.rs`).
   - The coordinator mints `run_id`, and nobody checks it.
   - Per request, every `StageWireMessage` carries `request_id` and `session_id` (u64)
     (`skippy-protocol/src/binary/types.rs`).
3. **`request_id` is per request; `session_id` is not.**
   - `request_id` is the first 8 bytes of a SHA-256 over the session label, a per-process
     random nonce and a per-process counter (`skippy-server/src/frontend/generation/cache_hints.rs`
     `request_id`; `frontend/util.rs` `stable_wire_id`).
   - `session_id` stays stable across a trusted agent session's requests.
4. **Hops form a chain, and replies take one of two paths.**
   - Forward frames go `stage0 → stage1 → … → final`.
   - Replies go back hop by hop (`Ack`, `PredictedToken`, `PredictedTokens`), **unless** the
     final stage returns predicted tokens directly to stage 0
     (`skippy-server/src/binary_transport/direct_return.rs`).
   - The direct lane is a preference: "a rejected preferred sink uses the existing reverse
     fallback" (same file).
5. **Frames cross a hop byte-for-byte on the wire, but not in the parsed message.**
   - On write, the codec writes the header, the ids, the tokens, the positions and the
     encoded activation verbatim (`skippy-protocol/src/binary/codec.rs`,
     `writer.write_all(&message.activation)`).
   - The mesh tunnel is a plain byte relay (`mesh-llm-host-runtime/src/network/tunnel.rs`
     `relay_bidirectional`).
   - **On read, the codec decodes the activation and re-encodes it as raw F32**
     (`codec.rs`: `decode_activation_frame(…)` then `encode_raw_activation_frame(&frame)`).
     So the parsed `StageWireMessage` a stage holds is **not** the bytes that crossed the
     hop.
   - The codec is chosen per hop (`skippy-server/src/binary_transport/forwarding.rs`
     `select_output_activation_codec`) from `RAW_F32_V1`, `F16_RNE_V1`, `BF16_RNE_V1` and
     `S8_ROW_F32_RNE_V1` (`skippy-protocol/proto/stage.proto`).
   - Each stage rewrites `source_stage_index` in the state header before forwarding
     (`forwarding.rs`).
   - **Consequence:** the fold must hash wire bytes as written, and wire bytes as read
     *before* the codec parses them. It cannot be rebuilt from the parsed message, or from
     tensors.
6. **The existing byte counters cannot be used for this.** In the per-request summary
   (`skippy-server/src/binary_transport/binary_messaging/connection.rs`):
   - `input_activation_bytes = message.activation.len()` is the size of the **raw-F32
     re-encoded** frame, descriptor included (fact 5).
   - `output_activation_bytes: output.payload.len()` is the raw payload **without** the
     descriptor.
   - The two neighbours' numbers therefore differ under every codec, raw F32 included. A new
     host measurement is needed (§8).
7. **Split serving starts with no generation hooks.** The split runtime is built with
   `serving_hooks_factory: None` (`local_split.rs`). The exact generation receipts are
   documented as "restricted to local single-stage execution"
   (`skippy-server/src/serving_hooks.rs`). No per-stage, per-request hook exists.
8. **Package identity is package-wide, not per slice.**
   - `computed_package_id()` is `sha256:` over the normalized package manifest
     (`skippy-package-format/src/lib.rs`). Stages reject a mismatch at admission, where
     `package_id` rides inside `LoadStage`'s admission descriptor.
   - `LoadStage` (`stage.proto`) also carries `manifest_sha256`, `source_model_sha256`,
     `layer_start` / `layer_end`, `topology_hash`, `participant_set_hash`, an optional
     coordinator id, `coordinator_term`, and `topology_stages` (the stage count is derived
     from that list).
   - **`layer_end` is exclusive.** Upstream counts layers as `layer_end - layer_start`
     (`skippy-server/src/binary_transport/stage_execution.rs`).
   - No per-slice weights digest exists.
9. **A stage lost mid-request fails that request.** Recovery re-plans at the generation
   level, with a new `run_id`, through `SplitLossRecoveryDecision`
   (`local_split/recovery.rs`). We found nothing that re-assigns a stage within a request.
10. **Token counts are known per stage.**
    - The per-request summary accumulates `prefill_token_count` and `decode_token_count`
      (`binary_messaging/summary.rs`).
    - Prefill counts include tokens restored from the prefix cache, so they count tokens
      received, not tokens computed.
    - Generated-token totals are known only at stage 0.

## 3. Keys and roles

**Stage exchange key:** `(coordinator_node_id, run_id, request_id, stage_index)`.
- `coordinator_node_id` is included because `run_id` is minted by the coordinator and
  checked by nobody (§2.2). A stage that serves splits for two coordinators must never let
  one coordinator's `run_id` alias the other's.
- `run_id` scopes the generation, so a re-plan never aliases an older run.
- `request_id` scopes the request.
- `stage_index` scopes the slice.
- **Two distinct records under one key** (for example a faulty stage 0 reusing a
  `request_id` within a run) are a `conflict`. Both are shown. Neither counts as `present`,
  and no hop involving that stage reads as agreeing.

**Split key:** `(coordinator_node_id, run_id, request_id)`. Every record for one split
request carries it, and the UI nests rows on it (§7).

**Join to the requester's exchange.** Only the coordinator's host knows both the
`openai.exchange.v1` `exchange_id` and the split key of the same request. The host seam (§8)
must emit both in one stage-0 event. A join on timing is never accepted as a key.

**Role words** (one per row, relative to the viewer):

| Viewer | Row | Role word |
|---|---|---|
| requester | requester ⟷ coordinator | `you asked` |
| coordinator | requester ⟷ coordinator (parent) | `you coordinated` |
| coordinator | coordinator ⟷ stage k (child, one per remote stage) | `you delegated` |
| stage node | coordinator ⟷ stage k | `you ran a stage` |
| any | an ordinary served exchange | `you served` |

## 4. The stage record

### 4.1 Shape

The stage record stays inside the existing exchange-record kind. The live producer's
`x-mesh-poc-v1.role` is a string with four values: `requested`, `served`, `conflict` and
`unknown` (`capsule.rs`, `MeshPocV1.role`). This design adds one value and one sibling block.

- **`x-mesh-poc-v1.role = "stage"`** marks a record of a coordinator ⟷ stage exchange, on
  either side of it.
  - A reader that does not know `stage` must treat it as `unknown`, never as `served`.
  - The four existing values keep their meaning.
- **`x-mesh-stage-v1`** is a new block in `compute_attestation`. It appears on:
  - `stage` records, with `side` = `stage` or `coordinator`;
  - the coordinator's `served` main record, with `side` = `coordinator` and
    `stage_index: 0`, describing the coordinator's own slice.

The stage node's own block:

```json
{
  "v": 1,
  "side": "stage",
  "coordinator_node_id": "…",
  "run_id": "mesh-split-…",
  "request_id": "18446744073709551615",
  "topology_hash": "…",
  "stage_index": 1,
  "stage_count": 3,
  "layer_start": 16,
  "layer_end": 32,
  "package_id": "sha256:…",
  "manifest_sha256": "…",
  "source_model_sha256": "…",
  "return_mode": "direct",
  "upstream":   { "received": { "frames": 412, "digest": "sha256:…" },
                  "replies_sent": { "frames": 380, "digest": "sha256:…" } },
  "downstream": { "sent": { "frames": 412, "digest": "sha256:…" },
                  "replies_received": { "frames": 380, "digest": "sha256:…" } },
  "direct_return": null,
  "tokens": { "prefill_received": 380, "decode": 32 },
  "terminal_state": "completed"
}
```

- **Layer range.** `layer_start` / `layer_end` are upstream's values: `layer_end` is
  exclusive. The console shows the inclusive range (`16–31`) and derives it from these.
- **`request_id`** is a decimal string. A u64 does not survive as a JSON number in every
  reader.
- **Hop fields.**
  - Stage 0 has no `upstream`.
  - The final stage has no `downstream`.
  - `direct_return` is `{sent}` on the final stage and `{received}` on stage 0 when
    `return_mode` is `direct`, and `null` otherwise.
- **`return_mode`** is `direct` or `relayed` (the reverse fallback). The mode decides which
  fields must be non-empty for a completed request (§7.2).
- **`terminal_state`** reuses the existing closed set (`mesh_record_verifier.py`
  `TERMINAL_STATES`). A stage whose request failed still seals what it saw up to the
  failure.
- **Tokens.** `tokens.prefill_received` is named for what it counts (§2.10).

### 4.2 The hop digest

For each hop and each lane (`forward`, `reply`, `direct_return`):

```
h_0 = SHA-256( "skippy-stage-frames/v1" ‖ salt ‖ coordinator_node_id ‖ run_id
               ‖ request_id ‖ hop_index ‖ lane )
h_i = SHA-256( h_{i-1} ‖ SHA-256(frame_i) )      for each frame of this request on this lane, in wire order
digest = h_n,   frames = n
```

- **`frame_i`** is the exact byte string of one message of this request as it crossed the
  hop: the bytes the codec wrote on the sender, and the bytes read off the stream on the
  receiver, **before** any parse or re-encode (§2.5).
  - Tokens, positions, ids and the header (including the rewritten `source_stage_index`)
    sit inside the frame. So a reordered sequence, a truncation of one side against the
    other, or an altered byte changes the digest.
- **`hop_index`** is the index of the upstream stage of the hop. It binds each digest to its
  hop, so a record from one hop cannot be replayed as another hop's, even when a hop carried
  only control frames.
- **`lane`** keeps forward frames, hop-by-hop replies and the direct return lane apart.
- **Other requests.** Forward frames of other requests sharing the stream are skipped by
  `request_id`, and each request keeps its own order.
- **Replies carry no `request_id`.** `StageReply` has no id field
  (`skippy-protocol/src/binary/types.rs`). Reply frames are therefore attributed to a request
  only by lane ownership: a persistent lane serves one request at a time and is handed back
  for reuse (`binary_messaging/async_forwarder.rs`). The `reply` lane's fold rests on that
  host invariant, which is Q-H5.
- **Frame counts** are carried for display only. Equal digests already imply equal counts.
  Equal digests with unequal counts mark a **malformed** record, not a disagreement.
- **When to seal.** Lanes can be reused across requests, and a teardown `Stop` can be
  written through another handle on the same socket, so a middle stage may never see a clean
  end of request. A stage therefore seals on whichever comes first:
  - the request's completion as the stage observes it (its final reply, or a `Stop` naming
    the request);
  - an idle timeout since the request's last frame, with `terminal_state: "timed_out"`;
  - a transport loss on either hop, with `terminal_state: "transport_error"`.
- **Open host questions** (§8): whether one request's frames on one hop can span more than
  one ordered stream (Q-H1), and where the fold sits when a socket has more than one writer
  (Q-H2).

### 4.3 What the stage record does not carry

- **No activations, tokens or text.** Digests and counts only.
- **No `session_id`.** It is stable across a trusted agent session (§2.3). In a record that
  may be shown to others it would link one party's requests across time. The stage already
  sees it on the wire, so leaving it out takes nothing from the stage.
- **No end-requester identity.** Stages are not told who the end requester is. A stage's
  counterparty is the coordinator.
- **No per-slice weights digest,** because none exists (§2.8). `package_id` + layer range is
  what the stage *claims* it loaded, with the same standing as row C4 in the threat model (`THREAT-MODEL.md`)
  §2.4. This record makes the claim signed and per-request. It does not make it verified.

### 4.4 Salt, and who it protects

A digest over frames that contain token ids and activations can be tested against a guess.
Someone holding the public weights and a candidate prompt could recompute stage 0's
outgoing frames and compare. That needs bit-identical kernels, codec and hardware class, so
it is plausible but not demonstrated.

- **Comparing digests never needs the salt.** A verifier compares the two ends' digests of
  one hop.
- **Who holds the salt.** Stage 0 would mint 32 random bytes per request and carry them on
  the request's first frame. That frame is forwarded down the chain, so **every stage in the
  run holds the salt**, and anyone outside the run does not. Which frame is first (control
  kinds such as `ConfigureGeneration` can precede activations) is a host question (Q-H3).
- **Who it protects against.** In v0, stage records go only to the coordinator and the
  requester, and both already know the prompt. So the salt protects nobody yet. It starts to
  matter the moment stage records reach anyone else: a history surface, a referee, a third
  party.
- **Recommendation.** Ship v0 unsalted to coordinator + requester only, with the limitation
  stated in the record's caveat. Stage records stay off every other surface until the salt
  lands (part B of §8).

## 5. The coordinator's side: stage-exchange records and citation

For each remote stage k, the coordinator seals a **stage-exchange record** (`role: "stage"`,
`x-mesh-stage-v1.side: "coordinator"`). It carries:

- **What the coordinator assigned.** Layer range, `package_id`, `manifest_sha256`,
  `topology_hash` and `coordinator_term`, as sent in `LoadStage`. These are producer claims
  about its own act of assignment.
- **What it observed first-hand,** and only that:
  - for stage 1: the coordinator's own `downstream` fields, since it is stage 0;
  - for the final stage under `direct` return: its own `direct_return.received`;
  - for any other stage: nothing on the data path. The record states
    `data_path_observed: false` rather than leaving empty fields that could read as a pass.
- **A `counterparty_half` reference** to the stage's own record once it arrives (§6). This is
  exactly how the live producer cites a counterparty today (`CITATION_PURPOSE_COUNTERPARTY_HALF`,
  defined in `capsule.rs` and sealed in `capsule_emit.rs`).

The coordinator's **main record** of the requester's exchange (`role: "served"`, as today)
gains:
- `x-mesh-stage-v1` for its own slice (`stage_index: 0`);
- one `references[]` entry per stage-exchange record, with a new provisional citation
  purpose, `split_stage`;
- the `x-mesh-coordinator-receipt-v1` block (`run_id`, `topology[]`, `stages[]`), moved from
  Python onto the live producer. `stages[].bundle` keeps its three states:
  - `present` — the stage record arrived and is cited;
  - `absent` — it was asked for and did not arrive;
  - `not_requested` — it was never part of the run.

  The existing producer rules hold: `present` implies a reference, and `absent` /
  `not_requested` imply none.

**The model stays pairwise, as it is today.** Each side seals its own view, and the other
side's record is cited, never copied into your chain. The nesting lives in references, not
in a new composite kind.

## 6. Delivery

1. **Stage → coordinator.** When a stage seals its record, it pushes it to
   `coordinator_node_id` over `record-push/1`. It uses the bundle form (record + inclusion
   proof + checkpoint) once that form lands.
   - This is the same channel and the same receiver-side sealing as the existing push. Only
     the addressee is new.
   - The stage learns the coordinator from the topology it was loaded with. It never learns
     the end requester.
2. **Coordinator waits, with a deadline.** The coordinator seals its main record once it holds
   every remote stage's record, or once the deadline passes.
   - **The deadline is new.** No bundle-push deadline exists on the reviewed refs, so it is
     built here (Q-D5).
   - A late stage is `absent`. It is never dropped silently, and never `present` without a
     reference.
   - A record that arrives after the seal is cited by a follow-up record
     (`chain.relation: "follows"`). The sealed main record is never rewritten.
3. **Coordinator → requester.** The coordinator's push to the requester carries its main
   record plus each cited stage record. The requester can then run the hop check (§7.2)
   offline, without contacting any stage.

**No new transport is needed.** `record-push/1` exists. The bundle form and the coordinator's
bundle ask are in progress on their own branches. This design adds one addressee, one
payload and one deadline.

**Forwarding a stage's record to a requester the stage never met** is a term of taking part
in the split, not a per-request choice. It must be declared when the stage joins the run,
the same "consent is protocol-declared" rule the Python coordinator-receipt design uses
(Q-D8).

## 7. Console

### 7.1 One grouping pattern: twins, referee, mixture, splits

Twins and splits need the same grouping, so it is built once. Twins come first because they
exist today.

- **Rows stay two-sided.** A row is still one exchange: your side and their side.
- **Rows nest by a shared bracket key:**
  - `twin-bracket:<id>` for twins (`twin_bracket_id`, which exists today);
  - `split:<coordinator>/<run_id>/<request_id>` for splits.
- **Today's grouping never reorders.** `groupStreamRows` merges only *adjacent* rows with one
  `twin-bracket` key (`crates/mesh-llm-ui/src/features/capsules/lib/exchange-pages.ts`), and
  it deliberately never reorders the time-ordered stream. Nesting by key when rows are not
  adjacent changes that rule. The parent would sit at its own timestamp, with its children
  under it. This is a decision (Q-D7).
- **One role word per row** (§3).
- **Parent state.** A parent row's state is its own outward exchange plus one inward summary
  line (for example `2 of 3 stages recorded · hand-offs agree`). A child never changes the
  parent's own state cell.
- **Role filter chips:** `asked · served · coordinated · delegated · ran a stage`.
- **Direct and indirect Peers are relative to the viewer:**
  - to the requester, the coordinator is direct and the stages are indirect;
  - to the coordinator, every stage is direct;
  - to a stage, the coordinator is direct and the requester is invisible.

### 7.2 The requester's row and the stage strip

Row: `you asked · C · split across 3 nodes`. Expanded, it shows the stage strip:

```
C 0–15 (coordinator's own slice) · B 16–31 ✓ · D 32–47 ◌ not received
hand-offs: C→B agree ✓ · B→D gap (D's record not received) · request: completed
```

**Stage cells:**
- **Stage 0 (the coordinator's own slice) never gets a `✓`.** Its cell would only compare
  the coordinator's claim against the coordinator's own assignment.
- `✓` — the stage record is present, verifies, and its layer range and `package_id` match
  what the coordinator's receipt says it assigned.
- `✕ disagrees` — present but not matching the assignment.
- `◌ not received` — `absent`.
- `—` — `not_requested`.
- `⚠ conflict` — two records under one key (§3).

**Hand-offs line.** Checked per hop, per lane: forward, then replies, then the direct return
lane when `return_mode` is `direct`.
- `agree ✓` — both ends present, and every lane's digests are equal and non-empty for the
  lanes that the request's `terminal_state` and `return_mode` require.
- `gap` — at least one end is missing, **or** a required lane is empty on both ends. A lane
  empty on both sides has nothing to compare, and it is never read as agreement. A gap is
  not a failure verdict.
- `break` — both ends are present and differ.
- `malformed` — equal digests with unequal counts.

**Rules for the line as a whole:**
- The run reads `hand-offs agree ✓` only when every hop and every required lane agrees.
- **The request's `terminal_state` is always shown beside the hand-offs line.** Agreement is
  not completion: a request that stopped early can have both neighbours agree on a truncated
  stream.
- **The UI must not overstate `agree ✓` (§0).** Its tooltip says that two stage records agree
  with each other. It does not say that the stages are independent or that their computation
  was correct.

### 7.3 Peers

The Peers tab adds a third group, **"Nodes that served you through a split"**.
- It is separate from "Nodes you have dealt with" and never merged into it, the same way the
  two existing groups are never merged (`LedgerPeersTable.tsx`).
- A node you reached only through a coordinator is not a node you dealt with.

### 7.4 The stage node's tab

Row: `you ran a stage · stage 2 of 3 · for C`.
- The end requester is not shown, because the stage was never told who it is.
- Showing it would take a sharing-policy switch that the requester controls. That switch is
  out of scope here (Q-D4).

### 7.5 Copy rules the build must keep

- Every new chip gets a tooltip and a tooltip-census entry (`tooltip-census.test.tsx`).
- The census forbids `half` / `halves` outside the checks panel. Console copy says "their
  record of this stage", never "counter-half".
- The retired-phrase test stays green.
- The existing CLOSED rule and its copy are unchanged for the requester ⟷ coordinator row.
  The stage strip is added beneath that row. It never replaces the row's state.

## 8. The host seam (upstream ask)

**Nothing in §4–§6 can be built from the plugin alone:**
- split serving has no per-request hook (§2.7);
- the fold must sit below the codec's parse (§2.5);
- the existing counters are not comparable (§2.6);
- only the coordinator's host knows the `exchange_id` ↔ split key mapping (§3).

**The ask is a `skippy.stage.v1` mesh channel in the host's stage runtime.** It follows the
payment lifecycle channel's pattern:
- plugins declare it in their manifest;
- delivery is best-effort, with a bounded queue and a publication timeout;
- serving never waits on it;
- there are no per-token events;
- there is no hashing at all unless some plugin declares the channel.

**Part A** is one event per request per stage, emitted when that stage seals (§4.2). It
carries:
- the §4.1 fields;
- the per-lane folds;
- `return_mode`;
- `terminal_state`;
- at stage 0 only, the request's `openai.exchange.v1` `exchange_id`.

**Part B**, optional, is the per-request salt (§4.4).

The issue text is drafted separately, in the host's own vocabulary. It is held until an
earlier host ask has an answer. It asks the maintainers:

- **Q-H1** — Can one request's frames on one hop span more than one ordered stream (async
  forwarder, reconnect)? If yes, the fold needs a sort key from inside the frame
  (`pos_start` / `decode_step`) instead of wire order.
- **Q-H2** — Where does the fold sit? It must sit where frames are serialised to, or read
  from, the socket. It cannot sit at the parsed message (§2.5), and it must hold even when a
  socket has more than one writer.
- **Q-H3** — Which frame is a request's first, for carrying the salt? Do `StateImport` frames
  belong to a request's forward lane?
- **Q-H4** — Is one SHA-256 per frame acceptable on the decode path?
- **Q-H5** — How does a stage know that a request has ended, on the reply fallback and on
  reused lanes? Does a lane carry exactly one request at a time? Replies have no
  `request_id`, so the reply fold depends on that.

## 9. What this does not establish, and what comes later

- **Correct computation.** Agreeing digests at every hop show that each hop's two ends saw
  the same bytes. They do not show that stage k's output is the right function of its input.
- **Collusion.** Two neighbouring stages can agree on anything. A coordinator can staff every
  stage (§0).
- **Later: a stage-level spot check.** A referee takes one stage's disclosed input frames and
  the salt, recomputes that slice, and compares the result with the stage's sent frames.
  - Recomputation on different hardware, or through a lossy codec (`S8_ROW_F32_RNE_V1`), will
    generally not be bit-identical.
  - So the spot check must compare **decoded tensors within a stated tolerance**. It must not
    compare digests.
  - The hop digest stays exact, because it compares bytes that were never recomputed.
- **Identity.** A stage record is signed by the stage's key. Who operates that key has the
  same standing as for any exchange record today (the threat model (`THREAT-MODEL.md`)).
- **Later: per-slice payment accounting** ("who is owed for which slice"). It depends on this
  record, because a slice nobody signed for cannot be accounted for. That design is out of
  scope here.

## 10. Open decisions, each with a recommendation

| # | Decision | Recommendation |
|---|---|---|
| Q-D1 | New role value `stage` plus the `x-mesh-stage-v1` block (with `side`), versus a separate record kind | **Role value plus block.** Sequence numbers, chaining, push and pairwise citation all stay as they are. |
| Q-D2 | Provisional citation purpose from the main record to a stage-exchange record | **`split_stage`**, flagged as provisional the same way `counterparty_half` is. |
| Q-D3 | Ship v0 unsalted, or wait for part B | **Ship unsalted, to coordinator + requester only** (§4.4). |
| Q-D4 | Whether a stage ever learns the end requester | **Not by default.** Any future switch is the requester's to set, not the coordinator's. |
| Q-D5 | The coordinator's seal deadline for stage records | **A new, explicit deadline**, recorded in the main record, so `absent` can be read against it. No deadline exists yet to reuse. |
| Q-D6 | How much of the Python coordinator receipt to keep | **Move the shape and producer rules to Rust.** Keep the Python emitter and verifier as the offline reference. Both pass one shared fixture (cross-implementation check). |
| Q-D7 | Nest rows by bracket key when they are not adjacent, which relaxes today's never-reorder rule | **Yes, for parent and child only.** Rows with no bracket keep strict time order. Children whose parent is missing render as ordinary rows, so there is never a parentless group. |
| Q-D8 | How a stage consents to its record being forwarded to the requester | **Declared once, as a condition of joining the run** (protocol-declared, not per request). A stage that does not accept it is not placed in the topology. |

## 11. Build plan, after review (not started)

1. **Twins nesting** (UI only, fixture mode). Parent/child rows by bracket key, role filter
   chips, tooltips and census entries. Splits reuse this.
2. **Stage-record producer** (`mesh/crates/mesh-llm-capsules-plugin/src/producer/stage.rs`). `role: "stage"` and `x-mesh-stage-v1`, fed
   from a *fixture* `skippy.stage.v1` event until the host seam exists. Tests:
   - an unknown role reads as `unknown`;
   - hop fields are present exactly per stage position;
   - `request_id` survives as a string;
   - `layer_end` stays exclusive.
3. **Coordinator side.** Stage-exchange records, the main record's `references[]`, and
   `x-mesh-coordinator-receipt-v1` in Rust. The Python producer rules carry over, each shown
   failing against its mutant:
   - `present` without a reference is rejected;
   - `absent` with a reference is rejected;
   - one stage record cited under two hops is rejected;
   - two records under one key read as `conflict`.
4. **Hop verifier.** Rust and the Python reference, run over one shared fixture:
   - equal lanes read `agree`;
   - a one-byte change in one frame reads `break`;
   - a swapped pair of frames reads `break`;
   - a missing end reads `gap`, never `agree`;
   - a required lane empty on both ends reads `gap`, never `agree`;
   - a record replayed at another `hop_index` reads `break`;
   - equal digests with unequal counts read `malformed`;
   - `relayed` return with a non-null `direct_return` is rejected.
5. **Delivery.** Stage → coordinator push, the coordinator deadline, the follow-up citation
   for late records, and carriage from coordinator to requester.
6. **UI.** Stage strip, the Peers split group and the stage node's row, in fixture mode.
7. **Host.** The `skippy.stage.v1` emitter, and only after the issue has an answer. Upstream
   text in host vocabulary only.

## 12. Build notes

Where the build had to choose something this document left open, the choice is recorded here.
The decisions in §10 are built as recommended.

- **Fold encoding.** "‖" in §4.2 is made unambiguous: every variable-length field is
  `u32be(len) ‖ bytes`, `request_id` is `u64be`, `hop_index` is `u32be`, and the salt is a
  zero-length field in v0. The direct return lane's `hop_index` is the final stage's index.
  `vectors/split-stage/fold-vectors.json` pins the bytes for both implementations.
- **The coordinator's first-hand observations** ride under `coordinator_observed`
  (`downstream` for stage 1, `direct_return.received` for the final stage under `direct`
  return) on its stage-exchange record, never under the stage's own hop field names. A reader
  keying on `upstream` / `downstream` cannot mistake the coordinator's end for the stage's.
- **Stage 0 in the receipt.** `topology[0]` is the coordinator (`role: coordinator`), and its
  `stages[0].bundle` is always `not_requested`: the coordinator asks nobody for its own slice,
  which rides on the main record's `x-mesh-stage-v1`.
- **`conflict` is a fourth bundle state.** Two distinct records under one stage key are cited
  together in `bundle_refs`, and neither is `present` (§3). The Python emitter accepts it, and
  its existing verifier reads it as a gap.
- **Topology entries carry the assignment** (`node_id`, `layer_start`, `layer_end`,
  `package_id`). The stage strip's `✓` compares each stage record against it.
- **Required lanes.** Forward is always required. Replies are required only for a `relayed`
  request that completed. The direct return lane is required only for a `direct` request that
  completed. A lane that is not required and is empty on both ends reads `not_applicable`,
  never `agree`, and the run line needs at least one lane that agrees.
- **Main-record citations.** The main record carries one `split_stage` reference per remote
  stage, to the coordinator's stage-exchange record, whether or not that stage's own record
  arrived. `stages[k].bundle_ref` names the stage's own record.
- **What the requester's check does not repeat.** It recomputes every `capsule_id`. Signatures
  are checked where records are received, and a stage key is not yet linked to
  `assignment.node_id` (that needs announced keys).

### 12.1 Delivery on the plugin

- **Event shape.** A `skippy.stage.v1` event is `{block, exchange_id?, coordinator_term?,
  topology?}`. The last three are stage 0's only. The channel is not declared in the manifest,
  since no host serves it yet.
- **Ordering the build relies on.** Stage 0's event arrives before the `openai.exchange.v1`
  terminal event of the same request, because the host holds both. The coordinator then holds
  that terminal event instead of sealing it. A terminal event that arrives first is sealed as an
  ordinary exchange, and the late stage-0 event is refused and counted. This goes to the host
  issue alongside Q-H1 to Q-H5.
- **Deadline.** `CAPSULES_SPLIT_STAGE_DEADLINE_MS`, 2000 ms by default, counted from
  stage 0's event and recorded in the receipt. A background tick seals released splits. The push
  to the requester waits for the next handler that has a host context, because pushing needs one.
- **A received stage record is cited twice.** Its ordinary `counterparty_half` citing record is
  sealed before the door's ack, as for every pushed record, and the coordinator's stage-exchange
  record cites it again. For a record that arrives after the seal, the ordinary citing record is
  the follow-up.
- **Carriage.** The coordinator's bundle gains `split_stage_records`. The door refuses the whole
  push `bundle_malformed` unless every carried record verifies, is a stage-side record of this
  split, is named by the receipt, and is carried once. The sender then re-pushes the plain
  bundle, so the main record still arrives and those stages read `◌ not received`. A bare push
  (checkpoint cadence off) cannot carry stage records.
- **Addressing.** A stage pushes to `coordinator_node_id` as a mesh peer id. That the stage
  runtime's coordinator node id is the id plugins address streams to is a host question.

### 12.2 Console

- **Where it lands.** The Evidence page now ships as the plugin's own web UI bundle
  (`web-ui/`, ported from the fork console's tab), so the strip is built
  there. Porting it back to the fork tab is a file copy.
- **Built:** the stage strip under a split row (§7.2), with cells, the hand-offs line folded to
  the worst lane per hop and the terminal state beside it. Also the stage node's own row line
  (§7.4), the Peers group "Nodes that served you through a split" (§7.3, shown only when
  non-empty), and tooltip-census entries for all three. The page runs its own copy of the
  requester's check, and it passes the same fixtures as the Rust and Python ones.
- **Not built yet:** nesting parent and child rows by bracket key (§7.1, Q-D7) and the role
  filter chips. The coordinator's `you delegated` child rows need a pane that emits
  stage-exchange rows, and no pane serves `split` yet (`DATA-ROUTES.md`). Until then the strip
  under the parent row carries the per-stage view, and twins keep today's adjacency grouping.
