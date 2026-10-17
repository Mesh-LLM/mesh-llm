# The verification chain: what each link proves, and how

*A technical companion: what this plugin's records and the Evidence page can and cannot
show. A plain-language version and the full threat model are linked from
[THREAT-MODEL.md](THREAT-MODEL.md). This document is the middle layer:
the actual cryptographic chain, link by link — for each link, **what it proves**, **the
mechanism (and RFC)**, **how a verifier checks it**, and **what it does *not* prove.***

The chain runs from a single served exchange up to a receipt from a log the serving
node does not control. Every link is **offline-verifiable** — a verifier needs the
artifacts, not access to any live service.

On the mesh's proxied path (`dispatch_path: RemoteMesh`), the routing node now runs this
same chain for its own requester-role half — joined to the peer's served-role chain by
the shared client nonce, never `exchange_id` (minted independently per node) — so a
proxied exchange yields two independently offline-verifiable capsules, not just the
peer's (requester-side seal on proxy, 2026-09-07).

Standards this builds on: **RFC 8785** (JCS canonical JSON), **RFC 9052/9053** (COSE /
`COSE_Sign1`), **RFC 9162** (the SHA-256 Merkle verifiable data structure), **RFC
9942** (COSE Receipts / SCITT receipts), **RFC 9943** (SCITT architecture), plus the
`draft-mih-scitt-agent-action-capsule` (the record format) and
`draft-mih-sokolov-scitt-payload-binding` (CPB, how the payload binds to a receipt).

---

## Link 0 — Canonicalization (the foundation everything digests over)

- **Proves:** nothing on its own — but it's the reason every later digest is
  reproducible. Two implementations, given the same record, produce the same bytes.
- **Mechanism:** **RFC 8785 JSON Canonicalization Scheme (JCS)** over the record's
  value domain (`agent_action_capsule/canonical.py`, §2/§5.1 of the AAC draft). A digest
  is `HEX(SHA-256(JCS(value)))` — "JSON-DIGEST."
- **How you check it:** you don't check this link directly; you rely on it when you
  recompute any digest below and it matches. The viewer's `mesh_verify.js` ports the
  exact same JCS so a browser reproduces the server's digests byte-for-byte.
- **Does not prove:** anything semantic. It's a serialization discipline, not evidence.

## Link 1 — Content-addressing: the `capsule_id` (integrity)

- **Proves:** the record has not been altered. The `capsule_id` **is** a hash of the
  record's own canonical bytes, so changing any committed field changes the id.
- **Mechanism:** `capsule_id = HEX(SHA-256(JCS(record)))`. The committed fields include
  the model attestation, the request/response/tool-call/reasoning digests, the
  generation parameters, the nonce source, timing — everything that describes the
  exchange.
- **How you check it:** recompute `SHA-256(JCS(record))` and compare to the stated
  `capsule_id`. Mismatch ⇒ the record was tampered with (or isn't canonical). The
  viewer does this **in your browser**; `agent-action-capsule verify` does it offline.
- **Does not prove:** *who* produced it, or that its *claims* are true — only that this
  exact set of bytes hangs together. (Attribution is Link 3; claim-honesty is the
  threat model's job.)

## Link 2 — Field digests (the disclosed bodies match what was sealed)

- **Proves:** a disclosed prompt / response / tool-call / reasoning body is exactly the
  one the record committed to — the answer you're reading is the answer that was
  sealed, not a later substitution.
- **Mechanism:** the record commits `agent_input_digest`, `agent_output_digest`,
  `tool_calls_digest`, `reasoning_digest` — each `SHA-256(JCS(body))`. Disclosure is
  per-field: a body is either shown (so you can recompute) or sealed to its digest only.
- **How you check it:** for any *disclosed* field, recompute its digest over the shown
  body and compare to the committed digest. (Honest note: on the host-served *observe*
  path the sealed `response_digest` is a digest of the served **facts** — model +
  usage — not the streamed text; the viewer says so, and verifies the facts it can.)
- **Does not prove:** anything about a field left *sealed* — you get the digest, not the
  content, by design (that's the privacy/selective-disclosure property).

## Link 3 — The signed statement (attribution + non-repudiation)

- **Proves:** the record was produced by the holder of a specific node key, and that
  holder cannot later deny it (non-repudiation).
- **Mechanism:** a **`COSE_Sign1`** signed statement (RFC 9052) over the capsule, using
  **Ed25519 / EdDSA** (RFC 9053). In SCITT terms the capsule is the **Signed
  Statement**; the signer's key id rides the COSE protected header. (`.cose` sidecar
  files alongside the ledger.)
- **How you check it:** verify the `COSE_Sign1` signature against the node's public key
  (`agent-action-capsule verify --store`, or `scitt-cose`). A bundle that *carries* a
  signed statement but is checked *without* a key is reported `UNVERIFIED`, never PASS.
- **Does not prove:** that the key belongs to any real-world person, company, or piece
  of hardware. The key is **node-generated and self-attested** — it proves *continuity*
  (the same signer across exchanges), not *identity*. Binding it to a hardware root or a
  vouched membership is a higher assurance tier, out of scope for this link.

## Link 4 — The hash-chain (order within one node's log)

- **Proves:** the ordering of a node's own records, and that none was silently inserted
  or removed between two chained entries.
- **Mechanism:** each capsule carries `chain.parent_capsule_id` with `relation:
  "follows"` — a per-node hash chain. A store-level verify walks it.
- **How you check it:** verify with the store (`verify --store <ledger>`); a broken or
  missing parent surfaces as `chain_parent_missing`. (Vintage `format_version 2` records
  excluded `chain` from the id preimage; the mesh producer emits `format_version 4`,
  which commits `chain`, so a forged parent is caught.)
- **Does not prove:** that the node keeps *only* this history. A hash chain orders one
  history; it cannot show it's the only one the node keeps — that gap is closed by
  publication to a log the node does not control (Links 6–7).

## Link 5 — The local append-only accumulator (inclusion + append-only)

- **Proves:** that a specific capsule is included in the node's log at a given size, and
  that the log grew append-only (no rewrite of earlier entries).
- **Mechanism:** a **Merkle Mountain Range (MMR)** maintained locally over the capsules
  (the checkpointed-local-log, CLL, which the plugin keeps through its checkpoint cadence). Its state is summarized by its
  **peak hashes**; it yields **inclusion proofs** for any leaf and **consistency
  proofs** between two sizes.
- **How you check it:** verify a leaf's inclusion proof against a peak commitment;
  verify a consistency proof between an earlier and later size (append-only).
- **Does not prove:** anything to an outsider *yet* — the MMR is the node's own
  structure. It becomes evidence against the node only once a commitment to it is
  published externally (Links 6–7).

## Link 6 — The signed checkpoint (a committed snapshot of the whole log)

- **Proves:** the node committed, under its signature, to the entire state of its log at
  a specific size and time — so it can't later present a different history for that size.
- **Mechanism:** a **`COSE_Sign1` `cll-checkpoint`** (the cll COSE-wire checkpoint
  spec): `kind="cll-checkpoint"`, `log_size`, a **peak-list `commitment`** (the MMR
  peaks, *not* a bagged root — the root derives from it), `prev_size` / `prev_commitment`
  for the previous checkpoint, and `issued_at`. The signing body is the spec's 8 fields;
  `log_id` rides the CWT `iss` claim, the key id the COSE `kid`.
- **How you check it:** `verify_checkpoint_cose_offline(cose_bytes)` — the `COSE_Sign1`
  verifies under `kid`, and the commitment/prev_commitment reconstruct from the peak
  lists. A real **consistency proof** chains each checkpoint from the previous size.
- **Does not prove:** that anyone *else* saw it. A node could still, in principle, sign
  two divergent checkpoints for different audiences — until it's registered with a
  witness (Link 7), which is what makes equivocation visible.

## Link 7 — Witness registration → the receipt (tamper-evidence, non-equivocation)

- **Proves:** the checkpoint was recorded in a transparency log the node **does not
  operate**, at a time — so the node's history is tamper-evident against something
  outside it, and it cannot show different histories to different parties without the
  disagreement being detectable.
- **Mechanism:** the checkpoint is `POST`ed (`application/cll-checkpoint+cbor`) to a
  **SCITT Transparency Service** (RFC 9943) that the operator named. There is **no
  default witness**: with none named, no checkpoint is registered anywhere and no
  witness is contacted. Any service that implements the protocol can be named;
  `witness.agentactioncapsule.org` is one public instance an operator may choose, like
  any other. The service returns a **COSE Receipt** (RFC 9942) whose inclusion
  proof uses the **RFC 9162 SHA-256** verifiable data structure.
- **How you check it:** verify the receipt offline with `scitt-cose` (RFC 9942 receipt,
  RFC 9162 SHA-256 inclusion) against the witness's public key — no access to the
  witness required at verify time. States are distinguished (`pending` / `submitted` /
  `anchored` / `rejected` / `failed`); "anchored" is never claimed on submission alone.
- **Does not prove:** that the witness itself is honest in isolation. One log is only as
  neutral as its operator; the structural answer is **more than one witness** —
  independent logs holding the same checkpoint mean neither can rewrite or withhold
  without visible disagreement. (This is why the witness is open for anyone to run one.)

## CPB — how the payload binds to the receipt (so the two can't be mixed up)

- **Proves:** the receipt is *about this class of payload* and this record, not some
  other — a capsule can't be re-pointed under a receipt for something else.
- **Mechanism:** the **CBOR Payload Binding** (`draft-mih-sokolov-scitt-payload-binding`)
  — the capsule is registered under the payload class **`mesh-inference-exchange`** in a
  machine-readable registry (vendored into the consumers), with a **never-reject**
  invariant so an unknown value degrades honestly rather than failing closed on a
  verifier that hasn't updated.
- **How you check it:** the payload-class value resolves against the vendored registry;
  a known-provisional value renders as such, an unknown one as `known_provisional` /
  honestly-unknown, never a silent pass.

---

## The whole chain, as one offline check

A third party, holding only the disclosed artifacts, runs:

1. **Recompute the `capsule_id`** — `SHA-256(JCS(record))` matches (Link 1).
2. **Recompute each disclosed field digest** — the bodies match what was sealed (Link 2).
3. **Verify the `COSE_Sign1`** — the node key signed this record (Link 3).
4. **Walk the chain** — parent links intact, append-only (Link 4).
5. **Verify inclusion** — this capsule is under the checkpoint's commitment (Links 5–6).
6. **Verify the checkpoint signature + consistency** — the log committed to this state,
   append-only from the last (Link 6).
7. **Verify the witness receipt** — that checkpoint is in a log the node doesn't run,
   RFC 9162 inclusion under the witness key (Link 7).

Each link is independent: a verifier can stop at any level of assurance it needs. A
requester settling a bilateral dispute may stop at Link 3; a third party who trusts
neither side needs Link 7.

## What the whole chain still does not prove (read this)

The chain makes an account **durable, attributable, ordered, and externally
tamper-evident.** It does **not**, by itself, make the account **true**:

- **Identity is self-attested** (Link 3) — a node key, not a hardware root or a named
  party. Continuity, not identity.
- **The hardware/model line is a signed *claim*** — `model_name_digest` is
  `SHA-256(model_name)`, *named as a name hash*, never a weights or package binding. The
  chain proves the node *said* it; consistency (sealed `latency_ms` vs. claimed
  model+hardware) and behavioural/fingerprint checks — and, eventually, a real
  weights/TEE binding — are what test whether it's so.
- **The record counts, it does not price** — units (tokens, compute time) are committed;
  no currency, rate, or invoice is (TRUST-MODEL §6).
- **History is evidence, not a score** — the chain supplies checkable inputs; a relying
  party computes over them, and no one is the authority for that computation
  (TRUST-MODEL §7).

Saying each of these plainly is what makes the parts that *are* proven usable as
evidence. For the per-role version of "what each party wants answered, and the evidence
that addresses it," see the threat model ([THREAT-MODEL.md](THREAT-MODEL.md)) §2.2–2.5.
