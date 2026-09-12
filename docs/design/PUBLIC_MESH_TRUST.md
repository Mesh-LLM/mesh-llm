# Public-mesh trust: encrypted and verifiable inference

**Status:** design proposal, 2026-08-29. Backed by a field survey of the
2024–2026 literature and by experiment 10 in this repository.

**The question.** Today a mesh is private: you invite machines you trust. The
open question blocking a public mesh is the pair of unsolved problems named in
the commercial strategy — *peers can read prompts* and *peers can forge
outputs*. This document answers both, and is explicit about which one is
solved and which one is not.

---

## The one-paragraph answer

> **Integrity is solvable now; confidentiality is not.** Output forgery can be
> eliminated almost entirely by moving sampling authority back to the
> coordinator and adding per-stage activation receipts — both of which have
> working precedent in this codebase and in production elsewhere. Prompt
> confidentiality on hardware an adversary physically owns has no solution in
> 2026: the cryptography is four to six orders of magnitude too slow, the
> hardware enclaves do not exist on consumer or Apple silicon, and the one
> cheap obfuscation left standing — secret basis rotation — is **falsified for
> open-weight models by experiment 10**. The honest architecture is therefore
> two-tier: a *verified* tier whose outputs you can check and whose prompts you
> should still not consider secret, and the existing *private* tier for
> anything confidential.

---

## 1. Threat model, grounded in the current code

Not hypothetical. Every item below is a live property of `HEAD`.

| Surface | Current state | Where |
|---|---|---|
| Stage protocol | Plaintext TCP, 4-byte `SRDY` magic, no auth, no HMAC, no replay protection | `skippy-protocol/src/binary/codec.rs:15`, `skippy-server/src/binary_transport/socket.rs:114` |
| Inter-machine encryption | Real, but only when mesh-launched (iroh QUIC, ed25519 peer identity); the loopback bridge hop is plaintext | `mesh-llm-host-runtime/src/mesh/stage_transport.rs:1042-1083` |
| What crosses the wire | Raw little-endian **f32 activations** (8 KiB/token/boundary), prompt token IDs, positions, full sampling config incl. seed, ≤8 MiB chat metadata JSON, ≤512 MiB KV state | `skippy-protocol/src/binary/types.rs:361-375`, `docs/skippy/DATA_FLOW.md:84-99` |
| Sampling authority | **The final stage samples and returns an `i32`.** The head never sees logits and cannot check the token | `skippy-server/src/binary_transport/stage_execution.rs:756-775` |
| Embeddings / lm_head | Stage 0 holds embeddings; final stage holds `lm_head`, and under `LayerPackage` mode **also gets embeddings** | `skippy-server/src/runtime_state.rs:337-339` |
| Admission | `evaluate_direct_peer_admission` returns `Accepted` immediately when policy is `None` | `mesh/requirements.rs:614-631` |
| Bearer-token hole | Under default `TrustPolicy::Off`/`PreferOwned`, `STREAM_TUNNEL_HTTP (0x04)` and `STREAM_ROUTE_REQUEST (0x05)` bypass admission quarantine. The code comment says it plainly: *"a leaked invite token is a bearer credential for inference."* | `mesh/peer_state.rs:573-604` |
| Invite token exposure | `MeshListing.invite_token` is field one of a **plaintext public Nostr event** on five public relays; `--auto` joins with a token read off a relay | `network/nostr/contracts.rs:19-25`, `publish.rs:477` |
| Reputation | `NODE_REP.md` is process-local routing health only. Never gossiped, never persisted, no ban list — and says so | `docs/NODE_REP.md:23-31` |

Two consequences worth stating without hedging. First, **the API-terminating
node sees plaintext prompts and completions** — it runs the chat template, the
tokenizer and the guardrails — and via `STREAM_TUNNEL_HTTP` that node can be a
*remote* peer serving someone else's request. Second, **the head/worker trust
asymmetry is total**: the head ships a seed and receives a token ID, with no
logit return, no commitment and no witness.

---

## 2. What the field settled, 2024–2026

Compressed from four parallel literature sweeps. Full citations in
`experiment-10/REPORT.md`.

**Confidentiality.**

- Plaintext activations *are* plaintext prompts. This is measured, on
  Petals-style meshes specifically: >90% token reconstruction with auxiliary
  data, >50% without ([CCS 2025](https://arxiv.org/abs/2503.09291)); ActInv
  reports >98% precision and still ~77% recovery at a depth-7 split
  ([2026](https://arxiv.org/html/2605.23158)). Gaussian noise and sparsification
  do not stop it.
- FHE is minutes-per-token at 7B on datacenter GPUs (Zama hybrid GPT-2: ~11
  s/token, ~2.2 MB/token). ASIC relief is 2028+.
- MPC needs designated non-colluding parties and GB–TB per inference — the
  opposite of an open mesh. Best case is SIGMA at ~44 s/token for 13B *with a
  trusted dealer*.
- Permutation schemes (STIP, PermLLM, Centaur) were the cheap option and were
  **broken in May 2025**: >99% prompt recovery from permuted hidden states
  across layers 1–26 ([2505.18332](https://arxiv.org/html/2505.18332)).
- Secret orthogonal rotation with an equivariant transformer (ConjFormer, June
  2026) survived that attack — ≤1.3% token recovery at +0.4% perplexity. It was
  the last cheap candidate. **Experiment 10 falsifies it for our case** (§4).
- TEEs: no GeForce card and no consumer Ryzen/Core part has confidential
  computing; Apple attests device and binary identity, never memory secrecy.
  And where TEEs do exist, physical possession beats them —
  [Battering RAM](https://batteringram.eu/) ($50 DDR4 interposer) and
  [TEE.Fail](https://thehackernews.com/2025/10/new-teefail-side-channel-attack.html)
  (<$1k, DDR5) extract keys and **forge attestations** on patched SEV-SNP/TDX.
  A mesh node operator is exactly the adversary the TEE threat model excludes.

**Integrity.**

- zkML tops out around 270M parameters end-to-end (DeepProve: GPT-2 at ~174
  tok/min, 7–54 MB proofs). Not applicable at our model sizes this decade.
- Activation commitments work and are production-proven on *pipeline splits
  over heterogeneous consumer GPUs*: TOPLOC is 258 bytes per 32 tokens, tolerant
  of cross-hardware numerical variance by design, and was used across ~1,250
  mixed GPUs with per-stage replay for blame assignment
  ([TOPLOC](https://arxiv.org/abs/2501.16007),
  [SYNTHETIC-2](https://www.primeintellect.ai/blog/synthetic-2-release)).
- Fixing the sampling seed makes decode a deterministic function of the logits,
  which collapses "did you sample honestly?" into an argmax check. DiFR reports
  >98% exact token match across A100/H200 and vLLM/HF, detecting a wrong seed in
  ~100 tokens and 4-bit quantisation in ~1,000
  ([2511.20621](https://arxiv.org/abs/2511.20621)).
- Verification by *re-prefill* costs ~1% of generation
  ([VeriLLM](https://arxiv.org/pdf/2509.24257)).
- Black-box statistical auditing is real but evadable by an adaptive cheater,
  and canaries fail once the adversary can recognise them
  ([Berkeley](https://arxiv.org/abs/2504.04715)).

**What survived contact with adversaries in the field.** Only four things:
hardware attestation, GPU fingerprinting bound to traffic decryption (Chutes
GraVal), statistical telemetry with fast deranking (OpenRouter Auto Exacto),
and reputation plus workload selection (Salad). Spot-check/arbitration
protocols were never deployed anywhere; io.net's incentives without attestation
produced ~1.8M fake GPUs.

---

## 3. The architecture

Five layers, ordered by confidence. Layer 0 is required regardless of anything
else; layers 1–2 are proven elsewhere; layer 3 is novel; layer 4 is the
measurement nobody has made.

### Layer 0 — close the doors that are open (not research)

1. Cover consumption with the same enforcement as joining: gate
   `STREAM_TUNNEL_HTTP` and `STREAM_ROUTE_REQUEST` on admission under **all**
   trust policies, not just `RequireOwned`/`Allowlist`.
2. Stop broadcasting a bearer credential. A public `MeshListing` should carry a
   contact endpoint, not `invite_token`; joining becomes a challenge-response
   against the mesh origin key, with the existing `SignedBootstrapToken`
   (`mesh/requirements.rs:45-55`) as the credential and expiry enforced.
3. Make the stage frame carry its own protection rather than inheriting it from
   how the process happened to be launched: an explicitly versioned
   authenticated-encryption field on `StageWireMessage`, fail-closed across
   generations. `SignedEncryptedEnvelope`
   (`mesh-llm-identity/src/envelope.rs`) already implements sign-then-encrypt
   with domain separation and is unused on the data path. Follow the fp8
   precedent in `SKIPPY_PROTOCOL_TODO.md:121` — a new versioned field, never a
   reinterpretation of an existing one.
4. Pad and pace inter-stage frames. Token-granular traffic leaks queries to a
   passive observer even when payloads are opaque
   ([2411.01076](https://arxiv.org/pdf/2411.01076)).

### Layer 1 — sampling authority returns to the coordinator

Today the final stage decides the token. Change it to return the top-*k* logits
plus its `TokenSignal`, and let the coordinator sample with the seed it already
ships in `StageSamplingConfig`.

This is not a new idea in this codebase — it is the generalisation of a move
already made and already load-bearing. `docs/skippy/PIPELINED_VERIFY_WINDOW.md:34`
says it outright: *"Acceptance now has one owner: the coordinator."*

Effect: **a node can no longer forge a token at all.** It can only corrupt the
residual stream it was given, which is a different and much more detectable
attack. Cost: top-*k* logits instead of one `i32` on the return path — trivial
against 8 KiB/token in the forward direction, and the direct final→stage-0
return path (`binary_transport/direct_return.rs`) already exists to carry it.

### Layer 2 — stage receipts

Each stage commits to the top-*k* values of its output boundary activation,
every 32 tokens, TOPLOC-style: ~258 bytes per stage per 32 tokens against 256
KiB of actual activation traffic over the same window. Acceptance stays
optimistic; on dispute the coordinator replays a single stage to assign blame,
then ejects.

Verification uses re-prefill, not re-decode — the ~100× asymmetry is what makes
this affordable. `crates/skippy-correctness/src/glm_dsa_trace.rs` already
implements per-tensor trace capture and comparison
(`compare_tensor_traces:484`) and is the natural basis for the replay path.

### Layer 3 — draft-acceptance fingerprinting (novel)

**The problem it solves.** Every black-box audit in the literature fails the
same way: the adversary can recognise the test. Canary prompts are recognisable;
benchmark traffic is recognisable; a cheater serves the real model to anything
that looks like an audit. The Berkeley paper's conclusion is that software-only
auditing is unreliable against an adaptive adversary.

**The mechanism.** In speculative decoding the acceptance rate α is a divergence
measure between the draft distribution q and the target distribution p. The
coordinator already owns acceptance and already runs a draft model. So:

- Let the coordinator hold a **secret draft model** — chosen per session from a
  pool, never disclosed to the mesh.
- α, measured per token on ordinary live traffic, is a statistic of the target
  model's identity. Substituting a smaller model, or quantising harder, moves p,
  which moves α.
- The audit is therefore **indistinguishable from the work**: there is no canary
  to recognise, because every token of every real request is a measurement.
- The adversary cannot tune its output to hit the expected α, because α depends
  on a draft model it does not have.

**Why it fits our problem specifically.** α is a *distributional* quantity, so
it tolerates the floating-point divergence between Metal, CUDA and ROCm that
makes byte-equality verification impossible on a heterogeneous mesh. And the
machinery is already built: the coordinator holds both the proposal and the
target's reply and already computes the accepted prefix
(`PIPELINED_VERIFY_WINDOW.md:52`), a coordinator-side draft model path already
exists (`resolver/speculative.rs:43`, mode `"draft"`), and `TokenSignal`
(`skippy-ffi/src/sampling.rs:33-54`) already computes entropy, top and second
logprob and margin per token.

**One correction to make it sound.** The two draft sources deployed today are
*not* usable as audit signals. Native MTP candidates come from the target model
running **on the mesh**, so a cheating node supplies both the draft and the
verdict; and the n-gram continuation is derived from committed history the mesh
has also seen. The audit therefore requires a *third* proposal source — a small
draft model held only by the coordinator, its identity drawn per session from a
pool. That is a configuration and bookkeeping change on an existing code path,
not new inference machinery.

**Prior-art position, checked adversarially.** No paper, preprint or repo found
uses draft-model acceptance rate as an audit statistic against an untrusted
provider. The closest work runs the other way: DiFR
([2511.20621](https://arxiv.org/html/2511.20621)) treats speculative decoding as
a sampling mode it cannot yet verify, and its Appendix F sketch assumes verifier
and provider share a *published* draft model — the inverse of using a secret
one. Every commitment scheme in the field (TOPLOC, VeriLLM, Proof-of-Logits,
TensorCommitments, DeepProve) requires provider cooperation; this does not.

Three corrections from that check, which change what we may claim:

1. **Do not sell it as the first undetectable natural-traffic audit.** The
   rank-based uniformity test ([2506.06975](https://arxiv.org/html/2506.06975v5))
   already audits live traffic without a probe pattern. Our real edge is cost
   and topology: RUT needs a full-size reference copy of the claimed model plus
   rank access, whereas α needs only a **small** secret draft. On a mesh the
   coordinator is often a laptop, so this is the difference between an audit
   that can run and one that cannot.
2. **Do not assert quantisation detection.** It is contested —
   [2505.22179](https://arxiv.org/html/2505.22179v1) reports quantisation has
   minimal impact on mean accepted length, while Activation-DiFR catches 4-bit
   at AUC > 0.999 from two tokens. So α is the detector for **model
   substitution**, where separation is large (published α gaps of 0.369 vs
   0.544 across same-size models, and 0.680 vs 0.038 across architectures), and
   **layer 2 receipts are the detector for quantisation**. The layers divide the
   work; neither covers both.
3. **Budget for α-forgery.** [Mistletoe](https://arxiv.org/abs/2605.14005) and
   [2607.21804](https://arxiv.org/abs/2607.21804) construct KL-bounded prompts
   that collapse draft acceptance while preserving the target's visible
   behaviour. Framed as denial-of-service on speculative *speed*, they read
   directly as false-positive injection against this audit, and the reverse —
   tuning a cheap model to sit at the expected α — needs analysis before launch.

**Where it is weak, stated up front.** α alone is spoofable in principle by a
node that accepts drafts at a tuned rate while generating cheaply — right α,
wrong tokens. So α is a *detector*, not a proof, and must be paired with layer 2
receipts and occasional DiFR-style logit-gap checks on sampled positions. α also
varies with prompt domain, so the test should be **relative** — nodes serving the
same claimed model are each other's controls, which the router is already
positioned to arrange — rather than against an absolute baseline. Finally, the
same channel leaks the other way: acceptance patterns fingerprint user queries
at >75% accuracy ([2411.01076](https://arxiv.org/pdf/2411.01076)), and a node
that watches our draft tokens learns about the secret draft over time, so the
draft must be rotated.

### Layer 4 — tolerance bands for Apple silicon (the missing measurement)

Every activation-commitment scheme needs to know how far honest hardware
disagrees, so it can set a threshold below which it calls cheating. Those bands
have been published for NVIDIA parts. **Nobody has published them for Metal or
ROCm**, which is exactly the hardware that makes this mesh interesting.

The experiment is well-defined and we have the equipment: run identical layer
packages on `micstudio` (Metal) and a CUDA node via jianyang, capture boundary
activations with the existing trace harness, and measure the divergence
distribution. Then measure the divergence induced by *actual* cheating — Q4
substitution, a skipped layer, a smaller model — and ask whether the two
distributions separate. If they do, stage receipts work on Apple silicon and we
are first to know it. If they do not, layer 2 is limited to same-class node
pairs and layer 3 carries more weight.

This is the load-bearing unknown in the whole design.

---

## 4. Experiment 10: secret basis rotation cannot carry this product

The one cheap confidentiality mechanism the literature still endorsed was
secret orthogonal rotation of the residual basis (ConjFormer). Experiment 10
measured it against real Qwen3-0.6B weights. Full result and method in
[`experiment-10/REPORT.md`](../../../experiment-10/REPORT.md).

**Measured.** Given a conjugated weight matrix and the public checkpoint it came
from, the secret rotation is recovered and **97.9–99.9% of tokens read exactly**,
from every matrix tested. Control (rotated state, no attack): 0.0%. The
embedding matrix is the worst case — 151,936 rows against 1,024 unknowns is a
148× overdetermined solve, and it still reads 98.6% when the deployed weights
are perturbed by a full 100% relative Frobenius norm, ten times the perturbation
4-bit quantisation introduces.

**Correction we owe the record.** This attack is not novel: it is Sections 3.3
and 3.4 of the ConjFormer paper itself, and the paper defeats it by having the
client **fine-tune before rotating** so the deployed weights diverge from the
public ones. Our sweep used isotropic random noise, which a least-squares solve
averages away, so it does not refute their structured-drift defence and we do
not claim it does.

**Why it still settles the question for us.** The security lives in the private
fine-tuning drift, not in the symmetry. A mesh's proposition is pooling machines
to serve *stock* open-weight models; requiring a privately fine-tuned model per
deployment, kept secret from every node, contradicts the product and breaks
layer packaging, model sharing and the catalog. The authors also state plainly
that it is not a cryptographic or differentially private mechanism, and that
norms, pairwise distances, repeated-token patterns and attention logits stay
visible to the server.

**One finding to act on regardless.** Under `LoadMode::LayerPackage` the final
stage already receives the embedding matrix
(`skippy-server/src/runtime_state.rs:337-339`) — the strongest attack surface
measured. If a rotated model is ever shipped to a worker that also holds the
embedding table, the rotation falls to a single solve. Worth a comment at that
call site whether or not rotation is adopted.

---

## 4a. The one confidentiality direction still open: dimension-sharded serving

Worth recording because it is the only architecture found that could genuinely
improve prompt confidentiality on untrusted nodes, and its blocker is a number
we can measure rather than a proof.

Every inversion result in the literature attacks a **full** residual state. But
a transformer can be sharded so that no node ever holds one. If each node owns a
subset of attention heads and a disjoint slice of the FF neurons, and the linear
layers are input-split, then node *i* only ever sees `h[:, S_i]` — 128 or 512
dimensions of a 4096-dimensional stream — plus a handful of all-reduced scalars
(the RMSNorm sum-of-squares, the attention partial sums). Nobody has published
how invertible a *slice* of a residual state is. It is a cheap experiment and
the answer is not obvious.

The blocker is that this is tensor parallelism, which needs an all-reduce per
layer. That is exactly the O(N·L) synchronisation cost that made pipeline
parallelism the right choice for WAN in the first place — the same conclusion
Prime Intellect reached independently. So the honest position is: **plausible on
a Thunderbolt/RDMA LAN, probably not over the public internet** — and a LAN mesh
is a private mesh, which did not need the confidentiality in the first place.

**Measured, experiment 11 ([`experiment-11/REPORT.md`](../../../experiment-11/REPORT.md)).**
The slice-invertibility question is now answered, and the answer is "no, but
with a real caveat."

- *At the input boundary, sharding gives nothing.* The layer-0 residual is the
  embedding row, and a node holding **4 of 1024 coordinates identifies the token
  98.5% of the time** — flat across every slice size and slice-choice strategy,
  because an embedding is a near-unique fingerprint in even a handful of
  coordinates. The intuition "each node sees only a few dimensions so it can't
  read anything" is false.
- *At interior layers a safe regime exists but is narrow.* Modelling a mixed
  state as `E[token] + σ·noise`, a small slice of a well-mixed state collapses to
  near-zero recovery (an 8-dim slice at σ=1 recovers 0.2%). But the noise needed
  to hide a slice scales with the slice — 8 dims safe by σ≈0.5, 128 dims not
  until σ≈4 — so safety demands *fine* sharding, which demands *more* all-reduce
  bandwidth, pulling directly against the WAN constraint that motivated sharding.
  And the measurement used a weak (nearest-neighbour) attacker and isotropic
  noise, both of which flatter the defender.

Still open, but scoped to one number: σ_k, the ratio of the non-token residual to
the token embedding at layer k, on *real* states. Residual norms grow with
depth, so there is plausibly a depth beyond which small slices are safe against a
nearest-neighbour adversary. That is a clean lab forward-pass experiment, and a
learned-inverter attack on slices is the companion that would decide whether the
safe regime survives a serious adversary. Neither is on the critical path, and
neither rescues the reduction-point exposure: the nonlinearities still need a
full pre-activation held in the clear unless computed under MPC.

---

## 5. What we are not building, and why

| Not building | Why |
|---|---|
| FHE or MPC inference | 10³–10⁶× too slow; MPC's non-collusion assumption is incompatible with an open mesh |
| zkML proofs of inference | Ceiling is ~270M parameters end-to-end |
| Permutation obfuscation | Broken, May 2025 |
| Secret basis rotation | Falsified for open-weight serving by experiment 10 |
| TEE attestation as the mesh's foundation | No consumer or Apple-silicon path; and physical possession forges attestations for <$1k. Use it *opportunistically* where a node has it, to lower that node's audit rate — never as the floor |
| A claim that public-mesh prompts are private | They are not, and saying so would be the one thing we cannot walk back |

---

## 6. Honest product framing

One statement per surface, in operator language, no hedging:

- **Private mesh** — machines you invited. Traffic between them is encrypted and
  authenticated. This is where confidential work goes.
- **Public mesh** — machines you did not invite. Their outputs are verified: a
  node cannot forge a token, and a node that corrupts its stage is detected and
  ejected. **A node operator may be able to reconstruct your prompt.** Do not
  send anything you would not send to that operator.

The second bullet is a feature of the writing, not a gap in the product. Every
network that pretended otherwise — Petals, the early GPU marketplaces — lost
credibility exactly there, and Petals' own documentation ended up saying the
same sentence.

---

## 7. Build order

1. **Layer 0** — admission covers consumption; stop publishing the invite token;
   versioned AEAD on `StageWireMessage`; pad/pace frames. No research
   dependency; do this first regardless of everything below.
2. **Two measurements, in parallel, both gating builds.** Neither is expensive
   and both can run on the existing lab through jianyang.
   - *Metal↔CUDA tolerance bands* (gates layer 2). Same layer package on
     `micstudio` and a CUDA node; capture boundary activations with the existing
     trace harness; measure honest cross-backend divergence against the
     divergence a real cheat induces (Q4 substitution, dropped layer, smaller
     model). If the distributions separate, stage receipts work on Apple
     silicon and we are first to know. If not, layer 2 is limited to same-class
     node pairs.
   - *α separation power* (gates layer 3). Fix a small coordinator-side draft;
     measure the acceptance-rate distribution against an honest target and
     against substituted targets; report tokens-to-detect at a fixed
     false-positive rate. Include a quantisation arm specifically to settle the
     contested question, and an adversarial arm reproducing a Mistletoe-style
     acceptance collapse to size the false-positive risk.
3. **Layer 1** — coordinator-owned sampling, top-*k* logit return.
4. **Layer 2** — stage receipts, optimistic acceptance, replay-on-dispute.
5. **Layer 3** — secret-draft acceptance fingerprinting, relative across nodes.
6. **Reputation** — feed layers 2–3 verdicts into the cross-node reputation that
   `docs/NODE_REP.md:97-106` already scopes as future work: what evidence is
   shared, Sybil resistance, private-mesh opt-out.

Order matters for one reason beyond dependency: step 2's first measurement is
the load-bearing unknown in the whole design, and it is cheap. If Metal and CUDA
disagree by more than a cheat does, half of this document changes.

Steps 1 and 3 are worth doing for the *private* mesh too, which is what makes
this fundable ahead of any public launch: coordinator-owned sampling and frame
authentication improve the product we sell today.
