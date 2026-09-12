# Confidential inference on random devices: the exhausted option space

**Status:** research synthesis, 2026-08-30. Goal: *exhaust* the options for
confidential LLM inference at production scale on random, untrusted consumer
devices — not survey them, but map the space completely enough to prove nothing
is missing, and say for each whether it can carry the product.

The bar throughout: nodes are RTX-class or Apple-Silicon consumer machines with
**no TEE**, churny, self-selected, some fraction actively adversarial and
logging, on residential uplinks. The device **owner** is a potential adversary —
the inverse of every datacenter threat model.

Companion to [`PUBLIC_MESH_TRUST.md`](PUBLIC_MESH_TRUST.md) (the integrity story,
which *is* solvable). This document is confidentiality, which is not — and says
exactly how-not, and what is reachable instead.

---

## The answer in three sentences

There is no single mechanism, and the option space is provably six buckets, not
an open-ended list. On **purely** random devices, cryptographic-grade
confidentiality ("the node mathematically cannot read it") is unreachable —
crypto is too slow, the hardware doesn't exist and is forged where it does, and
every transform is broken by a motivated node that trains an inverter. What *is*
reachable is a **defense-in-depth stack** that delivers strong practical
deniability and targeted-attack resistance — a real tier above plaintext mesh —
and a **cryptographic guarantee only returns if you re-introduce one non-random
trusted element**: the client itself, or one operator-run attested node.

---

## The completeness argument: why exactly six, and why no seventh

To stop a node from learning the prompt, ask one question — *what does the node
physically possess, and what can it do with it?* — and the space partitions by
exhaustive dichotomy:

The node either **receives the sensitive bits or it does not.**

- **Does not receive them** — two ways to withhold:
  - it never gets *enough* of them → **SPLIT / SHARD**
  - it can't tell *whose or which* they are → **ANONYMITY**
- **Receives them** — then they must be unreadable-in-practice, and unreadability
  has exactly three sources:
  - hard by math → **CRYPTO**
  - walled off by the machine → **HARDWARE**
  - scrambled by a secret the node lacks → **TRANSFORM**
- **Receives them and they are readable** — the only remaining lever is not on the
  data at all, but on the adversary's incentive → **ECONOMIC**

That is two withhold-strategies × three unreadability-sources + one
incentive-lever = **six mutually exclusive buckets**. Every technique the
literature offers lands in exactly one:

| # | Bucket | Unreadable/withheld by | Techniques |
|---|---|---|---|
| 1 | **CRYPTO** | math | FHE, 2PC/MPC, FSS, PIR |
| 2 | **HARDWARE** | isolation | GPU TEE + attestation, CPU-TEE+accelerator |
| 3 | **TRANSFORM** | secret encoding | orthogonal rotation (ConjFormer), GELO, obfuscation |
| 4 | **SPLIT** | insufficient data | U-shape split, PrivateLoRA, dimension-shard, plaintext secret-share |
| 5 | **ECONOMIC** | incentive | stake/slash, honeypot canaries, workload selection |
| 6 | **ANONYMITY** | unlinkability | OHTTP, mixnets (Funion), PCC non-targetability, oblivious batching |

**No seventh survives.** Four candidates that look new each collapse:
- *PII redaction / "just don't send it"* = SPLIT at token granularity.
- *Trusted relay / small anchor* = HARDWARE on one node + SPLIT for what it holds.
- *Legal / contractual / jurisdiction* = ECONOMIC (raise the cost of reading).
- *Steganography / decoy prompts* = TRANSFORM (encode among decoys) or ANONYMITY.

A genuine seventh would need a new physical location for the data, a fourth
source of unreadability beyond math/machine/encoding, or a lever other than
possession and incentive. None exists. **The space is closed.**

---

## Per-bucket verdict on random devices

Each bucket, its best 2026 position, the measured or cited evidence, and whether
it can stand alone on random consumer hardware.

### 1. CRYPTO — **dead on random devices**
FHE transformer inference is minutes/token at 7B on datacenter GPUs (Zama hybrid
GPT-2 ~11 s/token); ASIC relief 2028+. MPC needs designated non-colluding
parties — which an open Sybil-able mesh cannot provide.

*The one live sub-idea, and it's ownable:* in **2PC where the client is one of
the two parties**, non-collusion is free (the data owner never colludes against
itself) — and the client can also be the FSS **dealer**, which removes the
trusted-dealer assumption that fast protocols (SIGMA) smuggle in. Nobody does
this explicitly for LLMs. But the wall is bandwidth, not trust: online 2PC
traffic is provably linear in circuit size — **~2 GB per 7B forward pass,
~0.2–0.5 GB per generated token.** On a 20–50 Mbps home uplink that is 30–80 s
*per token* — dead. It is viable only on symmetric ≥1 Gbps, <2 ms-RTT links, i.e.
fiber-to-fiber or datacenter — *not* random residential devices. Co-design
(MPCFormer/Marill) buys ~1 order of magnitude; a second needs a real
breakthrough (silent generators for nonlinearity FSS keys, or non-interactive
decode). **Verdict: a fundable fiber/datacenter tier with a novel differentiator,
not a random-device mechanism.**

### 2. HARDWARE — **dead on target, alive only off-target**
No GeForce/RTX card and no Apple-Silicon GPU/NPU exposes an attestable
confidential mode; CC is exclusively H100/H200/Blackwell/RTX-PRO datacenter
silicon. And "wait for the hardware" is not a strategy on any fundable timeline:
Arm CCA Realms are unexposed in every shipping consumer SoC and can't yet pull
the GPU/NPU into a Realm; x86 memory encryption is being *withdrawn* from consumer
(AMD dropped it from Ryzen in 2025) or never offered (Intel Core); Android
AVF/pKVM is ubiquitous but solves the *inverse* problem (protect the VM from the
host, not the network from the host's owner) and roots trust in an owner-controlled
bootloader. Even Google ships confidential Gemini in datacenter TPUs, not on the
phone. **And the physical-owner attack is decisive**: Battering RAM ($50) and
TEE.Fail (<$1k) forge attestation on TDX/SEV-SNP *today*, and consumer silicon
will inherit the same missing integrity+freshness memory protection — which is
being removed for cost, not added. Realistic timeline for a usable,
owner-resistant consumer path: **not before ~2031, partial and flagship-only.**
The one milestone to watch: Arm CCA reaching consumer SoCs *with* RME-DA device
assignment *and* integrity+freshness memory. **Verdict: only ever a
verified-operator datacenter tier; falsified for random consumer hardware this
generation.**

### 3. TRANSFORM — **stops casual, not motivated**
Secret orthogonal rotation (ConjFormer) is cheap (<2% perplexity) and, measured
in [experiment 10](../../../experiment-10/REPORT.md), invertible in one Procrustes
solve when the base weights are public unless the client fine-tunes to diverge
them — which needs a private model per deployment, contradicting a stock-model
mesh. And the deeper break is general: any transform cheap enough to preserve
utility leaves enough structure to align, and a node can manufacture
(transformed-state, token) pairs to **train an inverter in the transformed
basis** — learned inverters hit 88–94% where they've been run, and the "empty
middle" study found 0 of 1,536 mechanisms reaching both moderate privacy and
moderate utility. **Verdict: real defense-in-depth against a node that greps;
theater if sold as confidentiality against a node that trains.**

### 4. SPLIT — **alive but weak alone**
Client holds embedding + first-k + last blocks, mesh holds the trunk. The
depth-vs-leakage curve is the problem: ActInv recovers 99.9% at 2 client blocks,
96.9% at 5, **86.8% at 7** — to starve inversion you must push the split so deep
the client runs most of the network, defeating the reason to offload. Measured
here two ways: [experiment 11](../../../experiment-11/REPORT.md) shows a slice of
the *input* residual identifies the token from 0.4% of coordinates (embeddings
are near-unique fingerprints), while 11c shows the *naive* attack collapses past
layer 0 on real interior states — but that is the weak attacker; the learned
inverter above is what a motivated node uses. Noise/DP on the boundary only moves
leakage ~90%→~45% at ~2× perplexity. **Verdict: a component, not a solution;
privacy and offload are directly antagonistic.**

### 5. ECONOMIC — **a real tier, not a guarantee**
Stake + slash + honeypot canary prompts that phone home if leaked, plus
Salad-style workload selection (route only non-sensitive traffic to untrusted
nodes). The honest limit: staking/slashing verifies *integrity* (attestable), but
confidentiality breach is a covert act leaving no evidence — only honeypots
detect it, probabilistically, and a node that logs silently and never re-shares
is uncatchable. **Verdict: "snooping is staked against a nonzero chance of a
honeypot burning your stake" is a defensible enterprise story as economic
deterrence, never as a cryptographic guarantee.**

### 6. ANONYMITY — **alive, partial-win**
Concede the node may read the activation; destroy its ability to *attribute* it.
OHTTP splits IP from payload; mixnets (Funion) give sender-receiver unlinkability
at production overhead; PCC-style non-targetability means an attacker who owns a
node can only see traffic that randomly routes there, and can't steer a target's
requests to it. **Verdict: defeats mass and targeted surveillance — turns "a node
read Jane's medical prompt" into "a node read *a* medical prompt it can't tie to
anyone" — but not content that self-identifies ("my name is Jane…"). A real
product tier for the right threat model.**

---

## The theorem, and the combination

**No single bucket reaches production-grade confidentiality on random devices** —
each fails a different way: crypto on speed, hardware on availability and physical
forge, transform and split on the learned inverter, economic and anonymity on the
guarantee. And the two that *could* give a mathematical guarantee (crypto,
hardware) are exactly the two that don't run on random consumer silicon.

**Corollary (the load-bearing claim):** a cryptographic-grade guarantee on random
devices provably requires re-introducing **one non-random trusted element** — the
client device itself (which reduces to SPLIT and its depth problem), or one
operator-run attested node (which is HARDWARE on a node you control). There is no
configuration of purely-random parts that yields the guarantee. This is not a gap
in current engineering; it falls out of the completeness partition.

**What the buckets buy in combination** — the achievable product, defense-in-depth:

- **ANONYMITY** (unlinkable, non-targetable transport) so no node can target or
  attribute a user — removes the *value* of reading.
- **SPLIT** (client keeps embedding + head + tail; mesh gets only mid-trunk
  activations) + **TRANSFORM** (secret rotation on the boundary) so a node sees
  scrambled mid-layer activations it cannot cheaply invert — raises the *cost* of
  reading.
- **ECONOMIC** (stake/slash + honeypot canaries) so a decode attempt is punished
  and probabilistically detected — removes the *incentive*.

A node in this stack sees rotated mid-layer activations it can't invert without
building a learned inverter, can't tie to a user, and is staking money against a
honeypot if it tries anyway. **That is strong practical deniability and
targeted-attack resistance — not "the node mathematically cannot read it."** Sell
the honest version: *targeted-snoop-proof, economically-deterred, obfuscated split
inference*, a real tier above plaintext mesh, explicit about its ceiling.

---

## The two watch-items where the guarantee tier arrives

1. **Consumer TEE silicon** — Arm CCA on consumer SoCs with GPU/NPU-in-Realm and
   integrity+freshness memory. Watch RME appearing in a Snapdragon/Dimensity spec
   and the RME-DA kernel series landing. Earliest ~2031, and still owner-forgeable
   without freshness silicon.
2. **FHE real-time** — ASICs (Niobium, Fabric) reaching interactive latency for
   small models. Earliest ~2028, small models first.

Until one lands, the guarantee tier lives only on operator-controlled datacenter
TEEs (the Apple/Google PCC pattern) or the client-as-dealer 2PC fiber tier.

---

## Recommendation

1. **Ship the defense-in-depth deniability tier** for random-device public mesh,
   sold honestly as deniability + anti-targeting, never as a cryptographic
   guarantee. It composes cleanly with the integrity work in
   `PUBLIC_MESH_TRUST.md`, which *is* a guarantee.
2. **Offer the cryptographic-guarantee tier only where a trusted anchor exists** —
   the private mesh (invited machines), operator-run attested datacenter nodes, or
   the client-as-dealer 2PC fiber tier. This is where confidential enterprise
   demand already is.
3. **Do not wait for consumer hardware** as a strategy; track the two watch-items
   as options, not plans.
4. **The one novel research position worth owning:** client-as-dealer 2PC — free
   non-collusion and a dealerless fast protocol — for the fiber/datacenter tier.
   It is defensible, unclaimed for LLMs, and the bandwidth wall is a co-design
   problem, not a trust one.

The space is exhausted: six buckets, none sufficient alone on random devices, a
combination that gives deniability but not a guarantee, and a proof that the
guarantee needs one trusted element. "Someone will figure it out" resolves to
one of exactly three futures — the silicon ships, FHE gets an ASIC, or the
product accepts a trusted anchor — and only the third is available now.
