# Trust Tiers and Request Trust Policy

Status: proposal

Related:

- [NODE_OWNER_IDENTITY.md](NODE_OWNER_IDENTITY.md)
- [../NODE_REP.md](../NODE_REP.md)
- [RECEIPTS.md](RECEIPTS.md)

## Summary

A request sent into the mesh runs on someone else's machine. Two separate
questions follow:

- **Privacy**: can the serving node keep, log, or read my prompt and response?
- **Model integrity**: did the serving node run the model, quant, and settings
  it claims?

No single mechanism answers both on every kind of hardware. This proposal
defines **trust tiers** that each answer them to a stated degree, a **request
trust policy** that lets a client require a tier, and routing that filters
candidates by that policy and fails closed.

Phase 1 adds the policy, weight-hash model identity, and a `trusted` tier built
on the owner attestation that already exists. Phase 2 adds a `measured` tier
based on TPM measured boot and remote attestation. A `tee` tier and a
receipt-based integrity check for public nodes are later work. They plug into
the same policy and are only outlined here.

## Goals

- Let a client say which nodes may serve a request, and guarantee that routing
  never silently widens that set.
- State plainly what each tier proves and what it does not.
- Reuse existing owner attestation and the trust store for the `trusted` tier.
- Make model identity a fact about weight bytes, not a model name.
- Offer a privacy tier that works on consumer Linux GPU hosts without
  confidential-computing GPUs.
- Keep mixed-version meshes working. Old nodes simply never qualify for a tier
  above `any`.

## Non-goals

- Secrecy against an owner with physical access to the machine. Only a TEE
  approaches that, and this proposal does not add one.
- Mesh-wide reputation or punishment. [NODE_REP.md](../NODE_REP.md) stays
  local.
- Changing admission. Tiers filter request routing. Admission policy stays as
  defined in [NODE_OWNER_IDENTITY.md](NODE_OWNER_IDENTITY.md).
- Verifying model outputs. Receipts are a separate proposal:
  [RECEIPTS.md](RECEIPTS.md).

## Current State

- **Model identity.** `ServedModelIdentity.identity_hash` is
  `identity_hash_for`, a SHA-256 of a reference string such as
  `repo@revision/file` (`mesh-llm-membership/src/model_identity.rs`). It is
  `None` for local paths and says nothing about the bytes served.
- **Weights digest.** `weights_digest` (`mesh-llm-membership/src/weights_digest.rs`)
  is a SHA-256 of the GGUF file bytes. It exists on the Rust
  `ServedModelIdentity` (`mesh-llm-types/src/mesh/mod.rs`) and appears in
  exchange `serving_provenance` (`host-runtime/src/plugin/openai_exchange.rs`).
  It is not a field of `ServedModelIdentity` in `node.proto`, so peers do not
  gossip it. Its own docs call it self-reported, and note a TOCTOU gap between
  the file the hasher opens and the file the runtime serves.
- **Owner attestation.** `SignedNodeOwnership`, `TrustStore`, `OwnershipStatus`,
  and the admission `TrustPolicy` (`off`, `prefer-owned`, `require-owned`,
  `allowlist`) live in `mesh-llm-identity/src/ownership.rs`. Owner attestations
  are on the wire as `NodeRecord.owner_attestation` and
  `PeerAnnouncement.owner_attestation` in `node.proto`. This policy governs
  admission, not which peer serves a given request.
- **Release attestation.** `ReleaseBuildAttestation`
  (`mesh-llm-membership/src/release_attestation.rs`,
  `NodeRecord.release_attestation`) proves a trusted signer signed a release
  binary. It does not prove what a remote machine is running.
- **Routing.** `handle_mesh_request` (`host-runtime/src/network/openai/transport.rs`)
  selects among `InferenceTarget::{Local, Remote}` (`mesh-llm-routing/src/lib.rs`).
  Candidates are ranked by context fit and throughput (`routing_rank.rs`), local
  health ([NODE_REP.md](../NODE_REP.md)), and, with the `payments` feature,
  price (`payment_routing.rs`). Clients can already narrow routing per request
  with the `x-mesh-target` and `x-mesh-exclude` headers (`request_parse.rs`).
  Remote requests travel as HTTP over an iroh QUIC stream (`open_http_tunnel`
  in `mesh/connections.rs`), encrypted to the peer's node key.

## Trust Tiers

| Tier | Privacy | Model integrity | Hardware |
|---|---|---|---|
| `any` | None | None beyond the self-reported digest | Any |
| `trusted` | Trust in an owner you chose | Trust in that owner | Any |
| `measured` | Running config proven; not secret from physical attackers | Measured binary checks weight bytes | Linux with TPM 2.0 |
| `tee` (later) | Strongest available | Covered by attestation | Confidential-computing CPU and GPU |
| `receipts` (later) | None | Verified by the client per response | Any |

### `trusted`

**Guarantee:** the serving node holds a valid, unexpired, unrevoked owner
attestation for an owner in the routing node's trusted set.

**Does not guarantee** anything about that owner's software or behavior. This
tier is as private as the owners are honest. It is the right default for
private meshes and for "only my machines".

### `measured`

**Guarantee:** a TPM signed a fresh quote showing that the node booted a
published, reproducible mesh-llm image whose root filesystem is verified
against a known hash. The quote is bound to the node's iroh key, so the
attested machine is the one terminating the encrypted connection. The image
enforces no shell, kernel lockdown, no persistent prompt storage, and no
logging of request content.

Non-guarantees, all of which must appear in user-facing docs:

- **Physical access.** Memory is not encrypted. An owner with physical access
  can read prompts through cold-boot attacks, DMA over PCIe or Thunderbolt, or
  hardware debug interfaces.
- **Discrete TPM interposers.** An active interposer on the bus of a discrete
  TPM can reset PCRs and replay measurements, which forges a quote entirely.
  Firmware TPMs (Intel PTT, AMD fTPM) avoid the exposed bus, but AMD fTPM has
  been broken by voltage glitching. See the TPM policy question below.
- **Runtime exploits.** Measurement happens at load time. A kernel, driver, or
  mesh-llm exploit after boot is not detected. The NVIDIA kernel driver is a
  large proprietary component inside the trusted base.
- **Firmware.** Firmware PCRs vary per machine and are not checked against
  published values, so firmware-level compromise is out of scope.

This tier stops a software-level or curious operator. It does not stop a
determined hardware attacker.

### `tee` (later)

A confidential VM (Intel TDX or AMD SEV-SNP) with an NVIDIA GPU in
confidential-computing mode. The attestation covers memory encryption as well
as the boot chain. It reuses most of the `measured` flow: quote verification,
reference values, and iroh key binding. It is deferred until operators with
qualifying data-center hardware exist.

### `receipts` (later)

Integrity for public nodes, including Macs. Each turn carries a signed
commitment. The paying client challenges one random layer and a few tokens, and
checks the openings on CPU against weight rows fetched from Hugging Face. It
provides no privacy. It is listed here only because the request policy must be
able to require it. The protocol and its lab measurements are in
[RECEIPTS.md](RECEIPTS.md).

## Model Identity as a Weight Hash

The tiers that make integrity claims need an identity based on bytes, not names.

Changes:

1. Add `optional string weights_digest = 10;` to `ServedModelIdentity` in
   `node.proto`, and gossip the value the Rust type already carries.
2. Define the **model identity for trust decisions** as the weights digest. Keep
   `identity_hash` as a naming and display aid. It never satisfies a trust
   check.
3. Resolve expected digests for Hugging Face references from the Hub's LFS
   SHA-256 metadata for the file. A client can then check that a peer serving
   `repo@revision/file` reports the published digest without downloading the
   file. Split GGUFs use a digest per part. Skippy layer packages use their
   manifest digest (`manifest_sha256` in
   `skippy/crates/skippy-protocol/proto/stage.proto`).
4. In the `measured` and `tee` tiers, the measured binary computes the digest
   from the file descriptor it actually maps for serving, which closes the
   TOCTOU gap. In `trusted` and `any`, the digest remains the operator's word.

## Request Trust Policy

### Expressing the policy

A policy is a set of acceptable tiers. A candidate satisfies the policy if it
qualifies for any tier in the set.

- **Request header:** `x-mesh-trust: trusted,measured`. This follows the
  existing `x-mesh-target` and `x-mesh-exclude` pattern and is removed before
  forwarding.
- **Node default:** `--request-trust <tiers>` or the equivalent config key.
  The default is `any`, which is today's behavior.
- **Combination:** the node default is a floor set by the operator. A header can
  only narrow it. If a header names no tier allowed by the node default, the
  request is rejected.
- **Evaluation point:** the node that performs routing evaluates the policy
  against its own trust store and verification cache. Usually that is the
  requester's own local node. A client that calls another node's API is already
  trusting that node to route honestly.

`Local` targets qualify as `trusted` when the routing node's own owner is in
its trusted set. A `Local` target qualifies as `measured` only if the routing
node itself is a measured node.

### Routing

Insert a trust filter immediately after model resolution and before any
ranking:

1. Resolve candidates for the model, or for auto routing, the candidate models.
2. **Trust filter:** drop every candidate that fails the policy. Apply it to
   `x-mesh-target` too: a pinned target that fails the policy is rejected, not
   served.
3. Context and capability eligibility (`routing_rank.rs`).
4. Local reputation ([NODE_REP.md](../NODE_REP.md)). Its "keep availability when
   every candidate is cooling" fallback operates only on the trust-filtered
   set.
5. Price ranking, when the `payments` feature is enabled.

For split serving, every stage host must satisfy the policy, because every
stage sees activations, and the first and last stages see tokens.

### Failing closed

- If no candidate satisfies the policy, return an error with code
  `trust_policy_unsatisfied` and a message naming the policy. Never fall back to
  a weaker tier.
- If a peer's tier evidence expires mid-session, the peer stops qualifying for
  new requests. In-flight requests finish.
- Unknown tier names in a header are an error, not ignored.

### Response surface

Add `x-mesh-trust-tier` to responses through `ResponseMetadata`, naming the tier
the serving node satisfied. For `measured`, add a short evidence ID (a hash of
the verified quote) so clients can log what they relied on.

## Trusted-Node Routing

This builds on what already ships in `mesh-llm-identity`.

- The trusted set is the routing node's own `owner_id` plus `trusted_owners`
  from the existing trust store. Revoked owners, certificate IDs, and node IDs
  in that store are honored.
- A peer qualifies as `trusted` when its owner attestation verifies as
  `OwnershipStatus::Verified` and its `owner_id` is in the trusted set.
- This is independent of the admission `TrustPolicy`. A mesh may admit
  unattributed peers while a given request is restricted to trusted owners.
- CLI and console surfaces show, per model, how many candidates qualify for each
  tier, so "no trusted node serves this model" is visible before a request
  fails.

## Measured Nodes

### Image requirements

A measured node boots a mesh-llm appliance image, not a general-purpose OS.

- **Reproducible build.** A NixOS image, built reproducibly so anyone can
  rebuild it from a tagged commit and get the same artifacts.
- **Unified kernel image** with kernel, initrd, and command line in one signed
  EFI binary, measured by `systemd-stub` into PCR 11. Secure Boot is
  recommended. PCR 7 records its state either way.
- **dm-verity root.** The root filesystem is read-only and verified. Its root
  hash is in the UKI command line, so it is covered by PCR 11. Mesh-llm, its
  configuration, and the GPU driver all live on the verified root.
- **Kernel hardening.** `lockdown=confidentiality`, enforced module signing,
  no debugfs, no `/proc/*/mem` or ptrace access to mesh-llm, and no eBPF for
  unprivileged users.
- **No operator shell.** No SSH, no console login, no remote management agent.
  Updates happen only by booting a newer published image.
- **No persistent request data.** No swap, or swap encrypted with a key
  generated at boot and discarded. Logs go only to tmpfs, and the image's
  logging configuration never records request or response content.
- **Writable state** (model files, cache, owner certificate) lives on a data
  partition mounted `noexec`. Nothing on it is executed or changes the trusted
  code.
- **Node key sealed to the TPM.** The iroh secret key is generated on first
  boot and sealed with a PCR policy, so it can only be unsealed by the measured
  image. This stops a relay attack, where an unmeasured machine copies the key
  from disk and forwards attestation challenges to the real node while serving
  traffic itself.

### Reference values

CI publishes for each release the expected PCR 4, 7, and 11 values and the
dm-verity root hash, signed with the release signer from
`release_attestation.rs`. Anyone can recompute them by rebuilding the image
from source, so trust does not rest on the signer alone.

Firmware PCRs (0 to 3) differ per machine and are recorded in the evidence but
not compared against published values.

### Attestation flow

1. **Registration, once per verifier-node pair.** The node sends its endorsement
   key (EK) certificate and attestation key (AK). The verifier checks the EK
   certificate chain against TPM manufacturer roots, runs
   `MakeCredential`/`ActivateCredential` to confirm the AK lives in that TPM,
   and caches the AK public key against the node's endpoint ID.
2. **Quote.** The verifier sends a fresh nonce. The node returns a TPM quote
   over the selected PCRs, signed by the AK. Its qualifying data is
   `SHA-256("mesh-llm-measured-v1" || nonce || iroh_public_key)`. The node also
   returns the boot event log.
3. **Verification.** The verifier checks:
   - the AK signature and the qualifying data, which binds the quote to this
     nonce and to the peer's live iroh key;
   - that replaying the event log reproduces the quoted PCR values;
   - that PCRs 4, 7, and 11 and the verity root hash in the log match a
     published reference set for a release it accepts;
   - the TPM clock info (`resetCount`, `restartCount`), to detect a reboot since
     the last quote;
   - optionally, that the TPM type satisfies the fTPM or discrete-TPM policy.
4. **Result.** On success the peer is `measured` until the evidence expires.

### Distribution and freshness

- **Challenge-response.** On demand, over a new protocol stream type. This is
  the strong path, used when a request requires `measured` and the cached
  evidence is older than its TTL.
- **Gossiped quotes.** For routing hints, a measured node re-quotes every 10
  minutes with qualifying data over a mesh time epoch instead of a verifier
  nonce, and attaches the quote to its `PeerAnnouncement`. Peers may use a
  gossiped quote to pre-classify candidates, but a request that requires
  `measured` triggers a challenge-response if the peer has no verified
  nonce-based quote within the TTL.
- **TTL.** Default 15 minutes for nonce-based evidence. A change in
  `resetCount` or `restartCount`, a new iroh key, or a new event-log entry for
  a PCR in the policy invalidates the cached result immediately.
- **Old nodes** never send quotes and never qualify as `measured`.

## Rollout

### Phase 1: policy and trusted tier

- Gossip `weights_digest` and add Hugging Face digest resolution.
- Add the `x-mesh-trust` header, the node default, the trust filter, fail-closed
  errors, and `x-mesh-trust-tier`.
- Implement `trusted` on top of the existing owner attestation and trust store.

### Phase 2: measured tier

- Build the NixOS appliance image and publish reference values from CI.
- Add TPM registration, quote, verification, and the new stream type.
- Add gossiped quotes and the freshness rules.
- Surface `measured` status in the CLI and console.

### Later

- `tee`, reusing the Phase 2 verification path, once qualifying hardware is
  available.
- `receipts`, as described in [RECEIPTS.md](RECEIPTS.md).

## Open Questions

- **Payments.** Paid serving already exists behind the `payments` feature. Does
  paid traffic need a tier floor by default? Paid public serving makes
  cheap-quant substitution profitable and raises the priority of `receipts`.
- **Node hardware mix.** What share of serving nodes run Linux with TPM 2.0?
  If most capacity is on Macs, `measured` covers little, and `receipts` matters
  more.
- **Macs.** macOS offers no third-party remote attestation of boot state, so
  Macs cannot be `measured`. Is that acceptable, or should Mac serving be
  limited to `trusted` and `receipts`?
- **fTPM vs discrete TPM.** Should `measured` require a firmware TPM by
  default, given discrete-TPM interposer attacks? That excludes some
  workstation boards. An alternative is to expose the TPM type and let the
  client policy decide.
- **Image scope.** Does the appliance image serve only, or also act as a
  client? A serve-only image keeps the trusted base smaller.
- **Reference-value trust.** Should clients accept only signed reference sets,
  only sets they rebuilt themselves, or either?
