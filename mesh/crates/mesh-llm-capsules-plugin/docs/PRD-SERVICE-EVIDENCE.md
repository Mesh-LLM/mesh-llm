<!-- SPDX-License-Identifier: Apache-2.0 -->
# Service evidence: Service History and evidence requests for mesh nodes

*Draft · 2026-10-12. A requirements document for this plugin: what a node
should be able to show a requester about the service it provides, what it must
not disclose, and how a requester checks it. It builds on the records,
checkpoints and evidence-request answers the plugin already produces. How
those are verified, link by link, is in
[VERIFICATION-CHAIN.md](VERIFICATION-CHAIN.md); who gets what is in
[SHARING-POLICY.md](SHARING-POLICY.md); the threat model is linked from
[THREAT-MODEL.md](THREAT-MODEL.md).*

**Ground truth.** `main` at `148311a8` (2026-10-13 00:22Z, includes #23);
release `v0.1.4` (2026-10-12 22:41Z). Status words used below:

- **live**: in a release and exercised on live hosts;
- **built**: in code and tests, not exercised live;
- **planned**: no code;
- **unverified**: stated somewhere, not checked against code.

---

## 1. Problem

A requester choosing an inference provider on a mesh can see what that
provider *says* about itself. This plugin already turns each exchange into a
signed, chained, checkpointed record and answers
`draft-mih-agent-evidence-request-00` requests in-process
(`mesh/crates/mesh-llm-capsules-plugin/src/evidence_answer.rs`), but:

- the only requester identity on the evidence path is the request's own
  `requester_id`, which SHARING-POLICY calls "a scope filter against casual
  reads, not access control";
- the record-free answers (checkpoint list, `history_card/1`,
  `served_summary/1`) are answered under every sharing tier, including `off`,
  so a node discloses cadence, model use and volume to anyone who asks
  (SHARING-POLICY, "Reading records back");
- a node's weights digest is a claim copied from the host, not measured by the
  plugin (VERIFICATION-CHAIN, "What the whole chain still does not prove"), so
  two nodes can claim the same weights and answer the same temperature-0
  request differently;
- there is no agreed statement of what evidence a paid session must produce,
  and no path for an agent outside the mesh to ask for it;
- checkpoint continuity is easy to over-read as "the model ran".

## 2. Decision

**A Service History is a new, versioned projection that embeds
`history_card/1` unchanged and declares which further members it carries.**
`history_card/1` is the derivation registered by
`draft-mih-agent-evidence-request-00` (its IANA table: checkpoints, receipts
and consistency proofs for a stream, no record content) and is served by
`evidence_answer.rs`. It is never renamed, extended in place or repurposed. The
working name for the projection is `mesh_service_history/1` (**planned**;
naming is an open question, §9).

Each member is separately declared and separately verifiable:

- the embedded `history_card/1` artifact, byte-identical to what the responder
  serves for that subject and anchor;
- optionally `served_summary/1` (**built**: per-model served counts and latency
  over the checkpointed range, "fewer than 5" below a floor of 5,
  `mesh/crates/mesh-llm-capsules-plugin/src/served_summary.rs`; not in -00's registry);
- witness receipt status;
- peer-confirmed exchange halves;
- referee verdicts;
- model, hardware and TEE claims, each labelled with who asserts it.

It never carries an aggregate, a figure blended across properties, or any
ordering of providers.

## 3. Users

| User | Wants | Gets from this document |
| --- | --- | --- |
| **Requester (Z)** choosing a provider | "Can X show what it served, under a log it can't rewrite, and will it produce evidence for my session?" | Service History; service evidence agreement; an Evidence Result per requirement |
| **Provider operator (X)** | Show verifiable service without leaking requesters, prompts, cadence or volume; control who gets what | Separate opt-in settings (§5.2); authorization gate (§5.3) |
| **Another node (Y)** asked about an exchange | Answer about its own half, in its own signed words, or refuse, without being crawled | Historical reference and live multi-party corroboration (§5.7) |
| **Outside verifier** | Check a response with its own code and keys it trusts | Test vectors; native verification of each source proof |

## 4. Non-goals

No score, rating, ranking or standing of nodes, and nothing that functions as
reputation; nothing pooled across requesters (SHARING-POLICY: every default
keys on relationship, "never on proximity, latency, or any computed
standing"). No automatic crawl of the mesh. No claim that continuity, a
witness receipt or a signature proves correct execution. No settlement
network: payment legs record observations, they do not move money.
`record-push/1` and `ledger-fetch/1` stay plugin-internal (SHARING-POLICY,
"The two mesh streams are plugin-internal").

## 5. Requirements

### 5.1 Units

- **R-1 Requested vs served kept apart.** Every count states its unit:
  exchanges this node *served*, exchanges it *requested*, checkpoints, witness
  receipts, referee verdicts. Records keep `role` ∈ `requested / served /
  conflict / unknown` ([DESIGN-split-stage-records.md](DESIGN-split-stage-records.md)
  §1). `served_summary/1` counts served records only and reads no
  requester-identity field (`served_summary.rs` module doc). No count mixes
  units.
- **R-2 Split-stage inference.** A split run is one requester⟷coordinator
  exchange today; stage nodes leave no record. When stage records exist
  (DESIGN-split-stage-records §4–§6, **planned**), the History reports per-hop
  agreement (`agree / break / gap / malformed`) and per-stage attribution only,
  and states what that design says it does not establish (correct computation,
  absence of collusion, stage independence). A withheld stage record is a
  **gap**, never agreement.
- **R-3 Checkpoints.** Coverage is a pinned checkpoint interval. Continuity is
  asserted only between endpoints a requester can verify (signed
  `cll-checkpoint` plus consistency proofs, VERIFICATION-CHAIN Links 5–6).
- **R-4 Witness receipts.** A receipt counts only after offline verification
  under the witness's key; a key pinned on first contact is labelled as such,
  distinct from a configured key (SHARING-POLICY, "Witness"). There is no
  default witness, so a default node's history card carries no receipts; the
  History says "no witness named", not "unwitnessed by failure". On the
  requester side, 0.1.4 holds no witness keys, so "no receipt in an answer is
  counted as witnessed" (CHANGELOG 0.1.4). Witnessed means only that a later
  rewrite is detectable (Link 7).
- **R-5 Peer-confirmed halves.** An exchange is two-sided only when the other
  node's own signed half is held and verified under that node's key
  (SHARING-POLICY, "The record at completion"). Push is off by default and
  needs the peer's key: `CAPSULES_PEER_KEYS`, or keys announced by the host
  (mesh-llm #2299, merged 2026-10-08, after the latest mesh-llm release
  v0.78.1; exercised only against a test build of the host, CHANGELOG 0.1.4).
  A one-sided exchange is shown as one-sided, never as a failure.
- **R-6 Model claims stay claims.** The model name digest and the weights
  digest are host-reported, not measured by this plugin. The History renders
  them as "provider states". Never "model executed", "model verified" or
  equivalent from continuity, signatures or receipts.

### 5.2 Privacy defaults and disclosure classes

Today the checkpoint list, `history_card/1` and `served_summary/1` are
answered under every `share_history_segments` value, including `off`
(SHARING-POLICY; parity cases in `evidence_request_parity.rs`). They
carry no record bodies, but they are not harmless:

- checkpoint `log_size` and `issued_at` over time give the node's **cadence
  and volume**;
- `served_summary/1` gives **model-use patterns** per model;
- the floor of 5 in `served_summary/1` protects a single answer, not a
  sequence of answers: two summaries over overlapping ranges (for example the
  range at checkpoint *n* and at *n + 1*) can be subtracted, and the
  difference can be a single exchange's model and latency.

Draft -00's own Privacy section notes that history cards leak cadence and
volume. This document therefore replaces "answered under every tier" with four
separate operator settings. Each is its own switch; none implies another.

| Setting (working names) | What it governs | Default |
| --- | --- | --- |
| `advertise_evidence_requests` | Whether the node advertises that it answers evidence requests at all (mesh capability, A2A Agent Card extension) | **off**: operator opt-in |
| `share_standard_history` | Standard Service History (checkpoints, `history_card/1`, `served_summary/1`) to a requester with no prior exchange with this node | **off** until the operator enables it |
| Protected exchange records (§5.3) | Any subject carrying record bodies | Authenticated **and** authorized requesters only; never on a stated id alone |
| `public_summary_policy` | A broad, public-safe summary any requester may read | **none**; only under an explicit policy with bounded periods and anti-differencing controls (R-9) |

- **R-7 Opt-in advertisement.** A node that has not enabled
  `advertise_evidence_requests` advertises no evidence capability. It may still
  answer a counterparty that asks, under the other settings.
- **R-8 Standard History is not public by default.** With
  `share_standard_history` off, a requester that is not an authenticated
  counterparty of this node receives a signed `policy_declined` for
  checkpoints, `history_card/1` and `served_summary/1`, uniformly (the same
  answer for every such requester, so the refusal itself discloses nothing
  per asker).
- **R-9 Public summaries need anti-differencing controls.** A broad summary
  is answered only for closed, aligned, non-overlapping periods of a fixed
  minimum length set by the policy; a request for any other interval is
  refused. Each period's summary is fixed once first produced (every later
  request gets the same bytes), the floor applies per period and per cell, and
  a cell whose value could be derived from published cells and totals is
  suppressed with it. The summary carries the policy's identifier and digest.
- **R-10 `prospective` resolved first.** SHARING-POLICY's table gives
  `prospective` (the default) the same behaviour as `counterparties`. Before
  any public A2A capability advertisement, either define what `prospective`
  adds, or rename or remove it, so that no external description relies on a
  distinction the code does not make.
- **R-11 Two different authorization cases.** The M1 gate (§5.3) remains
  release-blocking for protected records. An operator's deliberate broad public
  summary under R-9 is a distinct, recorded authorization case: the disclosure
  decision names the policy, not a principal, and it never releases record
  bodies.

### 5.3 Release-blocking authorization gate (protected records)

**Protected** = any subject carrying record bodies (`record`, `range`,
`full_history`, `correlation`, `exchange`), plus split-stage records,
settlement legs, verdict bodies and any TEE quote sidecar.

- **R-12** No protected answer leaves the responder until the **authenticated
  transport principal is bound to the requester's authorization and to the
  final disclosure decision**, and that binding is recorded with the request
  digest.
- **R-13** Until R-12 is met, the responder refuses protected subjects to any
  asker whose identity rests only on `requester_id`, except under an explicit
  operator choice of `peers`. Today the gate trusts the stated id
  (`may_serve_all` in `evidence_answer.rs`; SHARING-POLICY, "This trusts the id
  the request states").
- **R-14 Spoofed requester refused before lookup.** A request whose claimed
  principal does not match the authenticated transport principal is refused
  `not_authorized`, signed, **before** any index resolution, extending the
  existing cheapest-first order in `evidence_answer.rs`.
- Grounding (**unverified for plugin streams**): stock hosts name the node
  that asked on tunnelled inference (`requested_by_node_id`, CHANGELOG 0.1.4),
  and `plugin.proto` `ChannelMessage` has a `source_peer_id` field, but
  SHARING-POLICY states "mesh-llm does not yet tell the plugin which peer
  opened a stream". Over A2A the principal is the A2A-authenticated client,
  bound separately.

### 5.4 Invariance: artifact, envelope, projection

- **R-15 Canonical artifact.** The canonical Evidence Artifact for a
  (subject, anchor, derivation) is byte-identical whatever the requester, the
  request nonce or the transport (Mesh-native, A2A, MCP). Authorization
  decides serve whole or refuse whole; it never changes artifact bytes
  (`evidence_answer.rs`: "never filtered per asker").
- **R-16 Response envelope.** The signed response envelope is per request: it
  binds the request digest (and with it the requester's nonce) to the
  artifact digest. Replay of an envelope to a different request fails
  verification.
- **R-17 Recipient-specific projections.** A History disclosure tailored to a
  recipient (for example a Detailed History that includes records released to
  one counterparty) is a distinct, identified projection: its own derivation
  identifier, the recipient it was produced for, and the disclosure decision
  that released it. It is never presented as, or substituted for, the
  canonical artifact.

### 5.5 Service evidence agreement and temporal ordering

- **R-18 Agreement.** One signed **service evidence agreement** per bounded
  session: parties (requester and provider keys), model policy (acceptable
  model ids and weights claims, by digest), evidence requirements (by
  reference and digest), permitted spend (cap, unit, payment reference type),
  expiry, and the session bound (count or time). The provider's acceptance
  names the agreement digest. A counter-offer is a new version. Record type
  and name: **planned**.
- **R-19 Citation.** Each paid inference cites the accepted agreement's digest,
  in the requested and served halves and in each settlement terms leg. An
  exchange outside the bound, past expiry or over the cap is not covered and
  is shown so.
- **R-20 Acceptance is not spend authorization.** The provider's acceptance
  means "I will produce this evidence", not "the user authorized this spend".
  Spend authorization cites the principal's own signed authorization; without
  one the agreement is labelled an **agent-origin assertion**. Today the
  payer's spend control is local wallet configuration, not a signed
  authorization.
- **R-21 Backdated agreements.** A provider can run the inference, see the
  outcome, and then sign an agreement whose own timestamp precedes it. The
  provider's signature and timestamp cannot show order. The claim "accepted
  before the action" is made only with **temporal-ordering evidence**, at
  least one of:
  1. a signed protocol exchange: the requester's own signed request half
     cites the acceptance digest, so the request could not exist before the
     acceptance;
  2. execution records causally bound to the agreement: the served half
     cites the acceptance digest **and** the acceptance is included under a
     checkpoint that a witness, or the requester, held before the served
     record existed;
  3. an independently registered commitment: the acceptance digest
     registered with a transparency service the provider does not operate,
     with a receipt earlier than the requester-observed request.

  Without one of these the result reads "agreement present; ordering not
  established", which is `not_evaluable` for any requirement that depends on
  prior acceptance (§5.10).

### 5.6 Evidence requests: Mesh-native and A2A, one exchange

- **R-22** The same request map, outcomes (artifact / signed refusal /
  requester-recorded absence) and verification run Mesh-native
  (`evidence-request/1`) and over A2A; MCP is a third adapter. Capability
  discovery (only under R-7): an A2A Agent Card extension naming the draft,
  the supported subjects and derivations and their versions; no history and no
  per-exchange ids in discovery.
- **R-23 Wire alignment.** The Mesh carrier today is the interop profile's
  reply in its JSON binding (agent-action-capsule `spec/interop-profile-v0.md`
  §6.2, `evidence-interop/0`; `evidence_answer.rs` module doc), whereas draft
  -00's peer-to-peer binding uses the same `evidence-request/1` identifier with
  deterministic CBOR frames. One normative encoding is chosen, and Mesh either
  conforms or uses a distinct identifier (§9).
- **R-24** A2A binding: **planned** (no A2A code in this repository).

### 5.7 Cross-node corroboration

A node Y can only corroborate an exchange it took part in or observed. Y has
no account of an X–Z exchange it never saw, so asking Y about it yields
nothing but "not mine". Corroboration is therefore two separate tests, run in
this order.

**(a) Historical reference** (the first cross-node milestone).

- **R-25** Z asks X for the History of a *previous* X–Y exchange (one in
  which X's record names Y as the other side). Z then, independently, asks Y
  for Y's own half of that exchange. The result says whether **two
  independently held accounts of the same exchange agree**.
- **R-26 Join.** The two halves are joined on a common exchange reference
  bound into both signed halves: the shared client nonce
  (VERIFICATION-CHAIN: halves are joined "by the shared client nonce, never
  `exchange_id`", which each node mints independently). Never a time-window
  guess. Halves with no bound common reference are "not joinable", not
  "disagree".
- **R-27 Outcomes per Y, never merged:** halves agree; halves disagree (which
  fields); one-sided (Y holds no half); signed refusal (with its reason:
  `not_authorized`, `policy_declined`, `no_such_subject`, `deadline_unmet`,
  …); recorded absence (no answer in the waiting window); unverifiable
  artifact. A refusal is never an absence and silence never a refusal.
- **R-28 Sampling frame.** Z states why it chose each Y (X's record names Y as
  counterparty; an operator list; …). One bounded ask per Y; no crawl.
- Note: Z is not a party to the X–Y exchange, so under default sharing both X
  and Y refuse protected subjects to Z. A refusal is a valid, recorded
  outcome. Agreement can be shown only where the operators' settings release
  the halves to Z, or through a minimal-disclosure derivation (§9).

**(b) Live multi-party.**

- **R-29** Z invokes X while Y is actually present as one of: a **witness**
  (Y registers X's checkpoint, Link 7); a **participant** (Y is the routing
  node on the proxied path, which seals its own requester half joined by the
  client nonce, VERIFICATION-CHAIN; or a stage node, R-2); a **replica** (Y
  is X's twin, [TWIN-REFEREE-SELECTION.md](TWIN-REFEREE-SELECTION.md) §1); or
  an **authorized referee** (§5.8). Y's statement covers only what Y observed
  in that role.

**For both tests:**

- **R-30 Signature is not independence.** Y's answer verifying under Y's
  announced key shows that Y's key signed it, bound to Z's request digest and
  anchored to Y's own checkpoint. It does not show that Y is independent of X:
  an authenticated key does not show Y is not operated or controlled by X.
  Independence is a separate, declared claim with its own stated basis, and
  without one it reads "independence not established". No result is turned
  into a figure about X.

### 5.8 Referee and twin verdicts

- **R-31** A verdict is evidence only when (a) signed by the referee under its
  announced key, (b) associated with a request this node made of that referee
  about that exact pair of halves (`verdict_counts.rs`), and (c) the referee
  was eligible: neither twin, and serving the twins' model hash and weights
  digest (TWIN-REFEREE-SELECTION §2). The selection tier is sealed in the
  verdict (§3 there).
- **R-32** Verdict values stay four separate counts (`corroborated /
  contradicted / inconclusive / not_comparable`), each a list of verdict ids;
  "Not adjudicated" carries its reason and counts against no one. A verdict
  "proves nothing about a node's future answers" (TWIN-REFEREE-SELECTION §5).
- **R-33** A verdict in a shared History is the asker's held record, labelled
  with whose ask it was. Verdict counts are this node's own and are sent to no
  one today (`verdict_counts.rs`); disclosing them is protected (§5.3).
- Status: referee code merged (#5, #15); **no live verdict** on 0.1.4
  (CHANGELOG: "waits on 0.1.5"). Halves from Hugging Face-served models were
  refused `model_mismatch`; fixed by #23, **unreleased**. Twin-pair host
  additions are proposed in mesh-llm issue #2264 (open).

### 5.9 Payment evidence

- **R-34** Payment evidence uses `draft-mih-agent-settlement-records` legs
  (terms, payer-observed, payee-observed, delivered) only when a payment
  occurred. **Lightning first:** the plugin seals the legs with
  `ln.payment_hash` (#8; the payee leg uses mesh-llm #2169). **Live**: four
  paid exchanges across two nodes verified (CHANGELOG 0.1.4). Other payment
  reference types (for example `x402.transaction`) are **planned** adapters.
- **R-35** Each foreign payment object is verified natively: the native
  object, issuer resolution, native verification, an exact predicate, and
  `PASS / FAIL / NOT_CHECKED` per property. Carrying it by digest proves
  possession only.
- **R-36 Known gaps shown, not hidden.** Observed in two-node testing on
  0.1.4:
  1. an unroutable small trailing invoice became provider-side debt and
     blocked the payer; the block did not expire, and the payer saw **HTTP 503
     `service_unavailable`, not 402**, with no reason;
  2. the payer's book had no paid step for an invoice its wallet paid (the host
     emitted no settlement event on an interrupted paid exchange; cause not
     confirmed).

  Settlement state is derived by the verifier (one-sided / agreed /
  mismatched); "outcome not reported" is never rendered "unpaid" (CHANGELOG
  0.1.4).

### 5.10 Evidence Result

- **R-37** Each requirement yields `met`, `not_met` or `not_evaluable`. `met`
  and `not_met` only when the evidence is sufficient; a gap, insufficient or
  unknown evidence gives `not_evaluable`, shown as "not established from the
  available evidence", with the follow-up request that would settle it. Mesh
  cases that are `not_evaluable`, never `not_met`: a node declining a
  protected request; a twin pair "Not adjudicated"; a one-sided exchange; a
  history card with no receipts; an agreement with no ordering evidence
  (R-21).

### 5.11 TEE and execution evidence (an optional producer)

- **R-38** TEE evidence is its own property, never implied by continuity or
  signatures. The record already has a `tee_attestation` slot, written empty
  today (`producer/capsule.rs`, `capsule_emit.rs`). How a quote binds to a
  record is a design still to be proposed here; until then the slot stays
  empty and the page never says "attested".
- Status: **planned** (wire change, tag namespace and unprivileged quote access
  are open, §9).

### 5.12 Lifecycle links

- **R-39** `amends` / `supersedes` only by the issuer of the earlier record (a
  provider correcting its own served half or terms leg); `disputes` /
  `contradicts` by any party (a requester disputing delivered tokens, a
  referee's contradicted verdict); `resolves` cites the resolving process. A
  non-issuer `supersedes` is invalid. The History shows the original claim
  and the latest supported state.

## 6. Milestones and sequencing

M1, M2 and the invariant-artifact work run **in parallel**. TEE evidence,
split-stage instrumentation and advanced referee topologies are valuable
evidence producers, but **none is a prerequisite** for ordinary evidence
interoperability (M1–M5).

| | Milestone | Content | Exit gate |
| --- | --- | --- | --- |
| **M0** | Current: 0.1.4 | **Live**: signed chained records, checkpoints, in-process `evidence-request/1` (six subjects, `history_card/1`, `served_summary/1`), signed refusals, policy gate on the stated id; paid two-node exchanges with settlement legs on both sides. **Not live**: referee verdict, host key announcement on a stock release, TEE in records | Done; R-36 gaps carried as open issues |
| **M1** | Authenticated requester binding and privacy defaults (**release-blocking** for protected records and for any advertisement) | R-7–R-14; principal recorded with the request digest; spoofed `requester_id` refused before lookup; `prospective` resolved | T3 and T7 pass on two live nodes |
| **M2** | Standard Service History | `mesh_service_history/1` (name open) embedding `history_card/1` byte-identical; declared members; requested/served units | T6 passes; R-6 copy rules pass review |
| **M2-I** | Invariant artifact (in parallel with M1, M2) | R-15–R-17; one normative encoding (R-23); projection identifiers | T8 passes |
| **M3** | Historical reference corroboration (first cross-node milestone) | R-25–R-28, R-30; nonce join | T1 passes |
| **M4** | Service evidence agreement and payment legs | R-18–R-21 on paid sessions; Lightning legs cite the agreement; R-36 gaps fixed or labelled | Two-node paid session under one agreement; over-cap exchange shown uncovered; T9 passes |
| **M5** | A2A binding and independent verifier | A2A discovery (under R-7) and requests; a verifier outside this codebase | T2 passes |
| later | Evidence producers (not prerequisites) | Live multi-party topologies (R-29), live referee verdicts (0.1.5), split-stage records, TEE property | T4; TEE checks fail closed on a tampered quote |

## 7. Acceptance tests

- **T1 — Historical reference.** Z asks X for its record of an earlier X–Y
  exchange, then asks Y for Y's half. **Pass only if** (a) both halves verify
  under keys from Z's trust policy, (b) they are joined by the client nonce
  bound in both, with a fixture pair that matches only by time window yielding
  "not joinable", (c) a fixture where Y refuses yields a signed refusal, not
  absence, and silence yields recorded absence, and (d) the result states
  independence separately (R-30).
- **T2 — Outside verifier.** A developer without this project's client or
  verifier libraries verifies a `history_card/1` answer, a Service History, a
  record answer with inclusion proof and a signed refusal; keeps units apart;
  returns the same Evidence Results. Fixtures: positive, tampered artifact,
  wrong request digest, forged refusal, declined, missing evidence,
  unsupported version. **Pass only if** each yields its specified result.
- **T3 — Spoofed requester refused before lookup.** Node A sends a
  protected-subject request whose claimed principal names B. **Pass only if**
  the answer is a signed `not_authorized` bound to the request digest, issued
  before index resolution (instrumented: no ledger read), and the same request
  from B over its own authenticated transport is answered.
- **T4 — Live multi-party.** Z invokes X with Y in one of the R-29 roles.
  **Pass only if** Y's statement verifies under Y's key alone, binds Z's
  request digest, anchors to Y's own checkpoint, is unchanged if X is offline
  or hostile, and covers only what Y observed in that role.
- **T5 — Continuity is not execution.** Rendered History, Evidence Result text
  and A2A artifact descriptions are checked against a deny-list ("model
  executed", "verified model", "attested" without a chain check, "ran on"). A
  fixture pair with the same claimed weights digest and different temperature-0
  answers must render both claims as "provider states", with no inference that
  either ran those weights.
- **T6 — `history_card/1` preserved.** The card embedded in a Service History
  is byte-identical to the direct `history_card/1` answer for the same subject
  and anchor, and a requester that knows only -00 can verify it.
- **T7 — Privacy defaults.** On a node with default settings: no evidence
  capability is advertised; checkpoints, `history_card/1` and
  `served_summary/1` to a non-counterparty are refused `policy_declined`,
  identically for every such asker. With a public summary policy on: a request
  for a non-aligned or overlapping interval is refused, and the summaries for
  two adjacent periods cannot be combined to isolate one exchange (fixture: a
  period with exactly one exchange of a model).
- **T8 — Invariance.** The same (subject, anchor, derivation) requested by two
  requesters, with different nonces, over two transports, yields
  byte-identical artifacts and two envelopes, each binding its own request
  digest; an envelope moved to the other request fails verification; a
  recipient-specific projection carries its own identifier and recipient.
- **T9 — Backdated agreement (negative fixture).** An agreement whose stated
  time precedes the exchange, but which the requester's request half does not
  cite, which first appears under a checkpoint later than the served record,
  and which has no earlier independent receipt. **Pass only if** the result
  reads "ordering not established" and every requirement that depends on
  prior acceptance is `not_evaluable`. The matching positive fixture (request
  half cites the acceptance digest) yields "accepted before the action".

## 8. Dependencies and status (checked 2026-10-12)

| Dependency | State | Source |
| --- | --- | --- |
| 0.1.4 release | **Released** 2026-10-12 22:41Z (#22) | releases |
| #23 (model ids vs weights; unblocks Hugging Face twins) | Merged 2026-10-13 00:22Z, **unreleased** | #23; CHANGELOG "Unreleased" |
| #8 settlement legs; #5, #15 referee | Merged | PRs |
| `evidence-request` 0.2.0 (the interop profile's envelope; was `capsule-emit-evidence-request` 0.0.3), `capsule-emit` 0.0.8, `checkpointed-local-log` 0.3.1 | Pinned `=0.2.0` from crates.io; one CLL line, no evidencebook | #21, #32; CHANGELOG "Unreleased" |
| mesh-llm #2138 (exchange names both nodes) | Merged 2026-10-03 | README |
| mesh-llm #2169 (wallet credited/fee on settlement) | Merged 2026-10-03 | mesh-llm #2169 |
| mesh-llm #2299 (plugin key announced, bound to node key) | Merged 2026-10-08, **after v0.78.1** (latest release, 2026-10-05) | mesh-llm #2299; releases |
| mesh-llm #2264 (twin-pair host additions) | Open issue | mesh-llm #2264 |
| Authenticated peer on plugin streams | **Not provided** (SHARING-POLICY); `ChannelMessage.source_peer_id` exists, semantics **unverified** | `plugin.proto` |
| Split-stage host seam | Design only | DESIGN-split-stage-records |
| Agreement record type, evidence profile, test vectors | **Planned** | this document |
| A2A binding | **Planned**; no code here | repository tree |
| Non-Lightning payment adapters | **Planned** | R-34 |
| TEE slot fill | **Planned**; slot written empty | `producer/capsule.rs` |
| Debt block / 503-not-402; payer book missing a paid step | Open; observed in two-node testing; no upstream issue yet | R-36 |

## 9. Open technical questions

1. **Requester authentication on plugin streams.** Should the host pass the
   authenticated source peer id on inbound plugin mesh streams (as it names
   `requested_by_node_id` for tunnelled inference), with the plugin binding it,
   recording it with the request digest and refusing mismatches before lookup?
   What does `ChannelMessage.source_peer_id` carry today?
2. **Settings shape.** Do the four settings of §5.2 join the existing
   `config_schema` switches, and how does `share_standard_history` relate to
   `share_history_segments`?
3. **`prospective`.** Define what it adds over `counterparties`, or rename or
   remove it (R-10).
4. **Public summary periods.** Period length, alignment, per-cell floor and
   complementary suppression for R-9; whether to drop latency figures from
   public summaries entirely.
5. **Name and registration** of the Service History projection, and whether
   `served_summary/1` is registered in a -01 derivation registry (it is served
   today but not in -00's IANA table).
6. **Wire encoding** of `evidence-request/1`: plugin-internal JSON vs -00's
   deterministic CBOR peer binding under the same identifier (R-23).
7. **Minimal-disclosure corroboration.** A derivation returning only a half's
   digest, its bound client nonce and an inclusion proof, so that historical
   reference (§5.7a) can work without releasing record bodies to a
   third-party requester. Can the `exchange` subject be addressed by client
   nonce?
8. **Ordering evidence.** Which of R-21's three forms is the default for paid
   sessions, and should the requester's request half always cite the
   acceptance digest?
9. **Streamed responses.** `agent_output_digest` binds only terminal facts on
   the streamed path ([PROTOCOL-NOTE.md](PROTOCOL-NOTE.md)); the History must
   say which exchanges have a response-body binding.
10. **Debt-block surfacing.** Should a provider-side debt block surface as a
    signed refusal or a 402 with a reason, so the requester's evidence shows
    why service stopped, and should the block expire (R-36)?
11. **TEE binding** (if pursued): tag namespace, report-data layout, and quote
    access without root privileges.
12. **Stale comment.** `evidence_answer.rs` says "this node has no referee
    yet" for the `adjudicate` request, though referee code exists on its own
    path. Doc fix only.

## Sources checked

- This repository at `148311a8`: [SHARING-POLICY.md](SHARING-POLICY.md),
  [THREAT-MODEL.md](THREAT-MODEL.md),
  [VERIFICATION-CHAIN.md](VERIFICATION-CHAIN.md),
  [TWIN-REFEREE-SELECTION.md](TWIN-REFEREE-SELECTION.md),
  [DESIGN-split-stage-records.md](DESIGN-split-stage-records.md),
  [PROTOCOL-NOTE.md](PROTOCOL-NOTE.md), `CHANGELOG.md`,
  `mesh/crates/mesh-llm-capsules-plugin/src/evidence_answer.rs`, `served_summary.rs`,
  `verdict_counts.rs`, `producer/capsule.rs`, `capsule_emit.rs`,
  `evidence_request_parity.rs`; PRs #1–#23; releases v0.1.3, v0.1.4.
- `Mesh-LLM/mesh-llm`: `plugin.proto`; PRs #2138, #2169, #2299; issue #2264;
  issue #1331; release v0.78.1.
- `capsule-emit-evidence-request` 0.0.3 on crates.io.
- `draft-mih-agent-evidence-request-00` (Caller Invariance, History Card,
  Transport Bindings, Privacy, IANA) and
  `draft-mih-agent-settlement-records-00`, IETF Datatracker.
