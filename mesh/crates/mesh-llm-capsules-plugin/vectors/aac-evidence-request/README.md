# Evidence request conformance vectors (draft-mih-agent-evidence-request-00)

Conformance vectors for the evidence request interaction defined in
`spec/draft-mih-agent-evidence-request-00.md`: the request map, the six subject
forms, coverage, the request digest, the signed refusal, the three outcomes and
the pending state, caller invariance, and retention commitments.

**Derived from the -00 text, not from any implementation.** Every expected
result is written by hand in the generator from the section it cites. The
generator only encodes inputs, hashes them and signs the refusal cases; it runs
no responder or verifier. Every case is marked `"provenance": "spec-derived"`.
Where -00 leaves a choice open, the case carries an `ambiguity` note with the
section and the reading used (the most literal one); the list is below, for -01.

## Files

| File | What it pins |
|---|---|
| `registry.json` | The exact strings -00 fixes: request fields and their presence, the six subject forms, the two coverage members, the eight refusal reasons, the `history_card/1` derivation, the three outcomes, the pending state, the `evidence-request/1` subprotocol identifier. |
| `request.json` | Request map validation. Each case gives the request as JSON (RFC 8785 text, `request_json`) and, where it is a CBOR value, as a deterministic CBOR frame (`request_cbor_hex`), with the SHA-256 of each. Expect `well_formed: true`, or `false` with the exact refusal reason. |
| `resolution.json` | Exact-match resolution against one declared responder state (`responder`, with per-case `responder_overrides`). Expect `artifact` or a refusal with the most specific reason (§4.2). |
| `digest.json` | The request digest is over the bytes as received (§4.2 item 1): JSON and CBOR, deterministic and not, and a refusal carrying the digest of a re-canonicalized form. |
| `refusal.json` | Signed refusals: one per registered reason, a CBOR response frame, tampering, and objects that are signed but not conforming. Carries the signature profile and both fixed keys (seed and public key). |
| `refusal-imported.json` | An existing cross-implementation refusal vector, byte for byte (SHA-256 `885d7981…a66e`). It conforms to -00: a request digest, a registered reason, an issuance time, and a signature over all three. |
| `outcomes.json` | The three outcomes and pending, as facts in and one state out, with the states each case must never be recorded as (§4.5). |
| `invariance.json` | Caller invariance (§5): observations of the same subject under the same anchor must carry byte-identical artifacts. |
| `retention.json` | Retention commitments (§9): every reason inside and after `until`, absence with and without a commitment, and which subject forms a commitment may name. |
| `manifest.json` | SHA-256 and case count of every generated file, and the ambiguity list. |
| `SHA256SUMS` | SHA-256 of every file in this directory except itself. |
| `DIFFS.md` | Known differences between one existing Python responder and -00. |

## Conventions

- Digests are SHA-256, 64 lowercase hex characters, as text in both bindings
  (**A2**).
- A subject is a map with one member named by the form token (**A1**):
  `{"full_history": null}`, `{"checkpoints": null}`, `{"record": <digest>}`,
  `{"range": [a, b]}`, `{"correlation": "<id>"}`, `{"exchange": <digest>}`.
- Coverage is `{"expected_pin": <digest>}` or `{"min_freshness": <size or time>}`
  (**A3**).
- Refusals use the signature profile in `refusal.json` (**A7**): Ed25519 over
  the RFC 8785 serialization of `{issued_at, reason, request_digest}`, with
  `key_id` (raw public key) and `sig` in lowercase hex. The keys are derived
  from fixed seeds, so every signature is reproducible.
- `issued_at`, `deadline` and `min_freshness` times are RFC 3339 UTC strings
  (**A5**).
- Resolution cases expect the most specific reason (§4.2). The one case with a
  uniform `policy_declined` posture (§4.2, §12) says so in `responder_overrides`.

## How to consume them

1. Check the pinned hashes first: `sha256sum -c SHA256SUMS` in this directory,
   or compare each file with `manifest.json`. A copy vendored into another
   repository should be checked against this `SHA256SUMS` in its CI.
2. `request.json`: feed `request_json` (UTF-8 bytes) to a JSON-binding parser,
   and the bytes of `request_cbor_hex` to a CBOR-binding parser. Both must give
   the expected result. The digest of the bytes fed must equal
   `request_json_digest` / `request_cbor_digest`.
3. `resolution.json`: load `responder` (plus any `responder_overrides`) as the
   responder's holdings, and check the answer's form and reason.
4. `refusal.json`: verify each `refusal` object (or the `file`) offline.
   `expect.signature` is the signature check alone, `expect.reason_registered`
   the registry check alone, and `expect.conformant` both plus the presence and
   form of every member §4.2 requires. Keep the two checks separate: a valid
   signature over an unregistered token is still the responder's signed act.
5. `outcomes.json`: map `facts` to a state; it must equal `expect.state` and
   must not be any of `expect.must_not_record_as`.
6. `invariance.json`: for observations with the same `resolved_anchor`, the
   `artifact_hex` values must be identical. Differing verification material is
   allowed. Different anchors are `not_comparable`.
7. `retention.json`: classify each `facts` pair, and check each `commitment` for
   well-formedness.

Regenerate from the repository root with
`python3 python/scripts/generate_evidence_request_vectors.py`. The output is a
pure function of that script (no clock, no randomness);
`python/tests/test_evidence_request_vectors.py` re-runs it and requires
byte-identical files. `README.md` and `DIFFS.md` are hand-written; the
generator hashes them into `SHA256SUMS` but never rewrites them.

## Cases

### `request.json` (33 cases)

| Case | § | Expect | Ambiguity |
|---|---|---|---|
| `pos-subject-full_history` | 3, 3.1, 3.2 | well formed | A1 |
| `pos-subject-checkpoints` | 3, 3.1, 3.2 | well formed | A1 |
| `pos-subject-record` | 3, 3.1, 3.2 | well formed | A1 |
| `pos-subject-range` | 3, 3.1, 3.2 | well formed | A1, A3 |
| `pos-subject-correlation` | 3, 3.1, 3.2 | well formed | A1, A3 |
| `pos-subject-exchange` | 3, 3.1, 3.2, 6 | well formed | A1 |
| `pos-nonce-absent` | 3, 3.5 | well formed |  |
| `pos-all-optional-fields` | 3, 3.3, 3.4, 3.5, 3.6 | well formed | A5 |
| `pos-unknown-field-ignored` | 3 | well formed |  |
| `pos-derivation-history-card` | 3.3, 8.1, 13 | well formed |  |
| `pos-derivation-by-digest` | 3.3 | well formed | A14 |
| `neg-not-a-map` | 3 | refuse `request_malformed` |  |
| `neg-not-well-formed` | 3 | refuse `request_malformed` |  |
| `neg-subject-missing` | 3 | refuse `request_malformed` |  |
| `neg-subject-unknown-form` | 3, 3.1 | refuse `request_malformed` | A1 |
| `neg-subject-two-forms` | 3, 3.1 | refuse `request_malformed` | A1 |
| `neg-subject-kind-member` | 3, 3.1 | refuse `request_malformed` | A1 |
| `neg-subject-none-form-with-content` | 3, 3.1 | refuse `request_malformed` | A1 |
| `neg-record-digest-malformed` | 3, 3.1 | refuse `request_malformed` | A2 |
| `neg-record-digest-prefix` | 3, 3.1 | refuse `request_malformed` | A2 |
| `neg-range-reversed` | 3, 3.1 | refuse `request_malformed` | A6 |
| `neg-range-not-a-pair` | 3, 3.1 | refuse `request_malformed` | A1 |
| `neg-derivation-not-text` | 3, 3.3 | refuse `request_malformed` | A14 |
| `neg-expected-pin-malformed-digest` | 3, 3.2 | refuse `request_malformed` | A4 |
| `neg-expected-pin-map` | 3, 3.2 | refuse `request_malformed` | A4 |
| `neg-min-freshness-max-age` | 3, 3.2 | refuse `request_malformed` | A3, A4 |
| `neg-coverage-not-a-map` | 3, 3.2 | refuse `request_malformed` | A4 |
| `neg-coverage-both` | 3, 3.2 | refuse `coverage_unsatisfiable` |  |
| `neg-coverage-neither` | 3, 3.2 | refuse `coverage_unsatisfiable` |  |
| `neg-coverage-missing` | 3, 3.2 | refuse `coverage_unsatisfiable` | A4 |
| `neg-coverage-both-pin-malformed` | 3, 3.2 | refuse `request_malformed` | A4 |
| `neg-coverage-both-freshness-malformed` | 3, 3.2 | refuse `request_malformed` | A3, A4 |
| `neg-coverage-both-both-malformed` | 3, 3.2 | refuse `request_malformed` | A3, A4 |

### `resolution.json` (19 cases)

| Case | § | Expect | Ambiguity |
|---|---|---|---|
| `res-record-exact` | 3.1 | artifact |  |
| `res-record-not-held` | 3.1, 4.2 | refuse `no_such_subject` |  |
| `res-record-prefix` | 3.1 | refuse `request_malformed` | A2 |
| `res-correlation-exact` | 3.1 | artifact |  |
| `res-correlation-prefix` | 3.1 | refuse `no_such_subject` |  |
| `res-correlation-case-folded` | 3.1 | refuse `no_such_subject` |  |
| `res-exchange-cited` | 3.1, 6 | artifact |  |
| `res-exchange-not-cited` | 3.1, 6 | refuse `no_such_subject` |  |
| `res-range-positional` | 3.1 | artifact |  |
| `res-range-no-positional-ordering` | 3.1 | refuse `coverage_unsatisfiable` |  |
| `res-pin-unknown-anchor` | 3.2 | refuse `coverage_unsatisfiable` |  |
| `res-min-freshness-size-met` | 3.2 | artifact | A3 |
| `res-min-freshness-size-unmet` | 3.2, 12 | refuse `coverage_unsatisfiable` | A3 |
| `res-min-freshness-time-met` | 3.2 | artifact | A3 |
| `res-min-freshness-time-unmet` | 3.2 | refuse `coverage_unsatisfiable` | A3 |
| `res-derivation-unsupported` | 3.3 | refuse `derivation_unsupported` |  |
| `res-derivation-not-for-subject` | 3.3 | refuse `no_such_subject` |  |
| `res-derivation-history-card` | 3.3, 8.1 | artifact |  |
| `res-uniform-policy-declined` | 4.2, 12 | refuse `policy_declined` |  |

### `digest.json` (7 cases)

| Case | § | Expect | Ambiguity |
|---|---|---|---|
| `digest-json-jcs` | 4.2, 3.8 | digest 8f5e0250575f… | A2 |
| `digest-json-as-received` | 4.2 | digest 5d5fd71bef55… | A2 |
| `digest-cbor-deterministic` | 4.2, 3.8, 10 | digest 0a4b2bbcd793… | A2 |
| `digest-cbor-as-received` | 4.2, 10 | digest a6fca795b66e… | A2 |
| `digest-nonce-distinguishes` | 3.5, 4.2 | digest 109c87c4ee97… | A2 |
| `neg-refusal-digest-recanonicalized` | 4.2 | does not identify | A2 |
| `pos-refusal-digest-as-received` | 4.2 | identifies | A2 |

### `refusal.json` (21 cases)

| Case | § | Expect | Ambiguity |
|---|---|---|---|
| `pos-reason-not_authorized` | 4.2, 13 | conformant (sig valid) | A7 |
| `pos-reason-no_such_subject` | 4.2, 13 | conformant (sig valid) | A7 |
| `pos-reason-coverage_unsatisfiable` | 4.2, 13 | conformant (sig valid) | A7 |
| `pos-reason-derivation_unsupported` | 4.2, 13 | conformant (sig valid) | A7 |
| `pos-reason-policy_declined` | 4.2, 13 | conformant (sig valid) | A7 |
| `pos-reason-deadline_unmet` | 4.2, 13 | conformant (sig valid) | A7 |
| `pos-reason-request_malformed` | 4.2, 13 | conformant (sig valid) | A7 |
| `pos-reason-retention_expired` | 4.2, 13 | conformant (sig valid) | A7 |
| `pos-imported-interop-vector` | 4.2, 13 | conformant (sig valid) | A7 |
| `pos-cbor-frame` | 4.2, 10 | conformant (sig valid) | A7 |
| `neg-tamper-reason` | 4.2 | not conformant (sig invalid) | A7 |
| `neg-tamper-issued_at` | 4.2 | not conformant (sig invalid) | A7 |
| `neg-tamper-request_digest` | 4.2 | not conformant (sig invalid) | A7 |
| `neg-tamper-key_id` | 4.2, 11 | not conformant (sig invalid) | A7 |
| `neg-unsigned` | 4.2 | not conformant (sig invalid) | A7 |
| `neg-unregistered-reason-no_such_record` | 4.2, 13 | not conformant (sig valid) | A13 |
| `neg-reason-recorded_absence` | 4.2, 4.5, 13 | not conformant (sig valid) | A13 |
| `neg-reason-wrong-case` | 4.2, 13 | not conformant (sig valid) | A13 |
| `neg-two-reasons` | 4.2 | not conformant (sig valid) | A13 |
| `neg-missing-issued_at` | 4.2 | not conformant (sig valid) | A7 |
| `neg-malformed-request_digest` | 4.2 | not conformant (sig valid) | A2, A7 |

### `outcomes.json` (13 cases)

| Case | § | Expect | Ambiguity |
|---|---|---|---|
| `out-artifact-verified` | 4, 4.1, 4.6 | `artifact_verified`; never `refusal`, `recorded_absence`, `pending` |  |
| `out-artifact-verification-failed` | 4.1, 4.5 | `artifact_verification_failed`; never `artifact_verified`, `recorded_absence`, `refusal` |  |
| `out-refusal-signed` | 4.2, 4.5 | `refusal`; never `recorded_absence`, `pending`, `artifact_verified` |  |
| `neg-refusal-as-absence` | 4.5 | `refusal`; never `recorded_absence` |  |
| `out-refusal-over-http` | 10, 4.5 | `refusal`; never `recorded_absence` |  |
| `out-pending` | 4.3, 4.5 | `pending`; never `recorded_absence`, `refusal` |  |
| `out-absence` | 4.4, 4.5 | `recorded_absence`; never `refusal`, `pending` |  |
| `neg-timeout-as-refusal` | 4.5 | `recorded_absence`; never `refusal` |  |
| `neg-transport-error-as-refusal` | 10, 4.5 | `recorded_absence`; never `refusal` |  |
| `out-transport-error-inside-window` | 4.3, 10 | `pending`; never `recorded_absence`, `refusal` | A8 |
| `neg-subprotocol-not-offered` | 10 | `recorded_absence`; never `refusal` |  |
| `neg-commitment-makes-refusal` | 4.4, 4.5, 9 | `recorded_absence`; never `refusal` |  |
| `neg-unregistered-reason-as-absence` | 4.2, 4.5 | `refusal_nonconformant`; never `recorded_absence`, `refusal` | A13 |

### `invariance.json` (6 cases)

| Case | § | Expect | Ambiguity |
|---|---|---|---|
| `inv-same-pin-byte-identical` | 5, 3.5, 3.6 | holds | A12 |
| `inv-verification-material-may-differ` | 5 | holds | A12 |
| `inv-min-freshness-same-selected-anchor` | 5 | holds | A12 |
| `inv-different-anchor-not-comparable` | 5 | not_comparable | A12 |
| `neg-inv-divergent-artifacts` | 5 | violated | A12 |
| `neg-inv-nondeterministic-serialization` | 5 | violated | A12 |

### `retention.json` (25 cases)

| Case | § | Expect | Ambiguity |
|---|---|---|---|
| `ret-inside-not_authorized` | 9 | no_breach |  |
| `ret-inside-no_such_subject` | 9 | breach |  |
| `ret-inside-coverage_unsatisfiable` | 9 | no_breach |  |
| `ret-inside-derivation_unsupported` | 9 | no_breach |  |
| `ret-inside-policy_declined` | 9 | breach |  |
| `ret-inside-deadline_unmet` | 9 | breach |  |
| `ret-inside-request_malformed` | 9 | no_breach |  |
| `ret-inside-retention_expired` | 9 | breach | A9 |
| `ret-after-not_authorized` | 9, 4.2 | no_breach | A10 |
| `ret-after-no_such_subject` | 9, 4.2 | wrong_token | A10 |
| `ret-after-coverage_unsatisfiable` | 9, 4.2 | no_breach | A10 |
| `ret-after-derivation_unsupported` | 9, 4.2 | no_breach | A10 |
| `ret-after-policy_declined` | 9, 4.2 | no_breach | A10 |
| `ret-after-deadline_unmet` | 9, 4.2 | no_breach | A10 |
| `ret-after-request_malformed` | 9, 4.2 | no_breach | A10 |
| `ret-after-retention_expired` | 9, 4.2 | correct |  |
| `ret-absence-inside` | 9, 4.4 | attributable_absence |  |
| `ret-absence-without-commitment` | 4.4 | attempt_only |  |
| `ret-commitment-subject-record` | 9, 3.1 | well formed | A1 |
| `ret-commitment-subject-exchange` | 9, 3.1 | well formed | A1 |
| `ret-commitment-subject-range` | 9, 3.1 | well formed | A1, A11 |
| `ret-commitment-subject-correlation` | 9, 3.1 | malformed | A1, A11 |
| `ret-commitment-subject-full_history` | 9, 3.1 | malformed | A1, A11 |
| `ret-commitment-subject-checkpoints` | 9, 3.1 | malformed | A1, A11 |
| `ret-commitment-missing-until` | 9 | malformed | A1 |

## Ambiguities in -00 (for -01)

Each is marked on the cases it affects. The vectors follow the reading given.

- **A1** §3.1 Subject: -00 gives six forms and their content but no wire shape. Vectors use a map with exactly one member, named by the form token, whose value is the content: null for full_history and checkpoints, a digest for record and exchange, [a, b] (two unsigned integers) for range, a text string for correlation.
- **A2** §3.1, §3.2, §4.2: -00 names digests but not the hash algorithm or representation. Vectors use SHA-256 as 64 lowercase hex characters, as text in both the JSON and CBOR bindings.
- **A3** §3.2 Coverage: min_freshness is 'size or time' with no shape. Vectors read the value itself: an unsigned integer is a log size, a text string is an RFC 3339 UTC time. Any other type is not a conforming value.
- **A4** §3 / §3.2: coverage_unsatisfiable is assigned for 'both, or neither'. Vectors read an absent coverage field as 'neither' (coverage_unsatisfiable), and a coverage value that is not a map, or a member whose value does not conform, as request_malformed. *Amended 2026-10-06 (spec ruling):* -00 does not state which refusal wins when coverage carries both members and one of them does not conform. Ruling: request_malformed wins. A responder first checks that each member present in `coverage` has a conforming value; a non-conforming value is refused request_malformed, whether or not the other member is present. Only then is coverage carrying both members, or neither, refused coverage_unsatisfiable. The normative sentence goes into -01 §3.2; the posted -00 text is unchanged. Cases: `neg-coverage-both-pin-malformed`, `neg-coverage-both-freshness-malformed`, `neg-coverage-both-both-malformed`. (The A4 note embedded in the cases and manifest shipped in 0.6.0 is left as it was so those bytes do not change; the three new cases carry the amended text.)
- **A5** §3.4-§3.6: deadline, nonce and route have no stated type. Vectors use text strings (deadline as an RFC 3339 UTC time).
- **A6** §3.1 range: -00 does not say whether a > b is malformed. Vectors treat it as a value that does not conform (request_malformed).
- **A7** §4.2 / §10: -00 names the refusal's contents but no member names or signature format. Vectors use request_digest, reason, issued_at, key_id and sig with the signature profile in refusal.json. In the CBOR binding the same map is the frame; the signature is still over the RFC 8785 body.
- **A8** §4.3 / §10: the stream binding says absence is 'the stream closed or timed out without a response frame'; §4.3 says only the close of the waiting window resolves a request. Vectors follow §4.3: a transport failure inside the window leaves the request pending.
- **A9** §9: -00 lists retention_expired only as the correct refusal after until. Inside until, vectors apply the literal 'any other refusal reason' rule: it is a breach.
- **A10** §9: after until, -00 names only retention_expired as correct. Vectors class the four request/capability reasons, policy_declined and deadline_unmet as no breach, and no_such_subject as the wrong token (the lapsed promise must not read as never held, §4.2 item 2).
- **A11** §9: 'a digest or a position under a checkpoint'. Vectors read full_history, checkpoints and correlation subjects in a commitment as malformed.
- **A12** §4.1: the artifact envelope has no member names. Invariance cases compare artifact bytes only; the observation members are local to these vectors.
- **A13** §4.2 item 2 / §4.5: -00 does not say what a correctly signed object with an unregistered reason is. Vectors: not a conforming refusal, and still never recorded as absence.
- **A14** §3.3: a registered derivation token and the digest of a definition share one field. Vectors tell them apart by form: a token has a '/N' suffix, a digest is 64 lowercase hex.
