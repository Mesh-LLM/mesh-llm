# `served_summary/1`: what a node served, per model, over its checkpointed range

This is the human-readable definition of the derivation whose definition document is
[`derivations/served-summary-1.json`](../derivations/served-summary-1.json). The document is what
the digest covers. This page explains it and changes nothing it says: a change of meaning is
`served_summary/2`, with a new document and a new digest.

## Identifiers

| Identifier | What it names | Where it appears |
|---|---|---|
| `d494544ca59a4d7f997e4d59f87a83670d08f555cae65fb3752614f669757260` | `served_summary/1`: the whole artifact. This is SHA-256 over the RFC 8785 bytes of `served-summary-1.json`. | A request's `derivation`, and the artifact's top-level `derivation` |
| `6eb6f1870dbcee613573c60692b79ab979e2a6188a0f4c6191d4af65a7b773b8` | `mesh.served_summary_fold/1`: only the per-model fold. This is SHA-256 over the RFC 8785 bytes of [`served-summary-fold-1.json`](../derivations/served-summary-fold-1.json). | Inside the summary, as `derivation.definition_digest`. **Never a request's `derivation`:** a request naming it is refused `derivation_unsupported`. |
| `served_summary/1` (token) | A legacy alias for the first row | Accepted on requests through 0.1.8, and refused `derivation_unsupported` from 0.1.9. It is not registered. |

## What is answered

- **The subject:** `full_history`.
- **The coverage:** an `expected_pin` naming the node's newest checkpoint.
  - The summary is a static export of that checkpoint.
  - `min_freshness`, an on-demand export, is refused `policy_declined`.
  - A pin naming an older checkpoint is refused `coverage_unsatisfiable`.
- **The gate:** it is Standard History, answered only as `share_standard_history` allows
  ([SHARING-POLICY.md](SHARING-POLICY.md)).

## The selected range

The document's top-level `selection` is the range everything in the summary is computed over: the
records under the anchor checkpoint, from leaf 0 to the checkpoint's leaf count minus one, in log
order. A record appended after the anchor never changes the summary for that anchor.

The artifact's `selection` member reports that same range, in `records`:
- `from_entry` (1, or 0 when nothing is covered);
- `to_entry`;
- `covered_entries`.

The two are one range: the definition states it, and the artifact reports it.

## Members

| Member | Reads (record fields, under the selected range) | Unit |
|---|---|---|
| `derivation`: the fold, `by_model` | `record_type` (padding is never counted), `effect.status`, and from `x-mesh-poc-v1`: `role`, `latency_ms`, `serving_provenance.model_canonical_ref`, `.quantization` and `.weights_digest`; also `x-mesh-lifecycle-v1.observation_point` | `served`. It counts only exchanges this node served, per model. A model with fewer than 5 publishes "fewer than 5" and no latency figures. |
| `adjudications_received` | `record_type`, `chain.relation`, `compute_attestation.epistemic_type`, `adjudication.half_a_capsule_id` and `.half_b_capsule_id`, and `capsule_id` | `referee_verdicts`, `source: self_held`. A verdict counts only when both it and a half it judges are under the anchor. |
| `selection` | the anchor checkpoint's `mmr_size` | `records` |

**Fixed members.** Each one is always present, and those with a value carry exactly the value the
definition states:
- `schema`: `mesh-served-summary/1`;
- `node_id`: from the checkpoint's `key_id`;
- `coverage`: the checkpoint's `root`, `mmr_size`, `log_id` and `timestamp`, and its witnesses'
  URLs;
- `no_requester_identifiers`;
- `not_a_score`.

**What it never reads:** any requester-identity field (`cross_party.initiator_ref`,
`serving_provenance.requesting_party`, `counterparty_ref`), or any release state for a particular
recipient.

## Claims

The model reference, the quantizations and the weights-digest counts are what the serving host
reported. The plugin does not measure them. Wherever they are shown, they are shown as **what the
provider states**, never as a model that ran or was verified. A checkpoint, a signature or a
witness receipt shows only that the log was not rewritten. It never shows what was computed.

## Not a score

The summary counts what was served. It ranks nothing, blends nothing across models or units, and
says nothing about who asked.
