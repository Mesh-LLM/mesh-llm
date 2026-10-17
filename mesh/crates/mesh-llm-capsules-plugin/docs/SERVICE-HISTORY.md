# `service_history/1`: a node's history card and served summary, in one answer

This is the human-readable definition of the derivation whose definition document is
[`derivations/service-history-1.json`](../derivations/service-history-1.json). The document is what
the digest covers. This page explains it and changes nothing it says: a change of meaning is
`service_history/2`, with a new document and a new digest.

## Identifier

- **A request names it by the digest of its definition:**
  `6f8d6d058322b909e249e49bbeaae921abcff63b828c36c41202c75e392ba834`. That is SHA-256 over the RFC
  8785 bytes of `service-history-1.json`.
- `service_history/1` is the document's name, not a token. A request naming it is refused
  `derivation_unsupported`.
- It is not registered.

## What is answered

- **The subject:** `full_history`.
- **The coverage:** an `expected_pin` naming the node's newest checkpoint, the same rule as the
  served summary.
  - To learn that pin, ask for `history_card/1` under `min_freshness`. Its anchor is the newest
    checkpoint.
  - `min_freshness` here is refused `policy_declined`.
  - An older pin is refused `coverage_unsatisfiable`.
- **The gate:** it is Standard History, answered only as `share_standard_history` allows
  ([SHARING-POLICY.md](SHARING-POLICY.md)).
- **Answered whole or refused whole.** If either member cannot be served, the whole answer is
  refused. It is never sent with a member missing, and never filtered for the requester.

## The artifact

One JSON object, carried as its RFC 8785 octets:

| Key | Value |
|---|---|
| `projection` | the definition digest above |
| `subject` | `{"full_history": null}` |
| `anchor` | the anchor checkpoint's digest |
| `declared` | the members this artifact carries, sorted. A member is present exactly when it is declared, and no other key appears. |
| `history_card` | **required.** The `history_card/1` artifact this node serves directly for the same subject and anchor, as a JSON value. |
| `history_card_digest` | SHA-256 of the card's RFC 8785 octets |
| `served_summary` | **optional.** The direct `served_summary/1` artifact, defined in [SERVED-SUMMARY.md](SERVED-SUMMARY.md). This node always carries it. |

The same subject, anchor and definition give the same bytes for every requester. Nothing in the
artifact depends on who asks. A projection made for one recipient would be a different
definition, with its own name.

## The embedded card

- RFC 8785 serializes a nested value to the same octets as the value alone. So the card's
  canonical octets are exactly the octets of the direct `history_card/1` answer for the same
  subject and anchor, and `history_card_digest` is their SHA-256.
- The responder's envelope signs the digest of the whole artifact, so the card is the responder's
  signed statement for this request.
- To check its content, verify each checkpoint's signature under the responder's key, check that
  the list starts at the stream's first checkpoint and ends at the anchor, and check each
  consistency proof and each witness receipt. These are the checks a `history_card/1` answer gets.
- To have the card as a standalone `history_card/1` answer, ask for `history_card/1` with the same
  pin, and compare its SHA-256 with `history_card_digest`.

## Checking a Service History

1. Check the envelope: the signature, the request digest and the artifact digest.
2. Check the anchor checkpoint and the coverage.
3. Check `projection`, `subject`, `anchor`, `declared`, and that the keys are exactly the declared
   ones.
4. Check the card (above).
5. Check the served summary as a served-summary answer for the same subject and anchor
   ([SERVED-SUMMARY.md](SERVED-SUMMARY.md)).

A declared member that a verifier does not implement is reported as not checked. It is never
reported as failed, and never ignored silently.

## How it reads

- Every count names its unit: checkpoints, witness receipts, served exchanges, referee verdicts,
  records.
- Checkpoints are this log's own continuity.
- A witness receipt means that a later rewrite of the log is detectable. With no witness named,
  the History says so.
- The model, quantization and weights-digest figures are shown as what the provider states.
- Nothing in a Service History is shown as a model having run, or as a model verified, and nothing
  in it scores or ranks a node.
