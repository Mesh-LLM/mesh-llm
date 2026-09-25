# Wallet boundaries and evidence

Decision record, September 2026. This captures the architectural reasoning and
fixture expectations discussed during [PR #1926](https://github.com/Mesh-LLM/mesh-llm/pull/1926),
not a live-test report or a claim that the proposed fixtures are implemented.

## Keep wallet observations separate from inference claims

The `wallet.v1` process boundary keeps a concrete wallet SDK outside the host's
payment-policy implementation. Trust boundaries: the host owns token metering,
the decode-thread output gate (host atomics only) and response bytes; the
payments engine (`mesh-llm-payments`, installed as an in-process builtin
serving `payments.v1`) owns budgets, the ledger, settlement bookkeeping and
recovery; the wallet plugin supplies wallet operations and observations.
Host-runtime links only `mesh-llm-payments-types`. The host resolves the capability rather than a particular plugin
name. See the [wallet contract](../../../crates/mesh-llm-wallet/src/contract.rs).

A separate process is not, by itself, proof of trustworthy settlement or a
sandbox. Wallet results remain reports from the configured wallet implementation.
Evidence must preserve the distinction between who made a claim and who observed
it rather than presenting every event as an independently verified fact:

- `provider_asserted`: invoice terms attributed to the inference provider, as
  observed by the payer.
- `wallet_reported`: terminal settlement reported by the wallet, not the
  inference provider's assertion that it was paid.
- `payer_asserted`: the payer's accepted terms and final accounting.

The [payer-side evidence contract](../../specs/lightning-payments.md#payer-side-evidence-hooks)
is authoritative for fields and amounts. Settlement amounts exclude fees; final
payer accounting includes successful debits and fees. Neither payment nor
matching records prove correct inference, honest token counts or proof of prefill.

## Retain lookup by payment hash

`wallet_lookup` is the contract operation for asking the configured wallet about
a particular payment hash. It supports reconciliation and a targeted evidence
question: "What does this wallet report about payment H?" It is not redundant
with balance or a bounded recent-transactions list.

A lookup response must retain its provenance. An absent result or a failed lookup
must not be promoted into successful settlement. Matching payer and provider
hashes is useful correlation between observations, not a new trust guarantee.
Wallet operations are host-invoked services, not agent-callable tools by default.

## Evidence fixture expectations

Use synthetic identities and payment hashes in committed fixtures. Cover both
the successful path and policy/recovery boundaries:

| Scenario | What the evidence should distinguish |
| --- | --- |
| Paid request | Payer and provider observations correlate by payment hash; invoice claims remain distinct from wallet settlement reports. |
| Exhausted automatic budget | Paid request is rejected with HTTP 402, not represented as a successful paid exchange. |
| Free-only policy against a paid-only route | HTTP 402 policy refusal, not a wallet outage or settlement failure. |
| Restart with a pending payment | Reconcile the existing payment without a second debit; successful terminal reconciliation releases its reservation. |
| Harness or observation failure | Preserve the failed attempt and its limitation separately from any corrected rerun; do not silently turn it into a pass. |

These are acceptance expectations, not a promise that every recovery path ends
with zero reservations. Uncertain payments must remain distinguishable from
terminal outcomes. A fully reconciled fixture should reach zero outstanding
reservations without discarding unresolved debt to obtain that result.

## Evidence gaps are explicit

The current payer lifecycle stream is best-effort, not a complete audit log.
Missing events may reflect queue drops, disconnects or process exit; restart
recovery currently emits no lifecycle events. Absence of an event is not evidence
that a payment did not happen.

Provider-side lifecycle observations are a separate follow-up, not implied by
payer events. In particular, a payer settlement event does not establish when the
provider observed incoming payment arrival or released output. Future provider
records should preserve their own source and correlation rather than being
synthesized from the payer's account.

Keep durable decisions and sanitized fixture expectations here. Keep private
machine topology, wallet backups, seeds, credentials and raw payment records out
of these notes. Payment hashes themselves are linkable metadata. A future public
validation record should identify the exact tested revision and its limitations;
results from one revision must not silently become evidence for a later head.
