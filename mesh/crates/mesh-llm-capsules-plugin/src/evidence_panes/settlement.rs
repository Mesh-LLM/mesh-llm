//! Settlement records in the Evidence panes: the payer node's own sealed
//! observations of `payment.lifecycle.v1`, joined to exchanges by
//! `exchange_id`.
//!
//! The plugin seals one record per lifecycle event it observed, carrying the
//! event's fields verbatim under
//! `model_attestation.compute_attestation["x-mesh-settlement-v1"]`. This
//! module only reads those records; it never computes a sum, a balance or a
//! rate, and it never reaches into the wallet.
//!
//! What one node's records can say, and what they cannot:
//!
//! - A payer book is built from this node's records as the paying side only.
//!   The provider of the same exchange keeps its book on its own node, so a
//!   payer book always reports it as `not_available`.
//! - Records this node sealed as a provider (`observed_by: "provider"`) are
//!   its own book. They never enter a payer book. The host names a provider's
//!   lifecycle events with the id of its own served exchange (the paid
//!   serving path publishes that exchange under the same id), so they join
//!   that row as its provider book (`provider_settlement`); every one is
//!   still counted (`settlement_provider_records`).
//! - An invoice with no settlement this node's wallet reported, and no final
//!   amount, is `outcome_not_reported`: the host emits no event for a payment
//!   whose outcome is uncertain (a lost wallet reply), one that failed, or an
//!   exchange that was interrupted, so the book cannot say which, and says so.
//!   It is never "unpaid".
//! - A paid request refused before any payment state (for example a payment
//!   protocol version the seller does not speak) leaves no lifecycle event at
//!   all: such a row has no payment summary, like a free exchange.
//! - The provider's side of a payment is not something the payer's events
//!   can see, so nothing about it is sent: per-peer counts carry only this
//!   node's own facts, beside `provider_book: "not_available"`.
//! - An exchange with no settlement records has no payment summary at all
//!   (`settlement: null`). That covers a free exchange, a node with payments
//!   off, and a paid request that failed before authorization (the host emits
//!   no lifecycle phase for it). None of those is "unpaid".
//! - The payer's invoice record can arrive after the first token (the host
//!   authorizes at the first canonical token), so nothing here orders
//!   settlement records against the exchange's own record.

use serde_json::{json, Value};
use std::collections::{BTreeSet, HashMap};

/// Where the plugin records the observed event.
const SETTLEMENT_BLOCK: &str = "/model_attestation/compute_attestation/x-mesh-settlement-v1";

/// The provider's book of a payer's exchange is kept on the provider's node,
/// so it is never available to this reader.
pub(super) const PROVIDER_BOOK_NOT_AVAILABLE: &str = "not_available";

/// Every invoice this node saw issued has a settlement its own wallet
/// reported, under the same payment hash and segment.
const PAYER_SETTLED: &str = "settled";
/// At least one invoice has no settlement reported by this node's wallet.
/// The payer's events cannot tell why (not yet paid, never paid, or paid and
/// not reported), so this is a fact about the payer's book, not a lapse.
const PAYER_NO_SETTLEMENT_SEEN: &str = "no_settlement_seen";
/// Terms were accepted but no invoice was recorded.
const PAYER_TERMS_ONLY: &str = "terms_only";
/// An invoice has no settlement this node's wallet reported, and the exchange
/// recorded no final amount: the host reported no outcome (uncertain after a
/// lost wallet reply, failed, interrupted, or still running). Which one, the
/// records cannot say.
const OUTCOME_NOT_REPORTED: &str = "outcome_not_reported";
/// A settlement names a payment hash no invoice of this exchange named.
const PAYER_UNMATCHED_SETTLEMENT: &str = "unmatched_settlement";

pub(super) fn settlement_block(record: &Value) -> Option<&Value> {
    record.pointer(SETTLEMENT_BLOCK)
}

/// A settlement record is identified by its block alone.
pub(super) fn is_settlement_record(record: &Value) -> bool {
    settlement_block(record).is_some()
}

fn block_str<'a>(record: &'a Value, key: &str) -> Option<&'a str> {
    settlement_block(record)?
        .get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
}

/// The settlement records this node holds, keyed by the `exchange_id` each
/// one records, in ledger order.
#[derive(Default)]
pub(super) struct SettlementIndex {
    by_exchange: HashMap<String, Vec<Value>>,
    /// Ledger order of first appearance, so unjoined ids are reported
    /// deterministically.
    order: Vec<String>,
    /// Settlement records carrying no usable `exchange_id`: nothing can join
    /// them, so they are counted rather than dropped.
    missing_exchange_id: usize,
    /// Records this node sealed as the provider of a paid exchange. They are
    /// its own book, never part of a payer book: every one is counted, and
    /// those naming an exchange id are kept by it, to join the row of the
    /// exchange this node served under that id.
    provider_records: usize,
    provider_by_exchange: HashMap<String, Vec<Value>>,
}

impl SettlementIndex {
    pub(super) fn push(&mut self, record: Value) {
        if block_str(&record, "observed_by") == Some("provider") {
            self.provider_records += 1;
            if let Some(exchange_id) = block_str(&record, "exchange_id").map(str::to_string) {
                self.provider_by_exchange
                    .entry(exchange_id)
                    .or_default()
                    .push(record);
            }
            return;
        }
        let Some(exchange_id) = block_str(&record, "exchange_id").map(str::to_string) else {
            self.missing_exchange_id += 1;
            return;
        };
        if !self.by_exchange.contains_key(&exchange_id) {
            self.order.push(exchange_id.clone());
        }
        self.by_exchange
            .entry(exchange_id)
            .or_default()
            .push(record);
    }

    pub(super) fn is_empty(&self) -> bool {
        self.by_exchange.is_empty() && self.missing_exchange_id == 0 && self.provider_records == 0
    }

    pub(super) fn provider_records(&self) -> usize {
        self.provider_records
    }

    pub(super) fn missing_exchange_id(&self) -> usize {
        self.missing_exchange_id
    }

    /// The payer-book summary for the exchange ids one row's records carry,
    /// or `None` when none of them has a settlement record.
    ///
    /// Each exchange id is its own book: one exchange's settlement never
    /// settles another's invoice. A row carrying more than one id shows the
    /// WORST of their states, with every entry, so a merged row can never
    /// read "settled" while one of its exchanges is not.
    pub(super) fn summary_for<'a>(
        &self,
        exchange_ids: impl IntoIterator<Item = &'a str>,
    ) -> Option<Value> {
        let mut seen = BTreeSet::new();
        let mut books: Vec<(&str, Value)> = Vec::new();
        for id in exchange_ids {
            if !seen.insert(id) {
                continue;
            }
            if let Some(records) = self.by_exchange.get(id) {
                let entries: Vec<&Value> = records.iter().collect();
                books.push((id, payer_book(&entries)));
            }
        }
        match books.len() {
            0 => None,
            1 => books.pop().map(|(id, mut book)| {
                book["exchange_ids"] = json!([id]);
                book
            }),
            _ => Some(worst_book(books)),
        }
    }

    /// This node's book as the provider of the exchange ids one row carries,
    /// or `None` when none of them has a provider record. A row carrying more
    /// than one id shows the worst state, with every entry.
    pub(super) fn provider_summary_for<'a>(
        &self,
        exchange_ids: impl IntoIterator<Item = &'a str>,
    ) -> Option<Value> {
        let mut seen = BTreeSet::new();
        let mut books: Vec<(&str, Value)> = Vec::new();
        for id in exchange_ids {
            if !seen.insert(id) {
                continue;
            }
            if let Some(records) = self.provider_by_exchange.get(id) {
                let entries: Vec<&Value> = records.iter().collect();
                books.push((id, provider_book(&entries)));
            }
        }
        match books.len() {
            0 => None,
            1 => books.pop().map(|(id, mut book)| {
                book["exchange_ids"] = json!([id]);
                book
            }),
            _ => {
                let mut row = worst_book(books);
                row["observed_by"] = json!("provider");
                row["who_paid"] = json!(WHO_PAID_REQUESTER);
                row["final_accounted_msat"] = Value::Null;
                Some(row)
            }
        }
    }

    /// Provider exchange ids no pane row carries.
    pub(super) fn provider_unjoined<'a>(
        &self,
        joined: impl IntoIterator<Item = &'a str>,
    ) -> Vec<String> {
        let joined: BTreeSet<&str> = joined.into_iter().collect();
        let mut ids: Vec<String> = self
            .provider_by_exchange
            .keys()
            .filter(|id| !joined.contains(id.as_str()))
            .cloned()
            .collect();
        ids.sort();
        ids
    }

    /// Settlement exchange ids no pane row carries, so the page can say the
    /// records exist instead of dropping them.
    pub(super) fn unjoined<'a>(&self, joined: impl IntoIterator<Item = &'a str>) -> Vec<String> {
        let joined: BTreeSet<&str> = joined.into_iter().collect();
        self.order
            .iter()
            .filter(|id| !joined.contains(id.as_str()))
            .cloned()
            .collect()
    }
}

/// How bad a payer-book state is, for a row that carries several exchanges:
/// the row shows the worst. A state this reader does not know ranks worst.
fn state_rank(state: &str) -> u8 {
    match state {
        PAYER_TERMS_ONLY => 0,
        PAYER_SETTLED => 1,
        PAYER_NO_SETTLEMENT_SEEN => 2,
        OUTCOME_NOT_REPORTED => 3,
        _ => 4,
    }
}

/// Several exchanges' books as one row summary: the worst state, every entry
/// in book order, every terms digest, and the exchange ids it covers.
fn worst_book(books: Vec<(&str, Value)>) -> Value {
    let mut state = PAYER_TERMS_ONLY;
    let mut matched_by_segment_only = false;
    let mut terms_digests: BTreeSet<String> = BTreeSet::new();
    let mut entries: Vec<Value> = Vec::new();
    let mut exchange_ids: Vec<&str> = Vec::new();
    for (id, book) in &books {
        let book_state = book["state"].as_str().unwrap_or_default();
        if state_rank(book_state) > state_rank(state) {
            state = match book_state {
                PAYER_SETTLED => PAYER_SETTLED,
                PAYER_NO_SETTLEMENT_SEEN => PAYER_NO_SETTLEMENT_SEEN,
                OUTCOME_NOT_REPORTED => OUTCOME_NOT_REPORTED,
                _ => PAYER_UNMATCHED_SETTLEMENT,
            };
        }
        // "No reference" describes a settled book; it carries to the row only
        // while the row still reads settled.
        matched_by_segment_only |=
            book_state == PAYER_SETTLED && book["matched_by_segment_only"].as_bool() == Some(true);
        if let Some(digests) = book["terms_digests"].as_array() {
            terms_digests.extend(digests.iter().filter_map(Value::as_str).map(str::to_string));
        }
        if let Some(rows) = book["entries"].as_array() {
            entries.extend(rows.iter().cloned());
        }
        exchange_ids.push(id);
    }
    // A merged row's final amount is not any one exchange's: it is left out
    // rather than summed.
    json!({
        "observed_by": "payer",
        "who_paid": WHO_PAID_THIS_NODE,
        "state": state,
        "terms_digests": terms_digests.into_iter().collect::<Vec<_>>(),
        "entries": entries,
        "matched_by_segment_only": state == PAYER_SETTLED && matched_by_segment_only,
        "final_accounted_msat": Value::Null,
        "provider_book": PROVIDER_BOOK_NOT_AVAILABLE,
        "exchange_ids": exchange_ids,
    })
}

/// The payer's book for one exchange, from its settlement records.
///
/// `state` compares invoices with settlements by `(segment, payment_hash)`:
/// the one identifier both wallets share, and the one the host's events carry.
/// The host copies a settlement's hash from the wallet's transaction, which
/// may not carry one; such a settlement can only be matched to its segment's
/// invoice, and the summary says so (`matched_by_segment_only`).
/// Amounts are copied from each record and never added up.
fn payer_book(entries: &[&Value]) -> Value {
    let mut invoices: BTreeSet<(u64, &str)> = BTreeSet::new();
    let mut settlements: BTreeSet<(u64, &str)> = BTreeSet::new();
    let mut hashless_settlement_segments: BTreeSet<u64> = BTreeSet::new();
    let mut terms_digests: BTreeSet<&str> = BTreeSet::new();
    let mut rows: Vec<Value> = Vec::with_capacity(entries.len());
    for record in entries {
        let Some(block) = settlement_block(record) else {
            continue;
        };
        let phase = block
            .get("phase")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let segment = block.get("segment").and_then(Value::as_u64);
        let payment_hash = block
            .get("payment_hash")
            .and_then(Value::as_str)
            .filter(|h| !h.is_empty());
        // Only the wallet's own report settles an invoice. A settlement record
        // stated by anyone else stays an entry on the row and never reads
        // "settled".
        let wallet_settlement = phase.ends_with("_settlement_observed")
            && block.get("source").and_then(Value::as_str) == Some("wallet_reported");
        match (segment, payment_hash) {
            (Some(segment), Some(hash)) if phase.ends_with("_invoice_issued") => {
                invoices.insert((segment, hash));
            }
            (Some(segment), Some(hash)) if wallet_settlement => {
                settlements.insert((segment, hash));
            }
            (Some(segment), None) if wallet_settlement => {
                hashless_settlement_segments.insert(segment);
            }
            _ => {}
        }
        if let Some(digest) = block
            .get("terms_digest")
            .and_then(Value::as_str)
            .filter(|d| !d.is_empty())
        {
            terms_digests.insert(digest);
        }
        rows.push(entry_row(record, block, phase, segment, payment_hash));
    }
    let invoice_segments: BTreeSet<u64> = invoices.iter().map(|(segment, _)| *segment).collect();
    let mut invoices_per_segment: HashMap<u64, usize> = HashMap::new();
    for (segment, _) in &invoices {
        *invoices_per_segment.entry(*segment).or_default() += 1;
    }
    let mut matched_by_segment_only = false;
    let all_invoices_settled = invoices.iter().all(|invoice| {
        if settlements.contains(invoice) {
            return true;
        }
        // A settlement with no hash can only name its segment, so it settles
        // an invoice only when that segment has exactly one: with two (a
        // retry under the same exchange id), it cannot say which was paid.
        let by_segment = hashless_settlement_segments.contains(&invoice.0)
            && invoices_per_segment.get(&invoice.0) == Some(&1);
        matched_by_segment_only |= by_segment;
        by_segment
    });
    // The payer's own total, as it recorded it (wallet amounts plus fees): one
    // record's value, copied, never computed here.
    let final_accounted_msat = entries.iter().rev().find_map(|record| {
        let block = settlement_block(record)?;
        (block.get("phase").and_then(Value::as_str) == Some("final_accounted"))
            .then(|| block.get("amount_msat").cloned())
            .flatten()
    });
    let state = if !settlements.is_subset(&invoices)
        || !hashless_settlement_segments.is_subset(&invoice_segments)
    {
        PAYER_UNMATCHED_SETTLEMENT
    } else if invoices.is_empty() {
        PAYER_TERMS_ONLY
    } else if all_invoices_settled {
        PAYER_SETTLED
    } else if final_accounted_msat.is_none() {
        OUTCOME_NOT_REPORTED
    } else {
        PAYER_NO_SETTLEMENT_SEEN
    };
    json!({
        "observed_by": "payer",
        "who_paid": WHO_PAID_THIS_NODE,
        "final_accounted_msat": final_accounted_msat.unwrap_or(Value::Null),
        "state": state,
        // One digest when every record agrees; all of them otherwise, so a
        // disagreement is shown rather than resolved here.
        "terms_digests": terms_digests.into_iter().collect::<Vec<_>>(),
        "entries": rows,
        "matched_by_segment_only": matched_by_segment_only,
        "provider_book": PROVIDER_BOOK_NOT_AVAILABLE,
    })
}

/// Who paid, as a book can say it: on a payer book, this node; on a provider
/// book, the node that requested the exchange (the page names it from the
/// row's own exchange record).
const WHO_PAID_THIS_NODE: &str = "this_node";
const WHO_PAID_REQUESTER: &str = "requester";

/// One recorded step as a book lists it: the record's own values, copied.
/// The wallet's credited amount and fee, and the delivered-token watermark,
/// appear only when the record carries them.
fn entry_row(
    record: &Value,
    block: &Value,
    phase: &str,
    segment: Option<u64>,
    payment_hash: Option<&str>,
) -> Value {
    let mut row = json!({
        "capsule_id": record.get("capsule_id").cloned().unwrap_or(Value::Null),
        "timestamp": record.get("timestamp").cloned().unwrap_or(Value::Null),
        "phase": phase,
        "source": block.get("source").cloned().unwrap_or(Value::Null),
        "segment": segment,
        "payment_hash": payment_hash,
        "amount_msat": block.get("amount_msat").cloned().unwrap_or(Value::Null),
    });
    for key in ["credited_msat", "fee_msat", "tokens"] {
        if let Some(value) = block.get(key).filter(|v| v.is_u64()) {
            row[key] = value.clone();
        }
    }
    row
}

/// This node's book as the provider of one exchange, from the records it
/// sealed as the provider: each invoice it issued, its receiving wallet's
/// report of each settlement (with what it credited and deducted, when the
/// host passed them), and the delivered-token watermark. `state` compares
/// invoices with wallet-reported settlements as the payer book does; an
/// invoice with no reported settlement is `outcome_not_reported` (a provider
/// records no final amount).
fn provider_book(entries: &[&Value]) -> Value {
    let mut invoices: BTreeSet<(u64, String)> = BTreeSet::new();
    let mut settled: BTreeSet<(u64, String)> = BTreeSet::new();
    let mut terms_digests: BTreeSet<String> = BTreeSet::new();
    let mut delivered_tokens: Option<Value> = None;
    let mut rows: Vec<Value> = Vec::with_capacity(entries.len());
    for record in entries {
        let Some(block) = settlement_block(record) else {
            continue;
        };
        let phase = block
            .get("phase")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let segment = block.get("segment").and_then(Value::as_u64);
        let payment_hash = block
            .get("payment_hash")
            .and_then(Value::as_str)
            .filter(|h| !h.is_empty());
        let wallet = block.get("source").and_then(Value::as_str) == Some("wallet_reported");
        if let (Some(segment), Some(hash)) = (segment, payment_hash) {
            if phase.ends_with("_invoice_issued") {
                invoices.insert((segment, hash.to_string()));
            } else if wallet && phase.ends_with("_settlement_observed") {
                settled.insert((segment, hash.to_string()));
            }
        }
        if phase == "delivered" {
            delivered_tokens = block.get("tokens").filter(|v| v.is_u64()).cloned();
        }
        if let Some(digest) = block
            .get("terms_digest")
            .and_then(Value::as_str)
            .filter(|d| !d.is_empty())
        {
            terms_digests.insert(digest.to_string());
        }
        rows.push(entry_row(record, block, phase, segment, payment_hash));
    }
    let state = if !settled.is_subset(&invoices) {
        PAYER_UNMATCHED_SETTLEMENT
    } else if invoices.is_empty() {
        PAYER_TERMS_ONLY
    } else if invoices.is_subset(&settled) {
        PAYER_SETTLED
    } else {
        OUTCOME_NOT_REPORTED
    };
    json!({
        "observed_by": "provider",
        "who_paid": WHO_PAID_REQUESTER,
        "state": state,
        "terms_digests": terms_digests.into_iter().collect::<Vec<_>>(),
        "entries": rows,
        "matched_by_segment_only": false,
        "final_accounted_msat": Value::Null,
        "delivered_tokens": delivered_tokens.unwrap_or(Value::Null),
    })
}

/// Per-peer settlement counts from the payer-book summaries of the peer's
/// exchanges. Counts only: no amounts, no rates, and nothing about the
/// provider's side, which this node cannot observe (`provider_book`).
pub(super) fn peer_counts(summaries: &[Value]) -> Value {
    let count = |state: &str| {
        summaries
            .iter()
            .filter(|s| s.get("state").and_then(Value::as_str) == Some(state))
            .count()
    };
    // An exchange whose terms were accepted but that never got an invoice is
    // not paid: it is counted on its own.
    json!({
        "paid_exchanges": summaries.len() - count(PAYER_TERMS_ONLY),
        "terms_only": count(PAYER_TERMS_ONLY),
        "settled_payer_observed": count(PAYER_SETTLED),
        "no_settlement_seen": count(PAYER_NO_SETTLEMENT_SEEN),
        "outcome_not_reported": count(OUTCOME_NOT_REPORTED),
        "provider_book": PROVIDER_BOOK_NOT_AVAILABLE,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(
        exchange_id: &str,
        phase: &str,
        source: &str,
        segment: Option<u64>,
        hash: Option<&str>,
        amount: u64,
    ) -> Value {
        let mut block = json!({
            "v": 1,
            "observed_by": "payer",
            "exchange_id": exchange_id,
            "event_ref": format!("{exchange_id}-{phase}"),
            "terms_digest": "t".repeat(64),
            "phase": phase,
            "source": source,
            "amount_msat": amount,
        });
        if let Some(segment) = segment {
            block["segment"] = json!(segment);
        }
        if let Some(hash) = hash {
            block["payment_hash"] = json!(hash);
        }
        json!({
            "capsule_id": format!("cap-{exchange_id}-{phase}"),
            "timestamp": "2026-09-27T00:00:00Z",
            "model_attestation": { "compute_attestation": { "x-mesh-settlement-v1": block } },
        })
    }

    fn paid_and_settled(exchange_id: &str) -> Vec<Value> {
        vec![
            event(
                exchange_id,
                "terms_accepted",
                "payer_asserted",
                None,
                None,
                900,
            ),
            event(
                exchange_id,
                "input_invoice_issued",
                "provider_asserted",
                Some(0),
                Some("aa"),
                120,
            ),
            event(
                exchange_id,
                "input_settlement_observed",
                "wallet_reported",
                Some(0),
                Some("aa"),
                120,
            ),
            event(
                exchange_id,
                "output_invoice_issued",
                "provider_asserted",
                Some(1),
                Some("bb"),
                457,
            ),
            event(
                exchange_id,
                "output_settlement_observed",
                "wallet_reported",
                Some(1),
                Some("bb"),
                457,
            ),
            event(
                exchange_id,
                "final_accounted",
                "payer_asserted",
                None,
                None,
                577,
            ),
        ]
    }

    fn index(records: Vec<Value>) -> SettlementIndex {
        let mut index = SettlementIndex::default();
        for record in records {
            index.push(record);
        }
        index
    }

    #[test]
    fn every_invoice_settled_by_this_wallet_reads_settled_with_amounts_verbatim() {
        let index = index(paid_and_settled("ex-1"));
        let summary = index.summary_for(["ex-1"]).expect("records exist");
        assert_eq!(summary["state"], "settled");
        assert_eq!(summary["observed_by"], "payer");
        assert_eq!(summary["provider_book"], "not_available");
        let entries = summary["entries"].as_array().unwrap();
        assert_eq!(entries.len(), 6);
        // Recorded values, never a computed total.
        assert_eq!(entries[3]["amount_msat"], 457);
        assert_eq!(entries[5]["amount_msat"], 577);
        assert!(summary.get("total_msat").is_none());
    }

    /// Two paid exchanges with the same request body. Exchange 1
    /// has invoice seg0 `aa` and a hash-less wallet settlement for seg0;
    /// exchange 2 has invoice seg0 `bb` and nothing paid. A row carrying both
    /// must not read settled: each id is its own book, the row shows the worst.
    #[test]
    fn one_exchange_settlement_never_settles_another_exchange_invoice() {
        let records = vec![
            event(
                "ex-1",
                "input_invoice_issued",
                "provider_asserted",
                Some(0),
                Some("aa"),
                120,
            ),
            event(
                "ex-1",
                "input_settlement_observed",
                "wallet_reported",
                Some(0),
                None,
                120,
            ),
            event(
                "ex-2",
                "input_invoice_issued",
                "provider_asserted",
                Some(0),
                Some("bb"),
                120,
            ),
        ];
        let index = index(records);
        let merged = index.summary_for(["ex-1", "ex-2"]).unwrap();
        assert_eq!(merged["state"], "outcome_not_reported");
        assert_eq!(merged["matched_by_segment_only"], false);
        assert_eq!(merged["exchange_ids"], json!(["ex-1", "ex-2"]));
        assert_eq!(merged["entries"].as_array().unwrap().len(), 3);
        // Each on its own: exchange 1 settled (by segment), exchange 2 not.
        let one = index.summary_for(["ex-1"]).unwrap();
        assert_eq!(one["state"], "settled");
        assert_eq!(one["matched_by_segment_only"], true);
        assert_eq!(
            index.summary_for(["ex-2"]).unwrap()["state"],
            "outcome_not_reported"
        );
    }

    /// The retry: a paid retry reuses the exchange id, so one
    /// book holds two seg0 invoices. A hash-less settlement cannot say which
    /// was paid, so it settles neither.
    #[test]
    fn a_hashless_settlement_under_a_retried_segment_settles_nothing() {
        let records = vec![
            event(
                "ex-r",
                "input_invoice_issued",
                "provider_asserted",
                Some(0),
                Some("aa"),
                120,
            ),
            event(
                "ex-r",
                "input_invoice_issued",
                "provider_asserted",
                Some(0),
                Some("bb"),
                120,
            ),
            event(
                "ex-r",
                "input_settlement_observed",
                "wallet_reported",
                Some(0),
                None,
                120,
            ),
        ];
        let summary = index(records).summary_for(["ex-r"]).unwrap();
        assert_eq!(summary["state"], "outcome_not_reported");
        assert_eq!(summary["matched_by_segment_only"], false);
    }

    #[test]
    fn a_row_of_settled_and_terms_only_exchanges_reads_settled() {
        let mut records = paid_and_settled("ex-a");
        records.push(event(
            "ex-b",
            "terms_accepted",
            "payer_asserted",
            None,
            None,
            5,
        ));
        let merged = index(records).summary_for(["ex-a", "ex-b"]).unwrap();
        assert_eq!(merged["state"], "settled");
    }

    #[test]
    fn a_settlement_not_reported_by_the_wallet_never_reads_settled() {
        for hash in [Some("bb"), None] {
            let mut records = paid_and_settled("ex-src");
            records[4] = event(
                "ex-src",
                "output_settlement_observed",
                "provider_asserted",
                Some(1),
                hash,
                457,
            );
            let summary = index(records).summary_for(["ex-src"]).unwrap();
            assert_eq!(summary["state"], "no_settlement_seen", "hash {hash:?}");
            // The record is still shown, as what it is.
            assert_eq!(summary["entries"][4]["source"], "provider_asserted");
        }
    }

    #[test]
    fn an_output_invoice_without_its_settlement_is_no_settlement_seen() {
        let mut records = paid_and_settled("ex-2");
        records.remove(4);
        let summary = index(records).summary_for(["ex-2"]).unwrap();
        assert_eq!(summary["state"], "no_settlement_seen");
    }

    #[test]
    fn a_settlement_under_another_segment_does_not_settle_the_invoice() {
        let records = vec![
            event(
                "ex-3",
                "input_invoice_issued",
                "provider_asserted",
                Some(0),
                Some("aa"),
                1,
            ),
            event(
                "ex-3",
                "input_settlement_observed",
                "wallet_reported",
                Some(1),
                Some("aa"),
                1,
            ),
        ];
        let summary = index(records).summary_for(["ex-3"]).unwrap();
        assert_eq!(summary["state"], "unmatched_settlement");
    }

    #[test]
    fn a_settlement_naming_no_invoice_is_unmatched_even_when_the_invoices_settled() {
        let mut records = paid_and_settled("ex-4");
        records.push(event(
            "ex-4",
            "output_settlement_observed",
            "wallet_reported",
            Some(2),
            Some("cc"),
            3,
        ));
        let summary = index(records).summary_for(["ex-4"]).unwrap();
        assert_eq!(summary["state"], "unmatched_settlement");
    }

    #[test]
    fn a_settlement_with_no_hash_settles_its_segment_and_says_how_it_matched() {
        let mut records = paid_and_settled("ex-7");
        records[2] = event(
            "ex-7",
            "input_settlement_observed",
            "wallet_reported",
            Some(0),
            None,
            120,
        );
        let summary = index(records).summary_for(["ex-7"]).unwrap();
        assert_eq!(summary["state"], "settled");
        assert_eq!(summary["matched_by_segment_only"], true);
        let full = index(paid_and_settled("ex-8"))
            .summary_for(["ex-8"])
            .unwrap();
        assert_eq!(full["matched_by_segment_only"], false);
    }

    #[test]
    fn a_hashless_settlement_for_a_segment_with_no_invoice_is_unmatched() {
        let records = vec![
            event(
                "ex-9",
                "input_invoice_issued",
                "provider_asserted",
                Some(0),
                Some("aa"),
                1,
            ),
            event(
                "ex-9",
                "input_settlement_observed",
                "wallet_reported",
                Some(0),
                Some("aa"),
                1,
            ),
            event(
                "ex-9",
                "output_settlement_observed",
                "wallet_reported",
                Some(1),
                None,
                1,
            ),
        ];
        let summary = index(records).summary_for(["ex-9"]).unwrap();
        assert_eq!(summary["state"], "unmatched_settlement");
    }

    #[test]
    fn terms_without_an_invoice_read_terms_only() {
        let records = vec![event(
            "ex-5",
            "terms_accepted",
            "payer_asserted",
            None,
            None,
            900,
        )];
        let summary = index(records).summary_for(["ex-5"]).unwrap();
        assert_eq!(summary["state"], "terms_only");
    }

    #[test]
    fn an_exchange_with_no_settlement_records_has_no_summary() {
        // Free, payments off, or failed before authorization: nothing to say,
        // and in particular never "unpaid".
        let index = index(paid_and_settled("ex-paid"));
        assert!(index.summary_for(["ex-free"]).is_none());
    }

    #[test]
    fn provider_records_are_counted_and_never_enter_a_payer_book() {
        let mut provider = event("ex-a", "terms_accepted", "provider_asserted", None, None, 0);
        provider["model_attestation"]["compute_attestation"]["x-mesh-settlement-v1"]
            ["observed_by"] = json!("provider");
        let index = index(vec![provider]);
        assert_eq!(index.provider_records(), 1);
        assert!(!index.is_empty());
        assert!(index.summary_for(["ex-a"]).is_none());
        assert!(index.unjoined([]).is_empty());
    }

    #[test]
    fn records_for_ids_no_row_carries_are_reported_as_unjoined() {
        let mut records = paid_and_settled("ex-a");
        records.extend(paid_and_settled("ex-b"));
        let index = index(records);
        assert_eq!(index.unjoined(["ex-a"]), vec!["ex-b".to_string()]);
        assert!(index.unjoined(["ex-a", "ex-b"]).is_empty());
    }

    #[test]
    fn differing_terms_digests_are_all_listed() {
        let mut records = paid_and_settled("ex-6");
        records[1]["model_attestation"]["compute_attestation"]["x-mesh-settlement-v1"]
            ["terms_digest"] = json!("u".repeat(64));
        let summary = index(records).summary_for(["ex-6"]).unwrap();
        assert_eq!(summary["terms_digests"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn peer_counts_are_counts_and_leave_provider_states_unavailable() {
        let mut records = paid_and_settled("ex-p1");
        let mut unsettled = paid_and_settled("ex-p2");
        unsettled.remove(2);
        records.extend(unsettled);
        let index = index(records);
        let summaries: Vec<Value> = ["ex-p1", "ex-p2"]
            .iter()
            .filter_map(|id| index.summary_for([*id]))
            .collect();
        let counts = peer_counts(&summaries);
        assert_eq!(counts["paid_exchanges"], 2);
        assert_eq!(counts["settled_payer_observed"], 1);
        assert_eq!(counts["no_settlement_seen"], 1);
        assert_eq!(counts["provider_book"], "not_available");
        // Nothing about the provider's side is sent, not even as null.
        for key in ["lapsed", "debt", "settled_both_books"] {
            assert!(counts.get(key).is_none(), "{key}");
        }
    }

    /// Set one field of a record's settlement block.
    fn with(mut record: Value, key: &str, value: Value) -> Value {
        record["model_attestation"]["compute_attestation"]["x-mesh-settlement-v1"][key] = value;
        record
    }

    fn as_provider(record: Value) -> Value {
        with(record, "observed_by", json!("provider"))
    }

    /// The hole an uncertain payment leaves: the host emits no event when a
    /// pay call ends without success (the outcome is uncertain after a lost
    /// wallet reply, or the payment failed), so the payer's records stop at
    /// the invoice. The book says the outcome was not reported -- never
    /// "unpaid", and never which of those it was.
    #[test]
    fn an_invoice_with_no_reported_outcome_reads_outcome_not_reported() {
        let records = paid_and_settled("ex-u")[..2].to_vec();
        let summary = index(records).summary_for(["ex-u"]).unwrap();
        assert_eq!(summary["state"], "outcome_not_reported");
        assert_eq!(summary["final_accounted_msat"], Value::Null);
        assert_eq!(summary["who_paid"], "this_node");
        assert_eq!(summary["entries"].as_array().unwrap().len(), 2);
        assert_eq!(state_rank(OUTCOME_NOT_REPORTED), 3);
    }

    /// What a completed exchange charged, as recorded: the wallet's amount and
    /// fee on each settlement, and the payer's own final amount, each copied
    /// from its record and never added up here.
    #[test]
    fn a_settled_exchange_copies_the_wallet_fee_and_the_final_amount() {
        let mut records = paid_and_settled("ex-f");
        records[2] = with(records[2].clone(), "fee_msat", json!(1));
        records[4] = with(records[4].clone(), "fee_msat", json!(2));
        let final_amount = records[5]["model_attestation"]["compute_attestation"]
            ["x-mesh-settlement-v1"]["amount_msat"]
            .clone();
        let summary = index(records).summary_for(["ex-f"]).unwrap();
        assert_eq!(summary["state"], "settled");
        assert_eq!(summary["final_accounted_msat"], final_amount);
        assert_eq!(summary["entries"][2]["fee_msat"], 1);
        assert_eq!(summary["entries"][4]["fee_msat"], 2);
        // A record without the wallet's numbers carries none.
        assert!(summary["entries"][1].get("fee_msat").is_none());
        assert!(summary["entries"][1].get("credited_msat").is_none());
    }

    /// A paid request refused before any payment state (a payment protocol
    /// version the seller does not speak) leaves no lifecycle event, so its
    /// row has no payment summary: never "unpaid", never a guess.
    #[test]
    fn a_request_refused_before_payment_has_no_payment_summary() {
        let index = index(paid_and_settled("ex-other"));
        assert!(index.summary_for(["ex-refused"]).is_none());
        assert!(index.provider_summary_for(["ex-refused"]).is_none());
    }

    /// The provider's own book, joined by the id of the exchange it served:
    /// each invoice, its wallet's report of each settlement with what it
    /// credited and deducted, and the delivered watermark. It never enters a
    /// payer book.
    #[test]
    fn the_provider_book_joins_its_served_exchange() {
        let mut records: Vec<Value> = paid_and_settled("ex-s")[..5]
            .iter()
            .cloned()
            .map(as_provider)
            .collect();
        records[2] = with(
            with(records[2].clone(), "credited_msat", json!(119)),
            "fee_msat",
            json!(1),
        );
        records.push(as_provider(with(
            event("ex-s", "delivered", "provider_asserted", None, None, 0),
            "tokens",
            json!(42),
        )));
        let index = index(records);
        assert!(index.summary_for(["ex-s"]).is_none(), "never a payer book");
        assert_eq!(index.provider_records(), 6);
        let book = index.provider_summary_for(["ex-s"]).unwrap();
        assert_eq!(book["observed_by"], "provider");
        assert_eq!(book["who_paid"], "requester");
        assert_eq!(book["state"], "settled");
        assert_eq!(book["delivered_tokens"], 42);
        assert_eq!(book["entries"][2]["credited_msat"], 119);
        assert_eq!(book["entries"][2]["fee_msat"], 1);
        assert_eq!(book["exchange_ids"], json!(["ex-s"]));
        assert_eq!(index.provider_unjoined(["ex-s"]), Vec::<String>::new());
        assert_eq!(index.provider_unjoined([]), vec!["ex-s".to_string()]);
    }

    #[test]
    fn a_provider_invoice_with_no_reported_settlement_reads_outcome_not_reported() {
        let records: Vec<Value> = paid_and_settled("ex-p")[..4]
            .iter()
            .cloned()
            .map(as_provider)
            .collect();
        let book = index(records).provider_summary_for(["ex-p"]).unwrap();
        assert_eq!(book["state"], "outcome_not_reported");
        assert_eq!(book["delivered_tokens"], Value::Null);
    }

    #[test]
    fn peer_counts_count_outcomes_not_reported() {
        let mut records = paid_and_settled("ex-c1");
        records.extend(paid_and_settled("ex-c2")[..2].to_vec());
        let index = index(records);
        let summaries: Vec<Value> = ["ex-c1", "ex-c2"]
            .iter()
            .filter_map(|id| index.summary_for([*id]))
            .collect();
        let counts = peer_counts(&summaries);
        assert_eq!(counts["paid_exchanges"], 2);
        assert_eq!(counts["outcome_not_reported"], 1);
        assert_eq!(counts["no_settlement_seen"], 0);
    }
}
