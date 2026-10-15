//! This plugin's ledger index: capsule-emit's counterparty citation keys,
//! plus one key per settlement event and per adjudication record, so a
//! settlement or verdict already on the ledger is never recorded twice.
//! Every ledger this plugin opens uses [`open_ledger`].

use std::path::Path;

use serde_json::Value;

use crate::producer::capsule::{
    ADJUDICATION_ACK_REFUSED_BLOCK, ADJUDICATION_ISSUED_BLOCK, ADJUDICATION_RECEIVED_BLOCK,
    SETTLEMENT_EXTENSION_KEY,
};
use crate::producer::ledger::{counterparty_citation_keys, Ledger, LedgerError, RecoveryReport};

/// The index key of a settlement record for `event_ref`.
pub fn settlement_event_key(event_ref: &str) -> String {
    format!("{SETTLEMENT_EXTENSION_KEY}:{event_ref}")
}

/// The index key of an adjudication record: `block` is one of the
/// `ADJUDICATION_*_BLOCK` names; `subject` is the verdict's capsule id, or
/// `<verdict capsule id>/<peer>` for a refused acknowledgement.
pub fn adjudication_key(block: &str, subject: &str) -> String {
    format!("{block}:{subject}")
}

/// The index key of a settlement-records leg: one per leg kind and payment
/// (`terms`, `payer_observed`, `payee_observed` by payment hash), and one per
/// delivered leg by the terms leg it cites and its direction.
pub fn settlement_leg_key(leg: &str, subject: &str) -> String {
    format!("settlement-leg:{leg}:{subject}")
}

/// The subject of a leg's index key: its payment hash, or for a delivered leg
/// `<terms_ref>/<direction>`.
pub fn settlement_leg_subject(member: &Value) -> Option<(String, String)> {
    let leg = member.get("leg")?.as_str()?.to_string();
    let subject = if leg == "delivered" {
        format!(
            "{}/{}",
            member.get("terms_ref")?.as_str()?,
            member.pointer("/delivery/direction")?.as_str()?
        )
    } else {
        member.pointer("/payment_ref/value")?.as_str()?.to_string()
    };
    Some((leg, subject))
}

/// Every index key `capsule` contributes.
pub fn record_index_keys(capsule: &Value) -> Vec<String> {
    let mut keys = counterparty_citation_keys(capsule);
    if let Some((leg, subject)) = capsule.get("settlement").and_then(settlement_leg_subject) {
        keys.push(settlement_leg_key(&leg, &subject));
    }
    let Some(attestation) = capsule.pointer("/model_attestation/compute_attestation") else {
        return keys;
    };
    if let Some(event_ref) = attestation
        .get(SETTLEMENT_EXTENSION_KEY)
        .and_then(|s| s.get("event_ref"))
        .and_then(Value::as_str)
    {
        keys.push(settlement_event_key(event_ref));
    }
    for block in [ADJUDICATION_ISSUED_BLOCK, ADJUDICATION_RECEIVED_BLOCK] {
        if let Some(verdict) = attestation
            .get(block)
            .and_then(|b| b.get("verdict_capsule_id"))
            .and_then(Value::as_str)
        {
            keys.push(adjudication_key(block, verdict));
        }
    }
    if let Some(refused) = attestation.get(ADJUDICATION_ACK_REFUSED_BLOCK) {
        if let (Some(verdict), Some(peer)) = (
            refused.get("verdict_capsule_id").and_then(Value::as_str),
            refused.get("refused_by").and_then(Value::as_str),
        ) {
            keys.push(adjudication_key(
                ADJUDICATION_ACK_REFUSED_BLOCK,
                &format!("{verdict}/{peer}"),
            ));
        }
    }
    keys
}

/// Open (or recover) a ledger under this plugin's index.
pub fn open_ledger(ledger_dir: &Path) -> Result<(Ledger, RecoveryReport), LedgerError> {
    Ledger::open_with_indexer(ledger_dir, record_index_keys)
}
