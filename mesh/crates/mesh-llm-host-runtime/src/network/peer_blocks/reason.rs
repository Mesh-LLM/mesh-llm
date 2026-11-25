//! A block reason the host builds from strings a peer controls, such as a
//! failed receipt audit's model name and failures, cut to the size the
//! store keeps so the peer cannot make its own block fail by sending long
//! ones.

use super::{MAX_REASON_BYTES, Requester};

/// The longest string kept whole on the first pass, in characters.
const FIRST_STRING_CHARS: usize = 1024;
/// The most entries of an array kept.
const ARRAY_ENTRIES: usize = 16;
const CUT_MARK: char = '…';

/// The reason the store keeps for a block `requested_by` asks for. The host
/// builds a receipt audit's reason from what the peer sent, so its size is
/// the peer's to choose: it is cut to size, never a way for the peer to make
/// its own block fail. Any other requester's reason is its own to keep
/// within the limit.
pub(super) fn kept_for(
    requested_by: &Requester,
    reason: Option<serde_json::Value>,
) -> Option<serde_json::Value> {
    match requested_by {
        Requester::Receipts => reason.map(bounded_reason),
        Requester::Operator | Requester::Plugin(_) => reason,
    }
}

/// `reason`, or a cut copy of it that serializes to at most
/// [`MAX_REASON_BYTES`]: long strings lose their ends and long arrays their
/// last entries, more on each pass, until it fits. Object keys are kept, so
/// what a reader looks for is still where it was.
fn bounded_reason(reason: serde_json::Value) -> serde_json::Value {
    if fits(&reason) {
        return reason;
    }
    let mut string_chars = FIRST_STRING_CHARS;
    loop {
        let mut cut = reason.clone();
        cut_value(&mut cut, string_chars);
        if fits(&cut) {
            return cut;
        }
        if string_chars == 0 {
            // only keys and structure are left, and they do not fit
            return serde_json::json!({ "reason_omitted": "too large to keep" });
        }
        string_chars /= 2;
    }
}

fn fits(reason: &serde_json::Value) -> bool {
    reason.to_string().len() <= MAX_REASON_BYTES
}

fn cut_value(value: &mut serde_json::Value, string_chars: usize) {
    match value {
        serde_json::Value::String(text) => {
            if text.chars().count() > string_chars {
                let mut kept: String = text.chars().take(string_chars).collect();
                kept.push(CUT_MARK);
                *text = kept;
            }
        }
        serde_json::Value::Array(entries) => {
            entries.truncate(ARRAY_ENTRIES);
            for entry in entries {
                cut_value(entry, string_chars);
            }
        }
        serde_json::Value::Object(fields) => {
            for field in fields.values_mut() {
                cut_value(field, string_chars);
            }
        }
        serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::super::{BlockLength, PeerBlocks, peer_key};
    use super::*;

    #[test]
    fn a_reason_that_fits_is_kept_as_it_is() {
        let reason = serde_json::json!({"receipt_audit": {"model": "m", "failures": ["a", "b"]}});
        assert_eq!(bounded_reason(reason.clone()), reason);
    }

    #[test]
    fn long_strings_and_arrays_are_cut_until_the_reason_fits() {
        // every character here takes six bytes of JSON
        let hostile = "\u{1}".repeat(50_000);
        let reason = serde_json::json!({
            "receipt_audit": {
                "receipt_id": hostile,
                "model": "x".repeat(100_000),
                "failures": vec!["y".repeat(5_000); 500],
                "evidence": null,
            }
        });
        let cut = bounded_reason(reason);
        assert!(cut.to_string().len() <= MAX_REASON_BYTES);
        let audit = &cut["receipt_audit"];
        assert!(audit["model"].as_str().unwrap().starts_with('x'));
        assert!(audit["model"].as_str().unwrap().ends_with(CUT_MARK));
        assert!(audit["failures"].as_array().unwrap().len() <= ARRAY_ENTRIES);
        assert!(audit["evidence"].is_null());
    }

    #[test]
    fn a_reason_of_keys_alone_that_cannot_fit_is_replaced() {
        let fields: serde_json::Map<String, serde_json::Value> = (0..2_000)
            .map(|n| (format!("key-{n}"), serde_json::Value::Null))
            .collect();
        let cut = bounded_reason(serde_json::Value::Object(fields));
        assert!(cut.to_string().len() <= MAX_REASON_BYTES);
        assert!(cut.get("reason_omitted").is_some());
    }

    #[test]
    fn a_receipt_audit_block_cannot_be_refused_for_its_reason() {
        let blocks = PeerBlocks::in_memory();
        let bad = iroh::SecretKey::generate().public();
        let reason = serde_json::json!({
            "receipt_audit": {"model": "m".repeat(3 * MAX_REASON_BYTES)}
        });
        blocks
            .block(
                &bad,
                BlockLength::SevenDays,
                Requester::Receipts,
                Some(reason),
                1,
            )
            .expect("a long reason must not stop the block");
        let kept = blocks.snapshot(2)[&peer_key(&bad)].reason.clone().unwrap();
        assert!(kept.to_string().len() <= MAX_REASON_BYTES);
    }
}
