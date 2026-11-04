//! Why a stage did not answer a receipts request, as its caller needs to
//! tell: an unknown receipt, a stage without receipts, a turn the stage
//! can no longer open, or a request it refuses.
//!
//! The reasons travel as errors inside a stage, as reply kinds between
//! stages ([`ReceiptsReply`]) and as HTTP statuses to a client, and keep
//! their meaning along a chain: what a later stage answered is what stage
//! 0 answers its client.

use skippy_protocol::binary::{MAX_STAGE_RECEIPTS_REFUSAL_BYTES, ReceiptsReply};
use skippy_receipts::wire::CannotOpen;

/// A request names a receipt the stage holds no record of: one it never
/// issued, or one it dropped after its challenge window.
#[derive(Debug)]
pub(super) struct NotFound(pub String);

impl std::fmt::Display for NotFound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for NotFound {}

/// The stage issues no receipts: they are off, or it is being unloaded.
#[derive(Debug)]
pub(super) struct Unavailable(pub String);

impl std::fmt::Display for Unavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Unavailable {}

pub(super) fn not_found(receipt_id: &str) -> anyhow::Error {
    anyhow::Error::new(NotFound(format!("no receipt {receipt_id}")))
}

/// The reply a stage sends the stage before it for a request it did not
/// answer.
pub(super) fn reply(error: &anyhow::Error) -> ReceiptsReply {
    if let Some(cannot) = error.downcast_ref::<CannotOpen>()
        && let Ok(body) = serde_json::to_vec(cannot)
        && body.len() <= MAX_STAGE_RECEIPTS_REFUSAL_BYTES
    {
        return ReceiptsReply::CannotOpen(body);
    }
    let mut reason = format!("{error:#}");
    let mut end = reason.len().min(MAX_STAGE_RECEIPTS_REFUSAL_BYTES);
    while !reason.is_char_boundary(end) {
        end -= 1;
    }
    reason.truncate(end);
    if error.downcast_ref::<NotFound>().is_some() {
        ReceiptsReply::NotFound(reason)
    } else if error.downcast_ref::<Unavailable>().is_some() {
        ReceiptsReply::Unavailable(reason)
    } else {
        ReceiptsReply::Refused(reason)
    }
}

/// The answer in a later stage's reply, or the error that says why there
/// is none, as this stage would have raised it itself.
pub(super) fn answer(reply: ReceiptsReply) -> anyhow::Result<Vec<u8>> {
    match reply {
        ReceiptsReply::Answer(body) => Ok(body),
        ReceiptsReply::Refused(reason) => anyhow::bail!("the next stage: {reason}"),
        ReceiptsReply::NotFound(reason) => Err(anyhow::Error::new(NotFound(reason))),
        ReceiptsReply::Unavailable(reason) => Err(anyhow::Error::new(Unavailable(reason))),
        ReceiptsReply::CannotOpen(body) => match serde_json::from_slice::<CannotOpen>(&body) {
            Ok(cannot) => Err(anyhow::Error::new(sanitized(cannot))),
            Err(error) => {
                anyhow::bail!(
                    "the next stage sent a cannot-open answer that does not parse: {error}"
                )
            }
        },
    }
}

/// Bytes of a later stage's cannot-open detail that stage 0 passes on and
/// signs, at most: enough for any reason a stage gives.
pub(super) const MAX_LATER_DETAIL_BYTES: usize = 256;

/// A later stage's cannot-open answer as stage 0 may sign it: its detail
/// cut to [`MAX_LATER_DETAIL_BYTES`], with no control characters, so a
/// stage cannot have stage 0 sign text that reads as more than a reason.
fn sanitized(mut cannot: CannotOpen) -> CannotOpen {
    let mut detail: String = cannot
        .detail
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect();
    let mut end = detail.len().min(MAX_LATER_DETAIL_BYTES);
    while !detail.is_char_boundary(end) {
        end -= 1;
    }
    detail.truncate(end);
    cannot.detail = detail;
    cannot
}

/// `error` from the stages after stage `stage_index`, for a request of
/// `receipt_id`. A cannot-open answer for another receipt, or from a stage
/// that is not a later one, becomes a plain refusal, which no stage signs.
pub(super) fn from_later_stages(
    error: anyhow::Error,
    receipt_id: &str,
    stage_index: u32,
) -> anyhow::Error {
    match error.downcast_ref::<CannotOpen>() {
        Some(cannot) if cannot.receipt_id != receipt_id || cannot.stage_index <= stage_index => {
            anyhow::anyhow!(
                "the next stage answered that stage {} cannot open another request",
                cannot.stage_index
            )
        }
        _ => error,
    }
}

#[cfg(test)]
mod tests {
    use skippy_receipts::wire::CannotOpenReason;

    use super::*;

    // Each reason survives the trip between two stages.
    #[test]
    fn a_reason_keeps_its_kind_between_stages() {
        let cannot = CannotOpen {
            receipt_id: "r".into(),
            stage_index: 2,
            reason: CannotOpenReason::CellsOverwritten,
            detail: "a later request wrote the cell".into(),
        };
        let error = answer(reply(&anyhow::Error::new(cannot.clone()))).unwrap_err();
        assert_eq!(error.downcast_ref::<CannotOpen>(), Some(&cannot));
        let error = answer(reply(&not_found("r"))).unwrap_err();
        assert!(error.downcast_ref::<NotFound>().is_some());
        let off = anyhow::Error::new(Unavailable("receipts are off".into()));
        let error = answer(reply(&off)).unwrap_err();
        assert!(error.downcast_ref::<Unavailable>().is_some());
        let refused = anyhow::anyhow!("x".repeat(2 * MAX_STAGE_RECEIPTS_REFUSAL_BYTES));
        let ReceiptsReply::Refused(reason) = reply(&refused) else {
            panic!("a refusal");
        };
        assert_eq!(reason.len(), MAX_STAGE_RECEIPTS_REFUSAL_BYTES);
        assert_eq!(
            answer(ReceiptsReply::Answer(b"[]".to_vec())).unwrap(),
            b"[]"
        );
    }

    // A later stage cannot have stage 0 sign text of its choosing, or a
    // cannot-open answer of another receipt or of an earlier stage.
    #[test]
    fn stage_zero_signs_only_a_later_stages_plain_reason() {
        let cannot = |receipt_id: &str, stage_index: u32, detail: String| CannotOpen {
            receipt_id: receipt_id.into(),
            stage_index,
            reason: CannotOpenReason::CellsOverwritten,
            detail,
        };
        let forged = cannot("r", 1, format!("ok\r\n{}", "x".repeat(4000)));
        let body = serde_json::to_vec(&forged).unwrap();
        let error = answer(ReceiptsReply::CannotOpen(body)).unwrap_err();
        let passed = error.downcast_ref::<CannotOpen>().unwrap();
        assert_eq!(passed.detail.len(), MAX_LATER_DETAIL_BYTES);
        assert!(passed.detail.starts_with("ok  x"));

        for (receipt_id, stage_index) in [("other", 1), ("r", 0)] {
            let error = anyhow::Error::new(cannot(receipt_id, stage_index, String::new()));
            let error = from_later_stages(error, "r", 0);
            assert!(error.downcast_ref::<CannotOpen>().is_none());
        }
        let error = from_later_stages(anyhow::Error::new(cannot("r", 1, String::new())), "r", 0);
        assert!(error.downcast_ref::<CannotOpen>().is_some());
    }
}
