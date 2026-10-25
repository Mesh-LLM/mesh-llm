use crate::process::{LineEnding, ObservedLine};
use serde::Deserialize;

#[cfg(test)]
#[path = "../../../tests/migration_lifecycle/daemon/correlation.rs"]
mod tests;

#[derive(Clone, Copy)]
pub(crate) struct RequestId(uuid::Uuid);

impl RequestId {
    pub(crate) fn generate() -> Result<Self, super::Error> {
        let mut bytes = [0; 16];
        getrandom::fill(&mut bytes)
            .map_err(|_| super::Error::Invalid("request ID entropy unavailable"))?;
        Ok(Self(uuid::Builder::from_random_bytes(bytes).into_uuid()))
    }

    pub(super) fn header(self) -> String {
        self.0.hyphenated().to_string()
    }
}

#[derive(Deserialize)]
struct Terminal {
    request_id: String,
    source: String,
    route: String,
    method: String,
    request_kind: String,
    status_code: u16,
    event: String,
    outcome: String,
}

pub(super) fn classify(line: ObservedLine<'_>, request_id: RequestId) -> Option<u16> {
    if line.ending != LineEnding::Lf || line.bytes.len() > 8192 {
        return None;
    }
    let text = std::str::from_utf8(line.bytes).ok()?;
    if !text.trim_start().starts_with('{') {
        return None;
    }
    let record: Terminal = serde_json::from_str(text).ok()?;
    let mut encoded = uuid::Uuid::encode_buffer();
    if record.request_id.as_str() != request_id.0.hyphenated().encode_lower(&mut encoded)
        || record.source != "direct_http"
        || record.route != "models"
        || record.method != "GET"
        || record.request_kind != "model_listing"
    {
        return None;
    }
    let terminal = match record.status_code {
        200..=299 => record.event == "request_completed" && record.outcome == "completed",
        300..=399 => record.event == "request_failed" && record.outcome == "failed",
        _ => false,
    };
    terminal.then_some(record.status_code)
}
