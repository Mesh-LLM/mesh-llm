//! PostHog batch transport.
//!
//! Everything here is best effort. Analytics must never slow a command down,
//! never change an exit code, and never surface an error to the user, so every
//! failure path ends in a `tracing::debug!` and nothing else.

use crate::event::{Event, Properties, Value};
use chrono::{DateTime, Utc};
use serde_json::{Map, json};
use std::time::Duration;

/// Upper bound on one ingestion request.
pub(crate) const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

/// One captured event awaiting delivery.
#[derive(Clone, Debug)]
pub(crate) struct Envelope {
    pub(crate) event: Event,
    pub(crate) properties: Properties,
    pub(crate) captured_at: DateTime<Utc>,
}

/// Render one envelope into PostHog's batch item shape.
///
/// Two privacy properties are attached to every item and are not overridable
/// by callers:
///
/// - `$geoip_disable: true` stops PostHog's GeoIP enrichment from deriving a
///   location from the connecting address.
/// - `$ip: null` asks ingestion to drop the client address.
///
/// `$ip` is belt and braces. The authoritative control is the project's
/// "Discard client IP data" setting, which `docs/ANALYTICS.md` requires to be
/// on; this property means a misconfigured project still does not collect a
/// location.
pub(crate) fn batch_item(
    envelope: &Envelope,
    distinct_id: &str,
    base: &Properties,
) -> serde_json::Value {
    let mut properties = Map::new();
    properties.insert("distinct_id".to_owned(), json!(distinct_id));

    for source in [base, &envelope.properties] {
        for (key, value) in source.entries() {
            properties.insert((*key).to_owned(), value_to_json(value));
        }
    }

    properties.insert("$geoip_disable".to_owned(), json!(true));
    properties.insert("$ip".to_owned(), serde_json::Value::Null);

    json!({
        "event": envelope.event.name(),
        "properties": properties,
        "timestamp": envelope.captured_at.to_rfc3339(),
    })
}

fn value_to_json(value: &Value) -> serde_json::Value {
    match value {
        Value::Bool(inner) => json!(inner),
        Value::Int(inner) => json!(inner),
        Value::Text(label) => json!(label.as_str()),
        Value::Static(inner) => json!(inner),
    }
}

/// Build the full `/batch/` request body.
pub(crate) fn batch_body(
    api_key: &str,
    distinct_id: &str,
    base: &Properties,
    envelopes: &[Envelope],
) -> serde_json::Value {
    let batch: Vec<_> = envelopes
        .iter()
        .map(|envelope| batch_item(envelope, distinct_id, base))
        .collect();
    json!({ "api_key": api_key, "batch": batch })
}

/// The `/batch/` endpoint for `host`.
pub(crate) fn batch_endpoint(host: &str) -> String {
    format!("{}/batch/", host.trim_end_matches('/'))
}

/// POST one batch, swallowing every failure.
///
/// Returns whether the batch was accepted, which only the tests care about.
pub(crate) async fn send_batch(
    http: &reqwest::Client,
    endpoint: &str,
    body: &serde_json::Value,
) -> bool {
    match http.post(endpoint).json(body).send().await {
        Ok(response) if response.status().is_success() => true,
        Ok(response) => {
            tracing::debug!(status = %response.status(), "analytics batch rejected");
            false
        }
        Err(error) => {
            tracing::debug!(%error, "analytics batch failed to send");
            false
        }
    }
}

#[cfg(test)]
#[path = "client/tests.rs"]
mod tests;
