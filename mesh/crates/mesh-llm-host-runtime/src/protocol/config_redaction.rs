//! Secret redaction for owner-control config snapshots.
//!
//! The owner-control get-config snapshot is served to the local console, and
//! telemetry headers can carry OTLP credentials, so the snapshot replaces the
//! real header values with a stable sentinel. When the console sends the
//! redacted snapshot back through apply-config, the apply path restores the
//! original values from the currently active config so the sentinel is never
//! persisted to the config file.

use crate::plugin::MeshConfig;

/// Sentinel value served in place of a real telemetry header value.
pub(crate) const REDACTED_TELEMETRY_HEADER_VALUE: &str = "__MESH_LLM_REDACTED__";

/// Replace telemetry header values in a snapshot before it is served.
pub(crate) fn redact_telemetry_headers(config: &mut MeshConfig) {
    for value in config.telemetry.headers.values_mut() {
        *value = REDACTED_TELEMETRY_HEADER_VALUE.to_owned();
    }
}

/// Restore header values that arrived as redaction sentinels from the
/// currently active config. Sentinels without a current value are dropped so
/// the sentinel is never persisted.
pub(crate) fn restore_redacted_telemetry_headers(
    incoming: &mut MeshConfig,
    current_headers: &std::collections::BTreeMap<String, String>,
) {
    let incoming_headers = &mut incoming.telemetry.headers;
    incoming_headers.retain(|key, value| {
        if value != REDACTED_TELEMETRY_HEADER_VALUE {
            return true;
        }
        match current_headers.get(key) {
            Some(kept) => {
                *value = kept.clone();
                true
            }
            None => false,
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_with_header(key: &str, value: &str) -> MeshConfig {
        let mut config = MeshConfig::default();
        config
            .telemetry
            .headers
            .insert(key.to_owned(), value.to_owned());
        config
    }

    #[test]
    fn redact_replaces_all_header_values_with_the_sentinel() {
        let mut config = config_with_header("authorization", "Bearer sk-live");
        redact_telemetry_headers(&mut config);
        assert_eq!(
            config
                .telemetry
                .headers
                .get("authorization")
                .map(String::as_str),
            Some(REDACTED_TELEMETRY_HEADER_VALUE)
        );
    }

    #[test]
    fn restore_keeps_the_current_value_for_a_sentinel() {
        let mut incoming = config_with_header("authorization", REDACTED_TELEMETRY_HEADER_VALUE);
        let current = config_with_header("authorization", "Bearer sk-live");
        restore_redacted_telemetry_headers(&mut incoming, &current.telemetry.headers);
        assert_eq!(
            incoming
                .telemetry
                .headers
                .get("authorization")
                .map(String::as_str),
            Some("Bearer sk-live")
        );
    }

    #[test]
    fn restore_drops_sentinel_without_a_current_value() {
        let mut incoming = config_with_header("x-stale", REDACTED_TELEMETRY_HEADER_VALUE);
        let current = MeshConfig::default();
        restore_redacted_telemetry_headers(&mut incoming, &current.telemetry.headers);
        assert!(
            !incoming.telemetry.headers.contains_key("x-stale"),
            "the sentinel must never be persisted"
        );
    }

    #[test]
    fn restore_leaves_real_values_untouched() {
        let mut incoming = config_with_header("x-label", "prod");
        let current = config_with_header("x-label", "other");
        restore_redacted_telemetry_headers(&mut incoming, &current.telemetry.headers);
        assert_eq!(
            incoming
                .telemetry
                .headers
                .get("x-label")
                .map(String::as_str),
            Some("prod"),
            "a non-sentinel value is an intentional change and must not be reverted"
        );
    }
}
