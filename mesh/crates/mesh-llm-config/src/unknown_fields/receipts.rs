//! Unknown keys inside `[receipts]`.

use crate::{ConfigDiagnostic, ConfigDiagnosticCode, ConfigDiagnosticSource, ConfigPath};

/// `[receipts]` as the config struct reads it, with the keys it does not
/// name left over in `extra`.
#[derive(serde::Deserialize)]
struct ReceiptsTableProbe {
    #[serde(flatten)]
    _known: crate::model::ReceiptsConfig,
    #[serde(flatten)]
    extra: std::collections::BTreeMap<String, toml::Value>,
}

/// Unknown keys inside `[receipts]`. The table's struct drops them while
/// parsing, and a misspelt `require` or `offline` would leave the policy
/// the operator meant to set off without a word. As for model rows, the raw
/// TOML is parsed again with the same struct, so there is no key inventory
/// to drift.
pub(crate) fn unknown_receipts_field_diagnostics(raw_toml: Option<&str>) -> Vec<ConfigDiagnostic> {
    let Some(raw_toml) = raw_toml else {
        return Vec::new();
    };
    let Ok(value) = toml::from_str::<toml::Value>(raw_toml) else {
        return Vec::new();
    };
    let Some(receipts) = value.get("receipts") else {
        return Vec::new();
    };
    let Ok(probe) = receipts.clone().try_into::<ReceiptsTableProbe>() else {
        return Vec::new();
    };
    probe
        .extra
        .keys()
        .map(|key| {
            let mut path = ConfigPath::field("receipts");
            path.push_field(key);
            ConfigDiagnostic::warning(
                ConfigDiagnosticCode::UnknownField,
                ConfigDiagnosticSource::Validation,
                format!("unknown key '{key}' in [receipts] is ignored"),
            )
            .with_help("check for a typo, or remove the key")
            .at_path(path)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::tests::unknown_field_keys;

    #[test]
    fn warns_about_a_misspelt_receipts_key() {
        let messages = unknown_field_keys(
            r#"
            [receipts]
            enabled = true
            requier = true
            "#,
        );

        assert_eq!(messages.len(), 1, "got {messages:?}");
        assert!(messages[0].contains("requier"), "got {messages:?}");
        assert!(messages[0].contains("[receipts]"), "got {messages:?}");
    }

    #[test]
    fn accepts_every_receipts_key_the_config_reads() {
        let messages = unknown_field_keys(
            r#"
            [receipts]
            enabled = true
            audit = true
            require = true
            offline = true
            audit_one_in = 1
            "#,
        );

        assert!(messages.is_empty(), "got {messages:?}");
    }
}
