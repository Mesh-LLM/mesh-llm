use super::*;

pub(super) fn exchange_grant_settings(prefix: &str) -> Vec<ConfigSettingSchema> {
    let prefix = format!("{prefix}.openai_exchange_grant");
    let mut settings = Vec::new();
    for key in [
        "request_body",
        "effective_request_body",
        "response_body",
        "admission",
        "metadata",
        "read_identity_bundle",
        "delegate_signing_key",
    ] {
        settings.push(grant_setting(
            &format!("{prefix}.{key}"),
            ConfigValueSchema::Boolean,
        ));
    }
    for key in ["endpoints", "phases", "headers", "signing_scopes"] {
        let value = match key {
            "endpoints" => string_enum_from_slice(crate::OPENAI_EXCHANGE_ENDPOINTS),
            "phases" => string_enum_from_slice(crate::OPENAI_EXCHANGE_PHASES),
            _ => ConfigValueSchema::String,
        };
        settings.push(grant_setting(
            &format!("{prefix}.{key}"),
            ConfigValueSchema::Array {
                items: Box::new(value),
            },
        ));
    }
    for (key, maximum) in [
        ("deadline_ms", 30_000),
        ("max_body_bytes", 16_777_216),
        ("max_queue_bytes", 67_108_864),
        ("max_in_flight", 1024),
        ("max_delegation_ttl_secs", 86_400),
    ] {
        let mut setting = grant_setting(&format!("{prefix}.{key}"), ConfigValueSchema::Integer);
        setting.constraints.push(ConfigConstraint::Range {
            min: Some(
                if key == "max_delegation_ttl_secs" {
                    "0"
                } else {
                    "1"
                }
                .into(),
            ),
            max: Some(maximum.to_string()),
        });
        settings.push(setting);
    }
    settings.push(grant_setting(
        &format!("{prefix}.failure_policy"),
        string_enum(["best_effort", "required"]),
    ));
    settings
}

fn grant_setting(path: &str, schema: ConfigValueSchema) -> ConfigSettingSchema {
    let mut setting = plugin_setting(path, schema);
    // Manifest declarations request access; they cannot author operator grants.
    setting.control_surfaces = vec![
        ConfigControlSurface::ConfigFile,
        ConfigControlSurface::OwnerControl,
    ];
    setting.apply_mode = ConfigApplyMode::DynamicApply;
    setting.restart_scope = ConfigRestartScope::None;
    setting.description = Some("Operator-owned OpenAI lifecycle permission; owner apply refreshes grants immediately. Absent grants provide no access.".into());
    setting
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_delegating_grants_expose_their_valid_zero_ttl() {
        let grant = crate::OpenAiExchangeGrant {
            endpoints: vec!["chat_completions".into()],
            phases: vec!["request_received".into()],
            deadline_ms: 250,
            max_body_bytes: 1024,
            max_queue_bytes: 1024,
            max_in_flight: 1,
            ..Default::default()
        };
        assert!(grant.validate().is_ok());
        let ttl = exchange_grant_settings("plugins[0]")
            .into_iter()
            .find(|setting| setting.path.render().ends_with("max_delegation_ttl_secs"))
            .unwrap();
        assert!(ttl.constraints.iter().any(|constraint| matches!(
            constraint,
            ConfigConstraint::Range { min, max }
                if min.as_deref() == Some("0") && max.as_deref() == Some("86400")
        )));
        assert!(
            crate::OpenAiExchangeGrant {
                delegate_signing_key: true,
                ..grant
            }
            .validate()
            .is_err()
        );
    }
}
