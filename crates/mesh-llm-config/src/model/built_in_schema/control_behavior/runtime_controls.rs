use super::shared::{
    absent_condition, equals_bool_condition, present_condition, push_allowed_pattern_constraint,
    push_dependency_disable, push_non_empty_constraint, push_range_constraint,
    push_requires_constraint, set_numeric, set_static_options, set_static_unavailable,
    set_text_format,
};
use super::*;

pub(super) fn apply_runtime_controls_behavior(setting: &mut ConfigSettingSchema, rendered: &str) {
    match rendered {
        "owner_control.advertise_addr" => {
            control_behavior_mut(setting)
                .enable_when
                .push(present_condition("owner_control.bind"));
            push_dependency_disable(
                setting,
                absent_condition("owner_control.bind"),
                "owner_control.advertise_addr requires owner_control.bind so the advertised port is actually listening".to_string(),
            );
            push_requires_constraint(setting, "owner_control.bind");
        }
        "telemetry.enabled" => set_static_options(setting),
        "telemetry.service_name" => {
            push_non_empty_constraint(setting);
            push_allowed_pattern_constraint(setting, r"^[A-Za-z0-9_-]+$");
        }
        "telemetry.endpoint" | "telemetry.metrics.endpoint" => {
            push_non_empty_constraint(setting);
        }
        "telemetry.export_interval_secs" => {
            set_numeric(setting, Some(1.0), None, Some(1.0), Some("sec"));
        }
        "telemetry.queue_size" => set_numeric(setting, Some(1.0), None, Some(1.0), None),
        "telemetry.prompt_shape_metrics" => set_static_unavailable(
            setting,
            "Prompt-shape telemetry is intentionally disabled until the telemetry surface is reviewed.",
        ),
        "mesh_requirements.min_node_version" | "mesh_requirements.max_node_version" => {
            set_text_format(setting, ConfigTextFormat::Semver);
            push_non_empty_constraint(setting);
        }
        "mesh_requirements.min_protocol_version" => {
            set_numeric(setting, Some(0.0), None, Some(1.0), None);
            push_range_constraint(
                setting,
                None,
                Some("mesh_requirements.max_protocol_version".to_string()),
            );
        }
        "mesh_requirements.max_protocol_version" => {
            set_numeric(setting, Some(0.0), None, Some(1.0), None);
            push_range_constraint(
                setting,
                Some("mesh_requirements.min_protocol_version".to_string()),
                None,
            );
        }
        "mesh_requirements.require_release_attestation" => set_static_options(setting),
        "mesh_requirements.release_signer_keys" => {
            set_text_format(setting, ConfigTextFormat::Ed25519Key);
            control_behavior_mut(setting)
                .enable_when
                .push(equals_bool_condition(
                    "mesh_requirements.require_release_attestation",
                    true,
                ));
            push_dependency_disable(
                setting,
                equals_bool_condition("mesh_requirements.require_release_attestation", false),
                "mesh_requirements.release_signer_keys requires mesh_requirements.require_release_attestation = true".to_string(),
            );
        }
        "plugin.<plugin-name>.startup.connect_timeout_secs"
        | "plugin.<plugin-name>.startup.init_timeout_secs" => {
            set_numeric(setting, Some(1.0), None, Some(1.0), Some("sec"));
        }
        "plugin.<plugin-name>.startup.optional" | "plugin.<plugin-name>.startup.lazy_start" => {
            set_static_options(setting)
        }
        _ => {}
    }
}

fn control_behavior_mut(setting: &mut ConfigSettingSchema) -> &mut ConfigControlBehavior {
    setting
        .control_behavior
        .get_or_insert_with(ConfigControlBehavior::default)
}
