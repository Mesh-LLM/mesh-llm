use super::shared::{
    equals_string_condition, not_in_condition, push_dependency_disable, push_non_empty_constraint,
    set_numeric, set_static_options, set_static_unavailable, set_text_format,
};
use super::*;

pub(super) fn apply_skippy_behavior(setting: &mut ConfigSettingSchema, prefix: &str, suffix: &str) {
    match suffix {
        "stage_model_path" => {
            set_text_format(setting, ConfigTextFormat::Path);
            push_non_empty_constraint(setting);
        }
        "stage_role" | "stage_topology" | "binary_stage_transport" => {
            push_non_empty_constraint(setting);
        }
        "activation_wire_dtype" | "prefill_chunking" => set_static_options(setting),
        "openai_frontend_mode" => set_static_unavailable(
            setting,
            "OpenAI frontend override wiring is intentionally rejected on the built-in schema surface.",
        ),
        "lifecycle_startup_timeout_ms"
        | "lifecycle_readiness_interval_ms"
        | "lifecycle_health_interval_ms" => {
            set_numeric(setting, Some(1.0), None, Some(1.0), Some("ms"));
        }
        "prefill_chunk_size" => {
            set_numeric(setting, Some(1.0), None, Some(1.0), Some("tokens"));
            push_chunk_dependency(setting, prefix, "fixed", "prefill_chunk_size");
        }
        "prefill_chunk_schedule" => {
            set_text_format(setting, ConfigTextFormat::CsvPositiveInts);
            push_non_empty_constraint(setting);
            push_chunk_dependency(setting, prefix, "schedule", "prefill_chunk_schedule");
        }
        _ => {}
    }
}

fn push_chunk_dependency(
    setting: &mut ConfigSettingSchema,
    prefix: &str,
    required_mode: &str,
    leaf: &str,
) {
    let path = format!("{prefix}.prefill_chunking");
    let allowed = vec![ConfigConditionValue::String(required_mode.to_string())];
    let current = format!("{prefix}.{leaf}");

    control_behavior_mut(setting)
        .enable_when
        .push(equals_string_condition(&path, required_mode));
    push_dependency_disable(
        setting,
        not_in_condition(&path, allowed),
        format!("{current} requires {path} = \"{required_mode}\""),
    );
}

fn control_behavior_mut(setting: &mut ConfigSettingSchema) -> &mut ConfigControlBehavior {
    setting
        .control_behavior
        .get_or_insert_with(ConfigControlBehavior::default)
}
