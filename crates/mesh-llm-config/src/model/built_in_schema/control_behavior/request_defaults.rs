use super::shared::{
    in_condition, not_in_condition, push_dependency_disable, push_non_empty_constraint,
    set_numeric, set_static_options, set_static_unavailable, set_text_format,
};
use super::*;

pub(super) fn apply_request_defaults_behavior(
    setting: &mut ConfigSettingSchema,
    prefix: &str,
    suffix: &str,
) {
    match suffix {
        "max_tokens" => set_numeric(setting, Some(1.0), None, Some(1.0), Some("tokens")),
        "temperature" => set_numeric(setting, Some(0.0), None, Some(0.01), None),
        "top_p" | "min_p" | "typical_p" => {
            set_numeric(setting, Some(0.0), Some(1.0), Some(0.01), None);
        }
        "top_k" => set_numeric(setting, Some(0.0), None, Some(1.0), None),
        "top_nsigma" | "dynatemp_range" | "dynatemp_exponent" | "repeat_penalty"
        | "presence_penalty" | "frequency_penalty" => {
            set_numeric(setting, Some(0.0), None, Some(0.01), None);
        }
        "repeat_last_n" => set_numeric(setting, Some(-1.0), None, Some(1.0), None),
        "dry" => set_static_unavailable(
            setting,
            "Reserved sampler object is accepted for compatibility but not wired into the current runtime.",
        ),
        "xtc" => set_static_unavailable(
            setting,
            "Reserved sampler object is accepted for compatibility but not wired into the current runtime.",
        ),
        "adaptive" => set_static_unavailable(
            setting,
            "Reserved sampler object is accepted for compatibility but not wired into the current runtime.",
        ),
        "mirostat_mode" | "reasoning_format" | "reasoning_enabled" | "reasoning_budget" => {
            set_static_options(setting);
        }
        "mirostat_entropy" => {
            set_numeric(setting, Some(0.0), None, Some(0.1), None);
            push_mirostat_dependency(setting, prefix, suffix);
        }
        "mirostat_learning_rate" => {
            set_numeric(setting, Some(0.0), None, Some(0.01), None);
            push_mirostat_dependency(setting, prefix, suffix);
        }
        "sampler_sequence" | "chat_template" | "system_prompt" => {
            push_non_empty_constraint(setting);
        }
        "chat_template_file" => {
            set_text_format(setting, ConfigTextFormat::Path);
            push_non_empty_constraint(setting);
        }
        "backend_sampling" => set_static_unavailable(
            setting,
            "Backend-owned sampler blocks are explicitly rejected from the built-in control surface.",
        ),
        "grammar" => set_static_unavailable(
            setting,
            "Grammar injection is explicitly rejected on the built-in config surface.",
        ),
        "json_schema" => set_static_unavailable(
            setting,
            "JSON schema response shaping is intentionally rejected until a stable runtime contract exists.",
        ),
        "logprobs" => set_static_unavailable(
            setting,
            "Logprobs request defaults are explicitly rejected from persisted config.",
        ),
        _ => {}
    }
}

fn push_mirostat_dependency(setting: &mut ConfigSettingSchema, prefix: &str, leaf: &str) {
    let mode_path = format!("{prefix}.mirostat_mode");
    let current = format!("{prefix}.{leaf}");
    let allowed = vec![
        ConfigConditionValue::Integer(1),
        ConfigConditionValue::Integer(2),
        ConfigConditionValue::String("1".to_string()),
        ConfigConditionValue::String("2".to_string()),
    ];

    control_behavior_mut(setting)
        .enable_when
        .push(in_condition(&mode_path, allowed.clone()));
    push_dependency_disable(
        setting,
        not_in_condition(&mode_path, allowed),
        format!("{current} requires {mode_path} = 1 or 2"),
    );
}

fn control_behavior_mut(setting: &mut ConfigSettingSchema) -> &mut ConfigControlBehavior {
    setting
        .control_behavior
        .get_or_insert_with(ConfigControlBehavior::default)
}
