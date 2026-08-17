use super::shared::{
    equals_bool_condition, falsy_condition, push_constraint, push_dependency_disable, set_numeric,
    set_static_options, setting_path,
};
use super::*;

pub(super) fn apply_model_fit_behavior(
    setting: &mut ConfigSettingSchema,
    prefix: &str,
    suffix: &str,
) {
    match suffix {
        "ctx_size" => set_numeric(setting, Some(1.0), None, Some(1.0), Some("tokens")),
        "batch" => set_numeric(setting, Some(1.0), None, Some(1.0), Some("tokens")),
        "ubatch" => {
            set_numeric(setting, Some(1.0), None, Some(1.0), Some("tokens"));
            push_constraint(
                setting,
                ConfigConstraint::Range {
                    min: None,
                    max: Some(format!("{prefix}.batch")),
                },
            );
        }
        "cache_type_k" | "cache_type_v" => {
            set_static_options(setting);
            push_constraint(setting, ConfigConstraint::NonEmpty);
        }
        "kv_cache_policy" => set_static_options(setting),
        "kv_offload" | "kv_unified" | "prompt_cache" | "context_shift" | "swa_full"
        | "flash_attention" => set_static_options(setting),
        "cache_ram_mib" => set_numeric(setting, Some(0.0), None, Some(1.0), Some("MiB")),
        "cache_idle_slots" => {
            set_numeric(setting, Some(0.0), None, Some(1.0), Some("slots"));
            push_dependency_disable(
                setting,
                equals_bool_condition(&format!("{prefix}.prompt_cache"), false),
                format!("{prefix}.cache_idle_slots requires {prefix}.prompt_cache = true"),
            );
        }
        "prefix_cache.enabled" => {}
        "prefix_cache.max_entries"
        | "prefix_cache.min_tokens"
        | "prefix_cache.shared_stride_tokens"
        | "prefix_cache.shared_record_limit" => {
            set_numeric(setting, Some(1.0), None, Some(1.0), None);
            push_prefix_cache_disable(setting, prefix);
        }
        "prefix_cache.max_bytes" => {
            set_numeric(setting, Some(0.0), None, Some(1.0), None);
            push_prefix_cache_disable(setting, prefix);
        }
        "prefix_cache.payload_mode" => {
            set_static_options(setting);
            push_prefix_cache_disable(setting, prefix);
        }
        "keep_tokens" => {
            set_numeric(setting, Some(0.0), None, Some(1.0), Some("tokens"));
            push_constraint(
                setting,
                ConfigConstraint::Range {
                    min: None,
                    max: Some(format!("{prefix}.ctx_size")),
                },
            );
        }
        "checkpoint_interval" | "checkpoint_count" => {
            set_numeric(setting, Some(1.0), None, Some(1.0), None);
        }
        "lookup_cache_static" | "lookup_cache_dynamic" => {
            push_constraint(setting, ConfigConstraint::NonEmpty);
        }
        _ => {}
    }
}

fn push_prefix_cache_disable(setting: &mut ConfigSettingSchema, prefix: &str) {
    push_dependency_disable(
        setting,
        falsy_condition(&format!("{prefix}.prefix_cache.enabled")),
        format!(
            "{} requires {prefix}.prefix_cache.enabled = true",
            setting_path(setting)
        ),
    );
}
