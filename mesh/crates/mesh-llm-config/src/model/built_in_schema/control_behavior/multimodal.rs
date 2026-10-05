use super::shared::{
    push_non_empty_constraint, push_range_constraint, set_numeric, set_static_options,
    set_static_unavailable, set_text_format,
};
use super::*;

pub(super) fn apply_multimodal_behavior(
    setting: &mut ConfigSettingSchema,
    prefix: &str,
    suffix: &str,
) {
    match suffix {
        "mmproj" => {
            set_text_format(setting, ConfigTextFormat::Path);
            push_non_empty_constraint(setting);
        }
        "mmproj_url" => {
            set_text_format(setting, ConfigTextFormat::Url);
            push_non_empty_constraint(setting);
        }
        "mmproj_offload" => set_static_options(setting),
        "image_min_tokens" => {
            set_numeric(setting, Some(0.0), None, Some(1.0), Some("tokens"));
            push_range_constraint(setting, None, Some(format!("{prefix}.image_max_tokens")));
        }
        "image_max_tokens" => {
            set_numeric(setting, Some(0.0), None, Some(1.0), Some("tokens"));
            push_range_constraint(setting, Some(format!("{prefix}.image_min_tokens")), None);
        }
        "embeddings" => set_static_unavailable(
            setting,
            "Built-in multimodal embeddings controls are explicitly rejected from persisted config.",
        ),
        "reranking" => set_static_unavailable(
            setting,
            "Built-in reranking controls are explicitly rejected from persisted config.",
        ),
        "pooling" => set_static_unavailable(
            setting,
            "Built-in pooling controls are explicitly rejected from persisted config.",
        ),
        "vocoder" => set_static_unavailable(
            setting,
            "Built-in vocoder controls are explicitly rejected from persisted config.",
        ),
        _ => {}
    }
}
