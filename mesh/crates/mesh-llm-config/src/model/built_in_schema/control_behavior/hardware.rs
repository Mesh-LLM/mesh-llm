use super::shared::{
    equals_string_condition, push_dependency_disable, push_enable_when, push_non_empty_constraint,
    push_range_constraint, push_requires_constraint, set_numeric, set_runtime_gpu_options,
    set_static_options, set_static_unavailable, set_static_unavailable_with_note, set_text_format,
    set_write_policy,
};
use super::*;

pub(super) fn apply_hardware_behavior(
    setting: &mut ConfigSettingSchema,
    prefix: &str,
    suffix: &str,
) {
    match suffix {
        "model_runtime" => {
            set_static_options(setting);
            set_static_unavailable(
                setting,
                "Model runtime is selected by the installed native runtime and hardware resolver.",
            );
        }
        "device" => {
            set_runtime_gpu_options(setting);
            push_enable_when(setting, equals_string_condition("gpu.assignment", "pinned"));
            push_dependency_disable(
                setting,
                equals_string_condition("gpu.assignment", "auto"),
                "Set gpu.assignment = \"pinned\" to edit a concrete GPU device.".to_string(),
            );
            push_non_empty_constraint(setting);
        }
        "gpu_layers" => {
            set_static_options(setting);
            set_numeric(setting, Some(-1.0), None, Some(1.0), Some("layers"));
        }
        "stage_layer_start" => {
            set_numeric(setting, Some(0.0), None, Some(1.0), Some("layers"));
            push_requires_constraint(setting, &format!("{prefix}.stage_layer_end"));
        }
        "stage_layer_end" => {
            set_numeric(setting, Some(0.0), None, Some(1.0), Some("layers"));
            push_requires_constraint(setting, &format!("{prefix}.stage_layer_start"));
            push_range_constraint(setting, Some(format!("{prefix}.stage_layer_start")), None);
        }
        "placement" | "split_mode" | "cpu_moe" | "fit_context" => set_static_options(setting),
        "main_gpu" => set_numeric(setting, Some(0.0), None, Some(1.0), None),
        "tensor_split" => {}
        "n_cpu_moe" => set_numeric(setting, Some(0.0), None, Some(1.0), None),
        "rpc_backend" => set_static_unavailable(
            setting,
            "The legacy rpc_backend escape hatch is explicitly unsupported by the embedded runtime.",
        ),
        "fit_target_mib" => set_numeric(setting, Some(0.0), None, Some(1.0), Some("MiB")),
        "safety_margin_gb" => set_numeric(setting, Some(0.0), None, Some(0.1), Some("GB")),
        "model_path" => {
            set_text_format(setting, ConfigTextFormat::Path);
            push_non_empty_constraint(setting);
        }
        "mmproj" => {
            set_text_format(setting, ConfigTextFormat::Path);
            let canonical = prefix.replacen(".hardware", ".multimodal", 1);
            set_static_unavailable_with_note(
                setting,
                &format!("Edit {canonical}.mmproj instead of the legacy hardware duplicate."),
                &format!(
                    "Existing values are preserved on save unless you change {canonical}.mmproj."
                ),
            );
            set_write_policy(setting, ConfigDisabledWritePolicy::PreserveExisting);
        }
        "mmproj_offload" => {
            set_static_options(setting);
            let canonical = prefix.replacen(".hardware", ".multimodal", 1);
            set_static_unavailable_with_note(
                setting,
                &format!(
                    "Edit {canonical}.mmproj_offload instead of the legacy hardware duplicate."
                ),
                &format!(
                    "Existing values are preserved on save unless you change {canonical}.mmproj_offload."
                ),
            );
            set_write_policy(setting, ConfigDisabledWritePolicy::PreserveExisting);
        }
        "hf_repo" => push_hf_pair_constraint(setting, &format!("{prefix}.hf_file")),
        "hf_file" => push_hf_pair_constraint(setting, &format!("{prefix}.hf_repo")),
        _ => {}
    }
}

fn push_hf_pair_constraint(setting: &mut ConfigSettingSchema, sibling_path: &str) {
    push_non_empty_constraint(setting);
    push_requires_constraint(setting, sibling_path);
}
