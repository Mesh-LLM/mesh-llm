use super::shared::{
    push_non_empty_constraint, set_numeric, set_static_options, set_static_unavailable,
};
use super::*;

pub(super) fn apply_throughput_behavior(
    setting: &mut ConfigSettingSchema,
    _prefix: &str,
    suffix: &str,
) {
    match suffix {
        "parallel" => set_numeric(setting, Some(1.0), None, Some(1.0), Some("slots")),
        "continuous_batching" | "poll" => set_static_options(setting),
        "threads" | "threads_batch" => {
            set_numeric(setting, Some(0.0), None, Some(1.0), Some("threads"));
        }
        "threads_http" => set_static_unavailable(
            setting,
            "Dedicated HTTP worker tuning is rejected on the current embedded runtime path.",
        ),
        "cpu_affinity" | "numa" => push_non_empty_constraint(setting),
        "slot_prompt_similarity" => set_numeric(setting, Some(0.0), None, Some(0.01), None),
        "sleep_idle_seconds" => set_static_unavailable(
            setting,
            "The sleep-idle tuning knob is documented as rejected and must never become a live exported identifier.",
        ),
        "tuning_profile" => {
            set_static_options(setting);
            push_non_empty_constraint(setting);
        }
        _ => {}
    }
}
