use super::*;
mod hardware;
mod model_fit;
mod multimodal;
mod request_defaults;
mod runtime_controls;
mod shared;
mod skippy;
mod speculative;
mod throughput;

use self::hardware::apply_hardware_behavior;
use self::model_fit::apply_model_fit_behavior;
use self::multimodal::apply_multimodal_behavior;
use self::request_defaults::apply_request_defaults_behavior;
use self::runtime_controls::apply_runtime_controls_behavior;
use self::shared::{set_numeric, set_static_options};
use self::skippy::apply_skippy_behavior;
use self::speculative::apply_speculative_behavior;
use self::throughput::apply_throughput_behavior;

pub(super) fn apply_built_in_control_behavior(setting: &mut ConfigSettingSchema) {
    let rendered = setting.path.render();

    match rendered.as_str() {
        "gpu.assignment" => {
            set_static_options(setting);
        }
        "gpu.parallel" => {
            set_numeric(setting, Some(1.0), None, Some(1.0), Some("models"));
        }
        _ => {
            if let Some(suffix) = rendered.strip_prefix("defaults.model_fit.") {
                apply_model_fit_behavior(setting, "defaults.model_fit", suffix);
            } else if let Some(suffix) = rendered.strip_prefix("models.<model-ref>.model_fit.") {
                apply_model_fit_behavior(setting, "models.<model-ref>.model_fit", suffix);
            } else if let Some(suffix) = rendered.strip_prefix("defaults.hardware.") {
                apply_hardware_behavior(setting, "defaults.hardware", suffix);
            } else if let Some(suffix) = rendered.strip_prefix("models.<model-ref>.hardware.") {
                apply_hardware_behavior(setting, "models.<model-ref>.hardware", suffix);
            } else if let Some(suffix) = rendered.strip_prefix("defaults.throughput.") {
                apply_throughput_behavior(setting, "defaults.throughput", suffix);
            } else if let Some(suffix) = rendered.strip_prefix("models.<model-ref>.throughput.") {
                apply_throughput_behavior(setting, "models.<model-ref>.throughput", suffix);
            } else if let Some(suffix) = rendered.strip_prefix("defaults.skippy.") {
                apply_skippy_behavior(setting, "defaults.skippy", suffix);
            } else if let Some(suffix) = rendered.strip_prefix("models.<model-ref>.skippy.") {
                apply_skippy_behavior(setting, "models.<model-ref>.skippy", suffix);
            } else if let Some(suffix) = rendered.strip_prefix("defaults.speculative.") {
                apply_speculative_behavior(setting, "defaults.speculative", suffix);
            } else if let Some(suffix) = rendered.strip_prefix("models.<model-ref>.speculative.") {
                apply_speculative_behavior(setting, "models.<model-ref>.speculative", suffix);
            } else if let Some(suffix) = rendered.strip_prefix("defaults.request_defaults.") {
                apply_request_defaults_behavior(setting, "defaults.request_defaults", suffix);
            } else if let Some(suffix) =
                rendered.strip_prefix("models.<model-ref>.request_defaults.")
            {
                apply_request_defaults_behavior(
                    setting,
                    "models.<model-ref>.request_defaults",
                    suffix,
                );
            } else if let Some(suffix) = rendered.strip_prefix("defaults.multimodal.") {
                apply_multimodal_behavior(setting, "defaults.multimodal", suffix);
            } else if let Some(suffix) = rendered.strip_prefix("models.<model-ref>.multimodal.") {
                apply_multimodal_behavior(setting, "models.<model-ref>.multimodal", suffix);
            } else {
                apply_runtime_controls_behavior(setting, rendered.as_str());
            }
        }
    }
}
