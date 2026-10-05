use super::shared::{set_numeric, set_static_options, set_text_format};
use super::*;

pub(super) fn apply_logging_behavior(setting: &mut ConfigSettingSchema, path: &str) {
    match path {
        "logging.enabled" | "logging.webhook.enabled" | "logging.audit.enabled" => {
            set_static_options(setting);
        }
        "logging.application_state_root" | "logging.audit.log_path" => {
            set_text_format(setting, ConfigTextFormat::Path);
        }
        "logging.webhook.url" => set_text_format(setting, ConfigTextFormat::Url),
        "logging.artifact.capture_mode"
        | "logging.audit.log_format"
        | "logging.audit.log_level" => {
            set_static_options(setting);
        }
        "logging.summary_line_limit" => numeric(setting, 1.0, 65_536.0, "characters"),
        "logging.event_buffer_size" => numeric(setting, 50.0, 100_000.0, "events"),
        "logging.retention_ttl_secs" => numeric(setting, 3_600.0, 7_776_000.0, "seconds"),
        "logging.replay_capacity" => numeric(setting, 1.0, 10_000.0, "events"),
        "logging.queue_capacity" => numeric(setting, 64.0, 131_072.0, "entries"),
        "logging.cleanup_cadence_secs" => numeric(setting, 300.0, 86_400.0, "seconds"),
        "logging.artifact.byte_limit_bytes" => numeric(setting, 1_024.0, 16_777_216.0, "bytes"),
        "logging.artifact.aggregate_limit_bytes" => {
            numeric(setting, 524_288.0, 524_288_000.0, "bytes")
        }
        "logging.export_limit_bytes" => numeric(setting, 65_536.0, 104_857_600.0, "bytes"),
        "logging.webhook.max_attempts" => numeric(setting, 1.0, 20.0, "attempts"),
        "logging.webhook.timeout_secs" => numeric(setting, 1.0, 60.0, "seconds"),
        "logging.webhook.dead_letter_retention_secs" => {
            numeric(setting, 3_600.0, 1_555_200.0, "seconds")
        }
        "logging.audit.max_file_size_mb" => {
            set_numeric(setting, Some(1.0), None, Some(1.0), Some("MB"))
        }
        "logging.audit.max_files" => {
            set_numeric(setting, Some(1.0), None, Some(1.0), Some("files"))
        }
        _ => {}
    }
}

fn numeric(setting: &mut ConfigSettingSchema, min: f64, max: f64, unit: &str) {
    set_numeric(setting, Some(min), Some(max), Some(1.0), Some(unit));
}
