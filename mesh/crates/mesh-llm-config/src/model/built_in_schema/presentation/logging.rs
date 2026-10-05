use super::{CategoryPresentation, SettingPresentation, sp};

const GENERAL: CategoryPresentation = CategoryPresentation {
    id: "logs-general",
    label: "General",
    summary: "Request logging, local storage, summaries, and exports",
    order: 10,
};
const RETENTION: CategoryPresentation = CategoryPresentation {
    id: "logs-retention",
    label: "Retention",
    summary: "Request history limits and automatic cleanup timing",
    order: 20,
};
const BUFFERS: CategoryPresentation = CategoryPresentation {
    id: "logs-buffers",
    label: "Buffers & Replay",
    summary: "In-memory event buffering, persistence, and live replay",
    order: 30,
};
const ARTIFACTS: CategoryPresentation = CategoryPresentation {
    id: "logs-artifacts",
    label: "Artifacts & Storage",
    summary: "Captured request content and per-request storage budgets",
    order: 40,
};
const WEBHOOKS: CategoryPresentation = CategoryPresentation {
    id: "logs-webhooks",
    label: "Webhooks",
    summary: "Terminal-event delivery and failed-delivery retention",
    order: 50,
};
const AUDIT: CategoryPresentation = CategoryPresentation {
    id: "logs-audit",
    label: "Security Audit",
    summary: "Advanced security-event file output and rotation",
    order: 60,
};

const BYTE_UNITS: &[(&str, &str, u64)] = &[
    ("bytes", "B", 1),
    ("kilobytes", "KB", 1_024),
    ("megabytes", "MB", 1_048_576),
    ("gigabytes", "GB", 1_073_741_824),
];
const CAPTURE_CHOICES: &[(&str, &str, &str)] = &[
    (
        "metadata_only",
        "Metadata only",
        "Timing, routing, status, and usage fields.",
    ),
    (
        "redacted_artifacts",
        "Redacted payloads",
        "Bounded request and response bodies after configured redaction.",
    ),
];
const AUDIT_FORMAT_CHOICES: &[(&str, &str, &str)] = &[(
    "json_lines",
    "JSON Lines",
    "One JSON object per line for streaming and command-line processing.",
)];
const AUDIT_LEVEL_CHOICES: &[(&str, &str, &str)] = &[
    (
        "info",
        "Informational",
        "Includes informational, warning, error, and critical events.",
    ),
    (
        "warn",
        "Warning",
        "Includes warning, error, and critical events.",
    ),
    ("error", "Error", "Includes error and critical events."),
    ("critical", "Critical", "Includes critical events only."),
];

pub(super) fn is_advanced_logging_category(category_id: &str) -> bool {
    category_id == AUDIT.id
}

fn byte_size(
    label: &'static str,
    help: &'static str,
    category: CategoryPresentation,
    order: u32,
) -> SettingPresentation {
    sp(label, help, category, order)
        .unit("bytes")
        .renderer("byte-size")
        .display_units(BYTE_UNITS)
}

pub(super) fn logging_presentation(rendered: &str) -> Option<SettingPresentation> {
    match rendered {
        "logging.enabled" => Some(sp("Request logging", "Records request lifecycle and operational events in the local log store.", GENERAL, 10).hint("toggle")),
        "logging.application_state_root" => Some(sp("Log storage location", "Directory containing the local log database and retained artifact files.", GENERAL, 20).placeholder("~/.mesh-llm/logging").renderer("host-directory-picker")),
        "logging.export_limit_bytes" => Some(byte_size("Export size limit", "Maximum size of one generated log export.", GENERAL, 30)),
        "logging.summary_line_limit" => Some(sp("Summary length", "Maximum number of Unicode characters in each generated request summary.", GENERAL, 40).unit("characters")),
        "logging.cleanup_cadence_secs" => Some(sp("Cleanup interval", "Time between automatic retention cleanup runs.", RETENTION, 10).unit("seconds")),
        "logging.retention_max_rows" => Some(sp("Request history limit", "Maximum number of retained request summary rows.", RETENTION, 20).unit("requests")),
        "logging.retention_ttl_secs" => Some(sp("Retention period", "Age at which retained request records become eligible for cleanup.", RETENTION, 30).unit("seconds")),
        "logging.event_buffer_size" => Some(sp("Event buffer", "Maximum number of event entries held in memory for replay.", BUFFERS, 10).unit("events")),
        "logging.queue_capacity" => Some(sp("Write queue", "Maximum number of pending log entries waiting for persistence and webhook dispatch.", BUFFERS, 20).unit("entries")),
        "logging.replay_capacity" => Some(sp("Live replay window", "Number of recent events available to reconnecting console clients.", BUFFERS, 30).unit("events")),
        "logging.artifact.aggregate_limit_bytes" => Some(byte_size("Request artifact budget", "Maximum combined retained payload size for one request.", ARTIFACTS, 10)),
        "logging.artifact.byte_limit_bytes" => Some(byte_size("Payload size limit", "Maximum retained size of an individual request or response payload.", ARTIFACTS, 20)),
        "logging.artifact.capture_mode" => Some(sp("Captured content", "Choose the request information retained with each log record.", ARTIFACTS, 30).hint("segmented").choices(CAPTURE_CHOICES)),
        "logging.webhook.enabled" => Some(sp("Webhook delivery", "Sends terminal log events to the configured endpoint.", WEBHOOKS, 10).hint("toggle")),
        "logging.webhook.url" => Some(sp("Endpoint URL", "Destination that receives terminal log event payloads.", WEBHOOKS, 20).placeholder("https://logs.example.com/events")),
        "logging.webhook.max_attempts" => Some(sp("Retry attempts", "Maximum delivery attempts for each webhook event.", WEBHOOKS, 30).unit("attempts")),
        "logging.webhook.timeout_secs" => Some(sp("Request timeout", "Maximum time allowed for one webhook delivery attempt.", WEBHOOKS, 40).unit("seconds")),
        "logging.webhook.dead_letter_retention_secs" => Some(sp("Failed delivery retention", "Time exhausted webhook deliveries remain available for inspection.", WEBHOOKS, 50).unit("seconds")),
        "logging.audit.enabled" => Some(sp("Security audit log", "Writes security-sensitive operator and authorization events to a dedicated local file.", AUDIT, 10).hint("toggle")),
        "logging.audit.log_path" => Some(sp("Audit file", "Filesystem path for the security audit log.", AUDIT, 20).placeholder("~/.mesh-llm/audit.jsonl")),
        "logging.audit.log_format" => Some(sp("File format", "Serialization format used for each audit record.", AUDIT, 30).hint("select").choices(AUDIT_FORMAT_CHOICES)),
        "logging.audit.log_level" => Some(sp("Minimum severity", "Lowest audit severity written to the security log.", AUDIT, 40).hint("select").choices(AUDIT_LEVEL_CHOICES)),
        "logging.audit.max_file_size_mb" => Some(sp("File size limit", "Size at which the active audit file rotates.", AUDIT, 50).unit("MB")),
        "logging.audit.max_files" => Some(sp("Rotated file count", "Maximum number of rotated audit files retained.", AUDIT, 60).unit("files")),
        _ => None,
    }
}
