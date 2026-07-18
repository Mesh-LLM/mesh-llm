pub use mesh_llm_events::{
    ConsoleSessionMode, DashboardAcceptedRequestBucket, DashboardEndpointRow, DashboardLaunchPlan,
    DashboardModelLane, DashboardModelRow, DashboardProcessRow, DashboardSnapshot,
    DashboardSnapshotFuture, DashboardSnapshotProvider, LlamaInstanceKind, LogFormat,
    ModelProgressStatus, OutputEvent, OutputLevel, OutputSink, OutputSinkFuture, RuntimeStatus,
    TuiControlFlow, TuiEvent, TuiKeyEvent,
};
use ratatui::{
    style::{Color, Style},
    text::Text,
};
use std::sync::OnceLock;
use tokio::time::Duration;

mod fatal;
pub use fatal::{emit_fatal_error, emit_fatal_panic};

mod dashboard;
mod formatting;
mod merging;
pub(in crate::output) mod rendering;
mod state;
#[cfg(test)]
mod tests;

pub use formatting::{
    DashboardFormatter, Formatter, InteractiveDashboardFormatter, JsonFormatter, OutputManager,
    PrettyFormatter, emit_event, flush_output, force_restore_tui_after_panic,
    force_restore_tui_terminal, interactive_tui_active, json_mode_enabled,
};
pub use merging::sort_dashboard_endpoint_rows;
pub use state::{
    EndpointState, LlamaInstanceState, MeshEventState, MultiModelModeState, PassiveModeState,
    RunningModelState, StartupComponentState, StartupLifecyclePhase, StartupLifecycleState,
};

use dashboard::*;
pub(crate) use formatting::{GLOBAL_OUTPUT_MANAGER, write_emergency_event};
use merging::*;
use rendering::*;
use state::*;

const DEFAULT_PRETTY_DASHBOARD_EVENT_HISTORY_LIMIT: usize = 1000;
const PRETTY_TUI_STARTUP_HISTORY_LIMIT: usize = 32;
const PRETTY_DASHBOARD_REQUEST_WINDOW_BUCKETS: usize = 30;
const PRETTY_DASHBOARD_REQUEST_MAX_WINDOW_SECS: u32 = 24 * 60 * 60;
const PRETTY_DASHBOARD_PANEL_COUNT: usize = 6;
const PRETTY_TUI_REDRAW_INTERVAL: Duration = Duration::from_millis(33);
const PRETTY_TUI_SNAPSHOT_INTERVAL: Duration = Duration::from_millis(250);
const PRETTY_TUI_JOIN_TOKEN_COPY_STATUS_TTL: Duration = Duration::from_secs(2);
const PRETTY_TUI_MODEL_CARD_HEIGHT: usize = 8;
const PRETTY_TUI_MODEL_CARD_STRIDE: usize = PRETTY_TUI_MODEL_CARD_HEIGHT;
#[cfg(test)]
const PRETTY_TUI_LIST_HIGHLIGHT_SYMBOL_WIDTH: u16 = 2;
const PRETTY_TUI_REQUEST_GRAPH_GUIDE_SYMBOL: &str = "·";
const PRETTY_TUI_REQUEST_GRAPH_BASELINE_SYMBOL: &str = "─";
const PRETTY_TUI_STARTUP_PROGRESS_MIN_STEPS: usize = 12;
const PRETTY_TUI_JOIN_TOKEN_PANEL_HEIGHT: u16 = 5;
const PRETTY_TUI_JOIN_TOKEN_HORIZONTAL_PADDING: u16 = 2;
const PRETTY_TUI_JOIN_TOKEN_COPY_BUTTON_LABEL: &str = " Copy ";
const PRETTY_TUI_EVENTS_COLUMN_PERCENT: u16 = 44;
const PRETTY_TUI_REMAINING_COLUMN_WEIGHT: u16 = 1;
const PRETTY_TUI_WEBSERVER_PROCESS_HEADER_LABEL: &str = "PROCESSES";
const PRETTY_TUI_MIN_DASHBOARD_WIDTH: u16 = 60;
const PRETTY_TUI_SPLASH_ANSI: &[u8] = include_bytes!("assets/pretty-tui-splash.ans");

static PRETTY_TUI_SPLASH_TEXT: OnceLock<Option<Text<'static>>> = OnceLock::new();
static PRETTY_TUI_READY_LOGO_TEXT: OnceLock<Option<Text<'static>>> = OnceLock::new();

#[derive(Clone, Copy)]
struct TuiTheme {
    surface: Color,
    surface_raised: Color,
    text: Color,
    muted: Color,
    dim: Color,
    accent: Color,
    accent_soft: Color,
    success: Color,
    warning: Color,
    error: Color,
    selection_bg: Color,
    status_bar: Style,
}

const fn tui_theme() -> TuiTheme {
    TuiTheme {
        surface: Color::Rgb(8, 10, 14),
        surface_raised: Color::Rgb(18, 22, 29),
        text: Color::Rgb(220, 226, 235),
        muted: Color::Rgb(138, 150, 166),
        dim: Color::Rgb(72, 82, 96),
        accent: Color::Rgb(69, 211, 255),
        accent_soft: Color::Rgb(84, 142, 188),
        success: Color::Rgb(95, 214, 130),
        warning: Color::Rgb(232, 190, 84),
        error: Color::Rgb(238, 93, 108),
        selection_bg: Color::Rgb(31, 40, 52),
        status_bar: Style::new()
            .fg(Color::Rgb(220, 226, 235))
            .bg(Color::Rgb(18, 22, 29)),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TuiEventListRenderer {
    #[cfg(test)]
    Legacy,
    Scrollbar,
}

const PRETTY_TUI_EVENT_LEVEL_WIDTH: usize = 6;

impl TuiEventListRenderer {
    const ACTIVE: Self = Self::Scrollbar;
}

fn strip_leading_severity_icon(message: &str) -> &str {
    message
        .strip_prefix("⚠️")
        .or_else(|| message.strip_prefix("❌"))
        .map(str::trim_start)
        .unwrap_or(message)
}

fn format_invite_mesh_label(mesh_name: Option<&str>, mesh_id: &str) -> String {
    match mesh_name.map(str::trim).filter(|name| !name.is_empty()) {
        Some(name) => format!("{name} ({mesh_id})"),
        None => mesh_id.to_string(),
    }
}
#[cfg(test)]
pub fn assert_startup_lifecycle_transitions_pending_partial_ready_failed() {
    tests::assert_startup_lifecycle_transitions_pending_partial_ready_failed();
}

#[cfg(test)]
pub fn assert_startup_lifecycle_keeps_runtime_ready_as_final_edge() {
    tests::assert_startup_lifecycle_keeps_runtime_ready_as_final_edge();
}

#[cfg(test)]
pub fn assert_startup_failures_surface_in_tui_events_and_status() {
    tests::assert_startup_failures_surface_in_tui_events_and_status();
}

#[cfg(test)]
pub fn assert_startup_failure_summary_sanitizes_multiline_detail() {
    tests::assert_startup_failure_summary_sanitizes_multiline_detail();
}

#[cfg(test)]
pub fn assert_rpc_and_llama_startup_failures_mark_components_failed() {
    tests::assert_rpc_and_llama_startup_failures_mark_components_failed();
}

#[cfg(test)]
pub fn assert_discovery_and_join_failures_mark_startup_mesh_component_failed() {
    tests::assert_discovery_and_join_failures_mark_startup_mesh_component_failed();
}

#[cfg(test)]
pub fn assert_post_ready_peer_churn_does_not_reopen_startup_failure() {
    tests::assert_post_ready_peer_churn_does_not_reopen_startup_failure();
}

#[cfg(test)]
pub fn assert_startup_history_is_visible_after_late_tui_attach() {
    tests::assert_startup_history_is_visible_after_late_tui_attach();
}

#[cfg(test)]
pub fn assert_startup_history_keeps_order_when_tui_attaches_late() {
    tests::assert_startup_history_keeps_order_when_tui_attaches_late();
}

#[cfg(test)]
pub fn assert_endpoint_rows_remain_starting_until_ready_events() {
    tests::assert_endpoint_rows_remain_starting_until_ready_events();
}

#[cfg(test)]
pub fn assert_startup_launch_plan_renders_not_ready_rows_before_actions() {
    tests::assert_startup_launch_plan_renders_not_ready_rows_before_actions();
}

#[cfg(test)]
pub fn assert_startup_progress_after_launch_plan_shows_dashboard_not_loader() {
    tests::assert_startup_progress_after_launch_plan_shows_dashboard_not_loader();
}

#[cfg(test)]
pub fn assert_tui_model_progress_renders_dashboard_without_loading_screen() {
    tests::assert_tui_model_progress_renders_dashboard_without_loading_screen();
}

#[cfg(test)]
pub fn assert_tui_startup_progress_continues_in_dashboard_after_model_download_ready() {
    tests::assert_tui_startup_progress_continues_in_dashboard_after_model_download_ready();
}

#[cfg(test)]
pub fn assert_planned_rows_transition_from_not_ready_to_ready_events() {
    tests::assert_planned_rows_transition_from_not_ready_to_ready_events();
}

#[cfg(test)]
pub fn assert_launch_plan_rows_survive_empty_startup_snapshot() {
    tests::assert_launch_plan_rows_survive_empty_startup_snapshot();
}

#[cfg(test)]
pub fn assert_launch_plan_preserves_distinct_port_zero_endpoint_rows() {
    tests::assert_launch_plan_preserves_distinct_port_zero_endpoint_rows();
}

#[cfg(test)]
pub fn assert_snapshot_upsert_preserves_distinct_port_zero_endpoint_rows() {
    tests::assert_snapshot_upsert_preserves_distinct_port_zero_endpoint_rows();
}

#[cfg(test)]
pub fn assert_planned_port_zero_process_rows_bind_to_concrete_startup_events() {
    tests::assert_planned_port_zero_process_rows_bind_to_concrete_startup_events();
}

#[cfg(test)]
pub fn assert_fallback_mode_surfaces_startup_failures_without_tui() {
    tests::assert_fallback_mode_surfaces_startup_failures_without_tui();
}

#[cfg(test)]
pub fn assert_shutdown_suppresses_late_ready_render() {
    tests::assert_shutdown_suppresses_late_ready_render();
}

#[cfg(test)]
pub fn assert_interactive_preterminal_render_uses_plain_event_output() {
    tests::assert_interactive_preterminal_render_uses_plain_event_output();
}

#[cfg(test)]
pub fn assert_interactive_post_terminal_exit_resumes_plain_event_output() {
    tests::assert_interactive_post_terminal_exit_resumes_plain_event_output();
}

#[cfg(test)]
pub fn assert_tui_model_card_separates_name_from_metadata_columns() {
    tests::assert_tui_model_card_separates_name_from_metadata_columns();
}
