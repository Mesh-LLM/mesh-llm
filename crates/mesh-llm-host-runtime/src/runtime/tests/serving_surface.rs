use super::*;

#[test]
fn early_tui_spawns_before_llama_ready_in_active_flow() {
    assert_active_serve_path_spawn_gate_behavior();
}

#[test]
fn passive_path_tui_still_starts_immediately() {
    assert_passive_path_immediate_spawn_behavior();
}

#[test]
fn interactive_handler_spawns_once_across_startup_callbacks() {
    assert_interactive_handler_spawns_once_across_startup_callbacks();
}

#[test]
fn initial_pretty_session_mode_allows_dashboard_for_explicit_surface() {
    assert_eq!(
        initial_console_session_mode_for_surface(
            Some(RuntimeSurface::Serve),
            ConsoleSessionMode::InteractiveDashboard
        ),
        ConsoleSessionMode::InteractiveDashboard
    );

    assert_eq!(
        initial_console_session_mode_for_surface(
            Some(RuntimeSurface::Client),
            ConsoleSessionMode::InteractiveDashboard
        ),
        ConsoleSessionMode::InteractiveDashboard
    );

    assert_eq!(
        initial_console_session_mode_for_surface(None, ConsoleSessionMode::InteractiveDashboard),
        ConsoleSessionMode::None
    );
}

#[test]
fn headless_host_logs_management_api_without_console_url() {
    let line = format_console_ready_line(true, "http://127.0.0.1:3131");
    assert!(
        line.contains("Management API"),
        "expected 'Management API' in headless output, got: {line}"
    );
    assert!(
        !line.contains("Console:"),
        "headless output must not contain 'Console:', got: {line}"
    );
}

#[test]
fn default_host_mode_still_logs_console_url() {
    let line = format_console_ready_line(false, "http://127.0.0.1:3131");
    assert!(
        line.contains("Console:"),
        "expected 'Console:' in default output, got: {line}"
    );
    assert!(
        !line.contains("Management API"),
        "default output must not contain 'Management API', got: {line}"
    );
}

#[test]
fn active_startup_passes_headless_to_management_server() {
    let headless_line = format_console_ready_line(true, "http://127.0.0.1:9090");
    let normal_line = format_console_ready_line(false, "http://127.0.0.1:9090");
    assert_ne!(
        headless_line, normal_line,
        "headless and non-headless output must differ"
    );
    assert!(headless_line.contains("9090"));
    assert!(normal_line.contains("9090"));
}

#[test]
fn headless_passive_mode_preserves_api_without_ui() {
    let line = format_console_ready_line(true, "http://127.0.0.1:3131");
    assert!(
        line.contains("Management API"),
        "passive headless output must contain 'Management API', got: {line}"
    );
    assert!(
        !line.contains("Console:"),
        "passive headless output must not contain 'Console:', got: {line}"
    );
}

#[test]
fn passive_headless_promotion_keeps_ui_disabled() {
    let promoted_line = format_console_ready_line(true, "http://127.0.0.1:3131");
    assert!(
        promoted_line.contains("Management API"),
        "promoted headless node must still advertise Management API, got: {promoted_line}"
    );
    assert!(
        !promoted_line.contains("Console:"),
        "promoted headless node must not show Console: URL, got: {promoted_line}"
    );
}

#[test]
fn default_passive_mode_still_serves_ui_when_not_headless() {
    let line = format_console_ready_line(false, "http://127.0.0.1:3131");
    assert!(
        line.contains("Console:"),
        "default passive output must contain 'Console:', got: {line}"
    );
    assert!(
        !line.contains("Management API"),
        "default passive output must not contain 'Management API', got: {line}"
    );
}

#[test]
fn test_console_session_mode_serve_uses_interactive_mode() {
    // When explicit_surface is Some(RuntimeSurface::Serve), should preserve current mode
    let result = initial_console_session_mode_for_surface(
        Some(RuntimeSurface::Serve),
        ConsoleSessionMode::InteractiveDashboard,
    );
    assert_eq!(result, ConsoleSessionMode::InteractiveDashboard);
}

#[test]
fn test_console_session_mode_client_uses_interactive_mode() {
    // Explicit client mode is a runtime surface, so it should inherit the
    // detected terminal mode and start the passive/client dashboard.
    let result = initial_console_session_mode_for_surface(
        Some(RuntimeSurface::Client),
        ConsoleSessionMode::InteractiveDashboard,
    );
    assert_eq!(result, ConsoleSessionMode::InteractiveDashboard);
}

#[test]
fn test_console_session_mode_no_explicit_surface_uses_none() {
    // When explicit_surface is None, should use None mode
    let result =
        initial_console_session_mode_for_surface(None, ConsoleSessionMode::InteractiveDashboard);
    assert_eq!(result, ConsoleSessionMode::None);
}

// ── Bootstrap-proxy gate ────────────────────────────────────────────
//
// Regression history: commit 1bd62389 ("feat(hardware): add hardware
// information enrichment") changed the serve --auto path so its join
// candidates land in `auto_join_candidates` instead of `options.join`. The
// bootstrap proxy gate keyed off `options.join` and silently stopped firing
// for `serve --auto`, leaving :9337 unbound while the local model
// loaded. These tests pin the gate so both client and serve get the
// bootstrap proxy whenever there is a candidate to tunnel to.
