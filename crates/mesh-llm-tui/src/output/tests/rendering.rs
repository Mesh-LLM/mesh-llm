use super::*;

#[test]
pub(super) fn tui_event_line_uses_compact_timestamp_level_message_layout() {
    let line = event_line(
        &MeshEventState {
            timestamp: "12:34:56".to_string(),
            level: OutputLevel::Info,
            summary: "✅   joined   mesh   poker-night".to_string(),
        },
        80,
    );

    assert_eq!(
        spans_plain_text(&line.spans),
        "12:34:56 OK    joined mesh poker-night"
    );
}
#[test]
pub(super) fn tui_full_screen_events_wraps_long_log_lines() {
    let mut formatter = InteractiveDashboardFormatter::default();
    formatter.handle_tui_event(TuiEvent::Resize {
        columns: 72,
        rows: 10,
    });
    formatter
        .handle_output_event(&info_event(
            "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu unique-wrap-tail",
        ))
        .expect("event render should succeed");
    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Enter));

    let rendered = render_tui_frame_snapshot(&formatter.state, 72, 10);

    assert!(rendered.contains("fullscreen  Esc=Back"));
    assert!(rendered.contains("alpha beta gamma"));
    assert!(
        rendered.contains("unique-wrap-tail"),
        "expected full-screen log panel to wrap the long event instead of truncating it: {rendered}"
    );
    assert!(!rendered.contains("Loaded Models"));
    assert!(!rendered.contains("[Tab] Next"));
}
#[test]
pub(super) fn tui_scrollbar_event_list_renders_standalone_vertical_slice() {
    let events = sample_mesh_event_states(7);

    let rendered = render_scrollbar_event_list_widget_snapshot(&events, 2, 42, 3);

    assert!(rendered.contains("event-02 tdd-scroll-marker"));
    assert!(rendered.contains("event-03 tdd-scroll-marker"));
    assert!(rendered.contains("event-04 tdd-scroll-marker"));
    assert!(!rendered.contains("event-01 tdd-scroll-marker"));
    assert!(!rendered.contains("event-05 tdd-scroll-marker"));
    assert!(
        rendered.lines().all(|line| !line.contains('─')),
        "new event list should use the vertical scrollbar only: {rendered}"
    );
    assert!(
        rendered
            .lines()
            .any(|line| line.ends_with('│') || line.ends_with('█')),
        "expected a vertical scrollbar in the rightmost column: {rendered}"
    );
}
#[test]
pub(super) fn tui_scrollbar_event_list_reaches_bottom_at_last_slice() {
    let events = sample_mesh_event_states(7);

    let rendered = render_scrollbar_event_list_widget_snapshot(&events, 4, 42, 3);
    let scrollbar_column: String = rendered
        .lines()
        .map(|line| line.chars().last().unwrap_or(' '))
        .collect();

    assert!(rendered.contains("event-04 tdd-scroll-marker"));
    assert!(rendered.contains("event-05 tdd-scroll-marker"));
    assert!(rendered.contains("event-06 tdd-scroll-marker"));
    assert!(
        scrollbar_column.ends_with('█'),
        "expected scrollbar thumb to reach bottom for final visible slice: {rendered}"
    );
}
#[test]
pub(super) fn tui_events_panel_can_swap_between_scrollbar_widget_and_legacy_list() {
    let mut formatter = InteractiveDashboardFormatter::default();
    formatter
        .state
        .reduce(DashboardAction::Resize(DashboardLayoutState::new(
            4, 2, 2, 2, 2,
        )));
    for index in 0..6 {
        formatter
            .handle_output_event(&info_event(format!("event-{index:02} swap-marker")))
            .expect("event render should succeed");
    }

    let scrollbar_rendered = render_events_panel_with_renderer_snapshot(
        &formatter.state,
        TuiEventListRenderer::Scrollbar,
        72,
        8,
    );
    let legacy_rendered = render_events_panel_with_renderer_snapshot(
        &formatter.state,
        TuiEventListRenderer::Legacy,
        72,
        8,
    );

    assert!(scrollbar_rendered.contains("event-05 swap-marker"));
    assert!(legacy_rendered.contains("event-05 swap-marker"));
}
#[test]
pub(super) fn tui_events_scrollbar_arrows_scroll_text_line_by_line() {
    let mut formatter = InteractiveDashboardFormatter::default();
    formatter
        .state
        .reduce(DashboardAction::Resize(DashboardLayoutState::new(
            4, 2, 2, 2, 2,
        )));

    for index in 0..8 {
        formatter
            .handle_output_event(&info_event(format!("event-{index:02} line-scroll-marker")))
            .expect("event render should succeed");
    }

    assert!(formatter.state.events_follow);
    assert_eq!(
        formatter
            .state
            .panel_view_state(DashboardPanel::Events)
            .scroll_offset,
        4
    );

    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Up));
    assert!(!formatter.state.events_follow);
    assert_eq!(
        formatter
            .state
            .panel_view_state(DashboardPanel::Events)
            .scroll_offset,
        3,
        "Up should scroll the event text up by exactly one line"
    );

    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Up));
    assert_eq!(
        formatter
            .state
            .panel_view_state(DashboardPanel::Events)
            .scroll_offset,
        2,
        "a second Up press should scroll one more line"
    );

    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Down));
    assert_eq!(
        formatter
            .state
            .panel_view_state(DashboardPanel::Events)
            .scroll_offset,
        3,
        "Down should scroll the event text down by exactly one line"
    );

    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Down));
    assert_eq!(
        formatter
            .state
            .panel_view_state(DashboardPanel::Events)
            .scroll_offset,
        4
    );
    assert!(
        formatter.state.events_follow,
        "scrolling down to the newest event should re-enable follow mode"
    );
}
#[test]
pub(super) fn tui_events_up_repaints_actual_viewport_without_top_pinning() {
    let mut formatter = InteractiveDashboardFormatter::default();
    formatter
        .state
        .reduce(DashboardAction::Resize(DashboardLayoutState::new(
            5, 2, 2, 2, 2,
        )));

    for index in 0..12 {
        formatter
            .handle_output_event(&info_event(format!("event-{index:02} no-pin-marker")))
            .expect("event render should succeed");
    }

    let backend = TestBackend::new(90, 14);
    let mut terminal = Terminal::new(backend).expect("test backend should initialize");
    let title_area = Rect::new(0, 0, 90, 1);
    let body_area = Rect::new(0, 1, 90, 12);
    terminal
        .draw(|frame| render_events_panel(frame, &formatter.state, title_area, body_area))
        .expect("initial event panel render should succeed");

    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Up));
    assert!(!formatter.state.events_follow);

    terminal
        .draw(|frame| render_events_panel(frame, &formatter.state, title_area, body_area))
        .expect("up-arrow event panel render should succeed");

    let buffer = terminal.backend().buffer();
    let rendered_lines: Vec<String> = (0..14)
        .map(|y| {
            (0..90)
                .map(|x| buffer[(x, y)].symbol())
                .collect::<String>()
                .trim_end()
                .to_string()
        })
        .collect();
    let rendered = rendered_lines.join("\n");

    assert!(
        rendered.contains("event-01 no-pin-marker"),
        "event renderer should use the actual panel height, not the stale state viewport: {rendered_lines:?}"
    );
    assert!(
        rendered.contains("event-11 no-pin-marker"),
        "latest row should remain visible after one Up press: {rendered_lines:?}"
    );
    assert!(
        !rendered.contains("event-00 no-pin-marker"),
        "top row should scroll out instead of pinning to the panel top: {rendered_lines:?}"
    );
    for index in 1..=11 {
        let marker = format!("event-{index:02} no-pin-marker");
        assert_eq!(
            rendered.matches(&marker).count(),
            1,
            "event rows should be painted exactly once after Up, without duplicated stale text: {rendered_lines:?}"
        );
    }
}
#[test]
pub(super) fn tui_events_scroll_repaints_long_rows_cleanly() {
    let mut formatter = InteractiveDashboardFormatter::default();
    formatter
        .state
        .reduce(DashboardAction::Resize(DashboardLayoutState::new(
            2, 1, 1, 1, 1,
        )));

    formatter
        .handle_output_event(&info_event("short pre-scroll"))
        .expect("event render should succeed");
    formatter
        .handle_output_event(&info_event(
            "this row is intentionally long so scrolling has to repaint cleanly unique-tail-marker",
        ))
        .expect("event render should succeed");
    formatter
        .handle_output_event(&info_event("short post-scroll"))
        .expect("event render should succeed");

    let initial_state = formatter.state.clone();
    let mut scrolled_state = initial_state.clone();
    scrolled_state.events_follow = false;
    let events_view = scrolled_state.panel_view_state_mut(DashboardPanel::Events);
    events_view.scroll_offset = 0;
    events_view.selected_row = Some(0);

    let backend = TestBackend::new(72, 16);
    let mut terminal = Terminal::new(backend).expect("test backend should initialize");
    terminal
        .draw(|frame| render_tui_frame(frame, &initial_state))
        .expect("initial frame render should succeed");
    terminal
        .draw(|frame| render_tui_frame(frame, &scrolled_state))
        .expect("scrolled frame render should succeed");

    let buffer = terminal.backend().buffer();
    let rendered_lines: Vec<String> = (0..16)
        .map(|y| {
            (0..72)
                .map(|x| buffer[(x, y)].symbol())
                .collect::<String>()
                .trim_end()
                .to_string()
        })
        .collect();
    let scrolled_event_line = rendered_lines
        .iter()
        .find(|line| line.contains("short pr"))
        .unwrap_or_else(|| {
            panic!("expected the scrolled short event to be visible: {rendered_lines:?}")
        });
    assert!(
        !scrolled_event_line.contains("unique-tail-marker"),
        "expected long event text to be truncated before repaint: {scrolled_event_line}"
    );
    assert!(
        rendered_lines
            .iter()
            .all(|line| !line.contains("unique-tail-marker")),
        "expected no stale long-event text after scrolling: {rendered_lines:?}"
    );
}
#[test]
pub(super) fn tui_events_filter_empty_state_repaints_over_previous_rows() {
    let mut formatter = InteractiveDashboardFormatter::default();
    formatter
        .state
        .reduce(DashboardAction::Resize(DashboardLayoutState::new(
            4, 2, 2, 2, 2,
        )));

    formatter
        .handle_output_event(&info_event("sticky-filter-marker before-filter"))
        .expect("event render should succeed");
    formatter
        .handle_output_event(&info_event("another visible row before-filter"))
        .expect("event render should succeed");

    let backend = TestBackend::new(80, 18);
    let mut terminal = Terminal::new(backend).expect("test backend should initialize");
    terminal
        .draw(|frame| render_tui_frame(frame, &formatter.state))
        .expect("initial frame render should succeed");

    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Char('/')));
    for ch in "zzzz".chars() {
        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Char(ch)));
    }

    assert_eq!(formatter.state.filtered_mesh_events().len(), 0);
    terminal
        .draw(|frame| render_tui_frame(frame, &formatter.state))
        .expect("filtered frame render should succeed");

    let buffer = terminal.backend().buffer();
    let rendered_lines: Vec<String> = (0..18)
        .map(|y| {
            (0..80)
                .map(|x| buffer[(x, y)].symbol())
                .collect::<String>()
                .trim_end()
                .to_string()
        })
        .collect();
    assert!(
        rendered_lines
            .iter()
            .any(|line| line.contains("no events match")),
        "expected filtered empty-state message: {rendered_lines:?}"
    );
    assert!(
        rendered_lines
            .iter()
            .all(|line| !line.contains("sticky-filter-marker")),
        "expected filtered empty state to repaint over stale event rows: {rendered_lines:?}"
    );
}
#[test]
pub(super) fn tui_events_live_filter_repaints_to_matching_badge_rows() {
    let mut formatter = InteractiveDashboardFormatter::default();
    formatter
        .state
        .reduce(DashboardAction::Resize(DashboardLayoutState::new(
            4, 2, 2, 2, 2,
        )));

    formatter
        .handle_output_event(&info_event("plain operational live-filter-marker"))
        .expect("event render should succeed");
    formatter
        .handle_output_event(&info_event("ok heartbeat stale-ok-marker"))
        .expect("event render should succeed");
    formatter
        .handle_output_event(&OutputEvent::Warning {
            message: "capacity stale-warn-marker".to_string(),
            context: None,
        })
        .expect("event render should succeed");

    let backend = TestBackend::new(80, 18);
    let mut terminal = Terminal::new(backend).expect("test backend should initialize");
    terminal
        .draw(|frame| render_tui_frame(frame, &formatter.state))
        .expect("initial frame render should succeed");

    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Char('/')));
    for ch in "info".chars() {
        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Char(ch)));
    }

    assert_eq!(formatter.state.filtered_mesh_events().len(), 1);
    terminal
        .draw(|frame| render_tui_frame(frame, &formatter.state))
        .expect("filtered frame render should succeed");

    let buffer = terminal.backend().buffer();
    let rendered_lines: Vec<String> = (0..18)
        .map(|y| {
            (0..80)
                .map(|x| buffer[(x, y)].symbol())
                .collect::<String>()
                .trim_end()
                .to_string()
        })
        .collect();
    assert!(
        rendered_lines
            .iter()
            .any(|line| line.contains("INFO  plain operati")),
        "expected INFO badge row to remain visible: {rendered_lines:?}"
    );
    assert!(
        rendered_lines
            .iter()
            .all(|line| !line.contains("stale-ok-marker") && !line.contains("stale-warn-marker")),
        "expected non-matching rows to be repainted away: {rendered_lines:?}"
    );
}
#[test]
pub(super) fn tui_events_snapshot_preserves_timestamp_readability() {
    let mut formatter = InteractiveDashboardFormatter::default();
    formatter
        .state
        .reduce(DashboardAction::Resize(DashboardLayoutState::new(
            4, 2, 2, 2, 2,
        )));
    formatter
        .handle_output_event(&OutputEvent::DiscoveryJoined {
            mesh: "poker-night".to_string(),
        })
        .expect("event render should succeed");

    let rendered = render_tui_events_snapshot(&formatter.state, 48, 20);
    let event_line = rendered
        .lines()
        .find(|line| line.contains("joined mesh poker-night"))
        .expect("expected rendered mesh event line");
    let timestamp = event_line
        .split_whitespace()
        .find(|token| token.len() == 8 && token.chars().nth(2) == Some(':'))
        .expect("expected timestamp token");
    assert_hh_mm_ss(timestamp);
    assert!(
        event_line.contains(" OK    joined mesh poker-night"),
        "expected compact log row in {event_line}"
    );
    assert!(event_line.contains("joined mesh poker-night"));
    assert!(!event_line.contains("✅"));
}
#[test]
pub(super) fn tui_list_scrollbar_layout_reserves_one_column_gutter_on_overflow() {
    let inner_area = Rect::new(12, 4, 18, 5);

    assert_eq!(
        tui_list_scrollbar_layout(inner_area, 9, 5),
        TuiListScrollbarLayout {
            list_area: Rect::new(12, 4, 17, 5),
            scrollbar_area: Some(Rect::new(29, 4, 1, 5)),
        }
    );
    assert_eq!(
        tui_list_scrollbar_layout(inner_area, 5, 5),
        TuiListScrollbarLayout {
            list_area: inner_area,
            scrollbar_area: None,
        }
    );
}
#[test]
pub(super) fn tui_list_scrollbar_state_uses_row_count_and_clamps_to_max_offset() {
    let state = tui_list_scrollbar_state(10, 3, 99);

    assert_eq!(
        state,
        ratatui::widgets::ScrollbarState::new(10)
            .position(9)
            .viewport_content_length(3)
    );
}
#[test]
pub(super) fn tui_layout_uses_join_token_band_with_nested_process_tables() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
        120, 24,
    )));

    let areas = tui_layout(Rect::new(0, 0, 120, 24), &state);

    assert_join_token_layout(&state, &areas);
    assert_process_table_layout(&state, &areas);
}
#[test]
pub(super) fn tui_main_columns_pin_events_and_split_remaining_width() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
        121, 24,
    )));

    let areas = tui_layout(Rect::new(0, 0, 121, 24), &state);
    let events_width = combine_panel_rect(areas.events.0, areas.events.1).width;
    let processes_width = areas.processes.width;
    let models_width = combine_panel_rect(areas.models.0, areas.models.1).width;
    let expected_events_width = areas
        .main_body
        .width
        .saturating_mul(PRETTY_TUI_EVENTS_COLUMN_PERCENT)
        / 100;

    assert!(
        events_width.abs_diff(expected_events_width) <= 1,
        "Mesh Events should stay at roughly {PRETTY_TUI_EVENTS_COLUMN_PERCENT}% of the main body"
    );
    assert!(
        processes_width.abs_diff(models_width) <= 1,
        "Loaded Models and Processes should split the remaining width evenly"
    );
    assert_eq!(
        events_width
            .saturating_add(processes_width)
            .saturating_add(models_width),
        areas.main_body.width
    );
}
#[test]
pub(super) fn tui_layout_bottom_anchors_dashboard_with_top_slack() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
        120, 24,
    )));

    let area = Rect::new(0, 0, 120, 48);
    let areas = tui_layout(area, &state);

    assert!(
        areas.loading.is_some(),
        "expected unused top space above dashboard"
    );
    assert_eq!(areas.status_bar.bottom(), area.bottom());
    assert!(
        areas.main_body.y > area.y,
        "dashboard should sit at the bottom"
    );
}
#[test]
pub(super) fn tui_band_heights_never_exceed_terminal_budget() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
        120, 12,
    )));

    let area = Rect::new(0, 0, 120, 12);
    let band_heights = tui_band_heights(area, &state);
    let areas = tui_layout(area, &state);
    let requests_inner = tui_panel_block(&state, DashboardPanel::Requests)
        .inner(combine_panel_rect(areas.requests.0, areas.requests.1));

    assert_eq!(
        band_heights
            .join_token
            .saturating_add(band_heights.main_body)
            .saturating_add(band_heights.requests)
            .saturating_add(band_heights.status),
        area.height,
        "expected top-level bands to fit the frame budget without overlapping pane borders"
    );
    assert_eq!(areas.status_bar.bottom(), area.bottom());
    assert!(
        requests_inner.height >= 3,
        "expected summary + at least two graph rows in constrained layout"
    );
}
#[test]
pub(super) fn tui_invite_token_event_populates_join_token_panel() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::OutputEvent(OutputEvent::InviteToken {
        token: "mesh-invite-token-123".to_string(),
        mesh_id: "mesh-alpha".to_string(),
        mesh_name: None,
    }));

    let join_token = state
        .join_token
        .as_ref()
        .expect("invite token event should populate dashboard join token state");
    assert_eq!(join_token.token, "mesh-invite-token-123");
    assert_eq!(join_token.mesh_id, "mesh-alpha");
    assert_eq!(join_token.copy_status, DashboardJoinTokenCopyStatus::Idle);

    let rendered = render_tui_frame_snapshot(&state, 120, 24);
    let (join_index, _) = find_rendered_line(&rendered, "Join Token");
    let (events_index, _) = find_rendered_line(&rendered, "Mesh Events");
    assert!(
        join_index < events_index,
        "join token panel should render above existing dashboard panels\n{rendered}"
    );
    assert!(rendered.contains("mesh-invite-token-123"));
    assert!(rendered.contains("Copy"));

    let lines: Vec<&str> = rendered.lines().collect();
    assert!(
        lines[join_index.saturating_add(1)]
            .trim_matches(|ch| ch == '│' || ch == ' ')
            .is_empty(),
        "join token panel should leave one blank body row above the token\n{rendered}"
    );
    assert!(
        lines[join_index.saturating_add(3)]
            .trim_matches(|ch| ch == '│' || ch == ' ')
            .is_empty(),
        "join token panel should leave one blank body row below the token\n{rendered}"
    );
}
#[test]
pub(super) fn tui_join_token_copy_button_hit_test_uses_latest_resize() {
    let mut state = DashboardState::default();
    state.apply_tui_event(TuiEvent::Resize {
        columns: 120,
        rows: 24,
    });
    state.reduce(DashboardAction::OutputEvent(OutputEvent::InviteToken {
        token: "mesh-invite-token-123".to_string(),
        mesh_id: "mesh-alpha".to_string(),
        mesh_name: None,
    }));
    let areas = tui_layout(Rect::new(0, 0, 120, 24), &state);

    assert!(state.join_token_copy_button_contains(
        areas.join_token_copy_button.x,
        areas.join_token_copy_button.y
    ));
    assert!(!state.join_token_copy_button_contains(0, 0));
}
#[test]
pub(super) fn tui_join_token_is_selectable_with_backtab_and_mouse() {
    let mut state = DashboardState::default();
    state.apply_tui_event(TuiEvent::Resize {
        columns: 120,
        rows: 24,
    });
    assert_eq!(state.panel_focus, DashboardPanel::Events);

    state.apply_tui_event(TuiEvent::Key(TuiKeyEvent::BackTab));
    assert_eq!(state.panel_focus, DashboardPanel::JoinToken);

    state.panel_focus = DashboardPanel::Events;
    let areas = tui_layout(Rect::new(0, 0, 120, 24), &state);
    state.apply_tui_event(TuiEvent::MouseDown {
        column: areas.join_token_panel.x.saturating_add(1),
        row: areas.join_token_panel.y.saturating_add(1),
    });
    assert_eq!(state.panel_focus, DashboardPanel::JoinToken);

    let rendered = render_tui_frame_snapshot(&state, 120, 24);
    assert!(
        rendered.contains("▶ Join Token"),
        "focused join-token panel should use the standard focus marker\n{rendered}"
    );
}
#[test]
pub(super) fn tui_join_token_copy_shortcut_does_not_require_panel_focus() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::OutputEvent(OutputEvent::InviteToken {
        token: "mesh-invite-token-123".to_string(),
        mesh_id: "mesh-alpha".to_string(),
        mesh_name: None,
    }));
    state.panel_focus = DashboardPanel::Events;

    assert!(state.join_token_copy_shortcut_enabled());

    state.events_filter.editing = true;
    assert!(!state.join_token_copy_shortcut_enabled());
}
#[test]
pub(super) fn tui_join_token_scrolls_horizontally_with_left_right_keys() {
    let mut state = DashboardState::default();
    state.apply_tui_event(TuiEvent::Resize {
        columns: 48,
        rows: 24,
    });
    state.reduce(DashboardAction::OutputEvent(OutputEvent::InviteToken {
        token: "mesh-invite-token-abcdefghijklmnopqrstuvwxyz-0123456789".to_string(),
        mesh_id: "mesh-alpha".to_string(),
        mesh_name: None,
    }));
    state.panel_focus = DashboardPanel::JoinToken;

    assert_eq!(
        state
            .panel_view_state(DashboardPanel::JoinToken)
            .scroll_offset,
        0
    );
    state.apply_tui_event(TuiEvent::Key(TuiKeyEvent::Right));
    assert_eq!(
        state
            .panel_view_state(DashboardPanel::JoinToken)
            .scroll_offset,
        1
    );
    state.apply_tui_event(TuiEvent::Key(TuiKeyEvent::Left));
    assert_eq!(
        state
            .panel_view_state(DashboardPanel::JoinToken)
            .scroll_offset,
        0
    );

    state.apply_tui_event(TuiEvent::Key(TuiKeyEvent::Char('G')));
    let view = state.panel_view_state(DashboardPanel::JoinToken);
    assert!(
        view.scroll_offset > 0,
        "G should jump to the end of the horizontally scrollable token"
    );
    state.apply_tui_event(TuiEvent::Key(TuiKeyEvent::Char('g')));
    assert_eq!(
        state
            .panel_view_state(DashboardPanel::JoinToken)
            .scroll_offset,
        0
    );
}
#[test]
pub(super) fn join_token_slice_indicates_hidden_content() {
    let token = "abcdefghij";

    assert_eq!(join_token_visible_slice(token, 0, 5), "abcd…");
    assert_eq!(join_token_visible_slice(token, 2, 5), "…cde…");
    assert_eq!(join_token_visible_slice(token, 5, 5), "…fghi");
    assert_eq!(join_token_visible_slice(token, 0, 10), token);
}
#[test]
pub(super) fn join_token_slice_handles_narrow_widths() {
    assert_eq!(join_token_visible_slice("", 0, 5), "");
    assert_eq!(join_token_visible_slice("abcdef", 0, 0), "");
    assert_eq!(join_token_visible_slice("abcdef", 0, 1), "…");
    assert_eq!(join_token_visible_slice("abcdef", 2, 1), "…");
    assert_eq!(join_token_visible_slice("abcdef", 5, 1), "…");
}
#[test]
pub(super) fn tui_join_token_status_renders_on_right_title_bar() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::OutputEvent(OutputEvent::InviteToken {
        token: "mesh-invite-token-123".to_string(),
        mesh_id: "mesh-alpha".to_string(),
        mesh_name: None,
    }));
    state.reduce(DashboardAction::SetJoinTokenCopyStatus(
        DashboardJoinTokenCopyStatus::Copied { at: Instant::now() },
    ));

    let rendered = render_tui_frame_snapshot(&state, 120, 24);
    let (_, join_title_line) = find_rendered_line(&rendered, "Join Token");
    let mesh_index = join_title_line
        .find("mesh=mesh-alpha")
        .expect("left title should include mesh id");
    let copied_index = join_title_line
        .rfind("copied to clipboard")
        .expect("right title should include copy status");
    assert!(
        mesh_index < 40,
        "mesh id should stay near the left title bar"
    );
    assert!(
        copied_index > 90,
        "copy status should be aligned toward the far right title bar: {join_title_line:?}"
    );
    assert!(
        rendered.contains("Copied"),
        "copy status should be visible on the copy control too\n{rendered}"
    );
}
#[test]
pub(super) fn tui_join_token_copy_status_clears_after_ttl() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::OutputEvent(OutputEvent::InviteToken {
        token: "mesh-invite-token-123".to_string(),
        mesh_id: "mesh-alpha".to_string(),
        mesh_name: None,
    }));
    let now = Instant::now();
    state.reduce(DashboardAction::SetJoinTokenCopyStatus(
        DashboardJoinTokenCopyStatus::Copied {
            at: now - Duration::from_secs(1),
        },
    ));

    assert!(!state.clear_expired_join_token_copy_status(now));
    assert!(matches!(
        state
            .join_token
            .as_ref()
            .map(|join_token| &join_token.copy_status),
        Some(DashboardJoinTokenCopyStatus::Copied { .. })
    ));

    state.reduce(DashboardAction::SetJoinTokenCopyStatus(
        DashboardJoinTokenCopyStatus::Failed {
            message: "clipboard unavailable".to_string(),
            at: now - PRETTY_TUI_JOIN_TOKEN_COPY_STATUS_TTL - Duration::from_millis(1),
        },
    ));

    assert!(state.clear_expired_join_token_copy_status(now));
    assert_eq!(
        state
            .join_token
            .as_ref()
            .map(|join_token| &join_token.copy_status),
        Some(&DashboardJoinTokenCopyStatus::Idle)
    );
}
#[test]
pub(super) fn tui_full_screen_join_token_wraps_long_token() {
    let mut state = DashboardState::default();
    state.apply_tui_event(TuiEvent::Resize {
        columns: 64,
        rows: 16,
    });
    state.reduce(DashboardAction::OutputEvent(OutputEvent::InviteToken {
        token: "mesh-invite-token-abcdefghijklmnopqrstuvwxyz-0123456789-tail".to_string(),
        mesh_id: "mesh-alpha".to_string(),
        mesh_name: None,
    }));
    state.panel_focus = DashboardPanel::JoinToken;
    state.apply_tui_event(TuiEvent::Key(TuiKeyEvent::Enter));
    assert_eq!(state.full_screen_panel, Some(DashboardPanel::JoinToken));

    let rendered = render_tui_frame_snapshot(&state, 64, 16);

    assert!(
        rendered.contains("789-tail"),
        "expected full-screen join-token panel to wrap instead of slicing the token tail\n{rendered}"
    );
}
#[test]
pub(super) fn tui_join_token_title_includes_mesh_name_when_available() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::OutputEvent(OutputEvent::InviteToken {
        token: "mesh-invite-token-123".to_string(),
        mesh_id: "abcd1230".to_string(),
        mesh_name: Some("mymesh".to_string()),
    }));

    let title = join_token_panel_left_title(&state, ' ');

    assert!(title.contains("mesh=mymesh (abcd1230)"));
}
#[test]
pub(super) fn tui_join_token_title_uses_mesh_id_without_name() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::OutputEvent(OutputEvent::InviteToken {
        token: "mesh-invite-token-123".to_string(),
        mesh_id: "abcde1230".to_string(),
        mesh_name: None,
    }));

    let title = join_token_panel_left_title(&state, ' ');

    assert!(title.contains("mesh=abcde1230"));
    assert!(!title.contains('('));
}
#[test]
pub(super) fn tui_frame_clears_stale_join_token_rows_between_draws() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::OutputEvent(OutputEvent::InviteToken {
        token: "mesh-invite-token-123".to_string(),
        mesh_id: "mesh-alpha".to_string(),
        mesh_name: None,
    }));

    let backend = ratatui::backend::TestBackend::new(120, 24);
    let mut terminal = Terminal::new(backend).expect("test backend should initialize");
    terminal
        .draw(|frame| render_tui_frame(frame, &state))
        .expect("initial frame render should succeed");

    terminal
        .draw(|frame| {
            frame.render_widget(
                Paragraph::new("stale Join Token mesh=mesh-alpha token mesh-invite-token-123 Copy"),
                Rect::new(0, 0, 120, 1),
            );
        })
        .expect("stale frame render should succeed");

    let loading_state = DashboardState {
        model_progress: Some(ModelProgressState {
            label: "qwen2.5".to_string(),
            file: Some("qwen.gguf".to_string()),
            downloaded_bytes: Some(1),
            total_bytes: Some(10),
            status: ModelProgressStatus::Downloading,
        }),
        ..DashboardState::default()
    };

    terminal
        .draw(|frame| render_tui_frame(frame, &loading_state))
        .expect("loading frame render should succeed");

    let buffer = terminal.backend().buffer();
    let mut rendered = String::new();
    for y in 0..24 {
        for x in 0..120 {
            rendered.push_str(buffer[(x, y)].symbol());
        }
        rendered.push('\n');
    }

    assert!(
        !rendered.contains("stale Join Token"),
        "full-frame redraw should clear stale join-token rows from previous frames\n{rendered}"
    );
    assert!(
        !rendered.contains("mesh-invite-token-123"),
        "full-frame redraw should clear stale token text from previous frames\n{rendered}"
    );
}
#[test]
pub(super) fn loading_progress_bar_keeps_zero_empty_and_positive_visible() {
    assert_eq!(loading_progress_bar(0.0, 8), "░░░░░░░░");
    assert_eq!(loading_progress_bar(0.01, 8), "█░░░░░░░");
}
#[test]
pub(super) fn tui_process_tables_render_empty_states_without_collapsing() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
        120, 24,
    )));

    let areas = tui_layout(Rect::new(0, 0, 120, 24), &state);
    let llama_inner = tui_panel_block(&state, DashboardPanel::LlamaCpp).inner(combine_panel_rect(
        areas.llama_processes.0,
        areas.llama_processes.1,
    ));
    let webserver_inner = tui_panel_block(&state, DashboardPanel::Webserver).inner(
        combine_panel_rect(areas.webserver_processes.0, areas.webserver_processes.1),
    );
    assert_eq!(
        llama_inner.height as usize,
        state.panel_layout.rows_for(DashboardPanel::LlamaCpp)
    );
    assert_eq!(
        webserver_inner.height as usize,
        state.panel_layout.rows_for(DashboardPanel::Webserver)
    );

    let rendered = render_tui_frame_snapshot(&state, 120, 24);
    assert!(rendered.contains("Processes"));
    assert!(rendered.contains("llama.cpp"));
    assert!(rendered.contains("mesh-llm"));
    assert!(rendered.contains("(no llama.cpp processes yet)"));
    assert!(rendered.contains("(no webserver processes yet)"));
}
#[test]
pub(super) fn tui_process_tables_render_headers_and_joined_model_metadata() {
    let mut formatter = InteractiveDashboardFormatter::default();
    let mut process_row = sample_process_row("llama-server", 8001);
    process_row.backend = "metal".to_string();
    let mut model_row = sample_model_row("Mistral-7B", 8001);
    model_row.device = Some("GPU0".to_string());
    model_row.ctx_size = Some(8192);
    formatter.handle_snapshot(DashboardSnapshot {
        llama_process_rows: vec![process_row],
        webserver_rows: vec![sample_endpoint_row("Console", 3131)],
        loaded_model_rows: vec![model_row],
        ..snapshot_fixture(0, 30)
    });
    formatter.handle_tui_event(TuiEvent::Resize {
        columns: 240,
        rows: 30,
    });

    let rendered = render_tui_frame_snapshot(&formatter.state, 240, 30);
    let (_, process_header_line) = find_rendered_line(&rendered, "MODEL");
    assert!(process_header_line.contains("PID"));
    assert!(process_header_line.contains("PORT"));
    assert!(process_header_line.contains("STATE"));
    assert!(!process_header_line.contains("SLOTS"));
    assert!(rendered.contains("Mistral-7B"));
    assert_eq!(PRETTY_TUI_WEBSERVER_PROCESS_HEADER_LABEL, "PROCESSES");
    assert!(!rendered.contains("ENDPOINT"));
    assert!(rendered.contains("PID"));
    assert!(!rendered.contains("URL"));
    assert!(rendered.contains("mesh-llm Processes"));
}
#[test]
pub(super) fn tui_llama_process_table_omits_model_variant_suffix() {
    let mut formatter = InteractiveDashboardFormatter::default();
    let mut process_row = sample_process_row("llama-server", 8001);
    process_row.name = "unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL".to_string();
    formatter.handle_snapshot(DashboardSnapshot {
        llama_process_rows: vec![process_row],
        ..snapshot_fixture(0, 30)
    });
    formatter.handle_tui_event(TuiEvent::Resize {
        columns: 160,
        rows: 30,
    });

    let rendered = render_tui_frame_snapshot(&formatter.state, 160, 30);

    assert!(
        rendered.contains("unsloth/Qwen3.5-4B-G"),
        "expected truncated base model ref in llama.cpp process table: {rendered}"
    );
    assert!(
        !rendered.contains(":UD-Q4_K_XL"),
        "TUI should omit GGUF variant suffix from llama.cpp process model names: {rendered}"
    );
}
#[test]
pub(super) fn tui_process_table_widths_give_text_columns_leftover_space() {
    let [model_width, pid_width, port_width, status_width] = llama_process_column_widths(52);

    assert_eq!(pid_width, 5);
    assert_eq!(port_width, 5);
    assert_eq!(status_width, RuntimeStatus::NotReady.as_str().len());
    assert_eq!(model_width, 28);

    let rows = [DashboardEndpointRow {
        label: "Plugin: browser-tools".to_string(),
        status: RuntimeStatus::Ready,
        url: "browser-tools".to_string(),
        port: 0,
        pid: Some(4321),
    }];
    let [label_width, web_pid_width, web_port_width, web_status_width] =
        webserver_process_column_widths(52);

    assert_eq!(web_pid_width, 5);
    assert_eq!(web_port_width, 5);
    assert_eq!(web_status_width, RuntimeStatus::NotReady.as_str().len());
    assert_eq!(label_width, 28);
    assert!(label_width >= rows[0].label.len());
    assert!(label_width >= PRETTY_TUI_WEBSERVER_PROCESS_HEADER_LABEL.len());
}
#[test]
pub(super) fn tui_dashboard_process_table_renders_missing_pid_as_dash() {
    assert_eq!(format_dashboard_pid(None), "-");
    assert_eq!(format_dashboard_pid(Some(4321)), "4321");
}
#[test]
pub(super) fn tui_process_table_renders_six_digit_pid_without_truncation() {
    let mut formatter = InteractiveDashboardFormatter::default();
    formatter.handle_snapshot(DashboardSnapshot {
        webserver_rows: vec![DashboardEndpointRow {
            label: "Plugin: blobstore".to_string(),
            status: RuntimeStatus::Ready,
            url: "blobstore".to_string(),
            port: 0,
            pid: Some(132098),
        }],
        ..snapshot_fixture(0, 30)
    });
    formatter.handle_tui_event(TuiEvent::Resize {
        columns: 120,
        rows: 24,
    });

    let rendered = render_tui_frame_snapshot(&formatter.state, 120, 24);

    assert!(
        rendered.contains("132098"),
        "expected full six-digit PID in process table: {rendered}"
    );
}
#[test]
pub(super) fn tui_hjkl_and_arrows_navigate_focused_panel_without_changing_focus() {
    let mut formatter = InteractiveDashboardFormatter::default();
    formatter.handle_snapshot(DashboardSnapshot {
        loaded_model_rows: vec![
            sample_model_row("Model-0", 4000),
            sample_model_row("Model-1", 4001),
            sample_model_row("Model-2", 4002),
        ],
        ..snapshot_fixture(0, 30)
    });
    formatter.handle_tui_event(TuiEvent::Resize {
        columns: 140,
        rows: 18,
    });
    formatter.state.panel_layout.widgets[DashboardPanel::Models.index()].selectable = true;
    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Tab));
    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Tab));
    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Tab));
    assert_eq!(formatter.state.panel_focus, DashboardPanel::Models);

    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Char('l')));
    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Right));
    assert_eq!(formatter.state.panel_focus, DashboardPanel::Models);
    assert_eq!(
        formatter
            .state
            .panel_view_state(DashboardPanel::Models)
            .selected_row,
        Some(2)
    );

    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Char('h')));
    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Left));
    assert_eq!(formatter.state.panel_focus, DashboardPanel::Models);
    assert_eq!(
        formatter
            .state
            .panel_view_state(DashboardPanel::Models)
            .selected_row,
        Some(0)
    );
}
#[test]
pub(super) fn tui_up_down_cycle_request_window_when_requests_panel_is_focused() {
    let mut formatter = InteractiveDashboardFormatter::default();
    formatter.handle_tui_event(TuiEvent::Resize {
        columns: 140,
        rows: 18,
    });
    for _ in 0..4 {
        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Tab));
    }
    assert_eq!(formatter.state.panel_focus, DashboardPanel::Requests);
    assert_eq!(
        formatter.state.request_window,
        DashboardRequestWindow::SixtySeconds
    );

    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Down));
    assert_eq!(
        formatter.state.request_window,
        DashboardRequestWindow::TenMinutes
    );
    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Down));
    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Down));
    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Down));
    assert_eq!(
        formatter.state.request_window,
        DashboardRequestWindow::TwentyFourHours
    );

    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Down));
    assert_eq!(
        formatter.state.request_window,
        DashboardRequestWindow::TwentyFourHours
    );
    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Up));
    assert_eq!(
        formatter.state.request_window,
        DashboardRequestWindow::TwelveHours
    );

    let rendered = render_tui_frame_snapshot(&formatter.state, 140, 18);
    assert!(rendered.contains("12h"));
    assert!(rendered.contains("30m buckets"));
}
#[test]
pub(super) fn tui_process_tables_support_focus_and_row_navigation() {
    let mut formatter = InteractiveDashboardFormatter::default();
    formatter.handle_snapshot(DashboardSnapshot {
        llama_process_rows: vec![
            sample_process_row("llama-0", 8001),
            sample_process_row("llama-1", 8002),
            sample_process_row("llama-2", 8003),
            sample_process_row("llama-3", 8004),
        ],
        webserver_rows: vec![
            sample_endpoint_row("Console", 3131),
            sample_endpoint_row("API", 9337),
            sample_endpoint_row("Metrics", 9393),
        ],
        ..snapshot_fixture(1, 30)
    });
    formatter.handle_tui_event(TuiEvent::Resize {
        columns: 120,
        rows: 12,
    });

    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Tab));
    assert_eq!(formatter.state.panel_focus, DashboardPanel::LlamaCpp);
    assert_eq!(
        formatter.state.panel_view_state(DashboardPanel::LlamaCpp),
        DashboardPanelViewState {
            scroll_offset: 0,
            selected_row: None,
            viewport_rows: formatter
                .state
                .panel_layout
                .rows_for(DashboardPanel::LlamaCpp),
        }
    );

    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Down));
    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Down));
    assert_eq!(
        formatter
            .state
            .panel_view_state(DashboardPanel::LlamaCpp)
            .selected_row,
        Some(2)
    );

    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::PageDown));
    let llama_viewport_rows = formatter
        .state
        .panel_layout
        .rows_for(DashboardPanel::LlamaCpp);
    assert_eq!(
        formatter.state.panel_view_state(DashboardPanel::LlamaCpp),
        DashboardPanelViewState {
            scroll_offset: 4usize.saturating_sub(llama_viewport_rows),
            selected_row: Some(3),
            viewport_rows: llama_viewport_rows,
        }
    );

    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Tab));
    assert_eq!(formatter.state.panel_focus, DashboardPanel::Webserver);
    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Char('G')));
    assert_eq!(
        formatter
            .state
            .panel_view_state(DashboardPanel::Webserver)
            .selected_row,
        Some(2)
    );
    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Char('g')));
    assert_eq!(
        formatter.state.panel_view_state(DashboardPanel::Webserver),
        DashboardPanelViewState {
            scroll_offset: 0,
            selected_row: Some(0),
            viewport_rows: formatter
                .state
                .panel_layout
                .rows_for(DashboardPanel::Webserver),
        }
    );
}
#[test]
pub(super) fn tui_request_chart_preserves_thirty_one_second_buckets_with_newest_last() {
    let history = DashboardRequestHistoryState::from_snapshot(&DashboardSnapshot {
        accepted_request_buckets: vec![
            DashboardAcceptedRequestBucket {
                second_offset: 0,
                accepted_count: 9,
            },
            DashboardAcceptedRequestBucket {
                second_offset: 5,
                accepted_count: 4,
            },
            DashboardAcceptedRequestBucket {
                second_offset: 29,
                accepted_count: 1,
            },
        ],
        ..DashboardSnapshot::default()
    });

    let chart_spec = tui_request_chart_spec(&history, DashboardRequestWindow::SixtySeconds, 160);

    assert_eq!(
        chart_spec.bucket_values.len(),
        PRETTY_DASHBOARD_REQUEST_WINDOW_BUCKETS,
        "expected 30 two-second buckets"
    );
    assert_eq!(chart_spec.bucket_values.get(15), Some(&1));
    assert_eq!(chart_spec.bucket_values.get(27), Some(&4));
    assert_eq!(chart_spec.bucket_values.last(), Some(&9));
}
#[test]
pub(super) fn tui_braille_bar_symbols_use_vertical_subcell_fill() {
    assert_eq!(tui_braille_bar_symbol(0, 0), '⠀');
    assert_eq!(tui_braille_bar_symbol(1, 1), '⣀');
    assert_eq!(tui_braille_bar_symbol(2, 2), '⣤');
    assert_eq!(tui_braille_bar_symbol(3, 3), '⣶');
    assert_eq!(tui_braille_bar_symbol(4, 4), '⣿');
    assert!(is_braille_bar_symbol(tui_braille_bar_symbol(1, 0)));
    assert_ne!(tui_braille_bar_symbol(1, 0), tui_braille_bar_symbol(0, 1));
}
#[test]
pub(super) fn tui_request_chart_scale_uses_bucket_max_and_headroom_for_every_window() {
    let quiet_history = DashboardRequestHistoryState::default();
    let quiet_spec =
        tui_request_chart_spec(&quiet_history, DashboardRequestWindow::TwentyFourHours, 160);
    assert_eq!(quiet_spec.scale_max, 1);
    assert!(quiet_spec.scale_width >= 3);

    let sparse_day_history = DashboardRequestHistoryState::from_snapshot(&DashboardSnapshot {
        accepted_request_buckets: vec![DashboardAcceptedRequestBucket {
            second_offset: 23 * 60 * 60,
            accepted_count: 1,
        }],
        ..DashboardSnapshot::default()
    });
    let sparse_day_spec = tui_request_chart_spec(
        &sparse_day_history,
        DashboardRequestWindow::TwentyFourHours,
        160,
    );
    assert_eq!(sparse_day_spec.scale_max, 2);

    let busy_history = DashboardRequestHistoryState::from_snapshot(&DashboardSnapshot {
        accepted_request_buckets: vec![DashboardAcceptedRequestBucket {
            second_offset: 0,
            accepted_count: 51,
        }],
        ..DashboardSnapshot::default()
    });
    let busy_spec =
        tui_request_chart_spec(&busy_history, DashboardRequestWindow::SixtySeconds, 160);
    assert!(busy_spec.scale_max > 51);
    assert_eq!(busy_spec.scale_max, 100);
}
#[test]
pub(super) fn tui_request_scale_omits_duplicate_midpoint_for_unit_range() {
    assert_eq!(tui_request_scale_labels(4, 1), vec![(0, 1), (3, 0)]);
    assert_eq!(tui_request_scale_labels(4, 2), vec![(0, 2), (2, 1), (3, 0)]);
}
#[test]
pub(super) fn tui_request_chart_uses_thirty_and_sixty_minute_long_window_buckets() {
    assert_eq!(
        DashboardRequestWindow::TwelveHours.bucket_seconds(),
        30 * 60
    );
    assert_eq!(
        DashboardRequestWindow::TwentyFourHours.bucket_seconds(),
        60 * 60
    );
    assert_eq!(
        DashboardRequestWindow::TwelveHours.bucket_label(),
        "30m buckets"
    );
    assert_eq!(
        DashboardRequestWindow::TwentyFourHours.bucket_label(),
        "60m buckets"
    );

    let history = DashboardRequestHistoryState::from_snapshot(&DashboardSnapshot {
        accepted_request_buckets: vec![
            DashboardAcceptedRequestBucket {
                second_offset: 30 * 60 - 1,
                accepted_count: 3,
            },
            DashboardAcceptedRequestBucket {
                second_offset: 30 * 60,
                accepted_count: 5,
            },
        ],
        ..DashboardSnapshot::default()
    });
    let chart_spec = tui_request_chart_spec(&history, DashboardRequestWindow::TwelveHours, 160);
    assert_eq!(chart_spec.bucket_values.last(), Some(&3));
    assert_eq!(
        chart_spec
            .bucket_values
            .get(PRETTY_DASHBOARD_REQUEST_WINDOW_BUCKETS - 2),
        Some(&5)
    );
}
#[test]
pub(super) fn tui_request_chart_right_aligns_newest_bucket() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
        160, 24,
    )));
    state.reduce(DashboardAction::SnapshotUpdated(DashboardSnapshot {
        accepted_request_buckets: vec![DashboardAcceptedRequestBucket {
            second_offset: 0,
            accepted_count: 9,
        }],
        ..snapshot_fixture(0, 0)
    }));

    let (_, buffer) = render_tui_frame_snapshot_with_buffer(&state, 160, 24);
    let requests_inner = requests_inner_area(&state, 160, 24);
    let [_, graph_slot] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(0)])
        .areas(requests_inner);
    let chart_spec = tui_request_chart_spec(
        &state.request_history,
        state.request_window,
        graph_slot.width,
    );
    let (_, plot_area) = tui_request_chart_areas(graph_slot, &chart_spec);

    assert!(
        (plot_area.y..plot_area.bottom()).any(|y| {
            buffer[(plot_area.right().saturating_sub(1), y)]
                .symbol()
                .chars()
                .next()
                .is_some_and(is_braille_bar_symbol)
        }),
        "expected newest request bucket to touch the right edge of the plot area"
    );
}
#[test]
pub(super) fn tui_request_chart_shrinks_long_window_bars() {
    let history = DashboardRequestHistoryState::from_snapshot(&DashboardSnapshot {
        accepted_request_buckets: vec![DashboardAcceptedRequestBucket {
            second_offset: 0,
            accepted_count: 9,
        }],
        ..DashboardSnapshot::default()
    });
    let short_spec = tui_request_chart_spec(&history, DashboardRequestWindow::SixtySeconds, 160);
    let day_spec = tui_request_chart_spec(&history, DashboardRequestWindow::TwentyFourHours, 160);

    assert!(
        short_spec.bar_width > day_spec.bar_width,
        "expected longer request windows to render narrower bars"
    );
    assert_eq!(day_spec.bar_width, 1);
}
#[test]
pub(super) fn tui_requests_panel_renders_multi_row_barchart_and_summary_values() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
        160, 24,
    )));
    state.reduce(DashboardAction::SnapshotUpdated(DashboardSnapshot {
        current_inflight_requests: 7,
        accepted_request_buckets: vec![
            DashboardAcceptedRequestBucket {
                second_offset: 0,
                accepted_count: 9,
            },
            DashboardAcceptedRequestBucket {
                second_offset: 1,
                accepted_count: 4,
            },
        ],
        latency_samples_ms: vec![11, 17, 19, 23],
        ..snapshot_fixture(0, 0)
    }));

    let (rendered, buffer) = render_tui_frame_snapshot_with_buffer(&state, 160, 24);
    let requests_inner = requests_inner_area(&state, 160, 24);
    let (_, line) = find_rendered_line(&rendered, "RPS ");

    assert!(
        line.contains("RPS 9"),
        "expected current-bucket RPS in {line}"
    );
    assert!(
        line.contains("inflight 7"),
        "expected inflight count in {line}"
    );
    assert!(line.contains("p50 18ms"), "expected p50 latency in {line}");
    assert!(
        line.contains("window 60s"),
        "expected request window in {line}"
    );
    assert!(
        line.contains("2s buckets"),
        "expected bucket size in {line}"
    );
    assert!(
        !line.contains('|'),
        "expected summary row, not old sparkline strip: {line}"
    );
    assert!(
        rendered.contains("Incoming Requests  60s  2s buckets"),
        "expected request panel title to show window and bucket size in {rendered}"
    );
    assert!(
        request_graph_visible_row_count(&buffer, requests_inner) >= 2,
        "expected multi-row request graph in area {requests_inner:?}\n{rendered}"
    );
    assert!(
        request_graph_contains_bars(&buffer, requests_inner),
        "expected real bar glyphs in request graph area {requests_inner:?}\n{rendered}"
    );
    assert!(
        rendered.contains("20"),
        "expected adaptive request scale label in {rendered}"
    );
    assert!(
        !rendered.contains('•'),
        "expected Braille bar glyphs instead of dot bullets in {rendered}"
    );
}
#[test]
pub(super) fn tui_requests_panel_shows_na_latency_when_window_empty() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
        160, 24,
    )));
    state.reduce(DashboardAction::SnapshotUpdated(DashboardSnapshot {
        current_inflight_requests: 2,
        accepted_request_buckets: vec![DashboardAcceptedRequestBucket {
            second_offset: 0,
            accepted_count: 3,
        }],
        latency_samples_ms: Vec::new(),
        ..snapshot_fixture(0, 0)
    }));

    let (rendered, buffer) = render_tui_frame_snapshot_with_buffer(&state, 160, 24);
    let requests_inner = requests_inner_area(&state, 160, 24);
    let (_, line) = find_rendered_line(&rendered, "RPS ");

    assert!(
        line.contains("p50 n/a"),
        "expected empty-window latency text in {line}"
    );
    assert!(
        request_graph_visible_row_count(&buffer, requests_inner) >= 2,
        "expected visible empty-state graph guides in area {requests_inner:?}\n{rendered}"
    );
    assert!(
        request_graph_contains_guides(&buffer, requests_inner),
        "expected empty-state graph guides in area {requests_inner:?}\n{rendered}"
    );
}
#[test]
pub(super) fn tui_requests_panel_zero_traffic_still_renders_visible_graph_area() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
        160, 24,
    )));

    let (rendered, buffer) = render_tui_frame_snapshot_with_buffer(&state, 160, 24);
    let requests_inner = requests_inner_area(&state, 160, 24);
    let (_, line) = find_rendered_line(&rendered, "RPS ");

    assert!(line.contains("RPS 0"), "expected zero RPS in {line}");
    assert!(
        line.contains("inflight 0"),
        "expected zero inflight in {line}"
    );
    assert!(line.contains("p50 n/a"), "expected n/a latency in {line}");
    assert!(
        request_graph_visible_row_count(&buffer, requests_inner) >= 2,
        "expected idle graph area to stay visibly chart-like in {requests_inner:?}\n{rendered}"
    );
    assert!(
        request_graph_contains_guides(&buffer, requests_inner),
        "expected idle graph guides in area {requests_inner:?}\n{rendered}"
    );
    assert!(
        !request_graph_contains_bars(&buffer, requests_inner),
        "expected idle graph to avoid fake traffic bars in area {requests_inner:?}\n{rendered}"
    );
}
#[test]
pub(super) fn tui_requests_panel_clears_stale_bars_before_redraw() {
    let mut busy_state = DashboardState::default();
    busy_state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
        160, 24,
    )));
    busy_state.reduce(DashboardAction::SnapshotUpdated(DashboardSnapshot {
        accepted_request_buckets: vec![
            DashboardAcceptedRequestBucket {
                second_offset: 0,
                accepted_count: 40,
            },
            DashboardAcceptedRequestBucket {
                second_offset: 1,
                accepted_count: 32,
            },
            DashboardAcceptedRequestBucket {
                second_offset: 2,
                accepted_count: 28,
            },
        ],
        ..snapshot_fixture(0, 0)
    }));

    let mut quiet_state = DashboardState::default();
    quiet_state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
        160, 24,
    )));

    let backend = ratatui::backend::TestBackend::new(160, 24);
    let mut terminal = Terminal::new(backend).expect("test backend should initialize");
    terminal
        .draw(|frame| render_tui_frame(frame, &busy_state))
        .expect("busy frame render should succeed");
    terminal
        .draw(|frame| render_tui_frame(frame, &quiet_state))
        .expect("quiet frame render should succeed");

    let buffer = terminal.backend().buffer().clone();
    let requests_inner = requests_inner_area(&quiet_state, 160, 24);

    assert!(
        !request_graph_contains_bars(&buffer, requests_inner),
        "expected quiet redraw to clear stale Braille bars"
    );
}
#[test]
pub(super) fn tui_requests_panel_stays_multi_row_at_tighter_live_height() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
        160, 23,
    )));

    let (rendered, buffer) = render_tui_frame_snapshot_with_buffer(&state, 160, 23);
    let requests_inner = requests_inner_area(&state, 160, 23);

    assert!(
        requests_inner.height >= 3,
        "expected summary + at least two graph rows in area {requests_inner:?}\n{rendered}"
    );
    assert!(
        request_graph_visible_row_count(&buffer, requests_inner) >= 2,
        "expected visible request graph rows in area {requests_inner:?}\n{rendered}"
    );
    assert!(
        request_graph_contains_guides(&buffer, requests_inner),
        "expected chart guides in tighter live-height area {requests_inner:?}\n{rendered}"
    );
}
#[test]
pub(super) fn tui_status_bar_reports_focus_follow_and_filter_state() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
        240, 24,
    )));
    state.reduce(DashboardAction::SnapshotUpdated(DashboardSnapshot {
        llama_process_rows: vec![sample_process_row("llama-0", 8001)],
        webserver_rows: vec![
            sample_endpoint_row("Console", 3131),
            sample_endpoint_row("API", 9337),
        ],
        loaded_model_rows: vec![
            sample_model_row("Model-0", 4000),
            sample_model_row("Model-1", 4001),
        ],
        ..snapshot_fixture(0, 30)
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::Startup {
        version: "v0.64.0".to_string(),
        message: None,
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::NodeIdentity {
        node_id: "node-7".to_string(),
        mesh_id: Some("poker-night".to_string()),
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::PeerJoined {
        peer_id: "peer-1".to_string(),
        label: Some("alice".to_string()),
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::PeerJoined {
        peer_id: "peer-2".to_string(),
        label: Some("bob".to_string()),
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::RuntimeReady {
        api_url: "http://localhost:9337".to_string(),
        console_url: Some("http://localhost:3131".to_string()),
        api_port: 9337,
        console_port: Some(3131),
        models_count: Some(2),
        pi_command: None,
        goose_command: None,
    }));
    state.reduce(DashboardAction::FocusNextPanel);
    state.reduce(DashboardAction::FocusNextPanel);
    state.reduce(DashboardAction::FocusNextPanel);
    state.reduce(DashboardAction::SetPanelSelection {
        panel: DashboardPanel::Models,
        selected_row: Some(1),
    });
    state.reduce(DashboardAction::StartEventsFilterEdit);
    state.reduce(DashboardAction::InsertEventsFilterChar('p'));
    state.reduce(DashboardAction::InsertEventsFilterChar('o'));
    state.reduce(DashboardAction::ConfirmEventsFilter);
    state.reduce(DashboardAction::FocusNextPanel);
    state.reduce(DashboardAction::FocusNextPanel);
    state.reduce(DashboardAction::FocusNextPanel);
    state.reduce(DashboardAction::ToggleEventsFollow);

    let rendered = render_tui_frame_snapshot(&state, 240, 24);
    assert!(rendered.contains("READY"));
    assert!(rendered.contains("uptime:"));
    assert!(
        rendered.contains("peers: 2"),
        "expected peer count in {rendered}"
    );
    assert!(
        rendered.contains("models: 2"),
        "expected model count in {rendered}"
    );
    assert!(
        rendered.contains("processes: 3"),
        "expected process count in {rendered}"
    );
    assert!(rendered.contains("[Tab]"));
    assert!(rendered.contains("[Enter/Z]"));
    assert!(rendered.contains("[Shift-Tab]"));
    assert!(rendered.contains("[/]"));
    assert!(rendered.contains("[F]"));
    assert!(rendered.contains("[↑/↓]"));
    assert!(rendered.contains("[R]"));
    assert!(rendered.contains("[Q]"));
}
#[test]
pub(super) fn tui_status_bar_uses_badge_uptime_and_key_hint_styles() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
        180, 24,
    )));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::RuntimeReady {
        api_url: "http://localhost:9337".to_string(),
        console_url: None,
        api_port: 9337,
        console_port: None,
        models_count: Some(0),
        pi_command: None,
        goose_command: None,
    }));

    let (rendered, buffer) = render_tui_frame_snapshot_with_buffer(&state, 180, 24);
    let (ready_y, ready_line) = find_rendered_line(&rendered, "READY");
    let ready_x = ready_line
        .find("READY")
        .expect("expected READY badge in status line");
    let (tab_y, tab_line) = find_rendered_line(&rendered, "[Tab]");
    let tab_x = tab_line
        .find("[Tab]")
        .expect("expected bracketed Tab hint in controls line");
    let peers_x = ready_line
        .find("peers:")
        .expect("expected peer stats in status line");
    let processes_x = ready_line
        .find("processes:")
        .expect("expected process stats in status line");
    let uptime_x = ready_line
        .find("uptime:")
        .expect("expected uptime in status line");
    let theme = tui_theme();

    assert!(
        rendered.contains("uptime:"),
        "expected uptime text in {rendered}"
    );
    assert!(
        rendered.contains("[Q] Quit"),
        "expected bracketed quit hint in {rendered}"
    );
    assert!(
        rendered.contains("[↑/↓] Window"),
        "expected bracketed request-window hint in {rendered}"
    );
    assert!(
        ready_x <= 1,
        "expected READY badge at the far left of status line: {ready_line}"
    );
    assert!(
        ready_x < tab_x,
        "expected READY badge to precede hotkeys in {ready_line}"
    );
    assert!(
        peers_x > tab_x,
        "expected status stats to stay pinned after the flexible gap in {ready_line}"
    );
    assert!(
        uptime_x > processes_x,
        "expected uptime to stay near the clock at the right edge in {ready_line}"
    );
    assert_eq!(
        buffer[(ready_x as u16, ready_y as u16)].style().fg,
        Some(theme.success)
    );
    assert_eq!(
        buffer[(tab_x as u16, tab_y as u16)].style().fg,
        Some(theme.accent)
    );
    assert_eq!(
        buffer[(tab_x as u16, tab_y as u16)].style().bg,
        Some(theme.surface_raised)
    );
}
#[test]
pub(super) fn planned_rows_transition_from_not_ready_to_ready_events() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::OutputEvent(OutputEvent::LaunchPlan {
        plan: sample_launch_plan(),
    }));

    assert!(
        state
            .llama_process_rows
            .iter()
            .all(|row| row.status == RuntimeStatus::Loading)
    );
    assert!(
        state
            .webserver_rows
            .iter()
            .all(|row| row.status == RuntimeStatus::NotReady)
    );
    assert_eq!(state.loaded_model_rows[0].status, RuntimeStatus::Loading);
    state.reduce(DashboardAction::OutputEvent(OutputEvent::LlamaStarting {
        model: Some("Planned-Model".to_string()),
        http_port: 9338,
        ctx_size: Some(8192),
        log_path: None,
    }));
    state.reduce(DashboardAction::OutputEvent(
        OutputEvent::WebserverStarting {
            url: "http://localhost:3131".to_string(),
        },
    ));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::ApiStarting {
        url: "http://localhost:9337".to_string(),
    }));
    assert_eq!(
        state
            .llama_process_rows
            .iter()
            .find(|row| row.port == 9338)
            .expect("expected planned llama row")
            .status,
        RuntimeStatus::Starting
    );
    assert_eq!(
        state
            .webserver_rows
            .iter()
            .find(|row| row.label == "Console")
            .expect("expected planned console row")
            .status,
        RuntimeStatus::Starting
    );
    assert_eq!(
        state
            .webserver_rows
            .iter()
            .find(|row| row.label == "API")
            .expect("expected planned api row")
            .status,
        RuntimeStatus::Starting
    );
    assert_eq!(state.loaded_model_rows[0].status, RuntimeStatus::Loading);
    state.reduce(DashboardAction::OutputEvent(OutputEvent::LlamaReady {
        model: Some("Planned-Model".to_string()),
        port: 9338,
        ctx_size: Some(8192),
        log_path: None,
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::WebserverReady {
        url: "http://localhost:3131".to_string(),
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::ApiReady {
        url: "http://localhost:9337".to_string(),
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::ModelReady {
        model: "Planned-Model".to_string(),
        internal_port: Some(9338),
        role: Some("host".to_string()),
    }));
    assert_eq!(
        state
            .llama_process_rows
            .iter()
            .find(|row| row.port == 9338)
            .expect("expected planned llama row")
            .status,
        RuntimeStatus::Ready
    );
    assert_eq!(
        state
            .webserver_rows
            .iter()
            .find(|row| row.label == "Console")
            .expect("expected planned console row")
            .status,
        RuntimeStatus::Ready
    );
    assert_eq!(
        state
            .webserver_rows
            .iter()
            .find(|row| row.label == "API")
            .expect("expected planned api row")
            .status,
        RuntimeStatus::Ready
    );
    assert_eq!(state.loaded_model_rows[0].status, RuntimeStatus::Ready);
}
