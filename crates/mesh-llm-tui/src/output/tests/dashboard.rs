use super::*;

#[test]
pub(super) fn tui_reducer_focus_cycle_wraps_across_dashboard_panels() {
    let mut fixture = DashboardReducerFixture::default();

    assert_eq!(fixture.state.panel_focus, DashboardPanel::Events);
    assert!(fixture.state.events_follow, "follow should default to ON");

    fixture.reduce(DashboardAction::ToggleEventsFollow);
    assert!(!fixture.state.events_follow);
    fixture.reduce(DashboardAction::ToggleEventsFollow);
    assert!(fixture.state.events_follow);

    let expected_forward_order = [
        DashboardPanel::LlamaCpp,
        DashboardPanel::Webserver,
        DashboardPanel::Models,
        DashboardPanel::Requests,
        DashboardPanel::JoinToken,
        DashboardPanel::Events,
    ];
    for expected_panel in expected_forward_order {
        fixture.reduce(DashboardAction::FocusNextPanel);
        assert_eq!(fixture.state.panel_focus, expected_panel);
    }

    fixture.reduce(DashboardAction::FocusPreviousPanel);
    assert_eq!(fixture.state.panel_focus, DashboardPanel::JoinToken);
}

#[test]
pub(super) fn tui_full_screen_panel_toggles_from_focused_panel_and_restores_layout() {
    let mut formatter = InteractiveDashboardFormatter::default();
    formatter.handle_tui_event(TuiEvent::Resize {
        columns: 120,
        rows: 30,
    });

    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Tab));
    assert_eq!(formatter.state.panel_focus, DashboardPanel::LlamaCpp);

    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Enter));
    assert_eq!(
        formatter.state.full_screen_panel,
        Some(DashboardPanel::LlamaCpp)
    );

    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Escape));
    assert_eq!(formatter.state.full_screen_panel, None);
    assert_eq!(formatter.state.panel_focus, DashboardPanel::LlamaCpp);

    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Char('z')));
    assert_eq!(
        formatter.state.full_screen_panel,
        Some(DashboardPanel::LlamaCpp)
    );
    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Char('z')));
    assert_eq!(formatter.state.full_screen_panel, None);
}

#[test]
pub(super) fn tui_reducer_filter_is_case_insensitive_substring() {
    let mut fixture = DashboardReducerFixture::default().with_events(vec![
        OutputEvent::DiscoveryJoined {
            mesh: "Poker-Night".to_string(),
        },
        info_event("background sync complete"),
        OutputEvent::Warning {
            message: "capacity estimate stale".to_string(),
            context: Some("model=Qwen3-32B".to_string()),
        },
    ]);

    fixture.reduce(DashboardAction::FocusNextPanel);
    assert_eq!(fixture.state.panel_focus, DashboardPanel::LlamaCpp);

    fixture.reduce(DashboardAction::StartEventsFilterEdit);
    assert_eq!(fixture.state.panel_focus, DashboardPanel::Events);
    assert!(fixture.state.events_filter.editing);

    for ch in "PoKeR".chars() {
        fixture.reduce(DashboardAction::InsertEventsFilterChar(ch));
    }

    let filtered_events = fixture.state.filtered_mesh_events();
    assert_eq!(filtered_events.len(), 1);
    assert!(filtered_events[0].summary.contains("Poker-Night"));

    fixture.reduce(DashboardAction::BackspaceEventsFilter);
    assert_eq!(fixture.state.events_filter.query, "PoKe");
    assert_eq!(fixture.state.filtered_mesh_events().len(), 1);

    fixture.reduce(DashboardAction::ConfirmEventsFilter);
    assert!(!fixture.state.events_filter.editing);
    assert_eq!(fixture.state.events_filter.query, "PoKe");

    fixture.reduce(DashboardAction::StartEventsFilterEdit);
    fixture.reduce(DashboardAction::CancelEventsFilter);
    assert!(!fixture.state.events_filter.editing);
    assert!(fixture.state.events_filter.query.is_empty());
    assert_eq!(fixture.state.filtered_mesh_events().len(), 3);

    fixture.reduce(DashboardAction::StartEventsFilterEdit);
    for ch in "mesh.*night".chars() {
        fixture.reduce(DashboardAction::InsertEventsFilterChar(ch));
    }
    assert_eq!(fixture.state.filtered_mesh_events().len(), 0);
}

#[test]
pub(super) fn tui_reducer_filter_matches_visible_event_badges() {
    let mut fixture = DashboardReducerFixture::default().with_events(vec![
        info_event("plain operational marker"),
        info_event("ok heartbeat marker"),
        OutputEvent::Warning {
            message: "capacity stale marker".to_string(),
            context: None,
        },
    ]);

    fixture.reduce(DashboardAction::StartEventsFilterEdit);
    for ch in "INFO".chars() {
        fixture.reduce(DashboardAction::InsertEventsFilterChar(ch));
    }

    let filtered_events = fixture.state.filtered_mesh_events();
    assert_eq!(filtered_events.len(), 1);
    assert_eq!(filtered_events[0].summary, "plain operational marker");
}

#[test]
pub(super) fn tui_reducer_preserves_scroll_on_resize() {
    let mut fixture = DashboardReducerFixture::default().with_snapshot(snapshot_fixture(12, 30));

    fixture.reduce(DashboardAction::FocusNextPanel);
    fixture.reduce(DashboardAction::FocusNextPanel);
    fixture.reduce(DashboardAction::FocusNextPanel);
    assert_eq!(fixture.state.panel_focus, DashboardPanel::Models);

    fixture.reduce(DashboardAction::Resize(DashboardLayoutState::new(
        4, 4, 4, 3, 2,
    )));
    fixture.reduce(DashboardAction::SetPanelSelection {
        panel: DashboardPanel::Models,
        selected_row: Some(5),
    });
    fixture.reduce(DashboardAction::SetPanelScroll {
        panel: DashboardPanel::Models,
        scroll_offset: 4,
    });

    let before_resize = fixture.state.panel_view_state(DashboardPanel::Models);
    assert_eq!(before_resize.selected_row, None);
    assert_eq!(before_resize.scroll_offset, 4);

    fixture.reduce(DashboardAction::Resize(DashboardLayoutState::new(
        6, 4, 4, 5, 2,
    )));

    let after_resize = fixture.state.panel_view_state(DashboardPanel::Models);
    assert_eq!(fixture.state.panel_focus, DashboardPanel::Models);
    assert_eq!(after_resize.selected_row, None);
    assert_eq!(after_resize.scroll_offset, 4);
    assert_eq!(
        after_resize.viewport_rows,
        tui_panel_viewport_rows(DashboardPanel::Models, 5)
    );
}

#[test]
pub(super) fn tui_reducer_caps_event_history_at_1000() {
    let mut fixture = DashboardReducerFixture::default().with_snapshot(snapshot_fixture(2, 35));

    for index in 0..1005 {
        fixture.reduce(DashboardAction::OutputEvent(info_event(format!(
            "event-{index}"
        ))));
    }

    assert_eq!(fixture.state.mesh_event_limit, 1000);
    assert_eq!(fixture.state.mesh_events.len(), 1000);
    assert_eq!(
        fixture.state.request_history.accepted_request_buckets.len(),
        PRETTY_DASHBOARD_REQUEST_MAX_WINDOW_SECS as usize
    );
    assert!(
        fixture
            .state
            .mesh_events
            .front()
            .expect("expected oldest retained event")
            .summary
            .contains("event-5")
    );
    assert!(
        fixture
            .state
            .mesh_events
            .back()
            .expect("expected newest retained event")
            .summary
            .contains("event-1004")
    );
}

#[test]
pub(super) fn tui_events_follow_mode_keeps_latest_row_visible() {
    let mut formatter = InteractiveDashboardFormatter::default();
    formatter
        .state
        .reduce(DashboardAction::Resize(DashboardLayoutState::new(
            4, 2, 2, 2, 2,
        )));

    for index in 0..8 {
        formatter
            .handle_output_event(&info_event(format!("event-{index}")))
            .expect("event render should succeed");
    }

    let before = formatter.state.panel_view_state(DashboardPanel::Events);
    assert!(formatter.state.events_follow);
    assert_eq!(before.selected_row, Some(7));
    assert_eq!(before.scroll_offset, 4);

    formatter
        .handle_output_event(&info_event("event-8"))
        .expect("event render should succeed");

    let after = formatter.state.panel_view_state(DashboardPanel::Events);
    assert!(formatter.state.events_follow);
    assert_eq!(after.selected_row, Some(8));
    assert_eq!(after.scroll_offset, 5);
}

#[test]
pub(super) fn tui_events_short_list_navigation_keeps_non_follow_anchor() {
    let mut formatter = InteractiveDashboardFormatter::default();
    formatter
        .state
        .reduce(DashboardAction::Resize(DashboardLayoutState::new(
            8, 2, 2, 2, 2,
        )));

    for index in 0..3 {
        formatter
            .handle_output_event(&info_event(format!("event-{index}")))
            .expect("event render should succeed");
    }

    assert!(formatter.state.events_follow);
    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Char('f')));
    assert!(!formatter.state.events_follow);

    let viewport_rows = formatter
        .state
        .panel_view_state(DashboardPanel::Events)
        .viewport_rows;
    assert!(
        formatter.state.row_count_for_panel(DashboardPanel::Events) < viewport_rows,
        "test must exercise the short-list path"
    );
    let first_event_before = visible_event_rows(&formatter.state, viewport_rows)
        .iter()
        .position(|row| matches!(row, TuiEventRow::Event { .. }))
        .expect("expected at least one event row");

    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Char('G')));

    let view_after = formatter.state.panel_view_state(DashboardPanel::Events);
    assert_eq!(view_after.selected_row, Some(2));
    assert_eq!(view_after.scroll_offset, 0);
    assert!(
        formatter.state.events_follow,
        "jumping to the end of a short scrollbar list should follow the newest event"
    );
    let first_event_after = visible_event_rows(&formatter.state, viewport_rows)
        .iter()
        .position(|row| matches!(row, TuiEventRow::Event { .. }))
        .expect("expected at least one event row");
    assert!(
        first_event_after >= first_event_before,
        "short scrollbar lists may bottom-anchor when follow is re-enabled, but must not scroll text out of range"
    );
}

#[test]
pub(super) fn tui_events_short_list_arrow_navigation_disables_follow() {
    let mut formatter = InteractiveDashboardFormatter::default();
    formatter
        .state
        .reduce(DashboardAction::Resize(DashboardLayoutState::new(
            8, 2, 2, 2, 2,
        )));

    for index in 0..3 {
        formatter
            .handle_output_event(&info_event(format!("event-{index}")))
            .expect("event render should succeed");
    }

    assert!(formatter.state.events_follow);
    assert_eq!(
        formatter
            .state
            .panel_view_state(DashboardPanel::Events)
            .selected_row,
        Some(2)
    );

    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Up));

    assert!(formatter.state.events_follow);
    assert_eq!(
        formatter.state.panel_view_state(DashboardPanel::Events),
        DashboardPanelViewState {
            scroll_offset: 0,
            selected_row: Some(2),
            viewport_rows: 8,
        }
    );

    formatter
        .state
        .reduce(DashboardAction::Resize(DashboardLayoutState::new(
            8, 2, 2, 2, 2,
        )));

    assert_eq!(
        formatter
            .state
            .panel_view_state(DashboardPanel::Events)
            .selected_row,
        Some(2),
        "short scrollbar lists do not move a selected row; arrows only scroll text"
    );
}

#[test]
pub(super) fn tui_events_pgup_pgdn_and_home_end_navigation() {
    let mut formatter = InteractiveDashboardFormatter::default();
    formatter
        .state
        .reduce(DashboardAction::Resize(DashboardLayoutState::new(
            5, 2, 2, 2, 2,
        )));

    for index in 0..12 {
        formatter
            .handle_output_event(&info_event(format!("event-{index}")))
            .expect("event render should succeed");
    }

    assert!(formatter.state.events_follow);
    assert_eq!(
        formatter.state.panel_view_state(DashboardPanel::Events),
        DashboardPanelViewState {
            scroll_offset: 7,
            selected_row: Some(11),
            viewport_rows: 5,
        }
    );

    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::PageUp));
    assert!(!formatter.state.events_follow);
    assert_eq!(
        formatter.state.panel_view_state(DashboardPanel::Events),
        DashboardPanelViewState {
            scroll_offset: 3,
            selected_row: Some(11),
            viewport_rows: 5,
        }
    );

    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::PageDown));
    assert!(formatter.state.events_follow);
    assert_eq!(
        formatter.state.panel_view_state(DashboardPanel::Events),
        DashboardPanelViewState {
            scroll_offset: 7,
            selected_row: Some(11),
            viewport_rows: 5,
        }
    );

    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Char('g')));
    assert!(!formatter.state.events_follow);
    assert_eq!(
        formatter.state.panel_view_state(DashboardPanel::Events),
        DashboardPanelViewState {
            scroll_offset: 0,
            selected_row: Some(11),
            viewport_rows: 5,
        }
    );

    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Char('G')));
    assert!(formatter.state.events_follow);
    assert_eq!(
        formatter.state.panel_view_state(DashboardPanel::Events),
        DashboardPanelViewState {
            scroll_offset: 7,
            selected_row: Some(11),
            viewport_rows: 5,
        }
    );
}

#[test]
pub(super) fn tui_events_filter_persists_across_focus_changes() {
    let mut formatter = InteractiveDashboardFormatter::default();
    formatter
        .handle_output_event(&OutputEvent::DiscoveryJoined {
            mesh: "Poker-Night".to_string(),
        })
        .expect("event render should succeed");
    formatter
        .handle_output_event(&info_event("background sync complete"))
        .expect("event render should succeed");

    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Tab));
    assert_eq!(formatter.state.panel_focus, DashboardPanel::LlamaCpp);

    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Char('/')));
    for ch in "poker".chars() {
        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Char(ch)));
    }
    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Enter));

    assert_eq!(formatter.state.panel_focus, DashboardPanel::Events);
    assert_eq!(formatter.state.events_filter.query, "poker");
    assert_eq!(formatter.state.filtered_mesh_events().len(), 1);

    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Tab));
    assert_eq!(formatter.state.panel_focus, DashboardPanel::LlamaCpp);
    assert!(!formatter.state.events_filter.editing);
    assert_eq!(formatter.state.events_filter.query, "poker");
    assert_eq!(formatter.state.filtered_mesh_events().len(), 1);

    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::BackTab));
    assert_eq!(formatter.state.panel_focus, DashboardPanel::Events);
    assert_eq!(formatter.state.events_filter.query, "poker");
    assert_eq!(formatter.state.filtered_mesh_events().len(), 1);
}

#[test]
pub(super) fn tui_events_fewer_items_than_viewport_scroll_offset_is_zero() {
    let mut formatter = InteractiveDashboardFormatter::default();
    formatter
        .state
        .reduce(DashboardAction::Resize(DashboardLayoutState::new(
            8, 2, 2, 2, 2,
        )));

    for index in 0..3 {
        formatter
            .handle_output_event(&info_event(format!("event-{index}")))
            .expect("event render should succeed");
    }

    let view = formatter.state.panel_view_state(DashboardPanel::Events);
    assert_eq!(view.scroll_offset, 0);
    assert_eq!(view.viewport_rows, 8);

    let rows = visible_event_rows(&formatter.state, view.viewport_rows);
    let event_count = rows
        .iter()
        .filter(|r| matches!(r, TuiEventRow::Event { .. }))
        .count();
    assert_eq!(event_count, 3);
}

#[test]
pub(super) fn tui_events_overflow_scroll_offset_tracks_last_event() {
    let mut formatter = InteractiveDashboardFormatter::default();
    formatter
        .state
        .reduce(DashboardAction::Resize(DashboardLayoutState::new(
            4, 2, 2, 2, 2,
        )));

    for index in 0..10 {
        formatter
            .handle_output_event(&info_event(format!("event-{index}")))
            .expect("event render should succeed");
    }

    assert!(formatter.state.events_follow);
    let view = formatter.state.panel_view_state(DashboardPanel::Events);
    assert_eq!(view.scroll_offset, 6);
    assert_eq!(view.selected_row, Some(9));
}

#[test]
pub(super) fn tui_events_manual_scroll_up_disables_follow() {
    let mut formatter = InteractiveDashboardFormatter::default();
    formatter
        .state
        .reduce(DashboardAction::Resize(DashboardLayoutState::new(
            4, 2, 2, 2, 2,
        )));

    for index in 0..8 {
        formatter
            .handle_output_event(&info_event(format!("event-{index}")))
            .expect("event render should succeed");
    }

    assert!(formatter.state.events_follow);
    let view_before = formatter.state.panel_view_state(DashboardPanel::Events);
    assert_eq!(view_before.scroll_offset, 4);
    assert_eq!(view_before.selected_row, Some(7));

    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Up));

    assert!(!formatter.state.events_follow);
    let view_after = formatter.state.panel_view_state(DashboardPanel::Events);
    assert_eq!(
        view_after.selected_row,
        Some(7),
        "Up should not move a selected event row in scrollbar mode"
    );
    assert_eq!(
        view_after.scroll_offset, 3,
        "Up should scroll the event text by exactly one line"
    );
}

#[test]
pub(super) fn startup_lifecycle_transitions_pending_partial_ready_failed() {
    let mut state = DashboardState::default();
    assert_eq!(
        state.startup_lifecycle().phase,
        StartupLifecyclePhase::Pending
    );
    assert_eq!(
        state.startup_lifecycle().api.phase,
        StartupLifecyclePhase::Pending
    );

    state.reduce(DashboardAction::OutputEvent(OutputEvent::Startup {
        version: "v0.68.0".to_string(),
        message: None,
    }));
    assert_eq!(
        state.startup_lifecycle().phase,
        StartupLifecyclePhase::Starting
    );

    state.reduce(DashboardAction::OutputEvent(OutputEvent::ApiStarting {
        url: "http://localhost:9337".to_string(),
    }));
    assert_eq!(
        state.startup_lifecycle().api.phase,
        StartupLifecyclePhase::Starting
    );
    assert_eq!(
        state.startup_lifecycle().phase,
        StartupLifecyclePhase::Starting
    );

    state.reduce(DashboardAction::OutputEvent(OutputEvent::ApiReady {
        url: "http://localhost:9337".to_string(),
    }));
    assert_eq!(
        state.startup_lifecycle().phase,
        StartupLifecyclePhase::Partial
    );
    assert_eq!(
        state.startup_lifecycle().api.phase,
        StartupLifecyclePhase::Ready
    );

    let partial_rendered = render_tui_frame_snapshot(&state, 160, 32);
    let partial_dashboard = render_dashboard_text(&state);
    assert!(partial_rendered.contains("startup=partial"));
    assert!(partial_dashboard.contains("mesh=pending  api=ready  console=pending"));
    assert!(partial_dashboard.contains("llama-server=pending  model readiness=pending"));

    state.reduce(DashboardAction::OutputEvent(OutputEvent::RuntimeReady {
        api_url: "http://localhost:9337".to_string(),
        console_url: Some("http://localhost:3131".to_string()),
        api_port: 9337,
        console_port: Some(3131),
        models_count: Some(1),
        pi_command: None,
        goose_command: None,
    }));
    assert_eq!(
        state.startup_lifecycle().phase,
        StartupLifecyclePhase::Ready
    );
    assert_eq!(
        state.startup_lifecycle().llama_server.phase,
        StartupLifecyclePhase::Ready
    );
    assert_eq!(
        state.startup_lifecycle().llama_server.detail.as_deref(),
        Some("embedded runtime ready")
    );

    let ready_rendered = render_tui_frame_snapshot(&state, 160, 32);
    let ready_dashboard = render_dashboard_text(&state);
    assert!(ready_rendered.contains("startup=ready"));
    assert!(ready_dashboard.contains("mesh=ready  api=ready  console=ready"));
    assert!(ready_dashboard.contains("llama-server=ready  model readiness=pending"));

    let mut failed = DashboardState::default();
    failed.reduce(DashboardAction::OutputEvent(OutputEvent::Startup {
        version: "v0.68.0".to_string(),
        message: None,
    }));
    failed.reduce(DashboardAction::OutputEvent(OutputEvent::Error {
        message: "mesh startup failed".to_string(),
        context: Some("startup".to_string()),
    }));
    assert_eq!(
        failed.startup_lifecycle().phase,
        StartupLifecyclePhase::Failed
    );
    let failed_rendered = render_tui_frame_snapshot(&failed, 160, 32);
    let failed_dashboard = render_dashboard_text(&failed);
    assert!(failed_rendered.contains("startup=failed"));
    assert!(failed_dashboard.contains("mesh=failed"));
}

#[test]
pub(super) fn startup_lifecycle_keeps_runtime_ready_as_final_edge() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::OutputEvent(OutputEvent::Startup {
        version: "v0.68.0".to_string(),
        message: None,
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::NodeIdentity {
        node_id: "node-7".to_string(),
        mesh_id: Some("poker-night".to_string()),
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::WebserverReady {
        url: "http://localhost:3131".to_string(),
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::ApiReady {
        url: "http://localhost:9337".to_string(),
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::LlamaReady {
        model: Some("Qwen3-32B".to_string()),
        port: 9338,
        ctx_size: Some(8192),
        log_path: None,
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::ModelReady {
        model: "Qwen3-32B".to_string(),
        internal_port: Some(9338),
        role: Some("host".to_string()),
    }));

    assert!(
        !state.runtime_ready,
        "RuntimeReady must remain the final edge"
    );
    assert_eq!(
        state.startup_lifecycle().phase,
        StartupLifecyclePhase::Partial
    );
    assert_eq!(
        state.startup_lifecycle().model_readiness.phase,
        StartupLifecyclePhase::Ready
    );

    state.reduce(DashboardAction::OutputEvent(OutputEvent::RuntimeReady {
        api_url: "http://localhost:9337".to_string(),
        console_url: Some("http://localhost:3131".to_string()),
        api_port: 9337,
        console_port: Some(3131),
        models_count: Some(1),
        pi_command: None,
        goose_command: None,
    }));

    assert!(state.runtime_ready);
    assert_eq!(
        state.startup_lifecycle().phase,
        StartupLifecyclePhase::Ready
    );
}

#[test]
pub(super) fn endpoint_rows_remain_starting_until_ready_events() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::OutputEvent(OutputEvent::Startup {
        version: "v0.68.0".to_string(),
        message: None,
    }));
    state.reduce(DashboardAction::SnapshotUpdated(DashboardSnapshot {
        llama_process_rows: vec![sample_process_row("llama-server", 9338)],
        webserver_rows: vec![
            sample_endpoint_row("Console", 3131),
            sample_endpoint_row("API", 9337),
        ],
        ..DashboardSnapshot::default()
    }));

    assert_eq!(
        state.webserver_rows,
        vec![
            DashboardEndpointRow {
                label: "Console".to_string(),
                status: RuntimeStatus::Starting,
                url: "http://127.0.0.1:3131".to_string(),
                port: 3131,
                pid: None,
            },
            DashboardEndpointRow {
                label: "API".to_string(),
                status: RuntimeStatus::Starting,
                url: "http://127.0.0.1:9337".to_string(),
                port: 9337,
                pid: None,
            },
        ]
    );
    assert_eq!(
        state
            .llama_process_rows
            .iter()
            .map(|row| (&row.name, &row.status))
            .collect::<Vec<_>>(),
        vec![(&"llama-server".to_string(), &RuntimeStatus::Starting)]
    );

    state.reduce(DashboardAction::OutputEvent(OutputEvent::WebserverReady {
        url: "http://localhost:3131".to_string(),
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::ApiReady {
        url: "http://localhost:9337".to_string(),
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::LlamaReady {
        model: Some("Qwen3-32B".to_string()),
        port: 9338,
        ctx_size: Some(8192),
        log_path: None,
    }));

    assert!(
        state
            .webserver_rows
            .iter()
            .all(|row| row.status == RuntimeStatus::Ready)
    );
    assert_eq!(state.llama_process_rows[0].status, RuntimeStatus::Ready);
}

#[test]
pub(super) fn fatal_events_do_not_consume_startup_history_slots() {
    let mut formatter = InteractiveDashboardFormatter::default();
    let fatal = OutputEvent::Fatal {
        message: "panic occurred".to_string(),
        context: Some("panic at crates/mesh-llm/src/lib.rs:42".to_string()),
    };

    formatter
        .handle_output_event(&OutputEvent::Startup {
            version: "v0.68.0".to_string(),
            message: None,
        })
        .expect("startup event should reduce cleanly");
    formatter
        .handle_output_event(&fatal)
        .expect("fatal event should reduce cleanly");

    assert_eq!(formatter.state.startup_history.len(), 1);
    assert!(
        formatter
            .state
            .startup_history
            .iter()
            .all(|event| !event.summary.contains("panic occurred"))
    );
    assert!(
        formatter
            .state
            .mesh_events
            .iter()
            .any(|event| event.summary.contains("panic occurred"))
    );
}

#[test]
pub(super) fn startup_failures_surface_in_tui_events_and_status() {
    let mut formatter = InteractiveDashboardFormatter::default();
    for event in [
        OutputEvent::Startup {
            version: "v0.68.0".to_string(),
            message: None,
        },
        OutputEvent::LlamaStarting {
            model: Some("Qwen3-32B".to_string()),
            http_port: 9338,
            ctx_size: Some(8192),
            log_path: Some("/tmp/llama.log".to_string()),
        },
        OutputEvent::LlamaStartupFailed {
            model: Some("Qwen3-32B".to_string()),
            http_port: 9338,
            ctx_size: Some(8192),
            log_path: Some("/tmp/llama.log".to_string()),
            detail: "llama-server exited before becoming healthy".to_string(),
        },
    ] {
        formatter
            .handle_output_event(&event)
            .expect("startup failure events should reduce cleanly");
    }

    formatter
        .handle_output_event(&OutputEvent::Info {
            message: "background retry skipped after startup failure".to_string(),
            context: None,
        })
        .expect("later info events should not clear startup failures");

    let rendered = render_tui_frame_snapshot(&formatter.state, 160, 32);
    let dashboard = render_dashboard_text(&formatter.state);
    assert!(
        rendered.contains("startup=failed"),
        "expected failed lifecycle in {rendered}"
    );
    assert!(dashboard.contains("llama-server=failed  model readiness=failed"));
    assert!(formatter.state.startup_history.iter().any(|event| {
        event
            .summary
            .contains("llama-server exited before becoming healthy")
    }));
}

#[test]
pub(super) fn llama_startup_failures_mark_components_failed() {
    let mut llama_failed = DashboardState::default();
    llama_failed.reduce(DashboardAction::OutputEvent(OutputEvent::Startup {
        version: "v0.68.0".to_string(),
        message: None,
    }));
    llama_failed.reduce(DashboardAction::OutputEvent(OutputEvent::ModelQueued {
        model: "Qwen3-32B".to_string(),
    }));
    llama_failed.reduce(DashboardAction::OutputEvent(OutputEvent::LlamaStarting {
        model: Some("Qwen3-32B".to_string()),
        http_port: 9338,
        ctx_size: Some(8192),
        log_path: None,
    }));
    llama_failed.reduce(DashboardAction::OutputEvent(
        OutputEvent::LlamaStartupFailed {
            model: Some("Qwen3-32B".to_string()),
            http_port: 9338,
            ctx_size: Some(8192),
            log_path: None,
            detail: "llama-server exited before becoming healthy".to_string(),
        },
    ));

    assert_eq!(
        llama_failed.startup_lifecycle().phase,
        StartupLifecyclePhase::Failed
    );
    assert_eq!(
        llama_failed.startup_lifecycle().llama_server.phase,
        StartupLifecyclePhase::Failed
    );
    assert_eq!(
        llama_failed.startup_lifecycle().model_readiness.phase,
        StartupLifecyclePhase::Failed
    );
    assert!(matches!(
        llama_failed
            .llama_instances
            .iter()
            .find(|instance| instance.kind == LlamaInstanceKind::LlamaServer)
            .map(|instance| &instance.status),
        Some(RuntimeStatus::Error)
    ));
    assert!(matches!(
        llama_failed
            .running_models
            .iter()
            .find(|model| model.model == "Qwen3-32B")
            .map(|model| &model.status),
        Some(RuntimeStatus::Error)
    ));
}

#[test]
pub(super) fn generic_error_does_not_guess_last_running_model() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::OutputEvent(OutputEvent::ModelReady {
        model: "Qwen3-32B".to_string(),
        internal_port: Some(9338),
        role: Some("host".to_string()),
    }));

    state.reduce(DashboardAction::OutputEvent(OutputEvent::Error {
        message: "transport stderr surfaced".to_string(),
        context: Some("stderr".to_string()),
    }));

    assert!(matches!(
        state
            .running_models
            .iter()
            .find(|model| model.model == "Qwen3-32B")
            .map(|model| &model.status),
        Some(RuntimeStatus::Ready)
    ));
}

#[test]
pub(super) fn discovery_and_join_failures_mark_startup_mesh_component_failed() {
    let mut discovery_failed = DashboardState::default();
    discovery_failed.reduce(DashboardAction::OutputEvent(OutputEvent::Startup {
        version: "v0.68.0".to_string(),
        message: None,
    }));
    discovery_failed.reduce(DashboardAction::OutputEvent(
        OutputEvent::DiscoveryStarting {
            source: "Nostr auto-discovery".to_string(),
        },
    ));
    discovery_failed.reduce(DashboardAction::OutputEvent(OutputEvent::DiscoveryFailed {
        message: "Nostr auto-discovery failed".to_string(),
        detail: Some("relay timeout".to_string()),
    }));

    assert_eq!(
        discovery_failed.startup_lifecycle().phase,
        StartupLifecyclePhase::Failed
    );
    assert_eq!(
        discovery_failed.startup_lifecycle().mesh.phase,
        StartupLifecyclePhase::Failed
    );
    assert_eq!(
        discovery_failed.startup_lifecycle().mesh.detail.as_deref(),
        Some("Nostr auto-discovery failed: relay timeout")
    );

    let mut join_failed = DashboardState::default();
    join_failed.reduce(DashboardAction::OutputEvent(OutputEvent::Startup {
        version: "v0.68.0".to_string(),
        message: None,
    }));
    join_failed.reduce(DashboardAction::OutputEvent(OutputEvent::WaitingForPeers {
        detail: Some("waiting for peers while joining mesh".to_string()),
    }));
    join_failed.reduce(DashboardAction::OutputEvent(OutputEvent::Warning {
        message: "Failed to join any peer — running standalone".to_string(),
        context: None,
    }));

    assert_eq!(
        join_failed.startup_lifecycle().phase,
        StartupLifecyclePhase::Failed
    );
    assert_eq!(
        join_failed.startup_lifecycle().mesh.phase,
        StartupLifecyclePhase::Failed
    );
    assert_eq!(
        join_failed.startup_lifecycle().mesh.detail.as_deref(),
        Some("Failed to join any peer — running standalone")
    );
}

#[test]
pub(super) fn post_ready_peer_churn_does_not_reopen_startup_failure() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::OutputEvent(OutputEvent::Startup {
        version: "v0.68.0".to_string(),
        message: None,
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::DiscoveryJoined {
        mesh: "poker-night".to_string(),
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::ApiReady {
        url: "http://localhost:9337".to_string(),
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::RuntimeReady {
        api_url: "http://localhost:9337".to_string(),
        console_url: Some("http://localhost:3131".to_string()),
        api_port: 9337,
        console_port: Some(3131),
        models_count: Some(1),
        pi_command: None,
        goose_command: None,
    }));

    assert!(state.runtime_ready);
    assert_eq!(
        state.startup_lifecycle().phase,
        StartupLifecyclePhase::Ready
    );
    assert_eq!(
        state.startup_lifecycle().mesh.phase,
        StartupLifecyclePhase::Ready
    );
    assert_eq!(
        state.startup_lifecycle().mesh.detail.as_deref(),
        Some("joined mesh poker-night")
    );

    for event in [
        OutputEvent::DiscoveryStarting {
            source: "Nostr re-discovery".to_string(),
        },
        OutputEvent::WaitingForPeers {
            detail: Some("waiting for peers after reconnect".to_string()),
        },
        OutputEvent::DiscoveryFailed {
            message: "Nostr re-discovery failed".to_string(),
            detail: Some("relay timeout".to_string()),
        },
        OutputEvent::Warning {
            message: "Failed to join any peer — running standalone".to_string(),
            context: None,
        },
    ] {
        state.reduce(DashboardAction::OutputEvent(event));
    }

    assert_eq!(
        state.startup_lifecycle().phase,
        StartupLifecyclePhase::Ready
    );
    assert_eq!(
        state.startup_lifecycle().mesh.phase,
        StartupLifecyclePhase::Ready
    );
    assert_eq!(
        state.startup_lifecycle().mesh.detail.as_deref(),
        Some("joined mesh poker-night")
    );
}

#[test]
pub(super) fn generic_error_after_runtime_ready_does_not_reopen_startup_failure() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::OutputEvent(OutputEvent::Startup {
        version: "v0.68.0".to_string(),
        message: None,
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::ApiReady {
        url: "http://localhost:9337".to_string(),
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::RuntimeReady {
        api_url: "http://localhost:9337".to_string(),
        console_url: Some("http://localhost:3131".to_string()),
        api_port: 9337,
        console_port: Some(3131),
        models_count: Some(1),
        pi_command: None,
        goose_command: None,
    }));

    assert!(state.runtime_ready);
    assert_eq!(
        state.startup_lifecycle().phase,
        StartupLifecyclePhase::Ready
    );

    state.reduce(DashboardAction::OutputEvent(OutputEvent::Error {
        message: "native stderr surfaced after startup".to_string(),
        context: Some("stderr".to_string()),
    }));

    assert!(state.runtime_ready);
    assert_eq!(
        state.startup_lifecycle().phase,
        StartupLifecyclePhase::Ready
    );
    assert_eq!(state.startup_lifecycle().failure, None);
    assert!(render_dashboard_text(&state).contains("startup=ready"));
}

#[test]
pub(super) fn shutdown_requested_marks_runtime_shutting_down() {
    let mut state = DashboardState::default();

    state.reduce(DashboardAction::OutputEvent(OutputEvent::Startup {
        version: "v0.68.0".to_string(),
        message: None,
    }));

    state.reduce(DashboardAction::OutputEvent(
        OutputEvent::ShutdownRequested { signal: "SIGINT" },
    ));

    assert_eq!(
        state.startup_lifecycle().phase,
        StartupLifecyclePhase::ShuttingDown,
        "ShutdownRequested should mark lifecycle as ShuttingDown"
    );
}

#[test]
pub(super) fn shutdown_suppresses_subsequent_model_ready_events() {
    let mut state = DashboardState::default();

    state.reduce(DashboardAction::OutputEvent(OutputEvent::Startup {
        version: "v0.68.0".to_string(),
        message: None,
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::ApiStarting {
        url: "http://localhost:9337".to_string(),
    }));
    state.reduce(DashboardAction::OutputEvent(
        OutputEvent::ShutdownRequested { signal: "SIGTERM" },
    ));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::ModelReady {
        model: "Qwen3-32B".to_string(),
        internal_port: Some(9338),
        role: Some("host".to_string()),
    }));

    assert_eq!(
        state.startup_lifecycle().phase,
        StartupLifecyclePhase::ShuttingDown,
        "Shutdown should suppress late ModelReady"
    );
}

#[test]
pub(super) fn model_unloading_updates_model_row_status() {
    let mut state = DashboardState::default();

    state.reduce(DashboardAction::OutputEvent(OutputEvent::Startup {
        version: "v0.68.0".to_string(),
        message: None,
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::ModelLoaded {
        model: "Qwen3-32B".to_string(),
        bytes: Some(8_000_000_000),
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::ModelReady {
        model: "Qwen3-32B".to_string(),
        internal_port: Some(9338),
        role: Some("host".to_string()),
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::ModelUnloading {
        model: "Qwen3-32B".to_string(),
    }));

    let rendered = render_dashboard_text(&state);
    assert!(
        rendered.contains("Qwen3-32B"),
        "model should still appear in dashboard after unloading"
    );
    assert!(
        rendered.contains("stopped"),
        "dashboard should show the unloading model as stopped"
    );
}

#[test]
pub(super) fn model_unloaded_preserves_model_in_dashboard() {
    let mut state = DashboardState::default();

    state.reduce(DashboardAction::OutputEvent(OutputEvent::Startup {
        version: "v0.68.0".to_string(),
        message: None,
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::ModelLoaded {
        model: "Llama-3.1-8B".to_string(),
        bytes: Some(4_500_000_000),
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::ModelReady {
        model: "Llama-3.1-8B".to_string(),
        internal_port: Some(9338),
        role: Some("host".to_string()),
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::ModelUnloading {
        model: "Llama-3.1-8B".to_string(),
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::ModelUnloaded {
        model: "Llama-3.1-8B".to_string(),
    }));

    let rendered = render_dashboard_text(&state);
    assert!(
        rendered.contains("Llama-3.1-8B"),
        "model should still be visible in dashboard after full unload cycle"
    );
    assert!(
        rendered.contains("stopped"),
        "dashboard should keep the unloaded model row stopped"
    );
}
#[test]
pub(super) fn startup_history_is_visible_after_late_tui_attach() {
    let mut formatter = InteractiveDashboardFormatter::default();
    for event in [
        OutputEvent::Startup {
            version: "v0.68.0".to_string(),
            message: None,
        },
        OutputEvent::NodeIdentity {
            node_id: "node-7".to_string(),
            mesh_id: Some("poker-night".to_string()),
        },
        OutputEvent::ApiStarting {
            url: "http://localhost:9337".to_string(),
        },
        OutputEvent::LlamaStarting {
            model: Some("Qwen3-32B".to_string()),
            http_port: 9338,
            ctx_size: Some(8192),
            log_path: None,
        },
    ] {
        formatter
            .handle_output_event(&event)
            .expect("pre-attach startup events should reduce cleanly");
    }

    let rendered = render_tui_frame_snapshot(&formatter.state, 160, 32);

    assert!(
        rendered.contains("startup=partial"),
        "expected lifecycle summary in {rendered}"
    );
    assert!(
        rendered.contains("mesh-llm starting"),
        "expected startup line in {rendered}"
    );
    assert!(
        rendered.contains("node node-7 joined mesh poker-night"),
        "expected node identity line in {rendered}"
    );
    assert!(
        rendered.contains("api starting at http://localhost:9337"),
        "expected API start line in {rendered}"
    );
    assert!(
        rendered.contains("llama-server starting: port=9338 model=Qwen3-32B"),
        "expected llama start line in {rendered}"
    );
    assert!(
        rendered.contains("Mesh Events"),
        "late attach should render the main dashboard now that the loading screen is gone"
    );
    assert!(
        formatter
            .state
            .startup_history
            .iter()
            .any(|event| event.summary.contains("llama-server starting: port=9338"))
    );
}

#[test]
pub(super) fn startup_history_keeps_order_when_tui_attaches_late() {
    let mut formatter = InteractiveDashboardFormatter::default();
    for event in [
        OutputEvent::Startup {
            version: "v0.68.0".to_string(),
            message: None,
        },
        OutputEvent::NodeIdentity {
            node_id: "node-7".to_string(),
            mesh_id: Some("poker-night".to_string()),
        },
        OutputEvent::ApiStarting {
            url: "http://localhost:9337".to_string(),
        },
        OutputEvent::ApiReady {
            url: "http://localhost:9337".to_string(),
        },
        OutputEvent::LlamaStarting {
            model: Some("Qwen3-32B".to_string()),
            http_port: 9338,
            ctx_size: Some(8192),
            log_path: None,
        },
    ] {
        formatter
            .handle_output_event(&event)
            .expect("pre-attach startup events should reduce cleanly");
    }

    let rendered = render_tui_frame_snapshot(&formatter.state, 160, 32);
    assert!(rendered.contains("Mesh Events"));
    let history: Vec<&str> = formatter
        .state
        .startup_history
        .iter()
        .map(|event| event.summary.as_str())
        .collect();
    let startup_index = history
        .iter()
        .position(|summary| summary.contains("mesh-llm starting"))
        .expect("expected startup line in retained history");
    let node_index = history
        .iter()
        .position(|summary| summary.contains("node node-7 joined mesh poker-night"))
        .expect("expected node identity line in retained history");
    let api_start_index = history
        .iter()
        .position(|summary| summary.contains("api starting at http://localhost:9337"))
        .expect("expected API start line in retained history");
    let api_ready_index = history
        .iter()
        .position(|summary| summary.contains("api ready at http://localhost:9337"))
        .expect("expected API ready line in retained history");

    assert!(startup_index < node_index);
    assert!(node_index < api_start_index);
    assert!(api_start_index < api_ready_index);
}

#[test]
pub(super) fn startup_launch_plan_renders_not_ready_rows_before_actions() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
        160, 32,
    )));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::Startup {
        version: "v0.68.0".to_string(),
        message: None,
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::LaunchPlan {
        plan: sample_launch_plan(),
    }));
    let rendered = render_tui_frame_snapshot(&state, 160, 32);

    assert!(
        rendered.contains("Mesh Events"),
        "expected dashboard in {rendered}"
    );
    assert!(
        rendered.contains("NOT READY"),
        "expected not-ready rows in {rendered}"
    );
    assert!(rendered.contains("Console"));
    assert!(rendered.contains("Planned-Model"));
    assert_eq!(state.llama_process_rows[0].status, RuntimeStatus::Loading);
    assert_eq!(state.webserver_rows[0].status, RuntimeStatus::NotReady);
    assert_eq!(state.loaded_model_rows[0].status, RuntimeStatus::Loading);
}

#[test]
pub(super) fn startup_progress_after_launch_plan_shows_dashboard_not_loader() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
        160, 32,
    )));
    state.reduce(DashboardAction::OutputEvent(
        OutputEvent::ModelDownloadProgress {
            label: "Planned-Model".to_string(),
            file: Some("planned-model.gguf".to_string()),
            downloaded_bytes: Some(100),
            total_bytes: Some(100),
            status: ModelProgressStatus::Ready,
        },
    ));

    let loader_render = render_tui_frame_snapshot(&state, 160, 32);
    assert!(state.active_loading_progress().is_some());
    assert!(
        loader_render.contains("Mesh Events"),
        "startup progress should use the dashboard instead of a full-screen loader: {loader_render}"
    );
    assert!(!loader_render.contains('█'));

    state.reduce(DashboardAction::OutputEvent(OutputEvent::LaunchPlan {
        plan: sample_launch_plan(),
    }));

    let dashboard_render = render_tui_frame_snapshot(&state, 160, 32);
    assert!(state.active_loading_progress().is_some());
    assert!(
        dashboard_render.contains("Mesh Events"),
        "expected dashboard after launch plan in {dashboard_render}"
    );
    assert!(dashboard_render.contains("NOT READY"));
    assert!(dashboard_render.contains("Planned-Model"));
}

#[test]
pub(super) fn shutdown_suppresses_late_ready_render() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::OutputEvent(OutputEvent::Startup {
        version: "v0.68.0".to_string(),
        message: None,
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::ApiStarting {
        url: "http://localhost:9337".to_string(),
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::Shutdown {
        reason: None,
    }));

    for event in [
        OutputEvent::ApiReady {
            url: "http://localhost:9337".to_string(),
        },
        OutputEvent::RuntimeReady {
            api_url: "http://localhost:9337".to_string(),
            console_url: Some("http://localhost:3131".to_string()),
            api_port: 9337,
            console_port: Some(3131),
            models_count: Some(1),
            pi_command: None,
            goose_command: None,
        },
    ] {
        state.reduce(DashboardAction::OutputEvent(event));
    }

    let dashboard = render_dashboard_text(&state);
    let rendered = render_tui_frame_snapshot(&state, 160, 32);

    assert_eq!(
        state.startup_lifecycle().phase,
        StartupLifecyclePhase::ShuttingDown
    );
    assert!(dashboard.contains("startup=shutting down"));
    assert!(rendered.contains("startup=shutting down"));
    assert!(!dashboard.contains("mesh-llm runtime ready"));
    assert!(!rendered.contains("mesh-llm runtime ready"));
    assert!(matches!(
        state.api.as_ref().map(|api| &api.status),
        Some(RuntimeStatus::ShuttingDown)
    ));
}
