use super::*;

#[test]
pub(super) fn layer_package_progress_message_names_artifact_and_package() {
    let message = format_model_download_progress_message(
        "layer package meshllm/demo-layers",
        Some("shared/embeddings.gguf"),
        Some(256_000_000),
        Some(512_000_000),
        &ModelProgressStatus::Downloading,
    );

    assert_eq!(
        message,
        "downloading layer package artifact shared/embeddings.gguf for meshllm/demo-layers 256MB/512MB"
    );
}

#[test]
pub(super) fn multipart_model_progress_message_reports_part_counts() {
    let message = format_model_download_progress_message(
        "parts::org/repo:model",
        None,
        Some(2),
        Some(3),
        &ModelProgressStatus::Downloading,
    );

    assert_eq!(message, "downloading model parts for org/repo:model 2/3");
}

#[test]
pub(super) fn startup_failure_summary_sanitizes_multiline_detail() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::OutputEvent(OutputEvent::Startup {
        version: "v0.68.0".to_string(),
        message: None,
    }));
    state.reduce(DashboardAction::OutputEvent(
        OutputEvent::LlamaStartupFailed {
            model: Some("Qwen3-32B".to_string()),
            http_port: 9338,
            ctx_size: Some(8192),
            log_path: Some("/tmp/skippy-native.log".to_string()),
            detail: "llama-server exited
See /tmp/skippy-native.log:
tail line"
                .to_string(),
        },
    ));

    let summary = render_startup_summary(&state);
    assert_eq!(
        summary[0],
        "startup=failed  failure=llama-server exited See /tmp/skippy-native.log: tail line"
    );
    assert!(!summary[0].contains('\n'));

    let tui_summary =
        spans_plain_text(&startup_lifecycle_summary_line(&state.startup_lifecycle, 160).spans);
    assert!(
        tui_summary.contains("failure=llama-server exited See /tmp/skippy-native.log: tail line")
    );
    assert!(!tui_summary.contains('\n'));

    let title = join_token_panel_right_title(&state);
    assert!(title.starts_with("startup failed: llama-server exited See"));
    assert!(!title.contains('\n'));
}

#[test]
pub(super) fn fallback_mode_surfaces_startup_failures_without_tui() {
    let mut formatter = DashboardFormatter::default();
    let mut rendered = String::new();

    for event in [
        OutputEvent::Startup {
            version: "v0.68.0".to_string(),
            message: None,
        },
        OutputEvent::LlamaStarting {
            model: Some("Qwen3-32B".to_string()),
            http_port: 9338,
            ctx_size: Some(8192),
            log_path: None,
        },
        OutputEvent::LlamaStartupFailed {
            model: Some("Qwen3-32B".to_string()),
            http_port: 9338,
            ctx_size: Some(8192),
            log_path: None,
            detail: "llama-server exited before listening".to_string(),
        },
    ] {
        rendered = formatter
            .format(&event)
            .expect("fallback formatter should keep rendering durable startup failures");
    }

    assert!(rendered.contains("startup=failed"));
    assert!(rendered.contains("llama-server=failed"));
    assert!(rendered.contains("llama-server exited before listening"));
}

#[test]
pub(super) fn json_formatter_emits_app_owned_ndjson() {
    let mut output = Vec::new();
    let mut formatter = JsonFormatter;

    output
        .write_all(
            formatter
                .format(&OutputEvent::RpcServerStarting {
                    port: 43683,
                    device: "CUDA0".to_string(),
                    log_path: Some("/tmp/rpc.log".to_string()),
                })
                .expect("json emit should succeed")
                .as_bytes(),
        )
        .expect("write should succeed");

    let rendered = String::from_utf8(output).expect("output should be utf8");
    let line = rendered.trim_end();
    let value: Value = serde_json::from_str(line).expect("line should parse as json");
    assert_eq!(value["event"], "rpc_server_starting");
    assert_eq!(value["device"], "CUDA0");
    assert_eq!(value["log_path"], "/tmp/rpc.log");
    assert!(rendered.ends_with('\n'));
}

#[test]
pub(super) fn json_formatter_emits_llama_server_starting_payload() {
    let mut formatter = JsonFormatter;
    let rendered = formatter
        .format(&OutputEvent::LlamaStarting {
            model: Some("Qwen3.6-35B".to_string()),
            http_port: 43683,
            ctx_size: Some(8192),
            log_path: Some("/tmp/llama.log".to_string()),
        })
        .expect("llama startup render should succeed");
    let value: Value = serde_json::from_str(rendered.trim_end()).expect("line should parse");

    assert_eq!(value["event"], "llama_starting");
    assert_eq!(value["model"], "Qwen3.6-35B");
    assert_eq!(value["http_port"], 43683);
    assert_eq!(value["ctx_size"], 8192);
    assert_eq!(value["log_path"], "/tmp/llama.log");
}

#[test]
pub(super) fn json_formatter_includes_invite_mesh_metadata() {
    let mut formatter = JsonFormatter;
    let rendered = formatter
        .format(&OutputEvent::InviteToken {
            token: "invite-token".to_string(),
            mesh_id: "mesh-123".to_string(),
            mesh_name: None,
        })
        .expect("invite render should succeed");
    let value: Value = serde_json::from_str(rendered.trim_end()).expect("line should parse");

    assert_eq!(value["event"], "invite_token");
    assert_eq!(value["token"], "invite-token");
    assert_eq!(value["mesh_id"], "mesh-123");
}

#[test]
pub(super) fn json_formatter_includes_discovery_payloads() {
    let mut formatter = JsonFormatter;

    let started = formatter
        .format(&OutputEvent::DiscoveryStarting {
            source: "Nostr re-discovery".to_string(),
        })
        .expect("discovery start render should succeed");
    let started_value: Value = serde_json::from_str(started.trim_end()).expect("json line");
    assert_eq!(started_value["event"], "discovery_starting");
    assert_eq!(started_value["source"], "Nostr re-discovery");

    let candidate = formatter
        .format(&OutputEvent::MeshFound {
            mesh: "poker-night".to_string(),
            peers: 7,
            region: None,
        })
        .expect("discovery candidate render should succeed");
    let candidate_value: Value = serde_json::from_str(candidate.trim_end()).expect("json line");
    assert_eq!(candidate_value["event"], "mesh_found");
    assert_eq!(candidate_value["mesh"], "poker-night");
    assert_eq!(candidate_value["peers"], 7);
    assert_eq!(candidate_value["region"], Value::Null);

    let joined = formatter
        .format(&OutputEvent::DiscoveryJoined {
            mesh: "poker-night".to_string(),
        })
        .expect("discovery join render should succeed");
    let joined_value: Value = serde_json::from_str(joined.trim_end()).expect("json line");
    assert_eq!(joined_value["event"], "discovery_joined");
    assert_eq!(joined_value["mesh"], "poker-night");

    let failed = formatter
        .format(&OutputEvent::DiscoveryFailed {
            message: "Could not re-join any mesh — will retry".to_string(),
            detail: None,
        })
        .expect("discovery failure render should succeed");
    let failed_value: Value = serde_json::from_str(failed.trim_end()).expect("json line");
    assert_eq!(failed_value["event"], "discovery_failed");
    assert_eq!(
        failed_value["message"],
        "Could not re-join any mesh — will retry"
    );
    assert_eq!(failed_value["detail"], Value::Null);
}

#[test]
pub(super) fn dashboard_formatter_renders_invite_and_waiting_events_into_mesh_history() {
    let mut formatter = DashboardFormatter::with_state(DashboardState::with_mesh_event_limit(4));

    let _ = formatter
        .format(&OutputEvent::InviteToken {
            token: "invite-token".to_string(),
            mesh_id: "mesh-123".to_string(),
            mesh_name: None,
        })
        .expect("invite render should succeed");
    let dashboard = formatter
        .format(&OutputEvent::WaitingForPeers { detail: None })
        .expect("waiting render should succeed");

    assert!(dashboard.contains("Mesh events (latest 4)"));
    assert!(dashboard.contains("Invite created for mesh mesh-123: invite-token"));
    assert!(dashboard.contains("Waiting for peers..."));
    assert!(!dashboard.contains('📡'));
    for line in dashboard
        .lines()
        .filter(|line| line.contains("Waiting for peers"))
    {
        assert!(
            !line.contains('⏳'),
            "mesh event line should be emoji-free: {line}"
        );
    }
}

#[test]
pub(super) fn tui_falls_back_to_legacy_stderr_render_when_not_tty() {
    let mut formatter = select_formatter(LogFormat::Pretty, ConsoleSessionMode::Fallback);

    assert_eq!(formatter.kind(), "pretty_fallback");

    let dashboard = formatter
        .format(&OutputEvent::RuntimeReady {
            api_url: "http://localhost:9337".to_string(),
            console_url: Some("http://localhost:3131".to_string()),
            api_port: 9337,
            console_port: Some(3131),
            models_count: Some(1),
            pi_command: None,
            goose_command: None,
        })
        .expect("fallback render should succeed");

    assert!(dashboard.contains("Running llama.cpp instances"));
    assert!(dashboard.contains("Running API"));
    assert!(dashboard.contains("OpenAI-compatible API   ready   http://localhost:9337"));
    assert!(!dashboard.contains("\u{1b}[?1049h"));
    assert!(!dashboard.contains("\u{1b}[?1049l"));
    assert!(!dashboard.contains("\u{1b}[?25l"));
    assert!(!dashboard.contains("\u{1b}[?25h"));
}

#[test]
pub(super) fn tui_event_loop_dispatches_quit_on_q() {
    let mut formatter = InteractiveDashboardFormatter::default();

    assert_eq!(
        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Char('q'))),
        TuiControlFlow::Quit
    );
    assert_eq!(
        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Interrupt)),
        TuiControlFlow::Quit
    );
}

#[test]
pub(super) fn interactive_preterminal_render_uses_plain_event_output() {
    let mut formatter = InteractiveDashboardFormatter::default();

    let rendered = formatter
        .handle_output_event(&OutputEvent::RuntimeReady {
            api_url: "http://localhost:9337".to_string(),
            console_url: Some("http://localhost:3131".to_string()),
            api_port: 9337,
            console_port: Some(3131),
            models_count: Some(1),
            pi_command: None,
            goose_command: None,
        })
        .expect("interactive pre-terminal render should succeed")
        .expect("interactive formatter should emit a normal console line");

    assert_eq!(rendered, "✅ Mesh runtime ready (1 model(s))\n");
    assert!(!rendered.contains("Incoming Requests"));
    assert!(!rendered.contains('─'));
    assert!(!rendered.contains('│'));
    assert!(!rendered.contains("Running llama.cpp instances"));
    assert!(!rendered.contains("Running models"));
}

#[test]
pub(super) fn interactive_post_terminal_exit_resumes_plain_event_output() {
    let mut formatter = InteractiveDashboardFormatter {
        terminal_active: true,
        ..Default::default()
    };

    let active_shutdown = formatter
        .handle_output_event(&OutputEvent::Shutdown { reason: None })
        .expect("active TUI event formatting should succeed");
    assert!(
        active_shutdown.is_none(),
        "active TUI should not emit normal console output"
    );

    formatter.terminal_active = false;

    let shutdown = formatter
        .handle_output_event(&OutputEvent::Shutdown { reason: None })
        .expect("inactive post-exit event formatting should succeed")
        .expect("post-exit event should resume normal pretty output");
    assert_eq!(shutdown, "mesh-llm shutting down\n");
    assert!(!shutdown.contains("Mesh Events"));
    assert!(!shutdown.contains('─'));

    let ready = formatter
        .handle_output_event(&OutputEvent::RuntimeReady {
            api_url: "http://localhost:9337".to_string(),
            console_url: Some("http://localhost:3131".to_string()),
            api_port: 9337,
            console_port: Some(3131),
            models_count: Some(1),
            pi_command: None,
            goose_command: None,
        })
        .expect("post-exit runtime event formatting should succeed")
        .expect("post-exit runtime event should remain visible as plain output");
    assert_eq!(ready, "✅ Mesh runtime ready (1 model(s))\n");
    assert!(!ready.contains("Incoming Requests"));
    assert!(!ready.contains('│'));
}

#[tokio::test]
pub(super) async fn dashboard_snapshot_registration_stays_pretty_only() {
    let dashboard_manager =
        OutputManager::new(LogFormat::Pretty, ConsoleSessionMode::InteractiveDashboard);
    let json_manager = OutputManager::new(LogFormat::Json, ConsoleSessionMode::None);
    let expected = DashboardSnapshot {
        current_inflight_requests: 3,
        ..DashboardSnapshot::default()
    };
    let provider = Arc::new(StaticDashboardSnapshotProvider {
        snapshot: expected.clone(),
    });

    dashboard_manager.register_dashboard_snapshot_provider(provider.clone());
    json_manager.register_dashboard_snapshot_provider(provider);

    assert_eq!(dashboard_manager.dashboard_snapshot().await, Some(expected));
    assert_eq!(json_manager.dashboard_snapshot().await, None);
}

#[tokio::test]
pub(super) async fn output_manager_reset_replaces_runtime_owned_state() {
    let manager = OutputManager::new(LogFormat::Json, ConsoleSessionMode::None);

    assert!(matches!(manager.mode(), LogFormat::Json));
    assert_eq!(manager.console_session_mode(), None);

    manager.reset(LogFormat::Pretty, ConsoleSessionMode::Fallback);

    assert!(matches!(manager.mode(), LogFormat::Pretty));
    assert_eq!(
        manager.console_session_mode(),
        Some(ConsoleSessionMode::Fallback)
    );
    assert!(manager.flush().await.is_ok());
}

#[test]
pub(super) fn json_formatter_writes_machine_output_to_stdout_only() {
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();

    write_rendered_output_to_writers(
        LogFormat::Json,
        "{\"event\":\"ready\"}\n",
        &mut stdout,
        &mut stderr,
    )
    .expect("json write should succeed");

    assert_eq!(
        String::from_utf8(stdout).expect("stdout should be utf-8"),
        "{\"event\":\"ready\"}\n"
    );
    assert!(
        stderr.is_empty(),
        "json output must not be routed to stderr"
    );
}

#[test]
pub(super) fn dashboard_formatter_renders_discovery_events_into_mesh_history() {
    let mut formatter = DashboardFormatter::with_state(DashboardState::with_mesh_event_limit(4));

    formatter
        .format(&OutputEvent::DiscoveryStarting {
            source: "Nostr re-discovery".to_string(),
        })
        .expect("discovery start render should succeed");
    formatter
        .format(&OutputEvent::MeshFound {
            mesh: "poker-night".to_string(),
            peers: 7,
            region: None,
        })
        .expect("discovery candidate render should succeed");
    let dashboard = formatter
        .format(&OutputEvent::DiscoveryJoined {
            mesh: "poker-night".to_string(),
        })
        .expect("discovery join render should succeed");

    assert!(dashboard.contains("discovering mesh via Nostr re-discovery"));
    assert!(dashboard.contains("discovered mesh poker-night (7 peer(s))"));
    assert!(dashboard.contains("joined mesh poker-night"));
    assert!(!dashboard.contains('🔍'));
    assert!(!dashboard.contains('📡'));
    assert!(!dashboard.contains('✅'));
}

#[test]
pub(super) fn dashboard_formatter_renders_discovery_failure_in_mesh_history() {
    let mut formatter = DashboardFormatter::with_state(DashboardState::with_mesh_event_limit(4));

    let dashboard = formatter
        .format(&OutputEvent::DiscoveryFailed {
            message: "Nostr re-discovery failed".to_string(),
            detail: Some("relay timeout".to_string()),
        })
        .expect("discovery failure render should succeed");

    assert!(dashboard.contains("Nostr re-discovery failed: relay timeout"));
    assert!(!dashboard.contains("⚠️ Nostr re-discovery failed"));
}

#[test]
pub(super) fn dashboard_formatter_renders_warning_context_in_mesh_history() {
    let mut formatter = DashboardFormatter::with_state(DashboardState::with_mesh_event_limit(4));

    let dashboard = formatter
        .format(&OutputEvent::Warning {
            message: "llama-server process exited unexpectedly".to_string(),
            context: Some("model=Qwen3-32B port=9337".to_string()),
        })
        .expect("warning render should succeed");

    assert!(
        dashboard.contains("model=Qwen3-32B port=9337: llama-server process exited unexpectedly")
    );
    assert!(!dashboard.contains("⚠️ model=Qwen3-32B port=9337"));

    let dashboard = formatter
            .format(&OutputEvent::Warning {
                message: "⚠️ top-level --client now maps to `mesh-llm client`; re-running with client semantics"
                    .to_string(),
                context: None,
            })
            .expect("warning render with embedded icon should succeed");

    assert!(dashboard.contains(
        "top-level --client now maps to `mesh-llm client`; re-running with client semantics"
    ));
    assert!(!dashboard.contains(
        "⚠️ ⚠️ top-level --client now maps to `mesh-llm client`; re-running with client semantics"
    ));
}

#[test]
pub(super) fn dashboard_formatter_renders_info_context_in_mesh_history() {
    let mut formatter = DashboardFormatter::with_state(DashboardState::with_mesh_event_limit(4));

    let dashboard = formatter
        .format(&OutputEvent::Info {
            message: "mesh named poker-night is private by default".to_string(),
            context: Some("publish=false".to_string()),
        })
        .expect("info render should succeed");

    assert!(dashboard.contains("publish=false: mesh named poker-night is private by default"));
}

#[test]
pub(super) fn dashboard_formatter_renders_multi_model_mode_in_running_models_section() {
    let mut formatter = DashboardFormatter::with_state(DashboardState::with_mesh_event_limit(4));

    formatter
        .format(&OutputEvent::MultiModelMode {
            count: 3,
            models: vec![
                "Qwen2.5-32B".to_string(),
                "GLM-4.7-Flash".to_string(),
                "MiniMax-M2.5".to_string(),
            ],
        })
        .expect("multi-model render should succeed");
    formatter
        .format(&OutputEvent::ModelReady {
            model: "GLM-4.7-Flash".to_string(),
            internal_port: Some(3001),
            role: Some("host".to_string()),
        })
        .expect("model render should succeed");
    let dashboard = formatter
        .format(&OutputEvent::ModelReady {
            model: "Qwen2.5-32B".to_string(),
            internal_port: Some(3002),
            role: Some("standby".to_string()),
        })
        .expect("model render should succeed");

    assert!(dashboard.contains("Running models"));
    assert!(dashboard.contains(
        "multi-model mode   3 model(s)   models=Qwen2.5-32B, GLM-4.7-Flash, MiniMax-M2.5"
    ));
    assert!(dashboard.contains("GLM-4.7-Flash   ready   port=3001   role=host"));
    assert!(dashboard.contains("Qwen2.5-32B   ready   port=3002   role=standby"));
}

#[test]
pub(super) fn dashboard_formatter_pins_host_elected_role_and_capacity() {
    let mut formatter = DashboardFormatter::with_state(DashboardState::with_mesh_event_limit(4));

    let dashboard = formatter
        .format(&OutputEvent::HostElected {
            model: "Qwen3-32B".to_string(),
            host: "node-7".to_string(),
            role: Some("host".to_string()),
            capacity_gb: Some(24.0),
        })
        .expect("host election render should succeed");

    assert!(dashboard.contains("Running models"));
    assert!(dashboard.contains("Qwen3-32B   starting   role=host   capacity=24.0GB"));
    assert!(dashboard.contains("Qwen3-32B elected node-7 as host (24.0GB capacity)"));
    assert!(!dashboard.contains('🗳'));
}

#[test]
pub(super) fn dashboard_formatter_pins_passive_mode_in_running_models() {
    let mut formatter = DashboardFormatter::with_state(DashboardState::with_mesh_event_limit(4));

    let dashboard = formatter
            .format(&OutputEvent::PassiveMode {
                role: "standby".to_string(),
                status: RuntimeStatus::Starting,
                capacity_gb: Some(24.0),
                models_on_disk: Some(vec!["Qwen2.5-32B".to_string(), "GLM-4.7-Flash".to_string()]),
                detail: Some("No matching model on disk — running as standby GPU node. Proxying requests to other nodes. Will activate when needed.".to_string()),
            })
            .expect("passive mode render should succeed");

    assert!(dashboard.contains("Running models"));
    assert!(
        dashboard
            .contains("standby   starting   capacity=24.0GB   models=Qwen2.5-32B, GLM-4.7-Flash")
    );
    assert!(dashboard.contains("No matching model on disk — running as standby GPU node."));
    assert!(dashboard.contains("No matching model on disk — running as standby GPU node. Proxying requests to other nodes. Will activate when needed. (24.0GB capacity) models=Qwen2.5-32B, GLM-4.7-Flash"));
    assert!(!dashboard.contains('💤'));
}

#[test]
pub(super) fn json_formatter_includes_multi_model_mode_payload() {
    let mut formatter = JsonFormatter;
    let rendered = formatter
        .format(&OutputEvent::MultiModelMode {
            count: 2,
            models: vec!["Qwen2.5-32B".to_string(), "GLM-4.7-Flash".to_string()],
        })
        .expect("multi-model render should succeed");
    let value: Value = serde_json::from_str(rendered.trim_end()).expect("line should parse");

    assert_eq!(value["event"], "multi_model_mode");
    assert_eq!(value["count"], 2);
    assert_eq!(
        value["models"],
        serde_json::json!(["Qwen2.5-32B", "GLM-4.7-Flash"])
    );
}

#[test]
pub(super) fn json_formatter_includes_warning_context() {
    let mut formatter = JsonFormatter;
    let rendered = formatter
        .format(&OutputEvent::Warning {
            message: "Failed to start llama-server: bind error".to_string(),
            context: Some("model=Qwen3-32B mode=dense port=9337".to_string()),
        })
        .expect("warning render should succeed");
    let value: Value = serde_json::from_str(rendered.trim_end()).expect("line should parse");

    assert_eq!(value["event"], "warning");
    assert_eq!(value["warning"], "Failed to start llama-server: bind error");
    assert_eq!(value["context"], "model=Qwen3-32B mode=dense port=9337");
}

#[test]
pub(super) fn json_formatter_includes_fatal_level_and_context() {
    let mut formatter = JsonFormatter;
    let rendered = formatter
        .format(&OutputEvent::Fatal {
            message: "panic occurred".to_string(),
            context: Some("panic at crates/mesh-llm/src/lib.rs:42".to_string()),
        })
        .expect("fatal render should succeed");
    let value: Value = serde_json::from_str(rendered.trim_end()).expect("line should parse");

    assert_eq!(value["event"], "fatal");
    assert_eq!(value["level"], "fatal");
    assert_eq!(value["fatal"], "panic occurred");
    assert_eq!(value["context"], "panic at crates/mesh-llm/src/lib.rs:42");
}

#[test]
pub(super) fn emergency_fatal_event_renders_without_dashboard_worker() {
    let event = OutputEvent::Fatal {
        message: "panic occurred".to_string(),
        context: Some("panic at crates/mesh-llm/src/lib.rs:42".to_string()),
    };

    let rendered = render_emergency_event(LogFormat::Pretty, &event)
        .expect("emergency fatal render should succeed");

    assert_eq!(
        rendered,
        "panic at crates/mesh-llm/src/lib.rs:42: panic occurred\n"
    );
}

#[test]
pub(super) fn json_formatter_includes_info_context() {
    let mut formatter = JsonFormatter;
    let rendered = formatter
        .format(&OutputEvent::Info {
            message: "joined mesh".to_string(),
            context: Some("mesh=mesh-123".to_string()),
        })
        .expect("info render should succeed");
    let value: Value = serde_json::from_str(rendered.trim_end()).expect("line should parse");

    assert_eq!(value["event"], "info");
    assert_eq!(value["message"], "joined mesh");
    assert_eq!(value["context"], "mesh=mesh-123");
}

#[test]
pub(super) fn json_formatter_includes_model_ready_port() {
    let mut formatter = JsonFormatter;
    let rendered = formatter
        .format(&OutputEvent::ModelReady {
            model: "Qwen3-32B".to_string(),
            internal_port: Some(3002),
            role: Some("host".to_string()),
        })
        .expect("model ready render should succeed");
    let value: Value = serde_json::from_str(rendered.trim_end()).expect("line should parse");

    assert_eq!(value["event"], "model_ready");
    assert_eq!(value["model"], "Qwen3-32B");
    assert_eq!(value["port"], serde_json::json!(3002));
    assert_eq!(value["internal_port"], serde_json::json!(3002));
    assert_eq!(value["role"], "host");
}

#[test]
pub(super) fn json_formatter_includes_host_elected_role_and_capacity() {
    let mut formatter = JsonFormatter;
    let rendered = formatter
        .format(&OutputEvent::HostElected {
            model: "Qwen3-32B".to_string(),
            host: "node-7".to_string(),
            role: Some("host".to_string()),
            capacity_gb: Some(24.0),
        })
        .expect("host election render should succeed");
    let value: Value = serde_json::from_str(rendered.trim_end()).expect("line should parse");

    assert_eq!(value["event"], "host_elected");
    assert_eq!(value["model"], "Qwen3-32B");
    assert_eq!(value["host"], "node-7");
    assert_eq!(value["role"], "host");
    assert_eq!(value["capacity_gb"], serde_json::json!(24.0));
}

#[test]
pub(super) fn json_formatter_includes_passive_mode_payload() {
    let mut formatter = JsonFormatter;
    let rendered = formatter
        .format(&OutputEvent::PassiveMode {
            role: "client".to_string(),
            status: RuntimeStatus::Ready,
            capacity_gb: None,
            models_on_disk: None,
            detail: Some("Client ready".to_string()),
        })
        .expect("passive mode render should succeed");
    let value: Value = serde_json::from_str(rendered.trim_end()).expect("line should parse");

    assert_eq!(value["event"], "passive_mode");
    assert_eq!(value["role"], "client");
    assert_eq!(value["status"], "ready");
    assert_eq!(value["capacity_gb"], Value::Null);
    assert_eq!(value["models_on_disk"], Value::Null);
    assert_eq!(value["detail"], "Client ready");
}

#[test]
pub(super) fn dashboard_formatter_keeps_pinned_sections_and_bounds_mesh_history() {
    let mut formatter = DashboardFormatter::with_state(DashboardState::with_mesh_event_limit(2));

    formatter
        .format(&OutputEvent::Startup {
            version: "v0.64.0".to_string(),
            message: None,
        })
        .expect("startup render should succeed");
    formatter
        .format(&OutputEvent::LlamaStarting {
            model: Some("Qwen3.6-35B".to_string()),
            http_port: 43683,
            ctx_size: Some(8192),
            log_path: Some("/tmp/llama.log".to_string()),
        })
        .expect("llama render should succeed");
    formatter
        .format(&OutputEvent::RpcServerStarting {
            port: 43683,
            device: "CUDA0".to_string(),
            log_path: Some("/tmp/rpc.log".to_string()),
        })
        .expect("rpc render should succeed");
    formatter
        .format(&OutputEvent::ModelReady {
            model: "Qwen3.6-35B".to_string(),
            internal_port: Some(38373),
            role: Some("host".to_string()),
        })
        .expect("model render should succeed");
    formatter
            .format(&OutputEvent::RuntimeReady {
                api_url: "http://localhost:9337".to_string(),
                console_url: Some("http://localhost:3131".to_string()),
                api_port: 9337,
                console_port: Some(3131),
                models_count: Some(1),
                pi_command: Some("mesh-llm pi --host 127.0.0.1:9337 --model 'Qwen3.6-35B'".to_string()),
                goose_command: Some(
                    "GOOSE_PROVIDER=openai OPENAI_HOST=http://localhost:9337 OPENAI_API_KEY=mesh GOOSE_MODEL=Qwen3.6-35B goose session"
                        .to_string(),
                ),
            })
            .expect("api render should succeed");
    formatter
        .format(&OutputEvent::PeerJoined {
            peer_id: "peer-1".to_string(),
            label: None,
        })
        .expect("peer render should succeed");
    let dashboard = formatter
        .format(&OutputEvent::PeerJoined {
            peer_id: "peer-2".to_string(),
            label: None,
        })
        .expect("peer render should succeed");

    assert!(dashboard.contains("Running llama.cpp instances"));
    assert!(dashboard.contains("Startup status"));
    assert!(dashboard.contains("Running models"));
    assert!(dashboard.contains("Running webserver"));
    assert!(dashboard.contains("Running API"));
    assert!(dashboard.contains("Mesh events (latest 2)"));
    assert!(dashboard.contains("startup=ready"));
    assert!(dashboard.contains("mesh=ready  api=ready  console=ready"));
    assert!(dashboard.contains("llama-server   starting   port=43683"));
    assert!(dashboard.contains("model=Qwen3.6-35B"));
    assert!(dashboard.contains("ctx=8192"));
    assert!(dashboard.contains("logs=/tmp/llama.log"));
    assert!(dashboard.contains("OpenAI-compatible API   ready   http://localhost:9337"));
    assert!(dashboard.contains("Console   ready   http://localhost:3131"));
    assert!(dashboard.contains("pi:    mesh-llm pi --host 127.0.0.1:9337 --model 'Qwen3.6-35B'"));
    assert!(dashboard.contains("goose: GOOSE_PROVIDER=openai OPENAI_HOST=http://localhost:9337 OPENAI_API_KEY=mesh GOOSE_MODEL=Qwen3.6-35B goose session"));
    assert!(dashboard.contains("peer-1"));
    assert!(dashboard.contains("peer-2"));
    assert!(!dashboard.contains("mesh-llm starting"));
}

#[test]
pub(super) fn dashboard_and_json_formatters_cover_all_output_variants_without_panics() {
    let events = sample_events_covering_all_variants();
    let mut pretty = DashboardFormatter::with_state(DashboardState::with_mesh_event_limit(64));
    let mut json = JsonFormatter;

    for event in &events {
        let dashboard_rendered = pretty
            .format(event)
            .expect("pretty formatter should render every event variant");
        assert!(
            dashboard_rendered.contains("Running llama.cpp instances")
                && dashboard_rendered.contains("Startup status")
                && dashboard_rendered.contains("Running models")
                && dashboard_rendered.contains("Running webserver")
                && dashboard_rendered.contains("Running API")
                && dashboard_rendered.contains("Mesh events"),
            "pretty formatter should keep pinned sections for {event:?}"
        );

        let json_rendered = json
            .format(event)
            .expect("json formatter should render every event variant");
        let value = parse_json_line(&json_rendered);
        assert_required_json_envelope(&value, event);
    }
}

#[test]
pub(super) fn json_formatter_includes_required_fields_for_every_output_variant() {
    let events = sample_events_covering_all_variants();
    let mut formatter = JsonFormatter;

    for event in &events {
        let rendered = formatter
            .format(event)
            .expect("json formatter should render every event variant");
        let value = parse_json_line(&rendered);
        assert_required_json_envelope(&value, event);
    }
}

#[test]
pub(super) fn json_formatter_preserves_representative_optional_metadata_fields() {
    let mut formatter = JsonFormatter;

    let model_ready = format_json_event(
        &mut formatter,
        OutputEvent::ModelReady {
            model: "Qwen3-32B".to_string(),
            internal_port: Some(38373),
            role: Some("host".to_string()),
        },
    );
    assert_model_ready_metadata(&model_ready);

    let rpc_starting = format_json_event(
        &mut formatter,
        OutputEvent::RpcServerStarting {
            port: 43683,
            device: "CUDA0".to_string(),
            log_path: Some("/tmp/rpc.log".to_string()),
        },
    );
    assert_rpc_starting_metadata(&rpc_starting);

    let llama_starting = format_json_event(
        &mut formatter,
        OutputEvent::LlamaStarting {
            model: Some("Qwen3-32B".to_string()),
            http_port: 8001,
            ctx_size: Some(8192),
            log_path: Some("/tmp/llama.log".to_string()),
        },
    );
    assert_llama_starting_metadata(&llama_starting);

    let info = format_json_event(
        &mut formatter,
        OutputEvent::Info {
            message: "joined mesh".to_string(),
            context: Some("mesh=mesh-123".to_string()),
        },
    );
    assert_eq!(info["context"], "mesh=mesh-123");

    let warning = format_json_event(
        &mut formatter,
        OutputEvent::Warning {
            message: "bind warning".to_string(),
            context: Some("model=Qwen3-32B".to_string()),
        },
    );
    assert_eq!(warning["warning"], "bind warning");
    assert_eq!(warning["context"], "model=Qwen3-32B");

    let runtime_ready = format_json_event(
        &mut formatter,
        OutputEvent::RuntimeReady {
            api_url: "http://localhost:9337".to_string(),
            console_url: Some("http://localhost:3131".to_string()),
            api_port: 9337,
            console_port: Some(3131),
            models_count: Some(2),
            pi_command: Some("mesh-llm pi --host 127.0.0.1:9337 --model 'Qwen3-32B'".to_string()),
            goose_command: Some("goose session".to_string()),
        },
    );
    assert_runtime_ready_metadata(&runtime_ready);
}

#[test]
pub(super) fn dashboard_formatter_mesh_history_keeps_timestamps_and_emoji_readable() {
    let mut formatter = DashboardFormatter::with_state(DashboardState::with_mesh_event_limit(8));

    formatter
        .format(&OutputEvent::InviteToken {
            token: "invite-token-1234567890".to_string(),
            mesh_id: "mesh-abc".to_string(),
            mesh_name: None,
        })
        .expect("invite render should succeed");
    formatter
        .format(&OutputEvent::DiscoveryStarting {
            source: "Nostr re-discovery".to_string(),
        })
        .expect("discovery start render should succeed");
    formatter
        .format(&OutputEvent::Warning {
            message: "legacy capacity estimate may be stale".to_string(),
            context: Some("model=Qwen3-32B".to_string()),
        })
        .expect("warning render should succeed");
    let dashboard = formatter
        .format(&OutputEvent::Info {
            message: "waiting for stage readiness".to_string(),
            context: Some("model=Qwen3-32B".to_string()),
        })
        .expect("stage readiness render should succeed");

    let mesh_lines: Vec<&str> = dashboard
        .lines()
        .filter(|line| line.starts_with("│ "))
        .filter(|line| {
            line.contains("Invite created")
                || line.contains("discovering mesh")
                || line.contains("legacy capacity estimate may be stale")
                || line.contains("waiting for stage readiness")
        })
        .collect();

    assert_eq!(
        mesh_lines.len(),
        4,
        "expected four readable mesh history lines"
    );
    for line in &mesh_lines {
        let timestamp: String = line.chars().skip(2).take(8).collect();
        assert_hh_mm_ss(&timestamp);
    }

    assert!(dashboard.contains("Invite created for mesh mesh-abc: invite-token-1234567890"));
    assert!(dashboard.contains("discovering mesh via Nostr re-discovery"));
    assert!(dashboard.contains("model=Qwen3-32B: legacy capacity estimate may be stale"));
    assert!(dashboard.contains("model=Qwen3-32B: waiting for stage readiness"));
    assert!(!dashboard.contains('📡'));
    assert!(!dashboard.contains('🔍'));
    assert!(!dashboard.contains("⚠️"));
}

#[test]
pub(super) fn dashboard_formatter_keeps_long_names_paths_and_tokens_readable() {
    let long_model = "Qwen3.6-35B-A3B-UD-Q4_K_XL-with-extra-routing-suffix";
    let long_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.super.long.mesh.invite.token.payload";
    let long_llama_log = "/Users/ndizazzo/.mesh-llm/runtime/3845607/logs/llama-server-8001-with-a-very-long-name.log";
    let mut formatter = DashboardFormatter::with_state(DashboardState::with_mesh_event_limit(8));

    formatter
        .format(&OutputEvent::InviteToken {
            token: long_token.to_string(),
            mesh_id: "mesh-readable".to_string(),
            mesh_name: None,
        })
        .expect("invite render should succeed");
    formatter
        .format(&OutputEvent::LlamaStarting {
            model: Some(long_model.to_string()),
            http_port: 8001,
            ctx_size: Some(8192),
            log_path: Some(long_llama_log.to_string()),
        })
        .expect("llama render should succeed");
    let dashboard = formatter
        .format(&OutputEvent::ModelReady {
            model: long_model.to_string(),
            internal_port: Some(38373),
            role: Some("host".to_string()),
        })
        .expect("model ready render should succeed");

    assert!(dashboard.contains(long_model));
    assert!(dashboard.contains(long_token));
    assert!(dashboard.contains(long_llama_log));
    assert!(dashboard.contains("Mesh events (latest 8)"));
    assert!(dashboard.contains("│ llama-server   starting   port=8001"));
    assert!(dashboard.contains("model=Qwen3.6-35B-A3B-UD-Q4_K_XL-with-extra-routing-suffix"));
    assert!(dashboard.contains("ctx=8192"));
    assert!(dashboard.contains("│              logs=/Users/ndizazzo/.mesh-llm/runtime/3845607/logs/llama-server-8001-with-a-very-long-name.log"));
    assert!(dashboard.contains(
        "│ Qwen3.6-35B-A3B-UD-Q4_K_XL-with-extra-routing-suffix   ready   port=38373   role=host"
    ));
    assert!(
        dashboard
            .lines()
            .any(|line| line.starts_with("┌ Running llama.cpp instances "))
    );
    assert!(
        dashboard
            .lines()
            .any(|line| line.starts_with("┌ Running models "))
    );
    assert!(
        dashboard
            .lines()
            .any(|line| line.starts_with("┌ Mesh events (latest 8) "))
    );
}

#[test]
pub(super) fn test_select_formatter_for_console_session_mode_none() {
    let formatter = select_formatter(LogFormat::Pretty, ConsoleSessionMode::None);
    assert!(matches!(formatter, FormatterSelection::Plain(_)));
}

#[test]
pub(super) fn test_select_formatter_for_console_session_mode_interactive_dashboard() {
    let formatter = select_formatter(LogFormat::Pretty, ConsoleSessionMode::InteractiveDashboard);
    assert!(matches!(
        formatter,
        FormatterSelection::InteractiveDashboard(_)
    ));
}

#[test]
pub(super) fn test_select_formatter_for_console_session_mode_fallback() {
    let formatter = select_formatter(LogFormat::Pretty, ConsoleSessionMode::Fallback);
    assert!(matches!(
        formatter,
        FormatterSelection::DashboardFallback(_)
    ));
}

#[test]
pub(super) fn test_select_formatter_for_json_mode() {
    let formatter = select_formatter(LogFormat::Json, ConsoleSessionMode::InteractiveDashboard);
    assert!(matches!(formatter, FormatterSelection::Json(_)));
}

#[test]
pub(super) fn test_pretty_formatter_outputs_simple_line() {
    let mut formatter = PrettyFormatter;
    let event = OutputEvent::Info {
        message: "test message".to_string(),
        context: None,
    };
    let result = formatter.format(&event).unwrap();
    assert_eq!(result, "test message\n");
}

#[test]
pub(super) fn llama_native_log_event_name_returns_category() {
    for category in ["backend", "model", "memory", "kv_cache", "tokenizer"] {
        let event = OutputEvent::LlamaNativeLog {
            message: format!("{category} init test"),
            category,
            params: Vec::new(),
        };
        assert_eq!(event.event_name(), category);
    }
}

#[test]
pub(super) fn llama_native_log_message_preserves_content() {
    let msg = "VRAM used: 12.4 GB";
    let event = OutputEvent::LlamaNativeLog {
        message: msg.to_string(),
        category: "memory",
        params: Vec::new(),
    };
    assert_eq!(event.message(), msg);
}

#[test]
pub(super) fn llama_native_log_json_fields_serializes_both() {
    let event = OutputEvent::LlamaNativeLog {
        message: "KV cache type: f16".to_string(),
        category: "kv_cache",
        params: Vec::new(),
    };
    let fields = event.json_fields();
    assert!(fields.get("message").is_none());
    assert!(fields.get("category").is_none());
}

#[test]
pub(super) fn llama_native_log_level_is_info() {
    let event = OutputEvent::LlamaNativeLog {
        message: "backend_init".to_string(),
        category: "backend",
        params: Vec::new(),
    };
    assert_eq!(event.level(), OutputLevel::Debug);
}

#[test]
pub(super) fn llama_native_log_message_renders_structured_params() {
    let event = OutputEvent::LlamaNativeLog {
        message: "Reading model metadata...".to_string(),
        category: "model",
        params: vec![
            (
                "architecture".to_string(),
                Value::String("qwen35".to_string()),
            ),
            ("ctx".to_string(), Value::from(262144_u64)),
        ],
    };
    assert_eq!(event.message(), "Reading model metadata...");
    assert_eq!(
        event.pretty_text(),
        "Reading model metadata...\n  ↳ architecture=qwen35\n  ↳ ctx=262144"
    );
    assert_eq!(event.summary_line(), "Reading model metadata...");
}

#[test]
pub(super) fn llama_native_log_json_fields_include_params() {
    let event = OutputEvent::LlamaNativeLog {
        message: "Reading tensor groups...".to_string(),
        category: "model",
        params: vec![
            ("f32".to_string(), Value::from(177_u64)),
            ("q4_K".to_string(), Value::from(74_u64)),
        ],
    };
    let fields = event.json_fields();
    assert_eq!(fields.get("f32").unwrap().as_u64().unwrap(), 177);
    assert_eq!(fields.get("q4_K").unwrap().as_u64().unwrap(), 74);
}

#[test]
pub(super) fn json_formatter_keeps_llama_native_log_message_concise() {
    let event = OutputEvent::LlamaNativeLog {
        message: "Reading model metadata...".to_string(),
        category: "model",
        params: vec![
            (
                "architecture".to_string(),
                Value::String("qwen35".to_string()),
            ),
            ("ctx".to_string(), Value::from(262144_u64)),
        ],
    };
    let mut formatter = JsonFormatter;
    let rendered = formatter.format(&event).unwrap();
    let record: Value = serde_json::from_str(rendered.trim()).unwrap();
    assert_eq!(
        record.get("message").and_then(Value::as_str).unwrap(),
        "Reading model metadata..."
    );
    assert_eq!(
        record.get("architecture").and_then(Value::as_str).unwrap(),
        "qwen35"
    );
    assert_eq!(record.get("ctx").and_then(Value::as_u64).unwrap(), 262144);
    assert_eq!(
        record.get("event").and_then(Value::as_str).unwrap(),
        "model"
    );
    assert_eq!(
        record.get("level").and_then(Value::as_str).unwrap(),
        "debug"
    );
}

#[test]
pub(super) fn pretty_formatter_renders_llama_native_log_params_on_followup_lines() {
    let event = OutputEvent::LlamaNativeLog {
        message: "Reading tensor groups...".to_string(),
        category: "model",
        params: vec![
            ("f32".to_string(), Value::from(177_u64)),
            ("q4_K".to_string(), Value::from(74_u64)),
        ],
    };
    let mut formatter = PrettyFormatter;
    let rendered = formatter.format(&event).unwrap();
    assert_eq!(
        rendered,
        "Reading tensor groups...\n  ↳ f32=177\n  ↳ q4_K=74\n"
    );
}

#[test]
pub(super) fn shutdown_requested_event_name_returns_signal() {
    for signal in ["SIGINT", "SIGTERM", "CTRL-C", "api"] {
        let event = OutputEvent::ShutdownRequested { signal };
        assert_eq!(event.event_name(), signal);
    }
}

#[test]
pub(super) fn shutdown_requested_message_includes_signal_type() {
    for signal in ["SIGINT", "SIGTERM", "CTRL-C", "api"] {
        let event = OutputEvent::ShutdownRequested { signal };
        assert!(
            event.message().contains(signal),
            "message should contain signal: {}",
            event.message()
        );
    }
}

#[test]
pub(super) fn shutdown_requested_json_fields_serializes_signal() {
    for signal in ["SIGINT", "SIGTERM"] {
        let event = OutputEvent::ShutdownRequested { signal };
        let fields = event.json_fields();
        assert_eq!(fields.get("signal").unwrap().as_str().unwrap(), signal);
    }
}

#[test]
pub(super) fn model_unloading_event_serialization() {
    let event = OutputEvent::ModelUnloading {
        model: "Qwen3-32B".to_string(),
    };
    assert_eq!(event.event_name(), "model_unloading");
    assert!(event.message().contains("Qwen3-32B"));
    let fields = event.json_fields();
    assert_eq!(fields.get("model").unwrap().as_str().unwrap(), "Qwen3-32B");
}

#[test]
pub(super) fn model_unloaded_event_serialization() {
    let event = OutputEvent::ModelUnloaded {
        model: "Llama-3.1-8B".to_string(),
    };
    assert_eq!(event.event_name(), "model_unloaded");
    assert!(event.message().contains("Llama-3.1-8B"));
    let fields = event.json_fields();
    assert_eq!(
        fields.get("model").unwrap().as_str().unwrap(),
        "Llama-3.1-8B"
    );
}

#[test]
pub(super) fn model_lifecycle_events_have_consistent_model_names() {
    let name = "Mistral-Nemo-12B".to_string();

    for event in [
        OutputEvent::ModelLoading {
            model: name.clone(),
            source: None,
        },
        OutputEvent::ModelLoaded {
            model: name.clone(),
            bytes: Some(8_000_000_000),
        },
        OutputEvent::ModelUnloading {
            model: name.clone(),
        },
        OutputEvent::ModelUnloaded {
            model: name.clone(),
        },
    ] {
        assert!(
            event.message().contains("Mistral-Nemo-12B"),
            "event {} message should contain model name: {}",
            event.event_name(),
            event.message()
        );
        let fields = event.json_fields();
        assert_eq!(
            fields.get("model").unwrap().as_str().unwrap(),
            "Mistral-Nemo-12B",
            "json_fields for {} should have correct model",
            event.event_name()
        );
    }
}
#[test]
pub(super) fn tui_terminal_setup_marks_cleanup_required_after_enter_escape() {
    let mut formatter = InteractiveDashboardFormatter::default();

    formatter.mark_terminal_escape_written();

    assert!(formatter.terminal_active);
    assert!(formatter.tui_entered());
    assert!(formatter.dirty);
    assert!(formatter.terminal.is_none());
}

#[test]
pub(super) fn tui_panic_restore_flag_tracks_terminal_entry() {
    let mut formatter = InteractiveDashboardFormatter::default();

    assert!(!formatter.tui_entered());
    formatter.mark_terminal_escape_written();
    assert!(formatter.tui_entered());
    formatter.exit_terminal().expect("exit should succeed");
    assert!(!formatter.tui_entered());
}

#[test]
pub(super) fn tui_panic_restore_disables_interactive_redraws() {
    let tui_entered = Arc::new(AtomicBool::new(false));
    let panic_restored = Arc::new(AtomicBool::new(false));
    let mut formatter =
        InteractiveDashboardFormatter::with_tui_state(tui_entered.clone(), panic_restored.clone());
    formatter.mark_terminal_escape_written();

    formatter.mark_panic_restored();

    assert!(!formatter.terminal_active);
    assert!(!formatter.dirty);
    assert!(!tui_entered.load(Ordering::Acquire));
    assert!(panic_restored.load(Ordering::Acquire));
    assert_eq!(
        formatter
            .handle_output_event(&OutputEvent::Shutdown { reason: None })
            .expect("panic-restored formatter should ignore output events"),
        None
    );
    assert_eq!(
        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Char('q'))),
        TuiControlFlow::Continue
    );
    assert!(
        !formatter
            .render_if_dirty()
            .expect("panic-restored formatter should skip redraws")
    );
}

#[test]
pub(super) fn tui_restores_terminal_state_on_exit() {
    let mut output = Vec::new();

    write_tui_enter_to_writer(&mut output).expect("enter should succeed");
    write_tui_frame_to_writer(&mut output, "dashboard").expect("frame render should succeed");
    write_tui_exit_to_writer(&mut output).expect("exit should succeed");

    let rendered = String::from_utf8(output).expect("terminal output should be utf8");
    let leave_index = rendered
        .rfind("[?1049l")
        .expect("expected leave-alternate-screen sequence in exit output");
    let clear_index = rendered
        .rfind("[2J")
        .expect("expected full-screen clear in exit output");

    assert!(rendered.contains("dashboard"));
    assert!(rendered.contains('\u{1b}'));
    assert!(
        clear_index > leave_index,
        "expected final clear after leaving alternate screen in {rendered:?}"
    );
    assert!(rendered.matches('\u{1b}').count() >= 6);
}

#[test]
pub(super) fn tui_enter_does_not_enable_mouse_capture() {
    let mut output = Vec::new();

    write_tui_enter_to_writer(&mut output).expect("enter should succeed");
    write_tui_exit_to_writer(&mut output).expect("exit should succeed");

    let rendered = String::from_utf8(output).expect("terminal output should be utf8");
    for sequence in ["[?1000h", "[?1002h", "[?1003h", "[?1006h"] {
        assert!(
            !rendered.contains(sequence),
            "TUI should leave native terminal text selection available: {rendered:?}"
        );
    }
}

#[test]
pub(super) fn tui_redraw_start_repositions_without_physical_clear() {
    let mut output = Vec::new();

    write_tui_redraw_start_to_writer(&mut output).expect("redraw start should succeed");

    let rendered = String::from_utf8(output).expect("terminal output should be utf8");
    assert!(
        rendered.contains("[?25l"),
        "redraw start should hide the cursor before repainting: {rendered:?}"
    );
    assert!(
        rendered.contains("[H") || rendered.contains("[1;1H"),
        "redraw start should move to the top-left before repainting: {rendered:?}"
    );
    assert!(
        !rendered.contains("[2J"),
        "redraw start should avoid a physical full-screen clear that flickers between frames: {rendered:?}"
    );
}

#[test]
pub(super) fn tui_handles_resize_without_resetting_focus() {
    let mut formatter = InteractiveDashboardFormatter::default();
    formatter.handle_snapshot(snapshot_fixture(12, 30));

    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Tab));
    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Tab));
    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Tab));
    assert_eq!(formatter.state.panel_focus, DashboardPanel::Models);

    formatter.handle_tui_event(TuiEvent::Resize {
        columns: 120,
        rows: 36,
    });

    assert_eq!(formatter.state.panel_focus, DashboardPanel::Models);
}
