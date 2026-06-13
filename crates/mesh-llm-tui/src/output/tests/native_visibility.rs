use super::*;

#[test]
fn llama_native_log_does_not_affect_dashboard_state() {
    let mut state = DashboardState::default();

    state.reduce(DashboardAction::OutputEvent(OutputEvent::Startup {
        version: "v0.68.0".to_string(),
        message: None,
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::LaunchPlan {
        plan: sample_launch_plan(),
    }));
    let phase_before_native_logs = state.startup_lifecycle().phase.clone();

    for (category, msg) in [
        ("backend", "backend_init succeeded"),
        ("model", "loading model from disk"),
        ("memory", "VRAM used: 12 GB"),
        ("kv_cache", "KV cache type: f16"),
        ("tokenizer", "vocab loaded: 32000 tokens"),
    ] {
        state.reduce(DashboardAction::OutputEvent(OutputEvent::LlamaNativeLog {
            message: msg.to_string(),
            category,
            params: Vec::new(),
        }));
    }

    assert_eq!(
        state.startup_lifecycle().phase,
        phase_before_native_logs,
        "LlamaNativeLog events should not change startup lifecycle phase"
    );
    assert_eq!(state.llama_process_rows[0].status, RuntimeStatus::Loading);
    assert!(
        state
            .webserver_rows
            .iter()
            .all(|row| row.status == RuntimeStatus::NotReady)
    );
    assert_eq!(state.loaded_model_rows[0].status, RuntimeStatus::Loading);
}

#[test]
fn typed_native_visibility_events_do_not_replace_rust_owned_startup_edges() {
    let mut state = DashboardState::default();

    state.reduce(DashboardAction::OutputEvent(OutputEvent::Startup {
        version: "v0.68.0".to_string(),
        message: None,
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::LaunchPlan {
        plan: sample_launch_plan(),
    }));

    for event in [
        OutputEvent::Info {
            message: "Native runtime started opening model 'Planned-Model'".to_string(),
            context: Some("sequence=1 status=Ok emitter=OpenThread".to_string()),
        },
        OutputEvent::Info {
            message: "Opening model 'Planned-Model' 50%".to_string(),
            context: Some("sequence=2 status=Ok emitter=OpenThread".to_string()),
        },
        OutputEvent::Info {
            message: "Native runtime finished opening model 'Planned-Model'; waiting for Rust runtime readiness".to_string(),
            context: Some("sequence=3 status=Ok emitter=OpenThread".to_string()),
        },
        OutputEvent::Warning {
            message: "Native runtime reported a handled model-open failure for 'Planned-Model'"
                .to_string(),
            context: Some(
                "sequence=4 status=Err emitter=OpenThread detail=simulated native error"
                    .to_string(),
            ),
        },
    ] {
        state.reduce(DashboardAction::OutputEvent(event));
    }

    assert_eq!(state.llama_process_rows[0].status, RuntimeStatus::Loading);
    assert!(
        state
            .webserver_rows
            .iter()
            .all(|row| row.status == RuntimeStatus::NotReady)
    );
    assert_eq!(state.loaded_model_rows[0].status, RuntimeStatus::Loading);
    assert!(!state.runtime_ready);

    state.reduce(DashboardAction::OutputEvent(OutputEvent::WebserverReady {
        url: "http://localhost:3131".to_string(),
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::ApiReady {
        url: "http://localhost:9337".to_string(),
    }));

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
    assert_eq!(state.loaded_model_rows[0].status, RuntimeStatus::Loading);
    assert!(!state.runtime_ready);

    state.reduce(DashboardAction::OutputEvent(OutputEvent::ModelReady {
        model: "Planned-Model".to_string(),
        internal_port: Some(9338),
        role: Some("host".to_string()),
    }));

    assert_eq!(state.loaded_model_rows[0].status, RuntimeStatus::Ready);
    assert!(
        !state.runtime_ready,
        "ModelReady must not replace RuntimeReady"
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
}
