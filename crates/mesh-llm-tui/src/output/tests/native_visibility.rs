use super::*;

fn structured_model_load_sequence() -> Vec<OutputEvent> {
    vec![
        OutputEvent::Startup {
            version: "v0.68.0".to_string(),
            message: None,
        },
        OutputEvent::LaunchPlan {
            plan: sample_launch_plan(),
        },
        OutputEvent::ModelQueued {
            model: "Planned-Model".to_string(),
        },
        OutputEvent::ModelLoading {
            model: "Planned-Model".to_string(),
            source: None,
        },
        OutputEvent::Info {
            message: "Opening model 'Planned-Model' 50%".to_string(),
            context: Some("sequence=2 status=Ok emitter=OpenThread".to_string()),
        },
        OutputEvent::ModelLoaded {
            model: "Planned-Model".to_string(),
            bytes: Some(4_000_000_000),
        },
        OutputEvent::LlamaReady {
            model: Some("Planned-Model".to_string()),
            port: 9338,
            ctx_size: Some(8192),
            log_path: None,
        },
        OutputEvent::ModelReady {
            model: "Planned-Model".to_string(),
            internal_port: Some(9338),
            role: Some("host".to_string()),
        },
        OutputEvent::WebserverReady {
            url: "http://localhost:3131".to_string(),
        },
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
    ]
}

fn parsed_native_log_noise() -> Vec<OutputEvent> {
    [
        ("backend", "llama_backend_init: GGML_CUDA"),
        (
            "model",
            "model load plan: metadata rows=10, tensor rows=100",
        ),
        ("model", "tensors 50% (50/100 tensors)"),
        (
            "memory",
            "load_tensors: CUDA0 model buffer size = 4321.00 MiB",
        ),
        ("kv_cache", "kv cache 100% (8/8 layers)"),
        (
            "tokenizer",
            "init_tokenizer: initializing tokenizer for type 2",
        ),
        ("model", "mesh-llm: skippy_model_open finished"),
    ]
    .into_iter()
    .map(|(category, message)| OutputEvent::LlamaNativeLog {
        message: message.to_string(),
        category,
        params: vec![("CUDA0".to_string(), Value::from(36_u64))],
    })
    .collect()
}

#[derive(Debug, PartialEq)]
struct LifecycleProjection {
    startup_lifecycle: StartupLifecycleState,
    startup_progress: Option<StartupProgressState>,
    startup_milestones: std::collections::BTreeSet<String>,
    startup_history: Vec<String>,
    loading_progress: Option<LoadingProgressState>,
    model_progress: Option<ModelProgressState>,
    llama_process_rows: Vec<DashboardProcessRow>,
    webserver_rows: Vec<DashboardEndpointRow>,
    loaded_model_rows: Vec<DashboardModelRow>,
    runtime_ready: bool,
}

fn lifecycle_after(events: impl IntoIterator<Item = OutputEvent>) -> LifecycleProjection {
    let mut state = DashboardState::default();
    for event in events {
        state.reduce(DashboardAction::OutputEvent(event));
    }
    LifecycleProjection {
        startup_lifecycle: state.startup_lifecycle().clone(),
        startup_progress: state.startup_progress.clone(),
        startup_milestones: state.startup_milestones.clone(),
        startup_history: state
            .startup_history
            .iter()
            .map(|entry| entry.summary.clone())
            .collect(),
        loading_progress: state.active_loading_progress(),
        model_progress: state.model_progress.clone(),
        llama_process_rows: state.llama_process_rows.clone(),
        webserver_rows: state.webserver_rows.clone(),
        loaded_model_rows: state.loaded_model_rows.clone(),
        runtime_ready: state.runtime_ready,
    }
}

#[test]
fn structured_model_load_lifecycle_is_identical_with_and_without_native_logs() {
    let structured = structured_model_load_sequence();
    let noise = parsed_native_log_noise();
    let interleaved = structured.iter().enumerate().flat_map(|(index, event)| {
        std::iter::once(event.clone()).chain(noise.iter().skip(index % noise.len()).cloned())
    });

    let without_native_logs = lifecycle_after(structured.clone());
    let with_native_logs = lifecycle_after(interleaved);

    assert!(without_native_logs.runtime_ready);
    assert_eq!(with_native_logs, without_native_logs);
}

#[test]
fn partial_structured_model_load_is_identical_with_and_without_native_logs() {
    let structured: Vec<_> = structured_model_load_sequence()
        .into_iter()
        .take(5)
        .collect();
    let noise = parsed_native_log_noise();
    let interleaved = structured
        .iter()
        .flat_map(|event| std::iter::once(event.clone()).chain(noise.iter().cloned()));

    let without_native_logs = lifecycle_after(structured.clone());
    let with_native_logs = lifecycle_after(interleaved);

    assert!(!without_native_logs.runtime_ready);
    assert_eq!(with_native_logs, without_native_logs);
}

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
