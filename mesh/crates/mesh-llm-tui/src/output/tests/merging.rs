use super::*;

#[test]
pub(super) fn launch_plan_rows_survive_empty_startup_snapshot() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::OutputEvent(OutputEvent::LaunchPlan {
        plan: sample_launch_plan(),
    }));

    state.reduce(DashboardAction::SnapshotUpdated(
        DashboardSnapshot::default(),
    ));

    assert_eq!(state.llama_process_rows.len(), 1);
    assert_eq!(state.webserver_rows.len(), 2);
    assert_eq!(state.loaded_model_rows.len(), 1);
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
    state.reduce(DashboardAction::SnapshotUpdated(
        DashboardSnapshot::default(),
    ));
    assert_eq!(
        state
            .llama_process_rows
            .iter()
            .find(|row| row.name == "llama-server")
            .expect("expected planned llama row")
            .status,
        RuntimeStatus::Loading
    );
}

#[test]
pub(super) fn launch_plan_preserves_distinct_port_zero_endpoint_rows() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::OutputEvent(OutputEvent::LaunchPlan {
        plan: port_zero_endpoint_launch_plan(),
    }));

    let rows = state
        .webserver_rows
        .iter()
        .map(|row| (row.label.clone(), row.port, row.status.clone()))
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), 3);
    assert_eq!(
        rows,
        vec![
            ("Plugin: alpha".to_string(), 0, RuntimeStatus::NotReady),
            ("Plugin: beta".to_string(), 0, RuntimeStatus::NotReady),
            ("Plugin: zebra".to_string(), 0, RuntimeStatus::NotReady),
        ]
    );
}

#[test]
pub(super) fn snapshot_upsert_preserves_distinct_port_zero_endpoint_rows() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::OutputEvent(OutputEvent::LaunchPlan {
        plan: port_zero_endpoint_launch_plan(),
    }));

    state.reduce(DashboardAction::SnapshotUpdated(DashboardSnapshot {
        webserver_rows: vec![
            DashboardEndpointRow {
                label: "Plugin: alpha".to_string(),
                status: RuntimeStatus::Ready,
                url: "alpha-plugin-live".to_string(),
                port: 0,
                pid: Some(2000),
            },
            DashboardEndpointRow {
                label: "Plugin: zebra".to_string(),
                status: RuntimeStatus::Warning,
                url: "zebra-plugin-live".to_string(),
                port: 0,
                pid: Some(2001),
            },
        ],
        ..DashboardSnapshot::default()
    }));

    assert_eq!(state.webserver_rows.len(), 3);
    assert_eq!(
        state
            .webserver_rows
            .iter()
            .find(|row| row.label == "Plugin: beta")
            .expect("expected beta plugin placeholder row")
            .status,
        RuntimeStatus::NotReady
    );
    assert_eq!(
        state
            .webserver_rows
            .iter()
            .find(|row| row.label == "Plugin: alpha")
            .expect("expected alpha plugin row"),
        &DashboardEndpointRow {
            label: "Plugin: alpha".to_string(),
            status: RuntimeStatus::Ready,
            url: "alpha-plugin-live".to_string(),
            port: 0,
            pid: Some(2000),
        }
    );
    assert_eq!(
        state
            .webserver_rows
            .iter()
            .find(|row| row.label == "Plugin: zebra")
            .expect("expected zebra plugin row"),
        &DashboardEndpointRow {
            label: "Plugin: zebra".to_string(),
            status: RuntimeStatus::Warning,
            url: "zebra-plugin-live".to_string(),
            port: 0,
            pid: Some(2001),
        }
    );
}

#[test]
pub(super) fn planned_port_zero_process_rows_bind_to_concrete_startup_events() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::OutputEvent(OutputEvent::LaunchPlan {
        plan: DashboardLaunchPlan {
            llama_process_rows: vec![
                DashboardProcessRow {
                    name: "llama-server Model-A".to_string(),
                    backend: String::new(),
                    status: RuntimeStatus::Loading,
                    port: 0,
                    pid: 0,
                },
                DashboardProcessRow {
                    name: "llama-server Model-B".to_string(),
                    backend: String::new(),
                    status: RuntimeStatus::Loading,
                    port: 0,
                    pid: 0,
                },
            ],
            webserver_rows: Vec::new(),
            loaded_model_rows: Vec::new(),
        },
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::LlamaStarting {
        model: Some("Model-B".to_string()),
        http_port: 9339,
        ctx_size: Some(4096),
        log_path: None,
    }));

    assert_eq!(state.llama_process_rows.len(), 2);
    assert!(state.webserver_rows.is_empty());
    assert_eq!(
        state
            .llama_process_rows
            .iter()
            .find(|row| row.name == "llama-server Model-A")
            .expect("unstarted planned llama row should remain visible")
            .status,
        RuntimeStatus::Loading
    );
    assert_eq!(
        state
            .llama_process_rows
            .iter()
            .find(|row| row.name == "llama-server Model-B")
            .expect("planned llama row should bind to concrete model startup event"),
        &DashboardProcessRow {
            name: "llama-server Model-B".to_string(),
            backend: String::new(),
            status: RuntimeStatus::Starting,
            port: 9339,
            pid: 0,
        }
    );
}

#[test]
pub(super) fn ready_llama_process_row_stays_ready_when_another_model_starts() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::OutputEvent(OutputEvent::Startup {
        version: "v0.68.0".to_string(),
        message: Some("starting multi-model runtime".to_string()),
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::LlamaStarting {
        model: Some("Model-A".to_string()),
        http_port: 9338,
        ctx_size: Some(4096),
        log_path: None,
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::LlamaReady {
        model: Some("Model-A".to_string()),
        port: 9338,
        ctx_size: Some(4096),
        log_path: None,
    }));

    assert_eq!(
        state
            .llama_process_rows
            .iter()
            .find(|row| row.name == "llama-server Model-A")
            .expect("Model-A row should be present after ready event")
            .status,
        RuntimeStatus::Ready
    );

    state.reduce(DashboardAction::OutputEvent(OutputEvent::LlamaStarting {
        model: Some("Model-B".to_string()),
        http_port: 9339,
        ctx_size: Some(4096),
        log_path: None,
    }));

    assert_eq!(state.llama_process_rows.len(), 2);
    assert_eq!(
        state
            .llama_process_rows
            .iter()
            .find(|row| row.name == "llama-server Model-A")
            .expect("ready Model-A row should remain present")
            .status,
        RuntimeStatus::Ready
    );
    assert_eq!(
        state
            .llama_process_rows
            .iter()
            .find(|row| row.name == "llama-server Model-B")
            .expect("starting Model-B row should be present")
            .status,
        RuntimeStatus::Starting
    );
}

#[test]
pub(super) fn ready_llama_process_row_survives_lagging_startup_snapshot() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::OutputEvent(OutputEvent::Startup {
        version: "v0.68.0".to_string(),
        message: Some("starting multi-model runtime".to_string()),
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::LlamaStarting {
        model: Some("Model-A".to_string()),
        http_port: 9338,
        ctx_size: Some(4096),
        log_path: None,
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::LlamaReady {
        model: Some("Model-A".to_string()),
        port: 9338,
        ctx_size: Some(4096),
        log_path: None,
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::LlamaStarting {
        model: Some("Model-B".to_string()),
        http_port: 9339,
        ctx_size: Some(4096),
        log_path: None,
    }));

    state.reduce(DashboardAction::SnapshotUpdated(DashboardSnapshot {
        llama_process_rows: vec![DashboardProcessRow {
            name: "llama-server Model-A".to_string(),
            backend: String::new(),
            status: RuntimeStatus::Starting,
            port: 9338,
            pid: 0,
        }],
        ..DashboardSnapshot::default()
    }));

    assert_eq!(
        state
            .llama_process_rows
            .iter()
            .find(|row| row.name == "llama-server Model-A")
            .expect("ready Model-A row should survive lagging snapshot")
            .status,
        RuntimeStatus::Ready
    );
}

#[test]
pub(super) fn model_loading_row_reconciles_with_canonical_ready_name() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::OutputEvent(OutputEvent::ModelLoading {
        model: "Qwen3.5-4B-UD-Q4_K_XL".to_string(),
        source: None,
    }));

    assert_eq!(state.loaded_model_rows.len(), 1);
    assert_eq!(state.loaded_model_rows[0].name, "Qwen3.5-4B-UD-Q4_K_XL");
    assert_eq!(state.loaded_model_rows[0].status, RuntimeStatus::Loading);
    assert_eq!(state.llama_process_rows.len(), 1);
    assert_eq!(
        state.llama_process_rows[0].name,
        "llama-server Qwen3.5-4B-UD-Q4_K_XL"
    );
    assert_eq!(state.llama_process_rows[0].status, RuntimeStatus::Loading);

    state.reduce(DashboardAction::OutputEvent(OutputEvent::ModelReady {
        model: "unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL".to_string(),
        internal_port: Some(9338),
        role: Some("host".to_string()),
    }));

    assert_eq!(state.loaded_model_rows.len(), 1);
    let row = &state.loaded_model_rows[0];
    assert_eq!(row.name, "unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL");
    assert_eq!(row.status, RuntimeStatus::Ready);
    assert_eq!(row.port, Some(9338));
    assert_eq!(row.role.as_deref(), Some("host"));
}

#[test]
pub(super) fn planned_process_row_reconciles_with_canonical_loading_name() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::OutputEvent(OutputEvent::LaunchPlan {
        plan: DashboardLaunchPlan {
            llama_process_rows: vec![DashboardProcessRow {
                name: "llama-server Qwen3.5-4B-UD-Q4_K_XL".to_string(),
                backend: String::new(),
                status: RuntimeStatus::Loading,
                port: 0,
                pid: 0,
            }],
            webserver_rows: Vec::new(),
            loaded_model_rows: Vec::new(),
        },
    }));

    state.reduce(DashboardAction::OutputEvent(OutputEvent::ModelLoading {
        model: "unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL".to_string(),
        source: None,
    }));

    assert_eq!(state.llama_process_rows.len(), 1);
    let row = &state.llama_process_rows[0];
    assert_eq!(row.name, "llama-server unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL");
    assert_eq!(row.status, RuntimeStatus::Loading);
    assert_eq!(row.port, 0);

    state.reduce(DashboardAction::SnapshotUpdated(DashboardSnapshot {
        llama_process_rows: vec![DashboardProcessRow {
            name: "llama-server Qwen3.5-4B-UD-Q4_K_XL".to_string(),
            backend: String::new(),
            status: RuntimeStatus::NotReady,
            port: 0,
            pid: 0,
        }],
        ..DashboardSnapshot::default()
    }));

    assert_eq!(state.llama_process_rows.len(), 1);
    let row = &state.llama_process_rows[0];
    assert_eq!(row.name, "llama-server unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL");
    assert_eq!(row.status, RuntimeStatus::Loading);
    assert_eq!(row.port, 0);
}

#[test]
pub(super) fn raw_snapshot_ready_row_reconciles_with_canonical_loading_row() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::OutputEvent(OutputEvent::Startup {
        version: "v0.68.0".to_string(),
        message: None,
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::LaunchPlan {
        plan: DashboardLaunchPlan {
            llama_process_rows: [
                "unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL",
                "unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL",
            ]
            .into_iter()
            .map(|model| DashboardProcessRow {
                name: llama_process_row_name(Some(model)),
                backend: String::new(),
                status: RuntimeStatus::Loading,
                port: 0,
                pid: 0,
            })
            .collect(),
            webserver_rows: Vec::new(),
            loaded_model_rows: [
                "unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL",
                "unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL",
            ]
            .into_iter()
            .map(|model| DashboardModelRow {
                name: model.to_string(),
                role: None,
                status: RuntimeStatus::Loading,
                port: None,
                device: None,
                slots: None,
                quantization: None,
                ctx_size: None,
                ctx_used_tokens: None,
                lanes: None,
                file_size_gb: None,
            })
            .collect(),
        },
    }));

    state.reduce(DashboardAction::SnapshotUpdated(DashboardSnapshot {
        llama_process_rows: vec![DashboardProcessRow {
            name: "unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL".to_string(),
            backend: String::new(),
            status: RuntimeStatus::Ready,
            port: 36561,
            pid: 1221,
        }],
        loaded_model_rows: vec![DashboardModelRow {
            name: "unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL".to_string(),
            role: Some("host".to_string()),
            status: RuntimeStatus::Ready,
            port: Some(36561),
            device: None,
            slots: None,
            quantization: None,
            ctx_size: None,
            ctx_used_tokens: None,
            lanes: None,
            file_size_gb: None,
        }],
        ..DashboardSnapshot::default()
    }));

    assert_eq!(state.llama_process_rows.len(), 2);
    let qwen_35 = state
        .llama_process_rows
        .iter()
        .find(|row| row.name.contains("Qwen3.5-4B"))
        .expect("expected 4B loading row");
    assert_eq!(qwen_35.status, RuntimeStatus::Loading);
    assert_eq!(qwen_35.port, 0);

    let qwen_36 = state
        .llama_process_rows
        .iter()
        .find(|row| row.name.contains("Qwen3.6-27B"))
        .expect("expected 27B ready row");
    assert_eq!(qwen_36.name, "unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL");
    assert_eq!(qwen_36.status, RuntimeStatus::Ready);
    assert_eq!(qwen_36.port, 36561);
    assert_eq!(qwen_36.pid, 1221);
}

#[test]
pub(super) fn single_model_local_path_loading_row_merges_with_ready_model_ref() {
    let mut state = DashboardState::default();
    let loading_name = "Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m";
    let ready_name = "Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q4_k_m";

    state.reduce(DashboardAction::OutputEvent(OutputEvent::LaunchPlan {
        plan: DashboardLaunchPlan {
            llama_process_rows: Vec::new(),
            webserver_rows: Vec::new(),
            loaded_model_rows: vec![DashboardModelRow {
                name: loading_name.to_string(),
                role: Some("primary".to_string()),
                status: RuntimeStatus::Loading,
                port: None,
                device: None,
                slots: None,
                quantization: None,
                ctx_size: None,
                ctx_used_tokens: None,
                lanes: None,
                file_size_gb: None,
            }],
        },
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::ModelReady {
        model: ready_name.to_string(),
        internal_port: Some(51744),
        role: Some("host".to_string()),
    }));

    assert_eq!(state.loaded_model_rows.len(), 1);
    let row = &state.loaded_model_rows[0];
    assert_eq!(row.name, ready_name);
    assert_eq!(row.status, RuntimeStatus::Ready);
    assert_eq!(row.port, Some(51744));
    assert_eq!(row.role.as_deref(), Some("host"));
}

#[test]
pub(super) fn loaded_model_row_preserves_launch_plan_device_when_ready_snapshot_reports_backend() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::OutputEvent(OutputEvent::LaunchPlan {
        plan: DashboardLaunchPlan {
            llama_process_rows: Vec::new(),
            webserver_rows: Vec::new(),
            loaded_model_rows: vec![DashboardModelRow {
                name: "unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL".to_string(),
                role: Some("primary".to_string()),
                status: RuntimeStatus::Loading,
                port: None,
                device: Some("CUDA0".to_string()),
                slots: Some(4),
                quantization: None,
                ctx_size: Some(65_536),
                ctx_used_tokens: None,
                lanes: None,
                file_size_gb: None,
            }],
        },
    }));

    state.reduce(DashboardAction::SnapshotUpdated(DashboardSnapshot {
        loaded_model_rows: vec![DashboardModelRow {
            name: "unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL".to_string(),
            role: Some("host".to_string()),
            status: RuntimeStatus::Ready,
            port: Some(40511),
            device: Some("skippy".to_string()),
            slots: Some(4),
            quantization: Some("Q4_K_XL".to_string()),
            ctx_size: Some(65_536),
            ctx_used_tokens: None,
            lanes: None,
            file_size_gb: Some(2.9),
        }],
        ..DashboardSnapshot::default()
    }));

    assert_eq!(state.loaded_model_rows.len(), 1);
    let row = &state.loaded_model_rows[0];
    assert_eq!(row.name, "unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL");
    assert_eq!(row.device.as_deref(), Some("CUDA0"));
    assert_eq!(row.status, RuntimeStatus::Ready);
    assert_eq!(row.port, Some(40511));
    assert_eq!(row.quantization.as_deref(), Some("Q4_K_XL"));
}

#[test]
pub(super) fn runtime_ready_snapshot_preserves_launch_plan_device_metadata() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::OutputEvent(OutputEvent::LaunchPlan {
        plan: DashboardLaunchPlan {
            llama_process_rows: Vec::new(),
            webserver_rows: Vec::new(),
            loaded_model_rows: vec![DashboardModelRow {
                name: "unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL".to_string(),
                role: Some("primary".to_string()),
                status: RuntimeStatus::Loading,
                port: None,
                device: Some("CUDA0".to_string()),
                slots: Some(4),
                quantization: None,
                ctx_size: Some(65_536),
                ctx_used_tokens: None,
                lanes: None,
                file_size_gb: None,
            }],
        },
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::RuntimeReady {
        api_url: "http://localhost:40511".to_string(),
        console_url: None,
        api_port: 40511,
        console_port: None,
        models_count: Some(1),
        pi_command: None,
        goose_command: None,
    }));

    state.reduce(DashboardAction::SnapshotUpdated(DashboardSnapshot {
        loaded_model_rows: vec![DashboardModelRow {
            name: "unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL".to_string(),
            role: Some("host".to_string()),
            status: RuntimeStatus::Ready,
            port: Some(40511),
            device: None,
            slots: Some(4),
            quantization: Some("Q4_K_XL".to_string()),
            ctx_size: None,
            ctx_used_tokens: None,
            lanes: None,
            file_size_gb: Some(2.9),
        }],
        ..DashboardSnapshot::default()
    }));

    assert_eq!(state.loaded_model_rows.len(), 1);
    let row = &state.loaded_model_rows[0];
    assert_eq!(row.device.as_deref(), Some("CUDA0"));
    assert_eq!(row.status, RuntimeStatus::Ready);
    assert_eq!(row.port, Some(40511));
    assert_eq!(row.quantization.as_deref(), Some("Q4_K_XL"));
    assert_eq!(row.file_size_gb, Some(2.9));
}

#[test]
pub(super) fn runtime_ready_process_only_snapshot_preserves_loaded_model_device_metadata() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::OutputEvent(OutputEvent::LaunchPlan {
        plan: DashboardLaunchPlan {
            llama_process_rows: Vec::new(),
            webserver_rows: Vec::new(),
            loaded_model_rows: vec![DashboardModelRow {
                name: "unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL".to_string(),
                role: Some("model".to_string()),
                status: RuntimeStatus::Loading,
                port: None,
                device: Some("CUDA1".to_string()),
                slots: Some(4),
                quantization: None,
                ctx_size: Some(65_536),
                ctx_used_tokens: None,
                lanes: None,
                file_size_gb: None,
            }],
        },
    }));
    state.reduce(DashboardAction::OutputEvent(OutputEvent::RuntimeReady {
        api_url: "http://localhost:40511".to_string(),
        console_url: None,
        api_port: 40511,
        console_port: None,
        models_count: Some(1),
        pi_command: None,
        goose_command: None,
    }));

    state.reduce(DashboardAction::SnapshotUpdated(DashboardSnapshot {
        llama_process_rows: vec![DashboardProcessRow {
            name: "unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL".to_string(),
            backend: String::new(),
            status: RuntimeStatus::Ready,
            port: 45145,
            pid: 132098,
        }],
        loaded_model_rows: Vec::new(),
        ..DashboardSnapshot::default()
    }));

    assert_eq!(state.loaded_model_rows.len(), 1);
    let row = &state.loaded_model_rows[0];
    assert_eq!(row.name, "unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL");
    assert_eq!(row.device.as_deref(), Some("CUDA1"));
    assert_eq!(row.status, RuntimeStatus::Loading);
}

#[test]
pub(super) fn planned_process_row_reconciles_with_canonical_ready_name() {
    let mut state = DashboardState::default();
    state.reduce(DashboardAction::OutputEvent(OutputEvent::LaunchPlan {
        plan: DashboardLaunchPlan {
            llama_process_rows: vec![DashboardProcessRow {
                name: "llama-server Qwen3.5-4B-UD-Q4_K_XL".to_string(),
                backend: String::new(),
                status: RuntimeStatus::Loading,
                port: 0,
                pid: 0,
            }],
            webserver_rows: Vec::new(),
            loaded_model_rows: Vec::new(),
        },
    }));

    state.reduce(DashboardAction::OutputEvent(OutputEvent::LlamaReady {
        model: Some("unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL".to_string()),
        port: 9338,
        ctx_size: Some(8192),
        log_path: None,
    }));

    assert_eq!(state.llama_process_rows.len(), 1);
    let row = &state.llama_process_rows[0];
    assert_eq!(row.name, "llama-server unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL");
    assert_eq!(row.status, RuntimeStatus::Ready);
    assert_eq!(row.port, 9338);
}
