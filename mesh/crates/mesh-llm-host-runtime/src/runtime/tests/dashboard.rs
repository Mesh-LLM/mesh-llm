use super::*;

#[test]
fn plugin_dashboard_command_name_trims_base_path() {
    let summary = plugin::PluginSummary {
        name: "browser".to_string(),
        kind: "stdio".to_string(),
        enabled: true,
        status: "running".to_string(),
        pid: Some(4242),
        version: None,
        capabilities: Vec::new(),
        command: Some("/Users/test/dev/mesh/plugins/browser-tools".to_string()),
        args: Vec::new(),
        tools: Vec::new(),
        manifest: None,
        web_ui: plugin::PluginWebUiState::default(),
        startup: None,
        error: None,
    };

    assert_eq!(plugin_dashboard_command_name(&summary), "browser-tools");
}

#[tokio::test]
async fn dashboard_snapshot_provider_reuses_cached_inventory_within_ttl() {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .expect("test node should initialize");
    let local_processes = Arc::new(tokio::sync::Mutex::new(Vec::new()));
    let load_count = Arc::new(AtomicUsize::new(0));
    let load_count_for_loader = load_count.clone();
    let provider = RuntimeDashboardSnapshotProvider::with_inventory_loader(
        node,
        local_processes,
        None,
        RuntimeDashboardSnapshotProviderTestOptions {
            api_port: 9337,
            console_port: Some(3131),
            headless: false,
            inventory_snapshot_ttl: Duration::from_secs(60),
            inventory_snapshot_loader: Arc::new(move || {
                load_count_for_loader.fetch_add(1, AtomicOrdering::SeqCst);
                crate::models::LocalModelInventorySnapshot::default()
            }),
        },
    );

    let _ = provider.snapshot().await;
    let _ = provider.snapshot().await;

    assert_eq!(load_count.load(AtomicOrdering::SeqCst), 1);
}

#[tokio::test]
async fn dashboard_snapshot_provider_uses_runtime_ctx_and_inventory_file_size() {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .expect("test node should initialize");
    let model_name = "Runtime-Model".to_string();
    set_advertised_model_context(&node, &model_name, Some(8192)).await;
    let local_processes = Arc::new(tokio::sync::Mutex::new(vec![api::RuntimeProcessPayload {
        name: model_name.clone(),
        instance_id: None,
        backend: "CUDA0".to_string(),
        status: "ready".to_string(),
        port: 4001,
        pid: 1234,
        slots: 4,
        context_length: Some(8192),
        profile: String::new(),
    }]));
    let inventory_model_name = model_name.clone();
    let provider = RuntimeDashboardSnapshotProvider::with_inventory_loader(
        node,
        local_processes,
        None,
        RuntimeDashboardSnapshotProviderTestOptions {
            api_port: 9337,
            console_port: Some(3131),
            headless: false,
            inventory_snapshot_ttl: Duration::from_secs(60),
            inventory_snapshot_loader: Arc::new(move || {
                let mut snapshot = crate::models::LocalModelInventorySnapshot::default();
                snapshot
                    .size_by_name
                    .insert(inventory_model_name.clone(), 24_000_000_000);
                snapshot.metadata_by_name.insert(
                    inventory_model_name.clone(),
                    crate::proto::node::CompactModelMetadata {
                        model_key: inventory_model_name.clone(),
                        context_length: 4096,
                        quantization_type: "Q4_K_M".to_string(),
                        ..Default::default()
                    },
                );
                snapshot
            }),
        },
    );
    provider
        .local_context_usage
        .lock()
        .await
        .entry(model_name.clone())
        .or_default()
        .insert(
            DashboardContextUsageSource {
                port: 4001,
                pid: 1234,
            },
            2048,
        );

    let snapshot = provider.snapshot().await;
    assert_eq!(snapshot.loaded_model_rows.len(), 1);
    assert_eq!(snapshot.loaded_model_rows[0].slots, Some(4));
    assert_eq!(snapshot.loaded_model_rows[0].ctx_size, Some(8192));
    assert_eq!(snapshot.loaded_model_rows[0].ctx_used_tokens, Some(2048));
    assert_eq!(snapshot.loaded_model_rows[0].file_size_gb, Some(24.0));
    assert_eq!(
        snapshot.loaded_model_rows[0].quantization.as_deref(),
        Some("Q4_K_M")
    );
}

#[tokio::test]
async fn dashboard_snapshot_provider_uses_per_model_runtime_slot_snapshots() {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .expect("test node should initialize");
    let producer = node
        .runtime_data_collector()
        .producer(crate::runtime_data::RuntimeDataSource {
            scope: "runtime",
            plugin_data_key: None,
            plugin_endpoint_key: None,
        });
    let local_processes = Arc::new(tokio::sync::Mutex::new(vec![
        api::RuntimeProcessPayload {
            name: "model-a".to_string(),
            instance_id: None,
            backend: "skippy".to_string(),
            status: "ready".to_string(),
            port: 4001,
            pid: 1234,
            slots: 2,
            context_length: Some(8192),
            profile: String::new(),
        },
        api::RuntimeProcessPayload {
            name: "model-b".to_string(),
            instance_id: None,
            backend: "skippy".to_string(),
            status: "ready".to_string(),
            port: 4002,
            pid: 1235,
            slots: 2,
            context_length: Some(8192),
            profile: String::new(),
        },
    ]));
    producer.publish_llama_slots_snapshot(crate::runtime_data::RuntimeLlamaSlotsSnapshot {
        status: crate::runtime_data::RuntimeLlamaEndpointStatus::Ready,
        model: Some("model-a".to_string()),
        instance_id: None,
        last_attempt_unix_ms: Some(1),
        last_success_unix_ms: Some(1),
        error: None,
        slots: vec![
            crate::runtime_data::RuntimeLlamaSlotSnapshot {
                id: Some(0),
                is_processing: Some(true),
                ..crate::runtime_data::RuntimeLlamaSlotSnapshot::default()
            },
            crate::runtime_data::RuntimeLlamaSlotSnapshot {
                id: Some(1),
                is_processing: Some(false),
                ..crate::runtime_data::RuntimeLlamaSlotSnapshot::default()
            },
        ],
    });
    producer.publish_llama_slots_snapshot(crate::runtime_data::RuntimeLlamaSlotsSnapshot {
        status: crate::runtime_data::RuntimeLlamaEndpointStatus::Ready,
        model: Some("model-b".to_string()),
        instance_id: None,
        last_attempt_unix_ms: Some(2),
        last_success_unix_ms: Some(2),
        error: None,
        slots: vec![
            crate::runtime_data::RuntimeLlamaSlotSnapshot {
                id: Some(0),
                is_processing: Some(false),
                ..crate::runtime_data::RuntimeLlamaSlotSnapshot::default()
            },
            crate::runtime_data::RuntimeLlamaSlotSnapshot {
                id: Some(1),
                is_processing: Some(true),
                ..crate::runtime_data::RuntimeLlamaSlotSnapshot::default()
            },
        ],
    });

    let provider = RuntimeDashboardSnapshotProvider::with_inventory_loader(
        node,
        local_processes,
        None,
        RuntimeDashboardSnapshotProviderTestOptions {
            api_port: 9337,
            console_port: Some(3131),
            headless: false,
            inventory_snapshot_ttl: Duration::from_secs(60),
            inventory_snapshot_loader: Arc::new(
                crate::models::LocalModelInventorySnapshot::default,
            ),
        },
    );

    let snapshot = provider.snapshot().await;
    let model_a = snapshot
        .loaded_model_rows
        .iter()
        .find(|row| row.name == "model-a")
        .expect("model-a row should be present");
    let model_b = snapshot
        .loaded_model_rows
        .iter()
        .find(|row| row.name == "model-b")
        .expect("model-b row should be present");
    assert_eq!(
        model_a.lanes.as_ref().map(|lanes| {
            lanes
                .iter()
                .map(|lane| (lane.index, lane.active))
                .collect::<Vec<_>>()
        }),
        Some(vec![(0, true), (1, false)])
    );
    assert_eq!(
        model_b.lanes.as_ref().map(|lanes| {
            lanes
                .iter()
                .map(|lane| (lane.index, lane.active))
                .collect::<Vec<_>>()
        }),
        Some(vec![(0, false), (1, true)])
    );
}

#[tokio::test]
async fn dashboard_snapshot_provider_maps_canonical_model_refs_to_inventory_metadata() {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .expect("test node should initialize");
    let runtime_model_name = "unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL".to_string();
    let inventory_model_name = "Qwen3.5-4B-UD-Q4_K_XL".to_string();
    let local_processes = Arc::new(tokio::sync::Mutex::new(vec![api::RuntimeProcessPayload {
        name: runtime_model_name.clone(),
        instance_id: None,
        backend: "skippy".to_string(),
        status: "ready".to_string(),
        port: 37615,
        pid: 132098,
        slots: 4,
        context_length: Some(65_536),
        profile: String::new(),
    }]));
    let provider = RuntimeDashboardSnapshotProvider::with_inventory_loader(
        node,
        local_processes,
        None,
        RuntimeDashboardSnapshotProviderTestOptions {
            api_port: 9337,
            console_port: Some(3131),
            headless: false,
            inventory_snapshot_ttl: Duration::from_secs(60),
            inventory_snapshot_loader: Arc::new(move || {
                let mut snapshot = crate::models::LocalModelInventorySnapshot::default();
                snapshot
                    .size_by_name
                    .insert(inventory_model_name.clone(), 9_876_000_000);
                snapshot.metadata_by_name.insert(
                    inventory_model_name.clone(),
                    crate::proto::node::CompactModelMetadata {
                        model_key: inventory_model_name.clone(),
                        context_length: 4096,
                        quantization_type: "Q4_K_XL".to_string(),
                        ..Default::default()
                    },
                );
                snapshot
            }),
        },
    );

    let snapshot = provider.snapshot().await;
    assert_eq!(snapshot.loaded_model_rows.len(), 1);
    let row = &snapshot.loaded_model_rows[0];
    assert_eq!(row.name, runtime_model_name);
    assert_eq!(row.device, None);
    assert_eq!(row.slots, Some(4));
    assert_eq!(row.ctx_size, Some(65_536));
    assert_eq!(row.quantization.as_deref(), Some("Q4_K_XL"));
    assert_eq!(row.file_size_gb, Some(9.876));
}

#[tokio::test]
async fn dashboard_snapshot_provider_prefers_node_context_over_inventory_metadata() {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .expect("test node should initialize");
    let model_name = "unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL".to_string();
    set_advertised_model_context(&node, &model_name, Some(131_072)).await;
    let local_processes = Arc::new(tokio::sync::Mutex::new(vec![api::RuntimeProcessPayload {
        name: model_name.clone(),
        instance_id: None,
        backend: "skippy".to_string(),
        status: "ready".to_string(),
        port: 34097,
        pid: 132099,
        slots: 4,
        context_length: None,
        profile: String::new(),
    }]));
    let provider = RuntimeDashboardSnapshotProvider::with_inventory_loader(
        node,
        local_processes,
        None,
        RuntimeDashboardSnapshotProviderTestOptions {
            api_port: 9337,
            console_port: Some(3131),
            headless: false,
            inventory_snapshot_ttl: Duration::from_secs(60),
            inventory_snapshot_loader: Arc::new(move || {
                let mut snapshot = crate::models::LocalModelInventorySnapshot::default();
                snapshot.metadata_by_name.insert(
                    "Qwen3.6-27B-UD-Q4_K_XL".to_string(),
                    crate::proto::node::CompactModelMetadata {
                        model_key: "Qwen3.6-27B-UD-Q4_K_XL".to_string(),
                        context_length: 4096,
                        quantization_type: "Q4_K_XL".to_string(),
                        ..Default::default()
                    },
                );
                snapshot
            }),
        },
    );

    let snapshot = provider.snapshot().await;
    assert_eq!(snapshot.loaded_model_rows.len(), 1);
    let row = &snapshot.loaded_model_rows[0];
    assert_eq!(row.ctx_size, Some(131_072));
    assert_eq!(row.quantization.as_deref(), Some("Q4_K_XL"));
}

#[test]
fn dashboard_quantization_fallback_strips_direct_gguf_extension() {
    assert_eq!(
        dashboard_quantization_from_model_name("/models/Qwen3.5-4B-Q4_K_M.gguf").as_deref(),
        Some("Q4_K_M")
    );
}

#[test]
fn dashboard_endpoint_rows_keep_builtins_grouped_before_plugins() {
    let mut rows = vec![
        DashboardEndpointRow {
            label: "Plugin: zebra".to_string(),
            status: RuntimeStatus::Ready,
            url: "zebra".to_string(),
            port: 0,
            pid: Some(1001),
        },
        DashboardEndpointRow {
            label: "Web console".to_string(),
            status: RuntimeStatus::Ready,
            url: "http://localhost:3131".to_string(),
            port: 3131,
            pid: None,
        },
        DashboardEndpointRow {
            label: "Plugin: alpha".to_string(),
            status: RuntimeStatus::Ready,
            url: "alpha".to_string(),
            port: 0,
            pid: Some(1000),
        },
        DashboardEndpointRow {
            label: "Metrics".to_string(),
            status: RuntimeStatus::Ready,
            url: "metrics".to_string(),
            port: 0,
            pid: None,
        },
        DashboardEndpointRow {
            label: "OpenAI-compatible API".to_string(),
            status: RuntimeStatus::Ready,
            url: "http://localhost:9337".to_string(),
            port: 9337,
            pid: None,
        },
    ];

    sort_dashboard_endpoint_rows(&mut rows);

    let labels = rows.into_iter().map(|row| row.label).collect::<Vec<_>>();
    assert_eq!(
        labels,
        vec![
            "Metrics".to_string(),
            "OpenAI-compatible API".to_string(),
            "Web console".to_string(),
            "Plugin: alpha".to_string(),
            "Plugin: zebra".to_string(),
        ]
    );
}
