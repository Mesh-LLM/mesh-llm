use super::status::current_time_unix_ms;
use super::{LocalRuntimeModelHandle, sort_dashboard_endpoint_rows};
use crate::{api, mesh, models, plugin};
use mesh_llm_events::{
    DashboardAcceptedRequestBucket, DashboardEndpointRow, DashboardModelLane, DashboardModelRow,
    DashboardProcessRow, DashboardSnapshot, DashboardSnapshotFuture, DashboardSnapshotProvider,
    RuntimeStatus,
};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub(super) const PRETTY_DASHBOARD_INVENTORY_CACHE_TTL: Duration = Duration::from_secs(5);
pub(super) const DASHBOARD_CONTEXT_USAGE_REFRESH_INTERVAL: Duration = Duration::from_millis(250);
pub(super) const DASHBOARD_FIRST_PAINT_TIMEOUT: Duration = Duration::from_secs(2);
pub(super) const SPLIT_STANDBY_RETRY_INTERVAL: Duration = Duration::from_secs(30);
pub(super) const MODEL_TARGET_RECONCILIATION_INTERVAL: Duration = Duration::from_secs(15);

pub(super) type DashboardContextUsage =
    Arc<tokio::sync::Mutex<HashMap<String, HashMap<DashboardContextUsageSource, u64>>>>;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct DashboardContextUsageSource {
    pub(super) port: u16,
    pub(super) pid: u32,
}

pub(super) struct RuntimeDashboardSnapshotProvider {
    pub(super) node: mesh::Node,
    pub(super) local_processes: Arc<tokio::sync::Mutex<Vec<api::RuntimeProcessPayload>>>,
    pub(super) local_context_usage: DashboardContextUsage,
    pub(super) runtime_data_collector: crate::runtime_data::RuntimeDataCollector,
    pub(super) plugin_manager: Option<plugin::PluginManager>,
    pub(super) api_port: u16,
    pub(super) console_port: Option<u16>,
    pub(super) headless: bool,
    pub(super) inventory_snapshot_cache: Arc<tokio::sync::Mutex<CachedDashboardInventorySnapshot>>,
    pub(super) inventory_snapshot_ttl: Duration,
    pub(super) inventory_snapshot_loader:
        Arc<dyn Fn() -> crate::models::LocalModelInventorySnapshot + Send + Sync>,
}

#[cfg(test)]
pub(super) struct RuntimeDashboardSnapshotProviderTestOptions {
    pub(super) api_port: u16,
    pub(super) console_port: Option<u16>,
    pub(super) headless: bool,
    pub(super) inventory_snapshot_ttl: Duration,
    pub(super) inventory_snapshot_loader:
        Arc<dyn Fn() -> crate::models::LocalModelInventorySnapshot + Send + Sync>,
}

#[derive(Clone, Default)]
pub(super) struct CachedDashboardInventorySnapshot {
    pub(super) snapshot: crate::models::LocalModelInventorySnapshot,
    pub(super) captured_at: Option<Instant>,
}

impl RuntimeDashboardSnapshotProvider {
    pub(super) fn new(
        node: mesh::Node,
        local_processes: Arc<tokio::sync::Mutex<Vec<api::RuntimeProcessPayload>>>,
        local_context_usage: DashboardContextUsage,
        plugin_manager: Option<plugin::PluginManager>,
        api_port: u16,
        console_port: Option<u16>,
        headless: bool,
    ) -> Self {
        Self {
            runtime_data_collector: node.runtime_data_collector(),
            node,
            local_processes,
            local_context_usage,
            plugin_manager,
            api_port,
            console_port,
            headless,
            inventory_snapshot_cache: Arc::new(tokio::sync::Mutex::new(
                CachedDashboardInventorySnapshot::default(),
            )),
            inventory_snapshot_ttl: PRETTY_DASHBOARD_INVENTORY_CACHE_TTL,
            inventory_snapshot_loader: Arc::new(|| {
                crate::models::scan_local_inventory_snapshot_with_progress(|_| {})
            }),
        }
    }

    #[cfg(test)]
    pub(super) fn with_inventory_loader(
        node: mesh::Node,
        local_processes: Arc<tokio::sync::Mutex<Vec<api::RuntimeProcessPayload>>>,
        plugin_manager: Option<plugin::PluginManager>,
        options: RuntimeDashboardSnapshotProviderTestOptions,
    ) -> Self {
        Self {
            runtime_data_collector: node.runtime_data_collector(),
            node,
            local_processes,
            local_context_usage: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            plugin_manager,
            api_port: options.api_port,
            console_port: options.console_port,
            headless: options.headless,
            inventory_snapshot_cache: Arc::new(tokio::sync::Mutex::new(
                CachedDashboardInventorySnapshot::default(),
            )),
            inventory_snapshot_ttl: options.inventory_snapshot_ttl,
            inventory_snapshot_loader: options.inventory_snapshot_loader,
        }
    }

    async fn inventory_snapshot(&self) -> crate::models::LocalModelInventorySnapshot {
        {
            let cache = self.inventory_snapshot_cache.lock().await;
            if let Some(captured_at) = cache.captured_at
                && captured_at.elapsed() < self.inventory_snapshot_ttl
            {
                return cache.snapshot.clone();
            }
        }

        let inventory_snapshot_loader = self.inventory_snapshot_loader.clone();
        let snapshot = match tokio::task::spawn_blocking(move || inventory_snapshot_loader()).await
        {
            Ok(snapshot) => snapshot,
            Err(err) => {
                tracing::warn!("pretty dashboard inventory snapshot failed: {err}");
                crate::models::LocalModelInventorySnapshot::default()
            }
        };

        let mut cache = self.inventory_snapshot_cache.lock().await;
        cache.snapshot = snapshot.clone();
        cache.captured_at = Some(Instant::now());
        snapshot
    }
}

pub(super) fn dashboard_inventory_value_for_model<'a, T>(
    values_by_name: &'a HashMap<String, T>,
    model_name: &str,
) -> Option<&'a T> {
    dashboard_inventory_model_keys(model_name)
        .into_iter()
        .find_map(|key| values_by_name.get(&key))
}

pub(super) fn dashboard_context_usage_for_process(
    values_by_name: &HashMap<String, HashMap<DashboardContextUsageSource, u64>>,
    process: &api::RuntimeProcessPayload,
) -> Option<u64> {
    let source = DashboardContextUsageSource {
        port: process.port,
        pid: process.pid,
    };
    dashboard_inventory_model_keys(&process.name)
        .into_iter()
        .filter_map(|key| values_by_name.get(&key))
        .find_map(|source_values| source_values.get(&source).copied())
}

pub(super) fn dashboard_lanes_for_process(
    snapshots_by_instance: &BTreeMap<String, crate::runtime_data::RuntimeLlamaRuntimeSnapshot>,
    snapshots_by_model: &BTreeMap<String, crate::runtime_data::RuntimeLlamaRuntimeSnapshot>,
    process: &api::RuntimeProcessPayload,
) -> Option<Vec<DashboardModelLane>> {
    let snapshot = process
        .instance_id
        .as_ref()
        .and_then(|instance_id| snapshots_by_instance.get(instance_id))
        .or_else(|| snapshots_by_model.get(&process.name))?;

    let mut lanes = snapshot
        .items
        .slots
        .iter()
        .map(|slot| DashboardModelLane {
            index: dashboard_lane_index_for_slot(slot),
            active: slot.is_processing,
        })
        .collect::<Vec<_>>();
    lanes.sort_by_key(|lane| lane.index);
    (!lanes.is_empty()).then_some(lanes)
}

pub(super) fn dashboard_lane_index_for_slot(
    slot: &crate::runtime_data::RuntimeLlamaSlotItem,
) -> usize {
    slot.id
        .and_then(|id| usize::try_from(id).ok())
        .unwrap_or(slot.index)
}

pub(super) fn dashboard_quantization_from_model_name(model_name: &str) -> Option<String> {
    dashboard_inventory_model_keys(model_name)
        .into_iter()
        .map(|key| models::inventory::derive_quantization_type(&key))
        .map(|quantization| quantization.trim().trim_end_matches(".gguf").to_string())
        .find(|quantization| !quantization.is_empty())
}

pub(super) fn dashboard_inventory_model_keys(model_name: &str) -> Vec<String> {
    let mut keys = Vec::new();
    push_dashboard_inventory_model_key(&mut keys, model_name.trim());
    if let Some(base_name) = model_name.trim().rsplit('/').next() {
        push_dashboard_inventory_model_key(&mut keys, base_name);
    }

    let seeds = keys.clone();
    for key in seeds {
        if let Some(without_gguf_variant) = strip_gguf_variant_marker(&key) {
            push_dashboard_inventory_model_key(&mut keys, &without_gguf_variant);
        }
        push_dashboard_inventory_model_key(&mut keys, &key.replace(':', "-"));
        if key.to_ascii_lowercase().ends_with(".gguf") {
            push_dashboard_inventory_model_key(&mut keys, &key[..key.len().saturating_sub(5)]);
        }
    }
    keys
}

pub(super) fn strip_gguf_variant_marker(model_name: &str) -> Option<String> {
    let lower = model_name.to_ascii_lowercase();
    for marker in ["-gguf:", ":gguf:"] {
        if let Some(index) = lower.find(marker) {
            let variant_start = index + marker.len();
            return Some(format!(
                "{}-{}",
                &model_name[..index],
                &model_name[variant_start..]
            ));
        }
    }
    None
}

pub(super) fn push_dashboard_inventory_model_key(keys: &mut Vec<String>, key: &str) {
    let key = key.trim();
    if !key.is_empty() && !keys.iter().any(|candidate| candidate == key) {
        keys.push(key.to_string());
    }
}

impl DashboardSnapshotProvider for RuntimeDashboardSnapshotProvider {
    fn snapshot(&self) -> DashboardSnapshotFuture<'_> {
        let node = self.node.clone();
        let local_processes = self.local_processes.clone();
        let local_context_usage = self.local_context_usage.clone();
        let runtime_data_collector = self.runtime_data_collector.clone();
        let api_port = self.api_port;
        let console_port = self.console_port;
        let headless = self.headless;
        let plugin_manager = self.plugin_manager.clone();
        let provider = self;

        Box::pin(async move {
            let process_rows = local_processes.lock().await.clone();
            let context_usage_by_name = local_context_usage.lock().await.clone();
            let llama_runtime_by_model = runtime_data_collector.runtime_llama_snapshots_by_model();
            let llama_runtime_by_instance =
                runtime_data_collector.runtime_llama_snapshots_by_instance();
            let request_metrics = node.local_request_metrics_snapshot();
            let accepted_request_counts_len = request_metrics.accepted_request_counts.len();
            let inventory_snapshot = provider.inventory_snapshot().await;
            let metadata_by_name = inventory_snapshot.metadata_by_name;
            let size_by_name = inventory_snapshot.size_by_name;
            let mut loaded_model_rows = Vec::with_capacity(process_rows.len());
            for process in &process_rows {
                let metadata =
                    dashboard_inventory_value_for_model(&metadata_by_name, &process.name);
                let quantization = metadata
                    .map(|model| model.quantization_type.trim())
                    .filter(|value| !value.is_empty())
                    .map(str::to_string)
                    .or_else(|| dashboard_quantization_from_model_name(&process.name));
                let ctx_size = if let Some(context_length) = process.context_length {
                    Some(context_length)
                } else {
                    node.local_model_context_length(&process.name)
                        .await
                        .or_else(|| {
                            metadata
                                .map(|model| model.context_length)
                                .filter(|value| *value > 0)
                        })
                };
                loaded_model_rows.push(DashboardModelRow {
                    name: process.name.clone(),
                    role: dashboard_role_for_local_process(process),
                    status: runtime_status_from_process_status(&process.status),
                    port: Some(process.port),
                    device: None,
                    slots: Some(process.slots),
                    quantization,
                    ctx_size,
                    ctx_used_tokens: dashboard_context_usage_for_process(
                        &context_usage_by_name,
                        process,
                    ),
                    lanes: dashboard_lanes_for_process(
                        &llama_runtime_by_instance,
                        &llama_runtime_by_model,
                        process,
                    ),
                    file_size_gb: dashboard_inventory_value_for_model(&size_by_name, &process.name)
                        .map(|size| *size as f64 / 1e9),
                });
            }
            loaded_model_rows.sort_by(|left, right| left.name.cmp(&right.name));

            let mut webserver_rows =
                build_dashboard_endpoint_rows(api_port, console_port, headless);
            if let Some(plugin_manager) = plugin_manager {
                webserver_rows.extend(plugin_dashboard_endpoint_rows(&plugin_manager).await);
            }
            sort_dashboard_endpoint_rows(&mut webserver_rows);

            DashboardSnapshot {
                llama_process_rows: process_rows
                    .into_iter()
                    .map(|process| DashboardProcessRow {
                        name: process.name,
                        backend: process.backend,
                        status: runtime_status_from_process_status(&process.status),
                        port: process.port,
                        pid: process.pid,
                    })
                    .collect(),
                webserver_rows,
                loaded_model_rows,
                current_inflight_requests: node.inflight_requests(),
                accepted_request_buckets: request_metrics
                    .accepted_request_counts
                    .into_iter()
                    .enumerate()
                    .map(|(index, accepted_count)| DashboardAcceptedRequestBucket {
                        second_offset: accepted_request_counts_len.saturating_sub(1 + index) as u32,
                        accepted_count,
                    })
                    .collect(),
                latency_samples_ms: request_metrics.latency_samples_ms,
            }
        })
    }
}

#[allow(dead_code)]
pub(super) fn runtime_status_from_process_status(status: &str) -> RuntimeStatus {
    match status {
        "ready" => RuntimeStatus::Ready,
        "shutting down" | "shutting_down" => RuntimeStatus::ShuttingDown,
        "stopped" => RuntimeStatus::Stopped,
        "exited" => RuntimeStatus::Exited,
        "warning" => RuntimeStatus::Warning,
        "error" => RuntimeStatus::Error,
        _ => RuntimeStatus::Starting,
    }
}

#[allow(dead_code)]
pub(super) fn runtime_status_from_plugin_status(status: &str) -> RuntimeStatus {
    match status {
        "running" | "ready" => RuntimeStatus::Ready,
        "shutting down" | "shutting_down" => RuntimeStatus::ShuttingDown,
        "stopped" | "disabled" => RuntimeStatus::Stopped,
        "error" => RuntimeStatus::Error,
        "restarting" => RuntimeStatus::Warning,
        _ => RuntimeStatus::Starting,
    }
}

#[allow(dead_code)]
pub(super) fn dashboard_role_for_local_process(
    _process: &api::RuntimeProcessPayload,
) -> Option<String> {
    // `local_processes` only tracks local model-serving processes that own a ready
    // listening port on this node, so the pretty-only Loaded Models panel should
    // present them as host entries rather than inferring from event text.
    Some("host".to_string())
}

#[allow(dead_code)]
pub(super) fn build_dashboard_endpoint_rows(
    api_port: u16,
    console_port: Option<u16>,
    headless: bool,
) -> Vec<DashboardEndpointRow> {
    let mut rows = vec![DashboardEndpointRow {
        label: "OpenAI-compatible API".to_string(),
        status: RuntimeStatus::Ready,
        url: format!("http://localhost:{api_port}"),
        port: api_port,
        pid: None,
    }];
    if let Some(console_port) = console_port.filter(|_| !headless) {
        rows.push(DashboardEndpointRow {
            label: "Web console".to_string(),
            status: RuntimeStatus::Ready,
            url: format!("http://localhost:{console_port}"),
            port: console_port,
            pid: None,
        });
    }
    sort_dashboard_endpoint_rows(&mut rows);
    rows
}

#[allow(dead_code)]
pub(super) async fn plugin_dashboard_endpoint_rows(
    plugin_manager: &plugin::PluginManager,
) -> Vec<DashboardEndpointRow> {
    plugin_manager
        .list()
        .await
        .into_iter()
        .map(|summary| {
            let url = plugin_dashboard_command_name(&summary);
            DashboardEndpointRow {
                label: format!("Plugin: {}", summary.name),
                status: runtime_status_from_plugin_status(&summary.status),
                url,
                port: 0,
                pid: summary.pid,
            }
        })
        .collect()
}

pub(super) fn plugin_dashboard_command_name(summary: &plugin::PluginSummary) -> String {
    summary
        .command
        .as_deref()
        .filter(|command| !command.is_empty())
        .and_then(|command| {
            Path::new(command)
                .file_name()
                .and_then(|name| name.to_str())
                .filter(|name| !name.is_empty())
        })
        .unwrap_or(&summary.kind)
        .to_string()
}

pub(super) fn runtime_process_payload_with_status(
    name: &str,
    instance_id: Option<&str>,
    handle: &LocalRuntimeModelHandle,
    status: &str,
) -> api::RuntimeProcessPayload {
    api::RuntimeProcessPayload {
        name: name.to_string(),
        instance_id: instance_id.map(str::to_string),
        profile: String::new(),
        backend: handle.backend.clone(),
        status: status.to_string(),
        port: handle.port,
        pid: handle.pid(),
        slots: handle.slots,
        context_length: Some(handle.context_length),
    }
}

pub(super) async fn upsert_dashboard_process(
    shared: &Arc<tokio::sync::Mutex<Vec<api::RuntimeProcessPayload>>>,
    process: api::RuntimeProcessPayload,
) {
    let mut guard = shared.lock().await;
    guard.retain(|existing| {
        runtime_process_payload_identity(existing) != runtime_process_payload_identity(&process)
    });
    guard.push(process);
    guard.sort_by(|left, right| {
        (
            left.name.to_lowercase(),
            left.instance_id.as_deref().unwrap_or(""),
            left.port,
        )
            .cmp(&(
                right.name.to_lowercase(),
                right.instance_id.as_deref().unwrap_or(""),
                right.port,
            ))
    });
}

pub(super) async fn remove_dashboard_process(
    shared: &Arc<tokio::sync::Mutex<Vec<api::RuntimeProcessPayload>>>,
    target: &str,
) {
    let mut guard = shared.lock().await;
    let has_instance_match = guard
        .iter()
        .any(|process| process.instance_id.as_deref() == Some(target));
    guard.retain(|process| {
        if has_instance_match {
            process.instance_id.as_deref() != Some(target)
        } else {
            process.name != target
        }
    });
}

pub(super) fn runtime_process_payload_identity(process: &api::RuntimeProcessPayload) -> &str {
    process.instance_id.as_deref().unwrap_or(&process.name)
}

pub(super) async fn refresh_dashboard_context_usage(
    shared: &DashboardContextUsage,
    model_name: &str,
    handle: &LocalRuntimeModelHandle,
) {
    upsert_dashboard_context_usage(
        shared,
        model_name,
        dashboard_context_usage_source(handle),
        handle.ctx_used_tokens(),
    )
    .await;
}

pub(super) fn publish_runtime_llama_slots(
    producer: Option<&crate::runtime_data::RuntimeDataProducer>,
    model_name: &str,
    instance_id: Option<&str>,
    handle: &LocalRuntimeModelHandle,
) {
    let Some(producer) = producer else {
        return;
    };
    if let Some(snapshot) = handle.llama_slots_snapshot(model_name, instance_id) {
        producer.publish_llama_slots_snapshot(snapshot);
    }
}

pub(super) fn publish_runtime_llama_unavailable(
    producer: Option<&crate::runtime_data::RuntimeDataProducer>,
    model_name: &str,
    instance_id: Option<&str>,
) {
    let Some(producer) = producer else {
        return;
    };
    producer.publish_llama_slots_snapshot(crate::runtime_data::RuntimeLlamaSlotsSnapshot {
        status: crate::runtime_data::RuntimeLlamaEndpointStatus::Unavailable,
        model: Some(model_name.to_string()),
        instance_id: instance_id.map(str::to_string),
        last_attempt_unix_ms: Some(current_time_unix_ms()),
        last_success_unix_ms: None,
        error: None,
        slots: Vec::new(),
    });
}

pub(super) async fn refresh_dashboard_context_usage_batch(
    shared: &DashboardContextUsage,
    updates: Vec<(String, DashboardContextUsageSource, Option<u64>)>,
) {
    let mut guard = shared.lock().await;
    for (model_name, source, ctx_used_tokens) in updates {
        if let Some(ctx_used_tokens) = ctx_used_tokens {
            guard
                .entry(model_name)
                .or_default()
                .insert(source, ctx_used_tokens);
        } else {
            remove_dashboard_context_usage_source_locked(&mut guard, &model_name, source);
        }
    }
}

pub(super) async fn upsert_dashboard_context_usage(
    shared: &DashboardContextUsage,
    model_name: &str,
    source: DashboardContextUsageSource,
    ctx_used_tokens: Option<u64>,
) {
    let mut guard = shared.lock().await;
    if let Some(ctx_used_tokens) = ctx_used_tokens {
        guard
            .entry(model_name.to_string())
            .or_default()
            .insert(source, ctx_used_tokens);
    } else {
        remove_dashboard_context_usage_source_locked(&mut guard, model_name, source);
    }
}

pub(super) async fn remove_dashboard_context_usage(
    shared: &DashboardContextUsage,
    model_name: &str,
    handle: &LocalRuntimeModelHandle,
) {
    let mut guard = shared.lock().await;
    remove_dashboard_context_usage_source_locked(
        &mut guard,
        model_name,
        dashboard_context_usage_source(handle),
    );
}

pub(super) fn remove_dashboard_context_usage_source_locked(
    guard: &mut HashMap<String, HashMap<DashboardContextUsageSource, u64>>,
    model_name: &str,
    source: DashboardContextUsageSource,
) {
    let should_remove_model = if let Some(source_values) = guard.get_mut(model_name) {
        source_values.remove(&source);
        source_values.is_empty()
    } else {
        false
    };
    if should_remove_model {
        guard.remove(model_name);
    }
}

pub(super) fn dashboard_context_usage_source(
    handle: &LocalRuntimeModelHandle,
) -> DashboardContextUsageSource {
    DashboardContextUsageSource {
        port: handle.port,
        pid: handle.pid(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runtime_process(name: &str, port: u16, pid: u32) -> api::RuntimeProcessPayload {
        api::RuntimeProcessPayload {
            name: name.to_string(),
            instance_id: None,
            profile: String::new(),
            backend: String::new(),
            status: "ready".to_string(),
            port,
            pid,
            slots: 1,
            context_length: None,
        }
    }

    #[test]
    fn dashboard_context_usage_for_process_requires_exact_source() {
        let mut values_by_name = HashMap::new();
        values_by_name.insert(
            "model.gguf".to_string(),
            HashMap::from([(
                DashboardContextUsageSource {
                    port: 9001,
                    pid: 42,
                },
                512,
            )]),
        );

        assert_eq!(
            dashboard_context_usage_for_process(
                &values_by_name,
                &runtime_process("model.gguf", 9002, 42),
            ),
            None
        );
        assert_eq!(
            dashboard_context_usage_for_process(
                &values_by_name,
                &runtime_process("model.gguf", 9001, 43),
            ),
            None
        );
        assert_eq!(
            dashboard_context_usage_for_process(
                &values_by_name,
                &runtime_process("model.gguf", 9001, 42),
            ),
            Some(512)
        );
    }
}
