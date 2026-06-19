use super::runtime::ConfigControlOption;
use crate::api::MeshApi;
use crate::mesh;
use crate::models::LocalModelInventorySnapshot;
use mesh_llm_config::{ConfigConditionValue, ConfigOptionsSource};
use mesh_llm_native_runtime::NativeRuntimeBackendKind;
use mesh_llm_plugin_manager::{InstalledPluginMetadata, PluginStore, default_store_root};

#[derive(Clone, Debug, Default)]
pub(crate) struct RuntimeControlStateSources {
    pub(crate) gpus: RuntimeOptionsState,
    pub(crate) native_backends: RuntimeOptionsState,
    pub(crate) local_models: RuntimeOptionsState,
    pub(crate) installed_plugins: RuntimeOptionsState,
    pub(crate) mesh_peers: RuntimeOptionsState,
}

#[derive(Clone, Debug, Default)]
pub(crate) enum RuntimeOptionsState {
    #[default]
    Unknown,
    Options(Vec<ConfigControlOption>),
    Unavailable {
        reason: String,
        note: Option<String>,
    },
}

pub(crate) async fn collect_runtime_control_state_sources(
    state: &MeshApi,
    installed_plugins: Option<Vec<InstalledPluginMetadata>>,
) -> RuntimeControlStateSources {
    #[cfg(test)]
    if let Some(sources) = test_override_sources() {
        return sources;
    }

    let survey = crate::system::hardware::survey();
    let local_models = state.local_inventory_snapshot().await;
    let peers = state.node().await.peers().await;
    RuntimeControlStateSources {
        gpus: gpu_state(&survey.gpus),
        native_backends: native_backend_state(),
        local_models: local_model_state(&local_models),
        installed_plugins: installed_plugin_state(installed_plugins),
        mesh_peers: mesh_peer_state(&peers),
    }
}

pub(crate) fn load_installed_plugins() -> Option<Vec<InstalledPluginMetadata>> {
    let root = default_store_root().ok()?;
    PluginStore::new(root).list().ok()
}

fn gpu_state(gpus: &[crate::system::hardware::GpuFacts]) -> RuntimeOptionsState {
    let options = gpus
        .iter()
        .filter_map(|gpu| gpu.backend_device.as_ref().map(|device| (gpu, device)))
        .map(|(gpu, device)| ConfigControlOption {
            value: ConfigConditionValue::String(device.clone()),
            label: Some(format!("{} ({device})", gpu.display_name)),
            note: Some(format!(
                "{:.1} GiB VRAM",
                gpu.vram_bytes as f64 / 1_073_741_824.0
            )),
            disabled: false,
            reason: None,
            source: ConfigOptionsSource::RuntimeGpus,
        })
        .collect::<Vec<_>>();
    if options.is_empty() {
        return RuntimeOptionsState::Unavailable {
            reason: "No compatible GPU was detected.".to_string(),
            note: None,
        };
    }
    RuntimeOptionsState::Options(options)
}

fn native_backend_state() -> RuntimeOptionsState {
    let mut kinds = crate::system::native_runtime_install::host_runtime_profile().available_flavors;
    if let Ok(cache) = crate::system::native_runtime_install::default_native_runtime_cache()
        && let Ok(installed) = cache.installed()
    {
        for runtime in installed {
            kinds.insert(runtime.manifest.runtime.backend.kind);
        }
    }
    RuntimeOptionsState::Options(
        kinds
            .into_iter()
            .map(|kind| ConfigControlOption {
                value: ConfigConditionValue::String(kind.as_str().to_string()),
                label: Some(native_backend_label(&kind).to_string()),
                note: None,
                disabled: false,
                reason: None,
                source: ConfigOptionsSource::RuntimeNativeBackends,
            })
            .collect(),
    )
}

fn local_model_state(snapshot: &LocalModelInventorySnapshot) -> RuntimeOptionsState {
    let mut model_names = snapshot.model_names.iter().cloned().collect::<Vec<_>>();
    model_names.sort();
    if model_names.is_empty() {
        return RuntimeOptionsState::Unavailable {
            reason: "No local models were found.".to_string(),
            note: None,
        };
    }
    RuntimeOptionsState::Options(
        model_names
            .into_iter()
            .map(|name| ConfigControlOption {
                note: snapshot
                    .size_by_name
                    .get(&name)
                    .map(|size| format!("{:.1} GiB", *size as f64 / 1_073_741_824.0)),
                value: ConfigConditionValue::String(name.clone()),
                label: Some(name),
                disabled: false,
                reason: None,
                source: ConfigOptionsSource::RuntimeLocalModels,
            })
            .collect(),
    )
}

fn installed_plugin_state(
    installed_plugins: Option<Vec<InstalledPluginMetadata>>,
) -> RuntimeOptionsState {
    let Some(installed_plugins) = installed_plugins else {
        return RuntimeOptionsState::Unknown;
    };
    if installed_plugins.is_empty() {
        return RuntimeOptionsState::Unavailable {
            reason: "No installed plugins were found.".to_string(),
            note: None,
        };
    }
    RuntimeOptionsState::Options(
        installed_plugins
            .into_iter()
            .map(|plugin| ConfigControlOption {
                value: ConfigConditionValue::String(plugin.name.clone()),
                label: Some(plugin.name),
                note: Some(plugin.installed_version),
                disabled: !plugin.enabled,
                reason: (!plugin.enabled).then(|| {
                    plugin
                        .last_error
                        .unwrap_or_else(|| "Installed plugin is disabled.".to_string())
                }),
                source: ConfigOptionsSource::RuntimeInstalledPlugins,
            })
            .collect(),
    )
}

fn mesh_peer_state(peers: &[mesh::PeerInfo]) -> RuntimeOptionsState {
    if peers.is_empty() {
        return RuntimeOptionsState::Unavailable {
            reason: "No mesh peers are currently available.".to_string(),
            note: None,
        };
    }
    RuntimeOptionsState::Options(
        peers
            .iter()
            .map(|peer| ConfigControlOption {
                value: ConfigConditionValue::String(peer.id.fmt_short().to_string()),
                label: Some(
                    peer.hostname
                        .clone()
                        .filter(|hostname| !hostname.trim().is_empty())
                        .unwrap_or_else(|| peer.id.fmt_short().to_string()),
                ),
                note: None,
                disabled: false,
                reason: None,
                source: ConfigOptionsSource::RuntimeMeshPeers,
            })
            .collect(),
    )
}

fn native_backend_label(kind: &NativeRuntimeBackendKind) -> &'static str {
    match kind {
        NativeRuntimeBackendKind::Cpu => "CPU",
        NativeRuntimeBackendKind::Metal => "Metal",
        NativeRuntimeBackendKind::Cuda => "CUDA",
        NativeRuntimeBackendKind::Rocm => "ROCm",
        NativeRuntimeBackendKind::Vulkan => "Vulkan",
        NativeRuntimeBackendKind::Other(_) => "Other",
    }
}

#[cfg(test)]
static TEST_RUNTIME_CONTROL_STATE_SOURCES: std::sync::OnceLock<
    std::sync::Mutex<Option<RuntimeControlStateSources>>,
> = std::sync::OnceLock::new();

#[cfg(test)]
fn test_runtime_control_state_sources()
-> &'static std::sync::Mutex<Option<RuntimeControlStateSources>> {
    TEST_RUNTIME_CONTROL_STATE_SOURCES.get_or_init(|| std::sync::Mutex::new(None))
}

#[cfg(test)]
fn test_override_sources() -> Option<RuntimeControlStateSources> {
    test_runtime_control_state_sources().lock().ok()?.clone()
}

#[cfg(test)]
pub(crate) fn set_test_runtime_control_state_sources(sources: Option<RuntimeControlStateSources>) {
    if let Ok(mut guard) = test_runtime_control_state_sources().lock() {
        *guard = sources;
    }
}
