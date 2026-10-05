use super::runtime::{ConfigControlStateEntry, ConfigControlStatePayload};
#[cfg(test)]
pub(crate) use super::runtime_control_state_sources::set_test_runtime_control_state_sources;
pub(crate) use super::runtime_control_state_sources::{
    RuntimeControlStateSources, RuntimeOptionsState,
};
use super::runtime_control_state_sources::{
    collect_runtime_control_state_sources, load_installed_plugins,
};
use crate::api::MeshApi;
use crate::config_schema::aggregate_config_schema_sources;
use mesh_llm_config::{
    ConfigControlAvailabilitySource, ConfigControlBehavior, ConfigDisabledWritePolicy,
    ConfigOptionsSource, ConfigSettingSchema,
};

pub(super) async fn collect_runtime_config_control_state_payload(
    state: &MeshApi,
) -> ConfigControlStatePayload {
    let installed_plugins = load_installed_plugins();
    let schema_result = aggregate_config_schema_sources(
        std::iter::empty(),
        installed_plugins.clone().unwrap_or_default(),
    );
    let Ok(schema) = schema_result else {
        let error = schema_result
            .err()
            .map(|error| error.to_string())
            .unwrap_or_else(|| "unknown schema aggregation error".to_string());
        tracing::warn!(%error, "failed to aggregate config schema for runtime control state");
        return ConfigControlStatePayload::default();
    };
    let sources = collect_runtime_control_state_sources(state, installed_plugins).await;
    build_runtime_control_state_payload(schema.iter().map(|(_, entry)| &entry.setting), &sources)
}

pub(crate) fn build_runtime_control_state_payload<'a>(
    settings: impl IntoIterator<Item = &'a ConfigSettingSchema>,
    sources: &RuntimeControlStateSources,
) -> ConfigControlStatePayload {
    let mut payload = ConfigControlStatePayload::default();
    for setting in settings {
        let Some(behavior) = setting.control_behavior.as_ref() else {
            continue;
        };
        let Some(options_source) = behavior.options_source else {
            continue;
        };
        if options_source == ConfigOptionsSource::Static {
            continue;
        }
        let rendered_path = setting.path.render();
        if let Some(entry) = schema_disabled_entry(setting, behavior) {
            payload.settings.insert(rendered_path, entry);
            continue;
        }
        let source_state = source_state(sources, options_source);
        let Some(entry) = runtime_entry(setting, &source_state) else {
            continue;
        };
        payload.settings.insert(rendered_path, entry);
    }
    payload
}

fn schema_disabled_entry(
    setting: &ConfigSettingSchema,
    behavior: &ConfigControlBehavior,
) -> Option<ConfigControlStateEntry> {
    let availability = behavior.availability.as_ref()?;
    if availability.enabled {
        return None;
    }
    Some(ConfigControlStateEntry {
        enabled: false,
        reason: availability.reason.clone(),
        note: availability.note.clone(),
        source: availability.source,
        write_policy: write_policy_for(setting, availability.source),
        options: None,
    })
}

fn runtime_entry(
    setting: &ConfigSettingSchema,
    state: &RuntimeOptionsState,
) -> Option<ConfigControlStateEntry> {
    match state {
        RuntimeOptionsState::Unknown => None,
        RuntimeOptionsState::Options(options) => Some(ConfigControlStateEntry {
            enabled: true,
            reason: None,
            note: None,
            source: ConfigControlAvailabilitySource::Runtime,
            write_policy: write_policy_for(setting, ConfigControlAvailabilitySource::Runtime),
            options: Some(options.to_vec()),
        }),
        RuntimeOptionsState::Unavailable { reason, note } => Some(ConfigControlStateEntry {
            enabled: false,
            reason: Some(reason.clone()),
            note: note.clone(),
            source: ConfigControlAvailabilitySource::Runtime,
            write_policy: write_policy_for(setting, ConfigControlAvailabilitySource::Runtime),
            options: None,
        }),
    }
}

fn source_state(
    sources: &RuntimeControlStateSources,
    source: ConfigOptionsSource,
) -> RuntimeOptionsState {
    match source {
        ConfigOptionsSource::Static => RuntimeOptionsState::Unknown,
        ConfigOptionsSource::RuntimeGpus => sources.gpus.clone(),
        ConfigOptionsSource::RuntimeNativeBackends => sources.native_backends.clone(),
        ConfigOptionsSource::RuntimeLocalModels => sources.local_models.clone(),
        ConfigOptionsSource::RuntimeInstalledPlugins => sources.installed_plugins.clone(),
        ConfigOptionsSource::RuntimeMeshPeers => sources.mesh_peers.clone(),
    }
}

fn write_policy_for(
    setting: &ConfigSettingSchema,
    source: ConfigControlAvailabilitySource,
) -> ConfigDisabledWritePolicy {
    setting
        .default_disabled_write_policy(Some(source))
        .unwrap_or(ConfigDisabledWritePolicy::PreserveExisting)
}
