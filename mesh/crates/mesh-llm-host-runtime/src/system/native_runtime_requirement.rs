//! Native serving eligibility at the host startup and local-load boundaries.

use crate::{RuntimeOptions, plugin};
use anyhow::{Result, bail};
use std::sync::Mutex;

// The CLI initializes native libraries before installing its output surface.
// Retain the failure until a live provider handshake confirms the fallback.
static EXTERNAL_STARTUP_FAILURE: Mutex<Option<String>> = Mutex::new(None);

#[cfg(test)]
mod tests;
#[cfg(feature = "dynamic-native-runtime")]
mod unused_gpu;

pub(crate) fn ensure_native_runtime_available() -> Result<()> {
    if !skippy_runtime::native_runtime_loaded() {
        bail!(
            "Local model and split-stage loading require a MeshLLM native runtime; run `mesh-llm runtime install` and restart this node"
        );
    }
    Ok(())
}

#[cfg(any(feature = "dynamic-native-runtime", test))]
pub(crate) fn permits_external_startup(
    options: &RuntimeOptions,
    config: &plugin::MeshConfig,
) -> bool {
    !options.client
        && !options.local_model_only
        && options.model.is_empty()
        && options.gguf.is_empty()
        && options.mmproj.is_none()
        && options.draft.is_none()
        && !options.split
        && options.native_serving_plugin.is_none()
        // A model entry served by an inference plugin needs no native runtime;
        // only entries without a provider load through Skippy. Requiring the
        // runtime for every entry would block exactly the provider-only node
        // this fallback exists for.
        && !config
            .models
            .iter()
            .any(|model| model.provider.as_deref().is_none_or(str::is_empty))
        && options.llama_flavor.is_none()
        && config.runtime.native_runtime.selection.is_none()
        && config.runtime.native_runtime.mesh_version.is_none()
        && config.runtime.native_runtime.skippy_abi.is_none()
        && config.plugins.iter().any(declares_external_provider)
}

/// An enabled plugin entry that runs a provider process of its own.
///
/// Either a `command` (a locally installed plugin binary, which is how
/// `mesh-llm plugins install` declares one) or a `url` selects a real provider.
/// Built-in plugins are configuration toggles that carry neither, and a blank
/// value is not a provider.
#[cfg(any(feature = "dynamic-native-runtime", test))]
fn declares_external_provider(entry: &mesh_llm_config::PluginConfigEntry) -> bool {
    entry.enabled.unwrap_or(true)
        && (entry
            .command
            .as_deref()
            .is_some_and(|command| !command.trim().is_empty())
            || entry
                .url
                .as_deref()
                .is_some_and(|url| !url.trim().is_empty()))
}

pub(crate) async fn initialize(options: &RuntimeOptions) -> Result<()> {
    *EXTERNAL_STARTUP_FAILURE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
    let config = plugin::load_config(options.config.as_deref())?;
    crate::initialize_logging_foundation(&config.logging).await;
    if options.client {
        return Ok(());
    }
    #[cfg(feature = "dynamic-native-runtime")]
    {
        let allow_external = permits_external_startup(options, &config);
        let selection = super::native_runtime::NativeRuntimeStartupSelection::from_config(
            config.runtime.native_runtime.clone(),
            options.llama_flavor,
        )?;
        let runtime_selection = selection.runtime_selection.clone();
        let mesh_version = selection.mesh_version.clone();
        match super::native_runtime::try_load_installed_native_runtime(selection).await {
            Ok(Some(runtime)) => {
                tracing::info!(
                    native_runtime_id = %runtime.native_runtime_id,
                    libraries = ?runtime.libraries,
                    "Loaded MeshLLM native runtime"
                );
                unused_gpu::record(
                    &runtime_selection,
                    &mesh_version,
                    &runtime.release_version,
                    &runtime.native_runtime_id,
                );
            }
            Ok(None) => {}
            Err(error) if allow_external && is_unavailable_runtime_error(&error, true)? => {
                // Resolve executables now. Missing/disabled optional packages must
                // never count as providers. Capabilities are checked after the
                // real authenticated handshake, using the serving manager.
                let resolved = plugin::resolve_plugins(
                    &config,
                    plugin::PluginHostMode {
                        mesh_visibility: mesh_llm_plugin::MeshVisibility::Private,
                    },
                )?;
                if !resolved
                    .externals
                    .iter()
                    .any(plugin::ExternalPluginSpec::serves_models_itself)
                {
                    return Err(error);
                }
                *EXTERNAL_STARTUP_FAILURE
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) =
                    Some(format!("{error:#}"));
            }
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

#[cfg(feature = "dynamic-native-runtime")]
fn is_unavailable_runtime_error(error: &anyhow::Error, runtime_is_unused: bool) -> Result<bool> {
    if !is_runtime_availability_error(error) {
        return Ok(false);
    }
    // A node whose every model is provider-served never loads a native runtime,
    // so an installed or bundled one this build rejects costs it nothing. That
    // is the common state of a workstation: an older runtime is still cached
    // from a previous version, and refusing to start would block exactly the
    // provider-only node this fallback exists for. Say so rather than fail.
    if runtime_is_unused {
        return Ok(true);
    }
    // For a node that does want local models an absent catalog is recoverable;
    // installed or bundled artifacts that cannot load are operator errors.
    // Preserve those errors, including cache entries the normal recommendation
    // path would skip leniently.
    let cache = super::native_runtime_install::default_native_runtime_cache()?;
    let scan = cache.installed_lenient()?;
    let bundles = super::native_runtime_install::discover_native_runtime_bundle_dirs(&[])?;
    Ok(runtime_artifacts_absent(
        scan.runtimes.len(),
        scan.skipped.len(),
        bundles.len(),
    ))
}

#[cfg(feature = "dynamic-native-runtime")]
fn is_runtime_availability_error(error: &anyhow::Error) -> bool {
    error
        .downcast_ref::<super::native_runtime_install::NativeRuntimeResolutionError>()
        .is_some_and(|error| !error.enumeration_failed)
        || error
            .downcast_ref::<reqwest::Error>()
            .is_some_and(|error| error.is_connect() || error.is_timeout())
}

#[cfg(feature = "dynamic-native-runtime")]
fn runtime_artifacts_absent(installed: usize, skipped: usize, bundles: usize) -> bool {
    installed == 0 && skipped == 0 && bundles == 0
}

fn require_external_inference_provider(failure: &str, endpoint_count: usize) -> Result<()> {
    if endpoint_count == 0 {
        bail!("{failure}; no enabled plugin completed a compatible inference endpoint handshake");
    }
    Ok(())
}

/// Shows the warnings about detected GPUs the loaded runtime leaves unused;
/// called once the output surface exists.
pub(crate) fn emit_unused_gpu_warnings() {
    #[cfg(feature = "dynamic-native-runtime")]
    unused_gpu::emit_recorded();
}

pub(crate) async fn confirm_external_provider(manager: &plugin::PluginManager) -> Result<()> {
    let failure = EXTERNAL_STARTUP_FAILURE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take();
    let Some(failure) = failure else {
        return Ok(());
    };
    require_external_inference_provider(&failure, manager.inference_endpoints().await?.len())?;
    let _ = mesh_llm_events::emit_event(mesh_llm_events::OutputEvent::Warning {
        message: format!(
            "Serving external plugin inference without a native runtime. Local model and split-stage loading are disabled. Run `mesh-llm runtime install` and restart to enable them. Runtime resolution: {failure}"
        ),
        context: None,
    });
    Ok(())
}
