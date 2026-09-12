use super::EmbeddedMeshNodeMode;
use anyhow::Result;
use std::path::Path;

pub(super) fn prepare_embedded_native_runtime(
    mode: &EmbeddedMeshNodeMode,
    config_path: Option<&Path>,
) -> Result<()> {
    #[cfg(feature = "dynamic-native-runtime")]
    {
        if *mode != EmbeddedMeshNodeMode::Serve || skippy_runtime::native_runtime_loaded() {
            return Ok(());
        }
        let cache = crate::system::native_runtime_install::default_native_runtime_cache()?;
        let skippy_abi = crate::system::native_runtime_install::current_skippy_abi_version();
        let requirement = EmbeddedNativeRuntimeRequirement {
            mesh_version: crate::RELEASE_VERSION,
            skippy_abi: &skippy_abi,
            cache_root: cache.root(),
        };
        let config = crate::plugin::load_config(config_path)?;
        let selection = embedded_native_runtime_selection(&config.runtime.native_runtime)?;
        let loaded = crate::system::native_runtime::load_local_native_runtime_for_embedded_serving(
            &selection,
        )?
        .is_some()
            || skippy_runtime::native_runtime_loaded();
        ensure_embedded_native_runtime_ready(mode, loaded, requirement)?;
    }
    #[cfg(not(feature = "dynamic-native-runtime"))]
    {
        let _ = (mode, config_path);
    }
    Ok(())
}

#[cfg(any(feature = "dynamic-native-runtime", test))]
fn embedded_native_runtime_selection(
    config: &mesh_llm_config::NativeRuntimeConfig,
) -> Result<mesh_llm_native_runtime::RuntimeSelection> {
    mesh_llm_native_runtime::RuntimeSelection::parse(config.selection.as_deref())
}

#[cfg(any(feature = "dynamic-native-runtime", test))]
struct EmbeddedNativeRuntimeRequirement<'a> {
    mesh_version: &'a str,
    skippy_abi: &'a str,
    cache_root: &'a Path,
}

#[cfg(any(feature = "dynamic-native-runtime", test))]
fn ensure_embedded_native_runtime_ready(
    mode: &EmbeddedMeshNodeMode,
    loaded: bool,
    requirement: EmbeddedNativeRuntimeRequirement<'_>,
) -> Result<()> {
    if *mode != EmbeddedMeshNodeMode::Serve || loaded {
        return Ok(());
    }
    anyhow::bail!(missing_native_runtime_message(requirement));
}

#[cfg(any(feature = "dynamic-native-runtime", test))]
fn missing_native_runtime_message(requirement: EmbeddedNativeRuntimeRequirement<'_>) -> String {
    format!(
        "embedded serving requires a compatible MeshLLM native runtime for MeshLLM {} / Skippy ABI {}, but none is loaded, packaged beside the host, or installed in {}; install it explicitly with `mesh_llm_sdk::native_runtime::install_native_runtime(NativeRuntimeInstallOptions {{ mesh_version: CURRENT_MESH_VERSION.to_string(), skippy_abi_version: Some(current_skippy_abi_version()), ..Default::default() }})`, then retry embedded serving (embedded startup never downloads native runtimes automatically)",
        requirement.mesh_version,
        requirement.skippy_abi,
        requirement.cache_root.display()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn requirement() -> EmbeddedNativeRuntimeRequirement<'static> {
        EmbeddedNativeRuntimeRequirement {
            mesh_version: "0.72.1",
            skippy_abi: "0.1.26",
            cache_root: Path::new("/cache/mesh-llm/native-runtimes"),
        }
    }

    #[test]
    fn missing_embedded_serve_runtime_error_names_versions_cache_and_fix() {
        let error = ensure_embedded_native_runtime_ready(
            &EmbeddedMeshNodeMode::Serve,
            false,
            requirement(),
        )
        .expect_err("missing native runtime should fail before embedded serving starts");
        let message = error.to_string();

        assert!(message.contains("MeshLLM 0.72.1 / Skippy ABI 0.1.26"));
        assert!(message.contains("/cache/mesh-llm/native-runtimes"));
        assert!(message.contains("install_native_runtime"));
        assert!(message.contains("CURRENT_MESH_VERSION"));
        assert!(message.contains("embedded startup never downloads"));
    }

    #[test]
    fn embedded_client_does_not_require_native_serving_runtime() {
        ensure_embedded_native_runtime_ready(&EmbeddedMeshNodeMode::Client, false, requirement())
            .expect("client-only embedding should not require a serving runtime");
    }

    #[test]
    fn shared_endpoint_never_requires_a_native_runtime() {
        let mode = EmbeddedMeshNodeMode::SharedEndpoint {
            address: "http://localhost:11434".to_string(),
        };
        prepare_embedded_native_runtime(&mode).unwrap();
        ensure_embedded_native_runtime_ready(&mode, false, requirement()).unwrap();
    }

    #[test]
    fn loaded_native_runtime_allows_embedded_serve() {
        ensure_embedded_native_runtime_ready(&EmbeddedMeshNodeMode::Serve, true, requirement())
            .expect("loaded runtime should allow embedded serving");
    }

    #[test]
    fn embedded_serving_uses_configured_runtime_selection() {
        let selection = embedded_native_runtime_selection(&mesh_llm_config::NativeRuntimeConfig {
            selection: Some("vulkan".to_string()),
            ..Default::default()
        })
        .expect("configured selection should parse");

        assert_eq!(
            selection,
            mesh_llm_native_runtime::RuntimeSelection::Backend {
                kind: mesh_llm_native_runtime::NativeRuntimeBackendKind::Vulkan,
                cuda_toolkit_major: None,
            }
        );
    }

    #[test]
    fn embedded_serving_defaults_to_recommended_runtime_selection() {
        assert_eq!(
            embedded_native_runtime_selection(&mesh_llm_config::NativeRuntimeConfig::default())
                .expect("default selection should parse"),
            mesh_llm_native_runtime::RuntimeSelection::Recommended
        );
    }
}
