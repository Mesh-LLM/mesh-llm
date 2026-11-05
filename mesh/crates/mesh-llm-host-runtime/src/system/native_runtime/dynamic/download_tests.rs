use super::*;
use std::sync::{Arc, Mutex};

fn linux_host_profile() -> HostRuntimeProfile {
    HostRuntimeProfile {
        os: "linux".to_string(),
        arch: "x86_64".to_string(),
        target_triple: None,
        glibc_version: Some("2.39".to_string()),
        available_flavors: [skippy_native_runtime::NativeRuntimeBackendKind::Cpu].into(),
        gpus: Vec::new(),
        cuda: None,
        rocm: None,
        vulkan: None,
    }
}

/// Starts up with an empty cache and returns the `allow_download` the
/// installer was asked for, and the startup error.
async fn installer_allow_download(config: mesh_llm_config::NativeRuntimeConfig) -> (bool, String) {
    let temp = tempfile::tempdir().unwrap();
    let cache = NativeRuntimeCache::new(temp.path().join("cache"));
    let selection = NativeRuntimeStartupSelection::from_config(config, None).unwrap();
    let requested = Arc::new(Mutex::new(None));
    let error = resolve_startup_native_runtime_plan_with(
        || Ok(cache.clone()),
        linux_host_profile,
        crate::system::native_runtime_install::mesh_native_runtime_install_options,
        {
            let requested = Arc::clone(&requested);
            move |options: NativeRuntimeInstallOptions| {
                *requested.lock().unwrap() = Some(options.allow_download);
                async { anyhow::bail!("no runtime in this test") }
            }
        },
        selection,
    )
    .await
    .unwrap_err();
    let allow_download = requested
        .lock()
        .unwrap()
        .expect("the installer must be asked");
    (allow_download, format!("{error:#}"))
}

#[tokio::test]
async fn startup_may_download_by_default() {
    skippy_runtime_install::rerun_without_bundle_env!(startup_may_download_by_default);
    let (allow_download, error) =
        installer_allow_download(mesh_llm_config::NativeRuntimeConfig::default()).await;
    assert!(allow_download);
    assert!(!error.contains("runtime.native_runtime.download"));
}

#[tokio::test]
async fn download_off_keeps_startup_from_downloading() {
    skippy_runtime_install::rerun_without_bundle_env!(download_off_keeps_startup_from_downloading);
    for config in [
        mesh_llm_config::NativeRuntimeConfig {
            download: Some(false),
            ..Default::default()
        },
        mesh_llm_config::NativeRuntimeConfig {
            mesh_version: Some("0.68.0".to_string()),
            download_override: Some(false),
            ..Default::default()
        },
    ] {
        let (allow_download, error) = installer_allow_download(config).await;
        assert!(!allow_download);
        assert!(
            error.contains("runtime.native_runtime.download"),
            "the error must say why nothing was downloaded: {error}"
        );
    }
}
