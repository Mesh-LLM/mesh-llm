#![cfg(feature = "serving")]

use mesh_llm_sdk::native_runtime::{
    CURRENT_MESH_VERSION, current_skippy_abi_version, native_runtime_versions_match_current_sdk,
};

#[test]
fn documented_version_check_tracks_sdk_release_and_skippy_abi() {
    let required_abi = current_skippy_abi_version();
    assert!(native_runtime_versions_match_current_sdk(
        CURRENT_MESH_VERSION,
        &required_abi
    ));
    assert!(!native_runtime_versions_match_current_sdk(
        "previous-sdk-release",
        &required_abi
    ));
    assert!(!native_runtime_versions_match_current_sdk(
        CURRENT_MESH_VERSION,
        "previous-skippy-abi"
    ));
}

#[test]
fn readme_keeps_explicit_check_install_initialize_start_order() {
    let readme = include_str!("../README.md");
    let check = readme
        .find("native_runtime_versions_match_current_sdk")
        .expect("README should check cached runtime versions");
    let install = readme[check..]
        .find("install_native_runtime(NativeRuntimeInstallOptions")
        .map(|offset| check + offset)
        .expect("README should explicitly install a missing runtime");
    let initialize = readme[install..]
        .find("initialize_host_runtime().await")
        .map(|offset| install + offset)
        .expect("README should initialize the installed runtime");
    let start = readme[initialize..]
        .find("MeshNode::builder().serve().start().await")
        .map(|offset| initialize + offset)
        .expect("README should start embedded serving after initialization");

    assert!(check < install && install < initialize && initialize < start);
    assert!(!readme.contains("cache.installed()?.into_iter().find"));
    assert!(readme.contains("resolves the\nrecommended runtime against the current host profile"));
    assert!(readme.contains("runtime.load_plan()?"));
    assert!(readme.contains("startup only loads a compatible cached\nruntime"));
    assert!(readme.contains("it never downloads one"));
}
