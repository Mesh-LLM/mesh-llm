//! Windows host/runtime adapter contracts, with real Windows path execution separately owned.
use std::{
    fs,
    path::{Path, PathBuf},
};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}
fn source(path: &str) -> String {
    fs::read_to_string(root().join(path))
        .unwrap()
        .replace("\r\n", "\n")
}
fn script() -> String {
    source("scripts/build-windows.ps1")
}
fn function(source: &str, name: &str) -> String {
    let anchor = format!("function {name} {{\n");
    let start = source
        .find(&anchor)
        .unwrap_or_else(|| panic!("missing {name}"));
    let end = source[start..].find("\n}\n").expect("function boundary");
    source[start..start + end + 3].into()
}
fn ui_entry(action: &str) -> bool {
    action.contains("$uiEntryPoint = Join-Path $uiDist \"index.html\"")
        && action.contains("Test-Path -LiteralPath $uiEntryPoint -PathType Leaf")
        && !action.contains("Get-ChildItem -LiteralPath $uiDist")
}
#[test]
fn windows_ui_fallback_requires_an_index_file_even_when_other_assets_exist() {
    let action = source(".github/actions/prepare-windows-host-input/action.yml");
    assert!(ui_entry(&action));
    assert!(!ui_entry(&action.replace(
        "Test-Path -LiteralPath $uiEntryPoint -PathType Leaf",
        "Test-Path -LiteralPath $uiDist"
    )));
    assert!(!ui_entry(
        &(action + "\nGet-ChildItem -LiteralPath $uiDist\n")
    ));
}
#[test]
fn windows_build_preserves_dynamic_host_and_shared_runtime_ownership() {
    let script = script();
    let features = function(&script, "Get-HostFeatureList");
    assert!(features.contains("web-ui,dynamic-native-runtime,payments"));
    assert!(script.contains("\"-DBUILD_SHARED_LIBS=ON\""));
    assert!(!script.contains("[switch]$AbiOnly"));
}
#[test]
fn windows_prepare_uses_the_shared_queue_and_restores_workdir_after_failure() {
    let prepare = function(&script(), "Prepare-Llama");
    assert!(
        prepare.contains("Invoke-NativeCommand \"bash\" @(\"scripts/prepare-llama.sh\", $mode)")
    );
    assert!(!prepare.contains("Get-ChildItem -Path $patchDir"));
    assert!(!prepare.contains("\"am\","));
    assert!(prepare.contains("finally {"));
    assert!(prepare.contains("$env:LLAMA_WORKDIR = $previousWorkdir"));
    assert!(prepare.contains("Remove-Item Env:LLAMA_WORKDIR -ErrorAction SilentlyContinue"));
    assert!(!prepare.contains("$env:LLAMA_WORKDIR = $llamaDir.Replace("));
}
fn prepare_before_configure(script: &str) -> bool {
    let Some((_, build)) = script.rsplit_once("Invoke-InRepo {\n    Prepare-Llama\n") else {
        return false;
    };
    let resolve = build.find("$script:buildDir = Resolve-StageBuildDir $backendName");
    let configure = build.find("\"-B\", $buildDir,");
    matches!((resolve, configure), (Some(resolve), Some(configure)) if resolve < configure)
}
#[test]
fn windows_prepares_the_pin_before_resolving_the_directory_used_by_cmake() {
    let script = script();
    assert!(prepare_before_configure(&script));
    assert!(
        function(&script, "Resolve-StageBuildDir")
            .contains("Join-Path $llamaDir \".mesh-llm-patched-sha\"")
    );
    assert!(!prepare_before_configure(&script.replace(
        "$script:buildDir = Resolve-StageBuildDir $backendName",
        "$script:buildDir = $buildDir"
    )));
    assert!(!prepare_before_configure(&script.replace(
        "Invoke-InRepo {\n    Prepare-Llama\n",
        "Invoke-InRepo {\n"
    )));
}
#[test]
fn windows_product_is_a_verified_composer_without_a_host_or_runtime_build() {
    super::check_windows_dynamic_runtime_contract(
        &source(".github/workflows/ci-windows-host-slice.yml"),
        &(source(".github/workflows/ci-windows-runtime-slice.yml")
            + &source(".github/workflows/ci-windows-product-slice.yml")),
        &source(".github/actions/prepare-windows-host-input/action.yml"),
        &source(".github/actions/prepare-native-runtime-input/action.yml"),
        &source(".github/actions/compose-product-input/action.yml"),
    )
    .unwrap();
    assert!(
        source(".github/workflows/ci-windows-runtime-slice.yml")
            .contains("restore-windows-abi-cache")
    );
}

#[cfg_attr(not(windows), allow(dead_code))]
#[path = "windows_build_native.rs"]
mod native;

#[path = "windows_installer/mod.rs"]
mod installer;

#[path = "windows_installer_static_tests.rs"]
mod installer_static;

#[cfg(windows)]
#[path = "windows_with_lld.rs"]
mod with_lld;
