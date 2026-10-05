//! Draft source contracts for the maintained consumer PowerShell adapter.
//! Register under windows_runtime_tests; these do not execute Windows installation.
use super::{function, source};

fn has_all(body: &str, tokens: &[&str]) -> bool {
    tokens.iter().all(|token| body.contains(token))
}
fn assert_required_tokens(body: &str, tokens: &[&str]) {
    assert!(has_all(body, tokens));
    for token in tokens {
        assert!(
            !has_all(&body.replace(token, "missing-contract"), tokens),
            "{token}"
        );
    }
}
#[test]
fn windows_installer_has_no_runtime_selection_or_service_policy() {
    let body = source("install.ps1");
    for token in ["runtime install", "runtime prune", "New-Service", "sc.exe"] {
        assert!(!body.contains(token), "{token}");
    }
}
#[test]
fn windows_installer_keeps_neutral_host_asset_and_setup_handoff() {
    let body = source("install.ps1");
    assert_required_tokens(
        &body,
        &[
            "mesh-llm-x86_64-pc-windows-msvc.zip",
            "[switch]$NoSetup",
            "Run this next:",
            "mesh-llm.exe setup",
            "Legacy compatibility flag",
        ],
    );
    for token in [
        "Get-RecommendedFlavor",
        "Choose-Flavor",
        "native-runtimes.json",
    ] {
        assert!(!body.contains(token), "{token}");
    }
}
#[test]
fn windows_installer_architecture_fallback_handles_null_runtime_information() {
    let body = function(&source("install.ps1"), "Require-WindowsX64");
    assert!(!body.contains("::OSArchitecture.ToString()"));
    assert_required_tokens(&body, &["$env:PROCESSOR_ARCHITECTURE"]);
}
#[test]
fn windows_installer_retains_product_integrity_and_legacy_version_boundary() {
    let body = source("install.ps1");
    assert_required_tokens(
        &body,
        &[
            "Assert-ProductBundle",
            "mesh-llm-product-v2",
            "runtime.manifest_sha256",
            "Get-DeterministicTreeSha256",
            "Assert-SafeRelativePath",
            "$ComposedProductMinVersion = [System.Version]::Parse(\"0.75.0\")",
            "Installing supported legacy MeshLLM",
            "requires product-manifest.json and native-runtimes",
        ],
    );
}
#[test]
fn windows_installer_tree_digest_normalizes_both_path_separator_forms() {
    let body = function(&source("install.ps1"), "Get-DeterministicTreeSha256");
    assert_required_tokens(
        &body,
        &[
            "[System.IO.Path]::DirectorySeparatorChar",
            "[System.IO.Path]::AltDirectorySeparatorChar",
            "TrimStart($pathSeparators)",
        ],
    );
}
#[test]
fn windows_installer_replacement_owns_staging_backups_and_stale_import_removal() {
    let body = function(&source("install.ps1"), "Install-MeshBinary");
    assert_required_tokens(
        &body,
        &[
            "mesh-llm.exe.incoming",
            "native-runtimes.incoming",
            "Stage-IncomingBundle",
            "Restore-InstallBackup",
            "host-imports.json.backup",
            "Remove-Item $hostImportsDestination -Force",
        ],
    );
}
fn before(body: &str, first: &str, second: &str) -> bool {
    matches!((body.find(first),body.find(second)),(Some(a),Some(b)) if a < b)
}
#[test]
fn windows_installer_stages_before_product_mutation_and_cleans_before_rollback() {
    let whole = function(&source("install.ps1"), "Install-MeshBinary");
    let (_, body) = whole
        .split_once("$paths = [PSCustomObject]")
        .expect("product branch");
    for (first, second) in [
        ("Stage-IncomingBundle", "Move-IfExists"),
        ("Stage-IncomingBundle", "Remove-StaleBinaries"),
        (
            "Remove-InstallStagingPath -Path $paths.MeshBinaryStaging",
            "Restore-InstallBackup",
        ),
        (
            "Remove-InstallBackups -Paths $paths",
            "Remove-StaleBinaries",
        ),
    ] {
        assert!(before(body, first, second));
        assert!(!before(
            &body.replace(first, "missing-contract"),
            first,
            second
        ));
    }
}
