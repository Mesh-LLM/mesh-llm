use std::fs;
use std::path::Path;

fn source(name: &str) -> String {
    fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../scripts")
            .join(name),
    )
    .unwrap()
}

fn function<'a>(source: &'a str, name: &str, end: &str) -> &'a str {
    let start = source.find(&format!("function {name}")).unwrap();
    let finish = source[start..].find(end).unwrap() + start;
    &source[start..finish]
}

#[test]
fn powershell_precomposed_tree_retains_verification_before_archive() {
    let source = source("package-release.ps1");
    let body = function(
        &source,
        "Copy-AndVerifyPrecomposedProduct",
        "$Version = Normalize-RecipeArgument",
    );
    for required in [
        "$resolvedSourceDir = Resolve-RepositoryPath $SourceDir",
        "[System.IO.Directory]::GetFileSystemEntries($resolvedSourceDir)",
        "Invoke-Automation native verify-host-dependencies",
        "Invoke-Automation native verify-runtime-package",
        "Invoke-Automation product compose",
        "--check",
    ] {
        assert!(body.contains(required), "{required}");
    }
}

#[test]
fn powershell_composer_paths_accept_git_windows_boundary() {
    let source = source("package-release.ps1");
    let body = function(
        &source,
        "Resolve-RepositoryPath",
        "function Assert-AttestationConfig",
    );
    assert!(body.contains(r"^/(?<drive>[A-Za-z])(?:/(?<tail>.*))?$"));
    assert!(body.contains(r#"GetFullPath("${drive}:\${tail}")"#));
}

#[test]
fn powershell_preverified_attestation_requires_immutable_composer() {
    let source = source("package-release.ps1");
    let config = function(
        &source,
        "Assert-AttestationConfig",
        "function Invoke-ReleaseAttestationStamp",
    );
    assert!(config.contains(r#"$env:MESH_RELEASE_HOST_PRESTAMPED -ne "1""#));
    assert!(config.contains("-not (Test-HasValue $precomposedProductDir)"));
    let body = function(
        &source,
        "Invoke-ReleaseAttestationStamp",
        "function Copy-AndVerifyPrecomposedProduct",
    );
    assert!(
        body.find(r#"if ($attestationPreverified -eq "1")"#)
            .unwrap()
            < body.find("release-attestation inspect").unwrap()
    );
}

#[test]
fn powershell_runtime_selection_keeps_token_array_and_checked_output() {
    let source = source("package-release.ps1");
    let start = source.find("$cudaMajor =").unwrap();
    let end = source[start..].find("$runtimeDestinationRoot =").unwrap() + start;
    let body = &source[start..end];
    for required in [
        "$selectorArgs = @(",
        "if (Test-HasValue $cudaMajor)",
        r#"$selectorArgs += @("--cuda-major", $cudaMajor)"#,
        "Invoke-Automation @selectorArgs",
        "if ($selectorExitCode -ne 0)",
        "ForEach-Object { $_.Trim() }",
        "Select-Object -Last 1",
    ] {
        assert!(body.contains(required), "{required}");
    }
    assert!(
        body.find("$selectorExitCode = $LASTEXITCODE").unwrap()
            < body.find("$runtimeDir =").unwrap()
    );
}

#[test]
fn powershell_prestamped_host_requires_checksum_verified_inspection() {
    let source = source("package-release.ps1");
    let body = function(
        &source,
        "Invoke-ReleaseAttestationStamp",
        "function Copy-AndVerifyPrecomposedProduct",
    );
    assert!(body.contains(r#"Assert-FileChecksum -Path $attestationVerifier -ChecksumPath "${attestationVerifier}.sha256""#));
    assert!(body.contains("& $attestationVerifier release-attestation inspect"));
    assert!(body.contains("Invoke-Automation release-attestation inspect"));
    assert!(body.contains(r#"if ($inspectStatus -ne "valid")"#));
}

#[test]
fn rc_adapter_keeps_install_inference_and_scoped_cleanup_contracts() {
    let source = source("rc-release-smoke.sh");
    for required in [
        "runtime list --available --json",
        "runtime install --json",
        "download \"$MODEL_REF\"",
        "--log-format json",
        "product rc-ok request",
        "product rc-ok verify",
        "trap cleanup EXIT",
        "kill_mesh_bundle_processes",
        "wait_for_no_mesh_bundle_processes 10",
    ] {
        assert!(source.contains(required), "{required}");
    }
}

#[test]
fn rc_adapter_remains_valid_bash() {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/rc-release-smoke.sh");
    assert!(
        std::process::Command::new("bash")
            .arg("-n")
            .arg(script)
            .status()
            .unwrap()
            .success()
    );
}
