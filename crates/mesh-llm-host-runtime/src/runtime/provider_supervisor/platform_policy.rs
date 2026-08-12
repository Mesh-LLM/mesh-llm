use super::*;

#[cfg(target_os = "macos")]
use std::ffi::OsString;

#[cfg(target_os = "macos")]
pub(super) fn validate_provider_platform_policy(runtime: &InstalledProviderRuntime) -> Result<()> {
    if runtime.manifest.runtime.provider_kind != APPLE_PROVIDER_KIND {
        return Ok(());
    }
    let executable = runtime.entrypoint();
    run_policy_command(
        "codesign",
        &[
            OsString::from("--verify"),
            OsString::from("--strict"),
            executable.clone().into(),
        ],
        "verify Apple provider code signature",
    )?;
    let details = run_policy_command(
        "codesign",
        &[
            OsString::from("-dv"),
            OsString::from("--verbose=4"),
            executable.clone().into(),
        ],
        "inspect Apple provider code signature",
    )?;
    let team_identifier = signing_detail(&details, "TeamIdentifier");
    let signing_identifier = signing_detail(&details, "Identifier");
    let is_ad_hoc = team_identifier
        .as_deref()
        .is_none_or(|team| team == "not set");
    if is_ad_hoc && !environment_flag("MESH_LLM_APPLE_PROVIDER_ALLOW_AD_HOC") {
        bail!(
            "Apple provider is ad-hoc signed; set MESH_LLM_APPLE_PROVIDER_ALLOW_AD_HOC=1 only for local experimental builds"
        );
    }
    if let Some(signature) = &runtime.manifest.runtime.signature {
        compare_signing_detail(
            "team identifier",
            signature.team_identifier.as_deref(),
            team_identifier.as_deref(),
        )?;
        compare_signing_detail(
            "signing identifier",
            signature.signing_identifier.as_deref(),
            signing_identifier.as_deref(),
        )?;
        validate_declared_entitlements(&executable, &signature.entitlements)?;
        if signature.notarized == Some(true) {
            run_policy_command(
                "spctl",
                &[
                    OsString::from("--assess"),
                    OsString::from("--type"),
                    OsString::from("execute"),
                    executable.into(),
                ],
                "assess Apple provider notarization",
            )?;
        }
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
pub(super) fn validate_provider_platform_policy(runtime: &InstalledProviderRuntime) -> Result<()> {
    if runtime.manifest.runtime.provider_kind == APPLE_PROVIDER_KIND {
        bail!("Apple provider runtimes may only be launched on macOS");
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn run_policy_command(program: &str, arguments: &[OsString], label: &str) -> Result<String> {
    let output = std::process::Command::new(program)
        .args(arguments)
        .output()
        .with_context(|| label.to_string())?;
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    if !output.status.success() {
        bail!("{label} failed: {}", combined.trim());
    }
    Ok(combined)
}

#[cfg(target_os = "macos")]
fn signing_detail(details: &str, name: &str) -> Option<String> {
    details.lines().find_map(|line| {
        line.strip_prefix(&format!("{name}="))
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    })
}

#[cfg(target_os = "macos")]
fn compare_signing_detail(label: &str, expected: Option<&str>, actual: Option<&str>) -> Result<()> {
    if let Some(expected) = expected
        && actual != Some(expected)
    {
        bail!(
            "Apple provider {label} mismatch: expected {expected}, got {}",
            actual.unwrap_or("missing")
        );
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn validate_declared_entitlements(executable: &Path, declared: &[String]) -> Result<()> {
    if declared.is_empty() {
        return Ok(());
    }
    let output = run_policy_command(
        "codesign",
        &[
            OsString::from("-d"),
            OsString::from("--entitlements"),
            OsString::from(":-"),
            executable.to_path_buf().into(),
        ],
        "inspect Apple provider entitlements",
    )?;
    for entitlement in declared {
        if !output.contains(&format!("<key>{entitlement}</key>")) {
            bail!("Apple provider is missing declared entitlement {entitlement}");
        }
    }
    Ok(())
}
