use crate::command::{
    DynResult, display_relative, ensure_eq, ensure_eq_option, ensure_status, manifest_section,
    package_section_uses_workspace_version, run_command, sourced_script_stdout,
    trimmed_stderr_or_stdout, unique_temp_dir,
};
use crate::installer_fixtures::{
    FixtureRow, check_installer_outcomes, fixture_release_tag, fixture_row, fixture_rows,
};
use crate::publish_consistency::check_publish_crates_consistency;
use crate::repo_consistency::{
    check_attestation_default_version, host_supports_shell_parity_checks, repo_root,
    workspace_metadata,
};
use crate::workflow_checks::{
    check_ci_script_workspace_members, check_docs_and_workflow_invariants,
};
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
use std::process::Command;

pub(crate) fn check_release_targets() -> DynResult<()> {
    let repo_root = repo_root()?;
    let fixture_rows = fixture_rows(&repo_root)?;
    let fixture_version = fixture_release_tag(&fixture_rows)?;

    if host_supports_shell_parity_checks() {
        check_installer_outcomes(&repo_root, &fixture_rows)?;
        check_package_release_assets(&repo_root, &fixture_rows, &fixture_version)?;
    } else {
        println!(
            "note: skipping bash-dependent release parity checks on native Windows; run `just check-release` on macOS/Linux for install.sh and package-release.sh parity"
        );
    }
    check_windows_name_invariance(&fixture_rows, &fixture_version)?;
    check_ci_script_workspace_members(&repo_root)?;
    check_workspace_package_versions(&repo_root)?;
    check_workspace_internal_dependency_versions(&repo_root)?;
    check_attestation_default_version(&repo_root)?;
    check_publish_crates_consistency(&repo_root)?;
    check_docs_and_workflow_invariants(&repo_root)?;

    println!("repo consistency checks passed: release-targets");
    Ok(())
}

fn check_workspace_package_versions(repo_root: &Path) -> DynResult<()> {
    let metadata = workspace_metadata(repo_root, "workspace package version ownership")?;
    let workspace_members = metadata
        .workspace_members
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();

    for package in metadata.packages {
        if !workspace_members.contains(&package.id) {
            continue;
        }

        let manifest = fs::read_to_string(&package.manifest_path)?;
        let package_section = manifest_section(&manifest, "package").ok_or_else(|| {
            format!(
                "{}: missing [package] section",
                display_relative(repo_root, &package.manifest_path)
            )
        })?;
        if !package_section_uses_workspace_version(package_section) {
            return Err(format!(
                "{}: workspace member `{}` must use `version.workspace = true`",
                display_relative(repo_root, &package.manifest_path),
                package.name
            )
            .into());
        }
    }

    Ok(())
}

fn check_workspace_internal_dependency_versions(repo_root: &Path) -> DynResult<()> {
    let manifest_path = repo_root.join("Cargo.toml");
    let manifest = fs::read_to_string(&manifest_path)?;
    let Some(workspace_deps) = manifest_section(&manifest, "workspace.dependencies") else {
        return Ok(());
    };

    for line in workspace_deps.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || !trimmed.contains("path = \"crates/") {
            continue;
        }
        if !trimmed.contains("version =") {
            return Err(format!(
                "Cargo.toml [workspace.dependencies]: internal dependency must carry a release-updated version: `{trimmed}`"
            )
            .into());
        }
    }

    Ok(())
}

fn check_package_release_assets(
    repo_root: &Path,
    rows: &[FixtureRow],
    fixture_version: &str,
) -> DynResult<()> {
    for row in rows {
        if row.os != "linux" && row.os != "macos" {
            continue;
        }
        if row.support == "recognized-unsupported" {
            continue;
        }

        for raw_case in raw_targets(row)? {
            let mut envs = vec![
                ("MESH_RELEASE_OS", raw_case.raw_os),
                ("MESH_RELEASE_ARCH", raw_case.raw_arch),
            ];
            if row.flavor != implicit_release_flavor(row) {
                envs.push(("MESH_RELEASE_FLAVOR", row.flavor.as_str()));
            }

            let actual_support = sourced_script_stdout(
                repo_root,
                "scripts/package-release.sh",
                "release_target_support",
                &envs,
                &[],
            )?;
            ensure_eq(
                shell_support(row),
                &actual_support,
                &format!(
                    "{}/{}/{} package support ({})",
                    row.os, row.arch, row.flavor, raw_case.label
                ),
            )?;

            if row.support != "supported" {
                let tmp_output_dir = unique_temp_dir("check-release-unsupported");
                let output = run_command(
                    Command::new("bash")
                        .current_dir(repo_root)
                        .envs(envs.iter().copied())
                        .arg("scripts/package-release.sh")
                        .arg(fixture_version)
                        .arg(&tmp_output_dir),
                );
                let _ = std::fs::remove_dir_all(&tmp_output_dir);
                let output = output?;
                ensure_status(
                    1,
                    output.status.code(),
                    &format!(
                        "{}/{}/{} unsupported packaging exit code ({})",
                        row.os, row.arch, row.flavor, raw_case.label
                    ),
                )?;
                ensure_eq(
                    &unsupported_release_target_message(&raw_case, row),
                    &trimmed_stderr_or_stdout(&output),
                    &format!(
                        "{}/{}/{} unsupported packaging message ({})",
                        row.os, row.arch, row.flavor, raw_case.label
                    ),
                )?;
                continue;
            }

            let actual_stable = sourced_script_stdout(
                repo_root,
                "scripts/package-release.sh",
                "resolve_release_target; printf '%s\\n' \"$STABLE_ASSET\"",
                &envs,
                &[],
            )?;
            ensure_eq_option(
                row.stable_asset.as_deref(),
                Some(actual_stable.as_str()),
                &format!(
                    "{}/{}/{} package stable asset ({})",
                    row.os, row.arch, row.flavor, raw_case.label
                ),
            )?;

            let actual_versioned = sourced_script_stdout(
                repo_root,
                "scripts/package-release.sh",
                "versioned_asset_name \"$2\"",
                &envs,
                &[fixture_version],
            )?;
            ensure_eq_option(
                row.versioned_asset.as_deref(),
                Some(actual_versioned.as_str()),
                &format!(
                    "{}/{}/{} package versioned asset ({})",
                    row.os, row.arch, row.flavor, raw_case.label
                ),
            )?;
        }
    }

    let arm_row = fixture_row(rows, "linux", "arm", "cpu")?;
    ensure_eq(
        "recognized-unsupported",
        &arm_row.support,
        "linux/arm fixture support",
    )?;
    ensure_eq_option(
        None,
        arm_row.stable_asset.as_deref(),
        "linux/arm fixture stable asset",
    )?;
    ensure_eq_option(
        None,
        arm_row.versioned_asset.as_deref(),
        "linux/arm fixture versioned asset",
    )?;

    let tmp_output_dir = unique_temp_dir("check-release");
    let output = run_command(
        Command::new("bash")
            .current_dir(repo_root)
            .env("MESH_RELEASE_OS", "Linux")
            .env("MESH_RELEASE_ARCH", "armv7l")
            .arg("scripts/package-release.sh")
            .arg(fixture_version)
            .arg(&tmp_output_dir),
    );
    // Clean up before propagating any error so the temp dir is always removed.
    let _ = std::fs::remove_dir_all(&tmp_output_dir);
    let output = output?;
    ensure_status(1, output.status.code(), "Linux/armv7l packaging exit code")?;
    let actual_message = trimmed_stderr_or_stdout(&output);
    ensure_eq(
        "Recognized but unsupported release target: Linux/armv7l (normalized: linux/arm)",
        &actual_message,
        "Linux/armv7l packaging error",
    )?;

    Ok(())
}

struct RawTargetCase {
    raw_os: &'static str,
    raw_arch: &'static str,
    label: &'static str,
}

fn raw_targets(row: &FixtureRow) -> DynResult<Vec<RawTargetCase>> {
    match (row.os.as_str(), row.arch.as_str()) {
        ("macos", "aarch64") => Ok(vec![RawTargetCase {
            raw_os: "Darwin",
            raw_arch: "arm64",
            label: "Darwin/arm64",
        }]),
        ("linux", "x86_64") => Ok(vec![RawTargetCase {
            raw_os: "Linux",
            raw_arch: "x86_64",
            label: "Linux/x86_64",
        }]),
        ("linux", "aarch64") => Ok(vec![
            RawTargetCase {
                raw_os: "Linux",
                raw_arch: "arm64",
                label: "Linux/arm64",
            },
            RawTargetCase {
                raw_os: "Linux",
                raw_arch: "aarch64",
                label: "Linux/aarch64",
            },
        ]),
        _ => Err(format!("unsupported raw target mapping for {}/{}", row.os, row.arch).into()),
    }
}

fn implicit_release_flavor(row: &FixtureRow) -> &'static str {
    match (row.os.as_str(), row.arch.as_str()) {
        ("macos", "aarch64") => "metal",
        ("linux", "x86_64") | ("linux", "aarch64") | ("linux", "arm") => "cpu",
        _ => "",
    }
}

fn shell_support(row: &FixtureRow) -> &str {
    match row.support.as_str() {
        "unknown" => "unsupported",
        other => other,
    }
}

fn unsupported_release_target_message(raw_case: &RawTargetCase, row: &FixtureRow) -> String {
    format!(
        "Unsupported release target/flavor for packaging: {}/{} with flavor {} (normalized: {}/{})",
        raw_case.raw_os, raw_case.raw_arch, row.flavor, row.os, row.arch
    )
}

fn check_windows_name_invariance(rows: &[FixtureRow], fixture_version: &str) -> DynResult<()> {
    for row in rows {
        if row.os != "windows" {
            continue;
        }

        ensure_eq(
            "x86_64",
            &row.arch,
            &format!("windows/{}/{}/canonical arch", row.arch, row.flavor),
        )?;
        ensure_eq(
            "supported",
            &row.support,
            &format!("windows/{}/{}/support", row.arch, row.flavor),
        )?;
        let stable_expected = windows_asset_name(&row.flavor, "");
        let versioned_expected = windows_asset_name(&row.flavor, &format!("-{fixture_version}"));
        ensure_eq_option(
            Some(stable_expected.as_str()),
            row.stable_asset.as_deref(),
            &format!("windows/{}/{}/stable asset", row.arch, row.flavor),
        )?;
        ensure_eq_option(
            Some(versioned_expected.as_str()),
            row.versioned_asset.as_deref(),
            &format!("windows/{}/{}/versioned asset", row.arch, row.flavor),
        )?;
    }

    Ok(())
}

fn windows_asset_name(flavor: &str, version_prefix: &str) -> String {
    let suffix = match flavor {
        "cpu" | "metal" => "",
        other => other,
    };

    if suffix.is_empty() {
        format!("mesh-llm{version_prefix}-x86_64-pc-windows-msvc.zip")
    } else {
        format!("mesh-llm{version_prefix}-x86_64-pc-windows-msvc-{suffix}.zip")
    }
}
