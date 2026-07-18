use crate::command::{DynResult, ensure_eq, sourced_script_stdout};
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct FixtureRow {
    pub(crate) os: String,
    pub(crate) arch: String,
    pub(crate) flavor: String,
    pub(crate) support: String,
    pub(crate) stable_asset: Option<String>,
    pub(crate) versioned_asset: Option<String>,
}

pub(crate) fn fixture_rows(repo_root: &Path) -> DynResult<Vec<FixtureRow>> {
    let fixture_path = fixture_path(repo_root);
    let contents = fs::read_to_string(&fixture_path)?;
    Ok(serde_json::from_str(&contents)?)
}

pub(crate) fn fixture_path(repo_root: &Path) -> PathBuf {
    repo_root
        .join("crates")
        .join("mesh-llm-system")
        .join("tests")
        .join("fixtures")
        .join("release-target-matrix.json")
}

pub(crate) fn fixture_release_tag(rows: &[FixtureRow]) -> DynResult<String> {
    for row in rows {
        let (Some(stable), Some(versioned)) = (&row.stable_asset, &row.versioned_asset) else {
            continue;
        };

        let stable_tail = stable
            .strip_prefix("mesh-llm-")
            .ok_or("stable asset missing mesh-llm- prefix")?;
        let versioned_tail = versioned
            .strip_prefix("mesh-llm-")
            .ok_or("versioned asset missing mesh-llm- prefix")?;
        let suffix = format!("-{stable_tail}");
        if let Some(version) = versioned_tail.strip_suffix(&suffix) {
            return Ok(version.to_string());
        }
    }

    Err("could not derive fixture release tag".into())
}

pub(crate) fn fixture_row<'a>(
    rows: &'a [FixtureRow],
    os: &str,
    arch: &str,
    flavor: &str,
) -> DynResult<&'a FixtureRow> {
    rows.iter()
        .find(|row| row.os == os && row.arch == arch && row.flavor == flavor)
        .ok_or_else(|| format!("missing fixture row for {os}/{arch}/{flavor}").into())
}

pub(crate) fn check_installer_outcomes(repo_root: &Path, rows: &[FixtureRow]) -> DynResult<()> {
    let linux_arm64_asset = fixture_row(rows, "linux", "aarch64", "cpu")?
        .stable_asset
        .clone()
        .ok_or("linux/aarch64/cpu stable asset missing")?;
    let linux_arm64_cuda_asset = fixture_row(rows, "linux", "aarch64", "cuda")?
        .stable_asset
        .clone()
        .ok_or("linux/aarch64/cuda stable asset missing")?;
    let macos_arm64_asset = fixture_row(rows, "macos", "aarch64", "metal")?
        .stable_asset
        .clone()
        .ok_or("macos/aarch64/metal stable asset missing")?;

    let cases = [
        InstallerCase {
            raw_os: "Linux",
            raw_arch: "arm64",
            flavor: "cpu",
            expected_platform: "Linux/aarch64",
            expected_supported_flavors: "cuda cpu",
            expected_asset: linux_arm64_asset.as_str(),
            label: "Linux/arm64",
        },
        InstallerCase {
            raw_os: "Linux",
            raw_arch: "aarch64",
            flavor: "cpu",
            expected_platform: "Linux/aarch64",
            expected_supported_flavors: "cuda cpu",
            expected_asset: linux_arm64_asset.as_str(),
            label: "Linux/aarch64",
        },
        InstallerCase {
            raw_os: "Darwin",
            raw_arch: "arm64",
            flavor: "metal",
            expected_platform: "Darwin/arm64",
            expected_supported_flavors: "metal",
            expected_asset: macos_arm64_asset.as_str(),
            label: "Darwin/arm64",
        },
    ];

    for case in cases {
        let envs = [
            ("MESH_LLM_TEST_UNAME_S", case.raw_os),
            ("MESH_LLM_TEST_UNAME_M", case.raw_arch),
        ];
        let actual_platform =
            sourced_script_stdout(repo_root, "install.sh", "platform_id", &envs, &[])?;
        ensure_eq(
            case.expected_platform,
            &actual_platform,
            &format!("{} normalized platform", case.label),
        )?;

        let actual_supported_flavors =
            sourced_script_stdout(repo_root, "install.sh", "supported_flavors", &envs, &[])?;
        ensure_eq(
            case.expected_supported_flavors,
            &actual_supported_flavors,
            &format!("{} supported flavors", case.label),
        )?;

        let actual_asset = sourced_script_stdout(
            repo_root,
            "install.sh",
            "asset_name \"$2\"",
            &envs,
            &[case.flavor],
        )?;
        ensure_eq(
            case.expected_asset,
            &actual_asset,
            &format!("{} asset parity", case.label),
        )?;
    }

    let orin_envs = [
        ("MESH_LLM_TEST_UNAME_S", "Linux"),
        ("MESH_LLM_TEST_UNAME_M", "aarch64"),
        ("MESH_LLM_TEST_TEGRA_MODEL", "NVIDIA Jetson AGX Orin"),
    ];
    let recommended = sourced_script_stdout(
        repo_root,
        "install.sh",
        "recommended_flavor",
        &orin_envs,
        &[],
    )?;
    ensure_eq(
        "cuda",
        &recommended,
        "Linux/aarch64 Orin recommended flavor",
    )?;
    let actual_cuda_asset = sourced_script_stdout(
        repo_root,
        "install.sh",
        "asset_name \"$2\"",
        &orin_envs,
        &["cuda"],
    )?;
    ensure_eq(
        linux_arm64_cuda_asset.as_str(),
        &actual_cuda_asset,
        "Linux/aarch64 Orin CUDA asset parity",
    )?;

    let arm_fixture = fixture_row(rows, "linux", "arm", "cpu")?;
    let arm_envs = [
        ("MESH_LLM_TEST_UNAME_S", "Linux"),
        ("MESH_LLM_TEST_UNAME_M", "armv7l"),
    ];
    let actual_support = sourced_script_stdout(
        repo_root,
        "install.sh",
        "platform_support_status",
        &arm_envs,
        &[],
    )?;
    ensure_eq(
        &arm_fixture.support,
        &actual_support,
        "Linux/armv7l installer support classification",
    )?;
    let actual_message = sourced_script_stdout(
        repo_root,
        "install.sh",
        "platform_error_message",
        &arm_envs,
        &[],
    )?;
    ensure_eq(
        "error: recognized but unsupported platform: Linux/arm (32-bit ARM release bundles are not published)",
        &actual_message,
        "Linux/armv7l installer error",
    )?;

    Ok(())
}

struct InstallerCase<'a> {
    raw_os: &'a str,
    raw_arch: &'a str,
    flavor: &'a str,
    expected_platform: &'a str,
    expected_supported_flavors: &'a str,
    expected_asset: &'a str,
    label: &'a str,
}
