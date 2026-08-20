use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};

use crate::backend;
use crate::release_target::ReleaseTarget;

mod platform_probe;
mod release_fetch;

pub use release_fetch::{latest_release_version, version_newer};

use platform_probe::{installed_bundle_flavor, preferred_bundle_flavor_for_current_host};
#[cfg(not(windows))]
use release_fetch::INSTALL_SCRIPT_URL;
use release_fetch::{
    InstallOutcome, PostInstallAction, RELEASES_URL, ReleaseAssetPreference,
    current_release_target, describe_requested_update, exec_current_binary, install_latest_bundle,
    latest_release_info, mesh_binary_name, path_is_writable, platform_has_release_assets,
    release_asset_candidates, release_has_any_platform_asset, resolve_release_asset_name,
    resolve_release_info,
};

const SELF_UPDATE_ATTEMPTED_ENV: &str = "MESH_LLM_SELF_UPDATE_ATTEMPTED";

struct UpdateTarget {
    exe: PathBuf,
    install_dir: PathBuf,
    release_target: ReleaseTarget,
    bundle_flavor: backend::BinaryFlavor,
}

#[derive(Clone, Copy, Debug)]
pub struct AutoUpdateOptions {
    pub auto_update: bool,
    pub plugin_requested: bool,
    pub command_is_update: bool,
    pub llama_flavor: Option<backend::BinaryFlavor>,
    pub current_version: &'static str,
}

#[derive(Clone, Copy, Debug)]
pub struct UpdateCommandOptions<'a> {
    pub flavor: Option<backend::BinaryFlavor>,
    pub detect_flavor: bool,
    pub requested_version: Option<&'a str>,
    pub current_version: &'static str,
}

enum NoticeGuidance {
    RunUpdateCommand,
    #[cfg(not(windows))]
    ReinstallScript,
    #[cfg(windows)]
    DownloadReleases,
}

/// A newer-version notice for callers to surface through their own output
/// facility; this crate performs no console I/O of its own.
#[derive(Clone, Debug)]
pub struct UpdateNotice {
    pub message: String,
}

fn notice_message(current_version: &str, latest_version: &str, guidance: NoticeGuidance) -> String {
    match guidance {
        NoticeGuidance::RunUpdateCommand => format!(
            "✨ New version: v{current_version} -> v{latest_version}. Run 'mesh-llm update'."
        ),
        #[cfg(not(windows))]
        NoticeGuidance::ReinstallScript => format!(
            "✨ New version: v{current_version} -> v{latest_version}. Reinstall with: curl -fsSL {} | bash",
            release_fetch::INSTALL_SCRIPT_URL
        ),
        #[cfg(windows)]
        NoticeGuidance::DownloadReleases => format!(
            "✨ New version: v{current_version} -> v{latest_version}. Download from {RELEASES_URL}"
        ),
    }
}

/// Checks for a newer release and returns a notice when one is available.
///
/// Returns `None` when the platform has no release assets, no release could be
/// fetched, or the local version is not older than the latest release. Callers
/// are responsible for presenting [`UpdateNotice::message`] through their own
/// output facility.
pub async fn check_for_update(current_version: &str) -> Option<UpdateNotice> {
    if !platform_has_release_assets() {
        return None;
    }
    let release = latest_release_info().await?;
    if !version_newer(&release.version, current_version) {
        return None;
    }
    // Determine whether this is a bundle install and, if so, whether the
    // specific installed flavor's asset is present in the new release.
    let bundle_asset = std::env::current_exe().ok().and_then(|exe| {
        let (_, flavor) = bundle_install_dir(&exe, None)?;
        current_release_target(flavor).and_then(|target| {
            resolve_release_asset_name(&release, target, ReleaseAssetPreference::StableFirst)
        })
    });
    let has_matching_bundle_asset = bundle_asset
        .as_ref()
        .is_some_and(|asset| release_fetch::release_has_asset(&release, asset));
    let guidance = if has_matching_bundle_asset {
        NoticeGuidance::RunUpdateCommand
    } else {
        // Either not a bundle install, or the installed flavor's asset is not
        // published in the new release — fall back to generic guidance.
        if !release_has_any_platform_asset(&release, std::env::consts::OS, std::env::consts::ARCH) {
            return None;
        }
        #[cfg(not(windows))]
        {
            NoticeGuidance::ReinstallScript
        }
        #[cfg(windows)]
        {
            NoticeGuidance::DownloadReleases
        }
    };
    Some(UpdateNotice {
        message: notice_message(current_version, &release.version, guidance),
    })
}

pub async fn maybe_auto_update(options: AutoUpdateOptions) -> Result<bool> {
    if !should_attempt_auto_update(options) {
        return Ok(false);
    }
    let Some(target) = discover_update_target(options.llama_flavor) else {
        return Ok(false);
    };
    apply_update_if_available(
        target,
        PostInstallAction::RestartCurrentProcess,
        options.current_version,
    )
    .await
}

pub async fn run_update_command(options: UpdateCommandOptions<'_>) -> Result<()> {
    let target = require_update_target(options.flavor, options.detect_flavor)?;
    let requested_version = options.requested_version;
    let Some(release) = resolve_release_info(requested_version).await? else {
        bail!("Could not check for a release right now. Try again shortly.");
    };
    if requested_version.is_none() && !version_newer(&release.version, options.current_version) {
        eprintln!(
            "mesh-llm is already up to date (v{}).",
            options.current_version
        );
        return Ok(());
    }
    let asset_preference = if requested_version.is_some() {
        ReleaseAssetPreference::VersionedFirst
    } else {
        ReleaseAssetPreference::StableFirst
    };
    let Some(asset_name) =
        resolve_release_asset_name(&release, target.release_target, asset_preference)
    else {
        bail!(
            "Release v{} does not include a bundle for this install (tried: {}).",
            release.version,
            release_asset_candidates(target.release_target, &release.tag, asset_preference)
                .join(", ")
        );
    };
    if !path_is_writable(&target.exe) {
        bail!("{} is not writable.", target.exe.display());
    }

    eprintln!(
        "⬇️ {} mesh-llm v{} -> v{} ({})...",
        describe_requested_update(
            &release.version,
            options.current_version,
            requested_version.is_some()
        ),
        options.current_version,
        release.version,
        target.bundle_flavor.suffix()
    );
    match install_latest_bundle(
        &target.exe,
        &target.install_dir,
        &release,
        &asset_name,
        target.bundle_flavor,
        PostInstallAction::ExitAfterInstall,
    )
    .await
    {
        Ok(InstallOutcome::ExitNow) => {
            eprintln!("✅ Updated to v{}", release.version);
            Ok(())
        }
        Ok(InstallOutcome::HandoffAndExit) => {
            eprintln!(
                "✅ Applying update to v{}; exiting so the installer can finish",
                release.version
            );
            std::process::exit(0);
        }
        Ok(InstallOutcome::RestartNow) => {
            eprintln!("✅ Updated to v{}", release.version);
            Ok(())
        }
        Err(err) => Err(err),
    }
}

fn should_attempt_auto_update(options: AutoUpdateOptions) -> bool {
    options.auto_update
        && !options.plugin_requested
        && !options.command_is_update
        && std::env::var_os(SELF_UPDATE_ATTEMPTED_ENV).is_none()
}

fn discover_update_target(llama_flavor: Option<backend::BinaryFlavor>) -> Option<UpdateTarget> {
    let exe = std::env::current_exe().ok()?;
    let (install_dir, bundle_flavor) = bundle_install_dir(&exe, llama_flavor)?;
    let release_target = current_release_target(bundle_flavor)?;
    Some(UpdateTarget {
        exe,
        install_dir,
        release_target,
        bundle_flavor,
    })
}

fn require_update_target(
    flavor: Option<backend::BinaryFlavor>,
    detect_flavor: bool,
) -> Result<UpdateTarget> {
    if !platform_has_release_assets() {
        bail!(
            "`mesh-llm update` is not supported on this platform. Download the latest release from {RELEASES_URL}."
        );
    }
    if detect_flavor && flavor.is_some() {
        bail!("`mesh-llm update --detect-flavor` cannot be combined with `--flavor`.");
    }

    let exe = std::env::current_exe().context("Cannot determine mesh-llm executable path")?;
    let selected_flavor = if detect_flavor {
        preferred_bundle_flavor_for_current_host()
    } else {
        flavor
    };
    let Some((install_dir, bundle_flavor)) = bundle_install_dir(&exe, selected_flavor) else {
        bail!(
            "`mesh-llm update` only works for release-bundle installs. Current executable: {}",
            exe.display()
        );
    };
    let Some(release_target) = current_release_target(bundle_flavor) else {
        #[cfg(not(windows))]
        bail!(
            "No published release bundle matches this install. Reinstall with {INSTALL_SCRIPT_URL}."
        );
        #[cfg(windows)]
        bail!(
            "No published release bundle matches this install. Download the latest release from {RELEASES_URL}."
        );
    };

    Ok(UpdateTarget {
        exe,
        install_dir,
        release_target,
        bundle_flavor,
    })
}

async fn apply_update_if_available(
    target: UpdateTarget,
    action: PostInstallAction,
    current_version: &str,
) -> Result<bool> {
    let Some(release) = latest_release_info().await else {
        return Ok(true);
    };
    if !version_newer(&release.version, current_version) {
        return Ok(true);
    }
    let Some(asset_name) = resolve_release_asset_name(
        &release,
        target.release_target,
        ReleaseAssetPreference::StableFirst,
    ) else {
        return Ok(false);
    };
    if !path_is_writable(&target.exe) {
        eprintln!(
            "⚠️  Auto-update skipped: {} is not writable",
            target.exe.display()
        );
        return Ok(true);
    }

    eprintln!(
        "⬇️ Updating mesh-llm v{current_version} -> v{} ({})...",
        release.version,
        target.bundle_flavor.suffix()
    );
    match install_latest_bundle(
        &target.exe,
        &target.install_dir,
        &release,
        &asset_name,
        target.bundle_flavor,
        action,
    )
    .await
    {
        Ok(InstallOutcome::RestartNow) => {
            eprintln!("✅ Updated to v{}; restarting", release.version);
            exec_current_binary(&target.exe, SELF_UPDATE_ATTEMPTED_ENV, "1")?;
        }
        Ok(InstallOutcome::ExitNow) => {
            eprintln!("✅ Updated to v{}", release.version);
        }
        Ok(InstallOutcome::HandoffAndExit) => {
            eprintln!("✅ Updated to v{}; restarting", release.version);
            std::process::exit(0);
        }
        Err(err) => {
            eprintln!("⚠️  Auto-update failed: {err}");
        }
    }

    Ok(true)
}

fn bundle_install_dir(
    exe: &Path,
    requested_flavor: Option<backend::BinaryFlavor>,
) -> Option<(PathBuf, backend::BinaryFlavor)> {
    let dir = exe.parent()?;
    let file_name = exe.file_name()?.to_str()?;
    #[cfg(windows)]
    {
        if !file_name.eq_ignore_ascii_case(&mesh_binary_name()) {
            return None;
        }
    }
    #[cfg(not(windows))]
    {
        if file_name != mesh_binary_name() {
            return None;
        }
    }
    let flavor = installed_bundle_flavor(dir, requested_flavor)?;
    Some((dir.to_path_buf(), flavor))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(name: &str) -> PathBuf {
        let unique = format!(
            "mesh-llm-{name}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        );
        let path = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    #[serial]
    fn test_should_attempt_auto_update_only_when_flag_is_set() {
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::remove_var(SELF_UPDATE_ATTEMPTED_ENV) };
        assert!(should_attempt_auto_update(AutoUpdateOptions {
            auto_update: true,
            plugin_requested: false,
            command_is_update: false,
            llama_flavor: None,
            current_version: "0.68.0",
        }));

        assert!(!should_attempt_auto_update(AutoUpdateOptions {
            auto_update: false,
            plugin_requested: false,
            command_is_update: false,
            llama_flavor: None,
            current_version: "0.68.0",
        }));

        assert!(!should_attempt_auto_update(AutoUpdateOptions {
            auto_update: true,
            plugin_requested: false,
            command_is_update: true,
            llama_flavor: None,
            current_version: "0.68.0",
        }));
    }

    #[test]
    #[serial]
    fn test_should_attempt_auto_update_respects_restart_guard() {
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::set_var(SELF_UPDATE_ATTEMPTED_ENV, "1") };
        assert!(!should_attempt_auto_update(AutoUpdateOptions {
            auto_update: true,
            plugin_requested: false,
            command_is_update: false,
            llama_flavor: None,
            current_version: "0.68.0",
        }));
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::remove_var(SELF_UPDATE_ATTEMPTED_ENV) };
    }

    #[test]
    fn test_detect_flavor_rejects_explicit_flavor() {
        let err = match require_update_target(Some(backend::BinaryFlavor::Vulkan), true) {
            Ok(_) => panic!("conflicting flavor options should fail before install inspection"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("cannot be combined"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn test_bundle_install_dir_uses_requested_or_detected_flavor() {
        let dir = temp_dir("bundle-install");
        let exe = dir.join(mesh_binary_name());
        std::fs::write(&exe, b"binary").unwrap();
        let detected_flavor = preferred_bundle_flavor_for_current_host().unwrap();

        assert_eq!(
            bundle_install_dir(&exe, None),
            Some((dir.clone(), detected_flavor))
        );
        assert_eq!(
            bundle_install_dir(&exe, Some(backend::BinaryFlavor::Vulkan)),
            Some((dir.clone(), backend::BinaryFlavor::Vulkan))
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_notice_message_run_update_command() {
        assert_eq!(
            notice_message("0.76.0-rc3", "0.99.0", NoticeGuidance::RunUpdateCommand),
            "✨ New version: v0.76.0-rc3 -> v0.99.0. Run 'mesh-llm update'."
        );
    }

    #[cfg(not(windows))]
    #[test]
    fn test_notice_message_reinstall_script() {
        let install_script_url = release_fetch::INSTALL_SCRIPT_URL;
        assert_eq!(
            notice_message("0.75.2", "1.4.0", NoticeGuidance::ReinstallScript),
            format!(
                "✨ New version: v0.75.2 -> v1.4.0. Reinstall with: curl -fsSL {install_script_url} | bash"
            )
        );
    }

    #[cfg(windows)]
    #[test]
    fn test_notice_message_download_releases() {
        assert_eq!(
            notice_message("0.75.2", "1.4.0", NoticeGuidance::DownloadReleases),
            format!("✨ New version: v0.75.2 -> v1.4.0. Download from {RELEASES_URL}")
        );
    }

    #[test]
    fn test_update_notice_carries_complete_user_facing_text() {
        let message = notice_message("0.1.0", "9.9.9", NoticeGuidance::RunUpdateCommand);
        let UpdateNotice { message: carried } = UpdateNotice { message };
        assert!(carried.starts_with("✨ New version: v0.1.0 -> v9.9.9."));
    }
}
