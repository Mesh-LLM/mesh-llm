use crate::terminal::{self, ConfirmDefault, style_muted, style_ok, style_warn};
use anyhow::{Context, Result, bail};
use serde::Serialize;
use std::{
    fs, io,
    path::{Path, PathBuf},
    process::Command,
};

const SERVICE_NAME: &str = "mesh-llm";
const LAUNCHD_LABEL: &str = "com.mesh-llm.mesh-llm";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UninstallOptions {
    pub dry_run: bool,
    pub yes: bool,
    pub keep_cache: bool,
    pub keep_service_files: bool,
    pub purge_config: bool,
    pub keep_config: bool,
    pub binary_path: Option<PathBuf>,
    pub json: bool,
    pub verbose: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UninstallEnvironment {
    pub platform: UninstallPlatform,
    pub home_dir: PathBuf,
    pub config_root: PathBuf,
    pub cache_root: PathBuf,
    pub binary_path: PathBuf,
    pub user_id: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UninstallPlatform {
    Linux,
    MacOs,
    Windows,
    Other,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum UninstallStep {
    StopProcesses,
    DisableSystemdUserService,
    ReloadSystemdUser,
    BootoutLaunchdAgent {
        user_id: String,
        plist_path: PathBuf,
    },
    RemovePath {
        path: PathBuf,
        purpose: RemovePurpose,
    },
    RemoveBinary {
        path: PathBuf,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RemovePurpose {
    SystemdUnit,
    LaunchdPlist,
    ServiceEnv,
    ServiceRunner,
    ServiceConfigDir,
    LaunchdLogs,
    NativeRuntimeCache,
    ConfigAndIdentity,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct UninstallPlan {
    pub dry_run: bool,
    pub requires_confirmation: bool,
    pub platform: UninstallPlatform,
    pub steps: Vec<UninstallStep>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct UninstallOutcome {
    pub dry_run: bool,
    pub removed: Vec<PathBuf>,
    pub scheduled_removal: Vec<PathBuf>,
    pub already_absent: Vec<PathBuf>,
    pub warnings: Vec<String>,
}

pub fn detect_uninstall_environment(binary_path: Option<PathBuf>) -> Result<UninstallEnvironment> {
    let home_dir =
        dirs::home_dir().context("could not determine the home directory for uninstall")?;
    let config_root = dirs::config_dir().unwrap_or_else(|| home_dir.join(".config"));
    let cache_root = dirs::cache_dir().unwrap_or_else(|| home_dir.join(".cache"));
    let binary_path = match binary_path {
        Some(path) => path,
        None => std::env::current_exe()
            .context("could not determine the running mesh-llm executable path")?,
    };
    let user_id = if cfg!(target_os = "macos") {
        detect_user_id().unwrap_or_default()
    } else {
        String::new()
    };
    Ok(UninstallEnvironment {
        platform: current_platform(),
        home_dir,
        config_root,
        cache_root,
        binary_path,
        user_id,
    })
}

pub fn plan_uninstall(options: &UninstallOptions, env: &UninstallEnvironment) -> UninstallPlan {
    let mut steps = vec![UninstallStep::StopProcesses];
    match env.platform {
        UninstallPlatform::Linux => {
            steps.push(UninstallStep::DisableSystemdUserService);
            steps.push(UninstallStep::RemovePath {
                path: env
                    .config_root
                    .join("systemd/user")
                    .join(format!("{SERVICE_NAME}.service")),
                purpose: RemovePurpose::SystemdUnit,
            });
            steps.push(UninstallStep::ReloadSystemdUser);
        }
        UninstallPlatform::MacOs => {
            let plist_path = env
                .home_dir
                .join("Library/LaunchAgents")
                .join(format!("{LAUNCHD_LABEL}.plist"));
            steps.push(UninstallStep::BootoutLaunchdAgent {
                user_id: env.user_id.clone(),
                plist_path: plist_path.clone(),
            });
            steps.push(UninstallStep::RemovePath {
                path: plist_path,
                purpose: RemovePurpose::LaunchdPlist,
            });
            steps.push(UninstallStep::RemovePath {
                path: env.home_dir.join("Library/Logs/mesh-llm"),
                purpose: RemovePurpose::LaunchdLogs,
            });
        }
        UninstallPlatform::Windows | UninstallPlatform::Other => {}
    }
    if !options.keep_service_files {
        let service_config_dir = env.config_root.join("mesh-llm");
        steps.extend([
            UninstallStep::RemovePath {
                path: service_config_dir.join("service.env"),
                purpose: RemovePurpose::ServiceEnv,
            },
            UninstallStep::RemovePath {
                path: service_config_dir.join("run-service.sh"),
                purpose: RemovePurpose::ServiceRunner,
            },
            UninstallStep::RemovePath {
                path: service_config_dir,
                purpose: RemovePurpose::ServiceConfigDir,
            },
        ]);
    }
    if !options.keep_cache {
        steps.push(UninstallStep::RemovePath {
            path: env.cache_root.join("mesh-llm/native-runtimes"),
            purpose: RemovePurpose::NativeRuntimeCache,
        });
    }
    if options.purge_config && !options.keep_config {
        steps.push(UninstallStep::RemovePath {
            path: env.home_dir.join(".mesh-llm"),
            purpose: RemovePurpose::ConfigAndIdentity,
        });
    }
    steps.push(UninstallStep::RemoveBinary {
        path: env.binary_path.clone(),
    });
    UninstallPlan {
        dry_run: options.dry_run,
        requires_confirmation: !options.dry_run && !options.yes,
        platform: env.platform,
        steps,
    }
}

pub fn run_uninstall_command<F>(options: UninstallOptions, mut stop_processes: F) -> Result<()>
where
    F: FnMut() -> Result<()>,
{
    let env = detect_uninstall_environment(options.binary_path.clone())?;
    let plan = plan_uninstall(&options, &env);
    if options.dry_run {
        render_plan(&plan, options.json, options.verbose)?;
        return Ok(());
    }
    if plan.requires_confirmation && !confirm_uninstall()? {
        bail!("uninstall cancelled");
    }
    let mut outcome = execute_uninstall_plan(&plan, &mut stop_processes)?;
    add_option_warnings(&options, &mut outcome);
    render_outcome(&outcome, options.json, options.verbose)
}

pub fn execute_uninstall_plan<F>(
    plan: &UninstallPlan,
    stop_processes: &mut F,
) -> Result<UninstallOutcome>
where
    F: FnMut() -> Result<()>,
{
    let mut outcome = UninstallOutcome {
        dry_run: plan.dry_run,
        removed: Vec::new(),
        scheduled_removal: Vec::new(),
        already_absent: Vec::new(),
        warnings: Vec::new(),
    };
    if plan.dry_run {
        return Ok(outcome);
    }
    for step in &plan.steps {
        execute_step(step, stop_processes, &mut outcome)?;
    }
    Ok(outcome)
}

fn execute_step<F>(
    step: &UninstallStep,
    stop_processes: &mut F,
    outcome: &mut UninstallOutcome,
) -> Result<()>
where
    F: FnMut() -> Result<()>,
{
    match step {
        UninstallStep::StopProcesses => {
            if let Err(error) = stop_processes() {
                outcome.warnings.push(format!(
                    "failed to stop tracked mesh-llm processes: {error:#}"
                ));
            }
        }
        UninstallStep::DisableSystemdUserService => {
            run_best_effort(
                Command::new("systemctl").args(["--user", "disable", "--now", "mesh-llm.service"]),
                outcome,
            );
        }
        UninstallStep::ReloadSystemdUser => {
            run_best_effort(
                Command::new("systemctl").args(["--user", "daemon-reload"]),
                outcome,
            );
        }
        UninstallStep::BootoutLaunchdAgent {
            user_id,
            plist_path,
        } => {
            if user_id.is_empty() {
                outcome
                    .warnings
                    .push("could not determine user id for launchd bootout".to_string());
            } else {
                run_best_effort(
                    Command::new("launchctl").args([
                        "bootout",
                        &format!("gui/{user_id}"),
                        &plist_path.display().to_string(),
                    ]),
                    outcome,
                );
            }
        }
        UninstallStep::RemovePath { path, purpose } => {
            remove_path(path, *purpose, outcome)
                .with_context(|| format!("failed to remove {}", path.display()))?;
        }
        UninstallStep::RemoveBinary { path } => remove_binary(path, outcome)
            .with_context(|| format!("failed to remove {}", path.display()))?,
    }
    Ok(())
}

fn remove_path(path: &Path, purpose: RemovePurpose, outcome: &mut UninstallOutcome) -> Result<()> {
    if purpose == RemovePurpose::ServiceConfigDir {
        return remove_empty_dir(path, outcome);
    }
    remove_recursively(path, outcome)
}

fn remove_binary(path: &Path, outcome: &mut UninstallOutcome) -> Result<()> {
    let current_exe = std::env::current_exe().ok();
    remove_binary_for_platform(path, current_platform(), current_exe.as_deref(), outcome)
}

fn remove_binary_for_platform(
    path: &Path,
    platform: UninstallPlatform,
    current_exe: Option<&Path>,
    outcome: &mut UninstallOutcome,
) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_dir() => {
            bail!(
                "refusing to remove binary path because it is a directory: {}",
                path.display()
            );
        }
        Ok(_)
            if binary_removal_action(path, platform, current_exe) == BinaryRemovalAction::Defer =>
        {
            schedule_windows_deferred_binary_delete(path)?;
            outcome.scheduled_removal.push(path.to_path_buf());
        }
        Ok(_) => {
            fs::remove_file(path)?;
            outcome.removed.push(path.to_path_buf());
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            outcome.already_absent.push(path.to_path_buf());
        }
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BinaryRemovalAction {
    RemoveNow,
    Defer,
}

fn binary_removal_action(
    path: &Path,
    platform: UninstallPlatform,
    current_exe: Option<&Path>,
) -> BinaryRemovalAction {
    if platform == UninstallPlatform::Windows
        && current_exe.is_some_and(|current_exe| paths_match(path, current_exe))
    {
        return BinaryRemovalAction::Defer;
    }
    BinaryRemovalAction::RemoveNow
}

fn paths_match(left: &Path, right: &Path) -> bool {
    match (left.canonicalize(), right.canonicalize()) {
        (Ok(left), Ok(right)) => left == right,
        _ => left == right,
    }
}

#[cfg(windows)]
fn schedule_windows_deferred_binary_delete(path: &Path) -> Result<()> {
    let escaped_path = path.to_string_lossy().replace('\'', "''");
    let script =
        format!("Start-Sleep -Seconds 1; Remove-Item -LiteralPath '{escaped_path}' -Force");
    Command::new("powershell.exe")
        .args([
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            &script,
        ])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .with_context(|| format!("failed to schedule deferred removal for {}", path.display()))?;
    Ok(())
}

#[cfg(not(windows))]
fn schedule_windows_deferred_binary_delete(path: &Path) -> Result<()> {
    let _ = fs::metadata(path)?;
    Ok(())
}

fn remove_empty_dir(path: &Path, outcome: &mut UninstallOutcome) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_dir() => match fs::remove_dir(path) {
            Ok(()) => outcome.removed.push(path.to_path_buf()),
            Err(error) if error.kind() == io::ErrorKind::DirectoryNotEmpty => {
                outcome.warnings.push(format!(
                    "leaving non-empty service config directory {}",
                    path.display()
                ));
            }
            Err(error) => return Err(error.into()),
        },
        Ok(_) => {
            fs::remove_file(path)?;
            outcome.removed.push(path.to_path_buf());
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            outcome.already_absent.push(path.to_path_buf());
        }
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

fn remove_recursively(path: &Path, outcome: &mut UninstallOutcome) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_dir() => match fs::remove_dir(path) {
            Ok(()) => outcome.removed.push(path.to_path_buf()),
            Err(error) if error.kind() == io::ErrorKind::DirectoryNotEmpty => {
                fs::remove_dir_all(path)?;
                outcome.removed.push(path.to_path_buf());
            }
            Err(error) => return Err(error.into()),
        },
        Ok(_) => {
            fs::remove_file(path)?;
            outcome.removed.push(path.to_path_buf());
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            outcome.already_absent.push(path.to_path_buf());
        }
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

fn run_best_effort(command: &mut Command, outcome: &mut UninstallOutcome) {
    match command.output() {
        Ok(output) if output.status.success() => {}
        Ok(output) => outcome.warnings.push(format!(
            "`{:?}` exited with status {}",
            command, output.status
        )),
        Err(error) => outcome
            .warnings
            .push(format!("failed to run `{:?}`: {error}", command)),
    }
}

fn add_option_warnings(options: &UninstallOptions, outcome: &mut UninstallOutcome) {
    if options.purge_config && options.keep_config {
        outcome.warnings.push(
            "--purge-config was ignored because --keep-config was also set; preserving configuration"
                .to_string(),
        );
    }
}

fn render_plan(plan: &UninstallPlan, json: bool, verbose: bool) -> Result<()> {
    if json {
        println!("{}", serde_json::to_string_pretty(plan)?);
        return Ok(());
    }
    for line in plan_lines(plan, verbose) {
        eprintln!("{line}");
    }
    Ok(())
}

fn render_outcome(outcome: &UninstallOutcome, json: bool, verbose: bool) -> Result<()> {
    if json {
        println!("{}", serde_json::to_string_pretty(outcome)?);
        return Ok(());
    }
    eprintln!();
    for line in outcome_lines(outcome, verbose) {
        eprintln!("{line}");
    }
    Ok(())
}

fn plan_lines(plan: &UninstallPlan, verbose: bool) -> Vec<String> {
    if verbose {
        let mut lines = vec!["Uninstall dry run".to_string()];
        lines.extend(
            plan.steps
                .iter()
                .map(|step| format!("  - {}", step_label(step))),
        );
        return lines;
    }

    let mut lines = vec![format!("{} Mesh uninstall dry run", style_warn("!"))];
    lines.push(format!("  Steps    {} planned", plan.steps.len()));
    lines.push(format!("  Config   {}", config_plan_label(plan)));
    if let Some(binary_path) = binary_path_from_plan(plan) {
        lines.push(format!("  Binary   {}", binary_path.display()));
    }
    lines.push("Run with `--yes` to uninstall, or `--verbose` to inspect every step.".to_string());
    lines
}

fn outcome_lines(outcome: &UninstallOutcome, verbose: bool) -> Vec<String> {
    let mut lines = Vec::new();
    for warning in &outcome.warnings {
        lines.push(format!("{} {warning}", style_warn("warning:")));
    }
    if verbose {
        lines.push("Mesh uninstall complete".to_string());
        push_path_group(&mut lines, "Removed", &outcome.removed);
        push_path_group(
            &mut lines,
            "Scheduled for removal after exit",
            &outcome.scheduled_removal,
        );
        push_path_group(&mut lines, "Already absent", &outcome.already_absent);
        return lines;
    }

    lines.push(format!("{} Mesh uninstall complete", style_ok("✓")));
    lines.push(format!(
        "  Removed  {}",
        style_ok(&item_count(outcome.removed.len()))
    ));
    if !outcome.scheduled_removal.is_empty() {
        lines.push(format!(
            "  Deferred {}",
            style_warn(&item_count(outcome.scheduled_removal.len()))
        ));
    }
    if !outcome.already_absent.is_empty() {
        lines.push(format!(
            "  Skipped  {}",
            style_muted(&format!("{} already absent", outcome.already_absent.len()))
        ));
    }
    if !outcome.warnings.is_empty() {
        lines.push(format!(
            "  Warnings {}",
            style_warn(&outcome.warnings.len().to_string())
        ));
    }
    lines
}

fn push_path_group(lines: &mut Vec<String>, title: &str, paths: &[PathBuf]) {
    if paths.is_empty() {
        return;
    }
    lines.push(format!("{title}:"));
    lines.extend(paths.iter().map(|path| format!("  - {}", path.display())));
}

fn item_count(count: usize) -> String {
    format!("{count} {}", if count == 1 { "item" } else { "items" })
}

fn binary_path_from_plan(plan: &UninstallPlan) -> Option<&Path> {
    plan.steps.iter().find_map(|step| match step {
        UninstallStep::RemoveBinary { path } => Some(path.as_path()),
        _ => None,
    })
}

fn config_plan_label(plan: &UninstallPlan) -> &'static str {
    if plan.steps.iter().any(|step| {
        matches!(
            step,
            UninstallStep::RemovePath {
                purpose: RemovePurpose::ConfigAndIdentity,
                ..
            }
        )
    }) {
        "will be removed"
    } else {
        "preserved"
    }
}

fn step_label(step: &UninstallStep) -> String {
    match step {
        UninstallStep::StopProcesses => "stop tracked mesh-llm processes".to_string(),
        UninstallStep::DisableSystemdUserService => {
            "disable and stop systemd user service".to_string()
        }
        UninstallStep::ReloadSystemdUser => "reload systemd user units".to_string(),
        UninstallStep::BootoutLaunchdAgent { .. } => "boot out launchd agent".to_string(),
        UninstallStep::RemovePath { path, purpose } => {
            format!("remove {}: {}", purpose_label(*purpose), path.display())
        }
        UninstallStep::RemoveBinary { path } => format!("remove binary: {}", path.display()),
    }
}

fn purpose_label(purpose: RemovePurpose) -> &'static str {
    match purpose {
        RemovePurpose::SystemdUnit => "systemd unit",
        RemovePurpose::LaunchdPlist => "launchd plist",
        RemovePurpose::ServiceEnv => "service environment file",
        RemovePurpose::ServiceRunner => "service runner",
        RemovePurpose::ServiceConfigDir => "service config directory if empty",
        RemovePurpose::LaunchdLogs => "launchd logs",
        RemovePurpose::NativeRuntimeCache => "native runtime cache",
        RemovePurpose::ConfigAndIdentity => "configuration and identity",
    }
}

fn confirm_uninstall() -> Result<bool> {
    terminal::confirm_yes_no("Remove mesh-llm from this machine?", ConfirmDefault::No)
        .map(|reply| reply.unwrap_or(false))
}

fn current_platform() -> UninstallPlatform {
    if cfg!(target_os = "linux") {
        UninstallPlatform::Linux
    } else if cfg!(target_os = "macos") {
        UninstallPlatform::MacOs
    } else if cfg!(target_os = "windows") {
        UninstallPlatform::Windows
    } else {
        UninstallPlatform::Other
    }
}

fn detect_user_id() -> Result<String> {
    let output = Command::new("id")
        .arg("-u")
        .output()
        .context("failed to run `id -u` for launchd cleanup")?;
    if !output.status.success() {
        bail!("`id -u` exited with status {}", output.status);
    }
    Ok(String::from_utf8(output.stdout)
        .context("`id -u` emitted non-UTF-8 output")?
        .trim()
        .to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(temp: &Path, platform: UninstallPlatform) -> UninstallEnvironment {
        UninstallEnvironment {
            platform,
            home_dir: temp.join("home"),
            config_root: temp.join("config"),
            cache_root: temp.join("cache"),
            binary_path: temp.join("bin/mesh-llm"),
            user_id: "501".to_string(),
        }
    }

    fn options() -> UninstallOptions {
        UninstallOptions {
            dry_run: false,
            yes: true,
            keep_cache: false,
            keep_service_files: false,
            purge_config: false,
            keep_config: false,
            binary_path: None,
            json: false,
            verbose: false,
        }
    }

    #[test]
    fn linux_plan_removes_service_cache_and_binary_but_preserves_config_by_default() {
        let temp = tempfile::tempdir().expect("tempdir");
        let plan = plan_uninstall(&options(), &env(temp.path(), UninstallPlatform::Linux));

        assert!(
            plan.steps
                .contains(&UninstallStep::DisableSystemdUserService)
        );
        assert!(plan.steps.contains(&UninstallStep::ReloadSystemdUser));
        assert!(plan.steps.iter().any(|step| matches!(
            step,
            UninstallStep::RemovePath {
                purpose: RemovePurpose::NativeRuntimeCache,
                ..
            }
        )));
        assert!(
            plan.steps
                .iter()
                .any(|step| matches!(step, UninstallStep::RemoveBinary { .. }))
        );
        assert!(!plan.steps.iter().any(|step| matches!(
            step,
            UninstallStep::RemovePath {
                purpose: RemovePurpose::ConfigAndIdentity,
                ..
            }
        )));
    }

    #[test]
    fn keep_flags_omit_cache_and_service_helper_paths() {
        let temp = tempfile::tempdir().expect("tempdir");
        let opts = UninstallOptions {
            keep_cache: true,
            keep_service_files: true,
            ..options()
        };
        let plan = plan_uninstall(&opts, &env(temp.path(), UninstallPlatform::Linux));

        assert!(!plan.steps.iter().any(|step| matches!(
            step,
            UninstallStep::RemovePath {
                purpose: RemovePurpose::NativeRuntimeCache
                    | RemovePurpose::ServiceEnv
                    | RemovePurpose::ServiceRunner
                    | RemovePurpose::ServiceConfigDir,
                ..
            }
        )));
    }

    #[test]
    fn purge_config_adds_identity_config_removal() {
        let temp = tempfile::tempdir().expect("tempdir");
        let opts = UninstallOptions {
            purge_config: true,
            ..options()
        };
        let plan = plan_uninstall(&opts, &env(temp.path(), UninstallPlatform::MacOs));

        assert!(plan.steps.iter().any(|step| matches!(
            step,
            UninstallStep::RemovePath {
                purpose: RemovePurpose::ConfigAndIdentity,
                ..
            }
        )));
    }

    #[test]
    fn purge_config_keep_config_conflict_reports_warning() {
        let mut outcome = UninstallOutcome {
            dry_run: false,
            removed: Vec::new(),
            scheduled_removal: Vec::new(),
            already_absent: Vec::new(),
            warnings: Vec::new(),
        };
        let opts = UninstallOptions {
            purge_config: true,
            keep_config: true,
            ..options()
        };

        add_option_warnings(&opts, &mut outcome);

        assert_eq!(
            outcome.warnings,
            vec![
                "--purge-config was ignored because --keep-config was also set; preserving configuration"
                    .to_string()
            ]
        );
    }

    #[test]
    fn execute_plan_removes_files_and_directories() {
        let temp = tempfile::tempdir().expect("tempdir");
        let env = env(temp.path(), UninstallPlatform::Other);
        fs::create_dir_all(env.binary_path.parent().expect("binary parent")).expect("bin dir");
        fs::write(&env.binary_path, "binary").expect("binary");
        let cache = env.cache_root.join("mesh-llm/native-runtimes");
        fs::create_dir_all(&cache).expect("cache dir");
        fs::write(cache.join("manifest.json"), "{}").expect("cache file");
        let plan = plan_uninstall(&options(), &env);
        let mut stopped = false;

        let outcome = execute_uninstall_plan(&plan, &mut || {
            stopped = true;
            Ok(())
        })
        .expect("uninstall should execute");

        assert!(stopped);
        assert!(!env.binary_path.exists());
        assert!(!cache.exists());
        assert!(outcome.warnings.is_empty());
    }

    #[test]
    fn binary_removal_rejects_directory_paths() {
        let temp = tempfile::tempdir().expect("tempdir");
        let mut outcome = UninstallOutcome {
            dry_run: false,
            removed: Vec::new(),
            scheduled_removal: Vec::new(),
            already_absent: Vec::new(),
            warnings: Vec::new(),
        };

        let error = remove_binary_for_platform(
            temp.path(),
            UninstallPlatform::Linux,
            Some(temp.path()),
            &mut outcome,
        )
        .expect_err("directories are not valid binary paths");

        assert!(
            error
                .to_string()
                .contains("refusing to remove binary path because it is a directory")
        );
        assert!(temp.path().exists());
    }

    #[test]
    fn windows_current_exe_binary_removal_is_deferred() {
        let temp = tempfile::tempdir().expect("tempdir");
        let binary = temp.path().join("mesh-llm.exe");
        fs::write(&binary, "binary").expect("binary");

        assert_eq!(
            binary_removal_action(&binary, UninstallPlatform::Windows, Some(&binary)),
            BinaryRemovalAction::Defer
        );
        assert_eq!(
            binary_removal_action(&binary, UninstallPlatform::Linux, Some(&binary)),
            BinaryRemovalAction::RemoveNow
        );
    }

    #[test]
    fn execute_plan_leaves_non_empty_service_config_directory() {
        let temp = tempfile::tempdir().expect("tempdir");
        let env = env(temp.path(), UninstallPlatform::Other);
        let service_config_dir = env.config_root.join("mesh-llm");
        fs::create_dir_all(&service_config_dir).expect("service config dir");
        fs::write(service_config_dir.join("service.env"), "MESH=1").expect("service env");
        fs::write(service_config_dir.join("custom.toml"), "owned_by=user").expect("custom file");
        fs::create_dir_all(env.binary_path.parent().expect("binary parent")).expect("bin dir");
        fs::write(&env.binary_path, "binary").expect("binary");
        let plan = plan_uninstall(&options(), &env);

        let outcome = execute_uninstall_plan(&plan, &mut || Ok(())).expect("uninstall");

        assert!(!service_config_dir.join("service.env").exists());
        assert!(service_config_dir.join("custom.toml").exists());
        assert!(service_config_dir.exists());
        assert!(
            outcome
                .warnings
                .iter()
                .any(|warning| warning.contains("leaving non-empty service config directory"))
        );
    }

    #[test]
    fn compact_dry_run_summarizes_without_listing_every_step() {
        let temp = tempfile::tempdir().expect("tempdir");
        let plan = plan_uninstall(&options(), &env(temp.path(), UninstallPlatform::Linux));

        let lines = plan_lines(&plan, false);

        assert_eq!(lines[0], "! Mesh uninstall dry run");
        assert!(lines.iter().any(|line| line == "  Config   preserved"));
        assert!(lines.iter().any(|line| line.starts_with("  Binary   ")));
        assert!(
            lines
                .iter()
                .all(|line| !line.contains("disable and stop systemd user service"))
        );
    }

    #[test]
    fn verbose_dry_run_lists_cleanup_steps() {
        let temp = tempfile::tempdir().expect("tempdir");
        let plan = plan_uninstall(&options(), &env(temp.path(), UninstallPlatform::Linux));

        let lines = plan_lines(&plan, true);

        assert_eq!(lines[0], "Uninstall dry run");
        assert!(
            lines
                .iter()
                .any(|line| line.contains("disable and stop systemd user service"))
        );
    }

    #[test]
    fn compact_outcome_summarizes_counts_without_paths() {
        let temp = tempfile::tempdir().expect("tempdir");
        let removed = temp.path().join("bin/mesh-llm");
        let outcome = UninstallOutcome {
            dry_run: false,
            removed: vec![removed.clone()],
            scheduled_removal: Vec::new(),
            already_absent: vec![temp.path().join("missing")],
            warnings: Vec::new(),
        };

        let lines = outcome_lines(&outcome, false);

        assert_eq!(lines[0], "✓ Mesh uninstall complete");
        assert!(lines.iter().any(|line| line == "  Removed  1 item"));
        assert!(
            lines
                .iter()
                .any(|line| line == "  Skipped  1 already absent")
        );
        assert!(
            lines
                .iter()
                .all(|line| !line.contains(&removed.display().to_string()))
        );
    }

    #[test]
    fn verbose_outcome_lists_removed_paths() {
        let temp = tempfile::tempdir().expect("tempdir");
        let removed = temp.path().join("bin/mesh-llm");
        let outcome = UninstallOutcome {
            dry_run: false,
            removed: vec![removed.clone()],
            scheduled_removal: Vec::new(),
            already_absent: Vec::new(),
            warnings: Vec::new(),
        };

        let lines = outcome_lines(&outcome, true);

        assert_eq!(lines[0], "Mesh uninstall complete");
        assert!(
            lines
                .iter()
                .any(|line| line == &format!("  - {}", removed.display()))
        );
    }
}
