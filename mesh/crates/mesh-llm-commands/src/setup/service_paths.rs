use super::SetupPlatform;
use super::service_templates::{SERVICE_LABEL, SERVICE_NAME};
use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ServiceInstallContext {
    pub(crate) platform: SetupPlatform,
    pub(crate) home_dir: PathBuf,
    pub(crate) config_root: PathBuf,
    /// Machine-local data root (`%LOCALAPPDATA%` on Windows), for the
    /// Windows service log.
    pub(crate) local_data_root: PathBuf,
    pub(crate) binary_path: PathBuf,
    /// The launchd user id on macOS, the account SID on Windows.
    pub(crate) user_id: String,
    pub(crate) start_service: bool,
}

impl ServiceInstallContext {
    pub(crate) fn detect(platform: SetupPlatform, start_service: bool) -> Result<Self> {
        let home_dir = dirs::home_dir()
            .context("could not determine the home directory for service installation")?;
        let config_root = dirs::config_dir().unwrap_or_else(|| home_dir.join(".config"));
        let local_data_root =
            dirs::data_local_dir().unwrap_or_else(|| home_dir.join(".local/share"));
        let binary_path = std::env::current_exe()
            .context("could not determine the installed mesh-llm binary path")?;
        let user_id = match platform {
            SetupPlatform::MacOs => detect_user_id()?,
            SetupPlatform::Windows => detect_windows_user_sid()?,
            SetupPlatform::Linux => String::new(),
        };

        Ok(Self {
            platform,
            home_dir,
            config_root,
            local_data_root,
            binary_path,
            user_id,
            start_service,
        })
    }
}

/// The scheduled task definition `setup` writes and registers on Windows.
pub(crate) const WINDOWS_TASK_DEFINITION_FILE: &str = "mesh-llm-task.xml";
pub(crate) const WINDOWS_SERVICE_LOG_FILE: &str = "service.log";
/// The running service host's id and start time, beside its log.
pub(crate) const WINDOWS_SERVICE_HOST_RECORD: &str = "service-host.pid";

/// Where the Windows service host writes the node's output.
pub(crate) fn windows_service_log_dir(local_data_root: &Path) -> PathBuf {
    local_data_root.join("mesh-llm").join("logs")
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ServicePaths {
    pub(crate) mesh_config_file: PathBuf,
    pub(crate) service_config_dir: PathBuf,
    pub(crate) service_env_file: PathBuf,
    pub(crate) service_runner: PathBuf,
    pub(crate) systemd_unit_dir: PathBuf,
    pub(crate) systemd_unit_path: PathBuf,
    pub(crate) launchd_agent_dir: PathBuf,
    pub(crate) launchd_plist_path: PathBuf,
    pub(crate) launchd_log_dir: PathBuf,
    pub(crate) launchd_stdout_log: PathBuf,
    pub(crate) launchd_stderr_log: PathBuf,
    pub(crate) windows_task_definition: PathBuf,
    pub(crate) windows_log_file: PathBuf,
}

impl ServicePaths {
    pub(crate) fn from_context(context: &ServiceInstallContext) -> Self {
        let service_config_dir = context.config_root.join("mesh-llm");
        Self {
            windows_task_definition: service_config_dir.join(WINDOWS_TASK_DEFINITION_FILE),
            windows_log_file: windows_service_log_dir(&context.local_data_root)
                .join(WINDOWS_SERVICE_LOG_FILE),
            mesh_config_file: context.home_dir.join(".mesh-llm").join("config.toml"),
            service_env_file: service_config_dir.join("service.env"),
            service_runner: service_config_dir.join("run-service.sh"),
            systemd_unit_dir: context.config_root.join("systemd/user"),
            systemd_unit_path: context
                .config_root
                .join("systemd/user")
                .join(format!("{SERVICE_NAME}.service")),
            launchd_agent_dir: context.home_dir.join("Library/LaunchAgents"),
            launchd_plist_path: context
                .home_dir
                .join("Library/LaunchAgents")
                .join(format!("{SERVICE_LABEL}.plist")),
            launchd_log_dir: context.home_dir.join("Library/Logs/mesh-llm"),
            launchd_stdout_log: context.home_dir.join("Library/Logs/mesh-llm/stdout.log"),
            launchd_stderr_log: context.home_dir.join("Library/Logs/mesh-llm/stderr.log"),
            service_config_dir,
        }
    }
}

fn detect_user_id() -> Result<String> {
    let output = Command::new("id")
        .arg("-u")
        .output()
        .context("failed to run `id -u` for launchd service installation")?;
    if !output.status.success() {
        bail!("`id -u` exited with status {}", output.status);
    }
    let user_id = String::from_utf8(output.stdout).context("`id -u` emitted non-UTF-8 output")?;
    let trimmed = user_id.trim();
    if trimmed.is_empty() {
        bail!("`id -u` returned an empty user id");
    }
    Ok(trimmed.to_string())
}

// The scheduled task names its user by SID, which needs no domain or account
// name lookup and survives a renamed account.
fn detect_windows_user_sid() -> Result<String> {
    let output = Command::new("whoami")
        .args(["/user", "/fo", "csv", "/nh"])
        .output()
        .context("failed to run `whoami /user` for scheduled task installation")?;
    if !output.status.success() {
        bail!("`whoami /user` exited with status {}", output.status);
    }
    let line = String::from_utf8_lossy(&output.stdout);
    parse_whoami_user_sid(&line)
        .with_context(|| format!("`whoami /user` printed no SID: {}", line.trim()))
}

/// The SID column of `whoami /user /fo csv /nh`: `"machine\user","S-1-5-..."`.
pub(crate) fn parse_whoami_user_sid(line: &str) -> Option<String> {
    let sid = line.trim().rsplit(',').next()?.trim().trim_matches('"');
    sid.starts_with("S-1-").then(|| sid.to_string())
}
