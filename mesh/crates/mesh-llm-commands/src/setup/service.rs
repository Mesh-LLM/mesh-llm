use super::SetupPlatform;
use super::service_files::{
    ensure_service_env_file, shell_quote, write_service_runner, write_windows_task_definition,
};
use super::service_paths::{ServiceInstallContext, ServicePaths};
use super::service_runner::{ServiceCommand, ServiceCommandRunner};
use super::service_templates::{
    SERVICE_LABEL, SERVICE_NAME, WINDOWS_SERVICE_HOST_COMMAND, render_launchd_plist,
    render_systemd_unit, render_windows_task_xml,
};
use anyhow::{Context, Result, anyhow, bail};
use std::fs;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ServiceInstallReport {
    pub(crate) status: ServiceInstallStatus,
    pub(crate) summary: String,
    pub(crate) messages: Vec<String>,
    pub(crate) service_file: std::path::PathBuf,
    /// `service.env`; Windows has none, the node reads its TOML config.
    pub(crate) env_file: Option<std::path::PathBuf>,
    pub(crate) runner_file: Option<std::path::PathBuf>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ServiceInstallStatus {
    Started,
    /// Registered to start at the user's next logon, but not running yet.
    StartsAtLogon,
    NeedsManualStart,
}

pub(crate) fn install_service(
    context: &ServiceInstallContext,
    runner: &mut dyn ServiceCommandRunner,
) -> Result<ServiceInstallReport> {
    match context.platform {
        SetupPlatform::Linux => install_systemd_service(context, runner),
        SetupPlatform::MacOs => install_launchd_service(context, runner),
        SetupPlatform::Windows => install_windows_task(context, runner),
    }
}

fn install_systemd_service(
    context: &ServiceInstallContext,
    runner: &mut dyn ServiceCommandRunner,
) -> Result<ServiceInstallReport> {
    let paths = ServicePaths::from_context(context);
    // ensure_service_env_file creates the service config directory owner-only.
    fs::create_dir_all(&paths.systemd_unit_dir)?;
    ensure_service_env_file(&paths.service_env_file)?;
    fs::write(
        &paths.systemd_unit_path,
        render_systemd_unit(
            &context.binary_path,
            &paths.service_env_file,
            &paths.mesh_config_file,
        ),
    )?;

    runner
        .run(&ServiceCommand::new(
            "systemctl",
            ["--user", "daemon-reload"],
        ))
        .context("reload systemd user service manager")?;

    let service_name = format!("{SERVICE_NAME}.service");
    let manual_start_hint = format!("Start it with: systemctl --user enable --now {service_name}");
    let started = if context.start_service {
        runner
            .run(&ServiceCommand::new(
                "systemctl",
                ["--user", "enable", service_name.as_str()],
            ))
            .with_context(|| format!("enable systemd user service {service_name}"))?;
        runner
            .run(&ServiceCommand::new(
                "systemctl",
                ["--user", "restart", service_name.as_str()],
            ))
            .or_else(|restart_error| {
                runner
                    .run(&ServiceCommand::new(
                        "systemctl",
                        ["--user", "start", service_name.as_str()],
                    ))
                    .with_context(|| {
                        format!(
                            "restart systemd user service {service_name} failed ({restart_error}); start fallback"
                        )
                    })
            })
            .with_context(|| format!("start systemd user service {service_name}"))?;
        true
    } else {
        false
    };

    let exec_line = format!("ExecStart={} serve", shell_quote(&context.binary_path));
    let mut messages = Vec::new();
    if started {
        messages.push(format!(
            "Installed and started systemd user service: {service_name}"
        ));
    } else {
        messages.push(format!("Installed {}", paths.systemd_unit_path.display()));
        messages.push(manual_start_hint.clone());
    }
    messages.push(format!("Command: {exec_line}"));
    messages.push(format!(
        "Optional env: {}",
        paths.service_env_file.display()
    ));
    messages.push(format!(
        "Private mesh: set MESH_LLM_JOIN_FILE=<token file> in {} (re-read on every rejoin)",
        paths.service_env_file.display()
    ));
    messages.push(format!(
        "Edit startup models: {}",
        paths.mesh_config_file.display()
    ));
    messages.push(format!("Logs: journalctl --user -u {service_name} -f"));
    messages.push("Boot without login (optional): sudo loginctl enable-linger $USER".to_string());

    Ok(ServiceInstallReport {
        status: if started {
            ServiceInstallStatus::Started
        } else {
            ServiceInstallStatus::NeedsManualStart
        },
        summary: if started {
            "installed and started".to_string()
        } else {
            "installed; automatic start needs manual follow-up".to_string()
        },
        messages,
        service_file: paths.systemd_unit_path,
        env_file: Some(paths.service_env_file),
        runner_file: None,
    })
}

fn install_launchd_service(
    context: &ServiceInstallContext,
    runner: &mut dyn ServiceCommandRunner,
) -> Result<ServiceInstallReport> {
    let paths = ServicePaths::from_context(context);
    // ensure_service_env_file creates the service config directory owner-only.
    fs::create_dir_all(&paths.launchd_agent_dir)?;
    fs::create_dir_all(&paths.launchd_log_dir)?;
    ensure_service_env_file(&paths.service_env_file)?;
    write_service_runner(
        &paths.service_runner,
        &context.binary_path,
        &paths.service_env_file,
    )?;

    let plist_existed = paths.launchd_plist_path.exists();
    fs::write(
        &paths.launchd_plist_path,
        render_launchd_plist(
            &paths.service_runner,
            &context.home_dir,
            &paths.launchd_stdout_log,
            &paths.launchd_stderr_log,
        ),
    )?;

    let launch_domain = format!("gui/{}", context.user_id);
    let manual_start_hint = format!(
        "Start it with: launchctl bootstrap {launch_domain} {}",
        paths.launchd_plist_path.display()
    );
    let mut warnings = Vec::new();
    let started = if context.start_service {
        if plist_existed
            && let Err(error) = runner.run(&ServiceCommand::new(
                "launchctl",
                [
                    "bootout",
                    launch_domain.as_str(),
                    paths.launchd_plist_path.to_string_lossy().as_ref(),
                ],
            ))
        {
            warnings.push(format!(
                "warning: could not unload the previous launchd agent before reinstalling: {error}"
            ));
        }

        runner
            .run(&ServiceCommand::new(
                "launchctl",
                [
                    "bootstrap",
                    launch_domain.as_str(),
                    paths.launchd_plist_path.to_string_lossy().as_ref(),
                ],
            ))
            .with_context(|| format!("bootstrap launchd agent {SERVICE_LABEL}"))?;
        runner
            .run(&ServiceCommand::new(
                "launchctl",
                [
                    "enable".to_string(),
                    format!("{launch_domain}/{SERVICE_LABEL}"),
                ],
            ))
            .with_context(|| format!("enable launchd agent {SERVICE_LABEL}"))?;
        runner
            .run(&ServiceCommand::new(
                "launchctl",
                [
                    "kickstart".to_string(),
                    "-k".to_string(),
                    format!("{launch_domain}/{SERVICE_LABEL}"),
                ],
            ))
            .with_context(|| format!("kickstart launchd agent {SERVICE_LABEL}"))?;
        true
    } else {
        false
    };

    let mut messages = Vec::new();
    if started {
        messages.push(format!(
            "Installed and started launchd agent: {SERVICE_LABEL}"
        ));
    } else {
        messages.push(format!("Installed {}", paths.launchd_plist_path.display()));
        messages.push(manual_start_hint.clone());
    }
    messages.extend(warnings);
    messages.push(format!(
        "Startup models: {}",
        paths.mesh_config_file.display()
    ));
    messages.push(format!(
        "Optional env: {}",
        paths.service_env_file.display()
    ));
    messages.push(format!(
        "Private mesh: set MESH_LLM_JOIN_FILE=<token file> in {} (re-read on every rejoin)",
        paths.service_env_file.display()
    ));
    messages.push(format!(
        "Logs: {} and {}",
        paths.launchd_stdout_log.display(),
        paths.launchd_stderr_log.display()
    ));

    Ok(ServiceInstallReport {
        status: if started {
            ServiceInstallStatus::Started
        } else {
            ServiceInstallStatus::NeedsManualStart
        },
        summary: if started {
            "installed and started".to_string()
        } else {
            "installed; automatic start needs manual follow-up".to_string()
        },
        messages,
        service_file: paths.launchd_plist_path,
        env_file: Some(paths.service_env_file),
        runner_file: Some(paths.service_runner),
    })
}

fn install_windows_task(
    context: &ServiceInstallContext,
    runner: &mut dyn ServiceCommandRunner,
) -> Result<ServiceInstallReport> {
    let paths = ServicePaths::from_context(context);
    if context.user_id.is_empty() {
        bail!("scheduled task installation needs the current user's SID");
    }
    let binary_dir = context.binary_path.parent().ok_or_else(|| {
        anyhow!(
            "mesh-llm binary path has no parent: {}",
            context.binary_path.display()
        )
    })?;
    let binary_file_name = context
        .binary_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            anyhow!(
                "mesh-llm binary path has no file name: {}",
                context.binary_path.display()
            )
        })?;
    // The task names the binary relative to its directory, unquoted, because
    // `conhost --headless` drops the quoting of an argument with spaces.
    if binary_file_name.contains(char::is_whitespace) || binary_file_name.contains('"') {
        bail!(
            "the background service needs a mesh-llm binary name without spaces or quotes: {binary_file_name}"
        );
    }
    write_windows_task_definition(
        &paths.windows_task_definition,
        &render_windows_task_xml(&context.user_id, binary_dir, binary_file_name),
    )?;

    let definition = paths.windows_task_definition.to_string_lossy().to_string();
    let manual_start_hint = format!(
        "Start it with: schtasks /Create /TN {SERVICE_NAME} /XML \"{definition}\" /F, then schtasks /Run /TN {SERVICE_NAME}"
    );
    let mut warnings = Vec::new();
    let status = if context.start_service {
        // A running instance would keep the old binary and its ports, and the
        // task ignores a second start while one runs. Ending the task ends its
        // service host, which takes the node down with it.
        if runner
            .run(&ServiceCommand::new(
                "schtasks",
                ["/Query", "/TN", SERVICE_NAME],
            ))
            .is_ok()
            && let Err(error) = runner.run(&ServiceCommand::new(
                "schtasks",
                ["/End", "/TN", SERVICE_NAME],
            ))
        {
            warnings.push(format!(
                "warning: could not stop the running scheduled task before reinstalling: {error}"
            ));
        }
        runner
            .run(&ServiceCommand::new(
                "schtasks",
                [
                    "/Create",
                    "/TN",
                    SERVICE_NAME,
                    "/XML",
                    definition.as_str(),
                    "/F",
                ],
            ))
            .with_context(|| format!("register scheduled task {SERVICE_NAME}"))?;
        runner
            .run(&ServiceCommand::new(
                "schtasks",
                ["/Run", "/TN", SERVICE_NAME],
            ))
            .with_context(|| format!("start scheduled task {SERVICE_NAME}"))?;
        if windows_task_started(runner) {
            ServiceInstallStatus::Started
        } else {
            ServiceInstallStatus::StartsAtLogon
        }
    } else {
        ServiceInstallStatus::NeedsManualStart
    };

    let mut messages = Vec::new();
    match status {
        ServiceInstallStatus::Started => messages.push(format!(
            "Installed and started scheduled task: {SERVICE_NAME} (runs at your logon)"
        )),
        ServiceInstallStatus::StartsAtLogon => {
            messages.push(format!(
                "Installed scheduled task: {SERVICE_NAME}; it starts at your next logon"
            ));
            messages.push(format!(
                "It could not start now: this account has no desktop session (setup over SSH, for example). Once signed in, start it with: schtasks /Run /TN {SERVICE_NAME}"
            ));
        }
        ServiceInstallStatus::NeedsManualStart => {
            messages.push(format!(
                "Installed {}",
                paths.windows_task_definition.display()
            ));
            messages.push(manual_start_hint);
        }
    }
    messages.extend(warnings);
    messages.push(format!(
        "Command: {} {WINDOWS_SERVICE_HOST_COMMAND} (runs serve, restarts it on failure)",
        context.binary_path.display()
    ));
    messages.push(format!(
        "Startup models: {}",
        paths.mesh_config_file.display()
    ));
    messages.push(format!(
        "Private mesh: write the invite token to {}, then restart the task (schtasks /End /TN {SERVICE_NAME}, then schtasks /Run /TN {SERVICE_NAME}): a node that started without a token starts its own mesh, and once it has one the file is re-read on every rejoin",
        paths
            .mesh_config_file
            .with_file_name("invite.token")
            .display()
    ));
    messages.push(format!("Logs: {}", paths.windows_log_file.display()));
    messages.push(format!("Stop it with: schtasks /End /TN {SERVICE_NAME}"));

    Ok(ServiceInstallReport {
        status,
        summary: match status {
            ServiceInstallStatus::Started => "installed and started",
            ServiceInstallStatus::StartsAtLogon => "installed; starts at the next logon",
            ServiceInstallStatus::NeedsManualStart => {
                "installed; automatic start needs manual follow-up"
            }
        }
        .to_string(),
        messages,
        service_file: paths.windows_task_definition,
        env_file: None,
        runner_file: None,
    })
}

/// `schtasks /Run` succeeds even when the task cannot start, as when setup runs
/// over SSH and the account has no desktop session: the task's last result then
/// stays "has not run" (SCHED_S_TASK_HAS_NOT_RUN, 0x41303). That code, unlike
/// the status column, reads the same in every display language.
fn windows_task_started(runner: &mut dyn ServiceCommandRunner) -> bool {
    const TASK_HAS_NOT_RUN: &str = "267011";
    let query = ServiceCommand::new(
        "schtasks",
        ["/Query", "/TN", SERVICE_NAME, "/FO", "CSV", "/NH", "/V"],
    );
    // Task Scheduler starts the instance asynchronously.
    for attempt in 0..8 {
        if attempt > 0 {
            std::thread::sleep(std::time::Duration::from_millis(250));
        }
        let Ok(csv) = runner.output(&query) else {
            return false;
        };
        if windows_task_last_result(&csv).is_some_and(|result| result != TASK_HAS_NOT_RUN) {
            return true;
        }
    }
    false
}

/// The last-result column of `schtasks /Query /FO CSV /NH /V`.
pub(crate) fn windows_task_last_result(csv: &str) -> Option<&str> {
    csv.lines()
        .next()?
        .trim()
        .trim_matches('"')
        .split("\",\"")
        .nth(6)
}
