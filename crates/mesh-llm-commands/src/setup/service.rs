use super::SetupPlatform;
use super::service_files::{ensure_service_env_file, shell_quote, write_service_runner};
use super::service_paths::{ServiceInstallContext, ServicePaths};
use super::service_runner::{ServiceCommand, ServiceCommandRunner};
use super::service_templates::{
    SERVICE_LABEL, SERVICE_NAME, render_launchd_plist, render_systemd_unit,
};
use anyhow::{Context, Result, bail};
use std::fs;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ServiceInstallReport {
    pub(crate) status: ServiceInstallStatus,
    pub(crate) summary: String,
    pub(crate) messages: Vec<String>,
    pub(crate) service_file: std::path::PathBuf,
    pub(crate) env_file: std::path::PathBuf,
    pub(crate) runner_file: Option<std::path::PathBuf>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ServiceInstallStatus {
    Started,
    NeedsManualStart,
}

pub(crate) fn install_service(
    context: &ServiceInstallContext,
    runner: &mut dyn ServiceCommandRunner,
) -> Result<ServiceInstallReport> {
    match context.platform {
        SetupPlatform::Linux => install_systemd_service(context, runner),
        SetupPlatform::MacOs => install_launchd_service(context, runner),
        SetupPlatform::Windows => bail!("setup cannot install the background service on windows"),
    }
}

fn install_systemd_service(
    context: &ServiceInstallContext,
    runner: &mut dyn ServiceCommandRunner,
) -> Result<ServiceInstallReport> {
    let paths = ServicePaths::from_context(context);
    fs::create_dir_all(&paths.service_config_dir)?;
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
        env_file: paths.service_env_file,
        runner_file: None,
    })
}

fn install_launchd_service(
    context: &ServiceInstallContext,
    runner: &mut dyn ServiceCommandRunner,
) -> Result<ServiceInstallReport> {
    let paths = ServicePaths::from_context(context);
    fs::create_dir_all(&paths.service_config_dir)?;
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
        env_file: paths.service_env_file,
        runner_file: Some(paths.service_runner),
    })
}
