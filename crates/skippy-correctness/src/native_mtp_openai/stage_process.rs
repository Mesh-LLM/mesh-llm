use std::{
    fs::{self, File},
    net::SocketAddr,
    path::Path,
    process::{Command, Stdio},
    thread,
    time::Duration,
};

use anyhow::{Context, Result, bail};
use reqwest::blocking::Client;
use serde_json::Value;

use crate::{
    cli::NativeMtpOpenAiAbArgs,
    support::{ChildGuard, connect_ready},
};

use super::{
    BATCHED_VERIFY_ENV, NATIVE_MTP_ENABLED_ENV,
    remote::{RemoteStageGuard, spawn_remote_stage1},
};

pub(super) struct Stage1LaunchReport {
    pub(super) launch_mode: &'static str,
    pub(super) remote_config: Option<String>,
    pub(super) remote_topology: Option<String>,
    pub(super) remote_log: Option<String>,
}

pub(super) enum StageHandle {
    Local(ChildGuard),
    Remote(RemoteStageGuard),
    External,
}

impl StageHandle {
    pub(super) fn stop_and_collect(self, stage1_log: &Path) -> Result<()> {
        match self {
            StageHandle::Local(guard) => {
                drop(guard);
                Ok(())
            }
            StageHandle::External => Ok(()),
            StageHandle::Remote(mut guard) => guard.stop_and_collect(stage1_log),
        }
    }
}

pub(super) fn start_stage1(
    args: &NativeMtpOpenAiAbArgs,
    run_id: &str,
    config_path: &Path,
    topology_path: &Path,
    log_path: &Path,
    native_mtp_enabled: bool,
    batched_verify_enabled: bool,
) -> Result<(StageHandle, Stage1LaunchReport)> {
    if args.external_stage1 {
        fs::write(log_path, "stage 1 was externally managed by the caller\n")
            .with_context(|| format!("failed to create {}", log_path.display()))?;
        return Ok((
            StageHandle::External,
            Stage1LaunchReport {
                launch_mode: "external",
                remote_config: None,
                remote_topology: None,
                remote_log: None,
            },
        ));
    }

    if let Some(host) = args.stage1_ssh_host.as_deref() {
        let (remote, report) = spawn_remote_stage1(
            args,
            host,
            run_id,
            config_path,
            topology_path,
            native_mtp_enabled,
            batched_verify_enabled,
        )?;
        return Ok((StageHandle::Remote(remote), report));
    }

    let local = spawn_stage(
        args,
        config_path,
        None,
        topology_path,
        log_path,
        native_mtp_enabled,
        batched_verify_enabled,
    )?;
    Ok((
        StageHandle::Local(local),
        Stage1LaunchReport {
            launch_mode: "local",
            remote_config: None,
            remote_topology: None,
            remote_log: None,
        },
    ))
}

pub(super) fn spawn_stage(
    args: &NativeMtpOpenAiAbArgs,
    config_path: &Path,
    openai_bind_addr: Option<SocketAddr>,
    topology_path: &Path,
    log_path: &Path,
    native_mtp_enabled: bool,
    batched_verify_enabled: bool,
) -> Result<ChildGuard> {
    let log = File::create(log_path)
        .with_context(|| format!("failed to create {}", log_path.display()))?;
    let mut command = Command::new(&args.server.stage_server_bin);
    command.args([
        "serve-binary",
        "--config",
        config_path
            .to_str()
            .context("stage config path is not valid UTF-8")?,
        "--topology",
        topology_path
            .to_str()
            .context("topology path is not valid UTF-8")?,
        "--activation-width",
        &args.activation_width.to_string(),
        "--activation-wire-dtype",
        &args.activation_wire_dtype,
        "--telemetry-level",
        "debug",
    ]);
    if let Some(openai_bind_addr) = openai_bind_addr {
        command.args(["--openai-bind-addr", &openai_bind_addr.to_string()]);
    }
    command.env("SKIPPY_TELEMETRY_STDERR", "1");
    if native_mtp_enabled {
        command.env(NATIVE_MTP_ENABLED_ENV, "1");
    } else {
        command.env(NATIVE_MTP_ENABLED_ENV, "0");
    }
    if batched_verify_enabled {
        command.env_remove(BATCHED_VERIFY_ENV);
    } else {
        command.env(BATCHED_VERIFY_ENV, "0");
    }
    command.stdout(Stdio::from(log.try_clone()?));
    command.stderr(Stdio::from(log));
    ChildGuard::spawn(command)
}

pub(super) fn wait_stage1_ready(
    stage1: &StageHandle,
    addr: SocketAddr,
    timeout_secs: u64,
) -> Result<()> {
    match stage1 {
        StageHandle::Local(_) | StageHandle::External => {
            drop(connect_ready(addr, timeout_secs)?);
            Ok(())
        }
        StageHandle::Remote(guard) => guard.wait_ready(timeout_secs),
    }
}

pub(super) fn wait_openai_ready(
    client: &Client,
    addr: SocketAddr,
    timeout_secs: u64,
) -> Result<()> {
    let attempts = timeout_secs.saturating_mul(4).max(1);
    let url = format!("http://{addr}/v1/models");
    let mut last_error = None;
    for _ in 0..attempts {
        match client.get(&url).send() {
            Ok(response) if response.status().is_success() => return Ok(()),
            Ok(response) => last_error = Some(format!("HTTP {}", response.status())),
            Err(error) => last_error = Some(error.to_string()),
        }
        thread::sleep(Duration::from_millis(250));
    }
    bail!(
        "timed out waiting for {url}: {}",
        last_error.unwrap_or_else(|| "no attempts made".to_string())
    );
}

pub(super) fn case_addr(addr: SocketAddr, port_offset: u16, case_index: u16) -> Result<SocketAddr> {
    let offset = port_offset
        .checked_mul(case_index)
        .context("case port offset exceeds u16")?;
    offset_port(addr, offset)
}

fn offset_port(mut addr: SocketAddr, offset: u16) -> Result<SocketAddr> {
    let port = addr
        .port()
        .checked_add(offset)
        .context("batched port offset exceeds u16")?;
    addr.set_port(port);
    Ok(addr)
}

pub(super) fn write_stage_config(path: &Path, value: &Value) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(value)?)
        .with_context(|| format!("failed to write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn case_addr_applies_per_case_port_offset() {
        let addr: SocketAddr = "127.0.0.1:9000".parse().unwrap();

        let shifted = case_addr(addr, 10, 2).unwrap();

        assert_eq!(shifted.to_string(), "127.0.0.1:9020");
    }

    #[test]
    fn case_addr_rejects_port_overflow() {
        let addr: SocketAddr = "127.0.0.1:65530".parse().unwrap();

        let error = case_addr(addr, 10, 1).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("batched port offset exceeds u16")
        );
    }
}
