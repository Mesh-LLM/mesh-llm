use std::{fs, path::Path, process::Command, thread, time::Duration};

use anyhow::{Context, Result, bail};

use crate::cli::NativeMtpOpenAiAbArgs;

use super::stage_process::Stage1LaunchReport;

pub(super) struct RemoteStageGuard {
    host: String,
    pid: Option<u32>,
    remote_log: String,
}

pub(super) fn spawn_remote_stage1(
    args: &NativeMtpOpenAiAbArgs,
    host: &str,
    run_id: &str,
    config_path: &Path,
    topology_path: &Path,
    native_mtp_enabled: bool,
    batched_verify_enabled: bool,
) -> Result<(RemoteStageGuard, Stage1LaunchReport)> {
    let remote_dir = format!(
        "{}/{}",
        args.stage1_remote_root.trim_end_matches('/'),
        run_id
    );
    let remote_config = format!("{remote_dir}/stage1.json");
    let remote_topology = format!("{remote_dir}/topology.json");
    let remote_log = format!("{remote_dir}/stage1.log");
    let remote_pid = format!("{remote_dir}/stage1.pid");
    let remote_bin = args
        .stage1_remote_stage_server_bin
        .as_deref()
        .context("missing stage 1 remote binary")?;

    ssh_success(host, &format!("mkdir -p {}", shell_quote(&remote_dir)))
        .with_context(|| format!("create remote stage directory on {host}"))?;
    scp_to(host, config_path, &remote_config).with_context(|| {
        format!(
            "copy stage 1 config {} to {host}:{remote_config}",
            config_path.display()
        )
    })?;
    scp_to(host, topology_path, &remote_topology).with_context(|| {
        format!(
            "copy topology {} to {host}:{remote_topology}",
            topology_path.display()
        )
    })?;

    let command = remote_stage_command(RemoteStageCommand {
        workdir: args.stage1_remote_workdir.as_deref(),
        remote_bin,
        remote_config: &remote_config,
        remote_topology: &remote_topology,
        remote_log: &remote_log,
        remote_pid: &remote_pid,
        activation_width: args.activation_width,
        activation_wire_dtype: &args.activation_wire_dtype,
        native_mtp_enabled,
        batched_verify_enabled,
    });
    let output = Command::new("ssh")
        .arg(host)
        .arg(command)
        .output()
        .with_context(|| format!("start remote stage 1 on {host}"))?;
    if !output.status.success() {
        bail!(
            "remote stage 1 start on {host} failed with status {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let pid = stdout
        .lines()
        .rev()
        .find_map(|line| line.trim().parse::<u32>().ok())
        .with_context(|| format!("remote stage 1 on {host} did not print a pid"))?;

    Ok((
        RemoteStageGuard {
            host: host.to_string(),
            pid: Some(pid),
            remote_log: remote_log.clone(),
        },
        Stage1LaunchReport {
            launch_mode: "ssh",
            remote_config: Some(remote_config),
            remote_topology: Some(remote_topology),
            remote_log: Some(remote_log),
        },
    ))
}

struct RemoteStageCommand<'a> {
    workdir: Option<&'a str>,
    remote_bin: &'a str,
    remote_config: &'a str,
    remote_topology: &'a str,
    remote_log: &'a str,
    remote_pid: &'a str,
    activation_width: i32,
    activation_wire_dtype: &'a str,
    native_mtp_enabled: bool,
    batched_verify_enabled: bool,
}

fn remote_stage_command(args: RemoteStageCommand<'_>) -> String {
    let mut command = String::new();
    if let Some(workdir) = args.workdir {
        command.push_str("cd ");
        command.push_str(&shell_quote(workdir));
        command.push_str(" && ");
    }
    command.push_str("SKIPPY_TELEMETRY_STDERR=1 ");
    command.push_str("SKIPPY_NATIVE_MTP_ENABLED=");
    command.push_str(if args.native_mtp_enabled { "1 " } else { "0 " });
    if !args.batched_verify_enabled {
        command.push_str("SKIPPY_NATIVE_MTP_BATCHED_VERIFY=0 ");
    }
    command.push_str("nohup ");
    command.push_str(&shell_quote(args.remote_bin));
    command.push_str(" serve-binary --config ");
    command.push_str(&shell_quote(args.remote_config));
    command.push_str(" --topology ");
    command.push_str(&shell_quote(args.remote_topology));
    command.push_str(" --activation-width ");
    command.push_str(&args.activation_width.to_string());
    command.push_str(" --activation-wire-dtype ");
    command.push_str(&shell_quote(args.activation_wire_dtype));
    command.push_str(" --telemetry-level debug > ");
    command.push_str(&shell_quote(args.remote_log));
    command.push_str(" 2>&1 < /dev/null & pid=$!; echo $pid > ");
    command.push_str(&shell_quote(args.remote_pid));
    command.push_str("; echo $pid");
    command
}

impl RemoteStageGuard {
    pub(super) fn wait_ready(&self, timeout_secs: u64) -> Result<()> {
        let Some(pid) = self.pid else {
            bail!("remote stage 1 has no pid");
        };
        let attempts = timeout_secs.saturating_mul(2).max(1);
        let log = shell_quote(&self.remote_log);
        let mut last_stderr = String::new();
        for _ in 0..attempts {
            let command = format!(
                "if ! kill -0 {pid} 2>/dev/null; then echo dead; exit 2; fi; \
                 grep -q 'skippy-server listening:' {log}"
            );
            let output = Command::new("ssh")
                .arg(&self.host)
                .arg(&command)
                .output()
                .with_context(|| format!("check remote stage 1 readiness on {}", self.host))?;
            if output.status.success() {
                return Ok(());
            }
            if output.status.code() == Some(2) {
                bail!("remote stage 1 on {} exited before readiness", self.host);
            }
            last_stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            thread::sleep(Duration::from_millis(500));
        }
        bail!(
            "timed out waiting for remote stage 1 on {} to log readiness{}",
            self.host,
            if last_stderr.is_empty() {
                String::new()
            } else {
                format!(": {last_stderr}")
            }
        )
    }

    pub(super) fn stop_and_collect(&mut self, local_log: &Path) -> Result<()> {
        self.terminate();
        match scp_from(&self.host, &self.remote_log, local_log) {
            Ok(()) => Ok(()),
            Err(error) => {
                let note = format!("failed to collect remote stage 1 log: {error:#}\n");
                fs::write(local_log, note)
                    .with_context(|| format!("failed to write {}", local_log.display()))
            }
        }
    }

    fn terminate(&mut self) {
        let Some(pid) = self.pid.take() else {
            return;
        };
        let command = format!("kill {pid} 2>/dev/null || true; sleep 0.2");
        let _ = Command::new("ssh").arg(&self.host).arg(command).status();
    }
}

impl Drop for RemoteStageGuard {
    fn drop(&mut self) {
        self.terminate();
    }
}

fn ssh_success(host: &str, remote_command: &str) -> Result<()> {
    let status = Command::new("ssh")
        .arg(host)
        .arg(remote_command)
        .status()
        .with_context(|| format!("run ssh command on {host}"))?;
    if !status.success() {
        bail!("ssh command on {host} failed with status {status}");
    }
    Ok(())
}

fn scp_to(host: &str, local_path: &Path, remote_path: &str) -> Result<()> {
    let status = Command::new("scp")
        .arg(local_path)
        .arg(format!("{host}:{remote_path}"))
        .status()
        .with_context(|| format!("copy {} to {host}:{remote_path}", local_path.display()))?;
    if !status.success() {
        bail!("scp to {host}:{remote_path} failed with status {status}");
    }
    Ok(())
}

fn scp_from(host: &str, remote_path: &str, local_path: &Path) -> Result<()> {
    let status = Command::new("scp")
        .arg(format!("{host}:{remote_path}"))
        .arg(local_path)
        .status()
        .with_context(|| format!("copy {host}:{remote_path} to {}", local_path.display()))?;
    if !status.success() {
        bail!("scp from {host}:{remote_path} failed with status {status}");
    }
    Ok(())
}

fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_stage_command_quotes_paths_and_sets_mtp_env() {
        let command = remote_stage_command(RemoteStageCommand {
            workdir: Some("/tmp/work dir"),
            remote_bin: "target/debug/skippy-server",
            remote_config: "/tmp/run/stage'1.json",
            remote_topology: "/tmp/run/topology.json",
            remote_log: "/tmp/run/stage1.log",
            remote_pid: "/tmp/run/stage1.pid",
            activation_width: 2048,
            activation_wire_dtype: "f16",
            native_mtp_enabled: true,
            batched_verify_enabled: false,
        });

        assert!(command.contains("cd '/tmp/work dir' && "));
        assert!(command.contains("SKIPPY_NATIVE_MTP_ENABLED=1"));
        assert!(command.contains("SKIPPY_NATIVE_MTP_BATCHED_VERIFY=0"));
        assert!(command.contains("'target/debug/skippy-server' serve-binary"));
        assert!(command.contains("'/tmp/run/stage'\"'\"'1.json'"));

        let batched_command = remote_stage_command(RemoteStageCommand {
            batched_verify_enabled: true,
            ..RemoteStageCommand {
                workdir: None,
                remote_bin: "/bin/skippy-server",
                remote_config: "/tmp/stage1.json",
                remote_topology: "/tmp/topology.json",
                remote_log: "/tmp/stage1.log",
                remote_pid: "/tmp/stage1.pid",
                activation_width: 2048,
                activation_wire_dtype: "f16",
                native_mtp_enabled: true,
                batched_verify_enabled: false,
            }
        });
        assert!(!batched_command.contains("SKIPPY_NATIVE_MTP_BATCHED_VERIFY"));
    }
}
