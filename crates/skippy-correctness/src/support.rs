use std::{
    net::{SocketAddr, TcpStream},
    path::PathBuf,
    process::{Child, Command, ExitStatus},
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow, bail};
use skippy_protocol::binary::{
    activation_payload_multiplier_from_state_flags, activation_state_flags_from_frame_flags,
    recv_ready,
};

pub struct ChildGuard {
    child: Child,
}

impl ChildGuard {
    pub fn spawn(mut command: Command) -> Result<Self> {
        let child = command
            .spawn()
            .with_context(|| format!("failed to spawn {:?}", command))?;
        Ok(Self { child })
    }

    pub fn try_wait(&mut self) -> Result<Option<ExitStatus>> {
        self.child.try_wait().context("poll child process")
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub fn connect_ready(addr: SocketAddr, timeout_secs: u64) -> Result<TcpStream> {
    connect_ready_until(addr, timeout_secs, || Ok(()))
}

pub fn connect_ready_child(
    addr: SocketAddr,
    timeout_secs: u64,
    child: &mut ChildGuard,
) -> Result<TcpStream> {
    connect_ready_until(addr, timeout_secs, || {
        if let Some(status) = child.try_wait()? {
            bail!("child process exited before readiness with status {status}");
        }
        Ok(())
    })
}

fn connect_ready_until(
    addr: SocketAddr,
    timeout_secs: u64,
    mut check_child: impl FnMut() -> Result<()>,
) -> Result<TcpStream> {
    let attempts = timeout_secs.saturating_mul(2).max(1);
    let mut last_error = None;
    for _ in 0..attempts {
        check_child()?;
        match TcpStream::connect(addr) {
            Ok(mut stream) => {
                stream.set_nodelay(true).ok();
                stream
                    .set_read_timeout(Some(Duration::from_millis(500)))
                    .ok();
                stream
                    .set_write_timeout(Some(Duration::from_millis(500)))
                    .ok();
                match recv_ready(&mut stream) {
                    Ok(()) => {
                        stream
                            .set_read_timeout(Some(Duration::from_secs(timeout_secs.max(1))))
                            .ok();
                        stream
                            .set_write_timeout(Some(Duration::from_secs(timeout_secs.max(1))))
                            .ok();
                        return Ok(stream);
                    }
                    Err(error) => {
                        last_error = Some(anyhow!(error).context("ready handshake failed"))
                    }
                }
            }
            Err(error) => last_error = Some(anyhow!(error).context("connect failed")),
        }
        thread::sleep(Duration::from_millis(500));
    }
    check_child()?;
    Err(last_error.unwrap_or_else(|| anyhow!("timed out")))
}

pub fn activation_width(frame: &skippy_runtime::ActivationFrame) -> Result<i32> {
    if frame.desc.token_count == 0 {
        bail!("activation frame token_count is zero");
    }
    let bytes_per_token = frame
        .payload
        .len()
        .checked_div(frame.desc.token_count as usize)
        .context("activation token_count overflow")?;
    let payload_multiplier = activation_payload_multiplier_from_state_flags(
        activation_state_flags_from_frame_flags(frame.desc.flags),
    );
    let bytes_per_hidden_token = bytes_per_token
        .checked_div(payload_multiplier)
        .context("activation sideband multiplier overflow")?;
    if !bytes_per_token.is_multiple_of(payload_multiplier)
        || !bytes_per_hidden_token.is_multiple_of(4)
    {
        bail!("activation payload is not F32 aligned");
    }
    i32::try_from(bytes_per_hidden_token / 4).context("activation width exceeds i32")
}

pub fn generate_run_id() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before Unix epoch")
        .as_millis();
    format!("correctness-{millis}")
}

pub fn temp_config_path_for(run_id: &str, stage_id: &str) -> PathBuf {
    std::env::temp_dir().join(format!("{run_id}-{stage_id}.json"))
}

#[cfg(test)]
mod tests {
    use std::{
        net::{SocketAddr, TcpListener},
        process::{Command, Stdio},
    };

    use super::{ChildGuard, connect_ready_child};

    #[test]
    fn readiness_reports_a_child_that_exits_before_listening() -> anyhow::Result<()> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let addr: SocketAddr = listener.local_addr()?;
        drop(listener);

        let mut command = Command::new(std::env::current_exe()?);
        command
            .arg("--list")
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        let mut child = ChildGuard::spawn(command)?;

        let error = connect_ready_child(addr, 10, &mut child)
            .expect_err("short-lived child must be reported")
            .to_string();

        assert!(
            error.contains("child process exited before readiness with status"),
            "unexpected error: {error}"
        );
        Ok(())
    }
}
