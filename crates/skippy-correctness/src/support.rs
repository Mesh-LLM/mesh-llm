use skippy_protocol::binary::StageStream as TcpStream;
use std::{
    net::SocketAddr,
    path::PathBuf,
    process::{Child, Command, ExitStatus},
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow, bail};
use skippy_protocol::binary::{
    StageActivationDesc, StageActivationPartDesc, encode_activation_frame,
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
    connect_ready_until(addr, timeout_secs, || Ok(()), None)
}

pub fn connect_ready_child(
    addr: SocketAddr,
    timeout_secs: u64,
    child: &mut ChildGuard,
) -> Result<TcpStream> {
    connect_ready_until(
        addr,
        timeout_secs,
        || {
            if let Some(status) = child.try_wait()? {
                bail!("child process exited before readiness with status {status}");
            }
            Ok(())
        },
        None,
    )
}

pub fn connect_ready_child_with_boundary(
    addr: SocketAddr,
    timeout_secs: u64,
    child: &mut ChildGuard,
    receiver: &serde_json::Value,
    boundary: Option<skippy_runtime::ActivationBoundaryDesc>,
) -> Result<TcpStream> {
    let Some(boundary) = boundary else {
        return connect_ready_child(addr, timeout_secs, child);
    };
    let receiver: skippy_protocol::StageConfig = serde_json::from_value(receiver.clone())?;
    let mut source = receiver.clone();
    let upstream = receiver
        .upstream
        .clone()
        .unwrap_or(skippy_protocol::PeerConfig {
            stage_id: "stage-0".into(),
            stage_index: receiver.stage_index.saturating_sub(1),
            endpoint: "driver".into(),
        });
    source.stage_id = upstream.stage_id;
    source.stage_index = upstream.stage_index;
    source.layer_start = 0;
    source.layer_end = receiver.layer_start;
    source.downstream = Some(skippy_protocol::PeerConfig {
        stage_id: receiver.stage_id,
        stage_index: receiver.stage_index,
        endpoint: addr.to_string(),
    });
    use skippy_protocol::binary::{
        ActivationDimension, ActivationPartProfile, ActivationProfile,
        MAX_STAGE_DECODED_ACTIVATION_BYTES,
    };
    let profile = ActivationProfile {
        id: 1,
        producer_stage_index: source.stage_index as i32,
        layer_start: 0,
        layer_end: source.layer_end as i32,
        frontier_identity: boundary.frontier_identity,
        max_tokens: i32::MAX as u32,
        max_sequences: u32::MAX,
        parts: boundary
            .parts()?
            .iter()
            .map(|p| ActivationPartProfile {
                identity: p.identity,
                ggml_type: p.ggml_type,
                rank: p.rank,
                token_axis: p.token_axis as u32,
                optional: p.flags & skippy_protocol::binary::STAGE_ACTIVATION_PART_OPTIONAL != 0,
                dimensions: std::array::from_fn(|axis| {
                    if axis == p.token_axis as usize {
                        ActivationDimension::Tokens
                    } else if axis >= p.rank as usize {
                        ActivationDimension::Fixed(1)
                    } else if p.dimensions[axis] > 0 {
                        ActivationDimension::Fixed(p.dimensions[axis] as u64)
                    } else {
                        ActivationDimension::Dynamic {
                            min: 1,
                            max: MAX_STAGE_DECODED_ACTIVATION_BYTES as u64,
                        }
                    }
                }),
            })
            .collect(),
    };
    connect_ready_until(
        addr,
        timeout_secs,
        || {
            if let Some(status) = child.try_wait()? {
                bail!("stage server exited before ready: {status}");
            }
            Ok(())
        },
        Some((&source, &profile)),
    )
}

fn connect_ready_until(
    addr: SocketAddr,
    timeout_secs: u64,
    mut check_child: impl FnMut() -> Result<()>,
    setup: Option<(
        &skippy_protocol::StageConfig,
        &skippy_protocol::binary::ActivationProfile,
    )>,
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
                match skippy_protocol::binary::client_setup(
                    &mut stream,
                    if setup.is_some() {
                        skippy_protocol::binary::ConnectionRole::Activation
                    } else {
                        skippy_protocol::binary::ConnectionRole::TokensAndControl
                    },
                    setup.map(|s| s.0),
                    setup.map(|s| s.1.clone()),
                    std::time::Instant::now() + Duration::from_millis(500),
                    &std::sync::atomic::AtomicBool::new(false),
                ) {
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
    let primary = frame
        .desc
        .parts()?
        .first()
        .context("activation frame has no primary part")?;
    anyhow::ensure!(
        primary.ggml_type == skippy_runtime::GGML_TYPE_F32,
        "primary activation part is not F32"
    );
    let rank = usize::try_from(primary.rank).context("primary activation rank exceeds usize")?;
    anyhow::ensure!(
        rank > 0 && rank <= primary.dimensions.len(),
        "primary activation part has an invalid rank"
    );
    let token_axis = usize::try_from(primary.token_axis)
        .context("primary activation part has a negative token axis")?;
    anyhow::ensure!(
        token_axis < rank,
        "primary activation part has an invalid token axis"
    );
    let mut width = 1_u64;
    for (axis, dimension) in primary.dimensions.iter().copied().enumerate().take(rank) {
        if axis != token_axis {
            width = width
                .checked_mul(u64::try_from(dimension).context("invalid activation dimension")?)
                .context("activation width overflow")?;
        }
    }
    i32::try_from(width).context("activation width exceeds i32")
}

pub fn encode_runtime_activation(
    codec: skippy_protocol::StageActivationCodec,
    frame: &skippy_runtime::ActivationFrame,
) -> Result<Vec<u8>> {
    let desc = StageActivationDesc {
        version: frame.desc.version,
        producer_stage_index: frame.desc.producer_stage_index,
        layer_start: frame.desc.layer_start,
        layer_end: frame.desc.layer_end,
        token_count: frame.desc.token_count,
        sequence_count: frame.desc.sequence_count,
        payload_bytes: frame.desc.payload_bytes,
        frontier_identity: frame.desc.frontier_identity,
        parts: frame
            .desc
            .parts()?
            .iter()
            .map(|part| StageActivationPartDesc {
                identity: part.identity,
                ggml_type: part.ggml_type,
                rank: part.rank,
                token_axis: part.token_axis,
                flags: part.flags,
                dimensions: part.dimensions,
                byte_strides: part.byte_strides,
                payload_offset: part.payload_offset,
                payload_bytes: part.payload_bytes,
            })
            .collect(),
    };
    encode_activation_frame(codec, &desc, &frame.payload)
        .context("failed to encode multipart activation frame")
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
