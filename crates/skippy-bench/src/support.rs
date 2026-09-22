use skippy_protocol::binary::StageStream as TcpStream;
use std::{
    net::SocketAddr,
    path::{Path, PathBuf},
    process::{Child, Command},
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow, bail};
use skippy_protocol::binary::{
    ActivationAgreement, ActivationDimension, ActivationPartProfile, ActivationProfile,
    ConnectionRole, StageActivationDesc, StageActivationPartDesc, client_setup,
    encode_activation_frame,
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

    pub fn keep_alive(self) {
        std::mem::forget(self);
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub fn retry(timeout_secs: u64, mut action: impl FnMut() -> Result<()>) -> Result<()> {
    let attempts = timeout_secs.saturating_mul(2).max(1);
    let mut last_error = None;
    for _ in 0..attempts {
        match action() {
            Ok(()) => return Ok(()),
            Err(error) => last_error = Some(error),
        }
        thread::sleep(Duration::from_millis(500));
    }
    Err(last_error.unwrap_or_else(|| anyhow!("timed out")))
}

pub fn ensure_release_skippy_server_bin(path: &Path) -> Result<()> {
    let path = path.to_string_lossy();
    let debug_path = path.contains("target/debug/skippy-server")
        || path.contains("target\\debug\\skippy-server");
    if debug_path {
        bail!(
            "SkippyBench benchmark-managed skippy-server runs require a release binary; run `just release-build` and use --stage-server-bin target/release/skippy-server"
        );
    }
    Ok(())
}

pub fn connect_ready(
    addr: SocketAddr,
    timeout_secs: u64,
    receiver: &serde_json::Value,
    boundary: skippy_runtime::ActivationBoundaryDesc,
) -> Result<TcpStream> {
    let receiver: skippy_protocol::StageConfig = serde_json::from_value(receiver.clone())?;
    let mut source = receiver.clone();
    source.stage_index = receiver
        .stage_index
        .checked_sub(1)
        .context("activation driver needs a downstream stage")?;
    source.stage_id = receiver
        .upstream
        .as_ref()
        .context("driver upstream identity absent")?
        .stage_id
        .clone();
    source.layer_start = 0;
    source.layer_end = receiver.layer_start;
    source.downstream = Some(skippy_protocol::PeerConfig {
        stage_id: receiver.stage_id,
        stage_index: receiver.stage_index,
        endpoint: addr.to_string(),
    });
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
                            max: skippy_protocol::binary::MAX_STAGE_DECODED_ACTIVATION_BYTES as u64,
                        }
                    }
                }),
            })
            .collect(),
    };
    ActivationAgreement {
        generation: [1; 16],
        profiles: vec![profile.clone()],
    }
    .validate()?;
    let deadline = std::time::Instant::now() + Duration::from_secs(timeout_secs);
    let shutdown = std::sync::atomic::AtomicBool::new(false);
    let mut last_error = None;
    while std::time::Instant::now() < deadline {
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        match TcpStream::connect_timeout(&addr, remaining.min(Duration::from_secs(1))) {
            Ok(mut stream) => {
                stream.set_nodelay(true)?;
                match client_setup(
                    &mut stream,
                    ConnectionRole::Activation,
                    Some(&source),
                    Some(profile.clone()),
                    deadline,
                    &shutdown,
                ) {
                    Ok(()) => return Ok(stream),
                    Err(error) => last_error = Some(error.context("activation setup failed")),
                }
            }
            Err(error) => last_error = Some(anyhow!(error).context("connect failed")),
        }
        thread::sleep(
            Duration::from_millis(100)
                .min(deadline.saturating_duration_since(std::time::Instant::now())),
        );
    }
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
    format!("run-local-single-{millis}")
}

pub fn temp_db_path(run_id: &str) -> PathBuf {
    std::env::temp_dir().join(format!("{run_id}.sqlite"))
}

pub fn temp_config_path(run_id: &str) -> PathBuf {
    std::env::temp_dir().join(format!("{run_id}-stage-0.json"))
}

pub fn temp_config_path_for(run_id: &str, stage_id: &str) -> PathBuf {
    std::env::temp_dir().join(format!("{run_id}-{stage_id}.json"))
}
