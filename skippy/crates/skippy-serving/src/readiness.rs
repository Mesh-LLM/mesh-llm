//! Binary-stage readiness handshakes and cancellable startup waits.
use anyhow::{Context, Result, anyhow};
use std::{
    net::SocketAddr,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};
use tokio::task::JoinHandle;

/// Owned startup probe. Keep it registered until completion or cancel and join it.
pub struct StageReadinessProbe {
    cancelled: Arc<AtomicBool>,
    handle: JoinHandle<Result<()>>,
}

impl StageReadinessProbe {
    /// Wait for a wire-level readiness handshake. May be interrupted and resumed.
    pub async fn wait(&mut self) -> Result<()> {
        (&mut self.handle)
            .await
            .context("join binary stage readiness probe")?
    }

    /// Cancel startup and join its blocking worker before returning.
    pub async fn cancel_and_join(mut self) {
        self.cancelled.store(true, Ordering::Release);
        let _ = (&mut self.handle).await;
    }
}

pub fn parse_bind_addr(bind_addr: &str) -> Result<SocketAddr> {
    bind_addr
        .parse()
        .with_context(|| format!("parse stage bind_addr {bind_addr:?}"))
}

pub fn materialize_stage_bind_addr(bind_addr: SocketAddr) -> Result<SocketAddr> {
    if bind_addr.port() != 0 {
        return Ok(bind_addr);
    }
    let listener = std::net::TcpListener::bind(bind_addr)
        .with_context(|| format!("reserve ephemeral stage bind address for {bind_addr}"))?;
    listener
        .local_addr()
        .context("read reserved ephemeral stage bind address")
}

pub fn start_binary_stage_ready_probe(
    bind_addr: SocketAddr,
    timeout: Duration,
) -> StageReadinessProbe {
    let cancelled = Arc::new(AtomicBool::new(false));
    let probe_cancelled = Arc::clone(&cancelled);
    let handle = tokio::task::spawn_blocking(move || {
        probe_binary_stage_ready(bind_addr, timeout, &probe_cancelled)
    });
    StageReadinessProbe { cancelled, handle }
}

pub fn stage_load_timeout(source_model_bytes: Option<u64>) -> Duration {
    const MIN_STAGE_LOAD_TIMEOUT_SECS: u64 = 900;
    const MAX_STAGE_LOAD_TIMEOUT_SECS: u64 = 4 * 60 * 60;
    const STAGE_LOAD_BYTES_PER_SEC: u64 = 128 * 1024 * 1024;

    let scaled_secs = source_model_bytes
        .map(|bytes| {
            bytes.saturating_add(STAGE_LOAD_BYTES_PER_SEC.saturating_sub(1))
                / STAGE_LOAD_BYTES_PER_SEC
        })
        .unwrap_or(MIN_STAGE_LOAD_TIMEOUT_SECS);
    Duration::from_secs(
        MIN_STAGE_LOAD_TIMEOUT_SECS
            .max(scaled_secs)
            .min(MAX_STAGE_LOAD_TIMEOUT_SECS),
    )
}

fn probe_binary_stage_ready(
    bind_addr: SocketAddr,
    timeout: Duration,
    cancelled: &AtomicBool,
) -> Result<()> {
    const PROBE_IO_TIMEOUT: Duration = Duration::from_secs(2);
    let deadline = std::time::Instant::now() + timeout;
    let mut last_error = None;
    while std::time::Instant::now() < deadline {
        if cancelled.load(Ordering::Acquire) {
            return Err(anyhow!("binary stage readiness probe cancelled"));
        }
        match std::net::TcpStream::connect_timeout(&bind_addr, PROBE_IO_TIMEOUT) {
            Ok(mut stream) => {
                stream.set_nodelay(true).ok();
                stream.set_read_timeout(Some(PROBE_IO_TIMEOUT)).ok();
                stream.set_write_timeout(Some(PROBE_IO_TIMEOUT)).ok();
                match skippy_protocol::binary::recv_ready(&mut stream) {
                    Ok(()) => return Ok(()),
                    Err(error) => {
                        last_error =
                            Some(anyhow!(error).context("binary stage ready handshake failed"));
                    }
                }
            }
            Err(error) => {
                last_error = Some(anyhow!(error).context("connect binary stage listener"));
            }
        }
        for _ in 0..25 {
            if cancelled.load(Ordering::Acquire) {
                return Err(anyhow!("binary stage readiness probe cancelled"));
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
    Err(last_error
        .unwrap_or_else(|| anyhow!("timed out waiting for binary stage ready at {bind_addr}"))
        .context(format!(
            "binary stage did not become ready at {bind_addr} before timeout"
        )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;
    #[test]
    fn materialize_stage_bind_addr_replaces_ephemeral_port() {
        let bind_addr = materialize_stage_bind_addr("127.0.0.1:0".parse().unwrap()).unwrap();
        assert_eq!(bind_addr.ip().to_string(), "127.0.0.1");
        assert_ne!(bind_addr.port(), 0);
    }

    #[tokio::test]
    async fn binary_stage_ready_probe_waits_for_wire_handshake() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let bind_addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(75));
            let (mut stream, _) = listener.accept().unwrap();
            skippy_protocol::binary::send_ready(&mut stream).unwrap();
        });

        let started = Instant::now();
        let mut probe = start_binary_stage_ready_probe(bind_addr, Duration::from_secs(2));
        probe.wait().await.unwrap();
        assert!(started.elapsed() >= Duration::from_millis(50));
        server.join().unwrap();
    }

    #[test]
    fn load_timeout_preserves_floor_scaling_and_cap() {
        assert_eq!(stage_load_timeout(None), Duration::from_secs(900));
        assert_eq!(
            stage_load_timeout(Some(170 * 1024 * 1024 * 1024)),
            Duration::from_secs(1360)
        );
        assert_eq!(
            stage_load_timeout(Some(u64::MAX)),
            Duration::from_secs(14400)
        );
    }

    #[tokio::test]
    async fn readiness_timeout_requires_a_wire_handshake() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            drop(stream);
        });
        let mut probe = start_binary_stage_ready_probe(address, Duration::from_millis(100));
        let error = probe.wait().await.unwrap_err();
        assert!(format!("{error:#}").contains("binary stage did not become ready"));
        server.join().unwrap();
    }
}
