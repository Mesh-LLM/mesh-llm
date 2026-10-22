//! Establish stage lanes and perform their existing ready handshake.
use super::socket::{
    connect_downstream_socket, connect_downstream_socket_cancellable, downstream_source_ip,
    resolve_downstream_endpoint, resolve_downstream_endpoint_cancellable,
};
use crate::stage_transport::StageConnection;
use anyhow::{Context, Result, anyhow, bail};
use skippy_protocol::{
    StageConfig,
    binary::{READY_MAGIC, send_ready},
};
use std::{
    env,
    io::{self, Read, Write},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

const CLIENT_READY_HELLO_ENV: &str = "SKIPPY_STAGE_CLIENT_READY_HELLO";
const CLIENT_READY_HELLO_OPT_IN_PEEK_MS: u64 = 500;
const DOWNSTREAM_SHUTDOWN_POLL: Duration = Duration::from_millis(100);

pub(in crate::binary_transport) fn warm_downstream_preconnect_enabled() -> bool {
    warm_downstream_preconnect_enabled_from(
        env::var("SKIPPY_BINARY_WARM_PRECONNECT").ok().as_deref(),
    )
}

pub(super) fn warm_downstream_preconnect_enabled_from(value: Option<&str>) -> bool {
    value.is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

#[cfg(test)]
pub(in crate::binary_transport) fn take_warm_or_connect_downstream(
    config: &StageConfig,
    transport: Option<&Arc<dyn crate::stage_transport::StageTransport>>,
    warm_downstream: &Arc<Mutex<Option<StageConnection>>>,
    timeout: Duration,
) -> Result<Option<StageConnection>> {
    if config.downstream.is_none() {
        return Ok(None);
    }
    let warm = warm_downstream
        .lock()
        .map_err(|_| anyhow!("warm downstream lock poisoned"))?
        .take();
    match warm {
        Some(stream) if warm_downstream_is_healthy(&stream)? => Ok(Some(stream)),
        Some(_) | None => connect_binary_downstream(config, transport, timeout),
    }
}

fn take_warm_or_connect_downstream_cancellable(
    config: &StageConfig,
    transport: Option<&Arc<dyn crate::stage_transport::StageTransport>>,
    warm_downstream: &Arc<Mutex<Option<StageConnection>>>,
    timeout: Duration,
    shutdown: &AtomicBool,
) -> Result<Option<StageConnection>> {
    ensure_downstream_acquisition_active(shutdown)?;
    let warm = warm_downstream
        .lock()
        .map_err(|_| anyhow!("warm downstream lock poisoned"))?
        .take();
    match warm {
        Some(stream) if warm_downstream_is_healthy(&stream)? => Ok(Some(stream)),
        Some(_) | None => {
            connect_binary_downstream_cancellable(config, transport, timeout, shutdown)
        }
    }
}

pub(in crate::binary_transport) fn take_ready_downstream(
    config: &StageConfig,
    transport: Option<&Arc<dyn crate::stage_transport::StageTransport>>,
    warm_downstream: &Arc<Mutex<Option<StageConnection>>>,
    timeout_secs: u64,
    shutdown: &AtomicBool,
) -> Result<Option<StageConnection>> {
    if config.downstream.is_none() {
        return Ok(None);
    }
    let deadline = Instant::now() + Duration::from_secs(timeout_secs.max(1));
    let mut last_error = None;
    loop {
        ensure_downstream_acquisition_active(shutdown)?;
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }
        match take_warm_or_connect_downstream_cancellable(
            config,
            transport,
            warm_downstream,
            remaining,
            shutdown,
        ) {
            Ok(Some(mut stream)) => {
                let handshake_remaining = deadline.saturating_duration_since(Instant::now());
                if handshake_remaining.is_zero() {
                    last_error = Some(anyhow!("downstream ready deadline expired after connect"));
                    continue;
                }
                match complete_downstream_ready(&mut stream, deadline, shutdown) {
                    Ok(()) => return Ok(Some(stream)),
                    Err(error) => last_error = Some(error),
                }
            }
            Ok(None) => return Ok(None),
            Err(error) => last_error = Some(error),
        }
        ensure_downstream_acquisition_active(shutdown)?;
        let retry_sleep = deadline
            .saturating_duration_since(Instant::now())
            .min(DOWNSTREAM_SHUTDOWN_POLL);
        if !retry_sleep.is_zero() {
            thread::sleep(retry_sleep);
        }
    }
    Err(last_error
        .unwrap_or_else(|| anyhow!("downstream ready deadline expired"))
        .context(format!(
            "downstream stage did not become ready within {}s",
            timeout_secs.max(1)
        )))
}

fn ensure_downstream_acquisition_active(shutdown: &AtomicBool) -> Result<()> {
    if shutdown.load(Ordering::Acquire) {
        bail!("downstream acquisition cancelled during shutdown");
    }
    Ok(())
}

fn complete_downstream_ready(
    stream: &mut StageConnection,
    deadline: Instant,
    shutdown: &AtomicBool,
) -> Result<()> {
    ensure_downstream_acquisition_active(shutdown)?;
    let timeout = deadline
        .saturating_duration_since(Instant::now())
        .min(DOWNSTREAM_SHUTDOWN_POLL);
    if timeout.is_zero() {
        bail!("downstream ready deadline expired before handshake");
    }
    stream
        .set_write_timeout(Some(timeout))
        .context("set downstream ready write timeout")?;
    let hello_result =
        send_client_ready_hello_if_enabled(stream).context("send downstream client ready hello");
    stream
        .set_write_timeout(None)
        .context("clear downstream ready write timeout")?;
    hello_result?;
    stream
        .set_read_timeout(Some(timeout))
        .context("set downstream ready timeout")?;
    let result = (|| -> Result<()> {
        let mut bytes = [0_u8; 4];
        let mut offset = 0;
        while offset < bytes.len() {
            ensure_downstream_acquisition_active(shutdown)?;
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                bail!("downstream ready deadline expired during handshake");
            }
            stream
                .set_read_timeout(Some(remaining.min(DOWNSTREAM_SHUTDOWN_POLL)))
                .context("update downstream ready timeout")?;
            match stream.read(&mut bytes[offset..]) {
                Ok(0) => bail!("downstream binary stage closed before becoming ready"),
                Ok(read) => offset += read,
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::Interrupted
                            | io::ErrorKind::WouldBlock
                            | io::ErrorKind::TimedOut
                    ) => {}
                Err(error) => {
                    return Err(error).context("read downstream binary stage ready handshake");
                }
            }
        }
        if i32::from_le_bytes(bytes) != READY_MAGIC {
            bail!("downstream binary stage ready magic mismatch");
        }
        Ok(())
    })()
    .context("downstream binary stage did not become ready");
    stream
        .set_read_timeout(None)
        .context("clear downstream ready timeout")?;
    result
}

pub(in crate::binary_transport) fn warm_downstream_is_healthy(
    stream: &StageConnection,
) -> Result<bool> {
    let previous_timeout = stream
        .read_timeout()
        .context("read warm downstream timeout")?;
    stream
        .set_read_timeout(Some(Duration::from_millis(1)))
        .context("set warm downstream health-check timeout")?;
    let mut byte = [0_u8; 1];
    let peek_result = stream.peek(&mut byte);
    stream
        .set_read_timeout(previous_timeout)
        .context("restore warm downstream timeout")?;

    Ok(match peek_result {
        Ok(0) => false,
        Ok(_) => true,
        Err(error)
            if matches!(
                error.kind(),
                io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
            ) =>
        {
            true
        }
        Err(_) => false,
    })
}

pub(in crate::binary_transport) fn prepare_binary_stage_connection(
    stream: &StageConnection,
) -> Result<()> {
    stream
        .set_nonblocking(false)
        .context("set binary stage connection blocking")?;
    stream.set_nodelay(true).ok();
    Ok(())
}

pub(crate) fn send_client_ready_hello_if_enabled(stream: &mut StageConnection) -> Result<()> {
    if !client_ready_hello_enabled() {
        return Ok(());
    }
    send_ready(&mut *stream).context("send client ready hello")?;
    stream.flush().ok();
    Ok(())
}

pub(in crate::binary_transport) fn consume_optional_client_ready_hello(
    stream: &mut StageConnection,
) -> Result<()> {
    if !client_ready_hello_enabled() {
        return Ok(());
    }
    let previous_timeout = stream
        .read_timeout()
        .context("read stage connection timeout")?;
    stream
        .set_read_timeout(Some(Duration::from_millis(
            CLIENT_READY_HELLO_OPT_IN_PEEK_MS,
        )))
        .context("set client ready hello peek timeout")?;
    let mut bytes = [0_u8; 4];
    let peek_result = stream.peek(&mut bytes);
    stream
        .set_read_timeout(previous_timeout)
        .context("restore stage connection timeout")?;

    match peek_result {
        Ok(4) if i32::from_le_bytes(bytes) == READY_MAGIC => {
            skippy_protocol::binary::recv_ready(&mut *stream)
                .context("consume client ready hello")?;
            tracing::debug!("binary consumed client ready hello");
        }
        Ok(_) => {}
        Err(error)
            if matches!(
                error.kind(),
                io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
            ) => {}
        Err(error) => return Err(error).context("peek optional client ready hello"),
    }
    Ok(())
}

fn client_ready_hello_enabled() -> bool {
    env::var(CLIENT_READY_HELLO_ENV)
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
        .unwrap_or(false)
}
pub(crate) fn connect_binary_downstream(
    config: &StageConfig,
    transport: Option<&Arc<dyn crate::stage_transport::StageTransport>>,
    timeout: Duration,
) -> Result<Option<StageConnection>> {
    connect_binary_downstream_inner(config, transport, timeout, None)
}

pub(crate) fn connect_binary_downstream_cancellable(
    config: &StageConfig,
    transport: Option<&Arc<dyn crate::stage_transport::StageTransport>>,
    timeout: Duration,
    shutdown: &AtomicBool,
) -> Result<Option<StageConnection>> {
    connect_binary_downstream_inner(config, transport, timeout, Some(shutdown))
}

fn connect_binary_downstream_inner(
    config: &StageConfig,
    transport: Option<&Arc<dyn crate::stage_transport::StageTransport>>,
    timeout: Duration,
    shutdown: Option<&AtomicBool>,
) -> Result<Option<StageConnection>> {
    if let Some(shutdown) = shutdown {
        ensure_downstream_acquisition_active(shutdown)?;
    }
    let Some(peer) = config.downstream.as_ref() else {
        return Ok(None);
    };
    let endpoint = peer
        .endpoint
        .strip_prefix("tcp://")
        .unwrap_or(&peer.endpoint);
    let deadline = Instant::now() + timeout.max(Duration::from_millis(1));
    if let Some(transport) = transport {
        return transport
            .connect(endpoint, deadline, shutdown)
            .map(Some)
            .map_err(Into::into);
    }
    let source_ip = downstream_source_ip(config)?;
    let downstream_addr = match shutdown {
        Some(shutdown) => {
            resolve_downstream_endpoint_cancellable(endpoint, source_ip, deadline, shutdown)?
        }
        None => resolve_downstream_endpoint(endpoint, source_ip)?,
    };
    let mut last_error = None;
    loop {
        if let Some(shutdown) = shutdown {
            ensure_downstream_acquisition_active(shutdown)?;
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }
        let connect_timeout = remaining.min(Duration::from_secs(2));
        let result = match shutdown {
            Some(shutdown) => connect_downstream_socket_cancellable(
                downstream_addr,
                source_ip,
                connect_timeout,
                shutdown,
            ),
            None => connect_downstream_socket(downstream_addr, source_ip, connect_timeout),
        };
        match result {
            Ok(stream) => {
                stream.set_nodelay(true).ok();
                return Ok(Some(stream.into()));
            }
            Err(error) => {
                last_error = Some(anyhow!(error));
                let retry_sleep =
                    deadline
                        .saturating_duration_since(Instant::now())
                        .min(if shutdown.is_some() {
                            DOWNSTREAM_SHUTDOWN_POLL
                        } else {
                            Duration::from_millis(500)
                        });
                if !retry_sleep.is_zero() {
                    thread::sleep(retry_sleep);
                }
            }
        }
    }
    Err(last_error
        .unwrap_or_else(|| anyhow!("timed out"))
        .context(format!(
            "connect downstream binary stage at {endpoint} ({downstream_addr})"
        )))
}
