//! Open prediction-return connections through the stage's supplied transport.
use super::{
    direct_return::prediction_return_open_message,
    socket::{connect_downstream_socket, downstream_source_ip, resolve_downstream_endpoint},
    stage_connections::send_client_ready_hello_if_enabled,
};
use crate::stage_transport::StageConnection;
use anyhow::{Context, Result, anyhow};
use skippy_protocol::{
    StageConfig, StageTopology,
    binary::{StageReply, recv_ready, send_reply_message, write_stage_message},
};
use std::{
    net::{IpAddr, SocketAddr},
    sync::Arc,
    time::{Duration, Instant},
};

// Match the forward-lane cold-setup budget and bound a stalled ready handshake.
const RETURN_SINK_READY_READ_TIMEOUT: Duration = Duration::from_secs(20);

/// Connect to `return_addr`, complete the ready handshake, and send the
/// prediction-return open message. Single bounded attempt — on failure the
/// caller falls back to the upstream reply path.
fn open_return_sink_once(
    return_addr: SocketAddr,
    source_ip: Option<IpAddr>,
    request_id: u64,
    session_id: u64,
    not_ready_context: &'static str,
) -> Result<StageConnection> {
    let stream = connect_downstream_socket(return_addr, source_ip, Duration::from_secs(2))
        .map_err(|error| anyhow!(error))?;
    finish_return_open(stream.into(), request_id, session_id, not_ready_context)
}

fn finish_return_open(
    mut stream: StageConnection,
    request_id: u64,
    session_id: u64,
    not_ready_context: &'static str,
) -> Result<StageConnection> {
    stream.set_nodelay(true).ok();
    send_client_ready_hello_if_enabled(&mut stream)
        .context("send prediction return client ready hello")?;
    // Bound the ready handshake read. `recv_ready` is a blocking `read_exact`;
    // without a timeout a stalled downstream connection hangs the return-sink
    // open forever, blocking generation from falling back to the upstream reply.
    // A single short deadline (no outer retry) fails fast to that fallback.
    // Both the set and the clear are propagated: if the set fails, `recv_ready`
    // would be unbounded (defeating the fix); if the clear fails, the handshake
    // timeout would leak into the sink's later reads.
    stream
        .set_read_timeout(Some(RETURN_SINK_READY_READ_TIMEOUT))
        .context("set prediction return ready read timeout")?;
    let ready = recv_ready(&mut stream).context(not_ready_context);
    stream
        .set_read_timeout(None)
        .context("clear prediction return ready read timeout")?;
    ready?;
    write_stage_message(
        &mut stream,
        &prediction_return_open_message(request_id, session_id),
    )
    .context("open prediction return stream")?;
    Ok(stream)
}

pub(crate) fn open_prediction_return_stream(
    config: &StageConfig,
    transport: Option<&Arc<dyn crate::stage_transport::StageTransport>>,
    topology: Option<&StageTopology>,
    request_id: u64,
    session_id: u64,
    _timeout_secs: u64,
) -> Result<StageConnection> {
    let endpoint = driver_stage_endpoint(config, topology)?;
    if let Some(transport) = transport {
        let stream = transport.connect(
            endpoint,
            Instant::now() + RETURN_SINK_READY_READ_TIMEOUT,
            None,
        )?;
        return finish_return_open(
            stream,
            request_id,
            session_id,
            "prediction return sink did not become ready",
        );
    }
    let source_ip = downstream_source_ip(config)?;
    let return_addr = resolve_downstream_endpoint(endpoint, source_ip)?;
    open_return_sink_once(
        return_addr,
        source_ip,
        request_id,
        session_id,
        "prediction return sink did not become ready",
    )
    .with_context(|| format!("connect direct prediction return sink at {endpoint}"))
}

pub(crate) fn open_downstream_prediction_return_stream(
    config: &StageConfig,
    transport: Option<&Arc<dyn crate::stage_transport::StageTransport>>,
    request_id: u64,
    session_id: u64,
) -> Result<StageConnection> {
    let downstream = config
        .downstream
        .as_ref()
        .ok_or_else(|| anyhow!("direct prediction return requires downstream stage"))?;
    let endpoint = strip_tcp_prefix(&downstream.endpoint);
    if let Some(transport) = transport {
        let stream = transport.connect(
            endpoint,
            Instant::now() + RETURN_SINK_READY_READ_TIMEOUT,
            None,
        )?;
        return finish_return_open(
            stream,
            request_id,
            session_id,
            "downstream prediction return sink did not become ready",
        );
    }
    let source_ip = downstream_source_ip(config)?;
    let return_addr = resolve_downstream_endpoint(endpoint, source_ip)?;
    open_return_sink_once(
        return_addr,
        source_ip,
        request_id,
        session_id,
        "downstream prediction return sink did not become ready",
    )
    .with_context(|| format!("connect downstream prediction return sink at {endpoint}"))
}

pub(crate) fn send_direct_prediction_return(
    stream: &mut StageConnection,
    reply: StageReply,
) -> Result<()> {
    send_reply_message(stream, &reply).context("send direct prediction return")
}

fn driver_stage_endpoint<'a>(
    config: &'a StageConfig,
    topology: Option<&'a StageTopology>,
) -> Result<&'a str> {
    if let Some(topology) = topology {
        return driver_stage_endpoint_from_topology(topology);
    }
    if let Some(upstream) = config
        .upstream
        .as_ref()
        .filter(|upstream| upstream.stage_index == 0)
    {
        return Ok(strip_tcp_prefix(&upstream.endpoint));
    }
    Err(anyhow!("direct prediction return requires topology"))
}

fn driver_stage_endpoint_from_topology(topology: &StageTopology) -> Result<&str> {
    topology
        .stages
        .iter()
        .find(|stage| stage.stage_index == 0)
        .map(|stage| strip_tcp_prefix(&stage.endpoint))
        .ok_or_else(|| anyhow!("topology does not contain driver-facing stage 0"))
}

fn strip_tcp_prefix(endpoint: &str) -> &str {
    endpoint.strip_prefix("tcp://").unwrap_or(endpoint)
}
