use crate::binary_transport::connect_binary_downstream;
use crate::binary_transport::send_client_ready_hello_if_enabled;
use crate::frontend::generation::OpenAiGenerationIds;
use crate::frontend::generation::PhaseTimer;
use crate::frontend::prefill::PrefillChunkObservation;
use crate::frontend::util::openai_backend_error;
use crate::frontend::util::us_to_ms;
use crate::telemetry::Telemetry;
use crate::telemetry::lifecycle_attrs;
use crate::telemetry::now_unix_nanos;
use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use openai_frontend::OpenAiError;
use openai_frontend::OpenAiResult;
use serde_json::json;
use skippy_protocol::StageConfig;
use skippy_protocol::binary::StageReplyStats;
use skippy_protocol::binary::recv_ready;
use std::collections::BTreeMap;
use std::net::TcpStream;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Duration;

pub(in crate::frontend) struct PersistentStageLanePool {
    pub(in crate::frontend) config: StageConfig,
    pub(in crate::frontend) timeout_secs: u64,
    pub(in crate::frontend) telemetry: Telemetry,
    pub(in crate::frontend) lanes: Mutex<Vec<PersistentStageLane>>,
    pub(in crate::frontend) prefill_transport: Mutex<PrefillTransportEstimate>,
    pub(in crate::frontend) next_lane_id: AtomicU64,
    pub(in crate::frontend) capacity: usize,
}

pub(in crate::frontend) struct PersistentStageLane {
    pub(in crate::frontend) id: u64,
    pub(in crate::frontend) stream: TcpStream,
}

#[derive(Clone, Copy, Debug, Default)]
pub(in crate::frontend) struct PrefillTransportEstimate {
    pub(in crate::frontend) write_ms: f64,
    pub(in crate::frontend) wait_ms: f64,
    pub(in crate::frontend) slowest_compute_ms: f64,
    pub(in crate::frontend) slowest_compute_ms_per_token: f64,
    pub(in crate::frontend) write_to_compute: f64,
    pub(in crate::frontend) wait_to_compute: f64,
    pub(in crate::frontend) edge_stage_index: i64,
    pub(in crate::frontend) bottleneck_stage_index: i64,
    pub(in crate::frontend) bottleneck_token_count: i64,
    pub(in crate::frontend) activation_bytes: i64,
    pub(in crate::frontend) observations: u64,
}

/// Bounds the blocking ready handshake before the steady-state I/O deadline is
/// installed on the persistent lane.
pub(in crate::frontend) const LANE_READY_READ_TIMEOUT: Duration = Duration::from_secs(20);
pub(in crate::frontend) const LANE_STEADY_CONNECT_TIMEOUT: Duration = Duration::from_secs(3);

impl PersistentStageLanePool {
    const PREFILL_TRANSPORT_EWMA_ALPHA: f64 = 0.25;

    pub(in crate::frontend) fn new(
        config: &StageConfig,
        capacity: usize,
        timeout_secs: u64,
        telemetry: Telemetry,
    ) -> Result<Option<Arc<Self>>> {
        if config.downstream.is_none() {
            return Ok(None);
        }
        let pool = Arc::new(Self {
            config: config.clone(),
            timeout_secs,
            telemetry,
            lanes: Mutex::new(Vec::with_capacity(capacity)),
            prefill_transport: Mutex::new(PrefillTransportEstimate::default()),
            next_lane_id: AtomicU64::new(0),
            capacity,
        });
        let timer = PhaseTimer::start();
        for _ in 0..capacity {
            let lane = pool.connect_lane(
                Duration::from_secs(pool.timeout_secs),
                LANE_READY_READ_TIMEOUT,
            )?;
            pool.return_lane(lane);
        }
        let mut attrs = lifecycle_attrs(config);
        attrs.insert(
            "llama_stage.openai_downstream_pool_capacity".to_string(),
            json!(capacity),
        );
        attrs.insert(
            "llama_stage.elapsed_ms".to_string(),
            json!(timer.elapsed_ms()),
        );
        pool.telemetry.emit_span(
            "stage.openai_downstream_pool_ready",
            attrs,
            timer.start_unix_nanos,
            now_unix_nanos() as u64,
        );
        Ok(Some(pool))
    }

    pub(in crate::frontend) fn checkout(
        &self,
        ids: &OpenAiGenerationIds,
    ) -> OpenAiResult<PersistentStageLane> {
        let timer = PhaseTimer::start();
        let lane = {
            let mut lanes = self
                .lanes
                .lock()
                .map_err(|_| OpenAiError::backend("persistent lane pool lock poisoned"))?;
            lanes.pop()
        };
        let live_pooled = lane.filter(|lane| lane_stream_is_live(&lane.stream));
        let lane = match live_pooled {
            Some(lane) => lane,
            None => self
                .connect_lane(LANE_STEADY_CONNECT_TIMEOUT, LANE_STEADY_CONNECT_TIMEOUT)
                .map_err(openai_backend_error)?,
        };
        let mut attrs = BTreeMap::from([
            (
                "llama_stage.openai_downstream_persistent".to_string(),
                json!(true),
            ),
            (
                "llama_stage.openai_downstream_lane_id".to_string(),
                json!(lane.id),
            ),
            (
                "llama_stage.openai_downstream_pool_capacity".to_string(),
                json!(self.capacity),
            ),
            (
                "llama_stage.request_id".to_string(),
                json!(ids.request_id_string()),
            ),
            (
                "llama_stage.session_id".to_string(),
                json!(ids.session_id_string()),
            ),
        ]);
        attrs.insert(
            "llama_stage.elapsed_ms".to_string(),
            json!(timer.elapsed_ms()),
        );
        self.telemetry.emit_span(
            "stage.openai_downstream_connect",
            attrs,
            timer.start_unix_nanos,
            now_unix_nanos() as u64,
        );
        Ok(lane)
    }

    pub(in crate::frontend) fn prefill_transport_seed(&self) -> Option<PrefillChunkObservation> {
        let estimate = *self.prefill_transport.lock().ok()?;
        if estimate.observations == 0 {
            return None;
        }
        Some(PrefillChunkObservation {
            compute_ms: 1.0,
            forward_write_ms: estimate.write_to_compute,
            downstream_wait_ms: estimate.wait_to_compute,
        })
    }

    pub(in crate::frontend) fn prefill_transport_estimate(
        &self,
    ) -> Option<PrefillTransportEstimate> {
        let estimate = *self.prefill_transport.lock().ok()?;
        (estimate.observations > 0).then_some(estimate)
    }

    pub(in crate::frontend) fn observe_prefill_transport(
        &self,
        stats: &StageReplyStats,
        stage0_compute_max_ms: f64,
        stage0_compute_max_tokens: usize,
        stage0_forward_write_max_ms: f64,
        stage0_downstream_wait_max_ms: f64,
    ) {
        if stage0_compute_max_tokens == 0 {
            return;
        }
        let downstream_compute_ms = us_to_ms(stats.prefill_compute_us_at_slowest_rate);
        let downstream_compute_tokens = stats.prefill_compute_token_count_at_slowest_rate.max(1);
        let downstream_compute_ms_per_token =
            downstream_compute_ms / downstream_compute_tokens as f64;
        let stage0_compute_ms_per_token =
            stage0_compute_max_ms / stage0_compute_max_tokens.max(1) as f64;
        let (compute_ms, bottleneck_stage_index, bottleneck_token_count) =
            if downstream_compute_ms_per_token > stage0_compute_ms_per_token {
                (
                    downstream_compute_ms,
                    stats.prefill_compute_stage_index,
                    downstream_compute_tokens,
                )
            } else {
                (
                    stage0_compute_max_ms,
                    i64::from(self.config.stage_index),
                    i64::try_from(stage0_compute_max_tokens).unwrap_or(i64::MAX),
                )
            };
        let compute_ms = compute_ms.max(0.001);
        let compute_ms_per_token = compute_ms / bottleneck_token_count.max(1) as f64;
        let downstream_write_ms = us_to_ms(stats.prefill_edge_write_us_max);
        let downstream_wait_ms = us_to_ms(stats.prefill_edge_wait_us_max);
        let write_ms = stage0_forward_write_max_ms.max(downstream_write_ms);
        let wait_ms = stage0_downstream_wait_max_ms.max(downstream_wait_ms);
        let local_transport_ms = stage0_forward_write_max_ms + stage0_downstream_wait_max_ms;
        let downstream_transport_ms = us_to_ms(stats.prefill_edge_total_us_max);
        let edge_stage_index = if downstream_transport_ms > local_transport_ms {
            stats.prefill_edge_stage_index
        } else {
            i64::from(self.config.stage_index)
        };
        let sample = PrefillTransportEstimate {
            write_ms,
            wait_ms,
            slowest_compute_ms: compute_ms,
            slowest_compute_ms_per_token: compute_ms_per_token,
            write_to_compute: write_ms / compute_ms,
            wait_to_compute: wait_ms / compute_ms,
            edge_stage_index,
            bottleneck_stage_index,
            bottleneck_token_count,
            activation_bytes: stats.prefill_edge_activation_bytes_max,
            observations: u64::try_from(stats.prefill_edge_observation_count.max(1)).unwrap_or(1),
        };
        let mut estimate = match self.prefill_transport.lock() {
            Ok(estimate) => estimate,
            Err(_) => return,
        };
        if estimate.observations == 0 {
            *estimate = sample;
        } else {
            estimate.write_ms = ewma(estimate.write_ms, sample.write_ms);
            estimate.wait_ms = ewma(estimate.wait_ms, sample.wait_ms);
            estimate.slowest_compute_ms =
                ewma(estimate.slowest_compute_ms, sample.slowest_compute_ms);
            estimate.slowest_compute_ms_per_token = ewma(
                estimate.slowest_compute_ms_per_token,
                sample.slowest_compute_ms_per_token,
            );
            estimate.write_to_compute = ewma(estimate.write_to_compute, sample.write_to_compute);
            estimate.wait_to_compute = ewma(estimate.wait_to_compute, sample.wait_to_compute);
            estimate.edge_stage_index = sample.edge_stage_index;
            estimate.bottleneck_stage_index = sample.bottleneck_stage_index;
            estimate.bottleneck_token_count = sample.bottleneck_token_count;
            estimate.activation_bytes = sample.activation_bytes;
            estimate.observations = estimate.observations.saturating_add(sample.observations);
        }
    }

    pub(in crate::frontend) fn return_lane(&self, lane: PersistentStageLane) {
        match self.lanes.lock() {
            Ok(mut lanes) => lanes.push(lane),
            Err(_) => {
                let mut attrs = lifecycle_attrs(&self.config);
                attrs.insert(
                    "llama_stage.error".to_string(),
                    json!("persistent lane pool lock poisoned"),
                );
                self.telemetry
                    .emit("stage.openai_downstream_lane_return_failed", attrs);
            }
        }
    }

    pub(in crate::frontend) fn replace_lane(&self, retired_lane_id: u64) {
        let timer = PhaseTimer::start();
        let mut attrs = lifecycle_attrs(&self.config);
        attrs.insert(
            "llama_stage.openai_downstream_retired_lane_id".to_string(),
            json!(retired_lane_id),
        );
        match self.connect_lane(LANE_STEADY_CONNECT_TIMEOUT, LANE_STEADY_CONNECT_TIMEOUT) {
            Ok(lane) => {
                attrs.insert(
                    "llama_stage.openai_downstream_lane_id".to_string(),
                    json!(lane.id),
                );
                attrs.insert(
                    "llama_stage.elapsed_ms".to_string(),
                    json!(timer.elapsed_ms()),
                );
                self.return_lane(lane);
                self.telemetry.emit_span(
                    "stage.openai_downstream_lane_replaced",
                    attrs,
                    timer.start_unix_nanos,
                    now_unix_nanos() as u64,
                );
            }
            Err(error) => {
                attrs.insert("llama_stage.error".to_string(), json!(error.to_string()));
                attrs.insert(
                    "llama_stage.elapsed_ms".to_string(),
                    json!(timer.elapsed_ms()),
                );
                self.telemetry.emit_span(
                    "stage.openai_downstream_lane_replace_failed",
                    attrs,
                    timer.start_unix_nanos,
                    now_unix_nanos() as u64,
                );
            }
        }
    }

    pub(in crate::frontend) fn connect_lane(
        &self,
        connect_timeout: Duration,
        ready_timeout: Duration,
    ) -> Result<PersistentStageLane> {
        let lane_id = self.next_lane_id.fetch_add(1, Ordering::Relaxed);
        let timer = PhaseTimer::start();
        let stream = self
            .connect_lane_once(lane_id, connect_timeout, ready_timeout)
            .inspect_err(|error| {
            eprintln!(
                "openai downstream lane handshake failed: stage_id={} lane_id={lane_id}: {error:#}",
                self.config.stage_id,
            );
            })?;
        let mut attrs = lifecycle_attrs(&self.config);
        attrs.insert(
            "llama_stage.openai_downstream_lane_id".to_string(),
            json!(lane_id),
        );
        attrs.insert(
            "llama_stage.openai_downstream_pool_capacity".to_string(),
            json!(self.capacity),
        );
        attrs.insert(
            "llama_stage.elapsed_ms".to_string(),
            json!(timer.elapsed_ms()),
        );
        self.telemetry.emit_span(
            "stage.openai_downstream_persistent_connect",
            attrs,
            timer.start_unix_nanos,
            now_unix_nanos() as u64,
        );
        Ok(PersistentStageLane {
            id: lane_id,
            stream,
        })
    }

    fn connect_lane_once(
        &self,
        lane_id: u64,
        connect_timeout: Duration,
        ready_timeout: Duration,
    ) -> Result<TcpStream> {
        let mut stream = connect_binary_downstream(&self.config, connect_timeout)?
            .ok_or_else(|| anyhow!("embedded stage0 has no downstream"))?;
        let local_addr = stream.local_addr().ok();
        let peer_addr = stream.peer_addr().ok();
        eprintln!(
            "openai downstream lane waiting ready: stage_id={} lane_id={lane_id} local={local_addr:?} peer={peer_addr:?}",
            self.config.stage_id
        );
        send_client_ready_hello_if_enabled(&mut stream)
            .context("send persistent downstream lane client ready hello")?;
        receive_persistent_lane_ready(&mut stream, ready_timeout)?;
        configure_persistent_lane_io_deadlines(&stream)?;
        eprintln!(
            "openai downstream lane received ready: stage_id={} lane_id={lane_id} local={local_addr:?} peer={peer_addr:?}",
            self.config.stage_id
        );
        Ok(stream)
    }
}

pub(in crate::frontend) fn configure_persistent_lane_io_deadlines(
    stream: &TcpStream,
) -> Result<()> {
    let timeout = super::stage_reply_timeout();
    stream
        .set_read_timeout(Some(timeout))
        .context("set persistent downstream lane read timeout")?;
    stream
        .set_write_timeout(Some(timeout))
        .context("set persistent downstream lane write timeout")
}

pub(in crate::frontend) fn receive_persistent_lane_ready(
    stream: &mut TcpStream,
    timeout: Duration,
) -> Result<()> {
    stream
        .set_read_timeout(Some(timeout))
        .context("set persistent downstream lane ready timeout")?;
    let ready = recv_ready(&mut *stream).context("persistent downstream lane did not become ready");
    stream
        .set_read_timeout(None)
        .context("restore persistent downstream lane read timeout")?;
    ready
}

fn lane_stream_is_live(stream: &TcpStream) -> bool {
    use std::io::ErrorKind;
    if stream.set_nonblocking(true).is_err() {
        return false;
    }
    let mut probe = [0u8; 1];
    let live = match stream.peek(&mut probe) {
        Ok(0) | Ok(_) => false,
        Err(ref error) if error.kind() == ErrorKind::WouldBlock => true,
        Err(_) => false,
    };
    if stream.set_nonblocking(false).is_err() {
        return false;
    }
    live
}

pub(in crate::frontend) fn ewma(old: f64, sample: f64) -> f64 {
    old.mul_add(
        1.0 - PersistentStageLanePool::PREFILL_TRANSPORT_EWMA_ALPHA,
        sample * PersistentStageLanePool::PREFILL_TRANSPORT_EWMA_ALPHA,
    )
}
