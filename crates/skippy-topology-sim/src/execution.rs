//! Discrete pipeline execution model over a planned topology.
//!
//! This layer answers "what tok/s will this plan actually deliver" —
//! complementing the placement layer (which decides the plan) with an
//! execution estimate per plan, calibrated against the measured anchors in
//! `docs/BENCHMARKS.md`.
//!
//! Two decode regimes:
//!
//! - **Serial** (single stream / `parallel_lanes == 1`): autoregressive
//!   decode depends on the previous token's logits, so every token
//!   traverses every stage and returns. TPOT = Σ stage service times +
//!   Σ edge times (including the prediction-return hop). This is why the
//!   measured anchors drop 68 → 21 → 12-13 tok/s across 1/2/3-way splits.
//! - **Pipelined** (`parallel_lanes > 1`): stages process consecutive
//!   tokens concurrently; throughput is bounded by the slowest stage plus
//!   its egress edge. TPOT per lane ≈ max stage+edge time.
//!
//! Per-stage service time is weight-streaming: the bytes a stage must read
//! from memory per token, divided by that node's sustained bandwidth. For
//! dense models that is the stage's parameter bytes; for MoE models only
//! the active expert bytes are touched per token (the calibration anchor
//! GLM-4.7-Flash streams ~7.2 GB/token of its ~17 GB).

use crate::{Scenario, ScenarioLink};

/// Modeled tok/s and per-token breakdown for one executed plan.
#[derive(Clone, Debug, PartialEq)]
pub struct ExecutionEstimate {
    /// Steady-state tokens per second for a single decoding stream.
    pub serial_tok_s: f64,
    /// Steady-state tokens per second per lane when lanes run concurrently
    /// (pipelined regime). `None` when the plan lacks the bandwidth signals
    /// to model pipelining.
    pub pipelined_tok_s_per_lane: Option<f64>,
    /// Total serial per-token time in microseconds (all stages + all hops).
    pub serial_token_us: u64,
    /// Per-stage service time in microseconds, in stage order.
    pub stage_service_us: Vec<u64>,
    /// Per-hop time in microseconds (activation transfer + RTT), in order,
    /// including the final prediction-return hop.
    pub hop_us: Vec<u64>,
}

impl Scenario {
    /// Bytes a node streams from memory per decoded token for a stage of
    /// `layer_count` layers. Dense models touch every parameter; MoE models
    /// touch only the active fraction, expressed as `active_weight_fraction`
    /// (0.0-1.0; 1.0 = dense).
    fn stage_streamed_bytes(&self, layer_count: u32) -> Option<u64> {
        let fraction = self.model.active_weight_fraction.unwrap_or(1.0);
        if !(0.0..=1.0).contains(&fraction) {
            return None;
        }
        Some((self.model.weight_bytes_per_layer as f64 * f64::from(layer_count) * fraction) as u64)
    }

    /// Directed link lookup between two nodes (exact direction, then the
    /// reverse as a symmetric fallback).
    fn link(&self, source: &str, target: &str) -> Option<&ScenarioLink> {
        self.links
            .get(&format!("{source} -> {target}"))
            .or_else(|| self.links.get(&format!("{target} -> {source}")))
    }

    /// Estimate execution of a planned topology (stages as
    /// `(node_id, layer_count)` in pipeline order).
    pub fn estimate_execution(&self, stages: &[(&str, u32)]) -> Result<ExecutionEstimate, String> {
        if stages.is_empty() {
            return Err("no stages".to_string());
        }
        let mut stage_service_us = Vec::with_capacity(stages.len());
        let stage_overhead_us = (self.model.per_stage_overhead_ms * 1_000.0) as u64;
        for (node_id, layers) in stages {
            let node = self
                .nodes
                .get(*node_id)
                .ok_or_else(|| format!("unknown node {node_id}"))?;
            let bandwidth = node
                .sustained_mem_bandwidth_mib_per_s
                .ok_or_else(|| format!("node {node_id} lacks bandwidth signal"))?;
            if bandwidth == 0 {
                return Err(format!("node {node_id} reports zero bandwidth"));
            }
            let streamed = self
                .stage_streamed_bytes(*layers)
                .ok_or("invalid active_weight_fraction")?;
            // bytes / (MiB/s) in microseconds, plus calibrated per-stage
            // software overhead.
            stage_service_us.push(
                (streamed as f64 * 1_000_000.0 / (f64::from(bandwidth) * 1_048_576.0)) as u64
                    + stage_overhead_us,
            );
        }

        let hop_overhead_us = (self.model.per_hop_overhead_ms * 1_000.0) as u64;
        let mut hop_us = Vec::new();
        for window in stages.windows(2) {
            let link = self
                .link(window[0].0, window[1].0)
                .ok_or_else(|| format!("missing link {} -> {}", window[0].0, window[1].0))?;
            let mut us = u64::from(link.rtt_ms) * 1_000 + hop_overhead_us;
            if let Some(bandwidth) = link.large_frame_mib_per_s.filter(|b| *b > 0) {
                let frame = self.model.activation_frame_bytes;
                if frame > 0 {
                    us +=
                        (frame as f64 * 1_000_000.0 / (f64::from(bandwidth) * 1_048_576.0)) as u64;
                }
            }
            hop_us.push(us);
        }
        // Prediction-return hop: final stage back to stage 0.
        if stages.len() > 1 {
            let (last, first) = (stages[stages.len() - 1].0, stages[0].0);
            let link = self
                .link(last, first)
                .ok_or_else(|| format!("missing return link {last} -> {first}"))?;
            let mut us = u64::from(link.rtt_ms) * 1_000 + hop_overhead_us;
            if let Some(bandwidth) = link.large_frame_mib_per_s.filter(|b| *b > 0) {
                let frame = self.model.activation_frame_bytes;
                if frame > 0 {
                    us +=
                        (frame as f64 * 1_000_000.0 / (f64::from(bandwidth) * 1_048_576.0)) as u64;
                }
            }
            hop_us.push(us);
        }

        let serial_us: u64 = stage_service_us.iter().sum::<u64>() + hop_us.iter().sum::<u64>();
        let serial_tok_s = 1_000_000.0 / serial_us as f64;
        // Pipelined regime: throughput bounded by the slowest stage+egress
        // pair. With no hops (single stage) it is just the stage time.
        let pipelined_us = stage_service_us
            .iter()
            .enumerate()
            .map(|(index, stage)| stage + hop_us.get(index).copied().unwrap_or(0))
            .max()
            .ok_or("no stages")?;
        let pipelined_tok_s_per_lane = 1_000_000.0 / pipelined_us as f64;
        Ok(ExecutionEstimate {
            serial_tok_s,
            pipelined_tok_s_per_lane: Some(pipelined_tok_s_per_lane),
            serial_token_us: serial_us,
            stage_service_us,
            hop_us,
        })
    }
}
