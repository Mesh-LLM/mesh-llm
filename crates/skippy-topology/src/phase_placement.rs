//! Cost-based prefill/decode phase placement.
//!
//! Prefill is compute-bound and decode is memory-bandwidth-bound, so on a
//! heterogeneous pair the compute-strong node should prefill and the
//! bandwidth-strong node should decode — but only when the handoff pays:
//! moving continuation state costs a fixed floor (recurrent/SSM snapshot for
//! hybrid families) plus per-token attention KV, so short prompts and poor
//! links must prefill in place. These are pure functions over capability
//! signals nodes already gossip (`compute_tflops_fp16`,
//! `mem_bandwidth_gbps` in `PeerAnnouncement`); the planner and the router
//! call them, they do not read the mesh themselves.

/// Capability signals for one candidate node, as gossiped.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhaseCandidate {
    pub compute_tflops_fp16: f64,
    pub mem_bandwidth_gbps: f64,
}

/// A phase role assignment for a pair of nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhaseAssignment {
    /// Index of the node that should prefill.
    pub prefill: usize,
    /// Index of the node that should decode.
    pub decode: usize,
}

/// Assign prefill/decode roles across a candidate pair.
///
/// The binding constraint decides: place the scarce resource where it is the
/// bottleneck. When the pair differs meaningfully in compute but not in
/// bandwidth (the M3 Ultra / M1 Ultra shape: ~2x compute apart, ~2%
/// bandwidth apart), the compute-strong node prefills and decode loses
/// almost nothing. When the pair differs in both, the ratio test below
/// picks the split that maximises the product of (prefill compute) and
/// (decode bandwidth) — equivalent to comparing the two assignments'
/// bottleneck utilisation.
pub fn assign_phase_roles(a: PhaseCandidate, b: PhaseCandidate) -> PhaseAssignment {
    // Compare: a prefills (a.compute * b.bandwidth) vs b prefills
    // (b.compute * a.bandwidth). Guard degenerate zero signals by treating
    // them as equal, which falls through to a-prefills for determinism.
    let a_prefills = a.compute_tflops_fp16.max(0.0) * b.mem_bandwidth_gbps.max(0.0);
    let b_prefills = b.compute_tflops_fp16.max(0.0) * a.mem_bandwidth_gbps.max(0.0);
    if b_prefills > a_prefills {
        PhaseAssignment {
            prefill: 1,
            decode: 0,
        }
    } else {
        PhaseAssignment {
            prefill: 0,
            decode: 1,
        }
    }
}

/// Inputs to the per-request disaggregation cost gate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HandoffCostModel {
    /// Attention-KV bytes per prompt token (measured, e.g. from a
    /// remote-handoff report's `state_bytes_per_prompt_token` minus the
    /// fixed floor).
    pub state_bytes_per_token: f64,
    /// Fixed handoff bytes independent of prompt length — the
    /// recurrent/SSM + conv snapshot for hybrid families, 0 for pure
    /// attention.
    pub fixed_state_bytes: f64,
    /// Usable link throughput between the pair, bytes/second (measured,
    /// not line rate).
    pub link_bytes_per_second: f64,
    /// Prefill throughput on the decode node, tokens/second.
    pub local_prefill_tokens_per_second: f64,
    /// Prefill throughput on the prefill node, tokens/second.
    pub remote_prefill_tokens_per_second: f64,
    /// Fraction of transfer hidden behind prefill compute by page
    /// streaming, in [0, 1]. 0 models a flat (non-overlapped) transfer;
    /// measured runs on the lab pair should calibrate this.
    pub transfer_overlap_fraction: f64,
}

impl HandoffCostModel {
    /// TTFT cost (seconds) of disaggregating a prompt of `tokens`.
    pub fn disaggregated_seconds(&self, tokens: u64) -> f64 {
        let tokens = tokens as f64;
        let remote_rate = self.remote_prefill_tokens_per_second.max(f64::EPSILON);
        let link = self.link_bytes_per_second.max(f64::EPSILON);
        let prefill = tokens / remote_rate;
        let transfer_bytes = tokens * self.state_bytes_per_token.max(0.0);
        let hidden = self.transfer_overlap_fraction.clamp(0.0, 1.0);
        // Per-token KV can hide behind prefill; the fixed snapshot is only
        // final after the last chunk and is always exposed.
        let exposed_transfer =
            (transfer_bytes * (1.0 - hidden) + self.fixed_state_bytes.max(0.0)) / link;
        prefill + exposed_transfer
    }

    /// TTFT cost (seconds) of prefilling in place on the decode node.
    pub fn local_seconds(&self, tokens: u64) -> f64 {
        tokens as f64 / self.local_prefill_tokens_per_second.max(f64::EPSILON)
    }

    /// Whether a prompt of `tokens` should hand off.
    pub fn should_disaggregate(&self, tokens: u64) -> bool {
        self.disaggregated_seconds(tokens) < self.local_seconds(tokens)
    }

    /// Smallest prompt length at which handoff wins, or `None` if it never
    /// wins (searched up to `max_tokens`). The router caches this per pair
    /// and compares incoming prompt lengths against it.
    pub fn break_even_tokens(&self, max_tokens: u64) -> Option<u64> {
        // Both cost curves are affine in `tokens`, so the crossover is
        // where the per-token slopes and the fixed offsets balance:
        //   tokens/local = tokens/remote + tokens*exposed_per_token/link + fixed/link
        // Solve directly, then round outward and verify against the model
        // to stay robust to degenerate inputs.
        let local_rate = self.local_prefill_tokens_per_second.max(f64::EPSILON);
        let remote_rate = self.remote_prefill_tokens_per_second.max(f64::EPSILON);
        let link = self.link_bytes_per_second.max(f64::EPSILON);
        let hidden = self.transfer_overlap_fraction.clamp(0.0, 1.0);
        let slope_local = 1.0 / local_rate;
        let slope_disaggregated =
            1.0 / remote_rate + self.state_bytes_per_token.max(0.0) * (1.0 - hidden) / link;
        if slope_disaggregated >= slope_local {
            // Handoff never catches up: the per-token cost alone is worse.
            return None;
        }
        let fixed = self.fixed_state_bytes.max(0.0) / link;
        let crossover = fixed / (slope_local - slope_disaggregated);
        let candidate = crossover.ceil().max(1.0) as u64;
        (candidate <= max_tokens && self.should_disaggregate(candidate)).then_some(candidate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The lab shape: M3 Ultra ≈ 2x the compute of the M1 Ultra with
    /// near-equal bandwidth. The compute-strong box must prefill.
    #[test]
    fn near_equal_bandwidth_pairs_split_on_compute() {
        let m3_ultra = PhaseCandidate {
            compute_tflops_fp16: 57.0,
            mem_bandwidth_gbps: 819.0,
        };
        let m1_ultra = PhaseCandidate {
            compute_tflops_fp16: 42.0,
            mem_bandwidth_gbps: 800.0,
        };
        assert_eq!(
            assign_phase_roles(m3_ultra, m1_ultra),
            PhaseAssignment {
                prefill: 0,
                decode: 1
            }
        );
        assert_eq!(
            assign_phase_roles(m1_ultra, m3_ultra),
            PhaseAssignment {
                prefill: 1,
                decode: 0
            }
        );
    }

    /// The DGX-Spark shape: huge compute, modest bandwidth, paired with a
    /// bandwidth-rich Mac. Compute prefills, bandwidth decodes.
    #[test]
    fn compute_heavy_bandwidth_light_node_prefills() {
        let spark = PhaseCandidate {
            compute_tflops_fp16: 250.0,
            mem_bandwidth_gbps: 273.0,
        };
        let mac = PhaseCandidate {
            compute_tflops_fp16: 57.0,
            mem_bandwidth_gbps: 819.0,
        };
        assert_eq!(
            assign_phase_roles(spark, mac),
            PhaseAssignment {
                prefill: 0,
                decode: 1
            }
        );
    }

    fn lab_model() -> HandoffCostModel {
        // Calibrated to the remote-handoff report shape: ~115 KiB/token
        // attention KV, no recurrent floor (dense), 1 GB/s usable link,
        // decode node prefills at 2k tok/s, prefill node at 4k tok/s.
        HandoffCostModel {
            state_bytes_per_token: 115.0 * 1024.0,
            fixed_state_bytes: 0.0,
            link_bytes_per_second: 1e9,
            local_prefill_tokens_per_second: 2000.0,
            remote_prefill_tokens_per_second: 4000.0,
            transfer_overlap_fraction: 0.9,
        }
    }

    #[test]
    fn fast_link_with_overlap_favors_handoff_beyond_break_even() {
        let model = lab_model();
        let break_even = model.break_even_tokens(65_536);
        // No fixed floor and a winning slope: handoff wins from the start.
        assert_eq!(break_even, Some(1));
        assert!(model.should_disaggregate(4096));
    }

    /// A hybrid family's fixed recurrent floor (~160 MiB) pushes the break
    /// even point out: short prompts must prefill in place.
    #[test]
    fn fixed_recurrent_floor_rejects_short_prompts() {
        let model = HandoffCostModel {
            fixed_state_bytes: 160.0 * 1024.0 * 1024.0,
            ..lab_model()
        };
        assert!(!model.should_disaggregate(256));
        let break_even = model
            .break_even_tokens(65_536)
            .expect("must eventually win");
        assert!(break_even > 256, "break even was {break_even}");
        assert!(model.should_disaggregate(break_even));
        assert!(!model.should_disaggregate(break_even.saturating_sub(64)));
    }

    /// A slow link with no overlap can make handoff lose at every length.
    #[test]
    fn slow_flat_link_never_disaggregates() {
        let model = HandoffCostModel {
            link_bytes_per_second: 50e6,
            transfer_overlap_fraction: 0.0,
            ..lab_model()
        };
        assert_eq!(model.break_even_tokens(1_000_000), None);
        assert!(!model.should_disaggregate(32_768));
    }

    /// No compute advantage and a real transfer cost: prefill in place.
    #[test]
    fn equal_nodes_prefill_in_place() {
        let model = HandoffCostModel {
            remote_prefill_tokens_per_second: 2000.0,
            ..lab_model()
        };
        assert_eq!(model.break_even_tokens(1_000_000), None);
    }
}
