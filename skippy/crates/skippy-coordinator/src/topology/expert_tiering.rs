//! Routed-expert tiering for mixture-of-experts stages.
//!
//! A MoE layer is mostly routed-expert weight, and a token touches only
//! `expert_used_count / expert_count` of it. When a stage's accelerator budget
//! cannot hold its whole layer range, the planner may keep the dense trunk,
//! attention and KV on the accelerator and move the routed experts of a
//! leading prefix of the stage's layers into host memory. Experts beyond the
//! host budget page from the node's model store on demand; that cost is
//! charged per token here so the stage-count decision can weigh "stream
//! experts on fewer nodes" against "add a pipeline hop".
//!
//! Placement never changes routing or precision. It changes where the weights
//! live and which kernels run them, which is why the cache identity also
//! carries it.

/// Whether the planner may spill routed experts off the accelerator.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ExpertOffloadPolicy {
    /// Every routed expert stays on the accelerator; a layer range that does
    /// not fit is rejected, which is the pre-tiering behaviour.
    #[default]
    Disabled,
    /// Spill the smallest leading prefix of a stage's routed experts that lets
    /// the stage fit, but only when the node set cannot fit the model with
    /// every expert resident.
    Auto,
}

/// Where a stage's routed experts live after planning.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StageExpertPlacement {
    /// Routed experts of the stage's blocks `[layer_start, host_layer_end)`
    /// load into host memory; later blocks keep theirs on the accelerator.
    pub host_layer_end: u32,
    /// Routed-expert bytes moved off the accelerator.
    pub host_expert_bytes: u64,
    /// Portion of `host_expert_bytes` the node's host budget holds resident.
    pub resident_host_bytes: u64,
    /// Portion of `host_expert_bytes` that pages from the model store.
    pub streamed_bytes: u64,
    /// Estimated per-token time spent paging streamed experts, in
    /// milliseconds. Zero when nothing streams.
    pub estimated_stream_ms_per_token: u32,
}

/// Per-layer expert accounting the planner derives once per plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ExpertLayerProfile {
    /// Routed-expert weight bytes per layer; zero for dense layers.
    pub(super) expert_bytes: Vec<u64>,
    /// Routed-expert bytes one token reads from a layer: the active share of
    /// the layer's expert bank.
    pub(super) bytes_touched_per_token: Vec<u64>,
}

impl ExpertLayerProfile {
    /// Build the profile when the input carries usable expert data.
    pub(super) fn from_input(
        layer_count: u32,
        layer_expert_weight_bytes: &[u64],
        expert_count: u32,
        expert_used_count: u32,
    ) -> Option<Self> {
        if layer_expert_weight_bytes.len() != layer_count as usize
            || layer_expert_weight_bytes.iter().all(|bytes| *bytes == 0)
        {
            return None;
        }
        let expert_count = u64::from(expert_count.max(1));
        let expert_used_count = u64::from(expert_used_count.clamp(1, expert_count as u32));
        let bytes_touched_per_token = layer_expert_weight_bytes
            .iter()
            .map(|bytes| {
                u64::try_from(
                    (u128::from(*bytes) * u128::from(expert_used_count))
                        .div_ceil(u128::from(expert_count)),
                )
                .unwrap_or(u64::MAX)
            })
            .collect();
        Some(Self {
            expert_bytes: layer_expert_weight_bytes.to_vec(),
            bytes_touched_per_token,
        })
    }
}

/// What one node can give to spilled experts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct HostExpertBudget {
    pub(super) host_bytes: u64,
    pub(super) stream_bytes_per_second: Option<u64>,
}

/// A stage range the tiering fit accepted for one node.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct TieredStageFit {
    pub(super) layer_span: u32,
    pub(super) accelerator_bytes: u64,
    pub(super) placement: Option<StageExpertPlacement>,
}

/// Find the layer span starting at `start` that fits `capacity` once the
/// smallest leading prefix of routed experts is spilled to the host.
///
/// `layer_required_bytes` is the accelerator cost with every expert resident.
/// Spilling layer `l` removes `profile.expert_bytes[l]` from that cost and
/// charges it to the host budget; whatever the host budget cannot hold must
/// stream, which is only allowed when the node has a measured stream rate.
///
/// A final stage (`must_cover`) has to take all `limit` layers, paging if it
/// must. Any other stage prefers its longest span that pages nothing, and
/// pages only when not even one layer fits without it, so the greedy packing
/// does not push paging onto a stage just to make it longer.
/// Returns `None` when no acceptable span fits.
pub(super) fn fit_tiered_span(
    layer_required_bytes: &[u64],
    profile: &ExpertLayerProfile,
    start: usize,
    limit: usize,
    capacity: u64,
    budget: HostExpertBudget,
    must_cover: bool,
) -> Option<TieredStageFit> {
    let max_span = limit.min(layer_required_bytes.len().saturating_sub(start));
    let fit = |span: usize, allow_stream: bool| {
        let budget = HostExpertBudget {
            stream_bytes_per_second: budget.stream_bytes_per_second.filter(|_| allow_stream),
            ..budget
        };
        fit_span_with_minimal_prefix(layer_required_bytes, profile, start, span, capacity, budget)
    };
    if must_cover {
        return fit(max_span, true);
    }
    (1..=max_span)
        .rev()
        .find_map(|span| fit(span, false))
        .or_else(|| (1..=max_span).rev().find_map(|span| fit(span, true)))
}

fn fit_span_with_minimal_prefix(
    layer_required_bytes: &[u64],
    profile: &ExpertLayerProfile,
    start: usize,
    span: usize,
    capacity: u64,
    budget: HostExpertBudget,
) -> Option<TieredStageFit> {
    let range = start..start + span;
    let resident_total = sum(&layer_required_bytes[range.clone()]);
    if resident_total <= capacity {
        return Some(TieredStageFit {
            layer_span: span as u32,
            accelerator_bytes: resident_total,
            placement: None,
        });
    }
    let mut accelerator_bytes = resident_total;
    let mut host_expert_bytes = 0u64;
    let mut touched_per_token = 0u64;
    for (prefix_len, layer) in range.enumerate() {
        accelerator_bytes = accelerator_bytes.saturating_sub(profile.expert_bytes[layer]);
        host_expert_bytes = host_expert_bytes.saturating_add(profile.expert_bytes[layer]);
        touched_per_token =
            touched_per_token.saturating_add(profile.bytes_touched_per_token[layer]);
        if accelerator_bytes > capacity {
            continue;
        }
        let resident_host_bytes = host_expert_bytes.min(budget.host_bytes);
        let streamed_bytes = host_expert_bytes - resident_host_bytes;
        let estimated_stream_ms_per_token = if streamed_bytes == 0 {
            0
        } else {
            let rate = budget.stream_bytes_per_second.filter(|rate| *rate > 0)?;
            stream_ms_per_token(touched_per_token, streamed_bytes, host_expert_bytes, rate)
        };
        return Some(TieredStageFit {
            layer_span: span as u32,
            accelerator_bytes,
            placement: Some(StageExpertPlacement {
                host_layer_end: (start + prefix_len + 1) as u32,
                host_expert_bytes,
                resident_host_bytes,
                streamed_bytes,
                estimated_stream_ms_per_token,
            }),
        });
    }
    None
}

/// Per-token paging time for the streamed share of the touched expert bytes,
/// assuming routing is uniform over the spilled bank.
fn stream_ms_per_token(
    touched_per_token: u64,
    streamed_bytes: u64,
    host_expert_bytes: u64,
    bytes_per_second: u64,
) -> u32 {
    let streamed_touched = (u128::from(touched_per_token) * u128::from(streamed_bytes))
        .div_ceil(u128::from(host_expert_bytes.max(1)));
    let millis = (streamed_touched * 1_000).div_ceil(u128::from(bytes_per_second));
    u32::try_from(millis).unwrap_or(u32::MAX)
}

fn sum(values: &[u64]) -> u64 {
    values
        .iter()
        .fold(0u64, |total, value| total.saturating_add(*value))
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    fn profile(layers: usize, expert_gib: u64) -> ExpertLayerProfile {
        ExpertLayerProfile::from_input(layers as u32, &vec![expert_gib * GIB; layers], 128, 8)
            .expect("expert profile")
    }

    #[test]
    fn profile_requires_layer_aligned_nonzero_expert_bytes() {
        assert_eq!(ExpertLayerProfile::from_input(4, &[1, 2, 3], 8, 2), None);
        assert_eq!(ExpertLayerProfile::from_input(2, &[0, 0], 8, 2), None);
        let profile = ExpertLayerProfile::from_input(2, &[800, 0], 8, 2).expect("profile");
        assert_eq!(profile.bytes_touched_per_token, vec![200, 0]);
    }

    #[test]
    fn resident_fit_reports_no_placement() {
        let required = vec![2 * GIB; 4];
        let fit = fit_tiered_span(
            &required,
            &profile(4, 1),
            0,
            4,
            8 * GIB,
            HostExpertBudget {
                host_bytes: 0,
                stream_bytes_per_second: None,
            },
            false,
        )
        .expect("fit");
        assert_eq!(fit.layer_span, 4);
        assert_eq!(fit.placement, None);
    }

    #[test]
    fn spills_the_smallest_prefix_that_fits_the_accelerator() {
        // Four layers of 2 GiB, half of each is routed experts; a 6 GiB budget
        // needs two layers' experts moved to host memory to hold all four.
        let required = vec![2 * GIB; 4];
        let fit = fit_tiered_span(
            &required,
            &profile(4, 1),
            0,
            4,
            6 * GIB,
            HostExpertBudget {
                host_bytes: 4 * GIB,
                stream_bytes_per_second: None,
            },
            false,
        )
        .expect("fit");
        assert_eq!(fit.layer_span, 4);
        assert_eq!(fit.accelerator_bytes, 6 * GIB);
        let placement = fit.placement.expect("placement");
        assert_eq!(placement.host_layer_end, 2);
        assert_eq!(placement.host_expert_bytes, 2 * GIB);
        assert_eq!(placement.resident_host_bytes, 2 * GIB);
        assert_eq!(placement.streamed_bytes, 0);
        assert_eq!(placement.estimated_stream_ms_per_token, 0);
    }

    #[test]
    fn streams_beyond_the_host_budget_only_with_a_measured_rate() {
        let required = vec![2 * GIB; 4];
        // 4 GiB accelerator budget: every layer's experts must leave; only
        // 1 GiB of the 4 GiB fits in host memory, the rest streams.
        let without_rate = fit_tiered_span(
            &required,
            &profile(4, 1),
            0,
            4,
            4 * GIB,
            HostExpertBudget {
                host_bytes: GIB,
                stream_bytes_per_second: None,
            },
            false,
        )
        .expect("fit");
        // Without a stream rate the span shrinks until host memory holds the
        // spilled experts: one layer resident on the host, one more fits.
        assert_eq!(without_rate.layer_span, 2);

        let with_rate = fit_tiered_span(
            &required,
            &profile(4, 1),
            0,
            4,
            4 * GIB,
            HostExpertBudget {
                host_bytes: GIB,
                stream_bytes_per_second: Some(4 * GIB),
            },
            true,
        )
        .expect("fit");
        assert_eq!(with_rate.layer_span, 4);
        let placement = with_rate.placement.expect("placement");
        assert_eq!(placement.host_layer_end, 4);
        assert_eq!(placement.resident_host_bytes, GIB);
        assert_eq!(placement.streamed_bytes, 3 * GIB);
        // A token touches 8/128 of each 1 GiB bank: 4 layers * 64 MiB = 256
        // MiB, three quarters of which streams at 4 GiB/s: just under 47 ms.
        assert_eq!(placement.estimated_stream_ms_per_token, 47);
    }

    #[test]
    fn an_intermediate_stage_prefers_a_shorter_span_to_paging() {
        let required = vec![2 * GIB; 4];
        let budget = HostExpertBudget {
            host_bytes: GIB,
            stream_bytes_per_second: Some(4 * GIB),
        };
        let intermediate =
            fit_tiered_span(&required, &profile(4, 1), 0, 4, 4 * GIB, budget, false).expect("fit");
        // Two layers fit fully resident; three or four would need paging.
        assert_eq!(intermediate.layer_span, 2);
        assert_eq!(intermediate.placement, None);
    }

    #[test]
    fn rejects_when_the_dense_trunk_alone_exceeds_capacity() {
        let required = vec![2 * GIB; 2];
        assert_eq!(
            fit_tiered_span(
                &required,
                &profile(2, 1),
                0,
                2,
                GIB / 2,
                HostExpertBudget {
                    host_bytes: 16 * GIB,
                    stream_bytes_per_second: Some(GIB),
                },
                true,
            ),
            None
        );
    }
}
