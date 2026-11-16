use crate::inference::election;
use crate::mesh;
use crate::network::router;

pub(super) const REQUEST_TOKEN_MARGIN: u32 = 256;

/// Decode headroom Skippy admission reserves, as a `u32` for token arithmetic.
fn decode_headroom_tokens() -> u32 {
    saturating_u32(skippy_serving::DECODE_BATCH_HEADROOM_TOKENS)
}

fn saturating_u32(value: usize) -> u32 {
    value.try_into().unwrap_or(u32::MAX)
}

pub(super) fn ceil_div_u32(value: u32, divisor: u32) -> u32 {
    value.saturating_add(divisor - 1) / divisor
}

#[cfg(test)]
fn request_budget_tokens(body: &serde_json::Value) -> Option<u32> {
    let serialized = serde_json::to_vec(body).ok()?;
    let completion_tokens = [
        "max_completion_tokens",
        "max_tokens",
        "max_output_tokens",
        "n_predict",
    ]
    .into_iter()
    .find_map(|key| body.get(key).and_then(|value| value.as_u64()))
    .map(|value| value.min(u32::MAX as u64) as u32);
    request_budget_tokens_from_parts(serialized.len(), completion_tokens)
}

/// Context a request needs from a target, in tokens.
///
/// The completion term is deliberately *not* the client's `max_tokens`. That
/// value is an upper bound, and clients commonly send an enormous one to mean
/// "no limit" (Buzz Agent Desktop defaults to 65,536). Skippy admission only
/// ever reserves `min(max_tokens, DECODE_BATCH_HEADROOM_TOKENS)`, so counting
/// the full ceiling here made routing refuse requests the server would have
/// served — issue #1350. Both layers now apply the same bound.
pub(crate) fn request_budget_tokens_from_parts(
    body_len_bytes: usize,
    completion_tokens: Option<u32>,
) -> Option<u32> {
    if body_len_bytes == 0 {
        return None;
    }
    let prompt_tokens = ceil_div_u32(saturating_u32(body_len_bytes), 4);
    let completion_tokens = completion_tokens.unwrap_or(0).min(decode_headroom_tokens());
    let requested_tokens = prompt_tokens.saturating_add(completion_tokens);
    Some(
        prompt_tokens
            .saturating_add(completion_tokens)
            .saturating_add(request_token_margin(requested_tokens)),
    )
}

pub(super) fn request_token_margin(requested_tokens: u32) -> u32 {
    const MIN_REQUEST_TOKEN_MARGIN: u32 = 16;
    if requested_tokens == 0 {
        return 0;
    }
    ceil_div_u32(requested_tokens, 4).clamp(MIN_REQUEST_TOKEN_MARGIN, REQUEST_TOKEN_MARGIN)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct TargetThroughputRank {
    pub(super) avg_tokens_per_second_milli: u64,
    pub(super) throughput_samples: u64,
    pub(super) local_observation: bool,
}

#[derive(Clone)]
struct RankedTarget<T> {
    index: usize,
    candidate: T,
    context_length: Option<u32>,
    throughput: Option<TargetThroughputRank>,
}

const LOCAL_THROUGHPUT_PRECEDENCE_SAMPLES: u64 = 3;
const TARGET_THROUGHPUT_MAX_SCORE_SAMPLES: u64 = 32;

/// Sort key produced by [`target_throughput_rank_key`]. Candidates with equal
/// keys are throughput-equivalent: the measurements give no reason to prefer
/// one over the other.
pub(super) type ThroughputRankKey = (bool, bool, u64, u64);

fn target_throughput_rank_key(throughput: Option<TargetThroughputRank>) -> ThroughputRankKey {
    let Some(throughput) = throughput else {
        return (false, false, 0, 0);
    };
    if throughput.avg_tokens_per_second_milli == 0 || throughput.throughput_samples == 0 {
        return (false, false, 0, 0);
    }
    let sample_weight = throughput
        .throughput_samples
        .min(TARGET_THROUGHPUT_MAX_SCORE_SAMPLES);
    (
        true,
        throughput.local_observation,
        throughput.avg_tokens_per_second_milli,
        sample_weight,
    )
}

fn sort_ranked_targets<T>(targets: &mut [RankedTarget<T>]) {
    targets.sort_by(|a, b| {
        target_throughput_rank_key(b.throughput)
            .cmp(&target_throughput_rank_key(a.throughput))
            .then_with(|| a.index.cmp(&b.index))
    });
}

/// Candidates ordered by context fit and measured throughput, plus the length
/// of the leading run of targets whose throughput rank ties with the best one.
///
/// `equivalent_prefix` bounds in-flight reservation spreading: only the
/// leading targets that the measurements cannot distinguish are equivalent
/// enough to trade off by local in-flight count. It never crosses the
/// adequate-context / unknown-context boundary.
pub(super) struct RankedCandidates<T> {
    pub(super) ordered: Vec<T>,
    pub(super) equivalent_prefix: usize,
    /// Per-target draw inputs for the two-choice host pick, including each
    /// target's dense rank class in `ordered`.
    pub(super) placements: std::collections::HashMap<T, PlacementWeight>,
}

use crate::network::router::PlacementWeight;

#[cfg(test)]
impl<T: Clone + std::hash::Hash + Eq> RankedCandidates<T> {
    /// Fixture whose leading `equivalent_prefix` targets share rank class 0
    /// and every later target is strictly worse than the one before it, all
    /// at neutral capacity.
    pub(crate) fn with_equivalent_prefix(ordered: Vec<T>, equivalent_prefix: usize) -> Self {
        let placements = ordered
            .iter()
            .enumerate()
            .map(|(index, target)| {
                (
                    target.clone(),
                    PlacementWeight {
                        rank: (index + 1).saturating_sub(equivalent_prefix),
                        ..PlacementWeight::unmeasured()
                    },
                )
            })
            .collect();
        Self {
            ordered,
            equivalent_prefix,
            placements,
        }
    }
}

fn placement_weight_for(throughput: Option<TargetThroughputRank>, rank: usize) -> PlacementWeight {
    PlacementWeight {
        capacity: crate::network::router::placement_weight(
            throughput.map(|rank| rank.avg_tokens_per_second_milli as f64 / 1000.0),
            throughput.map(|rank| rank.throughput_samples).unwrap_or(0),
        ),
        rtt_ms: None,
        rank,
    }
}

/// Dense rank classes over consecutive tiers of sorted targets: targets with
/// equal throughput rank keys inside one tier share a class; every tier
/// boundary and every key change starts the next class.
fn placements_for<T: Clone + std::hash::Hash + Eq>(
    tiers: &[&[RankedTarget<T>]],
) -> std::collections::HashMap<T, PlacementWeight> {
    let mut placements = std::collections::HashMap::new();
    let mut rank = 0usize;
    for tier in tiers {
        let mut previous_key: Option<ThroughputRankKey> = None;
        for target in tier.iter() {
            let key = target_throughput_rank_key(target.throughput);
            if previous_key.is_some_and(|previous| previous != key) {
                rank += 1;
            }
            previous_key = Some(key);
            placements.insert(
                target.candidate.clone(),
                placement_weight_for(target.throughput, rank),
            );
        }
        if previous_key.is_some() {
            rank += 1;
        }
    }
    placements
}

fn equivalent_prefix_len<T>(targets: &[RankedTarget<T>]) -> usize {
    let Some(first) = targets.first() else {
        return 0;
    };
    let key = target_throughput_rank_key(first.throughput);
    targets
        .iter()
        .take_while(|target| target_throughput_rank_key(target.throughput) == key)
        .count()
}

pub(super) fn rank_candidates_by_context_and_throughput<T: Clone + std::hash::Hash + Eq>(
    candidates: &[(T, Option<u32>, Option<TargetThroughputRank>)],
    required_tokens: Option<u32>,
) -> RankedCandidates<T> {
    let ranked = candidates
        .iter()
        .enumerate()
        .map(
            |(index, (candidate, context_length, throughput))| RankedTarget {
                index,
                candidate: candidate.clone(),
                context_length: *context_length,
                throughput: *throughput,
            },
        )
        .collect::<Vec<_>>();

    let Some(required_tokens) = required_tokens else {
        let mut ranked = ranked;
        sort_ranked_targets(&mut ranked);
        let equivalent_prefix = equivalent_prefix_len(&ranked);
        let placements = placements_for(&[ranked.as_slice()]);
        return RankedCandidates {
            ordered: ranked.into_iter().map(|ranked| ranked.candidate).collect(),
            equivalent_prefix,
            placements,
        };
    };

    let mut adequate = Vec::new();
    let mut unknown = Vec::new();
    for ranked in ranked {
        match ranked.context_length {
            Some(value) if value >= required_tokens => adequate.push(ranked),
            Some(_) => {}
            None => unknown.push(ranked),
        }
    }

    if adequate.is_empty() && unknown.is_empty() {
        return RankedCandidates {
            ordered: Vec::new(),
            equivalent_prefix: 0,
            placements: Default::default(),
        };
    }

    sort_ranked_targets(&mut adequate);
    sort_ranked_targets(&mut unknown);
    let equivalent_prefix = if adequate.is_empty() {
        equivalent_prefix_len(&unknown)
    } else {
        equivalent_prefix_len(&adequate)
    };
    let placements = placements_for(&[adequate.as_slice(), unknown.as_slice()]);
    RankedCandidates {
        ordered: adequate
            .into_iter()
            .chain(unknown)
            .map(|ranked| ranked.candidate)
            .collect(),
        equivalent_prefix,
        placements,
    }
}

#[cfg(test)]
fn reorder_candidates_by_context_and_throughput<T: Clone + std::hash::Hash + Eq>(
    candidates: &[(T, Option<u32>, Option<TargetThroughputRank>)],
    required_tokens: Option<u32>,
) -> Vec<T> {
    rank_candidates_by_context_and_throughput(candidates, required_tokens).ordered
}

fn local_target_throughput_rank(
    node: &mesh::Node,
    model: &str,
    target: &election::InferenceTarget,
) -> Option<TargetThroughputRank> {
    let attempt_target = match target {
        election::InferenceTarget::Local(port) => {
            crate::network::metrics::AttemptTarget::Local(format!("127.0.0.1:{port}"))
        }
        election::InferenceTarget::Remote(peer_id) => {
            crate::network::metrics::AttemptTarget::Remote(peer_id.fmt_short().to_string())
        }
        election::InferenceTarget::None => return None,
    };
    node.routing_metrics()
        .throughput_hint_for_target(model, attempt_target)
        .map(|hint| TargetThroughputRank {
            avg_tokens_per_second_milli: hint.avg_tokens_per_second_milli,
            throughput_samples: hint.throughput_samples,
            local_observation: true,
        })
}

/// This host's own serving throughput for `model`: the same local-target-only
/// evidence it advertises to peers (`advertisable_model_throughput`), never the
/// model-wide aggregate that also counts what this node observed while
/// routing to other hosts.
pub(super) fn local_serving_throughput_rank(
    node: &mesh::Node,
    model: &str,
) -> Option<TargetThroughputRank> {
    node.routing_metrics()
        .advertisable_model_throughput(std::slice::from_ref(&model.to_string()))
        .into_iter()
        .find(|hint| hint.model_name == model)
        .map(|hint| TargetThroughputRank {
            avg_tokens_per_second_milli: hint.avg_tokens_per_second_milli,
            throughput_samples: hint.throughput_samples,
            local_observation: true,
        })
}

pub(super) async fn remote_target_throughput_rank(
    node: &mesh::Node,
    model: &str,
    peer_id: iroh::EndpointId,
) -> Option<TargetThroughputRank> {
    let target = election::InferenceTarget::Remote(peer_id);
    let local = local_target_throughput_rank(node, model, &target);
    if local
        .map(|hint| hint.throughput_samples >= LOCAL_THROUGHPUT_PRECEDENCE_SAMPLES)
        .unwrap_or(false)
    {
        return local;
    }

    let gossiped = node
        .peer_model_throughput_hint(peer_id, model)
        .await
        .map(|hint| TargetThroughputRank {
            avg_tokens_per_second_milli: hint.avg_tokens_per_second_milli,
            throughput_samples: hint.throughput_samples,
            local_observation: false,
        });
    gossiped.or(local)
}

pub(super) async fn rank_remote_hosts_by_context(
    node: &mesh::Node,
    model: &str,
    required_tokens: Option<u32>,
    hosts: &[iroh::EndpointId],
) -> RankedCandidates<iroh::EndpointId> {
    let mut candidates = Vec::with_capacity(hosts.len());
    for host in hosts {
        candidates.push((
            *host,
            node.peer_model_context_length(*host, model).await,
            remote_target_throughput_rank(node, model, *host).await,
        ));
    }
    let mut ranked = rank_candidates_by_context_and_throughput(&candidates, required_tokens);
    for host in hosts {
        if let Some(placement) = ranked.placements.get_mut(host) {
            placement.rtt_ms = node.peer_rtt_ms(host).await;
        }
    }
    ranked
}

#[cfg(test)]
pub(super) async fn order_remote_hosts_by_context(
    node: &mesh::Node,
    model: &str,
    required_tokens: Option<u32>,
    hosts: &[iroh::EndpointId],
) -> Vec<iroh::EndpointId> {
    rank_remote_hosts_by_context(node, model, required_tokens, hosts)
        .await
        .ordered
}

pub(super) async fn rank_targets_by_context(
    node: &mesh::Node,
    model: &str,
    required_tokens: Option<u32>,
    targets: &[election::InferenceTarget],
) -> RankedCandidates<election::InferenceTarget> {
    let mut candidates = Vec::with_capacity(targets.len());
    for target in targets {
        let context_length = match target {
            election::InferenceTarget::Local(_) => node.local_model_context_length(model).await,
            election::InferenceTarget::Remote(peer_id) => {
                node.peer_model_context_length(*peer_id, model).await
            }
            election::InferenceTarget::None => None,
        };
        let throughput = match target {
            election::InferenceTarget::Remote(peer_id) => {
                remote_target_throughput_rank(node, model, *peer_id).await
            }
            _ => local_target_throughput_rank(node, model, target),
        };
        candidates.push((target.clone(), context_length, throughput));
    }
    let mut ranked = rank_candidates_by_context_and_throughput(&candidates, required_tokens);
    for target in targets {
        let rtt_ms = match target {
            election::InferenceTarget::Local(_) => Some(0),
            election::InferenceTarget::Remote(peer_id) => node.peer_rtt_ms(peer_id).await,
            election::InferenceTarget::None => None,
        };
        if let Some(placement) = ranked.placements.get_mut(target) {
            placement.rtt_ms = rtt_ms;
        }
    }
    ranked
}

#[cfg(test)]
pub(super) async fn order_targets_by_context(
    node: &mesh::Node,
    model: &str,
    required_tokens: Option<u32>,
    targets: &[election::InferenceTarget],
) -> Vec<election::InferenceTarget> {
    rank_targets_by_context(node, model, required_tokens, targets)
        .await
        .ordered
}

pub(super) fn move_target_first<T: PartialEq>(targets: &mut [T], target: &T) -> bool {
    if let Some(pos) = targets.iter().position(|candidate| candidate == target) {
        targets[..=pos].rotate_right(1);
        true
    } else {
        false
    }
}

pub(super) fn descriptor_for_model<'a>(
    descriptors: &'a [mesh::ServedModelDescriptor],
    model_name: &str,
) -> Option<&'a mesh::ServedModelDescriptor> {
    descriptors
        .iter()
        .find(|descriptor| descriptor.identity.model_name == model_name)
        .or_else(|| {
            descriptors.iter().find(|descriptor| {
                super::model_names::public_model_id(
                    &descriptor.identity.model_name,
                    Some(descriptor),
                    "",
                ) == model_name
            })
        })
}

pub(super) fn cached_auto_model_satisfies_media_requirements(
    model: &str,
    media: &router::MediaRequirements,
    descriptors: &[mesh::ServedModelDescriptor],
) -> bool {
    let caps = capabilities_for_model(model, descriptors);
    router::model_satisfies_media_requirements(&caps, media)
}

pub(crate) fn capabilities_for_model(
    model: &str,
    descriptors: &[mesh::ServedModelDescriptor],
) -> crate::models::ModelCapabilities {
    descriptor_for_model(descriptors, model)
        .filter(|descriptor| descriptor.capabilities_known)
        .map(|descriptor| descriptor.capabilities)
        .unwrap_or_else(|| crate::models::installed_model_capabilities(model))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn local_gguf_descriptor(model_name: &str) -> mesh::ServedModelDescriptor {
        mesh::ServedModelDescriptor {
            identity: mesh::ServedModelIdentity {
                model_name: model_name.to_string(),
                source_kind: mesh::ModelSourceKind::LocalGguf,
                local_file_name: Some(format!("{model_name}.gguf")),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn local_gguf_descriptor_with_capabilities(
        model_name: &str,
        capabilities: crate::models::ModelCapabilities,
    ) -> mesh::ServedModelDescriptor {
        mesh::ServedModelDescriptor {
            capabilities_known: true,
            capabilities,
            ..local_gguf_descriptor(model_name)
        }
    }

    #[test]
    fn test_cached_auto_model_rejects_text_model_for_image_request() {
        let body = serde_json::json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
                ]
            }]
        });
        let media = router::media_requirements(&body);

        assert!(!cached_auto_model_satisfies_media_requirements(
            "Qwen3-8B-Q4_K_M",
            &media,
            &[]
        ));
        assert!(cached_auto_model_satisfies_media_requirements(
            "Qwen3.5-0.8B-Vision-Q4_K_M",
            &media,
            &[]
        ));
    }

    #[test]
    fn cached_auto_model_rejects_descriptor_text_only_even_when_name_looks_vision() {
        let body = serde_json::json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
                ]
            }]
        });
        let media = router::media_requirements(&body);
        let model = "Qwen3VL-2B-Instruct-Q4_K_M";
        let descriptors = vec![local_gguf_descriptor_with_capabilities(
            model,
            crate::models::ModelCapabilities::default(),
        )];

        assert!(!cached_auto_model_satisfies_media_requirements(
            model,
            &media,
            &descriptors
        ));
    }

    #[test]
    fn cached_auto_model_uses_static_fallback_for_unknown_descriptor_capabilities() {
        let body = serde_json::json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
                ]
            }]
        });
        let media = router::media_requirements(&body);
        let model = "Qwen3VL-2B-Instruct-Q4_K_M";
        let descriptors = vec![local_gguf_descriptor(model)];

        assert!(cached_auto_model_satisfies_media_requirements(
            model,
            &media,
            &descriptors
        ));
    }
    #[test]
    fn test_request_budget_tokens_includes_output_budget_and_scaled_margin() {
        let body = serde_json::json!({
            "model": "qwen",
            "max_tokens": 512,
            "messages": [{"role": "user", "content": "hello world"}],
        });

        let budget = request_budget_tokens(&body).unwrap();
        let prompt_tokens = ceil_div_u32(serde_json::to_vec(&body).unwrap().len() as u32, 4);
        assert_eq!(
            budget,
            prompt_tokens + 512 + request_token_margin(prompt_tokens + 512)
        );
    }

    #[test]
    fn test_request_budget_tokens_uses_bounded_margin_for_small_requests() {
        let budget = request_budget_tokens_from_parts(128, Some(4)).unwrap();

        assert!(
            budget <= 256,
            "small smoke requests should fit a tiny CI context: {budget}"
        );
    }

    #[test]
    fn test_request_budget_tokens_keeps_full_margin_for_large_requests() {
        let budget = request_budget_tokens_from_parts(10_000, Some(512)).unwrap();

        assert_eq!(budget, 2_500 + 512 + REQUEST_TOKEN_MARGIN);
    }
    #[test]
    fn test_request_budget_tokens_bounds_generous_client_ceiling() {
        // Regression for #1350: Buzz Agent Desktop defaults max_completion_tokens
        // to 65,536, which meant a ~35.6 KB request "needed" 74,703 tokens and was
        // refused by every 65,536-context target, even though Skippy admission
        // would only have reserved prompt + 512.
        let prompt_tokens = ceil_div_u32(35_644, 4);
        let budget = request_budget_tokens_from_parts(35_644, Some(65_536)).unwrap();

        assert_eq!(
            budget,
            prompt_tokens + 512 + request_token_margin(prompt_tokens + 512)
        );
        assert!(
            budget <= 65_536,
            "generous client ceiling should still fit a 65,536-token target: {budget}"
        );
    }

    #[test]
    fn test_request_budget_tokens_matches_skippy_decode_headroom() {
        // Routing and Skippy admission must apply the same completion bound or
        // routing can refuse requests the backend would serve (#1350).
        let huge = request_budget_tokens_from_parts(4_096, Some(u32::MAX)).unwrap();
        let headroom = request_budget_tokens_from_parts(
            4_096,
            Some(skippy_serving::DECODE_BATCH_HEADROOM_TOKENS as u32),
        )
        .unwrap();

        assert_eq!(huge, headroom);
    }

    #[test]
    fn test_request_budget_tokens_still_counts_full_prompt() {
        // Only the completion term is bounded. An oversized prompt must still be
        // excluded from an undersized target.
        let budget = request_budget_tokens_from_parts(400_000, None).unwrap();

        assert!(budget > 65_536, "oversized prompt must not be bounded away");
    }

    #[test]
    fn test_request_budget_tokens_bounds_all_completion_limit_aliases() {
        let expected = request_budget_tokens_from_parts(1_024, Some(512)).unwrap();
        for key in [
            "max_completion_tokens",
            "max_tokens",
            "max_output_tokens",
            "n_predict",
        ] {
            let body = serde_json::json!({ key: 65_536u64 });
            let padded_len = 1_024;
            let completion = body.get(key).and_then(|v| v.as_u64()).map(|v| v as u32);
            assert_eq!(
                request_budget_tokens_from_parts(padded_len, completion).unwrap(),
                expected,
                "{key} should be bounded like every other completion alias"
            );
        }
    }

    #[test]
    fn test_request_budget_tokens_absent_completion_limit_reserves_nothing() {
        let prompt_tokens = ceil_div_u32(1_024, 4);
        let budget = request_budget_tokens_from_parts(1_024, None).unwrap();

        assert_eq!(budget, prompt_tokens + request_token_margin(prompt_tokens));
    }

    fn throughput(milli: u64) -> Option<TargetThroughputRank> {
        Some(TargetThroughputRank {
            avg_tokens_per_second_milli: milli,
            throughput_samples: 4,
            local_observation: false,
        })
    }

    #[test]
    fn rank_classes_follow_measurements_not_positions() {
        // Two tied fast targets share class 0 whatever their input order; the
        // slower one is class 1; the unknown-context tier starts a new class
        // even when its throughput ties the best adequate target; two
        // unmeasured targets inside one tier share a class.
        for order in [[0usize, 1, 2, 3, 4], [1, 0, 3, 2, 4]] {
            let specs = [
                ("fast-a", Some(8_192), throughput(40_000)),
                ("fast-b", Some(8_192), throughput(40_000)),
                ("slow", Some(8_192), throughput(10_000)),
                ("unknown-ctx-fast", None, throughput(40_000)),
                ("unknown-ctx-cold", None, None),
            ];
            let candidates: Vec<_> = order.iter().map(|index| specs[*index]).collect();
            let ranked = rank_candidates_by_context_and_throughput(&candidates, Some(4_096));
            let rank = |name: &str| ranked.placements[name].rank;
            assert_eq!(rank("fast-a"), 0);
            assert_eq!(rank("fast-b"), 0);
            assert_eq!(rank("slow"), 1);
            assert_eq!(rank("unknown-ctx-fast"), 2);
            assert_eq!(rank("unknown-ctx-cold"), 3);
            assert_eq!(ranked.equivalent_prefix, 2);
            assert_eq!(ranked.ordered[2], "slow");
        }
        let cold = [("a", None, None), ("b", None, None)];
        let ranked = rank_candidates_by_context_and_throughput(&cold, None);
        assert_eq!(ranked.placements["a"].rank, 0);
        assert_eq!(ranked.placements["b"].rank, 0);
    }

    #[test]
    fn test_equivalent_prefix_is_one_when_ranks_differ() {
        let ranked = rank_candidates_by_context_and_throughput(
            &[
                (1u8, Some(8192), throughput(40_000)),
                (2u8, Some(8192), throughput(10_000)),
            ],
            Some(4096),
        );

        assert_eq!(ranked.ordered, vec![1, 2]);
        assert_eq!(ranked.equivalent_prefix, 1);
    }

    #[test]
    fn test_equivalent_prefix_covers_tied_throughput_ranks() {
        let ranked = rank_candidates_by_context_and_throughput(
            &[
                (1u8, Some(8192), throughput(40_000)),
                (2u8, Some(8192), throughput(40_000)),
                (3u8, Some(8192), throughput(10_000)),
            ],
            Some(4096),
        );

        assert_eq!(ranked.ordered, vec![1, 2, 3]);
        assert_eq!(ranked.equivalent_prefix, 2);
    }

    #[test]
    fn test_equivalent_prefix_covers_all_unmeasured_candidates() {
        let ranked = rank_candidates_by_context_and_throughput(
            &[
                (1u8, Some(8192), None),
                (2u8, Some(8192), None),
                (3u8, None, None),
            ],
            Some(4096),
        );

        assert_eq!(ranked.ordered, vec![1, 2, 3]);
        assert_eq!(ranked.equivalent_prefix, 2);
    }

    #[test]
    fn test_equivalent_prefix_does_not_cross_into_unknown_context_tier() {
        let ranked = rank_candidates_by_context_and_throughput(
            &[(1u8, Some(8192), None), (2u8, None, None)],
            Some(4096),
        );

        assert_eq!(ranked.ordered, vec![1, 2]);
        assert_eq!(ranked.equivalent_prefix, 1);
    }

    #[test]
    fn test_reorder_candidates_by_context_prefers_known_fit_then_unknown() {
        let ordered = reorder_candidates_by_context_and_throughput(
            &[
                (1u8, Some(4096), None),
                (2u8, None, None),
                (3u8, Some(16384), None),
            ],
            Some(8192),
        );

        assert_eq!(ordered, vec![3, 2]);
    }

    #[test]
    fn test_reorder_candidates_by_context_rejects_all_known_too_small() {
        let ordered = reorder_candidates_by_context_and_throughput(
            &[(1u8, Some(4096), None), (2u8, Some(6144), None)],
            Some(8192),
        );

        assert!(ordered.is_empty());
    }

    #[test]
    fn test_reorder_candidates_by_context_keeps_unknown_when_known_too_small() {
        let ordered = reorder_candidates_by_context_and_throughput(
            &[(1u8, Some(4096), None), (2u8, None, None)],
            Some(8192),
        );

        assert_eq!(ordered, vec![2]);
    }

    #[test]
    fn test_reorder_candidates_without_throughput_preserves_stable_order() {
        let ordered = reorder_candidates_by_context_and_throughput(
            &[
                (1u8, Some(8192), None),
                (2u8, Some(8192), None),
                (3u8, None, None),
            ],
            Some(4096),
        );

        assert_eq!(ordered, vec![1, 2, 3]);
    }

    #[test]
    fn test_reorder_candidates_by_throughput_prefers_stronger_hint() {
        let ordered = reorder_candidates_by_context_and_throughput(
            &[
                (
                    1u8,
                    Some(8192),
                    Some(TargetThroughputRank {
                        avg_tokens_per_second_milli: 10_000,
                        throughput_samples: 4,
                        local_observation: false,
                    }),
                ),
                (
                    2u8,
                    Some(8192),
                    Some(TargetThroughputRank {
                        avg_tokens_per_second_milli: 40_000,
                        throughput_samples: 4,
                        local_observation: false,
                    }),
                ),
            ],
            Some(4096),
        );

        assert_eq!(ordered, vec![2, 1]);
    }

    #[test]
    fn test_reorder_candidates_uses_samples_as_tiebreaker_not_multiplier() {
        let ordered = reorder_candidates_by_context_and_throughput(
            &[
                (
                    1u8,
                    Some(8192),
                    Some(TargetThroughputRank {
                        avg_tokens_per_second_milli: 20_000,
                        throughput_samples: 32,
                        local_observation: false,
                    }),
                ),
                (
                    2u8,
                    Some(8192),
                    Some(TargetThroughputRank {
                        avg_tokens_per_second_milli: 40_000,
                        throughput_samples: 2,
                        local_observation: false,
                    }),
                ),
            ],
            Some(4096),
        );

        assert_eq!(ordered, vec![2, 1]);
    }

    #[test]
    fn test_reorder_candidates_keeps_context_fit_ahead_of_faster_unknown() {
        let ordered = reorder_candidates_by_context_and_throughput(
            &[
                (
                    1u8,
                    Some(8192),
                    Some(TargetThroughputRank {
                        avg_tokens_per_second_milli: 10_000,
                        throughput_samples: 4,
                        local_observation: false,
                    }),
                ),
                (
                    2u8,
                    None,
                    Some(TargetThroughputRank {
                        avg_tokens_per_second_milli: 90_000,
                        throughput_samples: 16,
                        local_observation: false,
                    }),
                ),
            ],
            Some(4096),
        );

        assert_eq!(ordered, vec![1, 2]);
    }

    #[test]
    fn test_reorder_candidates_weights_local_observations_above_gossip() {
        let ordered = reorder_candidates_by_context_and_throughput(
            &[
                (
                    1u8,
                    Some(8192),
                    Some(TargetThroughputRank {
                        avg_tokens_per_second_milli: 20_000,
                        throughput_samples: 3,
                        local_observation: true,
                    }),
                ),
                (
                    2u8,
                    Some(8192),
                    Some(TargetThroughputRank {
                        avg_tokens_per_second_milli: 50_000,
                        throughput_samples: 4,
                        local_observation: false,
                    }),
                ),
            ],
            Some(4096),
        );

        assert_eq!(ordered, vec![1, 2]);
    }
}
