use crate::inference::election;
use crate::mesh;
use crate::network::router;

pub(super) const REQUEST_TOKEN_MARGIN: u32 = 256;

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

pub(crate) fn request_budget_tokens_from_parts(
    body_len_bytes: usize,
    completion_tokens: Option<u32>,
) -> Option<u32> {
    if body_len_bytes == 0 {
        return None;
    }
    let prompt_tokens = ceil_div_u32(saturating_u32(body_len_bytes), 4);
    let completion_tokens = completion_tokens.unwrap_or(0);
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

fn target_throughput_rank_key(throughput: Option<TargetThroughputRank>) -> (bool, bool, u64, u64) {
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

pub(super) fn reorder_candidates_by_context_and_throughput<T: Clone>(
    candidates: &[(T, Option<u32>, Option<TargetThroughputRank>)],
    required_tokens: Option<u32>,
) -> Vec<T> {
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
        return ranked.into_iter().map(|ranked| ranked.candidate).collect();
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
        return Vec::new();
    }

    sort_ranked_targets(&mut adequate);
    sort_ranked_targets(&mut unknown);
    adequate
        .into_iter()
        .chain(unknown)
        .map(|ranked| ranked.candidate)
        .collect()
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

async fn remote_target_throughput_rank(
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

pub(super) async fn order_remote_hosts_by_context(
    node: &mesh::Node,
    model: &str,
    required_tokens: Option<u32>,
    hosts: &[iroh::EndpointId],
) -> Vec<iroh::EndpointId> {
    let mut candidates = Vec::with_capacity(hosts.len());
    for host in hosts {
        candidates.push((
            *host,
            node.peer_model_context_length(*host, model).await,
            remote_target_throughput_rank(node, model, *host).await,
        ));
    }
    reorder_candidates_by_context_and_throughput(&candidates, required_tokens)
}

pub(super) async fn order_targets_by_context(
    node: &mesh::Node,
    model: &str,
    required_tokens: Option<u32>,
    targets: &[election::InferenceTarget],
) -> Vec<election::InferenceTarget> {
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
    reorder_candidates_by_context_and_throughput(&candidates, required_tokens)
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

pub(crate) fn descriptor_metadata_for_model<'a>(
    model: &str,
    descriptors: &'a [mesh::ServedModelDescriptor],
) -> Option<&'a mesh::ServedModelMetadata> {
    descriptor_for_model(descriptors, model).and_then(|descriptor| descriptor.metadata.as_ref())
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
