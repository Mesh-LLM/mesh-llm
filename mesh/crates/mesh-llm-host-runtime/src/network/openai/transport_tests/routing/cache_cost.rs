//! Cache-hit routing against cold peers, costed on the current request.

use super::*;

/// The issue's fixture: 4,096 prompt tokens, 1,024 provably shared. A hit on a
/// peer prefilling at 2 ms/token costs 6.144 s (plus the 256-token margin),
/// while a peer that holds nothing for this prefix but prefills at 0.25
/// ms/token serves it cold in 1.024 s, so the cold peer is chosen. The hit
/// only wins once the cold peer is slower than the warm one's estimate.
#[tokio::test]
async fn a_slow_cache_hit_loses_to_a_faster_cold_peer() -> Result<()> {
    use mesh_llm_routing::cache_inventory::{
        CACHE_AFFINITY_SALT_BYTES, CacheAffinityAdvertisement, CacheAffinityEntry, CacheTier,
        prefix_digest,
    };
    use mesh_llm_routing::request_cost::RequestCostInput;

    let model = "acme/code-model:Q4_K_M";
    let warm_peer = iroh::EndpointId::from(iroh::SecretKey::generate().public());
    let cold_peer = iroh::EndpointId::from(iroh::SecretKey::generate().public());
    let salt = [5; CACHE_AFFINITY_SALT_BYTES];
    let advertisement =
        |prefix_hash: u64, matched_tokens: u32, rate: u64| CacheAffinityAdvertisement {
            salt,
            epoch: 1,
            generated_at_unix_ms: crate::mesh::current_time_unix_ms(),
            ttl_ms: 120_000,
            entries: vec![CacheAffinityEntry {
                model: model.to_string(),
                prefix_digest: prefix_digest(&salt, model, prefix_hash),
                matched_tokens,
                suffix_prefill_tokens: 0,
                tier: CacheTier::L1,
                restore_micros: 0,
                queue_delay_micros: 0,
                prefill_micros_per_token: rate,
            }],
        };

    for (cold_rate, expect_hit) in [(250, false), (4_000, true)] {
        let node = test_node_with_remote_models(&[(model, warm_peer), (model, cold_peer)]).await;
        let affinity = AffinityRouter::new();
        let mut request = unparsed_chat_request(model);
        let mut prepared = prepare_mesh_targets(
            &mut request,
            Some(model),
            &[warm_peer, cold_peer],
            &affinity,
        );
        let prefix_hash = prepared.prefix_hash.expect("derived prefix");
        let mut warm = test_peer_serving_model(warm_peer, model);
        warm.cache_affinity = Some(advertisement(prefix_hash, 1_024, 2_000));
        node.insert_test_peer(warm).await;
        // The cold peer's only entry is for another prefix: it carries its
        // prefill rate, not a hit for this request.
        let mut cold = test_peer_serving_model(cold_peer, model);
        cold.cache_affinity = Some(advertisement(prefix_hash ^ 1, 1_024, cold_rate));
        node.insert_test_peer(cold).await;
        // A lease from an earlier decision does not override a costed one.
        affinity.remember_cache_lease(
            model,
            prefix_hash,
            &election::InferenceTarget::Remote(warm_peer),
        );

        let (ordered, _, _) = order_mesh_target_hosts(
            &node,
            Some(model),
            None,
            &mut prepared,
            &affinity,
            RequestCostInput::heuristic(4_096, 1_024),
        )
        .await;

        // The cheaper option leads: the hit, or the faster peer served cold.
        let expected = if expect_hit { warm_peer } else { cold_peer };
        assert_eq!(
            prepared.cache_target,
            Some(election::InferenceTarget::Remote(expected)),
            "cold rate {cold_rate}"
        );
        assert_eq!(ordered.first(), Some(&expected));
    }
    Ok(())
}

/// A peer holding no cache entry is costed cold from the prefill rate on its
/// throughput hint, so a fast idle peer is chosen over a slow hit; once that
/// rate is stale it is uncosted again and the hit wins.
#[tokio::test]
async fn a_cold_peer_with_only_its_advertised_rate_can_veto_a_slow_hit() -> Result<()> {
    use crate::network::metrics::{ModelThroughputHint, PrefillMeasurement};
    use mesh_llm_routing::cache_inventory::{
        CACHE_AFFINITY_SALT_BYTES, CacheAffinityAdvertisement, CacheAffinityEntry, CacheTier,
        prefix_digest,
    };
    use mesh_llm_routing::request_cost::RequestCostInput;

    let model = "acme/code-model:Q4_K_M";
    let warm_peer = iroh::EndpointId::from(iroh::SecretKey::generate().public());
    let cold_peer = iroh::EndpointId::from(iroh::SecretKey::generate().public());
    let salt = [6; CACHE_AFFINITY_SALT_BYTES];
    let now = crate::mesh::current_time_unix_ms();
    let fifteen_minutes = 15 * 60 * 1000;

    for (rate_age_ms, expect_hit) in [(1_000, false), (fifteen_minutes + 1_000, true)] {
        let node = test_node_with_remote_models(&[(model, warm_peer), (model, cold_peer)]).await;
        let affinity = AffinityRouter::new();
        let mut request = unparsed_chat_request(model);
        let mut prepared = prepare_mesh_targets(
            &mut request,
            Some(model),
            &[warm_peer, cold_peer],
            &affinity,
        );
        let prefix_hash = prepared.prefix_hash.expect("derived prefix");
        let mut warm = test_peer_serving_model(warm_peer, model);
        warm.cache_affinity = Some(CacheAffinityAdvertisement {
            salt,
            epoch: 1,
            generated_at_unix_ms: now,
            ttl_ms: 120_000,
            entries: vec![CacheAffinityEntry {
                model: model.to_string(),
                prefix_digest: prefix_digest(&salt, model, prefix_hash),
                matched_tokens: 1_024,
                suffix_prefill_tokens: 0,
                tier: CacheTier::L1,
                restore_micros: 0,
                queue_delay_micros: 0,
                prefill_micros_per_token: 2_000,
            }],
        });
        node.insert_test_peer(warm).await;
        let mut cold = test_peer_serving_model(cold_peer, model);
        cold.cache_affinity = None;
        cold.advertised_model_throughput = vec![ModelThroughputHint {
            model_name: model.to_string(),
            avg_tokens_per_second_milli: 40_000,
            throughput_samples: 8,
            measurement: None,
            prefill: Some(PrefillMeasurement {
                micros_per_token: 250,
                observed_at_unix_ms: now - rate_age_ms,
            }),
        }];
        node.insert_test_peer(cold).await;

        order_mesh_target_hosts(
            &node,
            Some(model),
            None,
            &mut prepared,
            &affinity,
            RequestCostInput::heuristic(4_096, 1_024),
        )
        .await;

        // A fresh rate makes the idle peer the cheaper target; a stale one
        // leaves it uncosted, and the hit wins against its own cold cost.
        let expected = if expect_hit { warm_peer } else { cold_peer };
        assert_eq!(
            prepared.cache_target,
            Some(election::InferenceTarget::Remote(expected)),
            "rate age {rate_age_ms} ms"
        );
    }
    Ok(())
}
