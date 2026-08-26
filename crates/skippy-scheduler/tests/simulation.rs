mod support;

use skippy_scheduler::SchedulerConfig;
use support::{RuntimeCostModel, burst_requests, simulate, staggered_prefill_requests};

#[test]
fn scheduler_simulation_is_deterministic() {
    let config = SchedulerConfig::default();
    let cost = RuntimeCostModel::default();
    let requests = burst_requests(4, 512, 16, 0);

    let first = simulate(config.clone(), cost, requests.clone()).unwrap();
    let second = simulate(config, cost, requests).unwrap();

    assert_eq!(first, second);
}

#[test]
fn prefix_restore_reduces_modeled_ttft() {
    let config = SchedulerConfig::default();
    let cost = RuntimeCostModel::default();
    let cold = simulate(config.clone(), cost, burst_requests(4, 4_096, 16, 0)).unwrap();
    let warm = simulate(config, cost, burst_requests(4, 4_096, 16, 4_080)).unwrap();

    assert!(
        warm.request("request-0").ttft_us().unwrap() < cold.request("request-0").ttft_us().unwrap()
    );
    assert!(warm.makespan_us < cold.makespan_us);
}

#[test]
fn staggered_prefill_exposes_decode_head_of_line_blocking() {
    let config = SchedulerConfig::default();
    let cost = RuntimeCostModel::default();
    let uninterrupted = simulate(
        config.clone(),
        cost,
        vec![support::SimRequest::new("decoder", 0, 32, 64)],
    )
    .unwrap();
    let staggered = simulate(config, cost, staggered_prefill_requests()).unwrap();

    assert_eq!(staggered.mixed_iterations, 0);
    assert!(
        staggered.request("decoder").max_inter_token_gap_us
            > uninterrupted.request("decoder").max_inter_token_gap_us * 4
    );
    assert!(staggered.request("decoder").completed_us.is_some());
}

#[test]
fn bounded_prefill_iterations_reduce_decode_head_of_line_blocking() {
    let unbounded = simulate(
        SchedulerConfig::default(),
        RuntimeCostModel::default(),
        staggered_prefill_requests(),
    )
    .unwrap();
    let bounded = simulate(
        SchedulerConfig {
            max_consecutive_prefill_iterations: 1,
            ..SchedulerConfig::default()
        },
        RuntimeCostModel::default(),
        staggered_prefill_requests(),
    )
    .unwrap();

    assert_eq!(bounded.mixed_iterations, 0);
    assert!(
        bounded
            .request("decoder")
            .max_inter_token_gap_us
            .saturating_mul(4)
            < unbounded.request("decoder").max_inter_token_gap_us
    );
}

#[test]
fn concurrent_burst_uses_batches_and_completes_every_request() {
    let report = simulate(
        SchedulerConfig::default(),
        RuntimeCostModel::default(),
        burst_requests(4, 128, 32, 0),
    )
    .unwrap();

    assert!(report.mean_batch_size > 1.0);
    assert!(report.mean_token_occupancy > 0.0);
    assert!(report.throughput_requests_per_second() > 0.0);
    assert!(report.requests.values().all(|request| {
        request.queue_wait_us().is_some()
            && request.latency_us().is_some()
            && request.generated_tokens == 32
    }));
}

#[test]
fn real_radix_affinity_prioritizes_the_high_value_prefix() {
    let mut radix = skippy_cache::UnifiedRadixCache::<&str, ()>::new();
    let cached_tokens = (0..768).collect::<Vec<i32>>();
    radix
        .insert_resident("stage-0", &cached_tokens, 768, "hot-prefix")
        .unwrap();
    let requests = vec![
        support::SimRequest::new("a-cold", 0, 1_024, 1).with_token_offset(10_000),
        support::SimRequest::new("b-hot", 0, 1_024, 1),
    ];
    let mut cache_aware_requests = requests;
    support::apply_resident_radix_affinity(
        &radix,
        "stage-0",
        0,
        RuntimeCostModel::default().prefill_token_us,
        &mut cache_aware_requests,
    );
    let mut fcfs_requests = cache_aware_requests.clone();
    for request in &mut fcfs_requests {
        request.cache_affinity = skippy_scheduler::CacheAffinity::default();
    }
    let fcfs = simulate(
        SchedulerConfig {
            max_active_sequences: 1,
            ..SchedulerConfig::default()
        },
        RuntimeCostModel::default(),
        fcfs_requests,
    )
    .unwrap();
    let cache_aware = simulate(
        SchedulerConfig {
            max_active_sequences: 1,
            ..SchedulerConfig::default()
        },
        RuntimeCostModel::default(),
        cache_aware_requests,
    )
    .unwrap();

    assert!(
        cache_aware.request("b-hot").ttft_us().unwrap() < fcfs.request("b-hot").ttft_us().unwrap()
    );
    assert_eq!(cache_aware.request("b-hot").queue_wait_us(), Some(0));
}
