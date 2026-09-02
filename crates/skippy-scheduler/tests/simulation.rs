mod support;

use std::collections::BTreeMap;

use serde::Deserialize;
use skippy_scheduler::SchedulerConfig;
use support::{RuntimeCostModel, burst_requests, simulate, staggered_prefill_requests};

#[derive(Deserialize)]
struct FixtureCatalog {
    profiles: BTreeMap<String, FixtureProfile>,
}

#[derive(Deserialize)]
struct FixtureProfile {
    ci_trace: FixtureTrace,
}

#[derive(Deserialize)]
struct FixtureTrace {
    family_order: Vec<i32>,
    prompt_tokens: usize,
    expected_fcfs_switches: usize,
    expected_dfs_switches: usize,
}

fn scheduler_fixture_catalog() -> FixtureCatalog {
    serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../evals/skippy-scheduler-fixtures.json"
    )))
    .expect("valid checked-in scheduler fixture catalog")
}

fn replay_family_switches(trace: &FixtureTrace, group_waiting_prefixes: bool) -> usize {
    let mut request_families = BTreeMap::new();
    let requests = trace
        .family_order
        .iter()
        .enumerate()
        .map(|(index, family)| {
            let id = format!("request-{index:03}-family-{family:02}");
            request_families.insert(id.clone(), *family);
            support::SimRequest::new(id, 0, trace.prompt_tokens, 1)
                .with_token_offset(family.saturating_add(1).saturating_mul(10_000))
        })
        .collect();
    let report = simulate(
        SchedulerConfig {
            max_active_sequences: 1,
            group_waiting_prefixes,
            ..SchedulerConfig::default()
        },
        RuntimeCostModel::default(),
        requests,
    )
    .expect("fixture replay completes");
    let mut service_order = report
        .requests
        .iter()
        .map(|(id, metrics)| {
            (
                metrics
                    .first_scheduled_us
                    .expect("fixture request was scheduled"),
                request_families[id],
            )
        })
        .collect::<Vec<_>>();
    service_order.sort_unstable();
    service_order
        .windows(2)
        .filter(|pair| pair[0].1 != pair[1].1)
        .count()
}

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
fn named_scheduler_profiles_replay_the_measured_locality_boundary() {
    for (name, profile) in scheduler_fixture_catalog().profiles {
        let fcfs_switches = replay_family_switches(&profile.ci_trace, false);
        let dfs_switches = replay_family_switches(&profile.ci_trace, true);

        assert_eq!(
            fcfs_switches, profile.ci_trace.expected_fcfs_switches,
            "{name} FCFS trace drifted"
        );
        assert_eq!(
            dfs_switches, profile.ci_trace.expected_dfs_switches,
            "{name} waiting-prefix trace drifted"
        );
    }
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
fn mixed_iterations_fill_prefill_capacity_without_starving_decode() {
    let bounded = simulate(
        SchedulerConfig {
            max_consecutive_prefill_iterations: 1,
            ..SchedulerConfig::default()
        },
        RuntimeCostModel::default(),
        staggered_prefill_requests(),
    )
    .unwrap();
    let mixed = simulate(
        SchedulerConfig {
            max_consecutive_prefill_iterations: 1,
            mixed_prefill_decode: true,
            ..SchedulerConfig::default()
        },
        RuntimeCostModel::default(),
        staggered_prefill_requests(),
    )
    .unwrap();

    assert!(mixed.mixed_iterations > 0);
    assert!(mixed.mean_token_occupancy > bounded.mean_token_occupancy);
    assert!(
        mixed.request("decoder").max_inter_token_gap_us
            < bounded.request("decoder").max_inter_token_gap_us
    );
    assert!(mixed.makespan_us < bounded.makespan_us);
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
