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
