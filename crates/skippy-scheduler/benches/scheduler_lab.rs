#[path = "../tests/support/mod.rs"]
mod support;

use std::hint::black_box;
use std::time::Instant;

use skippy_scheduler::SchedulerConfig;
use support::{
    RuntimeCostModel, SimRequest, SimulationReport, burst_requests, simulate,
    staggered_prefill_requests,
};

const BENCH_REPETITIONS: usize = 200;

struct Scenario {
    name: String,
    requests: Vec<SimRequest>,
    max_consecutive_prefill_iterations: usize,
}

fn main() {
    let config = SchedulerConfig::default();
    let cost = RuntimeCostModel::default();
    println!(
        "scenario\trequests\tmakespan_ms\trequest_s\tp95_queue_ms\tp50_ttft_ms\tp95_ttft_ms\tp95_latency_ms\tp95_max_itl_ms\tmean_batch\tmean_token_occupancy_pct\tprefill_iterations\tdecode_iterations\tmixed_iterations\tbench_us_per_run"
    );
    for scenario in scenarios() {
        let scenario_config = SchedulerConfig {
            max_consecutive_prefill_iterations: scenario.max_consecutive_prefill_iterations,
            ..config.clone()
        };
        let report = simulate(scenario_config.clone(), cost, scenario.requests.clone())
            .unwrap_or_else(|error| panic!("{}: {error}", scenario.name));
        let started = Instant::now();
        for _ in 0..BENCH_REPETITIONS {
            black_box(
                simulate(scenario_config.clone(), cost, scenario.requests.clone())
                    .unwrap_or_else(|error| panic!("{}: {error}", scenario.name)),
            );
        }
        let bench_us = started.elapsed().as_secs_f64() * 1_000_000.0 / BENCH_REPETITIONS as f64;
        print_report(&scenario.name, &report, bench_us);
    }
}

fn scenarios() -> Vec<Scenario> {
    let mut scenarios = Vec::new();
    for concurrency in [1, 2, 4] {
        scenarios.push(Scenario {
            name: format!("cold-burst-n{concurrency}"),
            requests: burst_requests(concurrency, 4_096, 32, 0),
            max_consecutive_prefill_iterations: usize::MAX,
        });
        scenarios.push(Scenario {
            name: format!("warm-divergent-n{concurrency}"),
            requests: burst_requests(concurrency, 4_096, 32, 4_080),
            max_consecutive_prefill_iterations: usize::MAX,
        });
    }
    scenarios.push(Scenario {
        name: "staggered-prefill".to_string(),
        requests: staggered_prefill_requests(),
        max_consecutive_prefill_iterations: usize::MAX,
    });
    scenarios.push(Scenario {
        name: "staggered-prefill-bounded".to_string(),
        requests: staggered_prefill_requests(),
        max_consecutive_prefill_iterations: 1,
    });
    scenarios
}

fn print_report(name: &str, report: &SimulationReport, bench_us: f64) {
    let queue_wait = report
        .requests
        .values()
        .filter_map(support::RequestMetrics::queue_wait_us)
        .collect::<Vec<_>>();
    let ttft = report
        .requests
        .values()
        .filter_map(support::RequestMetrics::ttft_us)
        .collect::<Vec<_>>();
    let latency = report
        .requests
        .values()
        .filter_map(support::RequestMetrics::latency_us)
        .collect::<Vec<_>>();
    let max_itl = report
        .requests
        .values()
        .map(|request| request.max_inter_token_gap_us)
        .collect::<Vec<_>>();
    black_box(
        report.request(
            report
                .requests
                .first_key_value()
                .expect("scenario must contain requests")
                .0,
        ),
    );
    println!(
        "{}\t{}\t{:.3}\t{:.2}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.2}\t{:.2}\t{}\t{}\t{}\t{:.2}",
        name,
        report.requests.len(),
        report.makespan_us as f64 / 1_000.0,
        report.throughput_requests_per_second(),
        percentile(&queue_wait, 95) as f64 / 1_000.0,
        percentile(&ttft, 50) as f64 / 1_000.0,
        percentile(&ttft, 95) as f64 / 1_000.0,
        percentile(&latency, 95) as f64 / 1_000.0,
        percentile(&max_itl, 95) as f64 / 1_000.0,
        report.mean_batch_size,
        report.mean_token_occupancy * 100.0,
        report.prefill_iterations,
        report.decode_iterations,
        report.mixed_iterations,
        bench_us,
    );
}

fn percentile(values: &[u64], percentile: usize) -> u64 {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let index = sorted
        .len()
        .saturating_mul(percentile)
        .div_ceil(100)
        .saturating_sub(1)
        .min(sorted.len().saturating_sub(1));
    sorted.get(index).copied().unwrap_or(0)
}
