#[path = "../tests/support/mod.rs"]
mod support;

use std::hint::black_box;
use std::io::{self, Write};
use std::time::Instant;

use skippy_scheduler::SchedulerConfig;
use support::{
    RuntimeCostModel, SimRequest, SimulationReport, apply_resident_radix_affinity, burst_requests,
    simulate, staggered_prefill_requests,
};

const BENCH_REPETITIONS: usize = 200;

struct Scenario {
    name: String,
    requests: Vec<SimRequest>,
    max_consecutive_prefill_iterations: usize,
    mixed_prefill_decode: bool,
    max_active_sequences: usize,
    group_waiting_prefixes: bool,
}

fn main() {
    let config = SchedulerConfig::default();
    let cost = RuntimeCostModel::default();
    let stdout = io::stdout();
    let mut output = io::BufWriter::new(stdout.lock());
    writeln!(
        output,
        "scenario\trequests\tmakespan_ms\trequest_s\tp95_queue_ms\tp50_ttft_ms\tp95_ttft_ms\tp95_latency_ms\tp95_max_itl_ms\tmean_batch\tmean_token_occupancy_pct\tprefill_iterations\tdecode_iterations\tmixed_iterations\tbench_us_per_run"
    )
    .expect("write scheduler lab header");
    for scenario in scenarios() {
        let scenario_config = SchedulerConfig {
            max_consecutive_prefill_iterations: scenario.max_consecutive_prefill_iterations,
            mixed_prefill_decode: scenario.mixed_prefill_decode,
            max_active_sequences: scenario.max_active_sequences,
            group_waiting_prefixes: scenario.group_waiting_prefixes,
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
        print_report(&mut output, &scenario.name, &report, bench_us);
    }
    output.flush().expect("flush scheduler lab report");
}

fn scenarios() -> Vec<Scenario> {
    let mut scenarios = Vec::new();
    for concurrency in [1, 2, 4] {
        scenarios.push(Scenario {
            name: format!("cold-burst-n{concurrency}"),
            requests: burst_requests(concurrency, 4_096, 32, 0),
            max_consecutive_prefill_iterations: usize::MAX,
            mixed_prefill_decode: false,
            max_active_sequences: 32,
            group_waiting_prefixes: true,
        });
        scenarios.push(Scenario {
            name: format!("warm-divergent-n{concurrency}"),
            requests: burst_requests(concurrency, 4_096, 32, 4_080),
            max_consecutive_prefill_iterations: usize::MAX,
            mixed_prefill_decode: false,
            max_active_sequences: 32,
            group_waiting_prefixes: true,
        });
    }
    scenarios.push(Scenario {
        name: "staggered-prefill".to_string(),
        requests: staggered_prefill_requests(),
        max_consecutive_prefill_iterations: usize::MAX,
        mixed_prefill_decode: false,
        max_active_sequences: 32,
        group_waiting_prefixes: true,
    });
    scenarios.push(Scenario {
        name: "staggered-prefill-bounded".to_string(),
        requests: staggered_prefill_requests(),
        max_consecutive_prefill_iterations: 1,
        mixed_prefill_decode: false,
        max_active_sequences: 32,
        group_waiting_prefixes: true,
    });
    scenarios.push(Scenario {
        name: "staggered-prefill-mixed".to_string(),
        requests: staggered_prefill_requests(),
        max_consecutive_prefill_iterations: 1,
        mixed_prefill_decode: true,
        max_active_sequences: 32,
        group_waiting_prefixes: true,
    });
    let mut radix = skippy_cache::UnifiedRadixCache::<&str, ()>::new();
    let cached_tokens = (0..3_072).collect::<Vec<i32>>();
    radix
        .insert_resident("stage-0", &cached_tokens, 3_072, "shared-prefix")
        .expect("valid scheduler-lab radix fixture");
    let mut cache_aware = vec![
        SimRequest::new("a-cold", 0, 4_096, 16).with_token_offset(10_000),
        SimRequest::new("b-cold", 0, 4_096, 16).with_token_offset(20_000),
        SimRequest::new("c-hot", 0, 4_096, 16),
    ];
    apply_resident_radix_affinity(
        &radix,
        "stage-0",
        0,
        RuntimeCostModel::default().prefill_token_us,
        &mut cache_aware,
    );
    let mut fcfs = cache_aware.clone();
    for request in &mut fcfs {
        request.cache_affinity = skippy_scheduler::CacheAffinity::default();
    }
    scenarios.push(Scenario {
        name: "radix-fcfs".to_string(),
        requests: fcfs,
        max_consecutive_prefill_iterations: 1,
        mixed_prefill_decode: false,
        max_active_sequences: 1,
        group_waiting_prefixes: true,
    });
    scenarios.push(Scenario {
        name: "radix-cache-aware".to_string(),
        requests: cache_aware,
        max_consecutive_prefill_iterations: 1,
        mixed_prefill_decode: false,
        max_active_sequences: 1,
        group_waiting_prefixes: true,
    });
    let waiting_prefix_requests = vec![
        SimRequest::new("a-unique", 0, 4_096, 16).with_token_offset(10_000),
        SimRequest::new("b-shared", 0, 4_096, 16),
        SimRequest::new("c-shared", 0, 4_096, 16),
    ];
    scenarios.push(Scenario {
        name: "waiting-prefix-fcfs".to_string(),
        requests: waiting_prefix_requests.clone(),
        max_consecutive_prefill_iterations: 1,
        mixed_prefill_decode: false,
        max_active_sequences: 1,
        group_waiting_prefixes: false,
    });
    scenarios.push(Scenario {
        name: "waiting-prefix-dfs".to_string(),
        requests: waiting_prefix_requests,
        max_consecutive_prefill_iterations: 1,
        mixed_prefill_decode: false,
        max_active_sequences: 1,
        group_waiting_prefixes: true,
    });
    scenarios
}

fn print_report(output: &mut impl Write, name: &str, report: &SimulationReport, bench_us: f64) {
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
    writeln!(
        output,
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
    )
    .expect("write scheduler lab report");
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
