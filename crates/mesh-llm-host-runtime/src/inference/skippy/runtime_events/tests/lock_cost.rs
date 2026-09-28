//! Caller-side cost of the adapter's `tracked` mutex. Generation threads call
//! `try_submit(Committed)` once per committed token batch, so the time a
//! producer spends inside the adapter (mutex wait included) is paid on the
//! decode path. This measures it under concurrent producers against a live
//! engine and holds it to the same budget as native callback ingress.

use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

use skippy_server::frontend::GenerationCommit;

use super::*;
use crate::runtime_events::config::CALLBACK_INGRESS_P99_BUDGET;

const PRODUCERS: usize = 8;
const COMMITS_PER_PRODUCER: usize = 2_000;

fn commit(request_id: u64, generated: usize) -> GenerationLifecycleObservation {
    GenerationLifecycleObservation::Committed(GenerationCommit {
        request_id,
        session_id: request_id,
        generated_token_count: generated,
        token_ids: Box::new([7]),
    })
}

fn producer_samples(
    adapter: &SkippyGenerationRuntimeEventAdapter,
    request_id: u64,
) -> Vec<Duration> {
    let mut samples = Vec::with_capacity(COMMITS_PER_PRODUCER);
    for generated in 1..=COMMITS_PER_PRODUCER {
        let begin = Instant::now();
        let _ = adapter.try_submit(commit(request_id, generated));
        samples.push(begin.elapsed());
    }
    samples
}

#[test]
#[serial_test::serial(runtime_event_engine_state)]
fn concurrent_commit_calls_stay_within_the_ingress_budget() {
    let engine = install_test_engine();
    let adapter = Arc::new(SkippyGenerationRuntimeEventAdapter::new());
    for request_id in 0..PRODUCERS as u64 {
        adapter
            .try_submit(GenerationLifecycleObservation::Started(start(
                request_id, request_id, None,
            )))
            .unwrap();
    }
    let barrier = Arc::new(Barrier::new(PRODUCERS));
    let workers: Vec<_> = (0..PRODUCERS as u64)
        .map(|request_id| {
            let adapter = adapter.clone();
            let barrier = barrier.clone();
            thread::spawn(move || {
                barrier.wait();
                producer_samples(&adapter, request_id)
            })
        })
        .collect();
    let mut samples: Vec<Duration> = workers
        .into_iter()
        .flat_map(|worker| worker.join().expect("producer panicked"))
        .collect();
    samples.sort_unstable();
    let p50 = samples[samples.len() / 2];
    let p99 = samples[samples.len() * 99 / 100];
    let max = *samples.last().unwrap();
    eprintln!(
        "adapter commit producers={PRODUCERS} calls={} p50={p50:?} p99={p99:?} max={max:?}",
        samples.len()
    );
    drop(engine);
    clear_runtime_event_engine();
    // Unoptimized builds inflate every call; only release numbers are held
    // to the budget. Debug runs still print the measurement.
    if cfg!(debug_assertions) {
        return;
    }
    assert!(
        p99 <= CALLBACK_INGRESS_P99_BUDGET,
        "adapter commit p99 {p99:?} (p50 {p50:?}, max {max:?}) exceeds {CALLBACK_INGRESS_P99_BUDGET:?}"
    );
}
