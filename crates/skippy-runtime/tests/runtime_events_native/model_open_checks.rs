//! Per-call `ModelOpenEventQueue` checks against the real native runtime:
//! single-path open, multi-part open, and a failed open.

use std::path::Path;
use std::time::Duration;

use skippy_runtime::{
    ModelOpenEventQueue, NativeEventRecord, RuntimeConfig, RuntimeEventCategory,
    RuntimeEventKind as Kind, StageModel, next_operation_id,
};

use super::evidence::Evidence;
use super::families::{FamilyTally, drain_global};

const RUNTIME_EVENT_V1_ABI_VERSION: u32 = 1;

/// Size of the pre-extension event layout: every record must cover at
/// least this much, whatever extension the runtime appends.
fn base_struct_size() -> u32 {
    let base = std::mem::offset_of!(skippy_ffi::SkippyRuntimeEventV1, numeric_summary_0);
    u32::try_from(base).expect("base event layout fits in u32")
}

fn drain_queue(queue: &ModelOpenEventQueue) -> Vec<NativeEventRecord> {
    let mut records = Vec::new();
    queue.drain(&mut records, usize::MAX);
    records
}

fn kinds(records: &[NativeEventRecord]) -> Vec<Kind> {
    records
        .iter()
        .map(|record| record.to_event().kind)
        .collect()
}

/// Header invariants every record must satisfy, success or failure.
fn assert_record_headers(label: &str, records: &[NativeEventRecord]) {
    let base = base_struct_size();
    for record in records {
        assert_eq!(
            record.abi_version, RUNTIME_EVENT_V1_ABI_VERSION,
            "{label}: record abi_version must be v1: {record:?}"
        );
        assert!(
            record.struct_size >= base,
            "{label}: record struct_size {} is below the base layout {base}",
            record.struct_size
        );
    }
    for pair in records.windows(2) {
        assert!(
            pair[1].sequence > pair[0].sequence,
            "{label}: sequence must strictly increase in delivery order: {} then {}",
            pair[0].sequence,
            pair[1].sequence
        );
    }
}

/// Assertions (a)/(b): a successful open produced a complete, ordered,
/// lossless model-open record stream.
fn assert_successful_open(label: &str, queue: &ModelOpenEventQueue, records: &[NativeEventRecord]) {
    assert!(
        !records.is_empty(),
        "{label}: open produced no queue records"
    );
    assert_record_headers(label, records);
    let kinds = kinds(records);
    assert_eq!(
        kinds.first(),
        Some(&Kind::ModelOpenStarted),
        "{label}: first record must be ModelOpenStarted: {kinds:?}"
    );
    let last_model_open = records
        .iter()
        .rev()
        .map(NativeEventRecord::to_event)
        .find(|event| event.category == RuntimeEventCategory::ModelOpen)
        .map(|event| event.kind);
    assert_eq!(
        last_model_open,
        Some(Kind::ModelOpenFinished),
        "{label}: ModelOpenFinished must be the last model-open record: {kinds:?}"
    );
    assert!(
        !kinds.contains(&Kind::ModelOpenFailedHandled),
        "{label}: a successful open must not report ModelOpenFailedHandled: {kinds:?}"
    );
    assert_eq!(queue.dropped(), 0, "{label}: queue dropped records");
    assert_eq!(queue.rejected(), 0, "{label}: queue rejected records");
}

fn summarize(records: &[NativeEventRecord]) -> String {
    let first = records.first().map_or(0, |record| record.sequence);
    let last = records.last().map_or(0, |record| record.sequence);
    // Run-length encoded in delivery order: progress repeats hundreds of times.
    let mut runs: Vec<(Kind, usize)> = Vec::new();
    for kind in kinds(records) {
        match runs.last_mut() {
            Some((last, count)) if *last == kind => *count += 1,
            _ => runs.push((kind, 1)),
        }
    }
    let tally = runs
        .iter()
        .map(|(kind, count)| match count {
            1 => format!("{kind:?}"),
            _ => format!("{kind:?}x{count}"),
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "records={} seq={first}..{last} kinds=[{tally}]",
        records.len()
    )
}

/// Runs (a) and (b). Unload records from dropping each model land on the
/// global reporter and are added to `global_tally`.
pub fn check_successful_opens(
    model_path: &str,
    config: &RuntimeConfig,
    evidence: &Evidence,
    global_tally: &mut FamilyTally,
) {
    let single = ModelOpenEventQueue::new(next_operation_id());
    let model = StageModel::open_with_events(model_path, config, &single)
        .unwrap_or_else(|error| panic!("open_with_events on a real model failed: {error}"));
    let records = drain_queue(&single);
    assert_successful_open("open_with_events", &single, &records);
    evidence.record(format!(
        "open-with-events: ok {} dropped=0 rejected=0 abi_version=1 struct_size>={}",
        summarize(&records),
        base_struct_size()
    ));
    drop(model);
    global_tally.add(&drain_global());

    let parts = ModelOpenEventQueue::new(next_operation_id());
    let model = StageModel::open_from_parts_with_events(&[model_path], config, &parts)
        .unwrap_or_else(|error| {
            panic!("open_from_parts_with_events on a real model failed: {error}")
        });
    let records = drain_queue(&parts);
    assert_successful_open("open_from_parts_with_events", &parts, &records);
    evidence.record(format!(
        "open-from-parts-with-events: ok {} dropped=0 rejected=0 abi_version=1 struct_size>={}",
        summarize(&records),
        base_struct_size()
    ));
    drop(model);
    global_tally.add(&drain_global());
}

/// Runs (c) for one bad source: the open must fail, and once it has
/// returned, the queue must receive nothing more.
fn check_one_failed_open(label: &str, path: &Path, config: &RuntimeConfig, evidence: &Evidence) {
    let queue = ModelOpenEventQueue::new(next_operation_id());
    let result = StageModel::open_with_events(path, config, &queue);
    let error = match result {
        Ok(_) => panic!("{label}: open_with_events on {} must fail", path.display()),
        Err(error) => error,
    };
    let records = drain_queue(&queue);
    assert_record_headers(label, &records);
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(
        queue.len(),
        0,
        "{label}: callbacks arrived after open_with_events returned"
    );
    assert_eq!(queue.dropped(), 0, "{label}: queue dropped records");
    assert_eq!(queue.rejected(), 0, "{label}: queue rejected records");
    let kinds = kinds(&records);
    let failed_handled = kinds.contains(&Kind::ModelOpenFailedHandled);
    assert!(
        !kinds.contains(&Kind::ModelOpenFinished),
        "{label}: a failed open must not report ModelOpenFinished: {kinds:?}"
    );
    let first_line = error.to_string().lines().next().unwrap_or("").to_owned();
    evidence.record(format!(
        "{label}: err=\"{first_line}\" {} post-return-len-after-200ms=0 failed-handled-observed={failed_handled}",
        summarize(&records)
    ));
}

/// Runs (c) against a garbage `.gguf` file and a path that does not exist.
/// Global-reporter records raised by the failures are added to `failure_tally`.
pub fn check_failed_opens(
    config: &RuntimeConfig,
    evidence: &Evidence,
    failure_tally: &mut FamilyTally,
) {
    let dir = tempfile::tempdir().expect("tempdir for invalid model");
    let garbage = dir.path().join("invalid.gguf");
    std::fs::write(&garbage, b"this is not a gguf file\0\x01\x02\x03").expect("write garbage");
    check_one_failed_open("failed-open-garbage-gguf", &garbage, config, evidence);
    failure_tally.add(&drain_global());

    let missing = dir.path().join("does-not-exist.gguf");
    check_one_failed_open("failed-open-missing-path", &missing, config, evidence);
    failure_tally.add(&drain_global());
}
