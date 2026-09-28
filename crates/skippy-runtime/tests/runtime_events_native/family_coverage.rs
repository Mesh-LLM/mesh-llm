//! Assertion (d): every confirmed event family fires on the process-global
//! reporter during a real load, session, and unload.

use skippy_runtime::{BackendDeviceType, CapabilityReport, RuntimeConfig, StageModel};

use super::evidence::Evidence;
use super::families::{Family, FamilyTally, drain_global};

const PROMPT: &str = "The capital of France is";
const DECODE_STEPS: usize = 4;

/// Device events are raised only when a backend device is explicitly
/// selected (`skippy_apply_selected_backend_device`), which is how the host
/// opens models. Prefer the first non-CPU device, as production does.
fn selected_device() -> String {
    let devices = skippy_runtime::backend_devices().unwrap_or_default();
    devices
        .iter()
        .find(|device| device.device_type != BackendDeviceType::Cpu)
        .map_or_else(|| "CPU".to_owned(), |device| device.name.clone())
}

/// Tokenize, prefill, decode a few steps, and reset: the operations that
/// raise KV events on a real session.
fn exercise_session(model: &StageModel, evidence: &Evidence) {
    let tokens = model
        .tokenize(PROMPT, true)
        .unwrap_or_else(|error| panic!("tokenize prompt: {error}"));
    assert!(tokens.len() >= 2, "prompt tokenized to too few tokens");
    let mut session = model
        .create_session()
        .unwrap_or_else(|error| panic!("create_session: {error}"));
    let (prefix, last) = tokens.split_at(tokens.len() - 1);
    session
        .prefill_chunked(prefix)
        .unwrap_or_else(|error| panic!("prefill: {error}"));
    let mut next = last[0];
    for _ in 0..DECODE_STEPS {
        next = session
            .decode_step(next)
            .unwrap_or_else(|error| panic!("decode_step: {error}"));
    }
    session
        .reset()
        .unwrap_or_else(|error| panic!("session reset: {error}"));
    evidence.record(format!(
        "session: prompt_tokens={} decoded_steps={DECODE_STEPS} reset=ok",
        tokens.len()
    ));
}

/// Loads with a selected device, runs a session, unloads, and adds every
/// global-reporter record to `tally`. Returns the plain open's structured
/// callback count for the legacy `structured-production-callbacks` marker.
pub fn run_load_session_unload(
    model_path: &str,
    base_config: &RuntimeConfig,
    evidence: &Evidence,
    tally: &mut FamilyTally,
) -> usize {
    let device = selected_device();
    let config = RuntimeConfig {
        selected_backend_device: Some(device.clone()),
        ..base_config.clone()
    };
    let model = StageModel::open(model_path, &config)
        .unwrap_or_else(|error| panic!("real model-open with device {device} failed: {error}"));
    let mut open_tally = FamilyTally::default();
    open_tally.add(&drain_global());
    evidence.record(format!(
        "model-open: single-part real model-open succeeded selected_backend_device={device}"
    ));

    let mut session_tally = FamilyTally::default();
    exercise_session(&model, evidence);
    session_tally.add(&drain_global());

    let mut unload_tally = FamilyTally::default();
    drop(model);
    unload_tally.add(&drain_global());
    evidence.record(format!(
        "unload-callbacks: {}",
        unload_tally.count(Family::Unload)
    ));

    evidence.record(format!(
        "session-kv-callbacks: {}",
        session_tally.count(Family::Kv)
    ));
    for phase in [&open_tally, &session_tally, &unload_tally] {
        tally.merge(phase);
    }
    open_tally.structured()
}

/// Per confirmed family: record OBSERVED/NOT_OBSERVED and return the
/// required families that never fired.
pub fn judge_family_coverage(
    report: &CapabilityReport,
    tally: &FamilyTally,
    evidence: &Evidence,
) -> Vec<&'static str> {
    let mut missing_required = Vec::new();
    for family in Family::ALL {
        let confirmed = report.family_confirmed(family.feature_bit());
        let count = tally.count(family);
        let verdict = match (confirmed, count > 0) {
            (false, _) => "NOT_CONFIRMED",
            (true, true) => "OBSERVED",
            (true, false) => "NOT_OBSERVED",
        };
        evidence.record(format!(
            "family-coverage: {} confirmed={confirmed} count={count} {verdict}",
            family.name()
        ));
        if confirmed && count == 0 && family.required_when_confirmed() {
            missing_required.push(family.name());
        }
    }
    missing_required
}
