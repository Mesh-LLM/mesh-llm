//! Real native-runtime integration test for the runtime-event ABI:
//! admission, capability probe, the process-global reporter, and per-call
//! `ModelOpenEventQueue` delivery. Gated behind
//! `MESH_LLM_RUNTIME_EVENTS_NATIVE_TEST=1` so it never touches a native
//! symbol during an ordinary `cargo test`. An ungated run still executes
//! (it is never skipped), but it must never claim `executed`: it prints a
//! `BLOCKED: <prerequisite>` line to stdout, writes only a
//! `blocked-when-ungated: <prerequisite>` evidence marker, and exits 0.
//!
//! Gated steps record their evidence in memory. `executed` is written, followed
//! by those markers, only after every step passed; a failing step writes
//! `failed: <panic>` followed by what the earlier steps established, and the
//! test still fails. A marker file that starts with `executed` therefore
//! reflects a genuine, complete run (review defect D10 -- see
//! `.omo/plans/event-system-fixes.md` task 11).
//!
//! The three prerequisites checked once the gate is set to `1` (the
//! `dynamic-native-runtime` feature, the native runtime bundle directory,
//! and the model path) each print their own `BLOCKED: <reason>` line before
//! panicking: a developer who explicitly opted into the real native gate
//! gets a loud, named failure for a misconfigured opt-in, not a silent
//! pass. Only the top-level gate-unset path is required to stay green,
//! since that is the path an ordinary `cargo test` run takes.

use std::env;
use std::path::PathBuf;

#[cfg(feature = "dynamic-native-runtime")]
#[path = "runtime_events_native/evidence.rs"]
mod evidence;
#[cfg(feature = "dynamic-native-runtime")]
#[path = "runtime_events_native/families.rs"]
mod families;
#[cfg(feature = "dynamic-native-runtime")]
#[path = "runtime_events_native/family_coverage.rs"]
mod family_coverage;
#[cfg(feature = "dynamic-native-runtime")]
#[path = "runtime_events_native/libraries.rs"]
mod libraries;
#[cfg(feature = "dynamic-native-runtime")]
#[path = "runtime_events_native/model_open_checks.rs"]
mod model_open_checks;

const GATE_ENV: &str = "MESH_LLM_RUNTIME_EVENTS_NATIVE_TEST";
#[cfg(feature = "dynamic-native-runtime")]
const BUNDLE_DIR_ENV: &str = "MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR";
#[cfg(feature = "dynamic-native-runtime")]
const MODEL_ENV: &str = "MESH_LLM_RUNTIME_EVENTS_MODEL";
const EVIDENCE_FILE_ENV: &str = "MESH_LLM_RUNTIME_EVENTS_EVIDENCE_FILE";

/// Appends one evidence marker line, or does nothing when
/// `MESH_LLM_RUNTIME_EVENTS_EVIDENCE_FILE` is unset. The actual file I/O is
/// `skippy_runtime::write_evidence_marker`, unit tested directly in
/// `crates/skippy-runtime/src/native_test_evidence.rs`.
fn write_marker(path: Option<&std::path::Path>, line: &str) {
    skippy_runtime::write_evidence_marker(path, line);
}

#[test]
fn runtime_events_native_gate() {
    // Resolve the evidence destination before loading native libraries. The
    // loader and model-open path are process-global; retaining this value also
    // makes the marker destination stable if native initialization mutates the
    // process environment.
    let evidence_path = env::var_os(EVIDENCE_FILE_ENV).map(PathBuf::from);

    if env::var(GATE_ENV).ok().as_deref() != Some("1") {
        println!("BLOCKED: {GATE_ENV} unset");
        write_marker(
            evidence_path.as_deref(),
            "blocked-when-ungated: gate unset, no native symbol was touched",
        );
        return;
    }

    #[cfg(not(feature = "dynamic-native-runtime"))]
    {
        println!("BLOCKED: dynamic-native-runtime feature not enabled");
        write_marker(
            evidence_path.as_deref(),
            "blocked: dynamic-native-runtime feature is not enabled for this run",
        );
        panic!("{GATE_ENV}=1 requires the dynamic-native-runtime feature");
    }

    #[cfg(feature = "dynamic-native-runtime")]
    {
        run_real_native_gate(evidence_path);
    }
}

#[cfg(feature = "dynamic-native-runtime")]
fn required_env(name: &str, evidence_path: &std::path::Path, purpose: &str) -> String {
    env::var(name).unwrap_or_else(|_| {
        println!("BLOCKED: {name} unset");
        write_marker(
            Some(evidence_path),
            &format!("blocked: {name} unset, required when {GATE_ENV}=1"),
        );
        panic!("{GATE_ENV}=1 requires {name} to {purpose}")
    })
}

#[cfg(feature = "dynamic-native-runtime")]
fn run_real_native_gate(evidence_path: Option<PathBuf>) {
    let evidence_path = evidence_path.unwrap_or_else(|| {
        panic!("{GATE_ENV}=1 requires {EVIDENCE_FILE_ENV} to name the evidence file")
    });
    let bundle_dir = required_env(
        BUNDLE_DIR_ENV,
        &evidence_path,
        "point at a dynamic native runtime",
    );
    let model_path = required_env(MODEL_ENV, &evidence_path, "name a readable model");

    let evidence = evidence::Evidence::new(evidence_path);
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        gated_steps(&PathBuf::from(bundle_dir), &model_path, &evidence);
    }));
    match outcome {
        Ok(()) => evidence.flush_executed(),
        Err(payload) => {
            // A failing step may leave the global reporter installed.
            skippy_runtime::clear_runtime_event_reporter();
            let reason = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("non-string panic");
            evidence.flush_failed(reason.lines().next().unwrap_or(reason));
            std::panic::resume_unwind(payload);
        }
    }
}

#[cfg(feature = "dynamic-native-runtime")]
fn load_runtime(bundle_dir: &std::path::Path) -> skippy_runtime::CapabilityReport {
    let libraries = libraries::discover_libraries(bundle_dir);
    assert!(
        !libraries.is_empty(),
        "no native runtime libraries found under {}",
        bundle_dir.display()
    );
    if !skippy_runtime::native_runtime_loaded() {
        unsafe { skippy_runtime::load_native_runtime_libraries(&libraries) }
            .expect("load native runtime libraries for the real ABI admission test");
    }
    skippy_runtime::probe_capabilities()
}

#[cfg(feature = "dynamic-native-runtime")]
fn gated_steps(bundle_dir: &std::path::Path, model_path: &str, evidence: &evidence::Evidence) {
    use families::{FamilyTally, drain_global};

    let report = load_runtime(bundle_dir);
    evidence.record(
        "exact-abi-admission: native runtime loaded (loader enforces exact major.minor.patch)",
    );
    evidence.record(format!(
        "capability-probe: confirmed={:#x} health_messages={}",
        report.confirmed,
        report.health_messages.len()
    ));

    // (e) This runtime must advertise per-call model-open events; the
    // no-event fallback is proven at the unit layer (see the marker).
    assert!(
        report.family_confirmed(skippy_ffi::FEATURE_RUNTIME_EVENTS),
        "capability probe must confirm FEATURE_RUNTIME_EVENTS on this runtime"
    );
    evidence.record(format!(
        "runtime-events-feature-bit: confirmed (FEATURE_RUNTIME_EVENTS={:#x})",
        skippy_ffi::FEATURE_RUNTIME_EVENTS
    ));
    evidence.record(
        "mixed-version-fallback: unit-covered by runtime_events::tests::\
         queue_supplied_but_events_unsupported_takes_legacy_path_and_queue_stays_empty and \
         assert_model_open_events_feature_missing_falls_back; not reachable in this process \
         because the loaded runtime advertises FEATURE_RUNTIME_EVENTS",
    );

    // Start from a clean ring: it is process-global and outlives any one
    // test, so a stale record would be miscounted as this run's evidence.
    let _ = drain_global();
    assert!(
        skippy_runtime::install_runtime_event_reporter(),
        "runtime event reporter must install when the explicit native gate is enabled"
    );
    evidence.record("reporter-install: true");

    let config = skippy_runtime::RuntimeConfig::default();
    let mut tally = FamilyTally::default();

    // (d) First, so the plain-open structured count reflects a fresh load.
    let structured =
        family_coverage::run_load_session_unload(model_path, &config, evidence, &mut tally);
    assert!(
        structured > 0,
        "successful model-open must produce at least one structured production callback; \
         old model-open progress alone is insufficient"
    );
    evidence.record(format!("structured-production-callbacks: {structured}"));

    // (a) + (b)
    model_open_checks::check_successful_opens(model_path, &config, evidence, &mut tally);
    // (c)
    model_open_checks::check_failed_opens(&config, evidence, &mut tally);

    let missing = family_coverage::judge_family_coverage(&report, &tally, evidence);
    evidence.record(format!(
        "global-reporter: dropped={} rejected={}",
        skippy_runtime::dropped_runtime_events(),
        skippy_runtime::rejected_runtime_events()
    ));
    skippy_runtime::clear_runtime_event_reporter();
    evidence.record("reporter-clear: returned");
    assert!(
        missing.is_empty(),
        "confirmed families never observed during a real load + session + unload: {missing:?}"
    );
}
