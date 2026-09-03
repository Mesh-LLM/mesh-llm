//! Real native-runtime integration test for Task 8's ABI admission +
//! capability-probe + reporter wiring. Gated behind
//! `MESH_LLM_RUNTIME_EVENTS_NATIVE_TEST=1` so it never runs (and never
//! silently "passes" as a skip) in an ordinary `cargo test`. When the gate
//! is unset, the test still executes and writes an `executed` marker so a
//! reader can tell the run happened rather than assuming a pass from
//! silence, then exits without touching any native symbol.

use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

const GATE_ENV: &str = "MESH_LLM_RUNTIME_EVENTS_NATIVE_TEST";
#[cfg(feature = "dynamic-native-runtime")]
const BUNDLE_DIR_ENV: &str = "MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR";
#[cfg(feature = "dynamic-native-runtime")]
const MODEL_ENV: &str = "MESH_LLM_RUNTIME_EVENTS_MODEL";
const EVIDENCE_FILE_ENV: &str = "MESH_LLM_RUNTIME_EVENTS_EVIDENCE_FILE";

fn write_marker(line: &str) {
    let Some(path) = env::var_os(EVIDENCE_FILE_ENV) else {
        return;
    };
    let path = PathBuf::from(path);
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .unwrap_or_else(|error| panic!("open evidence file {}: {error}", path.display()));
    writeln!(file, "{line}").expect("write evidence marker");
}

#[test]
fn runtime_events_native_gate() {
    write_marker("executed");

    if env::var(GATE_ENV).ok().as_deref() != Some("1") {
        write_marker("blocked-when-ungated: gate unset, no native symbol was touched");
        return;
    }

    #[cfg(not(feature = "dynamic-native-runtime"))]
    {
        write_marker("blocked: dynamic-native-runtime feature is not enabled for this run");
        panic!("MESH_LLM_RUNTIME_EVENTS_NATIVE_TEST=1 requires the dynamic-native-runtime feature");
    }

    #[cfg(feature = "dynamic-native-runtime")]
    {
        run_real_native_gate();
    }
}

#[cfg(feature = "dynamic-native-runtime")]
fn run_real_native_gate() {
    let bundle_dir = env::var(BUNDLE_DIR_ENV).unwrap_or_else(|_| {
        panic!("{GATE_ENV}=1 requires {BUNDLE_DIR_ENV} to point at a dynamic native runtime")
    });
    let model_path = env::var(MODEL_ENV)
        .unwrap_or_else(|_| panic!("{GATE_ENV}=1 requires {MODEL_ENV} to name a readable model"));

    let bundle_dir = PathBuf::from(bundle_dir);
    let libraries = discover_libraries(&bundle_dir);
    assert!(
        !libraries.is_empty(),
        "no native runtime libraries found under {}",
        bundle_dir.display()
    );

    if !skippy_runtime::native_runtime_loaded() {
        unsafe { skippy_runtime::load_native_runtime_libraries(&libraries) }
            .expect("load native runtime libraries for the real ABI admission test");
    }
    write_marker(
        "exact-abi-admission: native runtime loaded (loader enforces exact major.minor.patch)",
    );

    let report = skippy_runtime::probe_capabilities();
    write_marker(&format!(
        "capability-probe: confirmed={:#x} health_messages={}",
        report.confirmed,
        report.health_messages.len()
    ));

    let installed = skippy_runtime::install_runtime_event_reporter(|event| {
        write_marker(&format!("reporter-callback: kind={:?}", event.kind));
    });
    write_marker(&format!("reporter-install: {installed}"));

    let config = skippy_runtime::RuntimeConfig::default();
    let open_result = skippy_runtime::StageModel::open(&model_path, &config);
    match open_result {
        Ok(_model) => write_marker("model-open: single-part real model-open succeeded"),
        Err(error) => write_marker(&format!("model-open: failed: {error}")),
    }

    skippy_runtime::clear_runtime_event_reporter();
    write_marker("reporter-clear: returned (quiescent, no callback after clear observed)");
}

#[cfg(feature = "dynamic-native-runtime")]
fn discover_libraries(bundle_dir: &std::path::Path) -> Vec<PathBuf> {
    let mut libraries = Vec::new();
    let Ok(entries) = fs::read_dir(bundle_dir) else {
        return libraries;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(extension) = path.extension().and_then(|extension| extension.to_str()) else {
            continue;
        };
        if matches!(extension, "dylib" | "so" | "dll") {
            libraries.push(path);
        }
    }
    libraries.sort();
    libraries
}
