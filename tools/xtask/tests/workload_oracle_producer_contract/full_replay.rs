//! Actual build adapter replay selection through inert private component observers.
#[path = "full_replay/fixture.rs"]
mod fixture;
use crate::process::{
    self, Cancellation, Completion, Limits, Outcome, ProcessSpec, RawCaptureOptions, Readiness,
    Value,
};
use fixture::{Fixture, write_tool};
use std::{collections::BTreeMap, ffi::OsString, fs, num::NonZeroUsize, time::Duration};

struct Call {
    tool: String,
    args: Vec<String>,
}
struct Replay {
    fixture: Fixture,
    trace: std::path::PathBuf,
}
impl Replay {
    fn new() -> Self {
        let fixture = Fixture::new(false);
        let trace = fixture.directory.path().join("replay-trace");
        write_tool(
            &fixture.tools.join("cmake"),
            include_str!("full_replay_cmake.sh"),
        );
        write_tool(&fixture.tools.join("git"), "#!/bin/sh\nexit 0\n");
        for name in ["ctest", "fixture-generator"] {
            let label = if name == "ctest" { "ctest" } else { "fixture" };
            write_tool(
                &fixture.tools.join(name),
                &format!(
                    "#!/bin/bash\nset -euo pipefail\nprintf '%s\\0' {label} \"$@\" '' >> \"${{FULL_REPLAY_TRACE:?}}\"\n"
                ),
            );
        }
        Self { fixture, trace }
    }
    fn run(&self, replay: bool, upstream: bool, force: bool, backend: &str) {
        let command = self.fixture.command(if backend == "metal" {
            "Darwin"
        } else {
            "Linux"
        });
        let mut environment: BTreeMap<OsString, Value> = command
            .get_envs()
            .filter_map(|(key, value)| {
                value.map(|value| (key.to_owned(), Value::Public(value.to_owned())))
            })
            .collect();
        for (key, value) in [
            ("LLAMA_STAGE_LINK_MODE", "static".into()),
            ("LLAMA_STAGE_BACKEND", backend.into()),
            (
                "LLAMA_STAGE_FULL_REPLAY",
                if replay { "ON" } else { "OFF" }.into(),
            ),
            (
                "LLAMA_STAGE_UPSTREAM_TESTS",
                if upstream { "ON" } else { "OFF" }.into(),
            ),
            (
                "LLAMA_STAGE_FORCE_BUILD",
                if force { "1" } else { "0" }.into(),
            ),
            ("FULL_REPLAY_TRACE", self.trace.as_os_str().to_owned()),
            (
                "FULL_REPLAY_BUILD",
                self.fixture.build.as_os_str().to_owned(),
            ),
            (
                "FULL_REPLAY_GENERATOR",
                self.fixture
                    .tools
                    .join("fixture-generator")
                    .into_os_string(),
            ),
            ("HOME", self.fixture.directory.path().into()),
        ] {
            environment.insert(key.into(), Value::Public(value));
        }
        let result = process::supervise_raw(
            &ProcessSpec {
                executable: command.get_program().into(),
                arguments: command
                    .get_args()
                    .map(|arg| Value::Public(arg.to_owned()))
                    .collect(),
                cwd: command.get_current_dir().unwrap().into(),
                environment,
            },
            &Limits {
                execution: Duration::from_secs(10),
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(1),
                retained_bytes_per_stream: 65536,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            &Cancellation::default(),
            RawCaptureOptions {
                stdout: NonZeroUsize::new(65536),
                stderr: NonZeroUsize::new(65536),
            },
        )
        .unwrap();
        assert!(
            result.process.success() && result.process.failure.is_none(),
            "{result:?}"
        );
        assert_eq!(result.process.outcome, Outcome::Exited);
        assert_eq!(result.process.status.unwrap().code(), Some(0));
        assert!(result.process.cleanup.complete);
        assert!(!result.process.cleanup.forced && !result.process.cleanup.graceful_signal_failed);
        assert!(result.process.cleanup.failure.is_none());
        let raw_stdout = result
            .stdout
            .as_ref()
            .expect("requested stdout capture absent");
        let raw_stderr = result
            .stderr
            .as_ref()
            .expect("requested stderr capture absent");
        assert_eq!(
            u64::try_from(raw_stdout.as_bytes().len()).unwrap(),
            result.process.stdout.bytes_seen
        );
        assert_eq!(
            u64::try_from(raw_stderr.as_bytes().len()).unwrap(),
            result.process.stderr.bytes_seen
        );
        for stream in [&result.process.stdout, &result.process.stderr] {
            assert!(!stream.truncated && stream.suppressed_lines == 0);
        }
    }
    fn calls(&self) -> Vec<Call> {
        let bytes = fs::read(&self.trace).unwrap();
        assert_eq!(bytes.last(), Some(&0));
        let mut record = Vec::new();
        let mut calls = Vec::new();
        for word in bytes.split(|byte| *byte == 0) {
            if word.is_empty() {
                if !record.is_empty() {
                    let tool = record.remove(0);
                    calls.push(Call {
                        tool,
                        args: std::mem::take(&mut record),
                    });
                }
            } else {
                record.push(String::from_utf8(word.to_vec()).unwrap());
            }
        }
        assert!(record.is_empty());
        calls
    }
    fn finish(self) {
        self.fixture
            .directory
            .close()
            .expect("owned replay fixture cleanup failed");
    }
}
fn first<'a>(calls: &'a [Call], tool: &str, build: bool) -> &'a [String] {
    &calls
        .iter()
        .find(|call| {
            call.tool == tool
                && (tool != "cmake"
                    || (call.args.first().is_some_and(|arg| arg == "--build") == build))
        })
        .unwrap()
        .args
}
fn contains(args: &[String], value: &str) -> bool {
    args.iter().any(|arg| arg == value)
}
fn option<'a>(args: &'a [String], key: &str) -> &'a str {
    &args[args.iter().position(|arg| arg == key).unwrap() + 1]
}
#[test]
fn replay_default_product_disables_standard_private_and_nonchat_gates() {
    let replay = Replay::new();
    replay.run(false, false, true, "cpu");
    let calls = replay.calls();
    let configure = first(&calls, "cmake", false);
    let build = first(&calls, "cmake", true);
    for flag in [
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_STAGE_BUILD_TESTS=OFF",
        "-DGGML_METAL=OFF",
    ] {
        assert!(contains(configure, flag));
    }
    assert!(
        !build
            .iter()
            .any(|arg| arg.starts_with("skippy-") || arg.starts_with("test-skippy-"))
    );
    assert!(!calls.iter().any(|call| call.tool == "ctest"));
    replay.finish();
}
#[test]
fn replay_explicit_builds_retained_skippy_gates_and_runs_bounded_filtered_ctest() {
    let replay = Replay::new();
    replay.run(true, false, true, "cpu");
    let calls = replay.calls();
    let configure = first(&calls, "cmake", false);
    let build = first(&calls, "cmake", true);
    let ctest = first(&calls, "ctest", false);
    for flag in [
        "-DLLAMA_BUILD_TESTS=ON",
        "-DLLAMA_STAGE_BUILD_TESTS=ON",
        "-DLLAMA_BUILD_SERVER=OFF",
    ] {
        assert!(contains(configure, flag));
    }
    for target in [
        "skippy-graph-build-inputs",
        "skippy-hardware-application-probe",
        "skippy-model-fixture-generator",
        "skippy-model-loader-accounting",
        "skippy-runtime-events-test",
        "skippy-noalloc-graph-planning",
        "skippy-renamed-multishard-planning",
        "skippy-stage-slice-plan",
        "test-skippy-kv-cells-contiguous",
        "test-skippy-kv-page-export",
        "test-skippy-model-loader-accounting",
        "test-skippy-recurrent-state-roundtrip",
        "test-skippy-rerank-template",
        "test-skippy-sampling-suppress",
        "test-skippy-verify-checkpoint-retirement",
    ] {
        assert!(contains(build, target), "{target}");
    }
    assert!(
        !contains(build, "test-skippy-activation-layout") && !contains(build, "test-llama-archs")
    );
    assert_eq!(option(ctest, "--timeout"), "900");
    assert_eq!(option(ctest, "--fixture-exclude-any"), "^generate-models$");
    assert_eq!(option(ctest, "-R"), "^(skippy_|test-skippy-)");
    let fixtures: Vec<_> = calls
        .iter()
        .filter(|call| call.tool == "fixture")
        .map(|call| call.args[..4].to_vec())
        .collect();
    assert_eq!(
        fixtures,
        vec![
            vec!["--arch", "gemma", "--seed", "1"],
            vec!["--arch", "qwen2moe", "--seed", "1"]
        ]
    );
    replay.finish();
}
#[test]
fn replay_cached_full_replay_rebuilds_and_reexecutes_gates() {
    let replay = Replay::new();
    replay.run(true, false, true, "cpu");
    replay.run(true, false, false, "cpu");
    let calls = replay.calls();
    assert_eq!(
        calls
            .iter()
            .filter(|c| c.tool == "cmake" && c.args[0] == "--build")
            .count(),
        2
    );
    assert_eq!(calls.iter().filter(|c| c.tool == "ctest").count(), 2);
    replay.finish();
}
#[test]
fn replay_metal_selects_cachegen_gate_without_qualifying_device_execution() {
    let replay = Replay::new();
    replay.run(true, false, true, "metal");
    let calls = replay.calls();
    assert!(contains(
        first(&calls, "cmake", true),
        "test-skippy-cachegen-metal"
    ));
    replay.finish();
}
#[test]
fn replay_cached_standard_product_keeps_build_shortcut_and_no_ctest() {
    let replay = Replay::new();
    replay.run(false, false, true, "cpu");
    replay.run(false, false, false, "cpu");
    let calls = replay.calls();
    assert_eq!(
        calls
            .iter()
            .filter(|c| c.tool == "cmake" && c.args[0] == "--build")
            .count(),
        1
    );
    assert!(!calls.iter().any(|c| c.tool == "ctest"));
    replay.finish();
}
#[test]
fn replay_explicit_upstream_suite_builds_defaults_and_runs_unfiltered_bounded_ctest() {
    let replay = Replay::new();
    replay.run(false, true, true, "cpu");
    let calls = replay.calls();
    let configure = first(&calls, "cmake", false);
    let build = first(&calls, "cmake", true);
    let ctest = first(&calls, "ctest", false);
    for flag in [
        "-DLLAMA_BUILD_TESTS=ON",
        "-DLLAMA_STAGE_BUILD_TESTS=ON",
        "-DLLAMA_BUILD_SERVER=ON",
    ] {
        assert!(contains(configure, flag));
    }
    assert!(!contains(build, "--target") && !contains(ctest, "-R"));
    assert!(contains(ctest, "--output-on-failure"));
    assert_eq!(option(ctest, "--timeout"), "900");
    replay.finish();
}
#[test]
fn replay_just_recipe_keeps_pinned_prepare_and_isolated_forced_static_replay() {
    let source = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../just/skippy.just"
    ));
    let body = source
        .split_once("skippy-native-full-replay backend=\"cpu\":")
        .unwrap()
        .1
        .split_once("\n\n")
        .unwrap()
        .0;
    for required in [
        "scripts/prepare-llama.sh pinned",
        "build-stage-full-replay-static-{{ backend }}",
        "LLAMA_STAGE_LINK_MODE=static",
        "LLAMA_STAGE_FULL_REPLAY=ON",
        "LLAMA_STAGE_FORCE_BUILD=1",
    ] {
        assert!(body.contains(required), "{required}");
    }
}
