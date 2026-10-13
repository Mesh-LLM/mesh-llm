//! Execute the current action scalars under finite, local-only dependencies.
use super::support::{Fixture, action, step};
use serde_json::{Value, json};
use std::{fs, process::Command};

#[test]
fn all_successful_configure_routes_reset_once_and_reset_failures_fail_closed() {
    let depot = vec![
        ("INPUT_ALLOW_DEPOT_REMOTE_CACHE", "true"),
        ("SCCACHE_WEBDAV_ENDPOINT", "http://webdav.fixture"),
        ("DEPOT_CACHE_TOKEN", "fixture-depot-token"),
    ];
    let mut depot_fallback = depot.clone();
    depot_fallback.push(("START_CODES", "[1,0]"));
    let routes = [
        ("depot", depot, "disk,webdav"),
        ("depot fallback", depot_fallback, "disk"),
        (
            "provider denied",
            vec![("INPUT_ALLOW_NATIVE_GITHUB_CACHE", "false")],
            "disk",
        ),
        (
            "native disabled",
            vec![("SCCACHE_GHA_ENABLED", "false")],
            "disk",
        ),
        (
            "pull request",
            vec![("GITHUB_EVENT_NAME", "pull_request")],
            "disk",
        ),
        ("native remote", vec![], "disk,gha"),
        ("native fallback", vec![("START_CODES", "[1,0]")], "disk"),
    ];
    for (name, env, chain) in routes {
        for reset in ["0", "1"] {
            let mut env = env.clone();
            env.push(("RESET_CODE", reset));
            let result = Fixture::new().cache(&env);
            assert_eq!(
                result["exports"]["SCCACHE_MULTILEVEL_CHAIN"], chain,
                "{name}"
            );
            assert_eq!(
                result["failures"].as_array().unwrap().len(),
                usize::from(reset != "0"),
                "{name}: reset {reset}"
            );
            let calls = result["calls"].as_array().unwrap();
            let zeros: Vec<_> = calls
                .iter()
                .enumerate()
                .filter(|(_, c)| c["args"][0] == "--zero-stats")
                .collect();
            assert_eq!(zeros.len(), 1, "{name}: each successful route resets once");
            let (index, zero) = zeros[0];
            assert_eq!(calls[index - 1]["args"][0], "--start-server");
            assert_eq!(
                zero["env"],
                calls[index - 1]["env"],
                "{name}: reset must address the selected server environment"
            );
            if chain != "disk" {
                assert_eq!(
                    calls[index - 1]["env"]["SCCACHE_MULTILEVEL_WRITE_ERROR_POLICY"],
                    "all",
                    "{name}: synchronous multilevel policy must exist when the remote server starts"
                );
            }
        }
    }
}

#[test]
fn direct_and_dispatched_pr_target_contexts_keep_compiler_credentials_isolated() {
    for event in ["pull_request", "pull_request_target"] {
        for dispatched in [false, true] {
            let env = if dispatched {
                vec![
                    ("GITHUB_EVENT_NAME", "workflow_dispatch"),
                    ("DISPATCH_ORIGINAL_EVENT_NAME", event),
                ]
            } else {
                vec![("GITHUB_EVENT_NAME", event)]
            };
            let result = Fixture::new().cache(&env);
            assert_eq!(result["exports"]["SCCACHE_MULTILEVEL_CHAIN"], "disk");
            assert_eq!(result["exports"]["SCCACHE_GHA_RW_MODE"], "READ_ONLY");
            assert_eq!(result["failures"], json!([]));
            assert_eq!(
                result["job"]["ACTIONS_RUNTIME_TOKEN"],
                "fixture-runtime-token"
            );
            for call in result["calls"].as_array().unwrap().iter().filter(|c| {
                matches!(
                    c["args"][0].as_str(),
                    Some("--start-server" | "--zero-stats")
                )
            }) {
                for key in [
                    "ACTIONS_CACHE_URL",
                    "ACTIONS_RESULTS_URL",
                    "ACTIONS_RUNTIME_TOKEN",
                    "SCCACHE_WEBDAV_ENDPOINT",
                    "SCCACHE_WEBDAV_TOKEN",
                    "SCCACHE_WEBDAV_USERNAME",
                    "SCCACHE_WEBDAV_PASSWORD",
                ] {
                    assert_eq!(call["env"][key], "", "{event} {key}");
                }
            }
        }
    }
}
fn payload() -> Value {
    json!({"stats": {
        "compile_requests": 12, "requests_executed": 10, "compilations": 4, "cache_writes": 3,
        "cache_read_errors": 0, "cache_write_errors": 0,
        "cache_hits": {"counts": {"/fixture/private?token=synthetic-secret": 6}, "adv_counts": {"Rust": 600}},
        "cache_misses": {"counts": {"Rust": 4}}, "cache_errors": {"counts": {}}
    }, "cache_location": "https://synthetic-secret@cache.fixture/private", "version": "synthetic-secret"})
}
fn capture(fixture: &Fixture, artifact: &str, floor: &str) -> std::process::Output {
    fs::write(
        fixture.path().join("payload.json"),
        serde_json::to_vec(&payload()).unwrap(),
    )
    .unwrap();
    fixture.executable(
        "sccache",
        r#"
[[ "$#" == 3 && "$1" == --show-stats && "$2" == --stats-format && "$3" == json ]] || exit 72
if [[ -f error ]]; then echo 'synthetic-secret' >&2; exit 23; fi
/bin/cat payload.json
"#,
    );
    let mut command = Command::new("bash");
    command
        .current_dir(fixture.path())
        .args(["-c", &step(&action("capture-sccache-stats"), "run")]);
    command
        .env_clear()
        .env(
            "PATH",
            format!("{}:/usr/bin:/bin", fixture.path().join("bin").display()),
        )
        .env("MESH_LLM_AUTOMATION_BIN", env!("CARGO_BIN_EXE_xtask"))
        .env("SCCACHE_STATS_ARTIFACT_NAME", artifact)
        .env(
            "SCCACHE_STATS_OUTPUT_DIR",
            fixture.path().join("evidence with spaces"),
        )
        .env("SCCACHE_CACHE_EXPECTATION", "warm")
        .env("SCCACHE_MINIMUM_HIT_RATE", floor)
        .env("GITHUB_OUTPUT", fixture.path().join("outputs"));
    fixture.run(command)
}
#[test]
fn copied_capture_scalar_runs_real_counter_owner_then_preserves_threshold_failure_evidence() {
    for (floor, passed, classification) in [
        ("0.5", "true", "warm-pass"),
        ("0.8", "false", "warm-failure"),
    ] {
        let fixture = Fixture::new();
        let output = capture(&fixture, "sccache-fixture", floor);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let outputs = fixture.outputs();
        assert_eq!(outputs["cache_passed"], passed);
        assert_eq!(outputs["cache_classification"], classification);
        assert_eq!(outputs["cache_hits"], "6");
        let document = fs::read_to_string(outputs["stats_file"].as_str().unwrap()).unwrap();
        let evidence: Value = serde_json::from_str(&document).unwrap();
        assert_eq!(
            evidence["stats"]["cache_hits"],
            json!({"counts": {"total": 6}})
        );
        assert!(!format!("{document}{outputs}{output:?}").contains("synthetic-secret"));
        if passed == "false" {
            let action = action("capture-sccache-stats");
            let super::Node::Seq(steps) = action.get("runs").unwrap().get("steps").unwrap() else {
                panic!("steps");
            };
            let scalar = steps.last().unwrap().get("run").unwrap().text().unwrap();
            let mut command = Command::new("bash");
            command.args(["-c", scalar]);
            let enforced = fixture.run(command);
            assert_eq!(enforced.status.code(), Some(1));
            assert!(String::from_utf8_lossy(&enforced.stderr).contains("configured minimum"));
            assert!(std::path::Path::new(outputs["stats_file"].as_str().unwrap()).is_file());
        }
    }
}
#[test]
fn copied_capture_scalar_rejects_escaping_names_and_suppresses_tool_errors() {
    for fail_tool in [false, true] {
        let fixture = Fixture::new();
        if fail_tool {
            fs::write(fixture.path().join("error"), "fixture").unwrap();
        }
        let artifact = if fail_tool {
            "sccache-fixture"
        } else {
            "../escape"
        };
        let result = capture(&fixture, artifact, "0");
        assert!(!result.status.success());
        assert!(!format!("{result:?}").contains("synthetic-secret"));
        assert!(
            !fixture
                .path()
                .join("evidence with spaces")
                .join(artifact)
                .join("sccache-stats.json")
                .exists()
        );
        assert!(!fixture.path().join("outputs").exists());
    }
}
#[test]
fn copied_restore_scalar_enforces_finite_local_cache_capacity() {
    let fixture = Fixture::new();
    for allowed in ["true", "false", "truthy"] {
        let env_file = fixture.path().join("env");
        let _ = fs::remove_file(&env_file);
        let mut command = Command::new("bash");
        command
            .args(["-c", &step(&action("restore-sccache-seed"), "run")])
            .env("ALLOW_TRUSTED_SEED", allowed)
            .env("RUNNER_TEMP", fixture.path())
            .env("GITHUB_ENV", &env_file);
        let result = fixture.run(command);
        assert_eq!(result.status.success(), allowed != "truthy");
        if allowed == "truthy" {
            assert!(!env_file.exists());
        } else {
            let env = fs::read_to_string(env_file).unwrap();
            assert!(env.lines().any(|line| line == "SCCACHE_CACHE_SIZE=2G"));
            assert_eq!(env.contains("SCCACHE_GHA_ENABLED=false"), allowed == "true");
        }
    }
}
