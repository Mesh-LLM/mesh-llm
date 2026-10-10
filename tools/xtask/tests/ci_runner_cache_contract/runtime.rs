use super::support::Fixture;
use serde_json::{Value, json};

fn assert_runner(values: &Value, depot: bool, native: bool) {
    assert_eq!(values["depot_enabled"], depot.to_string());
    assert_eq!(values["allow_depot_remote_cache"], "false");
    assert_eq!(values["allow_native_github_cache"], native.to_string());
    assert_eq!(values["allow_trusted_sccache_seed"], (!depot).to_string());
    for (key, hosted, provider) in [
        ("runner", "ubuntu-24.04", "depot-ubuntu-24.04"),
        ("runner_arm", "ubuntu-24.04-arm", "depot-ubuntu-24.04-arm"),
        ("runner_macos", "macos-15", "depot-macos-15"),
        ("runner_windows", "windows-2022", "depot-windows-2022"),
    ] {
        assert_eq!(values[key], if depot { provider } else { hosted });
    }
    for size in ["4", "8", "16"] {
        for (key, label) in [
            ("runner", "ubuntu-24.04"),
            ("runner_arm", "ubuntu-24.04-arm"),
        ] {
            assert_eq!(
                values[format!("{key}_{size}")],
                if depot {
                    format!("depot-{label}-{size}")
                } else {
                    label.to_owned()
                }
            );
        }
    }
}

type RunnerCase = (Vec<(&'static str, &'static str)>, bool, bool);
fn direct_cases() -> Vec<RunnerCase> {
    vec![
        (vec![], false, true),
        (vec![("INPUT_DEPOT_PR_ENABLED", "true")], true, true),
        (
            vec![
                ("INPUT_DEPOT_PR_ENABLED", "true"),
                ("FIXTURE_DATE", "2026-09-14"),
            ],
            false,
            true,
        ),
        (
            vec![
                ("INPUT_PR_CANARY_REF", "refs/pull/12/merge"),
                ("FIXTURE_DATE", "2026-10-02"),
            ],
            true,
            false,
        ),
        (
            vec![
                ("INPUT_DEPOT_PR_ENABLED", "true"),
                ("INPUT_REPOSITORY", "attacker/mesh-llm"),
            ],
            false,
            true,
        ),
        (
            vec![
                ("INPUT_DEPOT_PR_ENABLED", "true"),
                ("INPUT_HEAD_REPOSITORY", "attacker/mesh-llm"),
            ],
            false,
            true,
        ),
        (
            vec![
                ("INPUT_DEPOT_PR_ENABLED", "true"),
                ("INPUT_FORCE_HOSTED", "true"),
            ],
            false,
            true,
        ),
        (
            vec![
                ("INPUT_DEPOT_PR_ENABLED", "true"),
                ("INPUT_REF", "refs/pull/12/head"),
            ],
            false,
            true,
        ),
        (
            vec![
                ("INPUT_EVENT_NAME", "pull_request_target"),
                ("INPUT_DEPOT_PR_ENABLED", "true"),
            ],
            false,
            true,
        ),
    ]
}
fn other_event_cases() -> Vec<RunnerCase> {
    vec![
        (
            vec![
                ("INPUT_EVENT_NAME", "push"),
                ("INPUT_REF", "refs/heads/main"),
                ("INPUT_DEPOT_MAIN_ENABLED", "true"),
                ("INPUT_DEPOT_PR_ENABLED", "true"),
            ],
            true,
            true,
        ),
        (
            vec![
                ("INPUT_EVENT_NAME", "push"),
                ("INPUT_REF", "refs/heads/main"),
                ("INPUT_DEPOT_MAIN_ENABLED", "true"),
                ("FIXTURE_DATE", "2026-10-02"),
            ],
            true,
            false,
        ),
        (
            vec![
                ("INPUT_EVENT_NAME", "push"),
                ("INPUT_REF", "refs/heads/feature"),
                ("INPUT_DEPOT_MAIN_ENABLED", "true"),
            ],
            false,
            true,
        ),
        (
            vec![
                ("INPUT_EVENT_NAME", "push"),
                ("INPUT_REF", "refs/tags/v1.2.3"),
                ("INPUT_DEPOT_MAIN_ENABLED", "true"),
            ],
            false,
            true,
        ),
        (
            vec![
                ("INPUT_EVENT_NAME", "schedule"),
                ("INPUT_REF", "refs/heads/main"),
                ("INPUT_DEPOT_MAIN_ENABLED", "true"),
            ],
            false,
            true,
        ),
        (
            vec![
                ("INPUT_EVENT_NAME", "workflow_dispatch"),
                ("INPUT_REF", "refs/heads/main"),
                ("INPUT_MANUAL_USE_DEPOT", "true"),
            ],
            true,
            false,
        ),
        (
            vec![
                ("INPUT_EVENT_NAME", "workflow_dispatch"),
                ("INPUT_REF", "refs/heads/main"),
                ("INPUT_DEPOT_MAIN_ENABLED", "true"),
                ("INPUT_ORIGINAL_EVENT_NAME", "pull_request"),
            ],
            false,
            true,
        ),
        (
            vec![
                ("INPUT_EVENT_NAME", "workflow_dispatch"),
                ("INPUT_REF", "refs/heads/main"),
                ("INPUT_DEPOT_MAIN_ENABLED", "true"),
                ("DISPATCH_ORIGINAL_EVENT_NAME", "pull_request_target"),
            ],
            false,
            true,
        ),
    ]
}
#[test]
fn exact_runner_identity_events_deadline_and_forced_hosted_preserve_cache_boundaries() {
    let cases = direct_cases().into_iter().chain(other_event_cases());
    for (env, depot, native) in cases {
        let f = Fixture::new();
        let out = f.selector(&env);
        assert!(
            out.status.success(),
            "{env:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert_runner(&f.outputs(), depot, native);
    }
}

#[test]
fn invalid_canary_selector_cannot_emit_runner_outputs() {
    for reference in ["refs/heads/main", "refs/pull/x/merge", "refs/pull/12/head"] {
        let f = Fixture::new();
        let out = f.selector(&[("INPUT_PR_CANARY_REF", reference)]);
        assert!(!out.status.success());
        assert!(!f.path().join("outputs").exists());
    }
}

fn assert_disk_isolated(value: &Value) {
    assert_eq!(value["exports"]["SCCACHE_MULTILEVEL_CHAIN"], "disk");
    let calls = value["calls"].as_array().unwrap();
    for call in calls
        .iter()
        .filter(|c| c["args"][0] == "--start-server" || c["args"][0] == "--zero-stats")
    {
        for key in [
            "ACTIONS_CACHE_URL",
            "ACTIONS_RESULTS_URL",
            "ACTIONS_RUNTIME_TOKEN",
            "SCCACHE_WEBDAV_ENDPOINT",
            "SCCACHE_WEBDAV_TOKEN",
            "SCCACHE_WEBDAV_USERNAME",
            "SCCACHE_WEBDAV_PASSWORD",
        ] {
            assert_eq!(call["env"][key], "", "{key}: {call}");
        }
        assert_eq!(
            call["env"]["SCCACHE_DIR"],
            "/fixture/runner temp/mesh-llm-sccache"
        );
    }
    assert_eq!(
        value["job"]["ACTIONS_RUNTIME_TOKEN"],
        "fixture-runtime-token"
    );
}

#[test]
fn denied_provider_and_dispatched_pr_use_disk_without_clearing_later_action_credentials() {
    for env in [
        vec![("INPUT_ALLOW_NATIVE_GITHUB_CACHE", "false")],
        vec![
            ("SCCACHE_WEBDAV_ENDPOINT", "http://webdav.fixture"),
            ("DEPOT_CACHE_TOKEN", "fixture-depot-token"),
        ],
        vec![("GITHUB_EVENT_NAME", "pull_request")],
        vec![
            ("GITHUB_EVENT_NAME", "workflow_dispatch"),
            ("DISPATCH_ORIGINAL_EVENT_NAME", "pull_request"),
        ],
        vec![("SCCACHE_GHA_ENABLED", "false")],
    ] {
        let v = Fixture::new().cache(&env);
        assert_disk_isolated(&v);
        assert_eq!(v["failures"], json!([]));
    }
}

#[test]
fn allowed_webdav_masks_token_and_native_remote_failures_fall_back_to_isolated_disk() {
    let v = Fixture::new().cache(&[
        ("INPUT_ALLOW_DEPOT_REMOTE_CACHE", "true"),
        ("SCCACHE_WEBDAV_ENDPOINT", "http://webdav.fixture"),
        ("DEPOT_CACHE_TOKEN", "fixture-depot-token"),
    ]);
    assert_eq!(v["exports"]["SCCACHE_MULTILEVEL_CHAIN"], "disk,webdav");
    assert_eq!(v["exports"]["SCCACHE_MULTILEVEL_WRITE_ERROR_POLICY"], "all");
    assert_eq!(v["secrets"], json!(["fixture-depot-token"]));
    assert_eq!(v["failures"], json!([]));
    for webdav in [false, true] {
        let mut env = vec![("START_CODES", "[1,0]")];
        if webdav {
            env.extend([
                ("INPUT_ALLOW_DEPOT_REMOTE_CACHE", "true"),
                ("SCCACHE_WEBDAV_ENDPOINT", "http://webdav.fixture"),
                ("DEPOT_CACHE_TOKEN", "fixture-depot-token"),
            ]);
        }
        let v = Fixture::new().cache(&env);
        assert_eq!(v["exports"]["SCCACHE_MULTILEVEL_CHAIN"], "disk");
        let calls = v["calls"].as_array().unwrap();
        let fallback = calls
            .iter()
            .filter(|c| c["args"][0] == "--start-server")
            .nth(1)
            .unwrap();
        assert_eq!(fallback["env"]["ACTIONS_RUNTIME_TOKEN"], "");
        assert_eq!(fallback["env"]["SCCACHE_WEBDAV_TOKEN"], "");
        assert_eq!(v["failures"], json!([]));
    }
}

#[test]
fn invalid_flags_and_failed_local_start_or_stats_fail_the_action() {
    for env in [
        vec![("INPUT_ALLOW_DEPOT_REMOTE_CACHE", "maybe")],
        vec![("INPUT_ALLOW_NATIVE_GITHUB_CACHE", "")],
        vec![
            ("INPUT_ALLOW_NATIVE_GITHUB_CACHE", "false"),
            ("START_CODES", "[1]"),
        ],
        vec![
            ("INPUT_ALLOW_NATIVE_GITHUB_CACHE", "false"),
            ("RESET_CODE", "1"),
        ],
    ] {
        let v = Fixture::new().cache(&env);
        assert_eq!(v["failures"].as_array().unwrap().len(), 1);
    }
    let v = Fixture::new().cache(&[]);
    assert_eq!(v["exports"]["SCCACHE_MULTILEVEL_CHAIN"], "disk,gha");
    assert_eq!(v["exports"]["SCCACHE_MULTILEVEL_WRITE_ERROR_POLICY"], "all");
    assert_eq!(v["failures"], json!([]));
}

// Append to existing ci_runner_cache_contract/runtime.rs.
#[test]
fn exact_sentinel_ref_does_not_authorize_other_prs_forks_dispatch_or_forced_hosted() {
    for (extra, accepted) in [
        (vec![], true),
        (vec![("INPUT_PR_CANARY_REF", "")], false),
        (vec![("INPUT_PR_CANARY_REF", "refs/pull/13/merge")], false),
        (vec![("INPUT_HEAD_REPOSITORY", "attacker/mesh-llm")], false),
        (vec![("INPUT_FORCE_HOSTED", "true")], false),
        (vec![("INPUT_EVENT_NAME", "pull_request_target")], false),
        (
            vec![
                ("INPUT_EVENT_NAME", "workflow_dispatch"),
                ("INPUT_REF", "refs/heads/main"),
            ],
            false,
        ),
        (vec![("INPUT_REF", "refs/pull/12/head")], false),
    ] {
        let fixture = Fixture::new();
        let mut fields = vec![
            ("INPUT_PR_CANARY_REF", "refs/pull/12/merge"),
            ("INPUT_DEPOT_PR_ENABLED", "false"),
            ("FIXTURE_DATE", "2026-10-02"),
        ];
        fields.extend(extra);
        let output = fixture.selector(&fields);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let values = fixture.outputs();
        assert_eq!(values["depot_enabled"], accepted.to_string());
        assert_eq!(
            values["runner"],
            if accepted {
                "depot-ubuntu-24.04"
            } else {
                "ubuntu-24.04"
            }
        );
        fixture.0.close().unwrap();
    }
}
