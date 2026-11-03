//! Drive the real planner's placement signal into the real selector action.
use super::support;
use serde_json::{Value, json};
use std::io::Write;
use std::process::{Command, Stdio};

fn plan(profile: &str, event: &str, base: &str, path: &str) -> Value {
    let request = json!({
        "profile": profile,
        "event_name": event,
        "source_sha": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "base_sha": base,
        "changed_files": [path],
        "affected_crates": ["mesh-llm"],
        "workspace_packages": [{"name": "mesh-llm", "path": "mesh/crates/mesh-llm"}]
    });
    let mut child = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .current_dir(support::root())
        .args(["ci", "plan"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(request.to_string().as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

fn nested_sdk_forwards_placement(workflow: &super::Node) -> bool {
    let nested = super::job(workflow, "produce_linux_static_abi");
    [
        "force_hosted",
        "use_depot",
        "original_event_name",
        "source_sha",
    ]
    .into_iter()
    .all(|key| super::input(nested, key) == Some(format!("${{{{ inputs.{key} }}}}").as_str()))
}

fn release_sdk_forces_hosted(workflow: &super::Node) -> bool {
    super::input(
        super::job(workflow, "build_native_sdk_runtime"),
        "force_hosted",
    ) == Some("true")
}

fn assert_wiring_mutation_rejected(
    file: &str,
    original: &str,
    replacement: &str,
    check: fn(&super::Node) -> bool,
) {
    let source = super::source(file);
    assert_eq!(source.matches(original).count(), 1, "{file}");
    let changed = source.replacen(original, replacement, 1);
    assert!(!check(&super::workflow_yaml::parse(&changed).unwrap()));
}

#[test]
fn pr_and_dispatch_decisions_follow_planned_work_and_checked_in_linux_callers() {
    let lane = super::document("ci-linux-lane.yml");
    let static_abi = super::job(&lane, "static_abi");
    let nested_sdk = super::job(&lane, "kotlin_sdk_input");
    for caller in [static_abi, nested_sdk] {
        assert_eq!(
            super::input(caller, "force_hosted"),
            Some("${{ toJSON(fromJson(inputs.lane_plan_json).signals.force_hosted) != 'false' }}")
        );
        assert_eq!(
            super::input(caller, "use_depot"),
            Some("${{ inputs.use_depot }}")
        );
        assert_eq!(
            super::input(caller, "source_sha"),
            Some("${{ inputs.source_sha }}")
        );
    }
    assert_eq!(
        super::input(static_abi, "artifact_name"),
        Some("ci-static-abi-${{ github.run_id }}")
    );
    assert_eq!(
        super::input(nested_sdk, "static_abi_artifact_name"),
        Some("ci-static-abi-${{ github.run_id }}")
    );
    for file in ["static-abi-artifact.yml", "native-sdk-artifact.yml"] {
        let workflow = super::document(file);
        let selectors = super::steps(super::job(&workflow, "runner_policy"))
            .iter()
            .filter(|step| super::text(step, "uses") == Some("./.github/actions/select-ci-runners"))
            .collect::<Vec<_>>();
        assert_eq!(selectors.len(), 1, "{file}");
        for (key, expected) in [
            ("force_hosted", "${{ inputs.force_hosted }}"),
            ("manual_use_depot", "${{ inputs.use_depot }}"),
            ("original_event_name", "${{ inputs.original_event_name }}"),
            (
                "depot_main_enabled",
                "${{ vars.DEPOT_RUNNERS_ENABLED == 'true' }}",
            ),
            (
                "depot_pr_enabled",
                "${{ vars.DEPOT_PR_RUNNERS_ENABLED == 'true' }}",
            ),
        ] {
            assert_eq!(
                super::input(selectors[0], key),
                Some(expected),
                "{file}/{key}"
            );
        }
    }
    assert!(nested_sdk_forwards_placement(&super::document(
        "native-sdk-artifact.yml"
    )));

    let base = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    let product = plan(
        "pr-ready",
        "pull_request",
        base,
        "mesh/crates/mesh-llm/src/main.rs",
    );
    assert_eq!(product["signals"]["force_hosted"], false);
    for required in [
        "static-abi",
        "rust-tests",
        "runtime-product",
        "product-smoke",
    ] {
        assert!(
            product["required_slices"]
                .as_array()
                .unwrap()
                .iter()
                .any(|value| value == required),
            "{required}"
        );
    }
    assert!(
        product["matrices"]["runtime_products"]
            .as_array()
            .unwrap()
            .iter()
            .any(|row| row["id"] == "linux-cpu")
    );
    assert!(
        product["matrices"]["rust_tests"]
            .as_array()
            .unwrap()
            .iter()
            .any(|row| row["crates"]
                .as_array()
                .unwrap()
                .iter()
                .any(|name| name == "mesh-llm"))
    );
    let control = plan(
        "pr-ready",
        "pull_request",
        base,
        ".github/workflows/main_linux.yml",
    );
    assert_eq!(control["signals"]["force_hosted"], true);

    // Fixed dates exercise the expired exception boundary without renewing it.
    let cases = [
        (
            "same-repo active",
            "pull_request",
            "",
            "Mesh-LLM/mesh-llm",
            "refs/pull/12/merge",
            "2026-09-13",
            "true",
            "false",
            false,
            true,
            true,
            false,
        ),
        (
            "expired",
            "pull_request",
            "",
            "Mesh-LLM/mesh-llm",
            "refs/pull/12/merge",
            "2026-09-14",
            "true",
            "false",
            false,
            false,
            true,
            true,
        ),
        (
            "fork",
            "pull_request",
            "",
            "other/mesh-llm",
            "refs/pull/12/merge",
            "2026-09-13",
            "true",
            "false",
            false,
            false,
            true,
            true,
        ),
        (
            "target event",
            "pull_request_target",
            "",
            "Mesh-LLM/mesh-llm",
            "refs/pull/12/merge",
            "2026-09-13",
            "true",
            "false",
            false,
            false,
            true,
            true,
        ),
        (
            "PR dispatch",
            "workflow_dispatch",
            "pull_request",
            "Mesh-LLM/mesh-llm",
            "refs/heads/main",
            "2026-09-13",
            "true",
            "true",
            false,
            false,
            true,
            true,
        ),
        (
            "policy override",
            "workflow_dispatch",
            "workflow_dispatch",
            "Mesh-LLM/mesh-llm",
            "refs/heads/main",
            "2026-09-13",
            "true",
            "true",
            true,
            false,
            true,
            true,
        ),
    ];
    for (
        name,
        event,
        original,
        head_repo,
        reference,
        date,
        pr_gate,
        manual,
        policy_change,
        depot,
        native_cache,
        seed,
    ) in cases
    {
        let plan = if policy_change { &control } else { &product };
        let force = plan["signals"]["force_hosted"].as_bool().unwrap();
        let fixture = support::Fixture::new();
        let output = fixture.selector(&[
            ("INPUT_EVENT_NAME", event),
            ("INPUT_ORIGINAL_EVENT_NAME", original),
            ("INPUT_HEAD_REPOSITORY", head_repo),
            ("INPUT_REF", reference),
            ("FIXTURE_DATE", date),
            ("INPUT_DEPOT_PR_ENABLED", pr_gate),
            ("INPUT_DEPOT_MAIN_ENABLED", "true"),
            ("INPUT_MANUAL_USE_DEPOT", manual),
            ("INPUT_FORCE_HOSTED", if force { "true" } else { "false" }),
        ]);
        assert!(output.status.success(), "{name}: {output:?}");
        let selected = fixture.outputs();
        assert_eq!(selected["depot_enabled"], depot.to_string(), "{name}");
        assert_eq!(selected["allow_depot_remote_cache"], "false", "{name}");
        assert_eq!(
            selected["allow_native_github_cache"],
            native_cache.to_string(),
            "{name}"
        );
        assert_eq!(
            selected["allow_trusted_sccache_seed"],
            seed.to_string(),
            "{name}"
        );
        let cache = fixture.cache(&[
            ("GITHUB_EVENT_NAME", event),
            ("DISPATCH_ORIGINAL_EVENT_NAME", original),
            (
                "INPUT_ALLOW_DEPOT_REMOTE_CACHE",
                selected["allow_depot_remote_cache"].as_str().unwrap(),
            ),
            (
                "INPUT_ALLOW_NATIVE_GITHUB_CACHE",
                selected["allow_native_github_cache"].as_str().unwrap(),
            ),
        ]);
        assert_eq!(cache["failures"], json!([]), "{name}");
        let expected_chain = if name == "policy override" {
            "disk,gha"
        } else {
            "disk"
        };
        assert_eq!(
            cache["exports"]["SCCACHE_MULTILEVEL_CHAIN"], expected_chain,
            "{name}"
        );
        fixture.0.close().unwrap();
    }

    let fixture = support::Fixture::new();
    let malformed = fixture.selector(&[("INPUT_DEPOT_PR_ENABLED", "not-a-boolean")]);
    assert!(!malformed.status.success());
    assert!(!fixture.path().join("outputs").exists());
}

#[test]
fn placement_guards_reject_nested_and_release_override_mutations() {
    let sdk = super::document("native-sdk-artifact.yml");
    assert!(nested_sdk_forwards_placement(&sdk));
    let static_abi = super::document("static-abi-artifact.yml");
    assert_eq!(
        static_abi
            .get("on")
            .and_then(|on| on.get("workflow_call"))
            .and_then(|call| call.get("inputs"))
            .and_then(|inputs| inputs.get("force_hosted"))
            .and_then(|input| super::text(input, "default")),
        Some("false")
    );
    assert_wiring_mutation_rejected(
        "native-sdk-artifact.yml",
        "      original_event_name: ${{ inputs.original_event_name }}\n      force_hosted: ${{ inputs.force_hosted }}\n      backend: ${{ inputs.backend }}",
        "      original_event_name: ${{ inputs.original_event_name }}\n      backend: ${{ inputs.backend }}",
        nested_sdk_forwards_placement,
    );

    let release = super::document("release.yml");
    assert!(release_sdk_forces_hosted(&release));
    assert_wiring_mutation_rejected(
        "release.yml",
        "      original_event_name: ${{ github.event_name }}\n      force_hosted: true\n      use_depot: ${{ inputs.use_depot == true }}",
        "      original_event_name: ${{ github.event_name }}\n      force_hosted: false\n      use_depot: ${{ inputs.use_depot == true }}",
        release_sdk_forces_hosted,
    );

    // Exercise the real selector with the correctly forwarded override and
    // with the false value caused by each synthetic wiring mutation.
    for (force_hosted, expected_depot) in [("true", "false"), ("false", "true")] {
        let fixture = support::Fixture::new();
        let result = fixture.selector(&[
            ("INPUT_EVENT_NAME", "workflow_dispatch"),
            ("INPUT_REF", "refs/heads/main"),
            ("INPUT_DEPOT_MAIN_ENABLED", "true"),
            ("INPUT_MANUAL_USE_DEPOT", "true"),
            ("INPUT_FORCE_HOSTED", force_hosted),
        ]);
        assert!(result.status.success(), "{result:?}");
        assert_eq!(fixture.outputs()["depot_enabled"], expected_depot);
        fixture.0.close().unwrap();
    }
}

mod caller_census;
mod canary_chain;
mod entrypoint_planner;
mod entrypoint_required;
mod event_matrix;
mod excluded_main;
mod host_roles;
mod platform_checks;
mod platform_runtimes;
mod product_roles;
mod quality;
mod release_roles;
mod selector_paths;

#[test]
fn exhaustive_coverage_and_provider_selection_remain_separate() {
    let base = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    let cases = [
        (
            "main",
            "push",
            base,
            "README.md",
            "true",
            "false",
            false,
            true,
        ),
        (
            "main",
            "push",
            base,
            "README.md",
            "false",
            "false",
            false,
            false,
        ),
        (
            "main",
            "push",
            base,
            "README.md",
            "true",
            "false",
            true,
            false,
        ),
        (
            "main",
            "push",
            base,
            ".github/workflows/main_linux.yml",
            "true",
            "false",
            false,
            false,
        ),
        (
            "main",
            "push",
            "",
            "__force_all__",
            "true",
            "false",
            false,
            false,
        ),
        (
            "manual-full",
            "workflow_dispatch",
            base,
            "README.md",
            "false",
            "true",
            false,
            true,
        ),
        (
            "manual-full",
            "workflow_dispatch",
            base,
            "README.md",
            "false",
            "false",
            false,
            false,
        ),
    ];
    for (profile, event, base, path, main_gate, manual_gate, override_hosted, depot) in cases {
        let plan = plan(profile, event, base, path);
        assert_eq!(plan["signals"]["runner_contract_required"], true);
        let force = override_hosted || plan["signals"]["force_hosted"].as_bool().unwrap();
        let fixture = support::Fixture::new();
        let output = fixture.selector(&[
            ("INPUT_EVENT_NAME", event),
            ("INPUT_ORIGINAL_EVENT_NAME", event),
            ("INPUT_REF", "refs/heads/main"),
            ("INPUT_DEPOT_MAIN_ENABLED", main_gate),
            ("INPUT_MANUAL_USE_DEPOT", manual_gate),
            ("INPUT_FORCE_HOSTED", if force { "true" } else { "false" }),
            ("FIXTURE_DATE", "2026-11-03"),
        ]);
        assert!(output.status.success(), "{profile} {path}: {output:?}");
        let selected = fixture.outputs();
        assert_eq!(
            selected["depot_enabled"],
            depot.to_string(),
            "{profile} {path}"
        );
        if depot {
            assert_eq!(selected["allow_depot_remote_cache"], "false");
            assert_eq!(selected["allow_native_github_cache"], "false");
        }
        fixture.0.close().unwrap();
    }
}

#[test]
fn composed_runner_selection_refuses_untrusted_events_and_expired_cache_authority() {
    let base = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    let pr = plan(
        "pr-ready",
        "pull_request",
        base,
        "mesh/crates/mesh-llm/src/main.rs",
    );
    let main = plan("main", "push", base, "README.md");
    let manual = plan("manual-full", "workflow_dispatch", base, "README.md");
    struct Case<'a> {
        name: &'static str,
        plan: &'a Value,
        inputs: &'a [(&'static str, &'static str)],
        depot: bool,
        native_cache: bool,
        trusted_seed: bool,
    }
    let cases = [
        Case {
            name: "expired matching canary stays cache-denied",
            plan: &pr,
            inputs: &[("INPUT_PR_CANARY_REF", "refs/pull/12/merge")],
            depot: true,
            native_cache: false,
            trusted_seed: false,
        },
        Case {
            name: "expired unrelated canary",
            plan: &pr,
            inputs: &[("INPUT_PR_CANARY_REF", "refs/pull/13/merge")],
            depot: false,
            native_cache: true,
            trusted_seed: true,
        },
        Case {
            name: "fork cannot use matching canary",
            plan: &pr,
            inputs: &[
                ("INPUT_PR_CANARY_REF", "refs/pull/12/merge"),
                ("INPUT_HEAD_REPOSITORY", "fork/mesh-llm"),
            ],
            depot: false,
            native_cache: true,
            trusted_seed: true,
        },
        Case {
            name: "PR head ref cannot use merge canary",
            plan: &pr,
            inputs: &[
                ("INPUT_PR_CANARY_REF", "refs/pull/12/merge"),
                ("INPUT_REF", "refs/pull/12/head"),
            ],
            depot: false,
            native_cache: true,
            trusted_seed: true,
        },
        Case {
            name: "disabled PR gate before expiry",
            plan: &pr,
            inputs: &[
                ("FIXTURE_DATE", "2026-09-13"),
                ("INPUT_DEPOT_PR_ENABLED", "false"),
            ],
            depot: false,
            native_cache: true,
            trusted_seed: true,
        },
        Case {
            name: "trusted main without the exception",
            plan: &main,
            inputs: &[
                ("INPUT_EVENT_NAME", "push"),
                ("INPUT_REF", "refs/heads/main"),
            ],
            depot: true,
            native_cache: false,
            trusted_seed: false,
        },
        Case {
            name: "push to another branch",
            plan: &main,
            inputs: &[
                ("INPUT_EVENT_NAME", "push"),
                ("INPUT_REF", "refs/heads/other"),
            ],
            depot: false,
            native_cache: true,
            trusted_seed: true,
        },
        Case {
            name: "PR-origin dispatch cannot opt in",
            plan: &manual,
            inputs: &[
                ("INPUT_EVENT_NAME", "workflow_dispatch"),
                ("INPUT_ORIGINAL_EVENT_NAME", "pull_request"),
                ("INPUT_REF", "refs/heads/main"),
                ("INPUT_MANUAL_USE_DEPOT", "true"),
            ],
            depot: false,
            native_cache: true,
            trusted_seed: true,
        },
        Case {
            name: "unsupported event fails closed",
            plan: &main,
            inputs: &[
                ("INPUT_EVENT_NAME", "schedule"),
                ("INPUT_REF", "refs/heads/main"),
            ],
            depot: false,
            native_cache: true,
            trusted_seed: true,
        },
    ];
    for case in cases {
        assert_eq!(case.plan["signals"]["force_hosted"], false, "{}", case.name);
        let fixture = support::Fixture::new();
        let mut inputs = vec![
            ("FIXTURE_DATE", "2026-09-14"),
            ("INPUT_DEPOT_PR_ENABLED", "true"),
            ("INPUT_DEPOT_MAIN_ENABLED", "true"),
            ("INPUT_FORCE_HOSTED", "false"),
        ];
        inputs.extend_from_slice(case.inputs);
        let result = fixture.selector(&inputs);
        assert!(result.status.success(), "{}: {result:?}", case.name);
        let selected = fixture.outputs();
        assert_eq!(
            selected["depot_enabled"],
            case.depot.to_string(),
            "{}",
            case.name
        );
        assert_eq!(
            selected["allow_depot_remote_cache"], "false",
            "{}",
            case.name
        );
        assert_eq!(
            selected["allow_native_github_cache"],
            case.native_cache.to_string(),
            "{}",
            case.name
        );
        assert_eq!(
            selected["allow_trusted_sccache_seed"],
            case.trusted_seed.to_string(),
            "{}",
            case.name
        );
        assert_eq!(
            selected["runner"],
            if case.depot {
                "depot-ubuntu-24.04"
            } else {
                "ubuntu-24.04"
            },
            "{}",
            case.name
        );
        assert_eq!(
            selected["runner_macos"],
            if case.depot {
                "depot-macos-15"
            } else {
                "macos-15"
            },
            "{}",
            case.name
        );
        assert_eq!(
            selected["runner_windows"],
            if case.depot {
                "depot-windows-2022"
            } else {
                "windows-2022"
            },
            "{}",
            case.name
        );
        if case.depot {
            let cache = fixture.cache(&[
                (
                    "INPUT_ALLOW_DEPOT_REMOTE_CACHE",
                    selected["allow_depot_remote_cache"].as_str().unwrap(),
                ),
                (
                    "INPUT_ALLOW_NATIVE_GITHUB_CACHE",
                    selected["allow_native_github_cache"].as_str().unwrap(),
                ),
            ]);
            assert_eq!(cache["failures"], json!([]), "{}", case.name);
            assert_eq!(
                cache["exports"]["SCCACHE_MULTILEVEL_CHAIN"], "disk",
                "{}",
                case.name
            );
        }
        fixture.0.close().unwrap();
    }
    for key in [
        "INPUT_FORCE_HOSTED",
        "INPUT_DEPOT_MAIN_ENABLED",
        "INPUT_MANUAL_USE_DEPOT",
    ] {
        let fixture = support::Fixture::new();
        let output = fixture.selector(&[(key, "TRUE")]);
        assert!(!output.status.success(), "{key}");
        assert!(!fixture.path().join("outputs").exists(), "{key}");
        fixture.0.close().unwrap();
    }

    let request = json!({
        "profile": "main", "event_name": "schedule",
        "source_sha": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "base_sha": base,
        "changed_files": ["README.md"], "affected_crates": ["mesh-llm"],
        "workspace_packages": [{"name": "mesh-llm", "path": "mesh/crates/mesh-llm"}]
    });
    let mut child = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .current_dir(support::root())
        .args(["ci", "plan"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(request.to_string().as_bytes())
        .unwrap();
    let rejected = child.wait_with_output().unwrap();
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("unsupported event_name"));
}

#[test]
fn selection_trace_preserves_decision_without_claiming_execution_or_echoing_untrusted_text() {
    let fixture = support::Fixture::new();
    let summary = fixture.path().join("selection-summary.md");
    let summary_path = summary.to_str().unwrap();
    let output = fixture.selector(&[
        ("FIXTURE_DATE", "2026-09-14"),
        ("INPUT_PR_CANARY_REF", "refs/pull/12/merge"),
        ("GITHUB_STEP_SUMMARY", summary_path),
    ]);
    assert!(output.status.success());
    let selected = fixture.outputs();
    let trace_text = selected["selection_trace"].as_str().unwrap();
    let trace: Value = serde_json::from_str(trace_text).unwrap();
    assert_eq!(trace["kind"], "selection_only");
    assert_eq!(trace["selected_provider"], "depot");
    assert_eq!(trace["selected_labels"]["linux"], selected["runner"]);
    assert_eq!(trace["selected_labels"]["linux_8"], selected["runner_8"]);
    assert_eq!(
        trace["selected_labels"]["linux_arm_8"],
        selected["runner_arm_8"]
    );
    assert_eq!(trace["selected_labels"]["macos"], selected["runner_macos"]);
    assert_eq!(
        trace["selected_labels"]["windows"],
        selected["runner_windows"]
    );
    assert_eq!(trace["reason"], "eligible_depot_selection");
    assert_eq!(trace["allow_native_github_cache"], false);
    assert_eq!(trace["exception_active"], false);
    for field in [
        "role",
        "policy_revision",
        "actual_runner",
        "effective_cache_mode",
    ] {
        assert!(trace[field].is_null(), "{field}");
    }
    let written = std::fs::read_to_string(&summary).unwrap();
    assert!(written.contains("not execution evidence"));
    assert!(written.contains(trace_text));
    fixture.0.close().unwrap();

    let fixture = support::Fixture::new();
    assert!(
        fixture
            .selector(&[
                ("INPUT_EVENT_NAME", "workflow_dispatch"),
                ("INPUT_REF", "refs/heads/main"),
                ("INPUT_MANUAL_USE_DEPOT", "true"),
                ("INPUT_FORCE_HOSTED", "true"),
            ])
            .status
            .success()
    );
    let selected = fixture.outputs();
    let trace: Value = serde_json::from_str(selected["selection_trace"].as_str().unwrap()).unwrap();
    assert_eq!(trace["selected_provider"], "github_hosted");
    assert_eq!(trace["reason"], "forced_hosted");
    assert_eq!(trace["force_hosted"], true);
    fixture.0.close().unwrap();

    let fixture = support::Fixture::new();
    let summary = fixture.path().join("untrusted-summary.md");
    assert!(
        fixture
            .selector(&[
                ("INPUT_EVENT_NAME", "unknown-sensitive-value"),
                ("INPUT_HEAD_SHA", "invalid-sensitive-value"),
                ("GITHUB_STEP_SUMMARY", summary.to_str().unwrap()),
            ])
            .status
            .success()
    );
    let selected = fixture.outputs();
    let text = selected["selection_trace"].as_str().unwrap();
    let trace: Value = serde_json::from_str(text).unwrap();
    assert_eq!(trace["event"], "other");
    assert!(trace["declared_head_sha"].is_null());
    assert!(!text.contains("sensitive-value"));
    assert!(
        !std::fs::read_to_string(summary)
            .unwrap()
            .contains("sensitive-value")
    );
    fixture.0.close().unwrap();

    let fixture = support::Fixture::new();
    let summary = fixture.path().join("invalid-summary.md");
    let output = fixture.selector(&[
        ("INPUT_FORCE_HOSTED", "TRUE"),
        ("GITHUB_STEP_SUMMARY", summary.to_str().unwrap()),
    ]);
    assert!(!output.status.success());
    assert!(!summary.exists());
    assert!(!fixture.path().join("outputs").exists());
    fixture.0.close().unwrap();
}

#[test]
fn execution_record_accepts_the_real_selector_trace_without_claiming_observation() {
    let fixture = support::Fixture::new();
    assert!(fixture.selector(&[]).status.success());
    let output = fixture.outputs();
    let trace = output["selection_trace"].as_str().unwrap();
    let request = json!({
        "selection_trace": trace,
        "context": {
            "role": "linux-native-16", "run_id": 42, "run_attempt": 1,
            "runner": {"name": "private-fixture-runner", "os": "Linux", "arch": "X64"}
        }
    });
    let input = serde_json::to_vec(&request).unwrap();
    let expected = ci_controller::plan::decision_evidence::render(&input).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["ci", "execution-record"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(&input).unwrap();
    let result = child.wait_with_output().unwrap();
    assert!(result.status.success(), "{:?}", result.stderr);
    assert_eq!(result.stdout, format!("{expected}\n").as_bytes());
    assert!(!expected.contains("private-fixture-runner"));
    let record: Value = serde_json::from_str(&expected).unwrap();
    assert_eq!(record["authenticity"], "unverified");
    assert_eq!(record["selection_to_job_binding"], "unverified");
    assert!(record["actual_runner"].is_null());
    assert!(record["effective_cache_mode"].is_null());
    assert_eq!(record["authorizes_execution"], false);
    fixture.0.close().unwrap();
}
