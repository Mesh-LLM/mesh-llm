//! Behavioral execution of the maintained routing policy; no regex reimplementation.
use super::workflows::tool;
#[path = "sdk_call_candidates.rs"]
mod sdk_call_candidates;
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use std::{collections::BTreeMap, num::NonZeroUsize, path::Path, time::Duration};

fn complete_bash(root: &Path, script: &str, values: &[(&str, &str)]) -> (bool, String, String) {
    let mut environment = BTreeMap::from([
        (
            "PATH".into(),
            Value::Public(std::env::var_os("PATH").unwrap()),
        ),
        ("HOME".into(), Value::Public(root.into())),
    ]);
    for (key, value) in values {
        environment.insert((*key).into(), Value::Public((*value).into()));
    }
    let report = process::supervise_raw(
        &ProcessSpec {
            executable: tool("bash"),
            cwd: root.into(),
            environment,
            arguments: ["-euo", "pipefail", "-c", script]
                .into_iter()
                .map(|arg| Value::Public(arg.into()))
                .collect(),
        },
        &Limits {
            execution: Duration::from_secs(5),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 16384,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        RawCaptureOptions {
            stdout: NonZeroUsize::new(16384),
            stderr: NonZeroUsize::new(16384),
        },
    )
    .unwrap();
    assert_eq!(report.process.outcome, process::Outcome::Exited);
    assert!(
        report.process.failure.is_none()
            && report.process.cleanup.complete
            && !report.process.cleanup.forced
            && !report.process.cleanup.graceful_signal_failed
            && report.process.cleanup.failure.is_none(),
        "{report:?}"
    );
    for stream in [&report.process.stdout, &report.process.stderr] {
        assert!(
            !stream.truncated && stream.suppressed_lines == 0,
            "routing fixture must retain complete nonsuppressed streams: {stream:?}"
        );
    }
    (
        report.process.status.unwrap().success(),
        String::from_utf8(report.stdout.unwrap().as_bytes().to_vec()).unwrap(),
        String::from_utf8(report.stderr.unwrap().as_bytes().to_vec()).unwrap(),
    )
}
const SOURCE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../.github/actions/compute-changes/derive-outputs.sh"
));

fn region(start: &str, end: &str) -> String {
    assert_eq!(SOURCE.matches(start).count(), 1, "unique routing start");
    assert_eq!(SOURCE.matches(end).count(), 1, "unique routing end");
    let body = SOURCE
        .split_once(start)
        .unwrap()
        .1
        .split_once(end)
        .unwrap()
        .0;
    format!("{start}{body}\n")
}

fn classify(changed: &str, force: &str, event: &str) -> String {
    let root = tempfile::tempdir().unwrap();
    let script = format!(
        "{}{}printf '%s %s %s %s %s' \"$RUNNER_CONTRACT_REQUIRED\" \"$BACKEND_CHANGED\" \"$WINDOWS_CPU_BUILD_REQUIRED\" \"$WINDOWS_GPU_BUILD_REQUIRED\" \"$SDK_SMOKE_REQUIRED\"\n",
        region(
            "RUNNER_CONTRACT_REQUIRED=\"false\"",
            "# Determine docs_only"
        ),
        region(
            "BACKEND_CHANGED=\"false\"",
            "# Inference artifacts are needed"
        )
    );
    let (ok, output, error) = complete_bash(
        root.path(),
        &script,
        &[
            ("CHANGED_FILES", changed),
            ("FORCE_ALL", force),
            ("EVENT_NAME", event),
            ("ALL_RUST", "false"),
            ("AFFECTED_CRATES", "[]"),
            ("BACKEND_RECIPE_CHANGED", "false"),
        ],
    );
    assert!(ok, "{changed}: {error}");
    root.close()
        .expect("owned routing fixture directory deletion failed");
    output
}

#[test]
fn windows_shared_product_primitives_route_both_products_in_supported_layouts() {
    for prefix in ["", "mesh/", "skippy/"] {
        for path in [
            "scripts/package-release.ps1",
            "scripts/package-native-runtime.sh",
            "scripts/verify-native-runtime-package.sh",
            "scripts/verify-checksum-sidecar.sh",
            "scripts/safe-extract-tar.sh",
            "scripts/compose-product-bundle.sh",
            "scripts/ci-compose-product-input.sh",
            "scripts/ci-client-readiness-smoke.sh",
        ] {
            let changed = format!("{prefix}{path}");
            let output = classify(&changed, "false", "push");
            let fields: Vec<_> = output.split_whitespace().collect();
            assert_eq!(&fields[2..4], &["true", "true"], "{changed}: {output}");
        }
    }
    for path in [
        ".github/actions/compute-changes/action.yml",
        ".github/actions/compute-changes/derive-outputs.sh",
        ".github/actions/prepare-windows-host-input/action.yml",
        ".github/actions/prepare-native-runtime-input/action.yml",
        ".github/actions/compose-product-input/action.yml",
        ".github/actions/save-and-verify-actions-cache/action.yml",
        ".github/workflows/ci.yml",
        ".github/workflows/main_windows.yml",
        ".github/workflows/pr_windows.yml",
        ".github/workflows/release.yml",
        ".github/workflows/windows-warm-caches.yml",
    ] {
        let output = classify(path, "false", "push");
        assert_eq!(
            &output.split_whitespace().collect::<Vec<_>>()[2..4],
            &["true", "true"],
            "{path}: {output}"
        );
    }
}

#[test]
fn windows_footer_is_cpu_only_and_force_sentinel_selects_both_products() {
    for prefix in ["", "mesh/", "skippy/"] {
        let changed = format!("{prefix}crates/mesh-llm-release-footer/src/lib.rs");
        assert_eq!(
            classify(&changed, "false", "push"),
            "false false true false false"
        );
    }
    assert_eq!(classify("", "true", "push"), "false false true true false");
    assert_eq!(
        classify("docs/README.md", "false", "push"),
        "false false false false false"
    );
}

#[test]
fn runner_cache_evidence_routes_runner_contract_and_epoch_routes_consumers() {
    for action in [
        "capture-sccache-stats",
        "configure-sccache-gha",
        "restore-sccache-seed",
        "select-ci-runners",
    ] {
        let changed = format!(".github/actions/{action}/action.yml");
        assert_eq!(
            classify(&changed, "false", "push"),
            "true false false false false"
        );
    }
    assert_eq!(
        classify(
            ".github/actions/resolve-native-toolchain-epoch/action.yml",
            "false",
            "push"
        ),
        "true true true true true"
    );
    assert_eq!(
        classify("", "false", "workflow_dispatch"),
        "true false false false true"
    );
}

#[test]
fn direct_sdk_smoke_and_shared_contract_inputs_select_sdk_in_both_layouts() {
    for prefix in ["", "mesh/", "skippy/"] {
        for path in [
            "scripts/ci-rust-sdk-smoke.sh",
            "scripts/ci-kotlin-sdk-smoke.sh",
            "scripts/ci-swift-sdk-smoke.sh",
            "scripts/ci-prepare-native-runtime.sh",
            "scripts/ci-sdk-fixture.sh",
            "scripts/restore-native-sdk-input.sh",
            "scripts/restore-static-abi-input.sh",
            "scripts/package-sdk-console-assets.sh",
            "scripts/check-sdk-contract.sh",
            "scripts/verify-sdk-console-assets.sh",
            "scripts/verify-swift-privacy-manifest.sh",
            "scripts/verify-swift-release-artifact.sh",
            "scripts/verify-checksum-sidecar.sh",
            "scripts/safe-extract-tar.sh",
            "scripts/verify-static-abi-build-stamp.sh",
        ] {
            let changed = format!("{prefix}{path}");
            let output = classify(&changed, "false", "push");
            assert_eq!(
                output.split_whitespace().last(),
                Some("true"),
                "{changed}: {output}"
            );
        }
    }
    for path in [
        ".github/actions/restore-smoke-inputs/action.yml",
        ".github/actions/compute-changes/action.yml",
        ".github/actions/compute-changes/derive-outputs.sh",
        ".github/workflows/ci.yml",
        ".github/workflows/main_linux.yml",
        ".github/workflows/pr_macos.yml",
        ".github/workflows/release.yml",
        "sdk/node/index.js",
        "mesh/sdk/node/index.js",
        "Package.swift",
    ] {
        let output = classify(path, "false", "push");
        assert_eq!(
            output.split_whitespace().last(),
            Some("true"),
            "{path}: {output}"
        );
    }
}

#[test]
fn source_visible_sdk_smoke_dependencies_refuse_unknown_forms_and_route_sdk_validation() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut calls = std::collections::BTreeSet::new();
    for smoke in [
        "ci-rust-sdk-smoke.sh",
        "ci-kotlin-sdk-smoke.sh",
        "ci-swift-sdk-smoke.sh",
    ] {
        let source = std::fs::read_to_string(root.join("scripts").join(smoke)).unwrap();
        let observed = sdk_call_candidates::collect(&source)
            .unwrap_or_else(|error| panic!("{smoke}: {error}"));
        assert!(
            !observed.adapters.is_empty(),
            "{smoke}: no reviewed adapters"
        );
        for library in observed.sourced_libraries {
            assert!(root.join(&library).is_file(), "{smoke}: {library}");
        }
        calls.extend(observed.adapters);
    }
    for call in calls {
        assert!(root.join(&call).is_file(), "{call}: dependency exists");
        let output = classify(&call, "false", "push");
        assert_eq!(
            output.split_whitespace().last(),
            Some("true"),
            "{call}: {output}"
        );
    }
}

fn workflow_sdk_calls(
    workflow: &crate::workflow_yaml::Node,
) -> Result<std::collections::BTreeSet<String>, String> {
    use crate::workflow_yaml::Node;
    let Some(Node::Map(jobs)) = workflow.get("jobs") else {
        return Err("workflow jobs must be a mapping".into());
    };
    let mut calls = std::collections::BTreeSet::new();
    for (name, job) in jobs {
        if !matches!(job, Node::Map(_)) {
            return Err(format!("job {name} must be a mapping"));
        }
        let Some(steps) = job.get("steps") else {
            // Reusable-workflow jobs have no executable steps here.
            continue;
        };
        let Node::Seq(steps) = steps else {
            return Err(format!("job {name} steps must be a sequence"));
        };
        for (index, step) in steps.iter().enumerate() {
            if !matches!(step, Node::Map(_)) {
                return Err(format!("job {name} step {index} must be a mapping"));
            }
            let Some(run) = step.get("run") else {
                continue;
            };
            let Node::Scalar(run) = run else {
                return Err(format!("job {name} step {index} run must be a scalar"));
            };
            let observed = sdk_call_candidates::collect(run)
                .map_err(|error| format!("job {name} step {index}: {error}"))?;
            calls.extend(observed.adapters);
        }
    }
    Ok(calls)
}

#[test]
fn parsed_sdk_workflow_run_blocks_select_each_sdk_smoke_and_route_every_visible_adapter() {
    let tree = super::workflows::workflow("sdk-smoke.yml");
    let calls = workflow_sdk_calls(&tree).unwrap();
    for sdk in ["rust", "kotlin", "swift"] {
        assert!(
            calls.contains(&format!("scripts/ci-{sdk}-sdk-smoke.sh")),
            "missing maintained {sdk} smoke entrypoint"
        );
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for call in calls {
        assert!(
            root.join(&call).is_file(),
            "workflow adapter absent: {call}"
        );
        assert_eq!(
            classify(&call, "false", "push").split_whitespace().last(),
            Some("true"),
            "workflow adapter does not select SDK: {call}"
        );
    }
}

#[test]
fn workflow_sdk_census_uses_only_direct_job_step_runs() {
    let workflow = crate::workflow_yaml::parse(
        r#"
env:
  run: scripts/unsupported-root-env.sh
run: scripts/unsupported-root.sh
jobs:
  native:
    run: scripts/unsupported-job.sh
    env:
      run: scripts/unsupported-job-env.sh
    steps:
      - uses: actions/checkout@v4
        with:
          run: scripts/unsupported-action-input.sh
      - run: scripts/ci-rust-sdk-smoke.sh --skip-build
        env:
          run: scripts/unsupported-step-env.sh
        with:
          run: scripts/unsupported-step-input.sh
      - name: scripts/unsupported-name.sh
        run: |
          retry_transient scripts/ci-kotlin-sdk-smoke.sh "$1"
  reusable:
    uses: ./.github/workflows/sdk-smoke.yml
    with:
      run: scripts/unsupported-reusable-input.sh
"#,
    )
    .unwrap();
    assert_eq!(
        workflow_sdk_calls(&workflow).unwrap(),
        std::collections::BTreeSet::from([
            "scripts/ci-rust-sdk-smoke.sh".into(),
            "scripts/ci-kotlin-sdk-smoke.sh".into(),
        ])
    );
}

#[test]
fn workflow_sdk_census_cannot_replace_missing_entrypoints_with_run_key_data() {
    let workflow = crate::workflow_yaml::parse(
        r#"
run: scripts/ci-rust-sdk-smoke.sh
env:
  run: scripts/ci-rust-sdk-smoke.sh
jobs:
  native:
    env:
      run: scripts/ci-kotlin-sdk-smoke.sh
    steps:
      - uses: actions/checkout@v4
        with:
          run: scripts/ci-swift-sdk-smoke.sh
          cache-key: "${{ hashFiles('scripts/ci-rust-sdk-smoke.sh') }}"
        env:
          run: scripts/ci-kotlin-sdk-smoke.sh
      - name: scripts/ci-swift-sdk-smoke.sh
        run: echo fixture
"#,
    )
    .unwrap();
    let calls = workflow_sdk_calls(&workflow).unwrap();
    assert!(calls.is_empty());
    for sdk in ["rust", "kotlin", "swift"] {
        assert!(!calls.contains(&format!("scripts/ci-{sdk}-sdk-smoke.sh")));
    }
}

#[test]
fn workflow_sdk_census_refuses_malformed_steps_and_unknown_actual_run_calls() {
    for source in [
        "jobs: []\n",
        "jobs:\n  native: invalid\n",
        "jobs:\n  native:\n    steps:\n      run: scripts/ci-rust-sdk-smoke.sh\n",
        "jobs:\n  native:\n    steps:\n      - invalid-step\n",
        "jobs:\n  native:\n    steps:\n      - run: []\n",
        "jobs:\n  native:\n    steps:\n      - run: scripts/ci-rust-sdk-smoke.sh\n      - run: bash scripts/unsupported.sh\n",
    ] {
        let workflow = crate::workflow_yaml::parse(source).unwrap();
        assert!(
            workflow_sdk_calls(&workflow).is_err(),
            "must refuse malformed or unreviewed executable source: {source}"
        );
    }
}

#[path = "producer_routes.rs"]
mod producer_routes;
