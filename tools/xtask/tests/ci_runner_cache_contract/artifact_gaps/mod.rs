//! Remaining original CI artifact intents, without hardware execution claims.
mod packaged_probe;
use super::{
    support,
    workflow_yaml::{self, Node},
};
use std::fs;
fn document(file: &str) -> Node {
    workflow_yaml::parse(
        &fs::read_to_string(support::root().join(format!(".github/workflows/{file}"))).unwrap(),
    )
    .unwrap()
}
fn text<'a>(node: &'a Node, key: &str) -> Option<&'a str> {
    node.get(key).and_then(Node::text)
}
fn job<'a>(node: &'a Node, key: &str) -> &'a Node {
    node.get("jobs").unwrap().get(key).unwrap()
}
fn steps(node: &Node) -> &[Node] {
    let Node::Seq(steps) = node.get("steps").unwrap() else {
        panic!("steps")
    };
    steps
}
fn input<'a>(node: &'a Node, key: &str) -> Option<&'a str> {
    node.get("with").and_then(|node| text(node, key))
}

#[test]
fn artifact_release_swift_producer_retains_exhaustive_target_results() {
    let release = document("release.yml");
    let producer = job(&release, "build_swift_sdk_artifact");
    assert_eq!(
        text(producer, "uses"),
        Some("./.github/workflows/swift-sdk-artifact.yml")
    );
    assert_eq!(input(producer, "fail_fast"), Some("false"));
    let swift = document("swift-sdk-artifact.yml");
    let strategies = swift
        .get("jobs")
        .unwrap()
        .entries()
        .iter()
        .filter_map(|(_, job)| job.get("strategy"))
        .collect::<Vec<_>>();
    assert!(!strategies.is_empty());
    for strategy in strategies {
        assert_eq!(text(strategy, "fail-fast"), Some("${{ inputs.fail_fast }}"));
    }
}

#[test]
fn artifact_safetensors_smoke_preserves_skip_ui_and_seed_before_compile() {
    let document = document("ci-rust-tests-slice.yml");
    let smoke = job(&document, "safetensors_runtime_smoke");
    assert_eq!(
        text(smoke.get("env").unwrap(), "MESH_LLM_SKIP_UI"),
        Some("1")
    );
    let steps = steps(smoke);
    let matches = |key, value| {
        steps
            .iter()
            .enumerate()
            .filter(|(_, step)| text(step, key) == Some(value))
            .collect::<Vec<_>>()
    };
    let seed = matches("uses", "./.github/actions/restore-sccache-seed");
    let build = matches("id", "safetensors_smoke_test");
    assert_eq!(seed.len(), 1);
    assert_eq!(build.len(), 1);
    assert!(seed[0].0 < build[0].0);
    assert_eq!(
        input(seed[0].1, "allow_trusted_seed"),
        Some("${{ needs.runner_policy.outputs.allow_trusted_sccache_seed }}")
    );
    assert!(
        text(build[0].1, "run")
            .unwrap()
            .starts_with("set -euo pipefail\n")
    );
}

#[test]
fn artifact_cuda_smoke_uses_only_the_registered_gpu_role_or_hosted_fallback() {
    let document = document("smoke.yml");
    let smoke = job(&document, "smoke_tests");
    let labels =
        r#"["self-hosted","Linux","X64","amd64","gpu-nvidia","mesh-llm-amd64","mesh-llm"]"#;
    assert_eq!(
        text(smoke, "runs-on"),
        Some(
            format!(
                "${{{{ inputs.runner == 'gpu-nvidia' && fromJSON('{labels}') || 'ubuntu-24.04' }}}}"
            )
            .as_str()
        )
    );
    for step in steps(smoke) {
        if let Some(run) = text(step, "run") {
            for package in ["cuda-cudart-12-9", "libcublas-12-9"] {
                assert!(!run.contains(package), "host toolkit injection");
            }
        }
    }
}

#[test]
fn artifact_every_lane_parallelism_input_consumes_its_declared_budget() {
    let mut observed = std::collections::BTreeSet::new();
    for lane in ["quality", "linux", "macos", "windows"] {
        let document = document(&format!("ci-{lane}-lane.yml"));
        for (_, job) in document.get("jobs").unwrap().entries() {
            let Some(inputs) = job.get("with") else {
                continue;
            };
            for (key, value) in inputs.entries() {
                if !matches!(
                    key.as_str(),
                    "max_parallel" | "clippy_max_parallel" | "total_max_workers"
                ) {
                    continue;
                }
                let budget = if key == "total_max_workers" {
                    key.clone()
                } else {
                    format!(
                        "{}_max_parallel",
                        if lane == "quality" { "linux" } else { lane }
                    )
                };
                assert_eq!(
                    value.text(),
                    Some(
                        format!("${{{{ fromJson(inputs.lane_plan_json).budgets.{budget} }}}}")
                            .as_str()
                    )
                );
                observed.insert(budget);
            }
        }
    }
    assert_eq!(
        observed,
        std::collections::BTreeSet::from(
            [
                "linux_max_parallel",
                "macos_max_parallel",
                "windows_max_parallel",
                "total_max_workers"
            ]
            .map(str::to_owned)
        )
    );
}

#[test]
fn artifact_cuda_probe_source_returns_before_benchmark_allocations() {
    let source = fs::read_to_string(
        support::root().join("skippy/crates/skippy-gpu-bench/native/cuda/membench-fingerprint.cu"),
    )
    .unwrap();
    let main = source
        .split_once("int main(int argc, char** argv) {")
        .unwrap()
        .1;
    let option = main.find("strcmp(argv[i], \"--probe\") == 0").unwrap();
    let device_check = main.find("cudaGetDeviceCount(&deviceCount)").unwrap();
    let probe = main.find("if (probeMode) {").unwrap();
    let benchmark = main
        .find("for (int dev = 0; dev < deviceCount; dev++) {")
        .unwrap();
    assert!(option < device_check && device_check < probe && probe < benchmark);
    assert!(main[probe..benchmark].contains("return 0;"));
    assert!(!main[probe..benchmark].contains("cudaMalloc"));
}

#[test]
fn artifact_smoke_handoffs_require_packaged_backends_without_remote_runtime_fallback() {
    let smoke = document("smoke.yml");
    for (name, backend) in [
        (
            "smoke_tests",
            "${{ inputs.runner == 'gpu-nvidia' && 'cuda' || 'cpu' }}",
        ),
        ("smoke_tests_macos", "metal"),
    ] {
        let job = job(&smoke, name);
        assert_eq!(
            text(
                job.get("env").unwrap(),
                "MESH_LLM_NATIVE_RUNTIME_MANIFEST_URL"
            ),
            Some("http://127.0.0.1:9/native-runtimes.json")
        );
        let dense = steps(job)
            .iter()
            .filter(|step| text(step, "id") == Some("dense_model"))
            .collect::<Vec<_>>();
        assert_eq!(dense.len(), 1);
        assert_eq!(
            text(dense[0], "uses"),
            Some("./.github/actions/restore-smoke-inputs")
        );
        assert_eq!(input(dense[0], "expected_backend"), Some(backend));
        assert_eq!(
            input(dense[0], "staged_binary_path"),
            Some("${{ inputs.mesh_binary_target }}")
        );
    }
    // These are source contracts for the retained adapters. Executable native
    // smoke and composition fixtures separately own their behavior.
    for (file, prefix) in [
        ("scripts/ci-smoke-test.sh", "MESH_CI"),
        ("scripts/ci-compat-smoke.sh", "MESH_COMPAT"),
    ] {
        let source = fs::read_to_string(support::root().join(file)).unwrap();
        for dimension in ["BATCH_SIZE", "UBATCH_SIZE"] {
            assert!(source.contains(&format!("{prefix}_{dimension}")));
        }
        if prefix == "MESH_CI" {
            assert!(source.contains("automation required-smoke run"));
        } else {
            assert!(source.contains("[defaults.model_fit]"));
        }
    }
}

const PREPARE: &str = "./.github/actions/prepare-automation";
const RUST_CACHE: &str = "Swatinem/rust-cache@";

/// Every (prepare, rust-cache) pair where the cache restore runs after the
/// preparation. Checks all occurrences, not only the first of each.
fn cache_restores_after_prepare(steps: &[Node]) -> Vec<(usize, usize)> {
    let at = |prefix: &str| -> Vec<usize> {
        steps
            .iter()
            .enumerate()
            .filter(|(_, step)| text(step, "uses").is_some_and(|u| u.starts_with(prefix)))
            .map(|(index, _)| index)
            .collect()
    };
    let caches = at(RUST_CACHE);
    at(PREPARE)
        .into_iter()
        .flat_map(|prepare| {
            caches
                .iter()
                .filter(move |&&cache| cache > prepare)
                .map(move |&cache| (prepare, cache))
        })
        .collect()
}

fn fixture_steps(uses: &[&str]) -> Vec<Node> {
    let yaml: String = std::iter::once("steps:\n".to_owned())
        .chain(uses.iter().map(|u| format!("  - uses: {u}\n")))
        .collect();
    let Node::Seq(steps) = workflow_yaml::parse(&yaml)
        .unwrap()
        .get("steps")
        .unwrap()
        .clone()
    else {
        panic!("steps")
    };
    steps
}

#[test]
fn artifact_automation_ordering_rejects_old_and_later_cache_restores() {
    let cache = "Swatinem/rust-cache@abc";
    // Release run 40804094970 shape: prepare, then rust-cache.
    assert_eq!(
        cache_restores_after_prepare(&fixture_steps(&[PREPARE, cache])),
        [(0, 1)]
    );
    // A second restore after preparation must not escape a first-match check.
    assert_eq!(
        cache_restores_after_prepare(&fixture_steps(&[cache, PREPARE, cache])),
        [(1, 2)]
    );
    // A later re-preparation does not excuse an earlier unsafe one.
    assert_eq!(
        cache_restores_after_prepare(&fixture_steps(&[PREPARE, cache, PREPARE])),
        [(0, 1)]
    );
    assert!(cache_restores_after_prepare(&fixture_steps(&[cache, cache, PREPARE])).is_empty());
}

/// Swatinem/rust-cache pre-cleans the shared target directory on a restore-key
/// (partial) hit, deleting workspace binaries such as `target/debug/xtask`.
/// Release run 40804094970 lost MESH_LLM_AUTOMATION_BIN this way, so in any job
/// every Rust cache restore must precede every automation preparation.
#[test]
fn artifact_automation_is_prepared_after_rust_cache_restore() {
    let mut checked = 0;
    for entry in fs::read_dir(support::root().join(".github/workflows")).unwrap() {
        let path = entry.unwrap().path();
        let file = path.file_name().unwrap().to_str().unwrap().to_owned();
        if !file.ends_with(".yml") {
            continue;
        }
        let workflow = document(&file);
        let Some(jobs) = workflow.get("jobs") else {
            continue;
        };
        for (name, job) in jobs.entries() {
            let Some(Node::Seq(steps)) = job.get("steps") else {
                continue;
            };
            let uses = |prefix: &str| {
                steps
                    .iter()
                    .any(|step| text(step, "uses").is_some_and(|u| u.starts_with(prefix)))
            };
            if !(uses(PREPARE) && uses(RUST_CACHE)) {
                continue;
            }
            checked += 1;
            let violations = cache_restores_after_prepare(steps);
            assert!(
                violations.is_empty(),
                "{file}/{name}: rust-cache restore after prepare-automation at step indices {violations:?}"
            );
        }
    }
    assert!(checked > 0, "no job combines automation and rust-cache");
}
