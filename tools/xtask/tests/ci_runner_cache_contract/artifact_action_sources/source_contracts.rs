//! Draft addition to the existing ci_runner_cache_contract target, not a new target.
//! These assertions qualify parsed source contracts only. They do not execute Windows.
use super::super::{
    support,
    workflow_yaml::{self, Node},
};
use std::fs;
fn source(path: &str) -> String {
    fs::read_to_string(support::root().join(path)).unwrap()
}
fn doc(path: &str) -> Node {
    workflow_yaml::parse(&source(path)).unwrap()
}
fn runs(node: &Node) -> String {
    match node {
        Node::Seq(items) => items.iter().map(runs).collect::<Vec<_>>().join("\n"),
        Node::Map(items) => items
            .iter()
            .map(|(key, child)| {
                if key == "run" {
                    child.text().unwrap_or_default().to_owned()
                } else {
                    runs(child)
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}
fn contains(run: &str, tokens: &[&str]) {
    for token in tokens {
        assert!(run.contains(token), "missing {token}");
    }
}
fn action_prepare(document: &Node) -> &Node {
    let Node::Seq(steps) = document.get("runs").unwrap().get("steps").unwrap() else {
        panic!("action steps");
    };
    let matches = steps
        .iter()
        .filter(|step| step.get("id").and_then(Node::text) == Some("prepare"))
        .collect::<Vec<_>>();
    assert_eq!(matches.len(), 1);
    matches[0]
}
#[test]
fn artifact_quality_job_keeps_tooling_release_targets_and_client_dependency_validation() {
    let document = doc(".github/workflows/ci-quality-slice.yml");
    let quality = document
        .get("jobs")
        .unwrap()
        .get("quality_contracts")
        .unwrap();
    let run = runs(quality);
    contains(
        &run,
        &[
            "actionlint -config-file .github/actionlint.yaml",
            "just ci-legacy-contracts",
            "cargo run -p xtask -- repo-consistency release-targets",
            "cargo tree -p mesh-llm-client",
        ],
    );
    let Node::Seq(steps) = quality.get("steps").unwrap() else {
        panic!("steps");
    };
    assert_eq!(
        steps
            .iter()
            .filter(|step| step.get("uses").and_then(Node::text)
                == Some("./.github/actions/install-actionlint"))
            .count(),
        1
    );
    assert!(steps.iter().all(|step| {
        !step
            .get("uses")
            .and_then(Node::text)
            .is_some_and(|value| value.starts_with("actions/setup-python@"))
    }));
    assert!(!run.contains("requirements-ci-python.txt"));
    assert!(!run.contains("pip install"));
}
#[test]
fn artifact_windows_host_source_preserves_neutral_integrity_and_explicit_commit_contracts() {
    let document = doc(".github/actions/prepare-windows-host-input/action.yml");
    let prepare = action_prepare(&document);
    assert_eq!(prepare.get("shell").and_then(Node::text), Some("pwsh"));
    assert_eq!(
        prepare
            .get("env")
            .unwrap()
            .get("INPUT_COMMIT")
            .and_then(Node::text),
        Some("${{ inputs.commit }}")
    );
    let run = prepare.get("run").and_then(Node::text).unwrap();
    contains(
        run,
        &[
            "& .\\scripts\\build-windows.ps1 -BuildProfile $profile -HostOnly",
            "cargo xtool native verify-host-dependencies",
            "mesh-llm.exe.sha256",
            "cargo build -q -p xtask --bin xtask",
            "release-attestation stamp",
            "release-attestation inspect",
            "--require-source-commit",
            "--commit",
            "INPUT_COMMIT",
            "\"$attestationVerifierPath.sha256\"",
            "\"$verifierHash  release-attestation-verifier.exe\"",
        ],
    );
    for forbidden in ["package-native-runtime.sh", "compose-product"] {
        assert!(!run.contains(forbidden));
    }
    let start = run.find("if ($profile -eq \"debug\")").unwrap();
    let end = run[start..]
        .find("if ($env:INPUT_SKIP_UI -eq \"true\")")
        .unwrap()
        + start;
    let debug = &run[start..end];
    contains(
        debug,
        &["cargo pkgid -p mesh-llm", "$env:MESH_LLM_BUILD_VERSION"],
    );
    assert!(!debug.contains("git "));
    let unix_document = doc(".github/actions/prepare-host-input/action.yml");
    let unix_prepare = action_prepare(&unix_document);
    assert_eq!(
        unix_prepare
            .get("env")
            .unwrap()
            .get("INPUT_COMMIT")
            .and_then(Node::text),
        Some("${{ inputs.commit }}")
    );
    let unix = unix_prepare.get("run").and_then(Node::text).unwrap();
    contains(
        unix,
        &["--require-source-commit", "--commit", "INPUT_COMMIT"],
    );
}
fn signing_callers(node: &Node, count: &mut usize) {
    if let Some(with) = node.get("with")
        && with.get("attestation_signing_key_file").is_some()
    {
        *count += 1;
        assert_eq!(
            with.get("commit").and_then(Node::text),
            Some("${{ needs.metadata.outputs.source_sha }}")
        );
    }
    match node {
        Node::Map(items) => {
            for (_, child) in items {
                signing_callers(child, count);
            }
        }
        Node::Seq(items) => {
            for child in items {
                signing_callers(child, count);
            }
        }
        _ => {}
    }
}
#[test]
fn artifact_release_stamps_every_signing_caller_with_immutable_metadata_commit() {
    let mut count = 0;
    signing_callers(&doc(".github/workflows/release.yml"), &mut count);
    assert!(count > 0);
}
#[test]
fn artifact_native_attestation_verifier_dependencies_remain_abi_free() {
    let xtask: toml::Value = toml::from_str(&source("tools/xtask/Cargo.toml")).unwrap();
    let dependencies = xtask["dependencies"].as_table().unwrap();
    assert_eq!(
        dependencies["mesh-llm-release-footer"]["workspace"].as_bool(),
        Some(true)
    );
    for name in ["mesh-llm-system", "skippy-ffi"] {
        assert!(!dependencies.contains_key(name));
    }
    let footer: toml::Value =
        toml::from_str(&source("mesh/crates/mesh-llm-release-footer/Cargo.toml")).unwrap();
    let names = footer["dependencies"]
        .as_table()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(names, ["hex", "sha2"].into_iter().collect());
}
#[test]
fn artifact_seed_hashes_bind_all_imported_just_sources_on_every_original_consumer() {
    for name in [
        "cache-warm-sccache.yml",
        "ci-quality-slice.yml",
        "ci-linux-host-slice.yml",
        "ci-rust-tests-slice.yml",
        "ci-linux-runtime-slice.yml",
    ] {
        let workflow = source(&format!(".github/workflows/{name}"));
        assert!(workflow.contains("hashFiles('Cargo.lock', '.github/cache-version.txt', '.cargo/config.toml', 'scripts/cargo-linker', 'scripts/cargo-linker-linux-*', 'scripts/lib/lld.sh', 'Justfile', 'just/**')"), "{name}");
    }
}
