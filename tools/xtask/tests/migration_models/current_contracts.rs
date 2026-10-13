//! Recognized workload/profile pairs and current native-event model admission.
use crate::support::{Stage, TestResult, repository_root, text, xtask};
use serde_json::Value;
use std::fs;

#[test]
fn model_registry_all_recognized_incompatible_workload_profile_pairs_refuse() -> TestResult {
    let registry: Value = serde_json::from_slice(&fs::read(
        repository_root().join("ci/model-artifacts/registry.json"),
    )?)?;
    let index = registry["artifacts"]
        .as_array()
        .ok_or("artifacts")?
        .iter()
        .position(|row| row.get("certification").is_some())
        .ok_or("certified artifact")?;
    let stage = Stage::new("recognized-class-profile-pairs")?;
    for (class, profile) in [
        ("embedding", "full"),
        ("rerank", "graph-only"),
        ("causal_generation", "workload-smoke"),
        ("causal_generation", "workload-oracle"),
    ] {
        let mut changed = registry.clone();
        changed["artifacts"][index]["certification"]["class"] = Value::from(class);
        changed["artifacts"][index]["certification"]["profile"] = Value::from(profile);
        stage.write("registry.json", &serde_json::to_vec(&changed)?)?;
        let output = xtask(
            stage.path(),
            &[
                "models",
                "generate",
                "--registry",
                "registry.json",
                "--check",
            ],
        )?;
        assert_eq!(
            output.status.code(),
            Some(2),
            "{class}/{profile}: {}",
            text(&output.stderr)
        );
        assert!(
            output.stdout.is_empty(),
            "failed generation published output"
        );
        assert!(
            text(&output.stderr).contains("class and profile are incompatible"),
            "{class}/{profile}: {}",
            text(&output.stderr)
        );
    }
    Ok(())
}

#[test]
fn model_registry_current_native_event_gate_model_resolves_at_pr_and_main() -> TestResult {
    // The owning parsed workflow guard binds these exact current action inputs
    // and original-event cadence, and rejects substitutions. Resolve the same
    // current checked-in model at both mandatory cadences, without downloading.
    let root = repository_root();
    for cadence in ["pull-request", "main"] {
        let output = xtask(
            &root,
            &[
                "models",
                "resolve",
                "ci/model-artifacts/manifests/skippy-ci-smoke.json",
                "--artifact-id",
                "family-qwen3-dense",
                "--cadence",
                cadence,
                "--require-single-file",
            ],
        )?;
        assert!(
            output.status.success(),
            "{cadence}: {}",
            text(&output.stderr)
        );
        let selected: Value = serde_json::from_slice(&output.stdout)?;
        assert_eq!(selected["artifact_id"], "family-qwen3-dense");
        let files: Vec<String> = serde_json::from_str(
            selected["files_json"]
                .as_str()
                .ok_or("selected file list")?,
        )?;
        assert_eq!(files.len(), 1);
        assert_eq!(selected["file"].as_str(), Some(files[0].as_str()));
    }
    Ok(())
}
