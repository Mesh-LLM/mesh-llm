//! The real checkout: the checked-in registry projects to the checked-in
//! manifests, the frozen fixtures still match them, and the required smoke
//! identities stay pinned.

use crate::support::{Stage, TestResult, assert_streams, fixture_dir, repository_root, xtask};
use serde_json::Value;
use std::fs;

const OUTPUTS: [&str; 13] = [
    "ci/llama-canary/family-certified.json",
    "ci/model-artifacts/manifests/product-smoke.json",
    "ci/model-artifacts/manifests/scripted-binary-smoke.json",
    "ci/model-artifacts/manifests/sdk-smoke.json",
    "ci/model-artifacts/manifests/hf-download-smoke.json",
    "ci/model-artifacts/manifests/openai-smoke.json",
    "ci/model-artifacts/manifests/skippy-correctness.json",
    "ci/model-artifacts/manifests/safetensors-runtime-smoke.json",
    "ci/model-artifacts/manifests/skippy-ci-smoke.json",
    "ci/model-artifacts/manifests/skippy-system-one-smoke.json",
    "ci/model-artifacts/manifests/skippy-parity.json",
    "ci/model-artifacts/manifests/competitive-benchmark.json",
    "ci/model-artifacts/manifests/radix-cache.json",
];

#[test]
fn migration_models_checked_in_projection_is_current() -> TestResult {
    // Given/When: the generator checks the real checkout.
    let output = xtask(&repository_root(), &["models", "generate", "--check"])?;
    // Then: every checked-in output already equals the projection.
    assert_streams("live check", &output, 0, "", "");
    Ok(())
}

#[test]
fn migration_models_regenerated_real_registry_is_byte_identical() -> TestResult {
    // Given: a copy of the real registry and nothing else.
    let root = repository_root();
    let stage = Stage::new("live")?;
    let registry = "ci/model-artifacts/registry.json";
    stage.write(registry, &fs::read(root.join(registry))?)?;
    // When: the generator writes a fresh projection.
    assert_streams(
        "regenerate",
        &xtask(stage.path(), &["models", "generate"])?,
        0,
        "",
        "",
    );
    // Then: all thirteen outputs equal the checked-in bytes.
    for path in OUTPUTS {
        assert_eq!(
            stage.read(path)?,
            fs::read_to_string(root.join(path))?,
            "{path}"
        );
    }
    Ok(())
}

#[test]
fn migration_models_frozen_manifests_match_the_checkout() -> TestResult {
    // Given/When: the frozen resolver inputs and the checked-in manifests.
    let root = repository_root();
    for path in &OUTPUTS[1..] {
        let name = path.rsplit('/').next().ok_or("manifest name")?;
        // Then: frozen artifact payloads remain unchanged apart from the
        // registry digest and Qwen's added System One membership.
        if name == "skippy-system-one-smoke.json" {
            continue;
        }
        let mut frozen: Value =
            serde_json::from_slice(&fs::read(fixture_dir().join("manifests").join(name))?)?;
        let mut current: Value = serde_json::from_slice(&fs::read(root.join(path))?)?;
        if name == "product-smoke.json" {
            current["artifacts"]
                .as_array_mut()
                .ok_or("product artifacts")?
                .retain(|row| row["id"] != "family-laya-multilingual");
        }
        if let Some(rows) = frozen["artifacts"].as_array_mut() {
            for row in rows {
                if row["id"] == "qwen3-q8-correctness" {
                    assert_eq!(
                        row["notes"],
                        "Pinned Skippy runtime/model-package correctness fixture."
                    );
                    // Main added the Skippy crate prefix to this human-readable note;
                    // frozen identities and every integrity field remain asserted below.
                    row["notes"] = Value::from(
                        "Pinned Skippy runtime/skippy-model-package correctness fixture.",
                    );
                }
                if row["id"] == "family-qwen3-dense" {
                    row["suites"]
                        .as_array_mut()
                        .ok_or("frozen Qwen suites")?
                        .push(Value::from("skippy-system-one-smoke"));
                }
            }
        }
        frozen["registry_sha256"] = current["registry_sha256"].clone();
        assert_eq!(frozen, current, "{name}");
    }
    Ok(())
}

#[test]
fn migration_models_required_smoke_identities_are_unchanged() -> TestResult {
    // Given: the checked-in product-smoke manifest.
    let root = repository_root();
    let manifest: Value = serde_json::from_slice(&fs::read(
        root.join("ci/model-artifacts/manifests/product-smoke.json"),
    )?)?;
    let pinned = [
        (
            "smollm2-q8-inference",
            "unsloth/SmolLM2-135M-Instruct-GGUF",
            "9e6855bc4be717fca1ef21360a1db4b29d5c559a",
            "c4a3dd037301b6ecea31d6da37f5cd793ead920dd5ddfe6d589294628d6ce66a",
        ),
        (
            "family-granite-hybrid",
            "ibm-granite/granite-4.0-h-350m-GGUF",
            "a864f823cce6e6048b5752e2816fe7a23987d790",
            "0a8d6a7373602fadfba274a640ba784b86cc6847f1c67f1b0a90fa2ec266b7fb",
        ),
        (
            "family-laya-multilingual",
            "meshllm/laya-multilingual-F16-GGUF",
            "bcc99560232b5a5c91cb14d46b9496acbeae2c43",
            "03881e45cdc5d0cf5e3a841cafe2ebdb8280e53e113c489e6847f7f45ca9c139",
        ),
    ];
    // When/Then: dense and recurrent identities are the pinned pair.
    let artifacts = manifest["artifacts"].as_array().ok_or("artifacts")?;
    for (id, repo, revision, sha) in pinned {
        let row = artifacts.iter().find(|row| row["id"] == id).ok_or(id)?;
        assert_eq!(row["repo"], repo, "{id}");
        assert_eq!(row["revision"], revision, "{id}");
        assert_eq!(row["sha256"], sha, "{id}");
        assert_eq!(row["files"].as_array().map(Vec::len), Some(1), "{id}");
    }
    assert_eq!(manifest["default_artifact_id"], "smollm2-q8-inference");
    Ok(())
}
