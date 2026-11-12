//! The real checkout: the checked-in registry projects to the checked-in
//! manifests, the frozen fixtures still match them, and the required smoke
//! identities stay pinned.

use crate::support::{Stage, TestResult, assert_streams, fixture_dir, repository_root, xtask};
use serde_json::Value;
use std::fs;

const FAMILY_ROSTER: &str = "ci/llama-canary/family-certified.json";
const MANIFESTS: &str = "ci/model-artifacts/manifests";
/// Suite manifests with a frozen fixture under `fixtures/models/manifests`.
const FROZEN: [&str; 11] = [
    "product-smoke",
    "scripted-binary-smoke",
    "sdk-smoke",
    "hf-download-smoke",
    "openai-smoke",
    "skippy-correctness",
    "safetensors-runtime-smoke",
    "skippy-ci-smoke",
    "skippy-parity",
    "competitive-benchmark",
    "radix-cache",
];

/// The family roster and every checked-in suite manifest, so a suite the
/// registry declares is compared without listing it here.
fn checked_in_outputs() -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut manifests = fs::read_dir(repository_root().join(MANIFESTS))?
        .map(|entry| Ok(entry?.file_name().to_string_lossy().into_owned()))
        .collect::<Result<Vec<_>, std::io::Error>>()?;
    manifests.sort();
    Ok(std::iter::once(FAMILY_ROSTER.to_owned())
        .chain(manifests.iter().map(|name| format!("{MANIFESTS}/{name}")))
        .collect())
}

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
    // Given: a copy of the real registry, its derived lists and nothing else.
    let root = repository_root();
    let stage = Stage::new("live")?;
    stage.real_registry(&root)?;
    // When: the generator writes a fresh projection.
    assert_streams(
        "regenerate",
        &xtask(stage.path(), &["models", "generate"])?,
        0,
        "",
        "",
    );
    // Then: every output equals the checked-in bytes.
    for path in checked_in_outputs()? {
        assert_eq!(
            stage.read(&path)?,
            fs::read_to_string(root.join(&path))?,
            "{path}"
        );
    }
    Ok(())
}

#[test]
fn migration_models_frozen_manifests_match_the_checkout() -> TestResult {
    // Given/When: the frozen resolver inputs and the checked-in manifests.
    let root = repository_root();
    for suite in FROZEN {
        let name = format!("{suite}.json");
        // Then: frozen artifact payloads remain unchanged apart from the
        // registry digest and Qwen's added System One membership.
        let mut frozen: Value =
            serde_json::from_slice(&fs::read(fixture_dir().join("manifests").join(&name))?)?;
        let mut current: Value =
            serde_json::from_slice(&fs::read(root.join(MANIFESTS).join(&name))?)?;
        if name == "product-smoke.json" {
            current["artifacts"]
                .as_array_mut()
                .ok_or("product artifacts")?
                .retain(|row| row["id"] != "family-laya-multilingual");
        }
        if name == "skippy-correctness.json" {
            // The split verify-rollback test added its own recurrent fixture to
            // this suite; the dense fixture stays the suite default.
            current["artifacts"]
                .as_array_mut()
                .ok_or("correctness artifacts")?
                .retain(|row| row["id"] != "qwen35-q4-rollback-correctness");
            assert_eq!(current["default_artifact_id"], "qwen3-q8-correctness");
            current
                .as_object_mut()
                .ok_or("correctness manifest")?
                .remove("default_artifact_id");
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
                if row["id"] == "smollm2-q8-inference" {
                    row["suites"]
                        .as_array_mut()
                        .ok_or("frozen SmolLM suites")?
                        .push(Value::from("skippy-cpp-api"));
                    row["cadences"]
                        .as_array_mut()
                        .ok_or("frozen SmolLM cadences")?
                        .push(Value::from("manual"));
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
