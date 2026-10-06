use crate::snapshot_promotion::fixtures;
#[path = "source_tests.rs"]
mod source_tests;
use super::*;
fn root() -> Vec<u8> {
    fixtures::manifest("metadata.gguf", 4, "a".repeat(64))
}
#[test]
fn layer_job_projection_uses_existing_root_identity_and_exact_source_binding() {
    let (_, row) = project(&root(), "fixture/model", &"b".repeat(40), true).unwrap();
    assert_eq!(row.layer_count, 1);
    assert_eq!(row.total_bytes, 4);
    assert_eq!(row.artifacts.len(), 1);
    assert!(row.experimental);
    assert!(row.scope.contains("no artifact-byte"));
    assert!(project(&root(), "other/model", &"b".repeat(40), false).is_err());
    let mut value: serde_json::Value = serde_json::from_slice(&root()).unwrap();
    value["layer_count"] = serde_json::json!(2);
    assert!(
        project(
            &serde_json::to_vec(&value).unwrap(),
            "fixture/model",
            &"b".repeat(40),
            false
        )
        .is_err()
    );
}
#[test]
fn layer_job_projection_refuses_catalog_overflow_duplicates_and_empty_sizes() {
    let initial: PackageManifest = serde_json::from_slice(&root()).unwrap();
    for mode in 0..3 {
        let mut manifest = initial.clone();
        let mut second = manifest.artifact_catalog.entries[0].clone();
        second.id = "other".into();
        second.path = "other.gguf".into();
        match mode {
            0 => {
                manifest.artifact_catalog.entries[0].byte_size = u64::MAX;
                second.byte_size = 1;
            }
            1 => second.path = manifest.artifact_catalog.entries[0].path.clone(),
            _ => second.byte_size = 0,
        }
        manifest.artifact_catalog.entries.push(second);
        manifest.package_id = manifest.computed_package_id().unwrap();
        assert!(
            project(
                &serde_json::to_vec(&manifest).unwrap(),
                "fixture/model",
                &"b".repeat(40),
                false
            )
            .is_err()
        );
    }
}
#[test]
fn layer_job_card_preserves_experimental_boundary_and_safe_structured_values() {
    let (mut manifest, projection) =
        project(&root(), "fixture/model", &"b".repeat(40), true).unwrap();
    manifest.format = "<script>|`bad`".into();
    let card = super::card::render(
        &manifest,
        &projection,
        "fixture/package",
        "text-generation",
        Some("apache-2.0"),
    )
    .unwrap();
    assert!(card.contains("license: \"apache-2.0\""));
    assert!(card.contains("does not promote"));
    assert!(card.contains("declared artifact identities"));
    assert!(card.contains("metadata.gguf"));
    assert!(card.contains(&projection.manifest_sha256));
    assert!(!card.contains("<script>"));
    assert!(!card.contains("|`bad`"));
    for title in [
        "Highlights",
        "Model Overview",
        "Recommended Use",
        "Quickstart",
        "Package Variant",
        "What Is Included",
        "Validation",
    ] {
        assert!(card.contains(&format!("## {title}")));
    }
    assert!(
        super::card::render(
            &manifest,
            &projection,
            "bad/repo/extra",
            "text-generation",
            None
        )
        .is_err()
    );
    assert!(
        super::card::render(
            &manifest,
            &projection,
            "fixture/package",
            "text-generation",
            Some("bad\nlicense")
        )
        .is_err()
    );
}

#[test]
fn layer_complete_card_preserves_license_fallback_metadata_artifacts_speculation_and_experimental_boundary()
 {
    let (mut manifest, projection) =
        project(&root(), "fixture/model", &"b".repeat(40), true).unwrap();
    manifest.source_model.distribution_id = Some("Qwen3-32B-UD-Q4_K_XL".into());
    manifest
        .model_metadata
        .insert("activation_width".into(), serde_json::json!(4096));
    let license = License {
        value: Some("mit".into()),
        repo: Some("base/first".into()),
        revision: Some("c".repeat(40)),
        warning: None,
    };
    let card = super::card::render_complete(
        &manifest,
        &projection,
        super::card::CardContext {
            target: "fixture/package",
            pipeline: "text-generation",
            source_file: "UD-Q4_K_XL/model-00001-of-00002.gguf",
            mesh_ref: "pinned-build",
            license: &license,
        },
    )
    .unwrap();
    for expected in [
        "license: \"mit\"",
        "base_model:",
        "fixture/model",
        "base/first",
        "Qwen3",
        "32B",
        "UD-Q4_K_XL",
        "activation_width",
        "4096",
        "metadata.gguf",
        "Manifest SHA-256",
        "pinned-build",
        "Tensor Catalog",
        "api/status",
        "/v1/chat/completions",
        "does not promote",
        "pipeline_tag: \"text-generation\"",
        "- experimental",
        "- openai-compatible",
        "Distributed GGUF inference package for Mesh LLM",
        "## Model Overview",
        "## Highlights",
        "## Recommended Use",
        "## Quickstart",
        "## Package Variant",
        "## What Is Included",
        "## Validation",
        "https://www.meshllm.cloud",
        "mesh-llm serve --model \"fixture/package\" --split",
    ] {
        assert!(card.contains(expected), "missing {expected}");
    }
    assert!(card.contains(&"c".repeat(40)));
    let absent = License {
        value: None,
        repo: None,
        revision: None,
        warning: Some("upstream license metadata unavailable; no license inferred"),
    };
    let card = super::card::render_complete(
        &manifest,
        &projection,
        super::card::CardContext {
            target: "fixture/package",
            pipeline: "text-generation",
            source_file: "model.gguf",
            mesh_ref: "pinned-build",
            license: &absent,
        },
    )
    .unwrap();
    assert!(!card.contains("license: "));
    assert!(card.contains("no license inferred"));
}
