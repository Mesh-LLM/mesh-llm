use super::*;
use std::collections::HashMap;

use serial_test::serial;

#[test]
#[serial]
fn refresh_backoff_suppresses_then_clears() {
    clear_refresh_backoff();
    assert!(!refresh_in_backoff(), "no backoff initially");

    set_refresh_backoff();
    assert!(refresh_in_backoff(), "backoff active after a failure");

    clear_refresh_backoff();
    assert!(!refresh_in_backoff(), "backoff cleared after success");
}

/// Hits the network: verifies that the live meshllm/catalog dataset
/// downloads successfully with the patched hf-hub (redirect Content-Length
/// no longer mistaken for the file size). Run with:
///   cargo test -p skippy-model-hf refresh_catalog_live -- --ignored --nocapture
#[test]
#[ignore = "network: downloads the live meshllm/catalog dataset"]
#[serial]
fn refresh_catalog_live() {
    refresh_catalog().expect("live catalog refresh should succeed");
    let entries = catalog_entries().expect("catalog entries loaded");
    assert!(!entries.is_empty(), "expected at least one catalog entry");
}

#[test]
fn deserializes_catalog_entry() {
    let json = r#"{
        "schema_version": 1,
        "source_repo": "unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF",
        "variants": {
            "Qwen3-Coder-480B-A35B-Instruct-UD-Q4_K_XL": {
                "source": { "repo": "unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF", "revision": "main", "file": "Qwen3-Coder-480B-A35B-Instruct-UD-Q4_K_XL.gguf" },
                "curated": { "name": "Qwen3 Coder 480B Q4_K_XL", "size": "294GB", "description": "Large MoE coding model", "draft": "Qwen3-Coder-Draft-Q4_K_M", "moe": "480B/35B", "extra_files": [], "mmproj": { "file": "mmproj-BF16.gguf", "repo": "unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF", "revision": "main" } },
                "packages": [
                    { "type": "layer-package", "repo": "meshllm/Qwen3-Coder-480B-A35B-Instruct-UD-Q4_K_XL-layers", "layer_count": 62, "total_bytes": 315680000000 }
                ]
            }
        }
    }"#;

    let entry: CatalogEntry = serde_json::from_str(json).unwrap();
    assert_eq!(entry.schema_version, 1);
    assert_eq!(
        entry.source_repo,
        "unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF"
    );
    assert_eq!(entry.variants.len(), 1);

    let variant = entry
        .variants
        .get("Qwen3-Coder-480B-A35B-Instruct-UD-Q4_K_XL")
        .unwrap();
    assert_eq!(variant.curated.name, "Qwen3 Coder 480B Q4_K_XL");
    assert_eq!(
        variant.curated.draft.as_deref(),
        Some("Qwen3-Coder-Draft-Q4_K_M")
    );
    assert_eq!(
        variant
            .curated
            .moe
            .as_ref()
            .and_then(|value| value.as_str()),
        Some("480B/35B")
    );
    assert!(matches!(
        variant.curated.mmproj.as_ref(),
        Some(CatalogSidecar::Asset(asset))
            if asset.file == "mmproj-BF16.gguf"
                && asset.repo == "unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF"
                && asset.revision.as_deref() == Some("main")
    ));
    assert_eq!(variant.packages.len(), 1);
    assert_eq!(variant.packages[0].package_type, "layer-package");
    assert_eq!(
        variant.packages[0].repo,
        "meshllm/Qwen3-Coder-480B-A35B-Instruct-UD-Q4_K_XL-layers"
    );
    assert_eq!(variant.packages[0].layer_count, Some(62));
}

#[test]
fn catalog_cache_dir_uses_hf_home() {
    // Just verify it returns a path (env-dependent)
    let dir = catalog_cache_dir();
    assert!(!dir.as_os_str().is_empty());
}

fn test_variant(curated_name: &str, repo: &str, package_repos: &[&str]) -> CatalogVariant {
    CatalogVariant {
        source: CatalogSource {
            repo: repo.to_string(),
            revision: Some("main".to_string()),
            file: Some(format!("{curated_name}.gguf")),
        },
        curated: CatalogCurated {
            name: curated_name.to_string(),
            size: None,
            description: None,
            draft: None,
            moe: None,
            extra_files: Vec::new(),
            mmproj: None,
        },
        packages: package_repos
            .iter()
            .map(|repo| CatalogPackage {
                package_type: "layer-package".to_string(),
                repo: (*repo).to_string(),
                layer_count: None,
                total_bytes: None,
            })
            .collect(),
    }
}

#[test]
fn asset_download_refs_preserve_revision_and_exact_source_file() {
    for revision in [None, Some("main"), Some("abc123")] {
        for source_file in ["mmproj-BF16.gguf", "vision/projectors/mmproj-F16.gguf"] {
            let asset = RemoteCatalogAsset {
                file: "display-projector.gguf".to_string(),
                repo: "example/vision-model".to_string(),
                revision: revision.map(str::to_string),
                source_file: source_file.to_string(),
            };
            let download_ref = asset.download_ref();
            let parsed = skippy_model_ref::ModelRef::parse(&download_ref)
                .expect("catalog asset download ref must select an exact file");
            assert_eq!(
                (parsed.repo, parsed.revision, parsed.selector.unwrap()),
                (
                    asset.repo.clone(),
                    asset.revision.clone(),
                    asset.source_file.clone(),
                ),
                "download ref {download_ref}"
            );
        }
    }
}

#[test]
fn catalog_model_exact_ref_retains_quant_selector() {
    for (revision, expected_ref) in [
        (None, "example/vision-model:Q4_K_M"),
        (Some("abc123"), "example/vision-model@abc123:Q4_K_M"),
    ] {
        let model = RemoteCatalogModel {
            name: "Vision Q4".to_string(),
            file: "vision-Q4_K_M.gguf".to_string(),
            repo: "example/vision-model".to_string(),
            revision: revision.map(str::to_string),
            source_file: "vision-Q4_K_M.gguf".to_string(),
            size: None,
            description: None,
            draft: None,
            extra_files: Vec::new(),
            mmproj: None,
        };
        assert_eq!(model.exact_ref(), expected_ref);
        let asset_ref = model.source_asset().download_ref();
        let parsed = skippy_model_ref::ModelRef::parse(&asset_ref).unwrap();
        assert_eq!(parsed.revision.as_deref(), revision);
        assert_eq!(parsed.selector.as_deref(), Some(model.source_file.as_str()));
    }
}

#[test]
fn remote_models_preserve_draft_and_structured_mmproj() {
    let mut variant = test_variant("Vision Draft", "example/vision-source", &[]);
    variant.curated.draft = Some("Vision-Draft-Q4_K_M".to_string());
    variant.curated.mmproj = Some(CatalogSidecar::Asset(CatalogSidecarAssetRef {
        file: "mmproj-BF16.gguf".to_string(),
        repo: "example/vision-source".to_string(),
        revision: Some("main".to_string()),
        source_file: None,
    }));

    let entry = CatalogEntry {
        schema_version: 1,
        source_repo: "example/vision-source".to_string(),
        variants: HashMap::from([("vision-q4".to_string(), variant)]),
    };

    let models = remote_models_from_entry(&entry).unwrap();

    assert_eq!(models.len(), 1);
    assert_eq!(models[0].draft.as_deref(), Some("Vision-Draft-Q4_K_M"));
    assert_eq!(
        models[0].mmproj,
        Some(RemoteCatalogAsset {
            file: "mmproj-BF16.gguf".to_string(),
            repo: "example/vision-source".to_string(),
            revision: Some("main".to_string()),
            source_file: "mmproj-BF16.gguf".to_string(),
        })
    );
}

#[test]
#[serial]
fn layer_package_lookup_uses_deterministic_variant_and_package_order() {
    let previous = CATALOG_ENTRIES.write().unwrap().take();
    let mut variants = HashMap::new();
    variants.insert(
        "z-variant".to_string(),
        test_variant(
            "Shared Match Z",
            "example/shared-source",
            &["meshllm/z-package"],
        ),
    );
    variants.insert(
        "a-variant".to_string(),
        test_variant(
            "Shared Match A",
            "example/shared-source",
            &["meshllm/b-package", "meshllm/a-package"],
        ),
    );
    *CATALOG_ENTRIES.write().unwrap() = Some(vec![CatalogEntry {
        schema_version: 1,
        source_repo: "example/shared-source".to_string(),
        variants,
    }]);

    assert_eq!(
        find_layer_package("shared"),
        Some("hf://meshllm/a-package".to_string())
    );

    *CATALOG_ENTRIES.write().unwrap() = previous;
}

#[test]
#[serial]
fn layer_package_lookup_matches_exact_repo_selector_refs() {
    let previous = CATALOG_ENTRIES.write().unwrap().take();
    let mut variants = HashMap::new();
    variants.insert(
        "Qwen3-8B-Q4_K_M".to_string(),
        test_variant(
            "Qwen3 8B Q4",
            "unsloth/Qwen3-8B-GGUF",
            &["meshllm/Qwen3-8B-Q4_K_M-layers"],
        ),
    );
    *CATALOG_ENTRIES.write().unwrap() = Some(vec![CatalogEntry {
        schema_version: 1,
        source_repo: "unsloth/Qwen3-8B-GGUF".to_string(),
        variants,
    }]);

    assert_eq!(
        find_layer_package("unsloth/Qwen3-8B-GGUF:Q4_K_M"),
        Some("hf://meshllm/Qwen3-8B-Q4_K_M-layers".to_string())
    );

    *CATALOG_ENTRIES.write().unwrap() = previous;
}

#[test]
#[serial]
fn layer_package_lookup_matches_exact_package_repo_refs() {
    let previous = CATALOG_ENTRIES.write().unwrap().take();
    let mut variants = HashMap::new();
    variants.insert(
        "Qwen3-8B-Q4_K_M".to_string(),
        test_variant(
            "Qwen3 8B Q4",
            "unsloth/Qwen3-8B-GGUF",
            &["meshllm/Qwen3-8B-Q4_K_M-layers"],
        ),
    );
    *CATALOG_ENTRIES.write().unwrap() = Some(vec![CatalogEntry {
        schema_version: 1,
        source_repo: "unsloth/Qwen3-8B-GGUF".to_string(),
        variants,
    }]);

    assert_eq!(
        find_layer_package("meshllm/Qwen3-8B-Q4_K_M-layers"),
        Some("hf://meshllm/Qwen3-8B-Q4_K_M-layers".to_string())
    );

    *CATALOG_ENTRIES.write().unwrap() = previous;
}

#[test]
#[serial]
fn hf_layer_package_probe_requires_manifest_not_repo_name() {
    let _probe_guard = set_hf_model_file_probe_for_test(|repo, revision, file| {
        repo == "meshllm/arbitrary-package-name"
            && revision == "main"
            && file == "model-package.json"
    });

    assert_eq!(
        find_huggingface_layer_package("meshllm/arbitrary-package-name"),
        Some("hf://meshllm/arbitrary-package-name".to_string())
    );
    assert_eq!(
        find_huggingface_layer_package("meshllm/arbitrary-package-name:Q4_K_M"),
        None
    );
    assert_eq!(
        find_huggingface_layer_package("meshllm/package-name-layers"),
        None
    );
}

#[test]
#[serial]
fn hf_layer_package_probe_preserves_explicit_revision() {
    let _probe_guard = set_hf_model_file_probe_for_test(|repo, revision, file| {
        repo == "meshllm/custom-package" && revision == "abc123" && file == "model-package.json"
    });

    assert_eq!(
        find_huggingface_layer_package("meshllm/custom-package@abc123"),
        Some("hf://meshllm/custom-package@abc123".to_string())
    );
}

#[test]
fn parse_entries_recursive_uses_sorted_directory_order() {
    let temp = tempfile::tempdir().unwrap();
    let z_dir = temp.path().join("z");
    let a_dir = temp.path().join("a");
    fs::create_dir_all(&z_dir).unwrap();
    fs::create_dir_all(&a_dir).unwrap();
    fs::write(
        z_dir.join("entry.json"),
        r#"{
            "schema_version": 1,
            "source_repo": "z/source",
            "variants": {}
        }"#,
    )
    .unwrap();
    fs::write(
        a_dir.join("entry.json"),
        r#"{
            "schema_version": 1,
            "source_repo": "a/source",
            "variants": {}
        }"#,
    )
    .unwrap();

    let entries = parse_entries_recursive(temp.path()).unwrap();
    let repos: Vec<_> = entries
        .iter()
        .map(|entry| entry.source_repo.as_str())
        .collect();
    assert_eq!(repos, vec!["a/source", "z/source"]);
}

#[test]
fn prune_stale_catalog_entry_files_removes_deleted_upstream_entries() {
    let temp = tempfile::tempdir().unwrap();
    let entries_dir = temp.path().join("entries");
    fs::create_dir_all(entries_dir.join("current")).unwrap();
    fs::create_dir_all(entries_dir.join("removed")).unwrap();
    let current = entries_dir.join("current/entry.json");
    let stale = entries_dir.join("removed/entry.json");
    fs::write(&current, b"{}").unwrap();
    fs::write(&stale, b"{}").unwrap();

    prune_stale_catalog_entry_files(temp.path(), &["entries/current/entry.json".to_string()])
        .unwrap();

    assert!(current.is_file());
    assert!(!stale.exists());
}

#[test]
fn catalog_entry_cache_path_rejects_paths_outside_entries_dir() {
    let temp = tempfile::tempdir().unwrap();

    for entry_file in [
        "entries/../../outside.json",
        "entries/../outside.json",
        "/entries/model.json",
        "other/model.json",
        "entries",
        "entries/model.txt",
    ] {
        assert!(
            catalog_entry_cache_path(temp.path(), entry_file).is_err(),
            "expected {entry_file} to be rejected"
        );
    }

    assert_eq!(
        catalog_entry_cache_path(temp.path(), "entries/org/model.json").unwrap(),
        temp.path().join("entries/org/model.json")
    );
}

#[test]
fn malformed_catalog_sidecars_fail_validation() {
    let mut variants = HashMap::new();
    let mut variant = test_variant("Broken", "example/source", &[]);
    variant.curated.extra_files = vec![serde_json::json!({
        "file": "tokenizer.json"
    })];
    variants.insert("broken".to_string(), variant);

    let entry = CatalogEntry {
        schema_version: 1,
        source_repo: "example/source".to_string(),
        variants,
    };

    assert!(remote_models_from_entry(&entry).is_err());
}

#[test]
#[serial]
fn stale_check_returns_true_for_nonexistent() {
    let prev = std::env::var_os("HF_HOME");
    // SAFETY: the enclosing test contract is `#[serial]`, so this process
    // environment mutation cannot race another test.
    unsafe { std::env::set_var("HF_HOME", "/tmp/meshllm-test-nonexistent-dir-xyz") };
    let result = is_catalog_stale();
    match prev {
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        Some(val) => unsafe { std::env::set_var("HF_HOME", val) },
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        None => unsafe { std::env::remove_var("HF_HOME") },
    }
    assert!(result);
}

#[test]
#[serial]
fn stale_check_uses_last_refresh_marker() {
    let prev = std::env::var_os("HF_HOME");
    let temp = std::env::temp_dir().join(format!(
        "meshllm-catalog-stale-marker-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    // SAFETY: the enclosing test contract is `#[serial]`, so this process
    // environment mutation cannot race another test.
    unsafe { std::env::set_var("HF_HOME", &temp) };

    let entries_dir = catalog_cache_dir().join("entries");
    fs::create_dir_all(&entries_dir).unwrap();
    assert!(is_catalog_stale());

    fs::File::create(entries_dir.join(".last_refresh")).unwrap();
    assert!(!is_catalog_stale());

    let _ = fs::remove_dir_all(&temp);
    match prev {
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        Some(val) => unsafe { std::env::set_var("HF_HOME", val) },
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        None => unsafe { std::env::remove_var("HF_HOME") },
    }
}
