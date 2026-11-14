use super::*;

fn catalog_model(name: &str, repo: &str, file: &str) -> models::remote_catalog::RemoteCatalogModel {
    models::remote_catalog::RemoteCatalogModel {
        name: name.to_string(),
        file: file.to_string(),
        repo: repo.to_string(),
        revision: None,
        source_file: file.to_string(),
        size: None,
        description: None,
        draft: None,
        extra_files: Vec::new(),
        mmproj: None,
    }
}

fn loaded<'a>(served_name: &'a str, source: ModelLoadSource) -> LoadedModel<'a> {
    LoadedModel {
        served_name,
        source,
        metadata: None,
        path: None,
        package_ref: None,
        dflash: None,
        device: None,
    }
}

#[test]
fn direct_gguf_never_reports_the_file_name() {
    // The label grammar does not save us here: `resolved_model_name` hands
    // over the file *stem*, and a bare stem passes the grammar cleanly, so
    // `--gguf /home/you/acme-merger-finetune.gguf` would go out verbatim.
    // `analytics.md` promises a path cannot reach the wire; the source rule
    // is what holds it, even when the stem happens to match a catalog name.
    let path = Path::new("/home/you/acme-merger-finetune.gguf");
    let name = crate::runtime::local::resolved_model_name(path);
    assert_eq!(name, "acme-merger-finetune", "stem is what gets served");
    let matches = CatalogMatches {
        by_name: Some("acme-merger-finetune".to_string()),
        ..CatalogMatches::default()
    };
    assert_eq!(
        reportable_catalog_name(ModelLoadSource::DirectGguf, matches),
        None
    );
}

#[test]
fn a_direct_load_from_the_cache_reports_its_catalog_entry() {
    // The common path: a catalog model resolved to its file in the Hugging
    // Face cache and loaded directly. Before, this was `redacted` on every
    // row, which left `inference_request` with no model breakdown at all.
    let catalog = [
        catalog_model(
            "Qwen3-8B-Q4_K_M",
            "unsloth/Qwen3-8B-GGUF",
            "Qwen3-8B-Q4_K_M.gguf",
        ),
        catalog_model(
            "Qwen3-8B-Q8_0",
            "unsloth/Qwen3-8B-GGUF",
            "Qwen3-8B-Q8_0.gguf",
        ),
    ];
    let by_file = catalog_name_for_file(&catalog, "Unsloth/Qwen3-8B-GGUF", "Qwen3-8B-Q8_0.gguf");
    assert_eq!(by_file.as_deref(), Some("Qwen3-8B-Q8_0"));
    assert_eq!(
        reportable_catalog_name(
            ModelLoadSource::DirectGguf,
            CatalogMatches {
                by_file,
                ..CatalogMatches::default()
            }
        )
        .as_deref(),
        Some("Qwen3-8B-Q8_0")
    );
}

#[test]
fn a_repository_the_catalog_does_not_list_matches_nothing() {
    // A private repository downloaded with a token sits in the same cache as
    // a public one. Only a catalog listing makes it public.
    let catalog = [catalog_model(
        "Qwen3-8B-Q4_K_M",
        "unsloth/Qwen3-8B-GGUF",
        "Qwen3-8B-Q4_K_M.gguf",
    )];
    assert_eq!(
        catalog_name_for_file(&catalog, "acme/private-finetune", "Qwen3-8B-Q4_K_M.gguf"),
        None
    );
    assert_eq!(
        catalog_name_for_file(&catalog, "unsloth/Qwen3-8B-GGUF", "other.gguf"),
        None
    );
}

#[test]
fn a_catalog_package_is_named_by_its_variant() {
    let entries: Vec<skippy_model_resolver::CatalogEntry> = serde_json::from_value(serde_json::json!([{
        "schema_version": 1,
        "source_repo": "unsloth/Qwen3-8B-GGUF",
        "variants": {
            "Q4_K_M": {
                "source": {"repo": "unsloth/Qwen3-8B-GGUF", "file": "Qwen3-8B-Q4_K_M.gguf"},
                "curated": {"name": "Qwen3-8B-Q4_K_M"},
                "packages": [{"type": "layer-package", "repo": "meshllm/Qwen3-8B-Q4_K_M-layers"}]
            }
        }
    }]))
    .expect("catalog entry");
    assert_eq!(
        catalog_name_for_package_repo(&entries, "meshllm/qwen3-8b-q4_k_m-layers").as_deref(),
        Some("Qwen3-8B-Q4_K_M")
    );
    assert_eq!(
        catalog_name_for_package_repo(&entries, "acme/private-layers"),
        None
    );
}

#[test]
fn model_loaded_names_the_system_one_backend_from_the_architecture() {
    let meta = |architecture: &str| models::gguf::GgufCompactMeta {
        architecture: architecture.to_string(),
        ..Default::default()
    };
    assert_eq!(system_one_backend(Some(&meta("laya"))), Some("laya"));
    assert_eq!(
        system_one_backend(Some(&meta("diffusion-gemma"))),
        Some("openjev")
    );
    assert_eq!(system_one_backend(Some(&meta("qwen3"))), None);
    assert_eq!(system_one_backend(None), None);
}

#[test]
fn a_split_from_a_local_file_never_reports_its_name() {
    // `--gguf ./acme-merger-finetune.gguf --model acme-merger-finetune`
    // splits under the alias, which is the file stem again.
    let everything = || CatalogMatches {
        by_name: Some("acme-merger-finetune".to_string()),
        by_file: Some("acme-merger-finetune".to_string()),
        by_package: Some("acme-merger-finetune".to_string()),
    };
    assert_eq!(
        reportable_catalog_name(
            split_model_load_source("/home/you/acme-merger-finetune.gguf"),
            everything()
        ),
        None
    );
    assert_eq!(
        reportable_catalog_name(ModelLoadSource::WhisperFile, everything()),
        None
    );
    assert_eq!(
        reportable_catalog_name(
            split_model_load_source("hf://unsloth/Qwen3-8B-GGUF"),
            CatalogMatches {
                by_name: Some("Qwen3-8B-Q4_K_M".to_string()),
                ..CatalogMatches::default()
            }
        )
        .as_deref(),
        Some("Qwen3-8B-Q4_K_M")
    );
}

#[test]
fn a_configured_alias_is_never_reported() {
    // `server.alias = "acme-codename"` on a Hugging Face package: the serving
    // name is operator configuration, so it is never what goes out. With no
    // catalog match the model is `redacted`; with a package match it is the
    // catalog's own name.
    for source in [
        ModelLoadSource::LayerPackage,
        split_model_load_source("hf://unsloth/Qwen3-8B-GGUF"),
    ] {
        assert_eq!(
            reportable_catalog_name(source, CatalogMatches::default()),
            None
        );
        let identity = model_identity(&loaded("acme-codename", source), None);
        assert_eq!(identity.model.as_str(), "redacted");
    }
}

#[test]
fn a_catalog_name_that_fails_the_grammar_still_redacts() {
    let identity = model_identity(
        &loaded("qwen3-8b", ModelLoadSource::LayerPackage),
        Some("/etc/passwd"),
    );
    assert_eq!(identity.model.as_str(), "redacted");
}

#[test]
fn a_redacted_model_is_still_described_by_architecture_and_size() {
    let meta = models::gguf::GgufCompactMeta {
        architecture: "qwen3moe".to_string(),
        parameter_size: Some("30B-A3B".to_string()),
        expert_count: 128,
        ..Default::default()
    };
    let model = LoadedModel {
        metadata: Some(&meta),
        ..loaded("acme-merger-finetune", ModelLoadSource::DirectGguf)
    };
    let identity = model_identity(&model, None);

    assert_eq!(identity.model.as_str(), "redacted");
    assert_eq!(
        identity.architecture.as_ref().map(|label| label.as_str()),
        Some("qwen3moe")
    );
    assert_eq!(identity.parameters, Some("20-39b"));
    assert_eq!(identity.moe, Some(true));
    let rendered = format!(
        "{:?}",
        identity.apply(mesh_llm_analytics::Properties::new())
    );
    assert!(!rendered.contains("acme"), "{rendered}");
}

#[test]
fn a_model_loaded_onto_the_cpu_reports_the_cpu_backend() {
    // A CUDA device can be registered while the model runs on the CPU; the
    // selected device, not the registered ones, decides the backend.
    let model = LoadedModel {
        device: Some("CPU"),
        ..loaded("Qwen3-8B-Q4_K_M", ModelLoadSource::DirectGguf)
    };
    assert_eq!(model_identity(&model, None).backend, Some("cpu"));
    let unselected = loaded("Qwen3-8B-Q4_K_M", ModelLoadSource::DirectGguf);
    assert_eq!(model_identity(&unselected, None).backend, None);
}
