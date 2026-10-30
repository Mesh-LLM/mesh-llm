use super::*;

#[test]
#[serial_test::serial]
fn unresolved_layer_package_ref_surfaces_actionable_diagnostic() {
    use crate::runtime_event_api::state_projection;
    use crate::runtime_events::engine::RuntimeEventEngine;
    use crate::runtime_events::{clear_runtime_event_engine, install_runtime_event_engine};

    clear_runtime_event_engine();
    let engine = RuntimeEventEngine::new();
    install_runtime_event_engine(engine.clone());

    // Mirrors the reported shape: the catalog lists the variant but its
    // `packages` list is empty, and the Hugging Face probe reports no
    // `model-package.json`, so layer-package resolution comes up empty.
    let mut variants = std::collections::HashMap::new();
    variants.insert(
        "RegressionUnavailable-Q4_K_M".to_string(),
        models::remote_catalog::CatalogVariant {
            source: models::remote_catalog::CatalogSource {
                repo: "mesh-test/regression-unavailable".to_string(),
                revision: Some("main".to_string()),
                file: Some("RegressionUnavailable-Q4_K_M.gguf".to_string()),
            },
            curated: models::remote_catalog::CatalogCurated {
                name: "Regression Unavailable Q4_K_M".to_string(),
                size: None,
                description: None,
                draft: None,
                moe: None,
                extra_files: Vec::new(),
                mmproj: None,
            },
            packages: Vec::new(),
        },
    );
    let _catalog_guard = models::remote_catalog::set_catalog_entries_for_test(vec![
        models::remote_catalog::CatalogEntry {
            schema_version: 1,
            source_repo: "mesh-test/regression-unavailable".to_string(),
            variants,
        },
    ]);
    let _probe_guard = models::remote_catalog::set_hf_model_file_probe_for_test(|_, _, _| false);

    let unresolved_ref = "meshllm/regression-unavailable-layers";
    let resolved = resolve_split_layer_package(unresolved_ref, Path::new(unresolved_ref));
    engine.drain();

    assert_eq!(
        resolved, None,
        "an unresolvable layer ref must keep resolving empty"
    );

    let node =
        serde_json::to_value(&state_projection::build(&engine).node).expect("serializable node");
    let warnings = node["diagnostics"]["active_warnings"]
        .as_array()
        .unwrap_or_else(|| panic!("no active warnings surfaced for {unresolved_ref}: {node}"));
    let matching = warnings
        .iter()
        .find(|warning| {
            warning["reason_code"]
                == serde_json::Value::String("model_ref_unresolvable".to_string())
        })
        .unwrap_or_else(|| panic!("expected a model_ref_unresolvable warning, got {warnings:?}"));

    let summary = matching["summary"].as_str().unwrap_or_default();
    assert!(
        summary.contains(unresolved_ref),
        "the surfaced diagnostic must name the unresolved ref, got {summary:?}"
    );

    clear_runtime_event_engine();
}

#[test]
#[serial_test::serial]
fn monolithic_hf_ref_without_layer_package_stays_silent() {
    use crate::runtime_event_api::state_projection;
    use crate::runtime_events::engine::RuntimeEventEngine;
    use crate::runtime_events::{clear_runtime_event_engine, install_runtime_event_engine};

    clear_runtime_event_engine();
    let engine = RuntimeEventEngine::new();
    install_runtime_event_engine(engine.clone());

    let _catalog_guard = models::remote_catalog::set_catalog_entries_for_test(Vec::new());
    let _probe_guard = models::remote_catalog::set_hf_model_file_probe_for_test(|_, _, _| false);

    // Every healthy node starts monolithic Hugging Face models that have no
    // layer package. That must not be reported as a fault.
    let monolithic = "unsloth/Qwen3-8B-GGUF";
    assert_eq!(
        resolve_split_layer_package(monolithic, Path::new(monolithic)),
        None
    );
    engine.drain();

    let node =
        serde_json::to_value(&state_projection::build(&engine).node).expect("serializable node");
    let warnings = node["diagnostics"]["active_warnings"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    assert!(
        warnings
            .iter()
            .all(|warning| warning["reason_code"] != serde_json::json!("model_ref_unresolvable")),
        "a monolithic ref must not raise the unresolved-ref warning: {warnings:?}"
    );

    clear_runtime_event_engine();
}
