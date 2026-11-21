use super::*;
use serial_test::serial;
fn empty_catalog_guard() -> skippy_model_hf::remote_catalog::CatalogEntriesOverrideGuard {
    skippy_model_hf::remote_catalog::set_catalog_entries_for_test(Vec::new())
}

#[tokio::test]
#[serial]
async fn show_exact_model_accepts_hf_layer_package_ref() {
    let _catalog = empty_catalog_guard();
    let _probe = skippy_model_hf::remote_catalog::set_hf_model_file_probe_for_test(
        |repo, revision, file| {
            repo == "meshllm/gemma-4-E4B-it-Q4_K_M-layers"
                && revision == "main"
                && file == "model-package.json"
        },
    );

    let details = show_exact_model("hf://meshllm/gemma-4-E4B-it-Q4_K_M-layers")
        .await
        .expect("models show must accept the hf:// ref that serve and models download accept");

    assert_eq!(
        details.exact_ref,
        "hf://meshllm/gemma-4-E4B-it-Q4_K_M-layers"
    );
    assert_eq!(
        crate::models::formatters::model_kind_code(details.kind),
        "layer_package"
    );
}

#[tokio::test]
#[serial]
async fn show_exact_model_accepts_pinned_hf_layer_package_ref() {
    let _catalog = empty_catalog_guard();
    let _probe = skippy_model_hf::remote_catalog::set_hf_model_file_probe_for_test(
        |repo, revision, file| {
            repo == "meshllm/gemma-4-E4B-it-Q4_K_M-layers"
                && revision == "d7a5df3d3a142d44e1e5d65b205812e0c8f1deac"
                && file == "model-package.json"
        },
    );

    let details = show_exact_model(
        "hf://meshllm/gemma-4-E4B-it-Q4_K_M-layers@d7a5df3d3a142d44e1e5d65b205812e0c8f1deac",
    )
    .await
    .expect("models show must accept a pinned hf:// layer-package ref");

    assert_eq!(
        details.exact_ref,
        "hf://meshllm/gemma-4-E4B-it-Q4_K_M-layers@d7a5df3d3a142d44e1e5d65b205812e0c8f1deac"
    );
    assert_eq!(
        crate::models::formatters::model_kind_code(details.kind),
        "layer_package"
    );
}

#[tokio::test]
#[serial]
async fn show_exact_model_accepts_hf_layer_package_ref_with_slash_revision() {
    let _catalog = empty_catalog_guard();
    let _probe = skippy_model_hf::remote_catalog::set_hf_model_file_probe_for_test(
        |repo, revision, file| {
            repo == "meshllm/gemma-4-E4B-it-Q4_K_M-layers"
                && revision == "refs/pr/42"
                && file == "model-package.json"
        },
    );

    let details = show_exact_model("hf://meshllm/gemma-4-E4B-it-Q4_K_M-layers@refs/pr/42")
        .await
        .expect("models show must preserve slash-containing HF package revisions");

    assert_eq!(
        details.exact_ref,
        "hf://meshllm/gemma-4-E4B-it-Q4_K_M-layers@refs/pr/42"
    );
}

#[tokio::test]
#[serial]
async fn show_exact_model_rejects_hf_ref_without_a_package_manifest() {
    let _catalog = empty_catalog_guard();
    let _probe = skippy_model_hf::remote_catalog::set_hf_model_file_probe_for_test(|_, _, _| false);

    assert!(
        show_exact_model("hf://meshllm/not-a-layer-package")
            .await
            .is_err()
    );
}
