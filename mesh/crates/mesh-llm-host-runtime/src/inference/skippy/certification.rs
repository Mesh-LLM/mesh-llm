//! Mesh catalog/cache policy for Skippy package certification.
use super::materialization::StagePackageRef;
use anyhow::{Context, Result, bail};
pub use skippy_api::package::certification::{
    CertificationGateStatus, SkippyCertificationReport, SkippyCertificationRequest,
};

fn acquisition() -> skippy_api::package::acquisition::remote::PackageAcquisition {
    mesh_llm_skippy_adapter::package::acquisition::with_client(|| {
        crate::models::build_hf_api(false)
    })
}

pub async fn certify_layer_package(
    request: SkippyCertificationRequest,
) -> Result<SkippyCertificationReport> {
    let resolved = resolve_certification_package_ref(&request.model_ref)?;
    skippy_api::package::certification::certify_layer_package(
        request,
        resolved,
        acquisition(),
        mesh_llm_skippy_adapter::hash_cache::open_default(),
    )
    .await
}

pub fn resolve_certification_package_ref(input: &str) -> Result<String> {
    if let Ok(package_ref) = StagePackageRef::parse(input) {
        if let Some(package_ref) = package_ref.as_package_ref() {
            return Ok(package_ref);
        }
        bail!("direct GGUF inputs are not layer-package certification targets");
    }
    crate::models::remote_catalog::find_layer_package(input)
        .with_context(|| format!("no layer package found for {input:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn package_v2_certification_selects_two_stage_artifact_closures() {
        let root = tempfile::tempdir().unwrap();
        crate::inference::skippy::write_test_package_v2_fixture(
            root.path(),
            "fixture/llama-1b",
            &[
                (
                    "layer-00000",
                    "layers/layer-00000.gguf",
                    "blk.0.attn.weight",
                ),
                (
                    "layer-00001",
                    "layers/layer-00001.gguf",
                    "blk.1.attn.weight",
                ),
            ],
        )
        .unwrap();

        let report = certify_layer_package(SkippyCertificationRequest {
            model_ref: root.path().to_string_lossy().into_owned(),
            package_only: true,
            api_base: None,
            prompt: "Say ok.".into(),
            max_tokens: 2,
        })
        .await
        .unwrap();
        assert_eq!(report.status, CertificationGateStatus::Passed);
        let stages = report.materialized_stages;

        assert_eq!(stages.len(), 2);
        assert_eq!(stages[0].selected_part_count, 2);
        assert_eq!(stages[1].selected_part_count, 2);
        assert!(stages.iter().all(|stage| stage.verified_artifacts == 2));
        assert!(stages.iter().all(|stage| {
            let path = std::path::Path::new(&stage.materialized_path);
            path.is_file() && path.ends_with("shared/metadata.gguf")
        }));
    }

    #[tokio::test]
    async fn package_v2_certification_rejects_unclassified_non_layer_artifact() {
        let root = tempfile::tempdir().unwrap();
        crate::inference::skippy::write_test_package_v2_fixture(
            root.path(),
            "fixture/llama-1b",
            &[
                (
                    "renamed-embeddings",
                    "shared/renamed.gguf",
                    "token_embd.weight",
                ),
                (
                    "layer-00000",
                    "layers/layer-00000.gguf",
                    "blk.0.attn.weight",
                ),
                (
                    "layer-00001",
                    "layers/layer-00001.gguf",
                    "blk.1.attn.weight",
                ),
            ],
        )
        .unwrap();

        let error = certify_layer_package(SkippyCertificationRequest {
            model_ref: root.path().to_string_lossy().into_owned(),
            package_only: true,
            api_base: None,
            prompt: "Say ok.".into(),
            max_tokens: 2,
        })
        .await
        .unwrap_err()
        .to_string();

        assert!(error.contains("unsupported artifact path"), "{error}");
    }
}
