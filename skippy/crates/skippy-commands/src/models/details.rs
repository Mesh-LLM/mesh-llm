//! Command presentation for exact model details.
use anyhow::{Context, Result, bail};
use skippy_model_artifact::capabilities::ModelCapabilities;
pub(super) use skippy_model_hf::details::ModelDetails;
use skippy_model_hf::remote_catalog;

/// Kind label for an `hf://` layer package. `model_kind_code` maps it to its
/// own code so `models show` does not go looking for GGUF quant variants.
const LAYER_PACKAGE_KIND: &str = "📦 Layer package";

/// Details for an `hf://` layer-package ref, the form `serve --model` and
/// `models download` accept. `models show` is how a ref gets checked without
/// downloading it, so the package manifest has to be present at the revision.
fn show_layer_package(input: &str) -> Result<ModelDetails> {
    let package_ref = skippy_api::package::acquisition::StagePackageRef::parse(input)
        .with_context(|| format!("Invalid layer-package ref {input}"))?;
    let skippy_api::package::acquisition::StagePackageRef::HuggingFacePackage { repo, revision } =
        package_ref
    else {
        bail!("Only Hugging Face layer packages can be shown: {input}");
    };
    let exact_ref =
        remote_catalog::find_huggingface_layer_package_at_revision(&repo, revision.as_deref())
            .with_context(|| format!("No layer package manifest found for {input}"))?;
    Ok(ModelDetails {
        display_name: repo.rsplit('/').next().unwrap_or(&repo).to_string(),
        exact_ref,
        source: "huggingface",
        kind: LAYER_PACKAGE_KIND,
        download_url: format!(
            "https://huggingface.co/{repo}/tree/{}",
            revision.as_deref().unwrap_or("main")
        ),
        size_label: None,
        description: Some(
            "Layer package. Serve it with `serve --model` or fetch it with `models download`."
                .to_string(),
        ),
        draft: None,
        capabilities: ModelCapabilities::default(),
    })
}

pub async fn show_exact_model(input: &str) -> Result<ModelDetails> {
    if input.starts_with("hf://") {
        return show_layer_package(input);
    }
    skippy_model_hf::details::show_exact_model(input, super::output::fit_budget_bytes()).await
}

pub async fn show_model_variants_with_progress<F>(
    input: &str,
    progress: F,
) -> Result<Option<Vec<ModelDetails>>>
where
    F: FnMut(skippy_model_hf::details::ShowVariantsProgress),
{
    skippy_model_hf::details::show_model_variants_with_progress(
        input,
        progress,
        super::output::fit_budget_bytes(),
    )
    .await
}

#[cfg(test)]
mod tests;
