//! Preserve source metadata without changing content-addressed split routing IDs.
use super::startup_models::StartupModelPlan;
use crate::mesh;
use std::path::Path;

pub(super) fn launch_source(source: &Path) -> String {
    if source.is_file() {
        source
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned()
    } else {
        let source = source.to_string_lossy();
        sanitized_model_url(&source).unwrap_or_else(|| source.into_owned())
    }
}

/// Remove credentials and request-scoped URL data before source metadata is
/// gossiped to peers. The stable URL path still lets the identity parser
/// recover Hugging Face coordinates or a direct-download filename.
fn sanitized_model_url(source: &str) -> Option<String> {
    let mut url = url::Url::parse(source).ok()?;
    if !matches!(url.scheme(), "http" | "https") {
        return None;
    }
    url.set_username("").ok()?;
    url.set_password(None).ok()?;
    url.set_query(None);
    url.set_fragment(None);
    Some(url.into())
}

pub(super) async fn advertise_startup_sources(node: &mesh::Node, models: &[StartupModelPlan]) {
    for (index, model) in models.iter().enumerate() {
        let descriptors = mesh::infer_served_model_descriptors(
            &model.declared_ref,
            std::slice::from_ref(&model.declared_ref),
            Some(&model.model_source),
            Some(&model.resolved_path),
        );
        for mut descriptor in descriptors {
            descriptor.identity.is_primary = index == 0;
            node.upsert_served_model_descriptor(descriptor).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_sources_do_not_publish_parent_directories() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("Gemma.gguf");
        std::fs::write(&path, b"GGUF").unwrap();
        assert_eq!(launch_source(&path), "Gemma.gguf");
        assert_eq!(
            launch_source(Path::new("unsloth/Gemma-GGUF@main:Q4_K_M")),
            "unsloth/Gemma-GGUF@main:Q4_K_M"
        );
    }

    #[test]
    fn remote_sources_do_not_publish_url_credentials_or_query_tokens() {
        assert_eq!(
            launch_source(Path::new(
                "https://user:password@example.com/private/model.gguf?token=secret#download"
            )),
            "https://example.com/private/model.gguf"
        );
    }
}

#[cfg(test)]
mod advertisement_tests {
    use super::*;
    use crate::runtime::startup_models::StartupModelPlan;

    fn plan(name: &str, source: &str) -> StartupModelPlan {
        StartupModelPlan {
            declared_ref: name.into(),
            model_source: source.into(),
            config_model_id: None,
            resolved_path: Path::new("/not-used/model.gguf").into(),
            preindexed_split_package: None,
            mmproj_path: None,
            ctx_size: None,
            gpu_id: None,
            pinned_gpu: None,
            parallel: None,
            cache_type_k: None,
            cache_type_v: None,
            n_batch: None,
            n_ubatch: None,
            flash_attention: skippy_protocol::FlashAttentionType::Auto,
            local_source_required: true,
            profile: String::new(),
        }
    }

    #[tokio::test]
    async fn source_metadata_for_primary_and_secondary_survives_serving_refresh() {
        let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
            .await
            .unwrap();
        let plans = [
            plan("local-gguf/sha256-a", "unsloth/Gemma-GGUF@main:Q4_K_M"),
            plan("local-gguf/sha256-b", "OtherModel.gguf"),
        ];
        let names: Vec<_> = plans.iter().map(|plan| plan.declared_ref.clone()).collect();
        node.set_serving_models(names.clone()).await;
        advertise_startup_sources(&node, &plans).await;
        node.set_serving_models(names).await;
        let descriptors = node.served_model_descriptors().await;
        assert_eq!(
            descriptors[0].identity.canonical_ref.as_deref(),
            Some("unsloth/Gemma-GGUF@main:Q4_K_M")
        );
        assert_eq!(
            descriptors[1].identity.local_file_name.as_deref(),
            Some("OtherModel.gguf")
        );
        assert!(descriptors[0].identity.is_primary);
        assert!(!descriptors[1].identity.is_primary);
    }
}
