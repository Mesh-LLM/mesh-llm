use super::*;

// Pure model-identity helpers moved to `mesh-llm-membership`; re-exported here
// so the host prelude (`use model_identity::*`) keeps exposing them to sibling
// modules and tests unchanged. The two functions below stay in host because
// they reach `skippy_model_ref::ModelRef` (a Skippy crate) and
// `ModelCapabilities` — the remaining Skippy-boundary deps of this module.
pub(crate) use mesh_llm_membership::{
    format_hf_canonical_ref, identity_hash_for, local_gguf_identity_from_source,
    parse_hf_ref_parts, parse_hf_resolve_url_parts, unknown_identity,
};

pub(crate) fn infer_remote_served_descriptors(
    primary_model_name: &str,
    serving_models: &[String],
    model_source: Option<&str>,
) -> Vec<ServedModelDescriptor> {
    let primary = model_source.and_then(identity_from_model_source);
    let primary_index = serving_models
        .iter()
        .position(|model_name| model_name == primary_model_name);
    serving_models
        .iter()
        .enumerate()
        .map(|(idx, model_name)| {
            let identity = if Some(idx) == primary_index {
                let mut identity = primary
                    .clone()
                    .unwrap_or_else(|| unknown_identity(model_name));
                identity.model_name = model_name.clone();
                identity.is_primary = true;
                if identity.local_file_name.is_none() {
                    identity.local_file_name = Some(format!("{model_name}.gguf"));
                }
                identity
            } else {
                unknown_identity(model_name)
            };
            ServedModelDescriptor {
                identity,
                capabilities_known: false,
                capabilities: crate::models::ModelCapabilities::default(),
                topology: None,
                metadata: None,
            }
        })
        .collect()
}

pub(crate) fn identity_from_model_source(source: &str) -> Option<ServedModelIdentity> {
    let trimmed = source.trim();
    if trimmed.is_empty() {
        return None;
    }

    if let Ok(model_ref) = skippy_model_ref::ModelRef::parse(trimmed) {
        let display_id = model_ref.display_id();
        return Some(ServedModelIdentity {
            model_name: String::new(),
            is_primary: false,
            source_kind: ModelSourceKind::HuggingFace,
            canonical_ref: Some(display_id.clone()),
            repository: Some(model_ref.repo),
            revision: model_ref.revision,
            artifact: model_ref.selector,
            local_file_name: None,
            identity_hash: Some(identity_hash_for(&display_id)),
            weights_digest: None,
        });
    }

    if trimmed.starts_with('/') || trimmed.starts_with("./") || trimmed.starts_with("../") {
        return Some(local_gguf_identity_from_source(trimmed));
    }

    if let Some((repo_id, revision, file)) = parse_hf_resolve_url_parts(trimmed) {
        let canonical_ref = format_hf_canonical_ref(&repo_id, revision.as_deref(), &file);
        return Some(ServedModelIdentity {
            model_name: String::new(),
            is_primary: false,
            source_kind: ModelSourceKind::HuggingFace,
            canonical_ref: Some(canonical_ref.clone()),
            repository: Some(repo_id),
            revision,
            artifact: Some(file.clone()),
            local_file_name: file.rsplit('/').next().map(str::to_string),
            identity_hash: Some(identity_hash_for(&canonical_ref)),
            weights_digest: None,
        });
    }

    if let Some((repo_id, revision, file)) = parse_hf_ref_parts(trimmed) {
        let canonical_ref = format_hf_canonical_ref(&repo_id, revision.as_deref(), &file);
        return Some(ServedModelIdentity {
            model_name: String::new(),
            is_primary: false,
            source_kind: ModelSourceKind::HuggingFace,
            canonical_ref: Some(canonical_ref.clone()),
            repository: Some(repo_id),
            revision,
            artifact: Some(file.clone()),
            local_file_name: file.rsplit('/').next().map(str::to_string),
            identity_hash: Some(identity_hash_for(&canonical_ref)),
            weights_digest: None,
        });
    }

    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        return Some(ServedModelIdentity {
            model_name: String::new(),
            is_primary: false,
            source_kind: ModelSourceKind::DirectUrl,
            canonical_ref: Some(trimmed.to_string()),
            repository: None,
            revision: None,
            artifact: None,
            local_file_name: trimmed.rsplit('/').next().map(str::to_string),
            identity_hash: Some(identity_hash_for(trimmed)),
            weights_digest: None,
        });
    }

    if trimmed.ends_with(".gguf")
        || (trimmed.contains('/') && !trimmed.ends_with('/') && trimmed.split('/').count() != 2)
    {
        return Some(local_gguf_identity_from_source(trimmed));
    }

    Some(ServedModelIdentity {
        model_name: String::new(),
        is_primary: false,
        source_kind: ModelSourceKind::Catalog,
        canonical_ref: Some(trimmed.to_string()),
        repository: None,
        revision: None,
        artifact: None,
        local_file_name: None,
        identity_hash: Some(identity_hash_for(&format!("catalog:{trimmed}"))),
        weights_digest: None,
    })
}

/// Expose only remotely resolvable model identities as public catalog IDs.
pub(crate) fn public_model_id_from_identity(identity: &ServedModelIdentity) -> Option<String> {
    match identity.source_kind {
        ModelSourceKind::HuggingFace => identity
            .repository
            .as_deref()
            .map(|repo| {
                let selector = identity
                    .artifact
                    .as_deref()
                    .and_then(skippy_model_ref::quant_selector_from_gguf_file)
                    .or_else(|| identity.artifact.clone());
                skippy_model_ref::format_model_ref(
                    repo,
                    identity.revision.as_deref(),
                    selector.as_deref(),
                )
            })
            .or_else(|| {
                identity
                    .canonical_ref
                    .as_deref()
                    .and_then(|model_ref| skippy_model_ref::ModelRef::parse(model_ref).ok())
                    .map(|model_ref| model_ref.display_id())
            }),
        ModelSourceKind::Catalog => identity
            .canonical_ref
            .as_deref()
            .and_then(|model_ref| skippy_model_ref::ModelRef::parse(model_ref).ok())
            .map(|model_ref| model_ref.display_id()),
        ModelSourceKind::LocalGguf | ModelSourceKind::DirectUrl | ModelSourceKind::Unknown => None,
    }
}

/// Normalize demand references without converting local paths into remote model identities.
pub(crate) fn canonical_demand_model_ref(model: &str) -> String {
    if let Ok(model_ref) = skippy_model_ref::ModelRef::parse(model) {
        return model_ref.display_id();
    }
    crate::models::find_loaded_remote_catalog_model_exact(model)
        .map(|remote_model| crate::models::remote_catalog_model_ref(&remote_model))
        .unwrap_or_else(|| model.to_string())
}

/// Match exactly the same public alias that peer HTTP discovery advertises.
/// Do not infer identity from similar basenames or borrow another peer's facts.
pub(crate) fn descriptor_matches_routable_name(
    descriptor: &ServedModelDescriptor,
    name: &str,
) -> bool {
    let identity = &descriptor.identity;
    identity.model_name == name
        || public_model_id_from_identity(identity)
            .unwrap_or_else(|| canonical_demand_model_ref(&identity.model_name))
            == name
}
