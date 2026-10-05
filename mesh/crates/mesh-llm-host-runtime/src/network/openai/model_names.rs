//! Public model names shared by listing, descriptor lookup and request resolution.
use crate::mesh;

pub(super) fn public_model_id(
    model_name: &str,
    descriptor: Option<&mesh::ServedModelDescriptor>,
    profile: &str,
) -> String {
    // A descriptor with an `artifact` field has enough information to
    // produce a public ID that round-trips to the same model. Without
    // it, the HuggingFace path collapses to just the repo name and
    // silently drops the quant-tag suffix the resolver needs (PR #566
    // review feedback — "some IDs in /v1/models dropped quant
    // suffixes"). Only use the descriptor-derived id when it can be
    // lossless; otherwise prefer the on-disk file (authoritative for
    // local models), and finally the internal model_name (which
    // always carries the quant suffix our resolver knows how to
    // route).
    let base_id = if let Some(descriptor) = descriptor
        && descriptor_can_produce_lossless_id(&descriptor.identity)
        && let Some(id) = mesh::public_model_id_from_identity(&descriptor.identity)
    {
        id
    } else if let Some(id) = public_model_id_from_local_path(model_name) {
        id
    } else {
        model_name.to_string()
    };

    // Append profile suffix for non-default profiles
    if profile.is_empty() {
        base_id
    } else {
        format!("{}#{}", base_id, profile)
    }
}

/// A descriptor identity carries enough information for
/// `public_model_id_from_identity` to produce an ID that round-trips
/// to the same model. For HuggingFace that means the `artifact` field
/// (the GGUF file name) is present so the quant selector can be
/// derived. Catalog identities always carry a `canonical_ref` with the
/// selector baked in.
fn descriptor_can_produce_lossless_id(identity: &mesh::ServedModelIdentity) -> bool {
    match identity.source_kind {
        mesh::ModelSourceKind::HuggingFace => identity.artifact.is_some(),
        mesh::ModelSourceKind::Catalog => identity.canonical_ref.is_some(),
        mesh::ModelSourceKind::LocalGguf
        | mesh::ModelSourceKind::DirectUrl
        | mesh::ModelSourceKind::Unknown => false,
    }
}

fn public_model_id_from_local_path(model_name: &str) -> Option<String> {
    let path = crate::models::find_model_path(model_name);
    if !path.is_file() {
        return None;
    }
    if path.extension().and_then(|extension| extension.to_str()) != Some("gguf") {
        return None;
    }
    Some(crate::models::model_ref_for_path(&path))
}
