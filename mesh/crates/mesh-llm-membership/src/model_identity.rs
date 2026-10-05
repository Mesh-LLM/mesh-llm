//! Content identity for a served model: the *reference string* hash and the
//! pure parsers that turn a source string into a `ServedModelIdentity`.
//!
//! These are the Skippy-free leaves of the host's old `model_identity`
//! module — they touch only `mesh_llm_types::mesh` types plus `sha2`/`hex`.
//! The two remaining host functions (`identity_from_model_source`, which
//! calls `skippy_model_ref::ModelRef::parse`, and
//! `infer_remote_served_descriptors`, which assembles `ModelCapabilities`)
//! stay in `mesh-llm-host-runtime` until a model-ref parse operation is
//! injected across the Skippy boundary.

use mesh_llm_types::mesh::{ModelSourceKind, ServedModelIdentity};

pub fn unknown_identity(model_name: &str) -> ServedModelIdentity {
    ServedModelIdentity {
        model_name: model_name.to_string(),
        is_primary: false,
        source_kind: ModelSourceKind::Unknown,
        canonical_ref: None,
        repository: None,
        revision: None,
        artifact: None,
        local_file_name: Some(format!("{model_name}.gguf")),
        identity_hash: None,
        weights_digest: None,
    }
}

pub fn local_gguf_identity_from_source(source: &str) -> ServedModelIdentity {
    let local_file_name = std::path::Path::new(source)
        .file_name()
        .and_then(|value| value.to_str())
        .map(str::to_string);
    ServedModelIdentity {
        model_name: String::new(),
        is_primary: false,
        source_kind: ModelSourceKind::LocalGguf,
        canonical_ref: None,
        repository: None,
        revision: None,
        artifact: None,
        local_file_name,
        identity_hash: None,
        weights_digest: None,
    }
}

pub fn parse_hf_ref_parts(input: &str) -> Option<(String, Option<String>, String)> {
    if input.starts_with('/') || input.starts_with("./") || input.starts_with("../") {
        return None;
    }
    let parts: Vec<&str> = input.splitn(3, '/').collect();
    if parts.len() != 3 {
        return None;
    }
    let (repo_tail, revision) = match parts[1].split_once('@') {
        Some((repo, revision)) => (repo, Some(revision.to_string())),
        None => (parts[1], None),
    };
    if parts[0].is_empty() || repo_tail.is_empty() || parts[2].is_empty() {
        return None;
    }
    Some((
        format!("{}/{}", parts[0], repo_tail),
        revision,
        parts[2].to_string(),
    ))
}

pub fn parse_hf_resolve_url_parts(url: &str) -> Option<(String, Option<String>, String)> {
    let path = url
        .strip_prefix("https://huggingface.co/")
        .or_else(|| url.strip_prefix("http://huggingface.co/"))?;
    let (repo, rest) = path.split_once("/resolve/")?;
    let (revision, file) = rest.split_once('/')?;
    if repo.is_empty() || revision.is_empty() || file.is_empty() {
        return None;
    }
    Some((
        repo.to_string(),
        Some(revision.to_string()),
        file.to_string(),
    ))
}

pub fn format_hf_canonical_ref(repo: &str, revision: Option<&str>, file: &str) -> String {
    match revision {
        Some(revision) => format!("{repo}@{revision}/{file}"),
        None => format!("{repo}/{file}"),
    }
}

pub fn identity_hash_for(input: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_url_supports_top_level_hugging_face_repository() {
        // Given: a resolve URL for a top-level Hugging Face repository.
        let url = "https://huggingface.co/gpt2/resolve/main/model.safetensors";

        // When: the URL identity is parsed.
        let parts = parse_hf_resolve_url_parts(url);

        // Then: the repository, revision, and artifact are preserved.
        assert_eq!(
            parts,
            Some((
                "gpt2".to_string(),
                Some("main".to_string()),
                "model.safetensors".to_string(),
            ))
        );
    }
}
