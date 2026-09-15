//! Human-readable labels for synthetic routing identities.
use crate::mesh::{ModelSourceKind, ServedModelIdentity};

pub(super) fn source_display_name(
    name: &str,
    identity: Option<&ServedModelIdentity>,
) -> Option<String> {
    if !name.starts_with("local-gguf/") {
        return None;
    }
    let identity = identity?;
    let label = match identity.source_kind {
        ModelSourceKind::HuggingFace | ModelSourceKind::Catalog => {
            identity.canonical_ref.as_deref()?
        }
        ModelSourceKind::LocalGguf | ModelSourceKind::DirectUrl => identity
            .local_file_name
            .as_deref()?
            .strip_suffix(".gguf")
            .unwrap_or(identity.local_file_name.as_deref()?),
        ModelSourceKind::Unknown => return None,
    };
    let label = label.trim();
    (!label.is_empty() && !label.starts_with("local-gguf/")).then(|| label.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_labels_do_not_change_routing_identity_or_explicit_aliases() {
        let name = "local-gguf/sha256-abcd";
        let mut identity = ServedModelIdentity {
            source_kind: ModelSourceKind::HuggingFace,
            canonical_ref: Some("unsloth/Gemma-GGUF@main:Q4_K_M".into()),
            ..Default::default()
        };
        assert_eq!(
            source_display_name(name, Some(&identity)).as_deref(),
            Some("unsloth/Gemma-GGUF@main:Q4_K_M")
        );
        assert_eq!(source_display_name("my-model", Some(&identity)), None);
        identity.source_kind = ModelSourceKind::LocalGguf;
        identity.local_file_name = Some("Gemma-Q4_K_M.gguf".into());
        assert_eq!(
            source_display_name(name, Some(&identity)).as_deref(),
            Some("Gemma-Q4_K_M")
        );
        identity.source_kind = ModelSourceKind::Unknown;
        assert_eq!(source_display_name(name, Some(&identity)), None);
        assert_eq!(source_display_name(name, None), None);
    }
}
