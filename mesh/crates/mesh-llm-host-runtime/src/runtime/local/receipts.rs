//! Inference receipts for a model this node serves alone, when `[receipts]`
//! is on (`mesh/docs/design/RECEIPTS.md`). The node key signs them, so a
//! client can tell which peer served a turn it audits.
//!
//! A single stage has no receipts address of its own: clients open its
//! receipts on its OpenAI address (`POST /v1/receipts/open`), which the
//! mesh routes like any request for the model.

use std::path::Path;

use super::LocalOpenAiModelStartSpec;

/// What the runtime of `model_name` needs to issue receipts, or `None` when
/// the node issues none.
pub(super) fn receipts_options(
    spec: &LocalOpenAiModelStartSpec<'_>,
    model_name: &str,
) -> Option<skippy_serving::receipts::ReceiptsOptions> {
    spec.mesh_config.receipts.enabled.then(|| {
        skippy_serving::receipts::ReceiptsOptions::embedded(
            receipt_model(spec.model_path, model_name),
            spec.receipts_signing_key,
        )
    })
}

/// The model a receipt names: `repo@commit/file` when the GGUF is a Hugging
/// Face file, or the first shard of a split one, which a client then reads
/// by range to audit the turn without the model; else `model_name`.
pub(in crate::runtime) fn receipt_model(path: &Path, model_name: &str) -> String {
    hugging_face_ref(path).unwrap_or_else(|| model_name.to_owned())
}

/// `repo@commit/file` for a GGUF in the Hugging Face cache: a whole GGUF,
/// or the first shard of a split one, which names the split. Another shard
/// is not the model, and names nothing.
fn hugging_face_ref(path: &Path) -> Option<String> {
    let file = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default();
    let names_model =
        skippy_model_ref::split_gguf_shard_info(file).is_none_or(|shard| shard.part == "00001");
    names_model
        .then(|| crate::models::huggingface_identity_for_path(path))
        .flatten()
        .map(|identity| identity.canonical_ref)
}

#[cfg(test)]
mod tests {
    use super::receipt_model;
    use std::path::Path;

    #[test]
    fn a_hugging_face_file_is_named_by_its_pinned_revision() {
        let path = Path::new(
            "/cache/huggingface/hub/models--unsloth--Qwen3-4B-GGUF/snapshots/0123456789abcdef0123456789abcdef01234567/Qwen3-4B-Q8_0.gguf",
        );
        let named = receipt_model(path, "Qwen3-4B");
        assert!(named.starts_with("unsloth/Qwen3-4B-GGUF@"), "{named}");
        assert!(named.ends_with("Qwen3-4B-Q8_0.gguf"), "{named}");
    }

    #[test]
    fn a_split_gguf_is_named_by_its_first_shard() {
        let snapshot = "/cache/huggingface/hub/models--o--m/snapshots/0123456789abcdef0123456789abcdef01234567";
        let first = format!("{snapshot}/M-00001-of-00002.gguf");
        let named = receipt_model(Path::new(&first), "M");
        assert!(named.starts_with("o/m@"), "{named}");
        assert!(named.ends_with("/M-00001-of-00002.gguf"), "{named}");
        let second = format!("{snapshot}/M-00002-of-00002.gguf");
        assert_eq!(receipt_model(Path::new(&second), "M"), "M");
    }

    #[test]
    fn a_local_file_is_named_by_the_model_name() {
        assert_eq!(receipt_model(Path::new("/models/m.gguf"), "m"), "m");
    }
}
