//! Mesh protobuf adapter for Skippy-owned local model inventory.

use std::collections::{HashMap, HashSet};

pub use skippy_model_hf::inventory::ModelMetadataCacheProgress;
pub(crate) use skippy_model_hf::inventory::derive_quantization_type;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct LocalModelInventorySnapshot {
    pub model_names: HashSet<String>,
    pub size_by_name: HashMap<String, u64>,
    pub metadata_by_name: HashMap<String, crate::proto::node::CompactModelMetadata>,
    pub display_name_by_name: HashMap<String, String>,
}

pub fn scan_local_inventory_snapshot_with_progress<F>(on_progress: F) -> LocalModelInventorySnapshot
where
    F: FnMut(ModelMetadataCacheProgress),
{
    let full_scan = std::env::var("MESH_LLM_ALLOW_FULL_HF_CACHE_SCAN").unwrap_or_default() == "1";
    skippy_model_hf::inventory::scan_local_inventory_snapshot_with_progress(full_scan, on_progress)
        .into()
}

impl From<skippy_model_hf::inventory::LocalModelInventorySnapshot> for LocalModelInventorySnapshot {
    fn from(snapshot: skippy_model_hf::inventory::LocalModelInventorySnapshot) -> Self {
        Self {
            model_names: snapshot.model_names,
            size_by_name: snapshot.size_by_name,
            metadata_by_name: snapshot
                .metadata_by_name
                .into_iter()
                .map(|(name, metadata)| (name, metadata_into_proto(metadata)))
                .collect(),
            display_name_by_name: snapshot.display_name_by_name,
        }
    }
}

fn metadata_into_proto(
    meta: skippy_model_hf::inventory::CompactModelMetadata,
) -> crate::proto::node::CompactModelMetadata {
    crate::proto::node::CompactModelMetadata {
        model_key: meta.model_key,
        parameter_size: meta.parameter_size,
        context_length: meta.context_length,
        vocab_size: meta.vocab_size,
        embedding_size: meta.embedding_size,
        head_count: meta.head_count,
        kv_head_count: meta.kv_head_count,
        layer_count: meta.layer_count,
        feed_forward_length: meta.feed_forward_length,
        key_length: meta.key_length,
        value_length: meta.value_length,
        architecture: meta.architecture,
        tokenizer_model_name: meta.tokenizer_model_name,
        special_tokens: vec![],
        rope_scale: meta.rope_scale,
        rope_freq_base: meta.rope_freq_base,
        is_moe: meta.is_moe,
        expert_count: meta.expert_count,
        used_expert_count: meta.used_expert_count,
        quantization_type: meta.quantization_type,
    }
}
