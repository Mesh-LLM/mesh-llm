use model_artifact::gguf::{
    GgufCompactMeta, GgufKvCacheQuant, GgufKvCacheType, GgufTensorByteProfile,
    scan_gguf_compact_meta, scan_gguf_tensor_byte_profile,
};
use std::fmt;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct TuneGgufMetadata {
    pub compact_meta: GgufCompactMeta,
    pub tensor_profile: TuneTensorProfile,
    pub model_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TuneTensorProfile {
    Exact(GgufTensorByteProfile),
    DegradedFallback { model_bytes: u64 },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TuneGgufMetadataError {
    CompactMetadataUnreadable {
        model: String,
    },
    MissingRequiredMetadata {
        model: String,
        missing_fields: Vec<&'static str>,
    },
    UnsupportedKvTypes {
        model: String,
        invalid_fields: Vec<InvalidKvType>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InvalidKvType {
    pub field_name: &'static str,
    pub value: String,
}

impl fmt::Display for TuneGgufMetadataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CompactMetadataUnreadable { model } => write!(
                f,
                "model `{model}`: could not read compact GGUF metadata from the local target"
            ),
            Self::MissingRequiredMetadata {
                model,
                missing_fields,
            } => write!(
                f,
                "model `{model}`: compact GGUF metadata is missing required fields: {}",
                missing_fields.join(", ")
            ),
            Self::UnsupportedKvTypes {
                model,
                invalid_fields,
            } => {
                let details = invalid_fields
                    .iter()
                    .map(|field| format!("{}=`{}`", field.field_name, field.value))
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(
                    f,
                    "model `{model}`: unsupported KV cache types ({details}); supported values are f16, q8_0, q4_0"
                )
            }
        }
    }
}

pub fn inspect_local_gguf_metadata(
    model: &str,
    path: &Path,
) -> Result<TuneGgufMetadata, TuneGgufMetadataError> {
    let compact_meta = scan_gguf_compact_meta(path).ok_or_else(|| {
        TuneGgufMetadataError::CompactMetadataUnreadable {
            model: model.to_string(),
        }
    })?;

    let missing_fields = missing_required_metadata_fields(&compact_meta);
    if !missing_fields.is_empty() {
        return Err(TuneGgufMetadataError::MissingRequiredMetadata {
            model: model.to_string(),
            missing_fields,
        });
    }

    let model_bytes = std::fs::metadata(path)
        .map(|metadata| metadata.len())
        .unwrap_or_default();
    let tensor_profile = match scan_gguf_tensor_byte_profile(path) {
        Some(profile) => TuneTensorProfile::Exact(profile),
        None => TuneTensorProfile::DegradedFallback { model_bytes },
    };

    Ok(TuneGgufMetadata {
        compact_meta,
        tensor_profile,
        model_bytes,
    })
}

pub fn validate_kv_cache_quant(
    model: &str,
    cache_type_k: &str,
    cache_type_v: &str,
) -> Result<GgufKvCacheQuant, TuneGgufMetadataError> {
    let parsed_k = GgufKvCacheType::from_llama_arg(cache_type_k);
    let parsed_v = GgufKvCacheType::from_llama_arg(cache_type_v);
    let mut invalid_fields = Vec::new();
    if parsed_k.is_none() {
        invalid_fields.push(InvalidKvType {
            field_name: "cache_type_k",
            value: cache_type_k.to_string(),
        });
    }
    if parsed_v.is_none() {
        invalid_fields.push(InvalidKvType {
            field_name: "cache_type_v",
            value: cache_type_v.to_string(),
        });
    }
    if !invalid_fields.is_empty() {
        return Err(TuneGgufMetadataError::UnsupportedKvTypes {
            model: model.to_string(),
            invalid_fields,
        });
    }

    GgufKvCacheQuant::from_llama_args(cache_type_k, cache_type_v).ok_or_else(|| {
        TuneGgufMetadataError::UnsupportedKvTypes {
            model: model.to_string(),
            invalid_fields: vec![
                InvalidKvType {
                    field_name: "cache_type_k",
                    value: cache_type_k.to_string(),
                },
                InvalidKvType {
                    field_name: "cache_type_v",
                    value: cache_type_v.to_string(),
                },
            ],
        }
    })
}

fn missing_required_metadata_fields(compact_meta: &GgufCompactMeta) -> Vec<&'static str> {
    let mut missing_fields = Vec::new();
    if compact_meta.architecture.is_empty() {
        missing_fields.push("architecture");
    }
    if compact_meta.context_length == 0 {
        missing_fields.push("context_length");
    }
    if compact_meta.layer_count == 0 {
        missing_fields.push("layer_count");
    }
    if compact_meta.effective_kv_head_count().is_none() {
        missing_fields.push("kv_head_count");
    }
    if compact_meta.key_length == 0 {
        missing_fields.push("key_length");
    }
    if compact_meta.value_length == 0 {
        missing_fields.push("value_length");
    }
    missing_fields
}
