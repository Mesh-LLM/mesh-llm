use model_artifact::gguf::{
    GgufCompactMeta, GgufKvCacheQuant, GgufKvCacheType, GgufTensorByteProfile,
    scan_gguf_compact_meta, scan_gguf_tensor_byte_profile,
};
use std::fmt;
use std::path::{Component, Path, PathBuf};

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
    LayerPackageMetadataUnreadable {
        model: String,
        reason: String,
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
            Self::LayerPackageMetadataUnreadable { model, reason } => write!(
                f,
                "model `{model}`: could not read layer package metadata: {reason}"
            ),
        }
    }
}

pub fn inspect_tune_target_metadata(
    model: &str,
    path: &Path,
) -> Result<TuneGgufMetadata, TuneGgufMetadataError> {
    let source = tune_metadata_source(model, path)?;
    inspect_gguf_metadata(model, &source.gguf_path, source.model_bytes)
}

pub fn inspect_local_gguf_metadata(
    model: &str,
    path: &Path,
) -> Result<TuneGgufMetadata, TuneGgufMetadataError> {
    inspect_gguf_metadata(model, path, None)
}

fn inspect_gguf_metadata(
    model: &str,
    path: &Path,
    model_bytes_override: Option<u64>,
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

    let model_bytes = model_bytes_override.unwrap_or_else(|| {
        std::fs::metadata(path)
            .map(|metadata| metadata.len())
            .unwrap_or_default()
    });
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

#[derive(Debug)]
struct TuneMetadataSource {
    gguf_path: PathBuf,
    model_bytes: Option<u64>,
}

fn tune_metadata_source(
    model: &str,
    path: &Path,
) -> Result<TuneMetadataSource, TuneGgufMetadataError> {
    let manifest_path = path.join("model-package.json");
    if !manifest_path.is_file() {
        return Ok(TuneMetadataSource {
            gguf_path: path.to_path_buf(),
            model_bytes: None,
        });
    }

    let manifest = read_package_manifest(model, &manifest_path)?;
    let metadata_path = package_artifact_path(
        model,
        path,
        manifest
            .pointer("/shared/metadata/path")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| package_metadata_error(model, "shared.metadata.path is missing"))?,
    )?;
    Ok(TuneMetadataSource {
        gguf_path: metadata_path,
        model_bytes: package_model_bytes(&manifest),
    })
}

fn read_package_manifest(
    model: &str,
    manifest_path: &Path,
) -> Result<serde_json::Value, TuneGgufMetadataError> {
    let bytes = std::fs::read(manifest_path)
        .map_err(|error| package_metadata_error(model, error.to_string()))?;
    serde_json::from_slice(&bytes).map_err(|error| package_metadata_error(model, error.to_string()))
}

fn package_artifact_path(
    model: &str,
    package_dir: &Path,
    relative_path: &str,
) -> Result<PathBuf, TuneGgufMetadataError> {
    let path = Path::new(relative_path);
    let safe = !relative_path.trim().is_empty()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_) | Component::CurDir));
    if !safe {
        return Err(package_metadata_error(
            model,
            format!("shared.metadata.path is not a safe relative path: {relative_path}"),
        ));
    }
    Ok(package_dir.join(path))
}

fn package_model_bytes(manifest: &serde_json::Value) -> Option<u64> {
    source_model_file_bytes(manifest).or_else(|| artifact_bytes(manifest))
}

fn source_model_file_bytes(manifest: &serde_json::Value) -> Option<u64> {
    let files = manifest.pointer("/source_model/files")?.as_array()?;
    checked_sum(
        files
            .iter()
            .filter_map(|file| file.get("size_bytes").and_then(serde_json::Value::as_u64)),
    )
}

fn artifact_bytes(manifest: &serde_json::Value) -> Option<u64> {
    let shared = [
        manifest.pointer("/shared/metadata"),
        manifest.pointer("/shared/embeddings"),
        manifest.pointer("/shared/output"),
    ]
    .into_iter()
    .flatten()
    .filter_map(|artifact| {
        artifact
            .get("artifact_bytes")
            .and_then(serde_json::Value::as_u64)
    });
    let layers = manifest
        .get("layers")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|artifact| {
            artifact
                .get("artifact_bytes")
                .and_then(serde_json::Value::as_u64)
        });
    let projectors = manifest
        .get("projectors")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|artifact| {
            artifact
                .get("artifact_bytes")
                .and_then(serde_json::Value::as_u64)
        });

    checked_sum(shared.chain(layers).chain(projectors))
}

fn checked_sum(values: impl IntoIterator<Item = u64>) -> Option<u64> {
    let mut total = 0u64;
    let mut saw_value = false;
    for value in values {
        saw_value = true;
        total = total.checked_add(value)?;
    }
    saw_value.then_some(total)
}

fn package_metadata_error(model: &str, reason: impl Into<String>) -> TuneGgufMetadataError {
    TuneGgufMetadataError::LayerPackageMetadataUnreadable {
        model: model.to_string(),
        reason: reason.into(),
    }
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
