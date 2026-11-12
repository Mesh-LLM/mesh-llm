//! Loads a trained BigHead pointer head embedded in a model GGUF.
//!
//! The head ships inside the model file: fp32 tensors `bighead.norm.weight`,
//! `bighead.norm.bias`, `bighead.q.weight`, `bighead.k.weight` (torch Linear
//! `[out][in]`), optional `bighead.q.bias`/`bighead.k.bias` for a
//! torch-default `nn.Linear` head, plus `bighead.*` metadata (schema version,
//! hidden size, pointer dimension, per-kind temperatures). This module reads
//! that head through skippy-model's existing GGUF catalog reader — no second
//! parser.
//!
//! Presence rule: a GGUF with NO `bighead.*` keys at all is an ordinary model
//! and yields `Ok(None)`, leaving every other backend unaffected. A GGUF with
//! SOME `bighead.*` keys but an incomplete head (missing required metadata or
//! tensors, wrong dtype or shape) is a hard error: a partially written head
//! must never silently disable the backend and fall through to a logit
//! readout with different probability semantics.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use anyhow::{Context, Result, ensure};
use serde_json::Value;
use skippy_model::gguf_catalog::{GgufCatalog, read_gguf_catalog};

use super::bighead::{BIGHEAD_SCHEMA_VERSION, BigHeadConfig, BigHeadWeights};
use super::bighead_serve::BigHeadModel;

/// GGML type id for fp32 payloads; the head trains and serves in f32.
const GGML_TYPE_F32: u32 = 0;
const BIGHEAD_PREFIX: &str = "bighead.";

/// Reads the embedded BigHead, if any, from a model GGUF.
///
/// `Ok(None)` when the file carries no `bighead.*` metadata or tensors.
/// Errors when it carries some but the head is incomplete or malformed.
pub fn load_bighead_from_gguf(path: &Path) -> Result<Option<(BigHeadConfig, BigHeadWeights)>> {
    let catalog = read_gguf_catalog(path)
        .with_context(|| format!("read GGUF catalog for BigHead head in {}", path.display()))?;
    if !carries_bighead(&catalog) {
        return Ok(None);
    }
    let config = read_config(&catalog)?;
    config
        .validate()
        .context("validate BigHead metadata in model GGUF")?;
    let weights = read_weights(path, &catalog, &config)?;
    weights
        .validate(&config)
        .context("validate BigHead tensor shapes in model GGUF")?;
    Ok(Some((config, weights)))
}

/// The decision-dispatch seam for model load: a GGUF carrying a BigHead
/// registers the trained-pointer-head backend, reporting
/// [`BigHeadModel::SEMANTICS`]; an ordinary GGUF yields `None` and the
/// existing causal/laya/diffusion dispatch continues unchanged.
///
/// The returned model has no hidden-state provider yet; it is attached at
/// serve time over the llama context (see [`super::bighead_hidden`]), and
/// `decide` fails closed until then.
pub fn load_bighead_backend(path: &Path) -> Result<Option<BigHeadModel>> {
    Ok(load_bighead_from_gguf(path)?
        .map(|(config, weights)| BigHeadModel::new(config, weights)))
}

fn carries_bighead(catalog: &GgufCatalog) -> bool {
    catalog.metadata.keys().any(|key| key.starts_with(BIGHEAD_PREFIX))
        || catalog
            .tensors
            .iter()
            .any(|tensor| tensor.name.starts_with(BIGHEAD_PREFIX))
}

fn read_config(catalog: &GgufCatalog) -> Result<BigHeadConfig> {
    let metadata = &catalog.metadata;
    let hidden_size = required_usize(metadata, "bighead.hidden_size")?;
    let pointer_dim = required_usize(metadata, "bighead.pointer_dim")?;
    let schema_version = match metadata.get("bighead.schema_version") {
        Some(value) => Some(
            value
                .as_u64()
                .map(|version| version.to_string())
                .context("bighead.schema_version must be an integer")?,
        ),
        None => None,
    };
    if let Some(version) = &schema_version {
        ensure!(
            version == BIGHEAD_SCHEMA_VERSION,
            "unsupported bighead schema {version:?} in model GGUF (expected {BIGHEAD_SCHEMA_VERSION:?})"
        );
    }
    let mut kinds = BTreeMap::new();
    for (key, value) in metadata {
        if let Some(kind) = key.strip_prefix("bighead.kind.") {
            let temperature = value
                .as_f64()
                .with_context(|| format!("bighead temperature for kind {kind:?} must be a float"))?;
            kinds.insert(kind.to_string(), temperature);
        }
    }
    Ok(BigHeadConfig {
        schema_version,
        hidden_size,
        pointer_dim,
        kinds,
    })
}

fn required_usize(metadata: &BTreeMap<String, Value>, key: &str) -> Result<usize> {
    let value = metadata
        .get(key)
        .with_context(|| format!("model GGUF carries bighead.* keys but is missing {key}"))?;
    usize::try_from(
        value
            .as_u64()
            .with_context(|| format!("{key} must be a non-negative integer"))?,
    )
    .context("{key} exceeds usize")
}

fn read_weights(path: &Path, catalog: &GgufCatalog, cfg: &BigHeadConfig) -> Result<BigHeadWeights> {
    let h = cfg.hidden_size as u64;
    let d = cfg.pointer_dim as u64;
    let mut weights = BigHeadWeights {
        norm_weight: read_f32_tensor(path, catalog, "bighead.norm.weight", &[h])?,
        norm_bias: read_f32_tensor(path, catalog, "bighead.norm.bias", &[h])?,
        q: read_f32_tensor(path, catalog, "bighead.q.weight", &[d, h])?,
        k: read_f32_tensor(path, catalog, "bighead.k.weight", &[d, h])?,
        q_bias: None,
        k_bias: None,
    };
    // Projection biases are required only when the head was packaged with a
    // torch-default nn.Linear; a bias=False head ships without them.
    if has_tensor(catalog, "bighead.q.bias") || has_tensor(catalog, "bighead.k.bias") {
        weights.q_bias = Some(read_f32_tensor(path, catalog, "bighead.q.bias", &[d])?);
        weights.k_bias = Some(read_f32_tensor(path, catalog, "bighead.k.bias", &[d])?);
    }
    Ok(weights)
}

fn has_tensor(catalog: &GgufCatalog, name: &str) -> bool {
    catalog.tensors.iter().any(|tensor| tensor.name == name)
}

/// Reads one fp32 tensor payload from the artifact. The catalog reader parses
/// the GGUF structure and resolves absolute payload offsets; the bytes
/// themselves are read here, once, per tensor the head needs.
fn read_f32_tensor(path: &Path, catalog: &GgufCatalog, name: &str, dims: &[u64]) -> Result<Vec<f32>> {
    let tensor = catalog
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .with_context(|| format!("model GGUF carries bighead.* keys but is missing tensor {name}"))?;
    ensure!(
        tensor.ggml_type == GGML_TYPE_F32,
        "bighead tensor {name} must be fp32 (ggml type {GGML_TYPE_F32}), got type {}",
        tensor.ggml_type
    );
    ensure!(
        tensor.dimensions == dims,
        "bighead tensor {name} has dimensions {:?}, expected {dims:?} (torch Linear [out][in])",
        tensor.dimensions
    );
    let elements: u64 = dims.iter().product();
    let byte_len: usize = elements
        .checked_mul(4)
        .and_then(|len| usize::try_from(len).ok())
        .context("bighead tensor {name} byte length exceeds usize")?;
    let end = tensor
        .data_offset
        .checked_add(byte_len as u64)
        .context("bighead tensor {name} payload offset overflow")?;
    ensure!(
        end <= catalog.artifact_bytes,
        "bighead tensor {name} payload extends past the artifact"
    );
    let mut file =
        File::open(path).with_context(|| format!("open GGUF artifact {}", path.display()))?;
    file.seek(SeekFrom::Start(tensor.data_offset))
        .with_context(|| format!("seek to bighead tensor {name}"))?;
    let mut bytes = vec![0_u8; byte_len];
    file.read_exact(&mut bytes)
        .with_context(|| format!("read bighead tensor {name} payload"))?;
    let (chunks, tail) = bytes.as_chunks::<4>();
    assert!(tail.is_empty(), "GGUF fp32 tensor payload is not a multiple of four bytes");
    Ok(chunks.iter().map(|c| f32::from_le_bytes(*c)).collect())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use super::*;

    const GGUF_TYPE_FLOAT32: u32 = 6;
    const GGUF_TYPE_UINT32: u32 = 4;

    struct TensorSpec {
        name: &'static str,
        dims: Vec<u64>,
        values: Vec<f32>,
    }

    /// Builds a synthetic GGUF on disk, following the byte-layout pattern of
    /// skippy-model's gguf_catalog tests: header, metadata, tensor table,
    /// alignment padding, then fp32 payloads at aligned offsets.
    fn write_gguf(metadata: &[(&str, u32, Vec<u8>)], tensors: &[TensorSpec]) -> PathBuf {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
        for (key, value_type, payload) in metadata {
            write_string(&mut bytes, key);
            bytes.extend_from_slice(&value_type.to_le_bytes());
            bytes.extend_from_slice(payload);
        }
        // Reserve the tensor table; payload offsets are relative to the
        // aligned data start, so write the table with placeholder offsets and
        // patch them after the table size is known.
        let table_start = bytes.len();
        for spec in tensors {
            write_string(&mut bytes, spec.name);
            bytes.extend_from_slice(&(spec.dims.len() as u32).to_le_bytes());
            for dim in &spec.dims {
                bytes.extend_from_slice(&dim.to_le_bytes());
            }
            bytes.extend_from_slice(&GGML_TYPE_F32.to_le_bytes());
            bytes.extend_from_slice(&0_u64.to_le_bytes()); // offset placeholder
        }
        let entry_offset = |index: usize| {
            table_start
                + tensors[..index]
                    .iter()
                    .map(|prior| 8 + prior.name.len() + 4 + 8 * prior.dims.len() + 4 + 8)
                    .sum::<usize>()
                + 8
                + tensors[index].name.len()
                + 4
                + 8 * tensors[index].dims.len()
                + 4
        };
        while !bytes.len().is_multiple_of(32) {
            bytes.push(0);
        }
        let data_start = bytes.len() as u64;
        for (index, spec) in tensors.iter().enumerate() {
            let offset = u64::try_from(bytes.len()).unwrap() - data_start;
            bytes[entry_offset(index)..entry_offset(index) + 8]
                .copy_from_slice(&offset.to_le_bytes());
            for value in &spec.values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            while !bytes.len().is_multiple_of(32) {
                bytes.push(0);
            }
        }
        let path = temp_path();
        fs::write(&path, bytes).unwrap();
        path
    }

    fn kv_u32(key: &str, value: u32) -> (&str, u32, Vec<u8>) {
        (Box::leak(key.to_string().into_boxed_str()), GGUF_TYPE_UINT32, value.to_le_bytes().to_vec())
    }

    fn kv_f32(key: &str, value: f32) -> (&str, u32, Vec<u8>) {
        (Box::leak(key.to_string().into_boxed_str()), GGUF_TYPE_FLOAT32, value.to_le_bytes().to_vec())
    }

    fn write_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    fn temp_path() -> PathBuf {
        // Nanoseconds alone collide when parallel tests call this in the same
        // tick (one test then deletes another's fixture); the atomic counter
        // makes every fixture path unique.
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::env::temp_dir().join(format!("skippy-bighead-gguf-{unique}-{seq}.gguf"))
    }

    /// A complete head: h=4, d=2, both projection biases packaged.
    type MetadataSpec = Vec<(&'static str, u32, Vec<u8>)>;

    fn head_specs() -> (MetadataSpec, Vec<TensorSpec>) {
        let metadata = vec![
            kv_u32("bighead.schema_version", 1),
            kv_u32("bighead.hidden_size", 4),
            kv_u32("bighead.pointer_dim", 2),
            kv_f32("bighead.kind.noul", 0.9),
            kv_f32("bighead.kind.choice", 1.1),
        ];
        let tensors = vec![
            TensorSpec { name: "bighead.norm.weight", dims: vec![4], values: vec![1.0, 1.1, 1.2, 1.3] },
            TensorSpec { name: "bighead.norm.bias", dims: vec![4], values: vec![0.1, 0.2, 0.3, 0.4] },
            TensorSpec { name: "bighead.q.weight", dims: vec![2, 4], values: vec![0.0, 8.0, 1.0, 9.0, 2.0, 10.0, 3.0, 11.0] },
            TensorSpec { name: "bighead.k.weight", dims: vec![2, 4], values: vec![4.0, 12.0, 5.0, 13.0, 6.0, 14.0, 7.0, 15.0] },
            TensorSpec { name: "bighead.q.bias", dims: vec![2], values: vec![0.5, 0.6] },
            TensorSpec { name: "bighead.k.bias", dims: vec![2], values: vec![0.7, 0.8] },
        ];
        (metadata, tensors)
    }

    #[test]
    fn round_trips_head_from_gguf() {
        let (metadata, tensors) = head_specs();
        let path = write_gguf(&metadata, &tensors);
        let (config, weights) = load_bighead_from_gguf(&path).unwrap().unwrap();
        assert_eq!(config.schema_version.as_deref(), Some("1"));
        assert_eq!(config.hidden_size, 4);
        assert_eq!(config.pointer_dim, 2);
        assert_eq!(config.kinds.len(), 2);
        assert!((config.kinds["noul"] - 0.9).abs() < 1e-6);
        assert!((config.kinds["choice"] - 1.1).abs() < 1e-6);
        assert_eq!(weights.norm_weight, vec![1.0, 1.1, 1.2, 1.3]);
        assert_eq!(weights.norm_bias, vec![0.1, 0.2, 0.3, 0.4]);
        assert_eq!(
            weights.q,
            vec![0.0, 8.0, 1.0, 9.0, 2.0, 10.0, 3.0, 11.0]
        );
        assert_eq!(weights.k, vec![4.0, 12.0, 5.0, 13.0, 6.0, 14.0, 7.0, 15.0]);
        assert_eq!(weights.q_bias.as_deref(), Some(&[0.5, 0.6][..]));
        assert_eq!(weights.k_bias.as_deref(), Some(&[0.7, 0.8][..]));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn bias_free_head_loads_without_biases() {
        let (metadata, mut tensors) = head_specs();
        tensors.retain(|spec| spec.name != "bighead.q.bias" && spec.name != "bighead.k.bias");
        let path = write_gguf(&metadata, &tensors);
        let (_, weights) = load_bighead_from_gguf(&path).unwrap().unwrap();
        assert_eq!(weights.q_bias, None);
        assert_eq!(weights.k_bias, None);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn ordinary_model_without_bighead_keys_loads_none() {
        let metadata = vec![kv_u32("general.alignment", 32)];
        let tensors = vec![TensorSpec {
            name: "blk.0.attn_q.weight",
            dims: vec![2, 2],
            values: vec![1.0, 2.0, 3.0, 4.0],
        }];
        let path = write_gguf(&metadata, &tensors);
        assert!(load_bighead_from_gguf(&path).unwrap().is_none());
        assert!(load_bighead_backend(&path).unwrap().is_none());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn partial_head_is_a_hard_error_not_silence() {
        // Only one tensor and no metadata: the file claims a head it does not have.
        let metadata = vec![kv_u32("general.alignment", 32)];
        let tensors = vec![TensorSpec {
            name: "bighead.norm.weight",
            dims: vec![4],
            values: vec![1.0; 4],
        }];
        let path = write_gguf(&metadata, &tensors);
        let error = load_bighead_from_gguf(&path).unwrap_err();
        let message = format!("{error:#}");
        assert!(
            message.contains("bighead.hidden_size"),
            "error should name the missing metadata: {message}"
        );
        fs::remove_file(path).unwrap();

        // Metadata but a missing tensor is equally hard.
        let (metadata, mut tensors) = head_specs();
        tensors.retain(|spec| spec.name != "bighead.k.weight");
        let path = write_gguf(&metadata, &tensors);
        let error = load_bighead_from_gguf(&path).unwrap_err();
        let message = format!("{error:#}");
        assert!(
            message.contains("bighead.k.weight"),
            "error should name the missing tensor: {message}"
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn backend_dispatch_reports_trained_pointer_semantics() {
        let (metadata, tensors) = head_specs();
        let path = write_gguf(&metadata, &tensors);
        let model = load_bighead_backend(&path).unwrap().unwrap();
        assert_eq!(
            model.semantics,
            "trained_pointer_head_probability_measure_v1"
        );
        fs::remove_file(path).unwrap();
    }
}
