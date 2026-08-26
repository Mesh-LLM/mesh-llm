use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use serde::Serialize;

use crate::gguf_metadata::{
    GGUF_TYPE_ARRAY, GGUF_TYPE_BOOL, GGUF_TYPE_FLOAT32, GGUF_TYPE_FLOAT64, GGUF_TYPE_INT8,
    GGUF_TYPE_INT32, GGUF_TYPE_INT64, GGUF_TYPE_STRING, GGUF_TYPE_UINT8, GGUF_TYPE_UINT16,
    GGUF_TYPE_UINT32, GGUF_TYPE_UINT64, GgufKv, write_kv,
};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const ALIGNMENT_DEFAULT: u64 = 32;
const COPY_BUFFER_BYTES: usize = 8 * 1024 * 1024;

/// Appends a converted MTP draft GGUF's tensors to the last shard of a
/// quantized target GGUF, producing a self-contained composite file.
///
/// Existing target tensors are stream-copied byte-for-byte after a rewritten
/// header/tensor-table; GGUF tensor offsets are relative to the data section,
/// so re-serializing the table preserves them. The MTP tensors are appended
/// at aligned offsets past the target data.
#[derive(Debug, Parser)]
pub(crate) struct ComposeMtpArgs {
    /// Last shard of the sharded target GGUF.
    #[arg(long)]
    pub(crate) target_shard: PathBuf,
    /// Standalone converted MTP draft GGUF.
    #[arg(long)]
    pub(crate) mtp_gguf: PathBuf,
    /// Output path for the composite last shard.
    #[arg(long)]
    pub(crate) output: PathBuf,
    /// Composite block index for the MTP layer (renames `blk.0.` → `blk.N.`).
    #[arg(long)]
    pub(crate) mtp_block: u32,
    /// First shard of the target GGUF carrying the global metadata KV
    /// (`*.block_count`, tokenizer). Sharded llama.cpp splits keep the full
    /// KV only in shard 1; pass it to patch block_count/overrides there.
    #[arg(long, requires = "metadata_output")]
    pub(crate) metadata_shard: Option<PathBuf>,
    /// Output path for the patched metadata (first) shard.
    #[arg(long, requires = "metadata_shard")]
    pub(crate) metadata_output: Option<PathBuf>,
    /// Metadata overrides in `key=value` (u32) form. Repeatable.
    #[arg(long = "set-kv")]
    pub(crate) set_kv: Vec<String>,
    /// Skip the automatic `block_count` increment.
    #[arg(long, default_value_t = false)]
    pub(crate) no_bump_block_count: bool,
    #[arg(long)]
    pub(crate) json: bool,
}

#[derive(Serialize)]
struct ComposeReport {
    target_shard: PathBuf,
    mtp_gguf: PathBuf,
    output: PathBuf,
    metadata_shard: Option<String>,
    target_tensors: usize,
    mtp_tensors: usize,
    appended_bytes: u64,
    block_count: Option<u32>,
}

struct GgufFileInfo {
    path: PathBuf,
    version: u32,
    kv: Vec<GgufKv>,
    tensors: Vec<TensorEntry>,
    alignment: u64,
    data_start: u64,
}

#[derive(Clone)]
struct TensorEntry {
    name: String,
    dims: Vec<u64>,
    tensor_type: u32,
    /// Offset relative to the owning file's data section.
    offset: u64,
    /// Resolved byte length; required only for appended (MTP) tensors.
    byte_len: Option<u64>,
    /// Output-section offset once appended to the composite; equals `offset`
    /// for tensors copied verbatim from the target shard.
    output_offset: u64,
}

pub(crate) fn run_compose_mtp(args: ComposeMtpArgs) -> Result<()> {
    let mut target = read_gguf_file_info(&args.target_shard)?;
    let mtp = read_gguf_file_info(&args.mtp_gguf)?;
    ensure!(!mtp.tensors.is_empty(), "MTP GGUF has no tensors");
    let mut appended = prepare_mtp_tensors(&mtp, args.mtp_block)?;
    let kv_overrides = parse_kv_overrides(&args.set_kv)?;
    let mut block_count = None;
    let metadata_shard = match (args.metadata_shard.as_ref(), args.metadata_output.as_ref()) {
        (Some(source), Some(output)) => {
            let mut metadata = read_gguf_file_info(source)?;
            let patched = plan_metadata(
                &mut metadata.kv,
                &mtp.kv,
                &kv_overrides,
                args.no_bump_block_count,
            )?;
            bump_split_tensors_count(&mut metadata.kv, appended.len())?;
            write_patched_metadata_shard(source, output, &metadata)?;
            block_count = patched;
            Some(output.display().to_string())
        }
        _ => None,
    };
    // block_count lives in the metadata shard when one was provided; applying
    // the plan to the last shard too would double-bump it, and split shards
    // carry only split.* KV anyway.
    if metadata_shard.is_none() {
        block_count = plan_metadata(
            &mut target.kv,
            &mtp.kv,
            &kv_overrides,
            args.no_bump_block_count,
        )?;
        bump_split_tensors_count(&mut target.kv, appended.len())?;
    }
    let appended_bytes = write_composite(
        &args.target_shard,
        &args.output,
        &target,
        &mtp,
        &mut appended,
    )?;
    let report = ComposeReport {
        target_shard: args.target_shard.clone(),
        mtp_gguf: args.mtp_gguf.clone(),
        output: args.output.clone(),
        metadata_shard,
        target_tensors: target.tensors.len(),
        mtp_tensors: appended.len(),
        appended_bytes,
        block_count,
    };
    if args.json {
        crate::output::print_json_pretty(&report)?;
    } else {
        crate::output::print_success(format!(
            "composed {}: {} target + {} MTP tensors, {} appended bytes, block_count={:?}",
            report.output.display(),
            report.target_tensors,
            report.mtp_tensors,
            report.appended_bytes,
            report.block_count,
        ));
    }
    Ok(())
}

fn parse_kv_overrides(overrides: &[String]) -> Result<Vec<(String, u32)>> {
    overrides
        .iter()
        .map(|item| {
            let (key, value) = item
                .split_once('=')
                .with_context(|| format!("--set-kv {item:?}"))?;
            Ok((key.to_string(), value.parse::<u32>()?))
        })
        .collect()
}

/// Mutates target metadata in place: applies `--set-kv` overrides and bumps
/// the architecture `block_count` unless disabled. Returns the final value.
/// When the block count grows, per-layer array metadata is extended to match.
fn plan_metadata(
    target_kv: &mut Vec<GgufKv>,
    mtp_kv: &[GgufKv],
    overrides: &[(String, u32)],
    no_bump: bool,
) -> Result<Option<u32>> {
    for (key, value) in overrides {
        apply_override(target_kv, key, *value)?;
    }
    let block_count = match target_kv
        .iter_mut()
        .find(|kv| kv.key().ends_with(".block_count"))
    {
        None => return Ok(None),
        Some(kv) => {
            let value = kv
                .u32_value_mut()
                .context("block_count exists but is not a u32 value")?;
            let previous = *value;
            if !no_bump {
                *value += 1;
            }
            previous
        }
    };
    if !no_bump {
        extend_per_layer_arrays(target_kv, mtp_kv, block_count as usize)?;
        return Ok(Some(block_count + 1));
    }
    Ok(Some(block_count))
}

/// llama.cpp's loader (`get_key_or_arr`) requires per-layer array metadata
/// whose length equals `block_count`. Keys whose array length matches the
/// pre-bump block count gain one entry for the appended MTP layer: taken from
/// the MTP GGUF's matching key when it carries one, otherwise duplicated from
/// the last target layer.
fn extend_per_layer_arrays(
    target_kv: &mut [GgufKv],
    mtp_kv: &[GgufKv],
    layer_count: usize,
) -> Result<()> {
    for kv in target_kv.iter_mut() {
        if per_layer_suffix(kv.key()).is_none() {
            continue;
        }
        extend_array_kv(kv, mtp_kv, layer_count)?;
    }
    Ok(())
}

/// Per-layer keys llama.cpp validates against `n_layer` when stored as arrays.
fn per_layer_suffix(key: &str) -> Option<&'static str> {
    const SUFFIXES: [&str; 3] = [
        ".feed_forward_length",
        ".attention.head_count",
        ".attention.head_count_kv",
    ];
    SUFFIXES.into_iter().find(|suffix| key.ends_with(*suffix))
}

fn extend_array_kv(kv: &mut GgufKv, mtp_kv: &[GgufKv], layer_count: usize) -> Result<()> {
    match kv {
        GgufKv::ArrayU32 { key, value } => {
            if value.len() == layer_count {
                let mtp_value = mtp_layer_integer(mtp_kv, key);
                let fallback = u64::from(value[value.len() - 1]);
                value.push(
                    u32::try_from(mtp_value.unwrap_or(fallback))
                        .context("per-layer array entry overflows uint32")?,
                );
            }
        }
        GgufKv::ArrayI32 { key, value } => {
            if value.len() == layer_count {
                let mtp_value = mtp_layer_integer(mtp_kv, key);
                let fallback = u64::try_from(i64::from(value[value.len() - 1]))
                    .context("negative per-layer array entry")?;
                value.push(
                    i32::try_from(mtp_value.unwrap_or(fallback))
                        .context("per-layer array entry overflows int32")?,
                );
            }
        }
        GgufKv::ArrayF32 { value, .. } => {
            if value.len() == layer_count && !value.is_empty() {
                let last = value[value.len() - 1];
                value.push(last);
            }
        }
        GgufKv::ArrayBool { value, .. } => {
            if value.len() == layer_count && !value.is_empty() {
                let last = value[value.len() - 1];
                value.push(last);
            }
        }
        GgufKv::ArrayString { value, .. } => {
            if value.len() == layer_count && !value.is_empty() {
                let last = value[value.len() - 1].clone();
                value.push(last);
            }
        }
        // Typed-array variants cover only u32/i32/f32/bool/string; other
        // element widths (e.g. the u16 `attention.head_count` the Nemotron
        // converter emits) round-trip as Raw. Layout: element type u32,
        // element count u64, then the elements.
        GgufKv::Raw {
            key,
            value_type,
            bytes,
        } if *value_type == GGUF_TYPE_ARRAY => {
            extend_raw_array_kv(key, bytes, mtp_kv, layer_count)?;
        }
        // Scalar per-layer keys are fine: llama.cpp's `get_key_or_arr`
        // repeats a scalar across all layers, so no extension is needed.
        _ => {}
    }
    Ok(())
}

fn extend_raw_array_kv(
    key: &str,
    bytes: &mut Vec<u8>,
    mtp_kv: &[GgufKv],
    layer_count: usize,
) -> Result<()> {
    ensure!(
        bytes.len() >= 12,
        "per-layer array {key:?} has a truncated header"
    );
    let element_type = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let count = u64::from_le_bytes(bytes[4..12].try_into().unwrap());
    if count != layer_count as u64 {
        return Ok(());
    }
    let element_size = array_element_size(element_type)
        .with_context(|| format!("per-layer array {key:?} element type"))?;
    let data_len = count as usize * element_size;
    ensure!(
        bytes.len() >= 12 + data_len,
        "per-layer array {key:?} is truncated"
    );
    // Duplicate the last element by default; prefer the MTP draft's scalar
    // value when it is representable in the array's element type.
    let mut element = bytes[12 + data_len - element_size..12 + data_len].to_vec();
    if let Some(scalar) = mtp_layer_scalar(mtp_kv, key, element_type) {
        element = scalar;
    }
    let new_count = count + 1;
    bytes[4..12].copy_from_slice(&new_count.to_le_bytes());
    bytes.extend_from_slice(&element);
    Ok(())
}

/// Encodes the MTP draft layer's value for `key` in a raw array's element
/// type, when the MTP GGUF carries a scalar or array of a compatible width.
fn mtp_layer_scalar(mtp_kv: &[GgufKv], key: &str, element_type: u32) -> Option<Vec<u8>> {
    let value = mtp_layer_integer(mtp_kv, key)?;
    match element_type {
        GGUF_TYPE_UINT8 | GGUF_TYPE_INT8 => Some(vec![value as u8]),
        GGUF_TYPE_UINT16 => Some((value as u16).to_le_bytes().to_vec()),
        GGUF_TYPE_UINT32 => Some((value as u32).to_le_bytes().to_vec()),
        GGUF_TYPE_INT32 => Some((value as i32).to_le_bytes().to_vec()),
        _ => None,
    }
}

/// Reads the MTP draft layer's integer value for a per-layer key: a scalar,
/// or the last entry of an array, in any integer representation.
fn mtp_layer_integer(mtp_kv: &[GgufKv], key: &str) -> Option<u64> {
    match mtp_kv.iter().find(|kv| kv.key() == key)? {
        GgufKv::U16 { value, .. } => Some(u64::from(*value)),
        GgufKv::U32 { value, .. } => Some(u64::from(*value)),
        GgufKv::I32 { value, .. } => u64::try_from(i64::from(*value)).ok(),
        GgufKv::U64 { value, .. } => Some(*value),
        GgufKv::ArrayU32 { value, .. } => value.last().map(|v| u64::from(*v)),
        GgufKv::ArrayI32 { value, .. } => {
            value.last().and_then(|v| u64::try_from(i64::from(*v)).ok())
        }
        _ => None,
    }
}

fn apply_override(target_kv: &mut Vec<GgufKv>, key: &str, value: u32) -> Result<()> {
    if let Some(kv) = target_kv.iter_mut().find(|kv| kv.key() == key) {
        let slot = kv.u32_value_mut().with_context(|| {
            format!("--set-kv {key:?} exists but is not a u32 value; refusing to overwrite")
        })?;
        *slot = value;
        return Ok(());
    }
    target_kv.push(GgufKv::u32(key, value));
    Ok(())
}

/// llama.cpp tracks the global tensor total across shards in
/// `split.tensors.count`; appended MTP tensors must be accounted for there or
/// the loader rejects the composite as inconsistent with the tensor names.
fn bump_split_tensors_count(target_kv: &mut [GgufKv], added: usize) -> Result<()> {
    let added = u64::try_from(added).context("appended tensor count does not fit u64")?;
    for kv in target_kv.iter_mut() {
        if kv.key() != "split.tensors.count" {
            continue;
        }
        // Producers disagree on the scalar type: llama.cpp gguf_split writes
        // int32, some converters write uint32/uint64. Bump in the existing
        // type so the patched shard stays byte-compatible with its siblings.
        bump_int_kv(kv, added, "split.tensors.count")?;
    }
    Ok(())
}

/// Adds `delta` to an integer KV value in place, preserving its scalar type.
fn bump_int_kv(kv: &mut GgufKv, delta: u64, key: &str) -> Result<()> {
    match kv {
        GgufKv::U16 { value, .. } => {
            *value = u16::try_from(
                u64::from(*value)
                    .checked_add(delta)
                    .context(overflow(key, "uint16"))?,
            )
            .context(overflow(key, "uint16"))?;
        }
        GgufKv::U32 { value, .. } => {
            *value = u32::try_from(
                u64::from(*value)
                    .checked_add(delta)
                    .context(overflow(key, "uint32"))?,
            )
            .context(overflow(key, "uint32"))?;
        }
        GgufKv::I32 { value, .. } => {
            let bumped = i64::from(*value)
                .checked_add(i64::try_from(delta).context(overflow(key, "int32"))?)
                .context(overflow(key, "int32"))?;
            *value = i32::try_from(bumped).context(overflow(key, "int32"))?;
        }
        GgufKv::U64 { value, .. } => {
            *value = value.checked_add(delta).context(overflow(key, "uint64"))?;
        }
        _ => anyhow::bail!("{key} exists but is not an integer value; refusing to rewrite it"),
    }
    Ok(())
}

fn overflow(key: &str, ty: &str) -> String {
    format!("{key} overflows {ty}")
}

fn prepare_mtp_tensors(mtp: &GgufFileInfo, mtp_block: u32) -> Result<Vec<TensorEntry>> {
    mtp.tensors
        .iter()
        .map(|tensor| {
            let name = rename_mtp_tensor(&tensor.name, mtp_block)?;
            let byte_len =
                tensor_byte_len(&tensor.dims, tensor.tensor_type).with_context(|| {
                    format!("MTP tensor {:?} (type {})", tensor.name, tensor.tensor_type)
                })?;
            Ok(TensorEntry {
                name,
                dims: tensor.dims.clone(),
                tensor_type: tensor.tensor_type,
                offset: tensor.offset,
                byte_len: Some(byte_len),
                output_offset: 0,
            })
        })
        .collect()
}

fn rename_mtp_tensor(name: &str, mtp_block: u32) -> Result<String> {
    if let Some(rest) = name.strip_prefix("blk.0.") {
        return Ok(format!("blk.{mtp_block}.{rest}"));
    }
    // Drafts produced by the pinned llama.cpp converter (`--mtp`) already
    // place the MTP block at its final trunk-relative index; keep verbatim.
    if mtp_block != 0 && name.starts_with(&format!("blk.{mtp_block}.")) {
        return Ok(name.to_string());
    }
    if name.starts_with("blk.") {
        bail!(
            "MTP tensor {name:?} is not anchored at blk.0 or blk.{mtp_block}; refusing to rename"
        );
    }
    Ok(name.to_string())
}

fn read_gguf_file_info(path: &Path) -> Result<GgufFileInfo> {
    let file = File::open(path).with_context(|| format!("open GGUF {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let (version, tensor_count, metadata_count) = read_header(&mut reader)?;
    ensure!(version >= 2, "unsupported GGUF version {version}");
    let mut kv = Vec::with_capacity(metadata_count as usize);
    for _ in 0..metadata_count {
        kv.push(read_kv(&mut reader)?);
    }
    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        tensors.push(read_tensor_entry(&mut reader)?);
    }
    let alignment = kv
        .iter()
        .find_map(|item| match item {
            GgufKv::U32 { key, value } if key == "general.alignment" => Some(*value as u64),
            GgufKv::U64 { key, value } if key == "general.alignment" => Some(*value),
            _ => None,
        })
        .unwrap_or(ALIGNMENT_DEFAULT);
    let position = reader.stream_position()?;
    let data_start = align_to(position, alignment);
    Ok(GgufFileInfo {
        path: path.to_path_buf(),
        version,
        kv,
        tensors,
        alignment,
        data_start,
    })
}

fn read_header<R: Read>(reader: &mut R) -> Result<(u32, u64, u64)> {
    let mut magic = [0_u8; 4];
    reader.read_exact(&mut magic)?;
    ensure!(&magic == GGUF_MAGIC, "not a GGUF file");
    let version = read_u32(reader)?;
    let tensor_count = read_u64(reader)?;
    let metadata_count = read_u64(reader)?;
    Ok((version, tensor_count, metadata_count))
}

fn read_kv<R: Read>(reader: &mut R) -> Result<GgufKv> {
    let key = read_string(reader)?;
    let value_type = read_u32(reader)?;
    let kv = match value_type {
        GGUF_TYPE_UINT8 => read_raw_scalar(reader, key, value_type, 1)?,
        GGUF_TYPE_INT8 => read_raw_scalar(reader, key, value_type, 1)?,
        GGUF_TYPE_UINT16 => GgufKv::U16 {
            key,
            value: read_u16(reader)?,
        },
        GGUF_TYPE_UINT32 => GgufKv::U32 {
            key,
            value: read_u32(reader)?,
        },
        GGUF_TYPE_INT32 => GgufKv::I32 {
            key,
            value: read_u32(reader)? as i32,
        },
        GGUF_TYPE_FLOAT32 => GgufKv::F32 {
            key,
            value: f32::from_le_bytes(read_u32(reader)?.to_le_bytes()),
        },
        GGUF_TYPE_BOOL => GgufKv::Bool {
            key,
            value: read_u8(reader)? != 0,
        },
        GGUF_TYPE_STRING => GgufKv::String {
            key,
            value: read_string(reader)?,
        },
        GGUF_TYPE_UINT64 => GgufKv::U64 {
            key,
            value: read_u64(reader)?,
        },
        GGUF_TYPE_INT64 | GGUF_TYPE_FLOAT64 => read_raw_scalar(reader, key, value_type, 8)?,
        GGUF_TYPE_ARRAY => read_kv_array(reader, key)?,
        other => bail!("unsupported GGUF metadata value type {other}"),
    };
    Ok(kv)
}

fn read_kv_array<R: Read>(reader: &mut R, key: String) -> Result<GgufKv> {
    let element_type = read_u32(reader)?;
    let len = read_u64(reader)? as usize;
    ensure!(len > 0, "GGUF array metadata {key:?} is empty");
    let kv = match element_type {
        GGUF_TYPE_UINT32 => GgufKv::ArrayU32 {
            key,
            value: (0..len).map(|_| read_u32(reader)).collect::<Result<_>>()?,
        },
        GGUF_TYPE_INT32 => GgufKv::ArrayI32 {
            key,
            value: (0..len)
                .map(|_| Ok(read_u32(reader)? as i32))
                .collect::<Result<_>>()?,
        },
        GGUF_TYPE_FLOAT32 => GgufKv::ArrayF32 {
            key,
            value: (0..len)
                .map(|_| Ok(f32::from_le_bytes(read_u32(reader)?.to_le_bytes())))
                .collect::<Result<_>>()?,
        },
        GGUF_TYPE_STRING => GgufKv::ArrayString {
            key,
            value: (0..len)
                .map(|_| read_string(reader))
                .collect::<Result<_>>()?,
        },
        GGUF_TYPE_BOOL => GgufKv::ArrayBool {
            key,
            value: (0..len)
                .map(|_| Ok(read_u8(reader)? != 0))
                .collect::<Result<_>>()?,
        },
        _ => {
            let element_size = array_element_size(element_type)?;
            // `write_kv` emits only the scalar type tag for `GgufKv::Raw`, so
            // the array header (element type + count) must be preserved inside
            // `bytes` or the following KV entries shift on round-trip.
            let mut bytes = Vec::with_capacity(12 + len * element_size);
            bytes.extend_from_slice(&element_type.to_le_bytes());
            bytes.extend_from_slice(&(len as u64).to_le_bytes());
            bytes.resize(12 + len * element_size, 0);
            reader.read_exact(&mut bytes[12..])?;
            GgufKv::Raw {
                key,
                value_type: GGUF_TYPE_ARRAY,
                bytes,
            }
        }
    };
    Ok(kv)
}

fn array_element_size(element_type: u32) -> Result<usize> {
    Ok(match element_type {
        GGUF_TYPE_UINT8 | GGUF_TYPE_BOOL => 1,
        GGUF_TYPE_UINT16 => 2,
        GGUF_TYPE_UINT32 | GGUF_TYPE_INT32 | GGUF_TYPE_FLOAT32 => 4,
        GGUF_TYPE_UINT64 | GGUF_TYPE_INT64 | GGUF_TYPE_FLOAT64 => 8,
        other => bail!("unsupported GGUF array element type {other}"),
    })
}

fn read_raw_scalar<R: Read>(
    reader: &mut R,
    key: String,
    value_type: u32,
    size: usize,
) -> Result<GgufKv> {
    let mut bytes = vec![0_u8; size];
    reader.read_exact(&mut bytes)?;
    Ok(GgufKv::Raw {
        key,
        value_type,
        bytes,
    })
}

fn read_tensor_entry<R: Read>(reader: &mut R) -> Result<TensorEntry> {
    let name = read_string(reader)?;
    let n_dims = read_u32(reader)?;
    ensure!(
        n_dims > 0 && n_dims <= 8,
        "tensor {name:?} has bad rank {n_dims}"
    );
    let dims = (0..n_dims)
        .map(|_| read_u64(reader))
        .collect::<Result<Vec<_>>>()?;
    let tensor_type = read_u32(reader)?;
    let offset = read_u64(reader)?;
    Ok(TensorEntry {
        name,
        dims,
        tensor_type,
        offset,
        byte_len: None,
        output_offset: offset,
    })
}

/// ggml (block_size, block_bytes) pairs for tensor types that may appear in a
/// converted MTP draft GGUF. Target-shard tensors never need this: their data
/// is copied wholesale and their offsets are preserved verbatim.
fn tensor_byte_len(dims: &[u64], tensor_type: u32) -> Result<u64> {
    let (block_elements, block_bytes) = match tensor_type {
        0 => (1, 4),      // F32
        1 => (1, 2),      // F16
        30 => (1, 2),     // BF16
        2 => (32, 18),    // Q4_0
        3 => (32, 20),    // Q4_1
        6 => (32, 22),    // Q5_0
        7 => (32, 24),    // Q5_1
        8 => (32, 34),    // Q8_0
        10 => (256, 84),  // Q2_K
        11 => (256, 110), // Q3_K
        12 => (256, 144), // Q4_K
        13 => (256, 176), // Q5_K
        14 => (256, 210), // Q6_K
        15 => (256, 292), // Q8_K
        16 => (256, 66),  // IQ2_XXS
        17 => (256, 74),  // IQ2_XS
        18 => (256, 104), // IQ3_XXS
        19 => (256, 38),  // IQ1_S
        20 => (32, 18),   // IQ4_NL
        21 => (256, 110), // IQ3_S
        22 => (256, 82),  // IQ2_S
        23 => (256, 144), // IQ4_XS
        24 => (256, 42),  // IQ1_M
        other => bail!("cannot size ggml tensor type {other}"),
    };
    let elements: u64 = dims.iter().product::<u64>().max(1);
    Ok(elements.div_ceil(block_elements) * block_bytes)
}

fn write_composite(
    target_path: &Path,
    output_path: &Path,
    target: &GgufFileInfo,
    mtp: &GgufFileInfo,
    appended: &mut [TensorEntry],
) -> Result<u64> {
    let mut target_file = File::open(target_path)?;
    let file_len = target_file.metadata()?.len();
    let target_data_len = file_len - target.data_start;

    let next_offset = assign_appended_offsets(appended, target_data_len, target.alignment);
    let data_section_len = next_offset;

    let output = File::create(output_path)?;
    let mut writer = BufWriter::with_capacity(COPY_BUFFER_BYTES, output);
    write_header_and_table(&mut writer, target, appended)?;
    pad_to_data_start(&mut writer, target.alignment)?;
    copy_section(
        &mut target_file,
        &mut writer,
        target.data_start,
        target_data_len,
    )?;
    pad_to_alignment(&mut writer, target_data_len, target.alignment)?;
    append_mtp_tensors(&mut writer, mtp, appended)?;
    writer.flush()?;
    Ok(data_section_len - target_data_len)
}

/// Rewrites the metadata (first) shard with the patched KV and tensor table,
/// byte-copying its data section. Tensor offsets are data-section-relative,
/// so a re-serialized table preserves them.
fn write_patched_metadata_shard(
    source_path: &Path,
    output_path: &Path,
    metadata: &GgufFileInfo,
) -> Result<()> {
    let mut source_file = File::open(source_path)?;
    let file_len = source_file.metadata()?.len();
    let data_len = file_len - metadata.data_start;
    let output = File::create(output_path)?;
    let mut writer = BufWriter::with_capacity(COPY_BUFFER_BYTES, output);
    write_header_and_table(&mut writer, metadata, &[])?;
    pad_to_data_start(&mut writer, metadata.alignment)?;
    copy_section(&mut source_file, &mut writer, metadata.data_start, data_len)?;
    writer.flush()?;
    Ok(())
}

fn assign_appended_offsets(
    appended: &mut [TensorEntry],
    target_data_len: u64,
    alignment: u64,
) -> u64 {
    let mut next_offset = target_data_len;
    for tensor in appended {
        let aligned = align_to(next_offset, alignment);
        tensor.output_offset = aligned;
        next_offset = aligned + tensor.byte_len.unwrap_or(0);
    }
    next_offset
}

fn write_header_and_table<W: Write>(
    writer: &mut W,
    target: &GgufFileInfo,
    appended: &[TensorEntry],
) -> Result<()> {
    writer.write_all(GGUF_MAGIC)?;
    writer.write_all(&target.version.to_le_bytes())?;
    writer.write_all(&((target.tensors.len() + appended.len()) as u64).to_le_bytes())?;
    writer.write_all(&(target.kv.len() as u64).to_le_bytes())?;
    for kv in &target.kv {
        write_kv(writer, kv)?;
    }
    for tensor in target.tensors.iter().chain(appended.iter()) {
        write_tensor_entry(writer, tensor)?;
    }
    Ok(())
}

fn write_tensor_entry<W: Write>(writer: &mut W, tensor: &TensorEntry) -> Result<()> {
    write_string(writer, &tensor.name)?;
    writer.write_all(&(tensor.dims.len() as u32).to_le_bytes())?;
    for dim in &tensor.dims {
        writer.write_all(&dim.to_le_bytes())?;
    }
    writer.write_all(&tensor.tensor_type.to_le_bytes())?;
    writer.write_all(&tensor.output_offset.to_le_bytes())?;
    Ok(())
}

fn append_mtp_tensors<W: Write>(
    writer: &mut W,
    mtp: &GgufFileInfo,
    renamed: &[TensorEntry],
) -> Result<()> {
    let mut mtp_file = BufReader::with_capacity(COPY_BUFFER_BYTES, File::open(&mtp.path)?);
    for tensor in renamed {
        mtp_file.seek(SeekFrom::Start(mtp.data_start + tensor.offset))?;
        copy_exact(&mut mtp_file, writer, tensor.byte_len.unwrap_or(0))?;
        pad_to_alignment(writer, tensor.byte_len.unwrap_or(0), mtp.alignment)?;
    }
    Ok(())
}

fn copy_section<R: Read + Seek, W: Write>(
    reader: &mut R,
    writer: &mut W,
    offset: u64,
    len: u64,
) -> Result<()> {
    reader.seek(SeekFrom::Start(offset))?;
    copy_exact(reader, writer, len)
}

fn copy_exact<R: Read, W: Write>(reader: &mut R, writer: &mut W, len: u64) -> Result<()> {
    let mut buffer = vec![0_u8; COPY_BUFFER_BYTES];
    let mut remaining = len;
    while remaining > 0 {
        let chunk = remaining.min(buffer.len() as u64) as usize;
        reader.read_exact(&mut buffer[..chunk])?;
        writer.write_all(&buffer[..chunk])?;
        remaining -= chunk as u64;
    }
    Ok(())
}

/// Pads from the current writer position to the aligned start of the tensor
/// data section, so re-serialized tables of differing length keep offsets
/// consistent with the alignment-relative GGUF offset convention.
fn pad_to_data_start<W: Write + Seek>(writer: &mut W, alignment: u64) -> Result<()> {
    let position = writer.stream_position()?;
    pad_to_alignment(writer, position, alignment)
}

fn pad_to_alignment<W: Write>(writer: &mut W, written: u64, alignment: u64) -> Result<()> {
    let padding = align_to(written, alignment) - written;
    const ZERO: [u8; 64] = [0_u8; 64];
    let mut remaining = padding;
    while remaining > 0 {
        let chunk = remaining.min(ZERO.len() as u64) as usize;
        writer.write_all(&ZERO[..chunk])?;
        remaining -= chunk as u64;
    }
    Ok(())
}

fn align_to(value: u64, alignment: u64) -> u64 {
    value.div_ceil(alignment) * alignment
}

fn read_u8<R: Read>(reader: &mut R) -> Result<u8> {
    let mut buffer = [0_u8; 1];
    reader.read_exact(&mut buffer)?;
    Ok(buffer[0])
}

fn read_u16<R: Read>(reader: &mut R) -> Result<u16> {
    let mut buffer = [0_u8; 2];
    reader.read_exact(&mut buffer)?;
    Ok(u16::from_le_bytes(buffer))
}

fn read_u32<R: Read>(reader: &mut R) -> Result<u32> {
    let mut buffer = [0_u8; 4];
    reader.read_exact(&mut buffer)?;
    Ok(u32::from_le_bytes(buffer))
}

fn read_u64<R: Read>(reader: &mut R) -> Result<u64> {
    let mut buffer = [0_u8; 8];
    reader.read_exact(&mut buffer)?;
    Ok(u64::from_le_bytes(buffer))
}

fn read_string<R: Read>(reader: &mut R) -> Result<String> {
    let len = read_u64(reader)? as usize;
    let mut buffer = vec![0_u8; len];
    reader.read_exact(&mut buffer)?;
    Ok(String::from_utf8(buffer)?)
}

fn write_string(writer: &mut impl Write, value: &str) -> Result<()> {
    writer.write_all(&(value.len() as u64).to_le_bytes())?;
    writer.write_all(value.as_bytes())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    fn put_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn put_u64(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    /// Minimal GGUF v2 writer for fixtures: F32 tensors, 32B alignment.
    fn write_fixture_gguf(path: &Path, kv: &[GgufKv], tensors: &[(&str, Vec<u32>, Vec<u8>)]) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(GGUF_MAGIC);
        put_u32(&mut bytes, 2);
        put_u64(&mut bytes, tensors.len() as u64);
        put_u64(&mut bytes, kv.len() as u64);
        for kv in kv {
            write_kv(&mut bytes, kv).unwrap();
        }
        for (name, dims, _) in tensors {
            put_string(&mut bytes, name);
            put_u32(&mut bytes, dims.len() as u32);
            for dim in dims.iter() {
                put_u64(&mut bytes, *dim as u64);
            }
            put_u32(&mut bytes, 0); // F32
            put_u64(&mut bytes, 0); // placeholder offset
        }
        let data_start = align_to(bytes.len() as u64, ALIGNMENT_DEFAULT);
        bytes.extend(std::iter::repeat_n(
            0,
            (data_start - bytes.len() as u64) as usize,
        ));
        // Patch offsets relative to data start, append data with alignment.
        let cursor = 24 + kv.iter().map(serialized_kv_len).sum::<usize>();
        let mut data_cursor = 0_u64;
        for (index, (name, dims, data)) in tensors.iter().enumerate() {
            let entry_len = 8 + name.len() + 4 + 8 * dims.len() + 4 + 8;
            let entry_start = cursor + index * entry_len;
            let offset_patch = entry_start + 8 + name.len() + 4 + 8 * dims.len() + 4;
            let offset_bytes = data_cursor.to_le_bytes();
            for (i, b) in offset_bytes.iter().enumerate() {
                bytes[offset_patch + i] = *b;
            }
            bytes.extend_from_slice(data);
            let padded = align_to(data.len() as u64, ALIGNMENT_DEFAULT);
            bytes.extend(std::iter::repeat_n(
                0,
                (padded - data.len() as u64) as usize,
            ));
            data_cursor = padded;
        }
        std::fs::write(path, bytes).unwrap();
    }

    fn serialized_kv_len(kv: &GgufKv) -> usize {
        let mut buffer = Vec::new();
        write_kv(&mut buffer, kv).unwrap();
        buffer.len()
    }

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let dir =
                std::env::temp_dir().join(format!("compose-mtp-test-{tag}-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }

        fn path(&self, name: &str) -> PathBuf {
            self.0.join(name)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn composes_mtp_tensors_and_bumps_block_count() {
        let dir = TempDir::new("basic");
        let target_path = dir.path("target-00002-of-00002.gguf");
        let mtp_path = dir.path("mtp.gguf");
        let output_path = dir.path("composite.gguf");
        write_fixture_gguf(
            &target_path,
            &[GgufKv::u32("nemotron.block_count", 88)],
            &[
                (
                    "blk.0.attn.weight",
                    vec![4],
                    vec![1.0_f32.to_le_bytes()[0]; 16],
                ),
                ("output.weight", vec![8], vec![2_u8; 32]),
            ],
        );
        write_fixture_gguf(
            &mtp_path,
            &[GgufKv::u32("nemotron.block_count", 1)],
            &[("blk.0.nextn.eh_proj.weight", vec![4], vec![7_u8; 16])],
        );

        run_compose_mtp(ComposeMtpArgs {
            target_shard: target_path.clone(),
            mtp_gguf: mtp_path.clone(),
            output: output_path.clone(),
            mtp_block: 88,
            metadata_shard: None,
            metadata_output: None,
            set_kv: vec![],
            no_bump_block_count: false,
            json: true,
        })
        .unwrap();

        let composite = read_gguf_file_info(&output_path).unwrap();
        assert_eq!(composite.version, 2);
        assert_eq!(
            composite
                .kv
                .iter()
                .find_map(|kv| match kv {
                    GgufKv::U32 { key, value } if key == "nemotron.block_count" => Some(*value),
                    _ => None,
                })
                .unwrap(),
            89
        );
        let names: Vec<&str> = composite.tensors.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "blk.0.attn.weight",
                "output.weight",
                "blk.88.nextn.eh_proj.weight"
            ]
        );
        // Existing target tensor bytes must be identical.
        let target = read_gguf_file_info(&target_path).unwrap();
        let target_bytes = std::fs::read(&target_path).unwrap();
        let output_bytes = std::fs::read(&output_path).unwrap();
        let t0 = &target_bytes[target.data_start as usize..];
        let c0 = &output_bytes[composite.data_start as usize..];
        assert_eq!(&t0[..64], &c0[..64]);
        // Appended MTP tensor bytes must match the source, at its new offset.
        let mtp_file = read_gguf_file_info(&mtp_path).unwrap();
        let mtp_bytes = std::fs::read(&mtp_path).unwrap();
        let appended = composite.tensors.last().unwrap();
        let src = &mtp_bytes[mtp_file.data_start as usize..][..16];
        let dst = &c0[appended.offset as usize..][..16];
        assert_eq!(src, dst);
        assert!(appended.offset >= 64);
    }

    #[test]
    fn set_kv_overrides_apply_without_touching_other_metadata() {
        let dir = TempDir::new("setkv");
        let target_path = dir.path("target.gguf");
        let mtp_path = dir.path("mtp.gguf");
        let output_path = dir.path("out.gguf");
        write_fixture_gguf(
            &target_path,
            &[GgufKv::u32("arch.block_count", 2)],
            &[("blk.0.weight", vec![2], vec![9_u8; 8])],
        );
        write_fixture_gguf(
            &mtp_path,
            &[GgufKv::u32("arch.block_count", 1)],
            &[("blk.0.nextn.weight", vec![2], vec![5_u8; 8])],
        );
        run_compose_mtp(ComposeMtpArgs {
            target_shard: target_path,
            mtp_gguf: mtp_path,
            output: output_path.clone(),
            mtp_block: 2,
            metadata_shard: None,
            metadata_output: None,
            set_kv: vec!["arch.attention.layer_count=2".to_string()],
            no_bump_block_count: true,
            json: false,
        })
        .unwrap();
        let composite = read_gguf_file_info(&output_path).unwrap();
        let get = |key: &str| {
            composite.kv.iter().find_map(|kv| match kv {
                GgufKv::U32 { key: k, value } if k == key => Some(*value),
                _ => None,
            })
        };
        assert_eq!(get("arch.block_count"), Some(2));
        assert_eq!(get("arch.attention.layer_count"), Some(2));
    }

    #[test]
    fn refuses_unanchored_mtp_blocks() {
        let dir = TempDir::new("rename");
        let mtp_path = dir.path("mtp.gguf");
        write_fixture_gguf(
            &mtp_path,
            &[GgufKv::u32("arch.block_count", 1)],
            &[("blk.7.nextn.weight", vec![2], vec![1_u8; 8])],
        );
        let mtp = read_gguf_file_info(&mtp_path).unwrap();
        assert!(prepare_mtp_tensors(&mtp, 88).is_err());
        // Drafts already positioned at the target block are accepted verbatim.
        write_fixture_gguf(
            &mtp_path,
            &[GgufKv::u32("arch.block_count", 1)],
            &[("blk.88.nextn.weight", vec![2], vec![1_u8; 8])],
        );
        let mtp = read_gguf_file_info(&mtp_path).unwrap();
        let prepared = prepare_mtp_tensors(&mtp, 88).unwrap();
        assert_eq!(prepared[0].name, "blk.88.nextn.weight");
    }

    #[test]
    fn patches_metadata_shard_for_sharded_targets() {
        let dir = TempDir::new("sharded");
        let first_path = dir.path("target-00001-of-00002.gguf");
        let last_path = dir.path("target-00002-of-00002.gguf");
        let mtp_path = dir.path("mtp.gguf");
        let patched_first = dir.path("composite-00001-of-00002.gguf");
        let composite_last = dir.path("composite-00002-of-00002.gguf");
        // Shard 1 carries the global KV (block_count, tokenizer-ish string,
        // global split tensor count) and no tensors; shard 2 carries tensors
        // with split-only KV.
        write_fixture_gguf(
            &first_path,
            &[
                GgufKv::u32("arch.block_count", 88),
                // The real split writer emits this as i32 (llama.cpp
                // gguf_split convention), not u32.
                GgufKv::i32("split.tensors.count", 1),
            ],
            &[],
        );
        write_fixture_gguf(
            &last_path,
            &[GgufKv::u32("split.no", 1)],
            &[("blk.87.weight", vec![2], vec![3_u8; 8])],
        );
        write_fixture_gguf(
            &mtp_path,
            &[GgufKv::u32("arch.block_count", 1)],
            &[("blk.0.nextn.weight", vec![2], vec![5_u8; 8])],
        );
        run_compose_mtp(ComposeMtpArgs {
            target_shard: last_path.clone(),
            mtp_gguf: mtp_path,
            output: composite_last.clone(),
            mtp_block: 88,
            metadata_shard: Some(first_path.clone()),
            metadata_output: Some(patched_first.clone()),
            set_kv: vec!["arch.nextn_predict_layers=1".to_string()],
            no_bump_block_count: false,
            json: false,
        })
        .unwrap();

        // Metadata shard: block_count bumped once, override added, no tensors.
        let patched = read_gguf_file_info(&patched_first).unwrap();
        let get = |key: &str| {
            patched.kv.iter().find_map(|kv| match kv {
                GgufKv::U32 { key: k, value } if k == key => Some(*value),
                _ => None,
            })
        };
        let get_i32 = |key: &str| {
            patched.kv.iter().find_map(|kv| match kv {
                GgufKv::I32 { key: k, value } if k == key => Some(*value),
                _ => None,
            })
        };
        assert_eq!(get("arch.block_count"), Some(89));
        assert_eq!(get("arch.nextn_predict_layers"), Some(1));
        // The appended MTP tensor must be accounted for in the global split
        // tensor total or llama.cpp rejects the composite shards. The i32
        // scalar type must be preserved (byte-compatible with siblings).
        assert_eq!(get_i32("split.tensors.count"), Some(2));
        assert!(patched.tensors.is_empty());

        // Last shard: split KV untouched (no block_count to double-bump),
        // MTP tensor appended and renamed.
        let composite = read_gguf_file_info(&composite_last).unwrap();
        assert_eq!(
            composite
                .kv
                .iter()
                .find_map(|kv| match kv {
                    GgufKv::U32 { key, value } if key == "split.no" => Some(*value),
                    _ => None,
                })
                .unwrap(),
            1
        );
        let names: Vec<&str> = composite.tensors.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, vec!["blk.87.weight", "blk.88.nextn.weight"]);
        // The override must not leak into the split-only shard KV set.
        assert!(
            composite
                .kv
                .iter()
                .all(|kv| kv.key() != "arch.nextn_predict_layers")
        );
    }

    #[test]
    fn raw_array_metadata_round_trips_with_array_header() {
        let dir = TempDir::new("rawarray");
        let target_path = dir.path("target.gguf");
        let mtp_path = dir.path("mtp.gguf");
        let output_path = dir.path("out.gguf");
        // Raw u16 array (no typed variant) followed by a regular u32 KV; a
        // lost array header would shift every following metadata entry.
        let raw_array = GgufKv::Raw {
            key: "arch.raw_u16_array".to_string(),
            value_type: GGUF_TYPE_ARRAY,
            bytes: {
                let mut bytes = GGUF_TYPE_UINT16.to_le_bytes().to_vec();
                bytes.extend_from_slice(&2_u64.to_le_bytes());
                bytes.extend_from_slice(&[0xAB, 0xCD, 0xEF, 0x01]);
                bytes
            },
        };
        write_fixture_gguf(
            &target_path,
            &[raw_array.clone(), GgufKv::u32("arch.block_count", 1)],
            &[("blk.0.weight", vec![2], vec![9_u8; 8])],
        );
        write_fixture_gguf(
            &mtp_path,
            &[GgufKv::u32("arch.block_count", 1)],
            &[("blk.0.nextn.weight", vec![2], vec![5_u8; 8])],
        );
        run_compose_mtp(ComposeMtpArgs {
            target_shard: target_path,
            mtp_gguf: mtp_path,
            output: output_path.clone(),
            mtp_block: 1,
            metadata_shard: None,
            metadata_output: None,
            set_kv: vec![],
            no_bump_block_count: true,
            json: false,
        })
        .unwrap();
        let composite = read_gguf_file_info(&output_path).unwrap();
        assert!(composite.kv.contains(&raw_array));
        assert!(
            composite
                .kv
                .iter()
                .any(|kv| matches!(kv, GgufKv::U32 { key, value: 1 } if key == "arch.block_count"))
        );
    }

    #[test]
    fn bumps_per_layer_array_metadata_with_block_count() {
        let dir = TempDir::new("perlayer");
        let target_path = dir.path("target.gguf");
        let mtp_path = dir.path("mtp.gguf");
        let output_path = dir.path("out.gguf");
        // nemotron-style metadata: per-layer arrays sized to block_count. The
        // MTP draft carries its own (different) ffn length but no head_count
        // override, exercising both value sources.
        write_fixture_gguf(
            &target_path,
            &[
                GgufKv::u32("arch.block_count", 2),
                GgufKv::array_u32("arch.feed_forward_length", vec![512, 1024]),
                GgufKv::array_u32("arch.attention.head_count", vec![8, 8]),
            ],
            &[("blk.1.weight", vec![2], vec![9_u8; 8])],
        );
        write_fixture_gguf(
            &mtp_path,
            &[
                GgufKv::u32("arch.block_count", 1),
                GgufKv::u32("arch.feed_forward_length", 2048),
            ],
            &[("blk.0.nextn.weight", vec![2], vec![5_u8; 8])],
        );
        run_compose_mtp(ComposeMtpArgs {
            target_shard: target_path,
            mtp_gguf: mtp_path,
            output: output_path.clone(),
            mtp_block: 2,
            metadata_shard: None,
            metadata_output: None,
            set_kv: vec![],
            no_bump_block_count: false,
            json: false,
        })
        .unwrap();
        let composite = read_gguf_file_info(&output_path).unwrap();
        let get_arr = |key: &str| {
            composite.kv.iter().find_map(|kv| match kv {
                GgufKv::ArrayU32 { key: k, value } if k == key => Some(value.clone()),
                _ => None,
            })
        };
        // ffn takes the MTP draft's value; head_count has no MTP source and
        // duplicates the last layer.
        assert_eq!(
            get_arr("arch.feed_forward_length"),
            Some(vec![512, 1024, 2048])
        );
        assert_eq!(get_arr("arch.attention.head_count"), Some(vec![8, 8, 8]));
    }

    #[test]
    fn extends_raw_u16_per_layer_arrays() {
        let dir = TempDir::new("rawperlayer");
        let target_path = dir.path("target.gguf");
        let mtp_path = dir.path("mtp.gguf");
        let output_path = dir.path("out.gguf");
        // Real-world Nemotron shape: `attention.head_count` is a u16 array,
        // which round-trips through the Raw KV variant (no typed u16 array).
        let head_count = {
            let mut bytes = GGUF_TYPE_UINT16.to_le_bytes().to_vec();
            bytes.extend_from_slice(&2_u64.to_le_bytes());
            for value in [32_u16, 32_u16] {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            GgufKv::Raw {
                key: "arch.attention.head_count".to_string(),
                value_type: GGUF_TYPE_ARRAY,
                bytes,
            }
        };
        write_fixture_gguf(
            &target_path,
            &[
                GgufKv::u32("arch.block_count", 2),
                GgufKv::array_u32("arch.feed_forward_length", vec![512, 1024]),
                head_count,
            ],
            &[("blk.1.weight", vec![2], vec![9_u8; 8])],
        );
        write_fixture_gguf(
            &mtp_path,
            &[
                GgufKv::u32("arch.block_count", 1),
                GgufKv::u32("arch.feed_forward_length", 2048),
                // Scalar u16 in the draft; must be encoded into the raw u16
                // array's element type.
                GgufKv::U16 {
                    key: "arch.attention.head_count".to_string(),
                    value: 8,
                },
            ],
            &[("blk.0.nextn.weight", vec![2], vec![5_u8; 8])],
        );
        run_compose_mtp(ComposeMtpArgs {
            target_shard: target_path,
            mtp_gguf: mtp_path,
            output: output_path.clone(),
            mtp_block: 2,
            metadata_shard: None,
            metadata_output: None,
            set_kv: vec![],
            no_bump_block_count: false,
            json: false,
        })
        .unwrap();
        let composite = read_gguf_file_info(&output_path).unwrap();
        assert_eq!(
            composite
                .kv
                .iter()
                .find_map(|kv| match kv {
                    GgufKv::ArrayU32 { key: k, value } if k == "arch.feed_forward_length" =>
                        Some(value.clone()),
                    _ => None,
                })
                .unwrap(),
            vec![512, 1024, 2048]
        );
        let raw = composite
            .kv
            .iter()
            .find(|kv| kv.key() == "arch.attention.head_count")
            .unwrap();
        let GgufKv::Raw { bytes, .. } = raw else {
            panic!("expected raw array");
        };
        assert_eq!(
            u64::from_le_bytes(bytes[4..12].try_into().unwrap()),
            3,
            "element count must be bumped"
        );
        assert_eq!(
            u16::from_le_bytes(bytes[16..18].try_into().unwrap()),
            8,
            "MTP scalar must be appended in the array's element type"
        );
    }

    #[test]
    fn set_kv_override_errors_on_non_u32_existing_key() {
        let mut kv = vec![GgufKv::u64("arch.block_count", 88)];
        assert!(apply_override(&mut kv, "arch.block_count", 89).is_err());
        // New keys are still appended.
        assert!(apply_override(&mut kv, "arch.nextn_predict_layers", 1).is_ok());
        assert!(
            kv.iter()
                .any(|item| item.key() == "arch.nextn_predict_layers")
        );
    }
}
