//! Byte-preserving payload-part writer for v2 packages.
//!
//! The native slice writer emits one whole-layer GGUF per artifact, which is
//! the right shape for ordinary layers. HF Jobs split packages must also cap
//! per-artifact bytes (the kubelet evicts pods above 50G of container-local
//! ephemeral storage), so oversized layers are subdivided into part artifacts
//! by the Rust planner. This module writes those parts directly from the
//! independent source inventory, streaming exact tensor payload bytes — no
//! per-shard scratch parts, no merge step, no transient 2x copy — while
//! emitting a legal GGUF whose descriptor table the native carrier writer and
//! the verifier both consume unchanged.
//!
//! Source metadata KV entries are copied as raw wire bytes rather than
//! re-encoded from parsed values: llama.cpp type-checks well-known keys
//! (`general.alignment` must be u32), and a serde_json round trip widens
//! them to u64, which the native reader rejects.
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use skippy_model::gguf_catalog::GgufTensor;
use skippy_package_format::{Tensor, TensorStorage};

use crate::source_inventory::SourceInventory;

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_VERSION: u32 = 3;
const GGUF_TYPE_U8: u32 = 0;
const GGUF_TYPE_I8: u32 = 1;
const GGUF_TYPE_U16: u32 = 2;
const GGUF_TYPE_I16: u32 = 3;
const GGUF_TYPE_U32: u32 = 4;
const GGUF_TYPE_I32: u32 = 5;
const GGUF_TYPE_F32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_U64: u32 = 10;
const GGUF_TYPE_I64: u32 = 11;
const GGUF_TYPE_F64: u32 = 12;
const COPY_BUFFER_BYTES: usize = 4 * 1024 * 1024;
const GENERAL_ALIGNMENT_KEY: &str = "general.alignment";
const SPLIT_BOOKKEEPING_KEYS: [&str; 3] = ["split.no", "split.count", "split.tensors.count"];

/// The primary shard's metadata section re-encoded as raw GGUF wire bytes with
/// split bookkeeping keys removed and `general.alignment` guaranteed present.
struct SourceMetadataWire {
    entry_count: u64,
    entries: Vec<u8>,
}

fn source_metadata_wire_bytes(
    shard: &crate::source_inventory::SourceShard,
) -> Result<SourceMetadataWire> {
    let mut reader = WireReader::open(&shard.path)?;
    let magic = reader.read_bytes(4)?;
    ensure!(
        magic == GGUF_MAGIC,
        "{} is not a GGUF file",
        shard.path.display()
    );
    reader.read_u32()?; // version; the inventory already validated it
    reader.read_u64()?; // tensor count
    let metadata_count = reader.read_u64()?;
    let mut wire = SourceMetadataWire {
        entry_count: 0,
        entries: Vec::new(),
    };
    let mut has_alignment = false;
    for _ in 0..metadata_count {
        let key_wire = reader.read_string_wire()?;
        let key = std::str::from_utf8(&key_wire[8..])
            .context("GGUF metadata key is not UTF-8")?
            .to_string();
        let value_type = reader.read_u32()?;
        let value_wire = reader.read_value_wire(value_type)?;
        if SPLIT_BOOKKEEPING_KEYS.contains(&key.as_str()) {
            continue;
        }
        has_alignment |= key == GENERAL_ALIGNMENT_KEY;
        wire.entries.extend_from_slice(&key_wire);
        extend_u32(&mut wire.entries, value_type);
        wire.entries.extend_from_slice(&value_wire);
        wire.entry_count += 1;
    }
    if !has_alignment {
        // The catalog reader defaults a missing alignment to 32; the part
        // inherits the same guarantee with the u32 type llama.cpp expects.
        write_string(&mut wire.entries, GENERAL_ALIGNMENT_KEY);
        extend_u32(&mut wire.entries, GGUF_TYPE_U32);
        extend_u32(&mut wire.entries, shard.directory.alignment as u32);
        wire.entry_count += 1;
    }
    Ok(wire)
}

/// Minimal sequential GGUF reader that keeps every consumed value as its exact
/// wire bytes. The source was already validated by the catalog reader when the
/// inventory was built, so this walker only needs to agree on value sizes.
struct WireReader {
    reader: File,
}

impl WireReader {
    fn open(path: &Path) -> Result<Self> {
        Ok(Self {
            reader: File::open(path).with_context(|| format!("open source {}", path.display()))?,
        })
    }

    fn read_bytes(&mut self, length: usize) -> Result<Vec<u8>> {
        let mut bytes = vec![0_u8; length];
        self.reader
            .read_exact(&mut bytes)
            .with_context(|| format!("read {} wire bytes", length))?;
        Ok(bytes)
    }

    fn read_u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.read_bytes(4)?.try_into().unwrap()))
    }

    fn read_u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.read_bytes(8)?.try_into().unwrap()))
    }

    /// A length-prefixed string, returned as its complete wire encoding
    /// (length bytes included).
    fn read_string_wire(&mut self) -> Result<Vec<u8>> {
        let length = self.read_u64()?;
        let length = usize::try_from(length).context("GGUF string length exceeds usize")?;
        let mut wire = Vec::with_capacity(8 + length);
        wire.extend_from_slice(&(length as u64).to_le_bytes());
        wire.extend_from_slice(&self.read_bytes(length)?);
        Ok(wire)
    }

    fn read_value_wire(&mut self, value_type: u32) -> Result<Vec<u8>> {
        match value_type {
            GGUF_TYPE_U8 | GGUF_TYPE_I8 | GGUF_TYPE_BOOL => self.read_bytes(1),
            GGUF_TYPE_U16 | GGUF_TYPE_I16 => self.read_bytes(2),
            GGUF_TYPE_U32 | GGUF_TYPE_I32 | GGUF_TYPE_F32 => self.read_bytes(4),
            GGUF_TYPE_U64 | GGUF_TYPE_I64 | GGUF_TYPE_F64 => self.read_bytes(8),
            GGUF_TYPE_STRING => self.read_string_wire(),
            GGUF_TYPE_ARRAY => {
                let element_type = self.read_u32()?;
                ensure!(
                    element_type != GGUF_TYPE_ARRAY,
                    "nested GGUF metadata arrays are unsupported"
                );
                let count = self.read_u64()?;
                let mut wire = Vec::new();
                extend_u32(&mut wire, element_type);
                wire.extend_from_slice(&count.to_le_bytes());
                for _ in 0..count {
                    wire.extend_from_slice(&self.read_value_wire(element_type)?);
                }
                Ok(wire)
            }
            other => anyhow::bail!("unsupported GGUF metadata type {other}"),
        }
    }
}

/// A tensor extent in the independent source, resolved through the
/// package-format catalog so aliases copy the target's bytes exactly once.
#[derive(Debug, Clone)]
struct SourceExtent {
    path: PathBuf,
    data_offset: u64,
    stored_length: u64,
}

/// Write one payload part holding exactly `tensor_names`, copying bytes from
/// the source shards. The output is a self-contained GGUF carrying the source
/// metadata KV (minus split bookkeeping) and one aligned payload extent per
/// tensor. Tensors are laid out in sorted-name order.
pub(crate) fn write_part(
    inventory: &SourceInventory,
    tensor_names: &[String],
    out: &Path,
) -> Result<()> {
    ensure!(
        !tensor_names.is_empty(),
        "a payload part must bind at least one tensor"
    );
    let alignment = inventory.shards[0].directory.alignment;
    ensure!(alignment > 0, "source alignment must be positive");
    let mut descriptors = BTreeMap::new();
    for shard in &inventory.shards {
        for tensor in &shard.tensors.entries {
            let gguf_tensor = shard
                .directory
                .tensors
                .iter()
                .find(|candidate| candidate.name == tensor.id)
                .with_context(|| {
                    format!(
                        "tensor {:?} missing from the shard descriptor table",
                        tensor.id
                    )
                })?;
            ensure!(
                descriptors
                    .insert(
                        tensor.id.clone(),
                        (gguf_tensor.clone(), shard.path.clone(), tensor.clone()),
                    )
                    .is_none(),
                "duplicate source tensor {:?} across shards",
                tensor.id
            );
        }
    }
    let mut selected: BTreeMap<String, SourceExtent> = BTreeMap::new();
    for name in tensor_names {
        let (_, path, tensor) = descriptors
            .get(name)
            .with_context(|| format!("part tensor {name:?} is absent from the source inventory"))?;
        let extent = owned_extent(&descriptors, &tensor.storage, path)?;
        selected.insert(name.clone(), extent);
    }

    if let Some(parent) = out.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create part parent directory {}", parent.display()))?;
    }
    let mut writer = File::create(out).with_context(|| format!("create part {}", out.display()))?;

    // Header: magic, version, tensor count, metadata count, KV pairs, then the
    // tensor descriptor table with freshly assigned aligned offsets. Metadata
    // KV entries are copied from the primary source shard as raw wire bytes
    // (minus split bookkeeping), preserving their exact GGUF types.
    let mut header: Vec<u8> = Vec::new();
    header.extend_from_slice(GGUF_MAGIC);
    extend_u32(&mut header, GGUF_VERSION);
    extend_u64(&mut header, selected.len() as u64);
    let metadata_bytes = source_metadata_wire_bytes(&inventory.shards[0])?;
    extend_u64(&mut header, metadata_bytes.entry_count);
    header.extend_from_slice(&metadata_bytes.entries);
    let mut offset: u64 = 0;
    let mut table: Vec<(&String, &GgufTensor, u64)> = Vec::new();
    for (name, extent) in &selected {
        let (descriptor, _, _) = &descriptors[name];
        table.push((name, descriptor, offset));
        offset = align_to(offset + extent.stored_length, alignment);
    }
    for (name, descriptor, tensor_offset) in &table {
        write_string(&mut header, name);
        extend_u32(&mut header, descriptor.dimensions.len() as u32);
        for dimension in &descriptor.dimensions {
            extend_u64(&mut header, *dimension);
        }
        extend_u32(&mut header, descriptor.ggml_type);
        extend_u64(&mut header, *tensor_offset);
    }
    let data_start = align_to(header.len() as u64, alignment);
    header.resize(data_start as usize, 0);
    writer
        .write_all(&header)
        .with_context(|| format!("write part header {}", out.display()))?;

    // Payload: stream each extent in table order; table offsets are already
    // aligned, so pad the cursor to each tensor's offset before copying and
    // consecutive writes land exactly where the table points.
    let mut cursor: u64 = 0;
    let zeros = vec![0_u8; COPY_BUFFER_BYTES];
    for (name, _, tensor_offset) in &table {
        while cursor < *tensor_offset {
            let gap = (*tensor_offset - cursor).min(zeros.len() as u64);
            writer.write_all(&zeros[..gap as usize])?;
            cursor += gap;
        }
        let extent = &selected[name.as_str()];
        copy_exact_extent(
            &extent.path,
            extent.data_offset,
            extent.stored_length,
            &mut writer,
        )
        .with_context(|| format!("copy payload of tensor {name:?} into part"))?;
        cursor += extent.stored_length;
    }
    writer
        .sync_all()
        .with_context(|| format!("sync part {}", out.display()))?;
    Ok(())
}

/// Resolve the byte extent that stores `storage`. Owned storage copies its own
/// extent; alias storage copies the target's extent (the alias shares bytes, it
/// does not own a second copy).
fn owned_extent(
    descriptors: &BTreeMap<String, (GgufTensor, PathBuf, Tensor)>,
    storage: &TensorStorage,
    path: &Path,
) -> Result<SourceExtent> {
    match storage {
        TensorStorage::Owned {
            data_offset,
            stored_length,
            ..
        } => Ok(SourceExtent {
            path: path.to_path_buf(),
            data_offset: *data_offset,
            stored_length: *stored_length,
        }),
        TensorStorage::Alias { target_tensor_id } => {
            let (_, target_path, tensor) = descriptors
                .get(target_tensor_id)
                .with_context(|| format!("alias target {target_tensor_id:?} is missing"))?;
            owned_extent(descriptors, &tensor.storage, target_path)
        }
    }
}

fn copy_exact_extent(source: &Path, offset: u64, length: u64, writer: &mut File) -> Result<()> {
    ensure!(length > 0, "refusing to copy a zero-length extent");
    let mut file =
        File::open(source).with_context(|| format!("open source {}", source.display()))?;
    file.seek(SeekFrom::Start(offset))
        .with_context(|| format!("seek source {}", source.display()))?;
    let mut buffer =
        vec![0_u8; COPY_BUFFER_BYTES.min(length.try_into().unwrap_or(COPY_BUFFER_BYTES))];
    let mut remaining = length;
    while remaining > 0 {
        let want = buffer
            .len()
            .min(usize::try_from(remaining).unwrap_or(buffer.len()));
        file.read_exact(&mut buffer[..want])
            .with_context(|| format!("read source {}", source.display()))?;
        writer.write_all(&buffer[..want])?;
        remaining -= want as u64;
    }
    Ok(())
}

fn extend_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn extend_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn write_string(bytes: &mut Vec<u8>, value: &str) {
    extend_u64(bytes, value.len() as u64);
    bytes.extend_from_slice(value.as_bytes());
}

fn align_to(value: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return value;
    }
    value.div_ceil(alignment) * alignment
}

#[cfg(test)]
mod tests;
