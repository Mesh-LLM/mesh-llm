//! The header of a claimed GGUF, read strictly.
//!
//! The node names the model, so every byte of the header may be chosen to
//! break the reader. Each length, count and offset is checked against the
//! [limits](super::limits) and the size of the file before it is used, with
//! arithmetic that cannot wrap. The header must also be one the runtime's
//! own loader takes: little-endian, version 2 or 3, names in UTF-8, each
//! name once, tensors of known types laid end to end at the alignment and
//! inside the file. A header that fails any of this is refused with
//! [`CannotAudit`](super::limits::CannotAudit) before the native loader sees
//! the file.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Result, bail};

use super::limits::{
    MAX_ARRAY_DEPTH, MAX_DIMS, MAX_HEADER_BYTES, MAX_METADATA_ENTRIES, MAX_NAME_BYTES,
    MAX_STRING_BYTES, MAX_TENSORS, refuse,
};
use super::{ByteSource, MetaValue, TensorInfo};
use crate::ggml_types::type_layout;

/// Bytes the first read of a header asks for: one request that holds the
/// header of most models, tokenizer and all.
pub(super) const FIRST_READ: u64 = 8 << 20;
/// The first read of a later shard, whose header has no tokenizer.
pub(super) const SHARD_FIRST_READ: u64 = 1 << 20;

const GGUF_UINT32: u32 = 4;
const GGUF_FLOAT32: u32 = 6;
const GGUF_BOOL: u32 = 7;
const GGUF_STRING: u32 = 8;
const GGUF_ARRAY: u32 = 9;
const GGUF_FLOAT64: u32 = 12;

const ALIGNMENT_KEY: &str = "general.alignment";
/// The tokens the runtime's sampler never picks, whatever a request asks.
pub(crate) const SUPPRESSED_TOKENS_KEY: &str = "tokenizer.ggml.suppress_tokens";
const DEFAULT_ALIGNMENT: u64 = 32;
/// ggml counts elements in an `int64_t`.
const MAX_ELEMENTS: u64 = i64::MAX as u64;

pub(super) struct Header {
    pub(super) tensors: BTreeMap<String, TensorInfo>,
    pub(super) metadata: BTreeMap<String, MetaValue>,
    /// Where the tensor data starts.
    pub(super) data_start: u64,
    /// Bytes of the header that were read.
    pub(super) header_bytes: u64,
}

/// Reads the header of the GGUF in `source`, asking for `first_read` bytes
/// first, and refuses a header of more than `max_bytes`: all of
/// [`MAX_HEADER_BYTES`], or what the shards before this one left of it.
pub(super) fn parse_header(
    source: &dyn ByteSource,
    first_read: u64,
    max_bytes: u64,
) -> Result<Header> {
    let mut c = Cursor {
        source,
        size: source.size(),
        buf: Vec::new(),
        pos: 0,
        first_read: first_read.max(1),
        max_bytes: max_bytes.min(MAX_HEADER_BYTES),
    };
    if c.bytes(4)? != b"GGUF" {
        return refuse("it is not a GGUF file");
    }
    check_version(c.u32()?)?;
    let n_tensors = c.u64()?;
    if n_tensors > MAX_TENSORS {
        return refuse(format!(
            "its header lists {n_tensors} tensors, more than {MAX_TENSORS}"
        ));
    }
    let n_kv = c.u64()?;
    if n_kv > MAX_METADATA_ENTRIES {
        return refuse(format!(
            "its header holds {n_kv} metadata entries, more than {MAX_METADATA_ENTRIES}"
        ));
    }
    let (metadata, alignment) = read_metadata(&mut c, n_kv)?;
    let listed = (0..n_tensors)
        .map(|_| read_tensor(&mut c))
        .collect::<Result<Vec<_>>>()?;
    let (tensors, data_start) = place_tensors(listed, c.pos as u64, alignment, c.size)?;
    Ok(Header {
        tensors,
        metadata,
        data_start,
        header_bytes: c.pos as u64,
    })
}

/// The runtime loads little-endian GGUF of versions 2 and 3.
fn check_version(version: u32) -> Result<()> {
    match version {
        2 | 3 => Ok(()),
        other if matches!(other.swap_bytes(), 1..=3) => {
            refuse("it is a big-endian GGUF, which this client does not read")
        }
        other => refuse(format!(
            "it is GGUF version {other}, which this client does not read"
        )),
    }
}

/// The bytes of a tensor of `dims` and `ggml_type`, as ggml sizes it.
pub(super) fn tensor_bytes(name: &str, dims: &[u64], ggml_type: u32) -> Result<u64> {
    let Some((block, block_bytes)) = type_layout(ggml_type) else {
        return refuse(format!(
            "tensor {name} has ggml type {ggml_type}, which this runtime does not load"
        ));
    };
    if dims.is_empty() || dims.len() > MAX_DIMS {
        return refuse(format!(
            "tensor {name} has {} dimensions, not 1 to {MAX_DIMS}",
            dims.len()
        ));
    }
    let mut elements = 1u64;
    for &dim in dims {
        let product = elements
            .checked_mul(dim)
            .filter(|&product| product > 0 && product <= MAX_ELEMENTS);
        let Some(product) = product else {
            return refuse(format!(
                "tensor {name} has dimensions {dims:?}: one is zero, or their product does not fit"
            ));
        };
        elements = product;
    }
    if !dims[0].is_multiple_of(block) {
        return refuse(format!(
            "tensor {name} has rows of {} values, not whole blocks of {block}",
            dims[0]
        ));
    }
    match (elements / block).checked_mul(block_bytes) {
        Some(bytes) => Ok(bytes),
        None => refuse(format!("tensor {name} has a size that does not fit")),
    }
}

/// A tensor as the header lists it, before the data start is known.
struct Listed {
    name: String,
    dims: Vec<u64>,
    ggml_type: u32,
    /// Its offset in the tensor data.
    relative: u64,
    bytes: u64,
}

fn read_metadata(c: &mut Cursor<'_>, count: u64) -> Result<(BTreeMap<String, MetaValue>, u64)> {
    let mut metadata = BTreeMap::new();
    // arrays are not kept, so the keys seen are more than the metadata's
    let mut seen = BTreeSet::new();
    let mut alignment = DEFAULT_ALIGNMENT;
    for _ in 0..count {
        let key = c.name("a metadata key")?;
        let ty = c.u32()?;
        let value = if ty == GGUF_ARRAY && key == SUPPRESSED_TOKENS_KEY {
            // the one array a check asks about: kept as its length
            Some(MetaValue::Int(c.skip_array(1)?))
        } else {
            c.value(ty)?
        };
        if key == ALIGNMENT_KEY {
            alignment = alignment_of(ty, value.as_ref())?;
        }
        if !seen.insert(key.clone()) {
            return refuse(format!("its header has the metadata key {key} twice"));
        }
        if let Some(value) = value {
            metadata.insert(key, value);
        }
    }
    Ok((metadata, alignment))
}

/// The alignment `general.alignment` gives: a `u32` power of two, as the
/// runtime's loader requires.
fn alignment_of(ty: u32, value: Option<&MetaValue>) -> Result<u64> {
    match value {
        Some(MetaValue::Int(alignment)) if ty == GGUF_UINT32 && alignment.is_power_of_two() => {
            Ok(*alignment)
        }
        _ => refuse(format!("its {ALIGNMENT_KEY} is not a u32 power of two")),
    }
}

fn read_tensor(c: &mut Cursor<'_>) -> Result<Listed> {
    let name = c.name("a tensor name")?;
    let rank = c.u32()?;
    if rank == 0 || rank as usize > MAX_DIMS {
        return refuse(format!(
            "tensor {name} has {rank} dimensions, not 1 to {MAX_DIMS}"
        ));
    }
    let dims = (0..rank).map(|_| c.u64()).collect::<Result<Vec<_>>>()?;
    let ggml_type = c.u32()?;
    let relative = c.u64()?;
    let bytes = tensor_bytes(&name, &dims, ggml_type)?;
    Ok(Listed {
        name,
        dims,
        ggml_type,
        relative,
        bytes,
    })
}

/// Gives each tensor its offset in the file. The tensors must lie end to
/// end at the alignment, in the order the header lists them, and inside the
/// file: so the runtime's loader requires, and so no two overlap.
fn place_tensors(
    listed: Vec<Listed>,
    header_end: u64,
    alignment: u64,
    size: u64,
) -> Result<(BTreeMap<String, TensorInfo>, u64)> {
    let Some(data_start) = header_end.checked_next_multiple_of(alignment) else {
        return refuse("its tensor data starts past any offset");
    };
    let mut tensors = BTreeMap::new();
    let mut expected = 0u64;
    for tensor in listed {
        let name = tensor.name;
        if tensor.relative != expected {
            return refuse(format!(
                "tensor {name} is at offset {} of the tensor data, not at {expected} where the tensor before it ends",
                tensor.relative
            ));
        }
        let offset = data_start.checked_add(tensor.relative);
        let end = offset
            .and_then(|offset| offset.checked_add(tensor.bytes))
            .filter(|&end| end <= size);
        let (Some(offset), Some(_)) = (offset, end) else {
            return refuse(format!(
                "tensor {name} runs past the end of the file ({size} bytes)"
            ));
        };
        let next = tensor
            .bytes
            .checked_next_multiple_of(alignment)
            .and_then(|padded| expected.checked_add(padded));
        let Some(next) = next else {
            return refuse(format!("tensor {name} ends past any offset"));
        };
        expected = next;
        let info = TensorInfo {
            name: name.clone(),
            dims: tensor.dims,
            ggml_type: tensor.ggml_type,
            offset,
        };
        if tensors.insert(name.clone(), info).is_some() {
            return refuse(format!("its header lists tensor {name} twice"));
        }
    }
    Ok((tensors, data_start))
}

/// Bytes of a metadata value of scalar type `ty`.
fn scalar_bytes(ty: u32) -> Option<usize> {
    match ty {
        0 | 1 | GGUF_BOOL => Some(1),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    }
}

struct Cursor<'a> {
    source: &'a dyn ByteSource,
    size: u64,
    /// The file from its start, as far as it was read.
    buf: Vec<u8>,
    pos: usize,
    first_read: u64,
    /// The most bytes the header may take.
    max_bytes: u64,
}

impl Cursor<'_> {
    /// The next `n` bytes of the header.
    fn bytes(&mut self, n: u64) -> Result<&[u8]> {
        let end = (self.pos as u64)
            .checked_add(n)
            .filter(|&end| end <= self.max_bytes);
        let Some(end) = end else {
            return refuse(format!(
                "its GGUF header, with those of the shards before it, is larger than {MAX_HEADER_BYTES} bytes"
            ));
        };
        if end > self.size {
            return refuse("its GGUF header runs past the end of the file");
        }
        self.fill(end)?;
        let end = usize::try_from(end)?;
        let bytes = &self.buf[self.pos..end];
        self.pos = end;
        Ok(bytes)
    }

    /// Reads the file up to `end` at least: twice what is held each time,
    /// so a header takes few requests of a remote file.
    fn fill(&mut self, end: u64) -> Result<()> {
        let ceiling = self.size.min(self.max_bytes);
        while (self.buf.len() as u64) < end {
            let held = self.buf.len() as u64;
            let len = held
                .max(self.first_read)
                .max(end - held)
                .min(ceiling - held);
            let chunk = self.source.read_at(held, len)?;
            if chunk.len() as u64 != len {
                bail!(
                    "read {} bytes of the GGUF header at {held}, not {len}",
                    chunk.len()
                );
            }
            self.buf.extend_from_slice(&chunk);
        }
        Ok(())
    }

    fn skip(&mut self, n: u64) -> Result<()> {
        self.bytes(n).map(|_| ())
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into()?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes(8)?.try_into()?))
    }

    /// A string of at most `max` bytes, which must be UTF-8: a lossy
    /// reading would name something the runtime's loader does not.
    fn text(&mut self, max: u64, what: &str) -> Result<String> {
        let n = self.u64()?;
        if n > max {
            return refuse(format!(
                "{what} of {n} bytes is longer than {max} in its header"
            ));
        }
        match std::str::from_utf8(self.bytes(n)?) {
            Ok(text) => Ok(text.to_owned()),
            Err(_) => refuse(format!("{what} in its header is not UTF-8")),
        }
    }

    /// A metadata key or tensor name.
    fn name(&mut self, what: &str) -> Result<String> {
        let name = self.text(MAX_NAME_BYTES, what)?;
        if name.is_empty() || name.chars().any(char::is_control) {
            return refuse(format!(
                "{what} in its header is empty or holds control characters: {name:?}"
            ));
        }
        Ok(name)
    }

    /// Reads a metadata value, returning it when it is a number or a
    /// string. Booleans and arrays are skipped.
    fn value(&mut self, ty: u32) -> Result<Option<MetaValue>> {
        if let Some(n) = scalar_bytes(ty) {
            let mut le = [0u8; 8];
            le[..n].copy_from_slice(self.bytes(n as u64)?);
            return Ok(match ty {
                GGUF_FLOAT32 => Some(MetaValue::Float(f64::from(f32::from_le_bytes(
                    le[..4].try_into()?,
                )))),
                GGUF_FLOAT64 => Some(MetaValue::Float(f64::from_le_bytes(le))),
                GGUF_BOOL => None,
                _ => Some(MetaValue::Int(u64::from_le_bytes(le))),
            });
        }
        match ty {
            GGUF_STRING => Ok(Some(MetaValue::Str(
                self.text(MAX_STRING_BYTES, "a metadata string")?,
            ))),
            GGUF_ARRAY => self.skip_array(1).map(|_| None),
            other => refuse(format!("its header has the unknown metadata type {other}")),
        }
    }

    /// Skips an array at nesting `depth`, 1 for an array that is a value.
    /// The recursion ends at [`MAX_ARRAY_DEPTH`], whatever the file says.
    /// Returns the number of its elements.
    fn skip_array(&mut self, depth: u32) -> Result<u64> {
        if depth > MAX_ARRAY_DEPTH {
            return refuse(format!(
                "its metadata arrays nest deeper than {MAX_ARRAY_DEPTH}"
            ));
        }
        let inner = self.u32()?;
        let count = self.u64()?;
        // the fewest bytes an element takes: a string its length, an array
        // its type and count
        let least = match inner {
            GGUF_STRING => 8,
            GGUF_ARRAY => 12,
            other => match scalar_bytes(other) {
                Some(n) => n as u64,
                None => {
                    return refuse(format!(
                        "its header has an array of the unknown metadata type {other}"
                    ));
                }
            },
        };
        let total = count
            .checked_mul(least)
            .filter(|&total| total <= MAX_HEADER_BYTES);
        let Some(total) = total else {
            return refuse(format!(
                "a metadata array of {count} elements does not fit a header of {MAX_HEADER_BYTES} bytes"
            ));
        };
        match inner {
            GGUF_STRING => {
                for _ in 0..count {
                    let n = self.u64()?;
                    if n > MAX_STRING_BYTES {
                        return refuse(format!(
                            "a metadata string of {n} bytes is longer than {MAX_STRING_BYTES} in its header"
                        ));
                    }
                    self.skip(n)?;
                }
            }
            GGUF_ARRAY => {
                for _ in 0..count {
                    self.skip_array(depth + 1)?;
                }
            }
            _ => self.skip(total)?,
        }
        Ok(count)
    }
}

#[cfg(test)]
mod tests;
