//! Small GGUF files in memory, for tests.

use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, bail};

use super::{ByteSource, MetaValue, WeightFile};
use crate::ggml_types::{GGML_F32, type_layout};

pub(crate) struct Memory(pub(crate) Vec<u8>);

impl ByteSource for Memory {
    fn read_at(&self, offset: u64, len: u64) -> Result<Vec<u8>> {
        let start = usize::try_from(offset)?;
        let end = start
            .checked_add(usize::try_from(len)?)
            .context("read past the end")?;
        self.0
            .get(start..end)
            .map(<[u8]>::to_vec)
            .context("read past the end")
    }

    fn size(&self) -> u64 {
        self.0.len() as u64
    }
}

/// A remote file of `size` bytes as a reader of its sparse copy sees it:
/// the `held` bytes at its start are here, and every byte past them is
/// fetched, and reads as zero.
pub(crate) struct Remote {
    pub(crate) held: Vec<u8>,
    pub(crate) size: u64,
    /// Bytes read past the held ones.
    pub(crate) fetched: AtomicU64,
}

impl Remote {
    pub(crate) fn new(held: Vec<u8>, size: u64) -> Self {
        Self {
            held,
            size,
            fetched: AtomicU64::new(0),
        }
    }

    /// The bytes of `[offset, offset + len)` past the held ones.
    fn past_held(&self, offset: u64, len: u64) -> Result<u64> {
        let end = offset.checked_add(len).filter(|&end| end <= self.size);
        let end = end.context("read past the end")?;
        Ok(end.saturating_sub(offset.max(self.held.len() as u64)))
    }
}

impl ByteSource for Remote {
    fn read_at(&self, offset: u64, len: u64) -> Result<Vec<u8>> {
        let fetched = self.past_held(offset, len)?;
        if len > 64 << 20 {
            bail!("a test reads at most 64 MiB at once, not {len} bytes");
        }
        self.fetched.fetch_add(fetched, Ordering::SeqCst);
        let mut out = vec![0u8; len as usize];
        let start = offset as usize;
        if let Some(held) = self.held.get(start..) {
            let n = held.len().min(out.len());
            out[..n].copy_from_slice(&held[..n]);
        }
        Ok(out)
    }

    fn size(&self) -> u64 {
        self.size
    }

    fn missing_bytes(&self, ranges: &[(u64, u64)]) -> Result<u64> {
        ranges.iter().try_fold(0u64, |sum, &(offset, len)| {
            Ok(sum + self.past_held(offset, len)?)
        })
    }
}

pub(crate) fn string(out: &mut Vec<u8>, s: &str) {
    out.extend((s.len() as u64).to_le_bytes());
    out.extend(s.as_bytes());
}

/// A GGUF with `metadata` and f32 `tensors` (name, dims, values).
pub(crate) fn open(
    metadata: &[(&str, MetaValue)],
    tensors: &[(&str, Vec<u64>, Vec<f32>)],
) -> WeightFile {
    WeightFile::open(Box::new(Memory(bytes(metadata, tensors)))).expect("a test GGUF")
}

/// The header of a GGUF with `metadata` and `tensors` (name, dims, ggml
/// type) laid end to end, up to where the tensor data starts, and the bytes
/// of that data.
pub(crate) fn header(
    metadata: &[(&str, MetaValue)],
    tensors: &[(&str, Vec<u64>, u32)],
) -> (Vec<u8>, u64) {
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend((tensors.len() as u64).to_le_bytes());
    out.extend((metadata.len() as u64).to_le_bytes());
    for (key, value) in metadata {
        string(&mut out, key);
        match value {
            MetaValue::Int(v) => {
                out.extend(10u32.to_le_bytes());
                out.extend(v.to_le_bytes());
            }
            MetaValue::Float(v) => {
                out.extend(12u32.to_le_bytes());
                out.extend(v.to_le_bytes());
            }
            MetaValue::Str(v) => {
                out.extend(8u32.to_le_bytes());
                string(&mut out, v);
            }
            MetaValue::Tokens(tokens) => {
                out.extend(9u32.to_le_bytes());
                out.extend(5u32.to_le_bytes());
                out.extend((tokens.len() as u64).to_le_bytes());
                for token in tokens {
                    out.extend(token.to_le_bytes());
                }
            }
        }
    }
    let mut offset = 0u64;
    for (name, dims, ggml_type) in tensors {
        string(&mut out, name);
        out.extend((dims.len() as u32).to_le_bytes());
        for d in dims {
            out.extend(d.to_le_bytes());
        }
        out.extend(ggml_type.to_le_bytes());
        out.extend(offset.to_le_bytes());
        let (block, bytes) = type_layout(*ggml_type).expect("a ggml type with a layout");
        let size = dims.iter().product::<u64>() / block * bytes;
        offset += size.div_ceil(32) * 32;
    }
    out.resize(out.len().div_ceil(32) * 32, 0);
    (out, offset)
}

/// The bytes of a GGUF with `metadata` and f32 `tensors`.
pub(crate) fn bytes(
    metadata: &[(&str, MetaValue)],
    tensors: &[(&str, Vec<u64>, Vec<f32>)],
) -> Vec<u8> {
    let listed: Vec<_> = tensors
        .iter()
        .map(|(name, dims, values)| {
            assert_eq!(
                dims.iter().product::<u64>(),
                values.len() as u64,
                "{name} has a value per element"
            );
            (*name, dims.clone(), GGML_F32)
        })
        .collect();
    let (mut out, _) = header(metadata, &listed);
    for (_, _, values) in tensors {
        for v in values {
            out.extend(v.to_le_bytes());
        }
        out.resize(out.len().div_ceil(32) * 32, 0);
    }
    out
}
