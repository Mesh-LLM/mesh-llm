//! Reads the claimed GGUF's header and individual weight rows, from a local
//! file or any ranged byte source (for example Hugging Face range requests).
//!
//! The node that issued a receipt names the claimed GGUF, so the file is
//! read as hostile input: the header strictly (`header`), and every read
//! within the [`limits`] of this client.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result, bail};

mod header;
pub mod limits;

use header::{FIRST_READ, SHARD_FIRST_READ, parse_header, tensor_bytes};
pub use header::{MetaArray, TokenList};
use limits::{FetchBudget, MAX_HEADER_BYTES, MAX_ROW_VALUES, MAX_SHARDS, refuse};

/// Random access to the GGUF bytes.
pub trait ByteSource: Send + Sync {
    fn read_at(&self, offset: u64, len: u64) -> Result<Vec<u8>>;
    fn size(&self) -> u64;
    /// Readies `ranges`, `(offset, len)` each, so later reads of them are
    /// local. A local file holds every byte already.
    fn prefetch(&self, _ranges: &[(u64, u64)]) -> Result<()> {
        Ok(())
    }
    /// Reads bytes needed once, which a sparse copy does not keep.
    fn read_once(&self, offset: u64, len: u64) -> Result<Vec<u8>> {
        self.read_at(offset, len)
    }
    /// Bytes of `ranges` that reading them would fetch from elsewhere. A
    /// local file holds every byte already.
    fn missing_bytes(&self, _ranges: &[(u64, u64)]) -> Result<u64> {
        Ok(0)
    }
}

pub struct LocalFile {
    file: Mutex<File>,
    size: u64,
}

impl LocalFile {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let size = file.metadata()?.len();
        Ok(Self {
            file: Mutex::new(file),
            size,
        })
    }
}

impl ByteSource for LocalFile {
    fn read_at(&self, offset: u64, len: u64) -> Result<Vec<u8>> {
        if offset.checked_add(len).is_none_or(|end| end > self.size) {
            bail!(
                "bytes {offset}+{len} are past the end of the file ({} bytes)",
                self.size
            );
        }
        let mut file = self
            .file
            .lock()
            .map_err(|_| anyhow::anyhow!("file lock poisoned"))?;
        file.seek(SeekFrom::Start(offset))?;
        let mut buf = vec![0u8; usize::try_from(len)?];
        file.read_exact(&mut buf)?;
        Ok(buf)
    }

    fn size(&self) -> u64 {
        self.size
    }
}

/// A source whose reads take what they fetch from a budget first: the
/// header's reads, which come before the file is a [`WeightFile`].
struct Charged<'a> {
    source: &'a dyn ByteSource,
    budget: &'a FetchBudget,
}

impl ByteSource for Charged<'_> {
    fn read_at(&self, offset: u64, len: u64) -> Result<Vec<u8>> {
        match self.source.missing_bytes(&[(offset, len)])? {
            0 => {}
            missing => self.budget.charge(missing)?,
        }
        self.source.read_at(offset, len)
    }

    fn size(&self) -> u64 {
        self.source.size()
    }
}

use crate::dequant::dequantize_row;
use crate::ggml_types::type_layout;
use crate::shards::Shards;

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    /// Absolute file offset of the tensor data.
    pub offset: u64,
}

impl TensorInfo {
    /// Values per row (the matmul's K).
    pub fn row_len(&self) -> u64 {
        self.dims.first().copied().unwrap_or(0)
    }

    /// Rows (the matmul's M).
    pub fn rows(&self) -> u64 {
        self.dims.get(1).copied().unwrap_or(1)
    }

    /// Bytes of one row, which is whole blocks of the tensor's type and
    /// never empty.
    pub fn row_bytes(&self) -> Result<u64> {
        let (block, bytes) = type_layout(self.ggml_type)
            .with_context(|| format!("{}: unknown ggml type {}", self.name, self.ggml_type))?;
        let len = self.row_len();
        if len == 0 || !len.is_multiple_of(block) {
            bail!(
                "{}: a row of {len} values is not whole blocks of {block}",
                self.name
            );
        }
        (len / block)
            .checked_mul(bytes)
            .with_context(|| format!("{}: the size of a row does not fit", self.name))
    }

    /// Rows over every dimension past the first.
    pub(crate) fn row_count(&self) -> Result<u64> {
        self.dims
            .iter()
            .skip(1)
            .try_fold(1u64, |rows, &dim| rows.checked_mul(dim))
            .with_context(|| format!("{}: the number of rows does not fit", self.name))
    }

    /// Bytes of the whole tensor. A tensor of a type the runtime does not
    /// load, or of a size that does not fit, is refused, never sized zero:
    /// a sparse copy would leave it a hole.
    pub fn total_bytes(&self) -> Result<u64> {
        tensor_bytes(&self.name, &self.dims, self.ggml_type)
    }

    pub fn layer(&self) -> Option<u32> {
        self.name
            .strip_prefix("blk.")?
            .split('.')
            .next()?
            .parse()
            .ok()
    }

    pub fn is_matrix(&self) -> bool {
        self.dims.len() >= 2 && self.dims[1] > 1 && self.name.ends_with(".weight")
    }
}

/// A scalar GGUF metadata value. Arrays are skipped, except the list of
/// suppressed tokens ([`WeightFile::suppressed_tokens`]).
#[derive(Clone, Debug, PartialEq)]
pub enum MetaValue {
    Int(u64),
    Float(f64),
    Str(String),
    Tokens(Vec<i32>),
}

/// The claimed model: header metadata and tensor directory.
pub struct WeightFile {
    source: Box<dyn ByteSource>,
    pub tensors: BTreeMap<String, TensorInfo>,
    pub metadata: BTreeMap<String, MetaValue>,
    /// The metadata arrays a check reads before the native loader sees the
    /// file: the vocabulary, per-layer counts, rope sections.
    pub arrays: BTreeMap<String, MetaArray>,
    pub header_bytes: u64,
    /// What the reads after the header may still fetch of a remote file.
    budget: FetchBudget,
}

impl WeightFile {
    /// Reads the header of the GGUF in `source`, a file held whole: it may
    /// fetch nothing ([`Self::open_with_budget`] is for a remote file).
    pub fn open(source: Box<dyn ByteSource>) -> Result<Self> {
        Self::open_with_budget(source, 0)
    }

    /// Reads the header of the GGUF in `source`, strictly: a file the
    /// runtime's loader would not take, or one past the
    /// [limits](limits), is refused with [`limits::CannotAudit`]. The reads
    /// may fetch `budget` bytes of a remote file, the header's included
    /// ([`limits::AUDIT_FETCH_BUDGET`] for an audit).
    pub fn open_with_budget(source: Box<dyn ByteSource>, budget: u64) -> Result<Self> {
        let budget = FetchBudget::new(budget);
        let charged = Charged {
            source: source.as_ref(),
            budget: &budget,
        };
        let header = parse_header(&charged, FIRST_READ, MAX_HEADER_BYTES)?;
        Ok(Self {
            source,
            tensors: header.tensors,
            metadata: header.metadata,
            arrays: header.arrays,
            header_bytes: header.data_start,
            budget,
        })
    }

    /// Opens a split GGUF held whole from its shards, in order
    /// ([`Self::open_shards_with_budget`]).
    pub fn open_shards(shards: Vec<Box<dyn ByteSource>>) -> Result<Self> {
        Self::open_shards_with_budget(shards, 0)
    }

    /// Opens a split GGUF from its shards, in order. The first shard's
    /// metadata describes the model; each tensor sits at its offset in the
    /// shards end to end ([`Shards`]), so ranges name bytes of the whole.
    /// The reads may fetch `budget` bytes, as [`Self::open_with_budget`].
    pub fn open_shards_with_budget(shards: Vec<Box<dyn ByteSource>>, budget: u64) -> Result<Self> {
        if shards.is_empty() || shards.len() > MAX_SHARDS {
            return refuse(format!(
                "it is split into {} shards, not 1 to {MAX_SHARDS}",
                shards.len()
            ));
        }
        let source = Shards::new(shards)?;
        let budget = FetchBudget::new(budget);
        let count = source.parts().count() as u64;
        let mut tensors = BTreeMap::new();
        let mut metadata = BTreeMap::new();
        let mut arrays = BTreeMap::new();
        let mut header_bytes = 0;
        // the headers of all the shards share the limit of one
        let mut headers_left = MAX_HEADER_BYTES;
        for (index, (base, shard)) in source.parts().enumerate() {
            let first_read = if index == 0 {
                FIRST_READ
            } else {
                SHARD_FIRST_READ
            };
            let charged = Charged {
                source: shard,
                budget: &budget,
            };
            let header = parse_header(&charged, first_read, headers_left)
                .with_context(|| format!("read shard {}", index + 1))?;
            headers_left = headers_left.saturating_sub(header.header_bytes);
            check_shard(&header.metadata, index as u64, count)?;
            if index == 0 {
                metadata = header.metadata;
                arrays = header.arrays;
                header_bytes = header.data_start;
            }
            for (name, mut info) in header.tensors {
                // a shard's tensors end inside it, so this fits
                info.offset = info
                    .offset
                    .checked_add(base)
                    .with_context(|| format!("tensor {name} is past any offset"))?;
                if tensors.insert(name.clone(), info).is_some() {
                    return refuse(format!("tensor {name} is in two of its shards"));
                }
            }
            if tensors.len() as u64 > limits::MAX_TENSORS {
                return refuse(format!(
                    "its shards list more than {} tensors",
                    limits::MAX_TENSORS
                ));
            }
        }
        if let Some(MetaValue::Int(expected)) = metadata.get("split.tensors.count")
            && *expected != tensors.len() as u64
        {
            return refuse(format!(
                "its shards hold {} tensors, not the {expected} the first says",
                tensors.len()
            ));
        }
        Ok(Self {
            source: Box::new(source),
            tensors,
            metadata,
            arrays,
            header_bytes,
            budget,
        })
    }

    /// Lets the reads from now on fetch `bytes` of a remote file, then
    /// refuse with [`limits::CannotAudit`]. A local file is never fetched,
    /// and bytes a sparse copy holds cost nothing. Without this the budget
    /// is the one the file opened with.
    pub fn set_fetch_budget(&self, bytes: u64) {
        self.budget.set(bytes);
    }

    /// Starts the fetch budget over, for the next audit against a file
    /// that stays open.
    pub fn reset_fetch_budget(&self) {
        self.budget.reset();
    }

    /// Bytes fetched of a remote file since the budget was set or started
    /// over, by the reads that count against it.
    pub fn fetched_bytes(&self) -> u64 {
        self.budget.spent()
    }

    /// Bytes of `ranges`, `(offset, len)` each, that reading them would
    /// fetch of a remote file.
    pub fn missing_bytes(&self, ranges: &[(u64, u64)]) -> Result<u64> {
        self.source.missing_bytes(ranges)
    }

    /// Takes what `ranges` would fetch from the budget.
    fn charge(&self, ranges: &[(u64, u64)]) -> Result<()> {
        match self.source.missing_bytes(ranges)? {
            0 => Ok(()),
            missing => self.budget.charge(missing),
        }
    }

    /// Reads bytes of the file within the budget; `once` for bytes a sparse
    /// copy need not keep.
    fn read(&self, offset: u64, len: u64, once: bool) -> Result<Vec<u8>> {
        if !once {
            self.charge(&[(offset, len)])?;
            return self.source.read_at(offset, len);
        }
        // a read that is not kept fetches all of it unless all is held
        if self.source.missing_bytes(&[(offset, len)])? > 0 {
            self.budget.charge(len)?;
        }
        self.source.read_once(offset, len)
    }

    /// The tensor `name` for a read that decodes its rows, the bytes of a
    /// row, and its values.
    fn decoded_rows(&self, name: &str) -> Result<(&TensorInfo, u64, usize)> {
        let t = self.tensor(name)?;
        if t.row_len() > MAX_ROW_VALUES {
            return refuse(format!(
                "{name} has rows of {} values, more than {MAX_ROW_VALUES}",
                t.row_len()
            ));
        }
        Ok((t, t.row_bytes()?, usize::try_from(t.row_len())?))
    }

    pub(crate) fn metadata_str(&self, key: &str) -> Result<&str> {
        match self.metadata.get(key) {
            Some(MetaValue::Str(value)) => Ok(value),
            _ => bail!("claimed GGUF has no string {key}"),
        }
    }

    /// The tokens the model names that its sampler never picks: the
    /// runtime takes them out of every turn's candidates.
    pub fn suppressed_tokens(&self) -> &[i32] {
        match self.metadata.get(header::SUPPRESSED_TOKENS_KEY) {
            Some(MetaValue::Tokens(tokens)) => tokens,
            _ => &[],
        }
    }

    pub fn metadata_f64(&self, key: &str) -> Result<f64> {
        match self.metadata.get(key) {
            Some(MetaValue::Float(value)) => Ok(*value),
            _ => bail!("claimed GGUF has no float {key}"),
        }
    }

    /// All values of a one-dimensional tensor, such as a norm weight, in f64.
    pub fn vector(&self, name: &str) -> Result<Vec<f64>> {
        let t = self.tensor(name)?;
        if t.dims.len() != 1 {
            bail!("{name} is not a vector");
        }
        self.row(name, 0)
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors
            .get(name)
            .with_context(|| format!("claimed GGUF has no tensor {name}"))
    }

    /// Readies `ranges` of the file, `(offset, len)` each, at once.
    pub fn prefetch(&self, ranges: &[(u64, u64)]) -> Result<()> {
        self.charge(ranges)?;
        self.source.prefetch(ranges)
    }

    /// The bytes of a tensor, `(offset, len)`.
    pub(crate) fn tensor_range(&self, name: &str) -> Result<(u64, u64)> {
        let t = self.tensor(name)?;
        Ok((t.offset, t.total_bytes()?))
    }

    /// The bytes of row `row` of a weight, `(offset, len)`.
    pub fn row_range(&self, name: &str, row: u64) -> Result<(u64, u64)> {
        let t = self.tensor(name)?;
        if row >= t.row_count()? {
            bail!("{name} has no row {row}");
        }
        let row_bytes = t.row_bytes()?;
        let offset = row
            .checked_mul(row_bytes)
            .and_then(|at| t.offset.checked_add(at))
            .with_context(|| format!("{name} row {row} is past any offset"))?;
        Ok((offset, row_bytes))
    }

    /// Dequantized row `row` of a 2D weight, in f64.
    pub fn row(&self, name: &str, row: u64) -> Result<Vec<f64>> {
        let (t, _, n) = self.decoded_rows(name)?;
        let (offset, len) = self.row_range(name, row)?;
        let raw = self.read(offset, len, false)?;
        dequantize_row(t.ggml_type, &raw, n).with_context(|| format!("dequantize {name} row {row}"))
    }

    /// Row `row` of a Q4_K or Q5_K weight as its sub-blocks of 32 values.
    pub(crate) fn min_blocks(&self, name: &str, row: u64) -> Result<Vec<crate::dequant::MinBlock>> {
        let (t, _, _) = self.decoded_rows(name)?;
        let (offset, len) = self.row_range(name, row)?;
        let raw = self.read(offset, len, false)?;
        crate::dequant::min_blocks(t.ggml_type, &raw)
            .with_context(|| format!("read {name} row {row}"))
    }

    /// Calls `visit` with each of `rows` of a 2D weight, dequantized, in
    /// order. Reads many rows at a time.
    pub(crate) fn each_row(
        &self,
        name: &str,
        rows: std::ops::Range<u64>,
        visit: &mut dyn FnMut(u64, &[f64]),
    ) -> Result<()> {
        self.visit_rows(name, rows, visit, false)
    }

    /// [`Self::each_row`] for rows read once: a sparse copy does not keep
    /// them.
    pub(crate) fn stream_rows(
        &self,
        name: &str,
        rows: std::ops::Range<u64>,
        visit: &mut dyn FnMut(u64, &[f64]),
    ) -> Result<()> {
        self.visit_rows(name, rows, visit, true)
    }

    fn visit_rows(
        &self,
        name: &str,
        rows: std::ops::Range<u64>,
        visit: &mut dyn FnMut(u64, &[f64]),
        once: bool,
    ) -> Result<()> {
        /// Bytes read at once.
        const CHUNK_BYTES: u64 = 4 << 20;
        let (t, row_bytes, n) = self.decoded_rows(name)?;
        if rows.end > t.rows() {
            bail!("{name} has {} rows, not {}", t.rows(), rows.end);
        }
        // a row is never empty, so the chunks below are not
        let row_size = usize::try_from(row_bytes)?;
        let per_chunk = (CHUNK_BYTES / row_bytes).max(1);
        let mut first = rows.start;
        while first < rows.end {
            let count = per_chunk.min(rows.end - first);
            let (offset, _) = self.row_range(name, first)?;
            let len = count
                .checked_mul(row_bytes)
                .with_context(|| format!("{name}: {count} rows do not fit a read"))?;
            let raw = self.read(offset, len, once)?;
            for (i, row) in raw.chunks_exact(row_size).enumerate() {
                let index = first + i as u64;
                let values = dequantize_row(t.ggml_type, row, n)
                    .with_context(|| format!("dequantize {name} row {index}"))?;
                visit(index, &values);
            }
            first += count;
        }
        Ok(())
    }

    /// Weight bytes per layer: an audit opens a layer with this probability.
    pub fn layer_bytes(&self) -> BTreeMap<u32, u64> {
        let mut out = BTreeMap::new();
        for t in self.tensors.values().filter(|t| t.is_matrix()) {
            if let Some(layer) = t.layer() {
                // a tensor of a file that opened has a size
                let bytes = t.total_bytes().unwrap_or(0);
                let total: &mut u64 = out.entry(layer).or_insert(0);
                *total = total.saturating_add(bytes);
            }
        }
        out
    }
}

/// Checks that the shard at `index` of `count` says it is.
fn check_shard(metadata: &BTreeMap<String, MetaValue>, index: u64, count: u64) -> Result<()> {
    let int = |key: &str| match metadata.get(key) {
        Some(MetaValue::Int(value)) => Ok(*value),
        _ => refuse(format!("its shard {} has no {key}", index + 1)),
    };
    if int("split.no")? != index || int("split.count")? != count {
        return refuse(format!(
            "its shard {} of {count} is not the shard its name says",
            index + 1
        ));
    }
    Ok(())
}

/// Small GGUF files in memory, for tests.
#[cfg(test)]
pub(crate) mod test_gguf;

#[cfg(test)]
mod tests;
