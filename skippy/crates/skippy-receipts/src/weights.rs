//! Reads the claimed GGUF's header and individual weight rows, from a local
//! file or any ranged byte source (for example Hugging Face range requests).

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result, bail};

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

use crate::dequant::{
    GGML_BF16, GGML_F16, GGML_F32, GGML_IQ4_NL, GGML_IQ4_XS, GGML_Q2_K, GGML_Q3_K, GGML_Q4_0,
    GGML_Q4_1, GGML_Q4_K, GGML_Q5_0, GGML_Q5_1, GGML_Q5_K, GGML_Q6_K, GGML_Q8_0, dequantize_row,
};
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

    pub fn row_bytes(&self) -> Result<u64> {
        let (block, bytes) = type_layout(self.ggml_type)
            .with_context(|| format!("{}: unknown ggml type {}", self.name, self.ggml_type))?;
        Ok(self.row_len() / block * bytes)
    }

    pub fn total_bytes(&self) -> u64 {
        let rows: u64 = self.dims.iter().skip(1).product();
        self.row_bytes().unwrap_or(0) * rows
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

/// (values per block, bytes per block) for the ggml types in GGUF files.
pub fn type_layout(t: u32) -> Option<(u64, u64)> {
    Some(match t {
        GGML_F32 => (1, 4),
        GGML_F16 => (1, 2),
        GGML_Q4_0 => (32, 18),
        GGML_Q4_1 => (32, 20),
        GGML_Q5_0 => (32, 22),
        GGML_Q5_1 => (32, 24),
        GGML_Q8_0 => (32, 34),
        GGML_Q2_K => (256, 84),
        GGML_Q3_K => (256, 110),
        GGML_Q4_K => (256, 144),
        GGML_Q5_K => (256, 176),
        GGML_Q6_K => (256, 210),
        GGML_IQ4_NL => (32, 18),
        GGML_IQ4_XS => (256, 136),
        GGML_BF16 => (1, 2),
        _ => return None,
    })
}

pub fn type_name(t: u32) -> String {
    match t {
        GGML_F32 => "f32".into(),
        GGML_F16 => "f16".into(),
        GGML_Q4_0 => "q4_0".into(),
        GGML_Q4_1 => "q4_1".into(),
        GGML_Q5_0 => "q5_0".into(),
        GGML_Q5_1 => "q5_1".into(),
        GGML_Q8_0 => "q8_0".into(),
        GGML_Q2_K => "q2_K".into(),
        GGML_Q3_K => "q3_K".into(),
        GGML_Q4_K => "q4_K".into(),
        GGML_Q5_K => "q5_K".into(),
        GGML_Q6_K => "q6_K".into(),
        GGML_IQ4_NL => "iq4_nl".into(),
        GGML_IQ4_XS => "iq4_xs".into(),
        GGML_BF16 => "bf16".into(),
        _ => format!("type{t}"),
    }
}

/// A scalar GGUF metadata value. Arrays are skipped.
#[derive(Clone, Debug, PartialEq)]
pub enum MetaValue {
    Int(u64),
    Float(f64),
    Str(String),
}

/// The claimed model: header metadata and tensor directory.
pub struct WeightFile {
    source: Box<dyn ByteSource>,
    pub tensors: BTreeMap<String, TensorInfo>,
    pub metadata: BTreeMap<String, MetaValue>,
    pub header_bytes: u64,
}

impl WeightFile {
    pub fn open(source: Box<dyn ByteSource>) -> Result<Self> {
        let header = parse_header(source.as_ref())?;
        Ok(Self {
            source,
            tensors: header.tensors,
            metadata: header.metadata,
            header_bytes: header.data_start,
        })
    }

    /// Opens a split GGUF from its shards, in order. The first shard's
    /// metadata describes the model; each tensor sits at its offset in the
    /// shards end to end ([`Shards`]), so ranges name bytes of the whole.
    pub fn open_shards(shards: Vec<Box<dyn ByteSource>>) -> Result<Self> {
        let source = Shards::new(shards);
        let count = source.parts().count() as u64;
        let mut tensors = BTreeMap::new();
        let mut metadata = BTreeMap::new();
        let mut header_bytes = 0;
        for (index, (base, shard)) in source.parts().enumerate() {
            let header =
                parse_header(shard).with_context(|| format!("read shard {}", index + 1))?;
            check_shard(&header.metadata, index as u64, count)?;
            if index == 0 {
                metadata = header.metadata;
                header_bytes = header.data_start;
            }
            for (name, mut info) in header.tensors {
                info.offset += base;
                if tensors.insert(name.clone(), info).is_some() {
                    bail!("tensor {name} is in two shards");
                }
            }
        }
        if let Some(MetaValue::Int(expected)) = metadata.get("split.tensors.count")
            && *expected != tensors.len() as u64
        {
            bail!("the shards hold {} tensors, not {expected}", tensors.len());
        }
        Ok(Self {
            source: Box::new(source),
            tensors,
            metadata,
            header_bytes,
        })
    }

    pub fn metadata_str(&self, key: &str) -> Result<&str> {
        match self.metadata.get(key) {
            Some(MetaValue::Str(value)) => Ok(value),
            _ => bail!("claimed GGUF has no string {key}"),
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
        self.source.prefetch(ranges)
    }

    /// The bytes of a tensor, `(offset, len)`.
    pub fn tensor_range(&self, name: &str) -> Result<(u64, u64)> {
        let t = self.tensor(name)?;
        Ok((t.offset, t.total_bytes()))
    }

    /// The bytes of row `row` of a weight, `(offset, len)`.
    pub fn row_range(&self, name: &str, row: u64) -> Result<(u64, u64)> {
        let t = self.tensor(name)?;
        if row >= t.dims.iter().skip(1).product() {
            bail!("{name} has no row {row}");
        }
        let row_bytes = t.row_bytes()?;
        Ok((t.offset + row * row_bytes, row_bytes))
    }

    /// Dequantized row `row` of a 2D weight, in f64.
    pub fn row(&self, name: &str, row: u64) -> Result<Vec<f64>> {
        let t = self.tensor(name)?;
        let row_bytes = t.row_bytes()?;
        let raw = self.source.read_at(t.offset + row * row_bytes, row_bytes)?;
        dequantize_row(t.ggml_type, &raw, usize::try_from(t.row_len())?)
            .with_context(|| format!("dequantize {name} row {row}"))
    }

    /// Row `row` of a Q4_K or Q5_K weight as its sub-blocks of 32 values.
    pub fn min_blocks(&self, name: &str, row: u64) -> Result<Vec<crate::dequant::MinBlock>> {
        let t = self.tensor(name)?;
        let row_bytes = t.row_bytes()?;
        let raw = self.source.read_at(t.offset + row * row_bytes, row_bytes)?;
        crate::dequant::min_blocks(t.ggml_type, &raw)
            .with_context(|| format!("read {name} row {row}"))
    }

    /// Calls `visit` with each of `rows` of a 2D weight, dequantized, in
    /// order. Reads many rows at a time.
    pub fn each_row(
        &self,
        name: &str,
        rows: std::ops::Range<u64>,
        visit: &mut dyn FnMut(u64, &[f64]),
    ) -> Result<()> {
        self.visit_rows(name, rows, visit, false)
    }

    /// [`Self::each_row`] for rows read once: a sparse copy does not keep
    /// them.
    pub fn stream_rows(
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
        let t = self.tensor(name)?;
        if rows.end > t.rows() {
            bail!("{name} has {} rows, not {}", t.rows(), rows.end);
        }
        let row_bytes = t.row_bytes()?;
        let n = usize::try_from(t.row_len())?;
        let per_chunk = (CHUNK_BYTES / row_bytes.max(1)).max(1);
        let mut first = rows.start;
        while first < rows.end {
            let count = per_chunk.min(rows.end - first);
            let (offset, len) = (t.offset + first * row_bytes, count * row_bytes);
            let raw = if once {
                self.source.read_once(offset, len)?
            } else {
                self.source.read_at(offset, len)?
            };
            for (i, row) in raw.chunks_exact(usize::try_from(row_bytes)?).enumerate() {
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
                *out.entry(layer).or_insert(0) += t.total_bytes();
            }
        }
        out
    }
}

/// Checks that the shard at `index` of `count` says it is.
fn check_shard(metadata: &BTreeMap<String, MetaValue>, index: u64, count: u64) -> Result<()> {
    let int = |key: &str| match metadata.get(key) {
        Some(MetaValue::Int(value)) => Ok(*value),
        _ => bail!("shard {} has no {key}", index + 1),
    };
    if int("split.no")? != index || int("split.count")? != count {
        bail!(
            "shard {} of {count} is not the shard its name says",
            index + 1
        );
    }
    Ok(())
}

struct Cursor<'a> {
    source: &'a dyn ByteSource,
    buf: Vec<u8>,
    pos: usize,
}

impl Cursor<'_> {
    fn bytes(&mut self, n: usize) -> Result<&[u8]> {
        while self.pos + n > self.buf.len() {
            let start = self.buf.len() as u64;
            let len = (self.buf.len() as u64)
                .max(1 << 20)
                .min(self.source.size() - start);
            if len == 0 {
                bail!("GGUF header runs past the end of the file");
            }
            let chunk = self.source.read_at(start, len)?;
            self.buf.extend_from_slice(&chunk);
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes(8)?.try_into()?))
    }
    fn string(&mut self) -> Result<String> {
        let n = usize::try_from(self.u64()?)?;
        Ok(String::from_utf8_lossy(self.bytes(n)?).into_owned())
    }
    /// Reads a metadata value, returning it when it is a scalar.
    fn value(&mut self, ty: u32) -> Result<Option<MetaValue>> {
        let fixed = |ty: u32| match ty {
            0 | 1 | 7 => Some(1usize),
            2 | 3 => Some(2),
            4..=6 => Some(4),
            10..=12 => Some(8),
            _ => None,
        };
        if let Some(n) = fixed(ty) {
            let b = self.bytes(n)?;
            let mut le = [0u8; 8];
            le[..n].copy_from_slice(b);
            return Ok(match ty {
                6 => Some(MetaValue::Float(f64::from(f32::from_le_bytes(
                    le[..4].try_into()?,
                )))),
                12 => Some(MetaValue::Float(f64::from_le_bytes(le))),
                7 => None,
                _ => Some(MetaValue::Int(u64::from_le_bytes(le))),
            });
        }
        match ty {
            8 => {
                return Ok(Some(MetaValue::Str(self.string()?)));
            }
            9 => {
                let inner = self.u32()?;
                let count = self.u64()?;
                for _ in 0..count {
                    self.value(inner)?;
                }
            }
            other => bail!("unknown GGUF metadata type {other}"),
        }
        Ok(None)
    }
}

struct Header {
    tensors: BTreeMap<String, TensorInfo>,
    metadata: BTreeMap<String, MetaValue>,
    data_start: u64,
}

fn parse_header(source: &dyn ByteSource) -> Result<Header> {
    let mut c = Cursor {
        source,
        buf: Vec::new(),
        pos: 0,
    };
    if c.bytes(4)? != b"GGUF" {
        bail!("not a GGUF file");
    }
    let version = c.u32()?;
    if version < 2 {
        bail!("GGUF version {version} is too old");
    }
    let n_tensors = c.u64()?;
    let n_kv = c.u64()?;
    let mut metadata = BTreeMap::new();
    for _ in 0..n_kv {
        let key = c.string()?;
        let ty = c.u32()?;
        if let Some(value) = c.value(ty)? {
            metadata.insert(key, value);
        }
    }
    let alignment = match metadata.get("general.alignment") {
        Some(MetaValue::Int(value)) => (*value).max(1),
        _ => 32,
    };
    let mut infos = Vec::new();
    for _ in 0..n_tensors {
        let name = c.string()?;
        let n_dims = c.u32()?;
        let dims = (0..n_dims).map(|_| c.u64()).collect::<Result<Vec<_>>>()?;
        let ggml_type = c.u32()?;
        let rel = c.u64()?;
        infos.push((name, dims, ggml_type, rel));
    }
    let data_start = (c.pos as u64).div_ceil(alignment) * alignment;
    let tensors = infos
        .into_iter()
        .map(|(name, dims, ggml_type, rel)| {
            let info = TensorInfo {
                name: name.clone(),
                dims,
                ggml_type,
                offset: data_start + rel,
            };
            (name, info)
        })
        .collect();
    Ok(Header {
        tensors,
        metadata,
        data_start,
    })
}

/// Small f32 GGUF files in memory, for tests.
#[cfg(test)]
pub(crate) mod test_gguf {
    use super::*;
    use crate::dequant::GGML_F32;

    pub(crate) struct Memory(pub(crate) Vec<u8>);

    impl ByteSource for Memory {
        fn read_at(&self, offset: u64, len: u64) -> Result<Vec<u8>> {
            let start = usize::try_from(offset)?;
            let end = start + usize::try_from(len)?;
            self.0
                .get(start..end)
                .map(<[u8]>::to_vec)
                .context("read past the end")
        }

        fn size(&self) -> u64 {
            self.0.len() as u64
        }
    }

    fn string(out: &mut Vec<u8>, s: &str) {
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

    /// The bytes of a GGUF with `metadata` and f32 `tensors`.
    pub(crate) fn bytes(
        metadata: &[(&str, MetaValue)],
        tensors: &[(&str, Vec<u64>, Vec<f32>)],
    ) -> Vec<u8> {
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
            }
        }
        let mut offset = 0u64;
        for (name, dims, values) in tensors {
            string(&mut out, name);
            out.extend((dims.len() as u32).to_le_bytes());
            for d in dims {
                out.extend(d.to_le_bytes());
            }
            out.extend(GGML_F32.to_le_bytes());
            out.extend(offset.to_le_bytes());
            offset += (values.len() as u64 * 4).div_ceil(32) * 32;
        }
        out.resize(out.len().div_ceil(32) * 32, 0);
        for (_, _, values) in tensors {
            for v in values {
                out.extend(v.to_le_bytes());
            }
            out.resize(out.len().div_ceil(32) * 32, 0);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_are_read_in_order_across_chunks() {
        let weights = test_gguf::open(
            &[],
            &[("w", vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])],
        );
        let mut seen = Vec::new();
        weights
            .each_row("w", 1..3, &mut |row, values| {
                seen.push((row, values.to_vec()))
            })
            .unwrap();
        assert_eq!(seen, vec![(1, vec![3.0, 4.0]), (2, vec![5.0, 6.0])]);
        assert!(weights.each_row("w", 0..4, &mut |_, _| {}).is_err());
    }

    fn shard(no: u64, tensors: &[(&str, Vec<u64>, Vec<f32>)]) -> Box<dyn ByteSource> {
        let mut metadata = vec![
            ("split.no", MetaValue::Int(no)),
            ("split.count", MetaValue::Int(2)),
        ];
        if no == 0 {
            metadata.push(("general.architecture", MetaValue::Str("test".into())));
            metadata.push(("split.tensors.count", MetaValue::Int(2)));
        }
        Box::new(test_gguf::Memory(test_gguf::bytes(&metadata, tensors)))
    }

    #[test]
    fn a_split_gguf_reads_as_one_file() {
        let weights = WeightFile::open_shards(vec![
            shard(0, &[("a", vec![2, 1], vec![1.0, 2.0])]),
            shard(1, &[("b", vec![2, 2], vec![3.0, 4.0, 5.0, 6.0])]),
        ])
        .unwrap();
        assert_eq!(
            weights.metadata_str("general.architecture").unwrap(),
            "test"
        );
        assert_eq!(weights.row("a", 0).unwrap(), vec![1.0, 2.0]);
        assert_eq!(weights.row("b", 1).unwrap(), vec![5.0, 6.0]);
        let (offset, len) = weights.row_range("b", 1).unwrap();
        let bytes = weights.source.read_at(offset, len).unwrap();
        let expected: Vec<u8> = [5.0f32, 6.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(bytes, expected);
    }

    #[test]
    fn shards_out_of_order_are_refused() {
        let swapped = WeightFile::open_shards(vec![
            shard(1, &[("b", vec![2, 2], vec![3.0, 4.0, 5.0, 6.0])]),
            shard(0, &[("a", vec![2, 1], vec![1.0, 2.0])]),
        ]);
        assert!(swapped.is_err());
    }
}
