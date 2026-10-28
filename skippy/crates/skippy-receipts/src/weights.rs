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

/// ggml type ids of the weight formats the verifier can dequantize.
pub const GGML_F32: u32 = 0;
pub const GGML_F16: u32 = 1;
pub const GGML_Q8_0: u32 = 8;
pub const GGML_BF16: u32 = 30;

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
        0 => (1, 4),
        1 => (1, 2),
        2 => (32, 18),
        3 => (32, 20),
        6 => (32, 22),
        7 => (32, 24),
        8 => (32, 34),
        10 => (256, 84),
        11 => (256, 110),
        12 => (256, 144),
        13 => (256, 176),
        14 => (256, 210),
        20 => (32, 18),
        23 => (256, 136),
        30 => (1, 2),
        _ => return None,
    })
}

pub fn type_name(t: u32) -> String {
    match t {
        0 => "f32".into(),
        1 => "f16".into(),
        2 => "q4_0".into(),
        3 => "q4_1".into(),
        8 => "q8_0".into(),
        12 => "q4_K".into(),
        13 => "q5_K".into(),
        14 => "q6_K".into(),
        30 => "bf16".into(),
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

    /// Dequantized row `row` of a 2D weight, in f64.
    pub fn row(&self, name: &str, row: u64) -> Result<Vec<f64>> {
        let t = self.tensor(name)?;
        let row_bytes = t.row_bytes()?;
        let raw = self.source.read_at(t.offset + row * row_bytes, row_bytes)?;
        dequantize_row(t.ggml_type, &raw, usize::try_from(t.row_len())?)
            .with_context(|| format!("dequantize {name} row {row}"))
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

/// Reference dequantization of one row.
pub fn dequantize_row(ggml_type: u32, raw: &[u8], n: usize) -> Result<Vec<f64>> {
    let mut out = Vec::with_capacity(n);
    match ggml_type {
        GGML_F32 => out.extend(
            raw.as_chunks::<4>()
                .0
                .iter()
                .map(|c| f64::from(f32::from_le_bytes(*c))),
        ),
        GGML_F16 => out.extend(
            raw.as_chunks::<2>()
                .0
                .iter()
                .map(|c| f64::from(half::f16::from_le_bytes(*c).to_f32())),
        ),
        GGML_BF16 => out.extend(
            raw.as_chunks::<2>()
                .0
                .iter()
                .map(|c| f64::from(half::bf16::from_le_bytes(*c).to_f32())),
        ),
        GGML_Q8_0 => {
            for block in raw.as_chunks::<34>().0 {
                let d = f64::from(half::f16::from_le_bytes([block[0], block[1]]).to_f32());
                out.extend(block[2..].iter().map(|&q| d * f64::from(q as i8)));
            }
        }
        other => bail!("no audit profile for weight type {} yet", type_name(other)),
    }
    if out.len() != n {
        bail!("row decoded to {} values, expected {n}", out.len());
    }
    Ok(out)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_0_block_dequantizes_to_scale_times_quant() {
        let mut block = half::f16::from_f32(0.5).to_le_bytes().to_vec();
        block.extend((0..32).map(|i| (i as i8 - 16) as u8));
        let row = dequantize_row(GGML_Q8_0, &block, 32).unwrap();
        assert_eq!(row[0], -8.0);
        assert_eq!(row[31], 7.5);
    }

    #[test]
    fn unknown_types_are_refused() {
        assert!(dequantize_row(12, &[0u8; 144], 256).is_err());
    }
}
