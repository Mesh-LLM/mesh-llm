//! A bounded reader for one SafeTensors file: header, then one tensor at a time.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::command::DynResult;

/// Headers of the drafts we convert are a few KiB; refuse anything absurd.
const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Dtype {
    F32,
    F16,
    Bf16,
}

impl Dtype {
    fn parse(name: &str) -> Option<Self> {
        match name {
            "F32" => Some(Self::F32),
            "F16" => Some(Self::F16),
            "BF16" => Some(Self::Bf16),
            _ => None,
        }
    }

    fn size(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct TensorInfo {
    pub(super) dtype: Dtype,
    /// Row-major, outermost first, as the checkpoint stores it.
    pub(super) shape: Vec<u64>,
    start: u64,
    end: u64,
}

impl TensorInfo {
    pub(super) fn element_count(&self) -> u64 {
        self.shape.iter().product()
    }
}

pub(super) struct Safetensors {
    file: File,
    data_start: u64,
    tensors: BTreeMap<String, TensorInfo>,
}

impl Safetensors {
    pub(super) fn open(path: &Path) -> DynResult<Self> {
        let mut file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let file_len = file.metadata()?.len();
        let mut len = [0u8; 8];
        file.read_exact(&mut len)?;
        let header_len = u64::from_le_bytes(len);
        if header_len > MAX_HEADER_BYTES || 8 + header_len > file_len {
            return Err(format!("{}: invalid SafeTensors header length", path.display()).into());
        }
        let mut header = vec![0u8; header_len as usize];
        file.read_exact(&mut header)?;
        let header: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&header)
            .map_err(|e| format!("{}: SafeTensors header: {e}", path.display()))?;
        let data_start = 8 + header_len;
        let mut tensors = BTreeMap::new();
        for (name, entry) in header {
            if name == "__metadata__" {
                continue;
            }
            let dtype = entry
                .get("dtype")
                .and_then(|v| v.as_str())
                .and_then(Dtype::parse)
                .ok_or_else(|| format!("{name}: unsupported or missing dtype"))?;
            let shape: Vec<u64> = entry
                .get("shape")
                .and_then(|v| v.as_array())
                .ok_or_else(|| format!("{name}: missing shape"))?
                .iter()
                .map(|v| v.as_u64().ok_or_else(|| format!("{name}: bad shape")))
                .collect::<Result<_, _>>()?;
            let offsets = entry
                .get("data_offsets")
                .and_then(|v| v.as_array())
                .filter(|v| v.len() == 2)
                .ok_or_else(|| format!("{name}: missing data_offsets"))?;
            let (start, end) = match (offsets[0].as_u64(), offsets[1].as_u64()) {
                (Some(start), Some(end)) if start <= end => (start, end),
                _ => return Err(format!("{name}: bad data_offsets").into()),
            };
            let bytes = shape
                .iter()
                .try_fold(dtype.size(), |acc, dim| acc.checked_mul(*dim))
                .ok_or_else(|| format!("{name}: shape overflows"))?;
            if end - start != bytes || data_start + end > file_len {
                return Err(format!("{name}: data does not match its shape").into());
            }
            tensors.insert(
                name,
                TensorInfo {
                    dtype,
                    shape,
                    start,
                    end,
                },
            );
        }
        Ok(Self {
            file,
            data_start,
            tensors,
        })
    }

    /// Tensor names in byte order, as llama.cpp's converter visits them.
    pub(super) fn tensors(&self) -> &BTreeMap<String, TensorInfo> {
        &self.tensors
    }

    /// Reads a tensor and widens it to `f32`, exactly.
    pub(super) fn read_f32(&mut self, name: &str) -> DynResult<Vec<f32>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("missing tensor {name}"))?
            .clone();
        let mut bytes = vec![0u8; (info.end - info.start) as usize];
        self.file
            .seek(SeekFrom::Start(self.data_start + info.start))?;
        self.file.read_exact(&mut bytes)?;
        Ok(match info.dtype {
            Dtype::F32 => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect(),
            Dtype::Bf16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16))
                .collect(),
            Dtype::F16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f16_to_f32(u16::from_le_bytes(*b)))
                .collect(),
        })
    }
}

/// IEEE binary16 to binary32; every value is exactly representable.
pub(super) fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exp = u32::from((bits >> 10) & 0x1f);
    let man = u32::from(bits & 0x03ff);
    let value = match (exp, man) {
        (0, 0) => sign,
        (0, _) => {
            // Subnormal: normalise the mantissa.
            let shift = man.leading_zeros() - 21;
            let man = (man << shift) & 0x03ff;
            sign | ((113 - shift) << 23) | (man << 13)
        }
        (0x1f, _) => sign | 0x7f80_0000 | (man << 13),
        _ => sign | ((exp + 112) << 23) | (man << 13),
    };
    f32::from_bits(value)
}
