//! A GGUF v3 writer for the F32 and Q8_0 tensors DFlash drafts carry.

use std::io::Write;

use crate::command::DynResult;

const ALIGNMENT: u64 = 32;
const Q8_0_BLOCK: usize = 32;
const Q8_0_BLOCK_BYTES: u64 = 34;

#[derive(Clone, Debug, PartialEq)]
pub(super) enum Value {
    U32(u32),
    F32(f32),
    Bool(bool),
    Str(String),
    I32s(Vec<i32>),
    Bools(Vec<bool>),
    Strs(Vec<String>),
}

impl Value {
    fn type_id(&self) -> u32 {
        match self {
            Self::U32(_) => 4,
            Self::F32(_) => 6,
            Self::Bool(_) => 7,
            Self::Str(_) => 8,
            Self::I32s(_) | Self::Bools(_) | Self::Strs(_) => 9,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TensorType {
    F32,
    Q8_0,
}

impl TensorType {
    fn id(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::Q8_0 => 8,
        }
    }

    pub(super) fn byte_len(self, elements: u64) -> u64 {
        match self {
            Self::F32 => elements * 4,
            Self::Q8_0 => elements / Q8_0_BLOCK as u64 * Q8_0_BLOCK_BYTES,
        }
    }
}

/// A tensor as GGUF lists it: `dims` innermost first.
#[derive(Clone, Debug)]
pub(super) struct TensorEntry {
    pub(super) name: String,
    pub(super) dims: Vec<u64>,
    pub(super) kind: TensorType,
}

impl TensorEntry {
    fn byte_len(&self) -> u64 {
        self.kind.byte_len(self.dims.iter().product())
    }
}

fn padding(len: u64) -> u64 {
    (ALIGNMENT - len % ALIGNMENT) % ALIGNMENT
}

fn put_str(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}

fn put_value(out: &mut Vec<u8>, value: &Value) {
    match value {
        Value::U32(v) => out.extend_from_slice(&v.to_le_bytes()),
        Value::F32(v) => out.extend_from_slice(&v.to_le_bytes()),
        Value::Bool(v) => out.push(u8::from(*v)),
        Value::Str(v) => put_str(out, v),
        Value::I32s(values) => {
            out.extend_from_slice(&5u32.to_le_bytes());
            out.extend_from_slice(&(values.len() as u64).to_le_bytes());
            values
                .iter()
                .for_each(|v| out.extend_from_slice(&v.to_le_bytes()));
        }
        Value::Bools(values) => {
            out.extend_from_slice(&7u32.to_le_bytes());
            out.extend_from_slice(&(values.len() as u64).to_le_bytes());
            values.iter().for_each(|v| out.push(u8::from(*v)));
        }
        Value::Strs(values) => {
            out.extend_from_slice(&8u32.to_le_bytes());
            out.extend_from_slice(&(values.len() as u64).to_le_bytes());
            values.iter().for_each(|v| put_str(out, v));
        }
    }
}

/// The header, tensor table and alignment padding before the first tensor.
pub(super) fn header(kvs: &[(String, Value)], tensors: &[TensorEntry]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
    for (key, value) in kvs {
        put_str(&mut out, key);
        out.extend_from_slice(&value.type_id().to_le_bytes());
        put_value(&mut out, value);
    }
    let mut offset = 0u64;
    for tensor in tensors {
        put_str(&mut out, &tensor.name);
        out.extend_from_slice(&(tensor.dims.len() as u32).to_le_bytes());
        tensor
            .dims
            .iter()
            .for_each(|d| out.extend_from_slice(&d.to_le_bytes()));
        out.extend_from_slice(&tensor.kind.id().to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
        offset += tensor.byte_len() + padding(tensor.byte_len());
    }
    out.resize(out.len() + padding(out.len() as u64) as usize, 0);
    out
}

/// Writes one tensor's data, padded to the alignment, refusing a length
/// other than the table promised.
pub(super) fn write_tensor(
    out: &mut impl Write,
    entry: &TensorEntry,
    values: &[f32],
) -> DynResult<()> {
    let bytes = match entry.kind {
        TensorType::F32 => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        TensorType::Q8_0 => quantize_q8_0(values)?,
    };
    if bytes.len() as u64 != entry.byte_len() {
        return Err(format!(
            "{}: {} bytes, expected {}",
            entry.name,
            bytes.len(),
            entry.byte_len()
        )
        .into());
    }
    out.write_all(&bytes)?;
    out.write_all(&vec![0u8; padding(bytes.len() as u64) as usize])?;
    Ok(())
}

/// ggml's reference Q8_0: per 32 values, `d = amax / 127` stored as f16 and
/// `round(x / d)` as i8, rounding half away from zero.
pub(super) fn quantize_q8_0(values: &[f32]) -> DynResult<Vec<u8>> {
    if !values.len().is_multiple_of(Q8_0_BLOCK) {
        return Err("Q8_0 needs a multiple of 32 values per row".into());
    }
    let mut out = Vec::with_capacity(values.len() / Q8_0_BLOCK * Q8_0_BLOCK_BYTES as usize);
    for block in values.as_chunks::<Q8_0_BLOCK>().0 {
        if block.iter().any(|v| !v.is_finite()) {
            return Err("Q8_0 input is not finite".into());
        }
        let amax = block.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        let d = amax / 127.0;
        let id = if d == 0.0 { 0.0 } else { 1.0 / d };
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        for value in block {
            let scaled = value * id;
            let magnitude = scaled.abs();
            let floored = magnitude.floor();
            let rounded = floored + (2.0 * (magnitude - floored)).floor();
            let q = if scaled < 0.0 { -rounded } else { rounded };
            out.push((q as i32) as i8 as u8);
        }
    }
    Ok(out)
}

/// IEEE binary32 to binary16, rounding to nearest even.
pub(super) fn f32_to_f16(value: f32) -> u16 {
    let x = value.to_bits();
    let sign = (x & 0x8000_0000) >> 16;
    let exp = x & 0x7f80_0000;
    let man = x & 0x007f_ffff;
    if exp == 0x7f80_0000 {
        let nan = if man == 0 { 0 } else { 0x0200 };
        return (sign | 0x7c00 | nan | (man >> 13)) as u16;
    }
    let half_exp = ((exp >> 23) as i32) - 127 + 15;
    if half_exp >= 0x1f {
        return (sign | 0x7c00) as u16;
    }
    if half_exp <= 0 {
        if 14 - half_exp > 24 {
            return sign as u16;
        }
        let man = man | 0x0080_0000;
        let shift = (14 - half_exp) as u32;
        let mut half_man = man >> shift;
        let round_bit = 1u32 << (shift - 1);
        if man & round_bit != 0 && man & (3 * round_bit - 1) != 0 {
            half_man += 1;
        }
        return (sign | half_man) as u16;
    }
    let bits = sign | ((half_exp as u32) << 10) | (man >> 13);
    let round_bit = 0x0000_1000;
    if man & round_bit != 0 && man & (3 * round_bit - 1) != 0 {
        (bits + 1) as u16
    } else {
        bits as u16
    }
}
