use std::io::Write;

use anyhow::{Context, Result};

use crate::types::ConvertOutputType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FloatDType {
    F32,
    F16,
    Bf16,
}

impl FloatDType {
    pub(crate) fn from_safetensor(dtype: &str) -> Option<Self> {
        match dtype {
            "F32" => Some(Self::F32),
            "F16" => Some(Self::F16),
            "BF16" => Some(Self::Bf16),
            _ => None,
        }
    }

    pub(crate) fn byte_size(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
        }
    }
}

pub(crate) fn target_dtype_for(
    source_dtype: FloatDType,
    output_type: Option<ConvertOutputType>,
) -> Result<FloatDType> {
    match output_type {
        None => Ok(source_dtype),
        Some(ConvertOutputType::F32) => Ok(FloatDType::F32),
        Some(ConvertOutputType::F16) => Ok(FloatDType::F16),
        Some(ConvertOutputType::Bf16) => Ok(FloatDType::Bf16),
        Some(other) => {
            anyhow::bail!(
                "native conversion does not support output type {}",
                other.as_arg()
            )
        }
    }
}

pub(crate) fn convert_float_chunk<W: Write>(
    input: &[u8],
    source_dtype: FloatDType,
    target_dtype: FloatDType,
    writer: &mut W,
) -> Result<u64> {
    let element_count = input.len() / source_dtype.byte_size() as usize;
    let output_len = element_count
        .checked_mul(target_dtype.byte_size() as usize)
        .context("converted chunk byte length overflow")?;
    let mut output = Vec::with_capacity(output_len);
    for index in 0..element_count {
        let value = read_float_element(input, source_dtype, index);
        write_float_element(&mut output, target_dtype, value);
    }
    writer.write_all(&output)?;
    Ok(output.len() as u64)
}

fn read_float_element(input: &[u8], dtype: FloatDType, index: usize) -> f32 {
    match dtype {
        FloatDType::F32 => {
            let start = index * 4;
            f32::from_le_bytes(input[start..start + 4].try_into().expect("slice length"))
        }
        FloatDType::F16 => {
            let start = index * 2;
            f16_bits_to_f32(u16::from_le_bytes(
                input[start..start + 2].try_into().expect("slice length"),
            ))
        }
        FloatDType::Bf16 => {
            let start = index * 2;
            f32::from_bits(
                u32::from(u16::from_le_bytes(
                    input[start..start + 2].try_into().expect("slice length"),
                )) << 16,
            )
        }
    }
}

fn write_float_element(output: &mut Vec<u8>, dtype: FloatDType, value: f32) {
    match dtype {
        FloatDType::F32 => output.extend_from_slice(&value.to_le_bytes()),
        FloatDType::F16 => output.extend_from_slice(&f32_to_f16_bits(value).to_le_bytes()),
        FloatDType::Bf16 => output.extend_from_slice(&f32_to_bf16_bits(value).to_le_bytes()),
    }
}

fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding_bias = ((bits >> 16) & 1) + 0x7fff;
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = (u32::from(bits & 0x8000)) << 16;
    let exp = (bits >> 10) & 0x1f;
    let frac = u32::from(bits & 0x03ff);
    let value = if exp == 0 {
        if frac == 0 {
            sign
        } else {
            let mut frac = frac;
            let mut exp_shift = -14_i32;
            while frac & 0x0400 == 0 {
                frac <<= 1;
                exp_shift -= 1;
            }
            frac &= 0x03ff;
            sign | (u32::try_from(exp_shift + 127).unwrap() << 23) | (frac << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (frac << 13)
    } else {
        sign | (u32::from(exp + 112) << 23) | (frac << 13)
    };
    f32::from_bits(value)
}

fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;

    if exp == 0xff {
        if mant == 0 {
            return sign | 0x7c00;
        }
        return sign | 0x7e00;
    }

    let half_exp = exp - 127 + 15;
    if half_exp >= 0x1f {
        return sign | 0x7c00;
    }
    if half_exp <= 0 {
        if half_exp < -10 {
            return sign;
        }
        let mantissa = mant | 0x0080_0000;
        let shift = u32::try_from(14 - half_exp).unwrap();
        let mut half_mant = (mantissa >> shift) as u16;
        if (mantissa >> (shift - 1)) & 1 != 0 {
            half_mant = half_mant.saturating_add(1);
        }
        return sign | half_mant;
    }

    let mut half = sign | ((half_exp as u16) << 10) | ((mant >> 13) as u16);
    if (mant & 0x0000_1000) != 0 {
        half = half.saturating_add(1);
    }
    half
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_dtype_rejects_unresolved_auto() {
        assert_eq!(
            target_dtype_for(FloatDType::F32, None).unwrap(),
            FloatDType::F32
        );
        assert_eq!(
            target_dtype_for(FloatDType::F32, Some(ConvertOutputType::Bf16)).unwrap(),
            FloatDType::Bf16
        );
        assert!(target_dtype_for(FloatDType::F32, Some(ConvertOutputType::Auto)).is_err());
    }
}
