use std::str::FromStr;

use clap::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ConvertOutputType {
    F32,
    F16,
    Bf16,
    Q8_0,
    TQ1_0,
    TQ2_0,
    Auto,
}

impl ConvertOutputType {
    pub fn as_arg(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::Q8_0 => "q8_0",
            Self::TQ1_0 => "tq1_0",
            Self::TQ2_0 => "tq2_0",
            Self::Auto => "auto",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum JobKind {
    ConvertHf,
    QuantizeGguf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum QuantType {
    Q1_0,
    Q2K,
    Q2KS,
    Q3K,
    Q3KS,
    Q3KM,
    Q3KL,
    Q4_0,
    Q4_1,
    Q4K,
    Q4KS,
    Q4KM,
    Q5_0,
    Q5_1,
    Q5K,
    Q5KS,
    Q5KM,
    Q6K,
    Q8_0,
    IQ1S,
    IQ1M,
    IQ2XXS,
    IQ2XS,
    IQ2S,
    IQ2M,
    IQ3XXS,
    IQ3XS,
    IQ3S,
    IQ3M,
    IQ4NL,
    IQ4XS,
    TQ1_0,
    TQ2_0,
    Mxfp4Moe,
    F16,
    Bf16,
    F32,
    Copy,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantSpec {
    base_quant: QuantType,
}

impl FromStr for QuantSpec {
    type Err = String;

    fn from_str(raw: &str) -> std::result::Result<Self, Self::Err> {
        let base_quant = raw.parse::<QuantType>()?;
        Ok(Self { base_quant })
    }
}

impl QuantSpec {
    pub fn base_quant(&self) -> QuantType {
        self.base_quant
    }

    pub fn output_name(&self) -> &'static str {
        self.base_quant.as_llama_name()
    }

    pub fn validate_recipe_requirements(
        &self,
        _has_tensor_type_file: bool,
    ) -> std::result::Result<(), String> {
        Ok(())
    }
}

impl From<QuantType> for QuantSpec {
    fn from(base_quant: QuantType) -> Self {
        Self { base_quant }
    }
}

impl FromStr for QuantType {
    type Err = String;

    fn from_str(raw: &str) -> std::result::Result<Self, Self::Err> {
        if let Ok(ftype) = raw.parse::<i32>() {
            return Self::from_llama_ftype_id(ftype)
                .ok_or_else(|| format!("unsupported quant ftype id {ftype}"));
        }
        let normalized = normalize_type_name(raw);
        let quant = match normalized.as_str() {
            "Q10" => Self::Q1_0,
            "Q2K" => Self::Q2K,
            "Q2KS" => Self::Q2KS,
            "Q3K" => Self::Q3K,
            "Q3KS" => Self::Q3KS,
            "Q3KM" => Self::Q3KM,
            "Q3KL" => Self::Q3KL,
            "Q40" => Self::Q4_0,
            "Q41" => Self::Q4_1,
            "Q4K" => Self::Q4K,
            "Q4KS" => Self::Q4KS,
            "Q4KM" => Self::Q4KM,
            "Q50" => Self::Q5_0,
            "Q51" => Self::Q5_1,
            "Q5K" => Self::Q5K,
            "Q5KS" => Self::Q5KS,
            "Q5KM" => Self::Q5KM,
            "Q6K" => Self::Q6K,
            "Q80" => Self::Q8_0,
            "IQ1S" => Self::IQ1S,
            "IQ1M" => Self::IQ1M,
            "IQ2XXS" => Self::IQ2XXS,
            "IQ2XS" => Self::IQ2XS,
            "IQ2S" => Self::IQ2S,
            "IQ2M" => Self::IQ2M,
            "IQ3XXS" => Self::IQ3XXS,
            "IQ3XS" => Self::IQ3XS,
            "IQ3S" => Self::IQ3S,
            "IQ3M" => Self::IQ3M,
            "IQ4NL" => Self::IQ4NL,
            "IQ4XS" => Self::IQ4XS,
            "TQ10" => Self::TQ1_0,
            "TQ20" => Self::TQ2_0,
            "MXFP4MOE" => Self::Mxfp4Moe,
            "F16" => Self::F16,
            "BF16" => Self::Bf16,
            "F32" => Self::F32,
            "COPY" => Self::Copy,
            _ => return Err(unsupported_quant_type_error(raw, &normalized)),
        };
        Ok(quant)
    }
}

impl QuantType {
    pub const ALL: &'static [Self] = &[
        Self::Q1_0,
        Self::Q2K,
        Self::Q2KS,
        Self::Q3K,
        Self::Q3KS,
        Self::Q3KM,
        Self::Q3KL,
        Self::Q4_0,
        Self::Q4_1,
        Self::Q4K,
        Self::Q4KS,
        Self::Q4KM,
        Self::Q5_0,
        Self::Q5_1,
        Self::Q5K,
        Self::Q5KS,
        Self::Q5KM,
        Self::Q6K,
        Self::Q8_0,
        Self::IQ1S,
        Self::IQ1M,
        Self::IQ2XXS,
        Self::IQ2XS,
        Self::IQ2S,
        Self::IQ2M,
        Self::IQ3XXS,
        Self::IQ3XS,
        Self::IQ3S,
        Self::IQ3M,
        Self::IQ4NL,
        Self::IQ4XS,
        Self::TQ1_0,
        Self::TQ2_0,
        Self::Mxfp4Moe,
        Self::F16,
        Self::Bf16,
        Self::F32,
        Self::Copy,
    ];

    pub fn as_llama_name(self) -> &'static str {
        match self {
            Self::Q1_0 => "Q1_0",
            Self::Q2K => "Q2_K",
            Self::Q2KS => "Q2_K_S",
            Self::Q3K => "Q3_K",
            Self::Q3KS => "Q3_K_S",
            Self::Q3KM => "Q3_K_M",
            Self::Q3KL => "Q3_K_L",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q4K => "Q4_K",
            Self::Q4KS => "Q4_K_S",
            Self::Q4KM => "Q4_K_M",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q5K => "Q5_K",
            Self::Q5KS => "Q5_K_S",
            Self::Q5KM => "Q5_K_M",
            Self::Q6K => "Q6_K",
            Self::Q8_0 => "Q8_0",
            Self::IQ1S => "IQ1_S",
            Self::IQ1M => "IQ1_M",
            Self::IQ2XXS => "IQ2_XXS",
            Self::IQ2XS => "IQ2_XS",
            Self::IQ2S => "IQ2_S",
            Self::IQ2M => "IQ2_M",
            Self::IQ3XXS => "IQ3_XXS",
            Self::IQ3XS => "IQ3_XS",
            Self::IQ3S => "IQ3_S",
            Self::IQ3M => "IQ3_M",
            Self::IQ4NL => "IQ4_NL",
            Self::IQ4XS => "IQ4_XS",
            Self::TQ1_0 => "TQ1_0",
            Self::TQ2_0 => "TQ2_0",
            Self::Mxfp4Moe => "MXFP4_MOE",
            Self::F16 => "F16",
            Self::Bf16 => "BF16",
            Self::F32 => "F32",
            Self::Copy => "COPY",
        }
    }

    pub fn from_llama_ftype_id(ftype: i32) -> Option<Self> {
        match ftype {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            7 => Some(Self::Q8_0),
            8 => Some(Self::Q5_0),
            9 => Some(Self::Q5_1),
            10 => Some(Self::Q2K),
            11 => Some(Self::Q3KS),
            12 => Some(Self::Q3K),
            13 => Some(Self::Q3KL),
            14 => Some(Self::Q4KS),
            15 => Some(Self::Q4K),
            16 => Some(Self::Q5KS),
            17 => Some(Self::Q5K),
            18 => Some(Self::Q6K),
            19 => Some(Self::IQ2XXS),
            20 => Some(Self::IQ2XS),
            21 => Some(Self::Q2KS),
            22 => Some(Self::IQ3XS),
            23 => Some(Self::IQ3XXS),
            24 => Some(Self::IQ1S),
            25 => Some(Self::IQ4NL),
            26 => Some(Self::IQ3S),
            27 => Some(Self::IQ3M),
            28 => Some(Self::IQ2S),
            29 => Some(Self::IQ2M),
            30 => Some(Self::IQ4XS),
            31 => Some(Self::IQ1M),
            32 => Some(Self::Bf16),
            36 => Some(Self::TQ1_0),
            37 => Some(Self::TQ2_0),
            38 => Some(Self::Mxfp4Moe),
            40 => Some(Self::Q1_0),
            _ => None,
        }
    }

    pub fn as_llama_file_type(self) -> llama_quant_ffi::LlamaFileType {
        match self {
            Self::Q1_0 => llama_quant_ffi::LlamaFileType::MostlyQ1_0,
            Self::Q2K => llama_quant_ffi::LlamaFileType::MostlyQ2K,
            Self::Q2KS => llama_quant_ffi::LlamaFileType::MostlyQ2KS,
            Self::Q3K | Self::Q3KM => llama_quant_ffi::LlamaFileType::MostlyQ3KM,
            Self::Q3KS => llama_quant_ffi::LlamaFileType::MostlyQ3KS,
            Self::Q3KL => llama_quant_ffi::LlamaFileType::MostlyQ3KL,
            Self::Q4_0 => llama_quant_ffi::LlamaFileType::MostlyQ4_0,
            Self::Q4_1 => llama_quant_ffi::LlamaFileType::MostlyQ4_1,
            Self::Q4K | Self::Q4KM => llama_quant_ffi::LlamaFileType::MostlyQ4KM,
            Self::Q4KS => llama_quant_ffi::LlamaFileType::MostlyQ4KS,
            Self::Q5_0 => llama_quant_ffi::LlamaFileType::MostlyQ5_0,
            Self::Q5_1 => llama_quant_ffi::LlamaFileType::MostlyQ5_1,
            Self::Q5K | Self::Q5KM => llama_quant_ffi::LlamaFileType::MostlyQ5KM,
            Self::Q5KS => llama_quant_ffi::LlamaFileType::MostlyQ5KS,
            Self::Q6K => llama_quant_ffi::LlamaFileType::MostlyQ6K,
            Self::Q8_0 => llama_quant_ffi::LlamaFileType::MostlyQ8_0,
            Self::IQ1S => llama_quant_ffi::LlamaFileType::MostlyIQ1S,
            Self::IQ1M => llama_quant_ffi::LlamaFileType::MostlyIQ1M,
            Self::IQ2XXS => llama_quant_ffi::LlamaFileType::MostlyIQ2XXS,
            Self::IQ2XS => llama_quant_ffi::LlamaFileType::MostlyIQ2XS,
            Self::IQ2S => llama_quant_ffi::LlamaFileType::MostlyIQ2S,
            Self::IQ2M => llama_quant_ffi::LlamaFileType::MostlyIQ2M,
            Self::IQ3XXS => llama_quant_ffi::LlamaFileType::MostlyIQ3XXS,
            Self::IQ3XS => llama_quant_ffi::LlamaFileType::MostlyIQ3XS,
            Self::IQ3S => llama_quant_ffi::LlamaFileType::MostlyIQ3S,
            Self::IQ3M => llama_quant_ffi::LlamaFileType::MostlyIQ3M,
            Self::IQ4NL => llama_quant_ffi::LlamaFileType::MostlyIQ4NL,
            Self::IQ4XS => llama_quant_ffi::LlamaFileType::MostlyIQ4XS,
            Self::TQ1_0 => llama_quant_ffi::LlamaFileType::MostlyTQ1_0,
            Self::TQ2_0 => llama_quant_ffi::LlamaFileType::MostlyTQ2_0,
            Self::Mxfp4Moe => llama_quant_ffi::LlamaFileType::MostlyMxfp4Moe,
            Self::F16 => llama_quant_ffi::LlamaFileType::MostlyF16,
            Self::Bf16 => llama_quant_ffi::LlamaFileType::MostlyBf16,
            Self::F32 | Self::Copy => llama_quant_ffi::LlamaFileType::AllF32,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TensorType {
    F32,
    F16,
    Q1_0,
    Q2_0,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    IQ1S,
    IQ1M,
    IQ2XXS,
    IQ2XS,
    IQ2S,
    IQ3XXS,
    IQ3S,
    IQ4NL,
    IQ4XS,
    I8,
    I16,
    I32,
    I64,
    F64,
    TQ1_0,
    TQ2_0,
    Mxfp4,
    Nvfp4,
    Bf16,
}

impl TensorType {
    pub fn as_ggml_type(self) -> Option<llama_quant_ffi::GgmlType> {
        match self {
            Self::F32 => Some(llama_quant_ffi::GgmlType::F32),
            Self::F16 => Some(llama_quant_ffi::GgmlType::F16),
            Self::Q1_0 => Some(llama_quant_ffi::GgmlType::Q1_0),
            Self::Q2_0 => Some(llama_quant_ffi::GgmlType::Q2_0),
            Self::Q4_0 => Some(llama_quant_ffi::GgmlType::Q4_0),
            Self::Q4_1 => Some(llama_quant_ffi::GgmlType::Q4_1),
            Self::Q5_0 => Some(llama_quant_ffi::GgmlType::Q5_0),
            Self::Q5_1 => Some(llama_quant_ffi::GgmlType::Q5_1),
            Self::Q8_0 => Some(llama_quant_ffi::GgmlType::Q8_0),
            Self::Q8_1 => Some(llama_quant_ffi::GgmlType::Q8_1),
            Self::Q2K => Some(llama_quant_ffi::GgmlType::Q2K),
            Self::Q3K => Some(llama_quant_ffi::GgmlType::Q3K),
            Self::Q4K => Some(llama_quant_ffi::GgmlType::Q4K),
            Self::Q5K => Some(llama_quant_ffi::GgmlType::Q5K),
            Self::Q6K => Some(llama_quant_ffi::GgmlType::Q6K),
            Self::Q8K => Some(llama_quant_ffi::GgmlType::Q8K),
            Self::IQ1S => Some(llama_quant_ffi::GgmlType::IQ1S),
            Self::IQ1M => Some(llama_quant_ffi::GgmlType::IQ1M),
            Self::IQ2XXS => Some(llama_quant_ffi::GgmlType::IQ2XXS),
            Self::IQ2XS => Some(llama_quant_ffi::GgmlType::IQ2XS),
            Self::IQ2S => Some(llama_quant_ffi::GgmlType::IQ2S),
            Self::IQ3XXS => Some(llama_quant_ffi::GgmlType::IQ3XXS),
            Self::IQ3S => Some(llama_quant_ffi::GgmlType::IQ3S),
            Self::IQ4NL => Some(llama_quant_ffi::GgmlType::IQ4NL),
            Self::IQ4XS => Some(llama_quant_ffi::GgmlType::IQ4XS),
            Self::I8 => Some(llama_quant_ffi::GgmlType::I8),
            Self::I16 => Some(llama_quant_ffi::GgmlType::I16),
            Self::I32 => Some(llama_quant_ffi::GgmlType::I32),
            Self::I64 => Some(llama_quant_ffi::GgmlType::I64),
            Self::F64 => Some(llama_quant_ffi::GgmlType::F64),
            Self::TQ1_0 => Some(llama_quant_ffi::GgmlType::TQ1_0),
            Self::TQ2_0 => Some(llama_quant_ffi::GgmlType::TQ2_0),
            Self::Mxfp4 => Some(llama_quant_ffi::GgmlType::Mxfp4),
            Self::Nvfp4 => Some(llama_quant_ffi::GgmlType::Nvfp4),
            Self::Bf16 => Some(llama_quant_ffi::GgmlType::Bf16),
        }
    }

    pub const ALL: &'static [Self] = &[
        Self::F32,
        Self::F16,
        Self::Q1_0,
        Self::Q2_0,
        Self::Q4_0,
        Self::Q4_1,
        Self::Q5_0,
        Self::Q5_1,
        Self::Q8_0,
        Self::Q8_1,
        Self::Q2K,
        Self::Q3K,
        Self::Q4K,
        Self::Q5K,
        Self::Q6K,
        Self::Q8K,
        Self::IQ1S,
        Self::IQ1M,
        Self::IQ2XXS,
        Self::IQ2XS,
        Self::IQ2S,
        Self::IQ3XXS,
        Self::IQ3S,
        Self::IQ4NL,
        Self::IQ4XS,
        Self::I8,
        Self::I16,
        Self::I32,
        Self::I64,
        Self::F64,
        Self::TQ1_0,
        Self::TQ2_0,
        Self::Mxfp4,
        Self::Nvfp4,
        Self::Bf16,
    ];

    pub fn as_ggml_name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q1_0 => "Q1_0",
            Self::Q2_0 => "Q2_0",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2K => "Q2_K",
            Self::Q3K => "Q3_K",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
            Self::Q8K => "Q8_K",
            Self::IQ1S => "IQ1_S",
            Self::IQ1M => "IQ1_M",
            Self::IQ2XXS => "IQ2_XXS",
            Self::IQ2XS => "IQ2_XS",
            Self::IQ2S => "IQ2_S",
            Self::IQ3XXS => "IQ3_XXS",
            Self::IQ3S => "IQ3_S",
            Self::IQ4NL => "IQ4_NL",
            Self::IQ4XS => "IQ4_XS",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::F64 => "F64",
            Self::TQ1_0 => "TQ1_0",
            Self::TQ2_0 => "TQ2_0",
            Self::Mxfp4 => "MXFP4",
            Self::Nvfp4 => "NVFP4",
            Self::Bf16 => "BF16",
        }
    }

    pub fn parse(raw: &str) -> Option<Self> {
        let normalized = normalize_type_name(raw);
        match normalized.as_str() {
            "F32" => Some(Self::F32),
            "F16" => Some(Self::F16),
            "Q10" => Some(Self::Q1_0),
            "Q20" => Some(Self::Q2_0),
            "Q40" => Some(Self::Q4_0),
            "Q41" => Some(Self::Q4_1),
            "Q50" => Some(Self::Q5_0),
            "Q51" => Some(Self::Q5_1),
            "Q80" => Some(Self::Q8_0),
            "Q81" => Some(Self::Q8_1),
            "Q2K" => Some(Self::Q2K),
            "Q3K" => Some(Self::Q3K),
            "Q4K" => Some(Self::Q4K),
            "Q5K" => Some(Self::Q5K),
            "Q6K" => Some(Self::Q6K),
            "Q8K" => Some(Self::Q8K),
            "IQ1S" => Some(Self::IQ1S),
            "IQ1M" => Some(Self::IQ1M),
            "IQ2XXS" => Some(Self::IQ2XXS),
            "IQ2XS" => Some(Self::IQ2XS),
            "IQ2S" => Some(Self::IQ2S),
            "IQ3XXS" => Some(Self::IQ3XXS),
            "IQ3S" => Some(Self::IQ3S),
            "IQ4NL" => Some(Self::IQ4NL),
            "IQ4XS" => Some(Self::IQ4XS),
            "I8" => Some(Self::I8),
            "I16" => Some(Self::I16),
            "I32" => Some(Self::I32),
            "I64" => Some(Self::I64),
            "F64" => Some(Self::F64),
            "TQ10" => Some(Self::TQ1_0),
            "TQ20" => Some(Self::TQ2_0),
            "MXFP4" => Some(Self::Mxfp4),
            "NVFP4" => Some(Self::Nvfp4),
            "BF16" => Some(Self::Bf16),
            _ => None,
        }
    }
}

fn normalize_type_name(raw: &str) -> String {
    raw.chars()
        .filter(|ch| !matches!(*ch, '_' | '-'))
        .flat_map(char::to_uppercase)
        .collect()
}

fn unsupported_quant_type_error(raw: &str, normalized: &str) -> String {
    if let Some(base) = normalized.strip_prefix("UD")
        && let Some(base_quant) = base_quant_from_profile_suffix(base)
    {
        return format!(
            "unsupported quant type {raw:?}: UD-* labels are custom tensor-type recipes, \
             not upstream llama-quantize whole-model modes; use base quant {base_quant:?} \
             with --tensor-type-file for the dynamic recipe"
        );
    }
    if normalized == "Q4KXL" {
        return "unsupported quant type \"Q4_K_XL\": Q4_K_XL is a custom high-quality recipe, \
                not an upstream llama-quantize whole-model mode; use base quant \"Q4_K_M\" \
                with --tensor-type-file for the XL recipe"
            .to_string();
    }
    if normalized.ends_with("MTPQ8") {
        return format!(
            "unsupported quant type {raw:?}: MTP-Q8 is a custom artifact profile, \
             not a whole-model quant mode; pass the base quant with --quant and \
             the tensor policy with --tensor-type-file"
        );
    }
    format!("unsupported quant type {raw:?}")
}

fn base_quant_from_profile_suffix(normalized_suffix: &str) -> Option<&'static str> {
    match normalized_suffix {
        "Q2K" => Some("Q2_K"),
        "Q2KS" => Some("Q2_K_S"),
        "Q3K" => Some("Q3_K"),
        "Q3KS" => Some("Q3_K_S"),
        "Q3KM" => Some("Q3_K_M"),
        "Q3KL" => Some("Q3_K_L"),
        "Q4K" => Some("Q4_K"),
        "Q4KS" => Some("Q4_K_S"),
        "Q4KM" => Some("Q4_K_M"),
        "Q5K" => Some("Q5_K"),
        "Q5KS" => Some("Q5_K_S"),
        "Q5KM" => Some("Q5_K_M"),
        "Q6K" => Some("Q6_K"),
        other => QuantType::from_str(other)
            .ok()
            .map(QuantType::as_llama_name),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::fs;
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn accepts_raw_tensor_types_but_not_ftype_mixtures() {
        assert!(TensorType::parse("Q1_0").is_some());
        assert!(TensorType::parse("Q2_0").is_some());
        assert!(TensorType::parse("Q3_K").is_some());
        assert!(TensorType::parse("q4_K").is_some());
        assert!(TensorType::parse("Q6_K").is_some());
        assert!(TensorType::parse("Q8_0").is_some());
        assert!(TensorType::parse("MXFP4").is_some());
        assert!(TensorType::parse("NVFP4").is_some());
        assert!(TensorType::parse("I8").is_some());
        assert!(TensorType::parse("F64").is_some());
        assert!(TensorType::parse("IQ2_M").is_none());
        assert!(TensorType::parse("IQ3_XS").is_none());
        assert!(TensorType::parse("IQ3_M").is_none());
        assert!(TensorType::parse("Q3_K_S").is_none());
        assert!(TensorType::parse("Q4_K_M").is_none());
    }

    #[test]
    fn quant_names_match_llama_cli() {
        assert_eq!(QuantType::Q2K.as_llama_name(), "Q2_K");
        assert_eq!(QuantType::Q3KS.as_llama_name(), "Q3_K_S");
        assert_eq!(QuantType::Mxfp4Moe.as_llama_name(), "MXFP4_MOE");
    }

    #[test]
    fn parses_llama_quant_names() {
        assert_eq!("Q2_K".parse::<QuantType>().unwrap(), QuantType::Q2K);
        assert_eq!("q2-k".parse::<QuantType>().unwrap(), QuantType::Q2K);
        assert_eq!("q2k".parse::<QuantType>().unwrap(), QuantType::Q2K);
        assert_eq!(
            "MXFP4_MOE".parse::<QuantType>().unwrap(),
            QuantType::Mxfp4Moe
        );
        assert!("NVFP4".parse::<QuantType>().is_err());
    }

    #[test]
    fn parses_quant_specs_from_llama_quant_names() {
        assert!("UD-Q3_K_S".parse::<QuantSpec>().is_err());
        assert!("Q4_K_XL".parse::<QuantSpec>().is_err());
        assert!("Q2_K-MTP-Q8".parse::<QuantSpec>().is_err());
        let regular = "Q4_K_M".parse::<QuantSpec>().unwrap();
        assert_eq!(regular.base_quant(), QuantType::Q4KM);
        assert_eq!(regular.output_name(), "Q4_K_M");
        assert!(regular.validate_recipe_requirements(false).is_ok());
    }

    #[test]
    fn parses_llama_numeric_ftype_ids() {
        assert_eq!("0".parse::<QuantType>().unwrap(), QuantType::F32);
        assert_eq!("12".parse::<QuantType>().unwrap(), QuantType::Q3K);
        assert_eq!("15".parse::<QuantType>().unwrap(), QuantType::Q4K);
        assert_eq!("17".parse::<QuantType>().unwrap(), QuantType::Q5K);
        assert_eq!("40".parse::<QuantType>().unwrap(), QuantType::Q1_0);
        assert!("999".parse::<QuantType>().is_err());
    }

    #[test]
    fn parses_every_current_llama_quant_cli_name() {
        for name in pinned_llama_quant_option_names() {
            assert!(
                name.parse::<QuantType>().is_ok(),
                "{name} should parse as a llama-quantize mode"
            );
        }
    }

    #[test]
    fn rejects_quant_modes_not_in_current_llama_cli() {
        let xl_error = "Q4_K_XL".parse::<QuantType>().unwrap_err();
        assert!(xl_error.contains("custom high-quality recipe"));
        assert!(xl_error.contains("Q4_K_M"));

        let ud_error = "UD-Q3_K_S".parse::<QuantType>().unwrap_err();
        assert!(ud_error.contains("custom tensor-type recipes"));
        assert!(ud_error.contains("Q3_K_S"));
    }

    #[test]
    fn parses_current_llama_ftype_names() {
        let names = ["MXFP4_MOE", "Q1_0"];
        for name in names {
            assert!(
                name.parse::<QuantType>().is_ok(),
                "{name} should parse as a llama ftype-backed mode"
            );
        }
    }

    #[test]
    fn public_quant_catalog_is_parseable() {
        assert!(QuantType::ALL.contains(&QuantType::Copy));
        assert!(QuantType::ALL.contains(&QuantType::Mxfp4Moe));
        for quant in QuantType::ALL {
            assert_eq!(quant.as_llama_name().parse::<QuantType>().unwrap(), *quant);
        }
    }

    #[test]
    fn public_quant_catalog_covers_pinned_llama_quantize_table() {
        let pinned = pinned_llama_quant_option_names()
            .into_iter()
            .collect::<BTreeSet<_>>();
        let local = QuantType::ALL
            .iter()
            .map(|quant| quant.as_llama_name().to_string())
            .collect::<BTreeSet<_>>();

        let missing = pinned.difference(&local).collect::<Vec<_>>();
        assert!(
            missing.is_empty(),
            "local quant catalog is missing pinned llama-quantize modes: {missing:?}"
        );

        let extra = local.difference(&pinned).collect::<Vec<_>>();
        assert!(
            extra.is_empty(),
            "local quant catalog has modes missing from pinned llama-quantize: {extra:?}"
        );
    }

    #[test]
    fn public_tensor_catalog_is_parseable() {
        assert!(TensorType::ALL.contains(&TensorType::Nvfp4));
        assert!(TensorType::ALL.contains(&TensorType::Mxfp4));
        for tensor_type in TensorType::ALL {
            assert_eq!(
                TensorType::parse(tensor_type.as_ggml_name()).unwrap(),
                *tensor_type
            );
        }
    }

    fn pinned_llama_quant_option_names() -> Vec<String> {
        const UNSUPPORTED_FFI_QUANT_MODES: &[&str] = &["Q2_0"];
        let quantize_cpp = repo_root().join(".deps/llama.cpp/tools/quantize/quantize.cpp");
        let source = fs::read_to_string(&quantize_cpp)
            .unwrap_or_else(|err| panic!("read {}: {err}", quantize_cpp.display()));
        let table = source
            .split("static const std::vector<quant_option> QUANT_OPTIONS = {")
            .nth(1)
            .and_then(|rest| rest.split_once("};").map(|(table, _)| table))
            .expect("find QUANT_OPTIONS table");
        table
            .lines()
            .filter_map(|line| {
                let line = line.trim();
                line.strip_prefix("{ \"")
                    .and_then(|rest| rest.split_once('"').map(|(name, _)| name.to_string()))
            })
            .filter(|name| !UNSUPPORTED_FFI_QUANT_MODES.contains(&name.as_str()))
            .collect()
    }

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|path| path.parent())
            .expect("crate lives under crates/skippy-quantize")
            .to_path_buf()
    }
}
