use anyhow::{Result, anyhow, ensure};

use super::{FlashAttentionType, RuntimeConfig, SplitMode};

pub const GGML_TYPE_F16: u32 = 1;
pub const GGML_TYPE_Q4_0: u32 = 2;
pub const GGML_TYPE_Q8_0: u32 = 8;
// This ID belongs to the packaged Skippy ABI 0.1.72 runtime, not the public fork.
pub const GGML_TYPE_TURBO4_0: u32 = 43;

pub fn parse_cache_type(value: &str) -> Result<u32> {
    let normalized = value.trim().to_ascii_lowercase().replace('-', "_");
    match normalized.as_str() {
        "" | "f16" => Ok(GGML_TYPE_F16),
        "q4" | "q4_0" => Ok(GGML_TYPE_Q4_0),
        "q8" | "q8_0" => Ok(GGML_TYPE_Q8_0),
        "turbo4" => Ok(GGML_TYPE_TURBO4_0),
        _ => Err(anyhow!("unsupported KV cache type {value:?}")),
    }
}

pub(super) fn validate_layout(config: &RuntimeConfig) -> Result<(), &'static str> {
    if config.cache_type_k == GGML_TYPE_TURBO4_0 {
        return Err("Turbo4 is supported only for the value cache");
    }
    if config.cache_type_v != GGML_TYPE_TURBO4_0 {
        return Ok(());
    }
    if config.cache_type_k != GGML_TYPE_Q8_0 {
        return Err("Turbo4 value cache requires q8_0 keys");
    }
    if config.flash_attn_type == FlashAttentionType::Disabled {
        return Err("Turbo4 value cache requires Flash Attention");
    }
    if config.split_mode == SplitMode::Tensor {
        return Err("Turbo4 value cache does not support tensor splitting");
    }
    Ok(())
}

pub(super) fn require_native_support(features: Option<u64>) -> Result<()> {
    ensure!(
        features.is_some_and(|bits| bits & skippy_ffi::FEATURE_TURBO4_KV != 0),
        "the loaded native runtime does not advertise Turbo4 KV support"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_existing_types_and_the_product_turbo4_id() -> Result<()> {
        for (name, expected) in [
            ("f16", GGML_TYPE_F16),
            ("q8_0", GGML_TYPE_Q8_0),
            ("q4_0", GGML_TYPE_Q4_0),
            (" Q8-0 ", GGML_TYPE_Q8_0),
            ("turbo4", 43),
        ] {
            assert_eq!(parse_cache_type(name)?, expected);
        }
        for unsupported in ["turbo2", "turbo3", "tq4_1s", "43", "47"] {
            assert!(parse_cache_type(unsupported).is_err());
        }
        Ok(())
    }

    #[test]
    fn turbo4_rejects_unsupported_layouts_before_native_context_creation() {
        let config = RuntimeConfig {
            cache_type_k: GGML_TYPE_Q8_0,
            cache_type_v: GGML_TYPE_TURBO4_0,
            flash_attn_type: FlashAttentionType::Enabled,
            ..RuntimeConfig::default()
        };
        assert!(validate_layout(&config).is_ok());
        for changed in [
            RuntimeConfig {
                cache_type_k: GGML_TYPE_TURBO4_0,
                ..config.clone()
            },
            RuntimeConfig {
                cache_type_k: GGML_TYPE_F16,
                ..config.clone()
            },
            RuntimeConfig {
                flash_attn_type: FlashAttentionType::Disabled,
                ..config.clone()
            },
            RuntimeConfig {
                split_mode: SplitMode::Tensor,
                ..config
            },
        ] {
            assert!(validate_layout(&changed).is_err());
        }
    }

    #[test]
    fn missing_native_capability_is_rejected_without_calling_native_code() {
        assert!(require_native_support(None).is_err());
        assert!(require_native_support(Some(0)).is_err());
        assert!(require_native_support(Some(skippy_ffi::FEATURE_LAYA_DECISIONS)).is_err());
        assert!(require_native_support(Some(skippy_ffi::FEATURE_TURBO4_KV)).is_ok());
        assert!(
            require_native_support(Some(
                skippy_ffi::FEATURE_TURBO4_KV | skippy_ffi::FEATURE_LAYA_DECISIONS
            ))
            .is_ok()
        );
    }
}
