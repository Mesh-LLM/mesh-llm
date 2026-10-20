use anyhow::{Result, anyhow, ensure};

use super::{FlashAttentionType, RuntimeConfig, SplitMode};

pub const GGML_TYPE_F16: u32 = 1;
pub const GGML_TYPE_Q4_0: u32 = 2;
pub const GGML_TYPE_Q8_0: u32 = 8;
// This ID belongs to the packaged Skippy runtime, not the public fork.
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
    if !matches!(config.cache_type_k, GGML_TYPE_F16 | GGML_TYPE_Q8_0) {
        return Err("Turbo4 value cache requires f16 or q8_0 keys");
    }
    if config.flash_attn_type == FlashAttentionType::Disabled {
        return Err("Turbo4 value cache requires Flash Attention");
    }
    if config.split_mode == SplitMode::Tensor {
        return Err("Turbo4 value cache does not support tensor splitting");
    }
    Ok(())
}

pub(super) fn require_native_support(features: Option<u64>, key_type: u32) -> Result<()> {
    let (required, name) = match key_type {
        GGML_TYPE_F16 => (skippy_ffi::FEATURE_TURBO4_F16_KV, "F16-key/Turbo4-value"),
        GGML_TYPE_Q8_0 => (skippy_ffi::FEATURE_TURBO4_KV, "Q8-key/Turbo4-value"),
        _ => return Err(anyhow!("unsupported Turbo4 key cache type {key_type}")),
    };
    ensure!(
        features.is_some_and(|bits| bits & required != 0),
        "the loaded native runtime does not advertise {name} support"
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
        assert!(
            validate_layout(&RuntimeConfig {
                cache_type_k: GGML_TYPE_F16,
                ..config.clone()
            })
            .is_ok()
        );
        for changed in [
            RuntimeConfig {
                cache_type_k: GGML_TYPE_TURBO4_0,
                ..config.clone()
            },
            RuntimeConfig {
                cache_type_k: GGML_TYPE_Q4_0,
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
        let q8 = skippy_ffi::FEATURE_TURBO4_KV;
        let f16 = skippy_ffi::FEATURE_TURBO4_F16_KV;
        assert_eq!(q8 & f16, 0);
        for (key, required, unrelated) in [(GGML_TYPE_F16, f16, q8), (GGML_TYPE_Q8_0, q8, f16)] {
            for absent in [
                None,
                Some(0),
                Some(skippy_ffi::FEATURE_LAYA_DECISIONS),
                Some(unrelated),
            ] {
                assert!(require_native_support(absent, key).is_err());
            }
            assert!(require_native_support(Some(required), key).is_ok());
            assert!(
                require_native_support(Some(q8 | f16 | skippy_ffi::FEATURE_LAYA_DECISIONS), key)
                    .is_ok()
            );
        }
        assert!(require_native_support(Some(q8 | f16), GGML_TYPE_Q4_0).is_err());
    }
}
