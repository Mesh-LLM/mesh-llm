#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KvCacheType {
    F16,
    Q8_0,
    Q4_0,
}

impl KvCacheType {
    pub(crate) fn as_config_value(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Q8_0 => "q8_0",
            Self::Q4_0 => "q4_0",
        }
    }

    fn to_gguf(self) -> crate::models::gguf::GgufKvCacheType {
        use crate::models::gguf::GgufKvCacheType;
        match self {
            Self::F16 => GgufKvCacheType::F16,
            Self::Q8_0 => GgufKvCacheType::Q8_0,
            Self::Q4_0 => GgufKvCacheType::Q4_0,
        }
    }

    fn from_gguf(value: crate::models::gguf::GgufKvCacheType) -> Self {
        use crate::models::gguf::GgufKvCacheType;
        match value {
            GgufKvCacheType::F16 => Self::F16,
            GgufKvCacheType::Q8_0 => Self::Q8_0,
            GgufKvCacheType::Q4_0 => Self::Q4_0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KvCachePolicy {
    pub(crate) k_type: KvCacheType,
    pub(crate) v_type: KvCacheType,
}

impl KvCachePolicy {
    const LARGE_MODEL_MIN_BYTES: u64 = 50 * 1024 * 1024 * 1024;

    /// Default KV cache policy, tiered by model size.
    ///
    /// Models >= 50 GB use Q4_0 K + Q4_0 V to keep KV cache small enough
    /// that unified-memory machines don't thrash.  On a 480B MoE split
    /// across two Apple Silicon nodes the difference between Q8_0 and Q4_0
    /// is the difference between swap-thrashing at 1 tok/s and running at
    /// 20+ tok/s.
    ///
    /// Smaller models use Q8_0 K + Q8_0 V which gives ~2× compression over
    /// f16 with negligible quality loss.
    ///
    /// Users can override via `--cache-type-k` / `--cache-type-v`.
    pub(crate) fn for_model_size(model_bytes: u64) -> Self {
        if model_bytes >= Self::LARGE_MODEL_MIN_BYTES {
            Self {
                k_type: KvCacheType::Q4_0,
                v_type: KvCacheType::Q4_0,
            }
        } else {
            Self {
                k_type: KvCacheType::Q8_0,
                v_type: KvCacheType::Q8_0,
            }
        }
    }

    fn as_gguf_quant(self) -> crate::models::gguf::GgufKvCacheQuant {
        crate::models::gguf::GgufKvCacheQuant::new(self.k_type.to_gguf(), self.v_type.to_gguf())
    }

    /// Downgrade this *default* policy to one the model can actually load.
    ///
    /// The size tiers above choose a quant purely from byte size; they do not
    /// know whether the model satisfies llama.cpp's quantised-KV constraints
    /// (Flash Attention availability, per-head block alignment). Without this
    /// guard, an incompatible model (e.g. Grok, or a head_dim not divisible by
    /// the q8_0/q4_0 block size of 32) fails the context build outright rather
    /// than degrading. When `meta` proves the chosen quant cannot load, fall
    /// back to f16 K/V.
    ///
    /// This is for policy/family *defaults* only — explicit user overrides must
    /// bypass it and fail loudly. With no metadata we cannot prove
    /// incompatibility, so the policy is returned unchanged (documented limit:
    /// a metadata-only guard also cannot see a backend Flash Attention probe
    /// failure).
    pub(crate) fn guarded_for_model(
        self,
        meta: Option<&crate::models::gguf::GgufCompactMeta>,
    ) -> Self {
        let Some(meta) = meta else {
            return self;
        };
        let compatible = meta.compatible_default_kv_cache_quant(self.as_gguf_quant());
        Self {
            k_type: KvCacheType::from_gguf(compatible.k),
            v_type: KvCacheType::from_gguf(compatible.v),
        }
    }

    pub(crate) fn cache_type_k(self) -> &'static str {
        self.k_type.as_config_value()
    }

    pub(crate) fn cache_type_v(self) -> &'static str {
        self.v_type.as_config_value()
    }

    pub(crate) fn label(self) -> String {
        format!(
            "{} K + {} V",
            self.cache_type_k().to_ascii_uppercase(),
            self.cache_type_v().to_ascii_uppercase()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_model_uses_q8_0() {
        let policy = KvCachePolicy::for_model_size(10 * 1024 * 1024 * 1024);
        assert_eq!(policy.k_type, KvCacheType::Q8_0);
        assert_eq!(policy.v_type, KvCacheType::Q8_0);
    }

    #[test]
    fn large_model_uses_q4_0() {
        let policy = KvCachePolicy::for_model_size(50 * 1024 * 1024 * 1024);
        assert_eq!(policy.k_type, KvCacheType::Q4_0);
        assert_eq!(policy.v_type, KvCacheType::Q4_0);
    }

    fn meta(architecture: &str, head_dim: u32) -> crate::models::gguf::GgufCompactMeta {
        crate::models::gguf::GgufCompactMeta {
            architecture: architecture.to_string(),
            head_count: 32,
            kv_head_count: 8,
            layer_count: 32,
            key_length: head_dim,
            value_length: head_dim,
            ..Default::default()
        }
    }

    #[test]
    fn guard_keeps_quant_for_block_aligned_model() {
        let policy = KvCachePolicy::for_model_size(10 * 1024 * 1024 * 1024);
        let guarded = policy.guarded_for_model(Some(&meta("qwen3", 128)));
        assert_eq!(guarded, policy);
    }

    #[test]
    fn guard_falls_back_to_f16_for_unaligned_head_dim() {
        let policy = KvCachePolicy::for_model_size(10 * 1024 * 1024 * 1024);
        let guarded = policy.guarded_for_model(Some(&meta("phi2", 80)));
        assert_eq!(guarded.k_type, KvCacheType::F16);
        assert_eq!(guarded.v_type, KvCacheType::F16);
    }

    #[test]
    fn guard_falls_back_to_f16_for_grok() {
        let policy = KvCachePolicy::for_model_size(60 * 1024 * 1024 * 1024);
        let guarded = policy.guarded_for_model(Some(&meta("grok", 128)));
        assert_eq!(guarded.k_type, KvCacheType::F16);
        assert_eq!(guarded.v_type, KvCacheType::F16);
    }

    #[test]
    fn guard_is_noop_without_metadata() {
        let policy = KvCachePolicy::for_model_size(10 * 1024 * 1024 * 1024);
        assert_eq!(policy.guarded_for_model(None), policy);
    }
}
