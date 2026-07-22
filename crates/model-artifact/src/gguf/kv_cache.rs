use super::GgufCompactMeta;

impl GgufCompactMeta {
    pub fn k_cache_bytes_per_token_f16(&self) -> Option<u64> {
        GgufKvCacheQuant::f16().k_cache_bytes_per_token(self)
    }

    pub fn v_cache_bytes_per_token_f16(&self) -> Option<u64> {
        GgufKvCacheQuant::f16().v_cache_bytes_per_token(self)
    }

    pub fn kv_cache_bytes_per_token_f16(&self) -> Option<u64> {
        GgufKvCacheQuant::f16().kv_cache_bytes_per_token(self)
    }

    fn kv_cache_head_count(&self) -> Option<u32> {
        // GLM-DSA uses absorbed MLA: cache one compressed KV group rather
        // than one expanded vector for every attention head.
        if self.architecture == "glm-dsa" {
            Some(1)
        } else {
            self.effective_kv_head_count()
        }
    }

    fn kv_cache_value_length(&self) -> u32 {
        // The cached V row is the compressed KV latent. The regular
        // attention value length describes the expanded per-head value.
        if self.architecture == "glm-dsa" && self.kv_lora_rank > 0 {
            self.kv_lora_rank
        } else {
            self.value_length
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GgufKvCacheType {
    F16,
    Q8_0,
    Q4_0,
}

impl GgufKvCacheType {
    pub fn from_llama_arg(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "f16" => Some(Self::F16),
            "q8_0" => Some(Self::Q8_0),
            "q4_0" => Some(Self::Q4_0),
            _ => None,
        }
    }

    pub const fn as_llama_arg(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Q8_0 => "q8_0",
            Self::Q4_0 => "q4_0",
        }
    }

    fn block_shape(self) -> (u64, u64) {
        match self {
            Self::F16 => (1, 2),
            Self::Q8_0 => (32, 34),
            Self::Q4_0 => (32, 18),
        }
    }

    fn bytes_for_elements(self, elements: u64) -> Option<u64> {
        let (block_elements, block_bytes) = self.block_shape();
        let blocks = elements
            .checked_add(block_elements.checked_sub(1)?)?
            .checked_div(block_elements)?;
        blocks.checked_mul(block_bytes)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GgufKvCacheQuant {
    pub k: GgufKvCacheType,
    pub v: GgufKvCacheType,
}

impl GgufKvCacheQuant {
    /// f16 K + f16 V — highest quality, largest KV cache.
    pub const F16: Self = Self {
        k: GgufKvCacheType::F16,
        v: GgufKvCacheType::F16,
    };

    /// q8_0 K + q8_0 V — moderate compression.
    pub const Q8_0: Self = Self {
        k: GgufKvCacheType::Q8_0,
        v: GgufKvCacheType::Q8_0,
    };

    /// q4_0 K + q4_0 V — most aggressive compression, smallest KV cache.
    pub const Q4_0: Self = Self {
        k: GgufKvCacheType::Q4_0,
        v: GgufKvCacheType::Q4_0,
    };

    pub const fn new(k: GgufKvCacheType, v: GgufKvCacheType) -> Self {
        Self { k, v }
    }

    pub const fn f16() -> Self {
        Self::F16
    }

    /// Returns `true` if `self` uses more aggressive (smaller) quantisation
    /// than `other`.
    pub const fn is_more_aggressive_than(self, other: Self) -> bool {
        Self::aggressiveness(self) > Self::aggressiveness(other)
    }

    const fn aggressiveness(q: Self) -> u8 {
        Self::type_aggressiveness(q.k) + Self::type_aggressiveness(q.v)
    }

    const fn type_aggressiveness(t: GgufKvCacheType) -> u8 {
        match t {
            GgufKvCacheType::F16 => 0,
            GgufKvCacheType::Q8_0 => 1,
            GgufKvCacheType::Q4_0 => 2,
        }
    }

    pub fn from_llama_args(cache_type_k: &str, cache_type_v: &str) -> Option<Self> {
        Some(Self {
            k: GgufKvCacheType::from_llama_arg(cache_type_k)?,
            v: GgufKvCacheType::from_llama_arg(cache_type_v)?,
        })
    }

    pub fn k_cache_bytes_per_token(self, meta: &GgufCompactMeta) -> Option<u64> {
        cache_bytes_per_token(meta, meta.key_length, self.k)
    }

    pub fn v_cache_bytes_per_token(self, meta: &GgufCompactMeta) -> Option<u64> {
        cache_bytes_per_token(meta, meta.kv_cache_value_length(), self.v)
    }

    pub fn kv_cache_bytes_per_token(self, meta: &GgufCompactMeta) -> Option<u64> {
        self.k_cache_bytes_per_token(meta)?
            .checked_add(self.v_cache_bytes_per_token(meta)?)
    }
}

fn cache_bytes_per_token(
    meta: &GgufCompactMeta,
    vector_length: u32,
    cache_type: GgufKvCacheType,
) -> Option<u64> {
    let kv_heads = u64::from(meta.kv_cache_head_count()?);
    let vector_length = u64::from((vector_length > 0).then_some(vector_length)?);
    let layers = u64::from((meta.layer_count > 0).then_some(meta.layer_count)?);
    let elements_per_layer = kv_heads.checked_mul(vector_length)?;
    cache_type
        .bytes_for_elements(elements_per_layer)?
        .checked_mul(layers)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prices_key_and_value_types_independently() {
        let meta = GgufCompactMeta {
            head_count: 32,
            kv_head_count: 8,
            layer_count: 24,
            key_length: 128,
            value_length: 128,
            ..Default::default()
        };
        let quant = GgufKvCacheQuant::new(GgufKvCacheType::Q8_0, GgufKvCacheType::Q4_0);

        assert_eq!(quant.k_cache_bytes_per_token(&meta), Some(26_112));
        assert_eq!(quant.v_cache_bytes_per_token(&meta), Some(13_824));
        assert_eq!(quant.kv_cache_bytes_per_token(&meta), Some(39_936));
    }

    #[test]
    fn prices_key_and_value_widths_independently() {
        let meta = GgufCompactMeta {
            head_count: 32,
            kv_head_count: 8,
            layer_count: 24,
            key_length: 64,
            value_length: 256,
            ..Default::default()
        };
        let quant = GgufKvCacheQuant::new(GgufKvCacheType::Q8_0, GgufKvCacheType::Q4_0);

        assert_eq!(quant.k_cache_bytes_per_token(&meta), Some(13_056));
        assert_eq!(quant.v_cache_bytes_per_token(&meta), Some(27_648));
        assert_eq!(quant.kv_cache_bytes_per_token(&meta), Some(40_704));
    }

    #[test]
    fn prices_glm_dsa_absorbed_mla_shape() {
        let meta = GgufCompactMeta {
            architecture: "glm-dsa".to_string(),
            head_count: 64,
            kv_head_count: 64,
            layer_count: 79,
            key_length: 576,
            value_length: 256,
            kv_lora_rank: 512,
            ..Default::default()
        };

        assert_eq!(
            GgufKvCacheQuant::Q4_0.k_cache_bytes_per_token(&meta),
            Some(25_596)
        );
        assert_eq!(
            GgufKvCacheQuant::Q4_0.v_cache_bytes_per_token(&meta),
            Some(22_752)
        );
        assert_eq!(
            GgufKvCacheQuant::Q4_0.kv_cache_bytes_per_token(&meta),
            Some(48_348)
        );
    }

    #[test]
    fn returns_none_when_required_fields_are_missing() {
        let meta = GgufCompactMeta {
            head_count: 32,
            layer_count: 24,
            key_length: 128,
            ..Default::default()
        };

        assert_eq!(meta.k_cache_bytes_per_token_f16(), Some(196_608));
        assert_eq!(meta.v_cache_bytes_per_token_f16(), None);
        assert_eq!(
            GgufKvCacheQuant::f16().kv_cache_bytes_per_token(&meta),
            None
        );
    }
}
