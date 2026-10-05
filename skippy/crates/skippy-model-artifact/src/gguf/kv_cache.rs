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

    /// Number of elements per quantisation block. `1` for f16 (no blocking),
    /// `32` for the q8_0/q4_0 formats. llama.cpp requires the cached per-head
    /// width to be an exact multiple of this when the type is quantised
    /// (`llama-context.cpp:3595` for K, `:3606` for V).
    pub const fn block_elements(self) -> u32 {
        match self {
            Self::F16 => 1,
            Self::Q8_0 | Self::Q4_0 => 32,
        }
    }

    /// Whether this type is a blocked/quantised format (anything but f16).
    pub const fn is_quantized(self) -> bool {
        self.block_elements() > 1
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
    let vector_length = u64::from((vector_length > 0).then_some(vector_length)?);
    let layers = u64::from((meta.layer_count > 0).then_some(meta.layer_count)?);
    // Models with per-layer KV head counts (e.g. inkling's hybrid attention)
    // price each layer by its own head count rather than a single global one.
    // GLM-DSA keeps its absorbed-MLA special case via kv_cache_head_count.
    if meta.architecture != "glm-dsa"
        && meta.kv_head_counts.len() == layers as usize
        && meta.kv_head_counts.iter().all(|head_count| *head_count > 0)
    {
        return meta
            .kv_head_counts
            .iter()
            .try_fold(0u64, |total, head_count| {
                let elements = u64::from(*head_count).checked_mul(vector_length)?;
                total.checked_add(cache_type.bytes_for_elements(elements)?)
            });
    }
    let kv_heads = u64::from(meta.kv_cache_head_count()?);
    let elements_per_layer = kv_heads.checked_mul(vector_length)?;
    cache_type
        .bytes_for_elements(elements_per_layer)?
        .checked_mul(layers)
}

/// Why a model cannot load a given quantised KV cache, per the llama.cpp
/// constraints in the pinned tree (`.deps/llama.cpp`, `llama-context.cpp`).
///
/// These are hard load failures (the context builder returns `nullptr` /
/// throws), not quality warnings — a default policy that selects an
/// unsupported quant would crash the model load outright.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvQuantUnsupported {
    /// Quantised V cache requires Flash Attention (`llama-context.cpp:3617`),
    /// but this architecture forces FA off (`:3579`, currently Grok), so a
    /// quantised V can never load.
    FlashAttentionUnavailable,
    /// Quantised K needs the cached per-head key width to be a multiple of the
    /// block size (`:3595`); this model's is not.
    KeyWidthNotBlockAligned { head_width: u32, block: u32 },
    /// Quantised V needs the cached per-head value width to be a multiple of
    /// the block size (`:3606`); this model's is not.
    ValueWidthNotBlockAligned { head_width: u32, block: u32 },
}

impl GgufCompactMeta {
    /// True when llama.cpp forces Flash Attention off for this architecture, in
    /// which case a quantised V cache cannot load (`llama-context.cpp:3579`).
    ///
    /// Grok is the only such arch in the pinned tree. This is a metadata-only
    /// view: it cannot see a *backend* FA probe failure (`:570`), which is a
    /// separate, documented limitation of guarding from metadata alone.
    fn flash_attention_forced_off(&self) -> bool {
        self.architecture == "grok"
    }

    /// Per-head `(key, value)` widths llama.cpp's block-alignment check can
    /// observe for this model. `n_embd_head_k/v(il)` returns the SWA width for
    /// sliding-window layers and the full width otherwise
    /// (`llama-hparams.cpp:115/:123`), and the quantised-KV check iterates
    /// every layer (`llama-context.cpp:3638/:3652`), so a quantised cache
    /// loads only when **every** distinct width the model uses is
    /// block-aligned. The SWA entry is `(0, 0)` when the GGUF carries no
    /// separate SWA widths; llama.cpp then reuses the full widths for SWA
    /// layers, which the first entry already covers.
    fn cached_head_widths(&self) -> [(u32, u32); 2] {
        [
            (self.key_length, self.kv_cache_value_length()),
            (self.key_length_swa, self.value_length_swa),
        ]
    }

    /// Returns `Ok(())` if this model can load `desired` per llama.cpp's
    /// flash-attention and block-alignment constraints, or the first reason it
    /// cannot. f16 K/V is always supported.
    ///
    /// Uses the *cached* per-head widths (`kv_cache_value_length` collapses the
    /// absorbed-MLA latent for glm-dsa), so the check reasons about what is
    /// actually stored, not the expanded attention width. Both the full and
    /// the SWA widths are validated whenever the GGUF carries distinct ones
    /// (21 model loaders create SWA layers whose widths can differ from the
    /// full-attention widths).
    pub fn kv_cache_quant_support(
        &self,
        desired: GgufKvCacheQuant,
    ) -> Result<(), KvQuantUnsupported> {
        if desired.v.is_quantized() && self.flash_attention_forced_off() {
            return Err(KvQuantUnsupported::FlashAttentionUnavailable);
        }
        for (head_k_width, head_v_width) in self.cached_head_widths() {
            if desired.k.is_quantized()
                && head_k_width > 0
                && !head_k_width.is_multiple_of(desired.k.block_elements())
            {
                return Err(KvQuantUnsupported::KeyWidthNotBlockAligned {
                    head_width: head_k_width,
                    block: desired.k.block_elements(),
                });
            }
            if desired.v.is_quantized()
                && head_v_width > 0
                && !head_v_width.is_multiple_of(desired.v.block_elements())
            {
                return Err(KvQuantUnsupported::ValueWidthNotBlockAligned {
                    head_width: head_v_width,
                    block: desired.v.block_elements(),
                });
            }
        }
        Ok(())
    }

    /// Resolve a *default* KV quant to one this model can actually load: returns
    /// `desired` when supported, else falls back to f16 K/V.
    ///
    /// Intended only for policy/family defaults. Explicit user overrides must
    /// NOT be routed through here — an override that cannot load should fail
    /// loudly with llama.cpp's own error rather than being silently rewritten.
    pub fn compatible_default_kv_cache_quant(&self, desired: GgufKvCacheQuant) -> GgufKvCacheQuant {
        match self.kv_cache_quant_support(desired) {
            Ok(()) => desired,
            Err(_) => GgufKvCacheQuant::F16,
        }
    }
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

    /// A conventional dense model (Llama/Qwen-shaped, head_dim 128) supports
    /// the quantised defaults the size policy hands out.
    #[test]
    fn block_aligned_dense_model_supports_quantized_kv() {
        let meta = GgufCompactMeta {
            architecture: "qwen3".to_string(),
            head_count: 32,
            kv_head_count: 8,
            layer_count: 36,
            key_length: 128,
            value_length: 128,
            ..Default::default()
        };

        assert_eq!(meta.kv_cache_quant_support(GgufKvCacheQuant::Q8_0), Ok(()));
        assert_eq!(meta.kv_cache_quant_support(GgufKvCacheQuant::Q4_0), Ok(()));
        assert_eq!(
            meta.compatible_default_kv_cache_quant(GgufKvCacheQuant::Q8_0),
            GgufKvCacheQuant::Q8_0
        );
        assert_eq!(
            meta.compatible_default_kv_cache_quant(GgufKvCacheQuant::Q4_0),
            GgufKvCacheQuant::Q4_0
        );
    }

    /// head_dim = 80 (e.g. Phi-2) is not a multiple of the q8_0/q4_0 block size
    /// (32), so llama.cpp rejects a quantised cache; the default must fall back
    /// to f16 instead of crashing the load.
    #[test]
    fn head_dim_not_block_aligned_falls_back_to_f16() {
        let meta = GgufCompactMeta {
            architecture: "phi2".to_string(),
            head_count: 32,
            kv_head_count: 32,
            layer_count: 32,
            key_length: 80,
            value_length: 80,
            ..Default::default()
        };

        assert_eq!(
            meta.kv_cache_quant_support(GgufKvCacheQuant::Q8_0),
            Err(KvQuantUnsupported::KeyWidthNotBlockAligned {
                head_width: 80,
                block: 32,
            })
        );
        assert_eq!(
            meta.compatible_default_kv_cache_quant(GgufKvCacheQuant::Q8_0),
            GgufKvCacheQuant::F16
        );
        assert_eq!(
            meta.compatible_default_kv_cache_quant(GgufKvCacheQuant::Q4_0),
            GgufKvCacheQuant::F16
        );
    }

    /// Grok forces Flash Attention off, and a quantised V cache requires FA, so
    /// any quantised default must fall back to f16 even though its head_dim
    /// (128) is block-aligned.
    #[test]
    fn grok_forces_f16_because_flash_attention_is_off() {
        let meta = GgufCompactMeta {
            architecture: "grok".to_string(),
            head_count: 48,
            kv_head_count: 8,
            layer_count: 64,
            key_length: 128,
            value_length: 128,
            ..Default::default()
        };

        assert_eq!(
            meta.kv_cache_quant_support(GgufKvCacheQuant::Q4_0),
            Err(KvQuantUnsupported::FlashAttentionUnavailable)
        );
        assert_eq!(
            meta.compatible_default_kv_cache_quant(GgufKvCacheQuant::Q4_0),
            GgufKvCacheQuant::F16
        );
        // f16 remains supported everywhere.
        assert_eq!(meta.kv_cache_quant_support(GgufKvCacheQuant::F16), Ok(()));
    }

    /// SWA layers can carry their own per-head widths. When both the full and
    /// the SWA widths are block-aligned the quantised default must be kept —
    /// this is the common shape (e.g. Gemma-3 with a 256 SWA head width).
    #[test]
    fn aligned_swa_widths_keep_quantized_kv() {
        let meta = GgufCompactMeta {
            architecture: "gemma3".to_string(),
            head_count: 32,
            kv_head_count: 8,
            layer_count: 24,
            key_length: 128,
            value_length: 128,
            key_length_swa: 256,
            value_length_swa: 256,
            ..Default::default()
        };

        assert_eq!(meta.kv_cache_quant_support(GgufKvCacheQuant::Q8_0), Ok(()));
        assert_eq!(
            meta.compatible_default_kv_cache_quant(GgufKvCacheQuant::Q4_0),
            GgufKvCacheQuant::Q4_0
        );
    }

    /// The exact hole from the review: full-attention widths aligned, SWA
    /// widths not. llama.cpp's per-layer check sees the SWA width for SWA
    /// layers, so the quantised cache cannot load and the default must fall
    /// back to f16 — previously the guard returned `Ok(())` and the context
    /// build crashed.
    #[test]
    fn unaligned_swa_widths_fall_back_to_f16() {
        let meta = GgufCompactMeta {
            architecture: "gemma3".to_string(),
            head_count: 32,
            kv_head_count: 8,
            layer_count: 24,
            key_length: 128,
            value_length: 128,
            key_length_swa: 100,
            value_length_swa: 100,
            ..Default::default()
        };

        assert_eq!(
            meta.kv_cache_quant_support(GgufKvCacheQuant::Q8_0),
            Err(KvQuantUnsupported::KeyWidthNotBlockAligned {
                head_width: 100,
                block: 32,
            })
        );
        assert_eq!(
            meta.compatible_default_kv_cache_quant(GgufKvCacheQuant::Q8_0),
            GgufKvCacheQuant::F16
        );
    }

    /// An unaligned SWA *value* width must independently degrade the V side,
    /// even when every key width is aligned.
    #[test]
    fn unaligned_swa_value_width_alone_degrades_to_f16() {
        let meta = GgufCompactMeta {
            architecture: "olmo2".to_string(),
            head_count: 32,
            kv_head_count: 8,
            layer_count: 24,
            key_length: 128,
            value_length: 128,
            key_length_swa: 128,
            value_length_swa: 100,
            ..Default::default()
        };

        assert_eq!(
            meta.kv_cache_quant_support(GgufKvCacheQuant::Q4_0),
            Err(KvQuantUnsupported::ValueWidthNotBlockAligned {
                head_width: 100,
                block: 32,
            })
        );
        assert_eq!(
            meta.compatible_default_kv_cache_quant(GgufKvCacheQuant::Q4_0),
            GgufKvCacheQuant::F16
        );
    }

    /// glm-dsa caches the absorbed-MLA latent (kv_lora_rank = 512), which is
    /// block-aligned, so the guard must use that width — not raw value_length —
    /// and keep the quantised default rather than misfiring to f16.
    #[test]
    fn glm_dsa_uses_cached_latent_width_and_stays_quantized() {
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

        assert_eq!(meta.kv_cache_quant_support(GgufKvCacheQuant::Q4_0), Ok(()));
        assert_eq!(
            meta.compatible_default_kv_cache_quant(GgufKvCacheQuant::Q4_0),
            GgufKvCacheQuant::Q4_0
        );
    }

    /// Mixed quant where only K is quantised and only V's width is misaligned:
    /// K passes, V trips, guard falls back.
    #[test]
    fn reports_value_width_misalignment_independently() {
        let meta = GgufCompactMeta {
            architecture: "custom".to_string(),
            head_count: 32,
            kv_head_count: 8,
            layer_count: 24,
            key_length: 128,
            value_length: 80,
            ..Default::default()
        };

        assert_eq!(
            meta.kv_cache_quant_support(GgufKvCacheQuant::new(
                GgufKvCacheType::Q8_0,
                GgufKvCacheType::Q8_0,
            )),
            Err(KvQuantUnsupported::ValueWidthNotBlockAligned {
                head_width: 80,
                block: 32,
            })
        );
    }
}
