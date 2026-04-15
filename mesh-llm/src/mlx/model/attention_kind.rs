use super::*;

pub enum AttentionKind {
    Standard(Attention),
    DeepseekV3(DeepseekV3Attention),
    KimiMla(KimiMlaAttention),
    KimiDelta(KimiDeltaAttention),
    Lfm2ShortConv(Lfm2ShortConv),
}

impl AttentionKind {
    pub(super) fn forward_no_cache(&self, x: &Array) -> Result<Array> {
        match self {
            Self::Standard(attn) => attn.forward_no_cache(x),
            Self::DeepseekV3(attn) => attn.forward_no_cache(x),
            Self::KimiMla(attn) => attn.forward_no_cache(x),
            Self::KimiDelta(attn) => attn.forward_no_cache(x),
            Self::Lfm2ShortConv(conv) => conv.forward_no_cache(x),
        }
    }

    pub(super) fn forward(
        &self,
        x: &Array,
        cache: &mut KVCache,
        shared_cache: Option<&KVCache>,
    ) -> Result<Array> {
        match self {
            Self::Standard(attn) => attn.forward(x, cache, shared_cache),
            Self::DeepseekV3(attn) => attn.forward(x, cache),
            Self::KimiMla(_) | Self::KimiDelta(_) => {
                bail!("Kimi Linear currently requires cacheless generation")
            }
            Self::Lfm2ShortConv(_) => {
                bail!("LFM2 ShortConv currently requires cacheless generation")
            }
        }
    }

    pub(super) fn kv_shared_source(&self) -> Option<usize> {
        match self {
            Self::Standard(attn) => attn.kv_shared_source,
            Self::DeepseekV3(_) => None,
            Self::KimiMla(_) | Self::KimiDelta(_) => None,
            Self::Lfm2ShortConv(_) => None,
        }
    }

    pub(super) fn sliding_window_size(&self) -> Option<usize> {
        match self {
            Self::Standard(attn) => attn.window_size.map(|size| size as usize),
            Self::DeepseekV3(_) => None,
            Self::KimiMla(_) | Self::KimiDelta(_) => None,
            Self::Lfm2ShortConv(_) => None,
        }
    }
}
