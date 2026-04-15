use super::*;

pub(crate) struct Layer {
    pub(super) attn: AttentionKind,
    pub(super) mlp: MlpKind,
    pub(super) attn_in_norm: Option<NormKind>,
    pub(super) attn_out_norm: Option<NormKind>,
    pub(super) mlp_in_norm: Option<NormKind>,
    pub(super) mlp_out_norm: Option<NormKind>,
    pub(super) per_layer_input: Option<PerLayerInputBlock>,
    pub(super) layer_scalar: Option<Array>,
}

impl Layer {
    pub(super) fn forward_no_cache(
        &self,
        x: &Array,
        per_layer_input: Option<&Array>,
    ) -> Result<Array> {
        let attn_input = if let Some(norm) = &self.attn_in_norm {
            norm.forward(x)?
        } else {
            x.clone()
        };
        let attn = self.attn.forward_no_cache(&attn_input)?;
        let attn = if let Some(norm) = &self.attn_out_norm {
            norm.forward(&attn)?
        } else {
            attn
        };
        let h = &attn + x;
        let mlp_input = if let Some(norm) = &self.mlp_in_norm {
            norm.forward(&h)?
        } else {
            h.clone()
        };
        let mlp = self.mlp.forward(&mlp_input)?;
        let mlp = if let Some(norm) = &self.mlp_out_norm {
            norm.forward(&mlp)?
        } else {
            mlp
        };
        let mut out = &mlp + &h;

        if let (Some(block), Some(per_layer_input)) = (&self.per_layer_input, per_layer_input) {
            let residual = out.clone();
            let mut gated = block.input_gate.forward(&out)?;
            gated = match block.activation {
                Activation::Silu => &mlx_rs::ops::sigmoid(&gated)? * &gated,
                Activation::GeluApproximate => mlx_rs::nn::gelu_approximate(&gated)?,
            };
            gated = gated.multiply(per_layer_input)?;
            gated = block.projection.forward(&gated)?;
            gated = block.post_norm.forward(&gated)?;
            out = &gated + &residual;
        }

        if let Some(layer_scalar) = &self.layer_scalar {
            out = out.multiply(layer_scalar)?;
        }

        Ok(out)
    }

    pub(super) fn forward(
        &self,
        x: &Array,
        per_layer_input: Option<&Array>,
        cache: &mut KVCache,
        shared_cache: Option<&KVCache>,
    ) -> Result<Array> {
        let attn_input = if let Some(norm) = &self.attn_in_norm {
            norm.forward(x)?
        } else {
            x.clone()
        };
        let attn = self.attn.forward(&attn_input, cache, shared_cache)?;
        let attn = if let Some(norm) = &self.attn_out_norm {
            norm.forward(&attn)?
        } else {
            attn
        };
        let h = &attn + x;
        let mlp_input = if let Some(norm) = &self.mlp_in_norm {
            norm.forward(&h)?
        } else {
            h.clone()
        };
        let mlp = self.mlp.forward(&mlp_input)?;
        let mlp = if let Some(norm) = &self.mlp_out_norm {
            norm.forward(&mlp)?
        } else {
            mlp
        };
        let mut out = &mlp + &h;

        if let (Some(block), Some(per_layer_input)) = (&self.per_layer_input, per_layer_input) {
            let residual = out.clone();
            let mut gated = block.input_gate.forward(&out)?;
            gated = match block.activation {
                Activation::Silu => &mlx_rs::ops::sigmoid(&gated)? * &gated,
                Activation::GeluApproximate => mlx_rs::nn::gelu_approximate(&gated)?,
            };
            gated = gated.multiply(per_layer_input)?;
            gated = block.projection.forward(&gated)?;
            gated = block.post_norm.forward(&gated)?;
            out = &gated + &residual;
        }

        if let Some(layer_scalar) = &self.layer_scalar {
            out = out.multiply(layer_scalar)?;
        }

        Ok(out)
    }
}

pub(crate) struct PerLayerInputBlock {
    pub(super) input_gate: QuantizedLinear,
    pub(super) projection: QuantizedLinear,
    pub(super) post_norm: NormKind,
    pub(super) activation: Activation,
}
