use super::*;
use mlx_rs::array;

pub(crate) struct KimiMlaAttention {
    pub(super) q_proj: QuantizedLinear,
    pub(super) kv_a_proj_with_mqa: QuantizedLinear,
    pub(super) kv_a_layernorm: RMSNorm,
    pub(super) embed_q: QuantizedMultiLinear,
    pub(super) unembed_out: QuantizedMultiLinear,
    pub(super) o_proj: QuantizedLinear,
    pub(super) num_heads: i32,
    pub(super) q_head_dim: i32,
    pub(super) qk_rope_head_dim: i32,
    pub(super) qk_nope_head_dim: i32,
    pub(super) kv_lora_rank: i32,
    pub(super) v_head_dim: i32,
    pub(super) scale: f32,
}

impl KimiMlaAttention {
    pub(super) fn forward_no_cache(&self, x: &Array) -> Result<Array> {
        let shape = x.shape();
        let (b, l) = (shape[0], shape[1]);

        let q = self
            .q_proj
            .forward(x)?
            .reshape(&[b, l, self.num_heads, self.q_head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let q_nope = q.index((
            std::ops::RangeFull,
            std::ops::RangeFull,
            std::ops::RangeFull,
            ..self.qk_nope_head_dim,
        ));
        let q_pe = q.index((
            std::ops::RangeFull,
            std::ops::RangeFull,
            std::ops::RangeFull,
            self.qk_nope_head_dim..,
        ));

        let compressed_kv = self.kv_a_proj_with_mqa.forward(x)?;
        let kv_latent = compressed_kv.index((
            std::ops::RangeFull,
            std::ops::RangeFull,
            ..self.kv_lora_rank,
        ));
        let k_pe = compressed_kv.index((
            std::ops::RangeFull,
            std::ops::RangeFull,
            self.kv_lora_rank..,
        ));
        let kv_latent = self.kv_a_layernorm.forward(&kv_latent)?.expand_dims(1)?;
        let k_pe = k_pe
            .reshape(&[b, l, 1, self.qk_rope_head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let pe_scores = mlx_rs::ops::matmul(
            &q_pe.multiply(&array!(self.scale))?,
            &k_pe.transpose_axes(&[0, 1, 3, 2])?,
        )?;

        let output = if l == 1 {
            let q_nope = self.embed_q.forward(&q_nope, true)?;
            let scores = mlx_rs::ops::matmul(
                &q_nope.multiply(&array!(self.scale))?,
                &kv_latent.transpose_axes(&[0, 1, 3, 2])?,
            )?
            .add(&pe_scores)?;
            let probs = mlx_rs::ops::softmax_axis(&scores, -1, true)?;
            let output = mlx_rs::ops::matmul(&probs, &kv_latent)?;
            self.unembed_out.forward(&output, true)?
        } else {
            let k = self.embed_q.forward(&kv_latent, false)?;
            let v = self.unembed_out.forward(&kv_latent, true)?;
            let mask = attention_mask(l, l, 0, 0, None)?.context("expected kimi mla mask")?;
            let scores = mlx_rs::ops::matmul(
                &q_nope.multiply(&array!(self.scale))?,
                &k.transpose_axes(&[0, 1, 3, 2])?,
            )?
            .add(&pe_scores)?;
            let fill = array!(scores.dtype().finfo_min()? as f32).as_dtype(scores.dtype())?;
            let scores = mlx_rs::ops::r#where(&mask, &scores, &fill)?;
            let probs = mlx_rs::ops::softmax_axis(&scores, -1, true)?;
            mlx_rs::ops::matmul(&probs, &v)?
        };

        let output = output.transpose_axes(&[0, 2, 1, 3])?.reshape(&[
            b,
            l,
            self.num_heads * self.v_head_dim,
        ])?;
        self.o_proj.forward(&output)
    }
}

pub(crate) struct KimiShortConv {
    pub(super) conv_weight: Array,
    pub(super) kernel_size: i32,
    pub(super) channels: i32,
}

impl KimiShortConv {
    pub(super) fn forward_no_cache(&self, x: &Array) -> Result<Array> {
        let x = pad(
            x,
            &[(0, 0), (self.kernel_size - 1, 0), (0, 0)],
            None::<Array>,
            None::<mlx_rs::ops::PadMode>,
        )?;
        let x = conv1d(
            &x,
            &self.conv_weight,
            None::<i32>,
            None::<i32>,
            None::<i32>,
            Some(self.channels),
        )?;
        Ok(&mlx_rs::ops::sigmoid(&x)? * &x)
    }
}

pub(crate) struct KimiDeltaAttention {
    pub(super) q_proj: QuantizedLinear,
    pub(super) k_proj: QuantizedLinear,
    pub(super) v_proj: QuantizedLinear,
    pub(super) q_conv: KimiShortConv,
    pub(super) k_conv: KimiShortConv,
    pub(super) v_conv: KimiShortConv,
    pub(super) f_a_proj: QuantizedLinear,
    pub(super) f_b_proj: QuantizedLinear,
    pub(super) b_proj: QuantizedLinear,
    pub(super) g_a_proj: QuantizedLinear,
    pub(super) g_b_proj: QuantizedLinear,
    pub(super) a_log: Array,
    pub(super) dt_bias: Array,
    pub(super) o_norm: RMSNorm,
    pub(super) o_proj: QuantizedLinear,
    pub(super) num_heads: i32,
    pub(super) head_dim: i32,
    pub(super) scale: f32,
}

impl KimiDeltaAttention {
    fn gated_delta_update(
        &self,
        q: &Array,
        k: &Array,
        v: &Array,
        a: &Array,
        b: &Array,
    ) -> Result<Array> {
        let bsz = q.shape()[0];
        let seq = q.shape()[1];
        let heads = q.shape()[2];
        let dim = q.shape()[3];
        let mut state = mlx_rs::ops::zeros_dtype(&[bsz, heads, dim, dim], q.dtype())?;
        let beta = mlx_rs::ops::sigmoid(b)?;
        let a = a.add(
            &self
                .dt_bias
                .reshape(&[1, 1, self.num_heads, self.head_dim])?,
        )?;
        let g = mlx_rs::ops::exp(&mlx_rs::ops::negative(
            &mlx_rs::ops::exp(
                &self
                    .a_log
                    .reshape(&[1, 1, self.num_heads, 1])?
                    .as_dtype(Dtype::Float32)?,
            )?
            .multiply(&mlx_rs::nn::softplus(&a)?)?,
        )?)?
        .as_dtype(q.dtype())?;

        let mut ys = Vec::with_capacity(seq as usize);
        for t in 0..seq {
            let q_t = q.index((
                std::ops::RangeFull,
                t,
                std::ops::RangeFull,
                std::ops::RangeFull,
            ));
            let k_t = k.index((
                std::ops::RangeFull,
                t,
                std::ops::RangeFull,
                std::ops::RangeFull,
            ));
            let v_t = v.index((
                std::ops::RangeFull,
                t,
                std::ops::RangeFull,
                std::ops::RangeFull,
            ));
            let beta_t = beta.index((
                std::ops::RangeFull,
                t,
                std::ops::RangeFull,
                std::ops::RangeFull,
            ));
            let g_t = g.index((
                std::ops::RangeFull,
                t,
                std::ops::RangeFull,
                std::ops::RangeFull,
            ));
            state = state.multiply(&g_t.expand_dims(2)?)?;
            let kv_mem = state
                .multiply(&k_t.expand_dims(2)?)?
                .sum_axes(&[-1], false)?;
            let delta = v_t.subtract(&kv_mem)?.multiply(&beta_t)?;
            state = state.add(&k_t.expand_dims(2)?.multiply(&delta.expand_dims(3)?)?)?;
            ys.push(
                state
                    .multiply(&q_t.expand_dims(2)?)?
                    .sum_axes(&[-1], false)?,
            );
        }
        let y_refs: Vec<&Array> = ys.iter().collect();
        Ok(mlx_rs::ops::stack(&y_refs)?.swap_axes(0, 1)?)
    }

    pub(super) fn forward_no_cache(&self, x: &Array) -> Result<Array> {
        let shape = x.shape();
        let (b, l) = (shape[0], shape[1]);
        let q_conv = self.q_conv.forward_no_cache(&self.q_proj.forward(x)?)?;
        let k_conv = self.k_conv.forward_no_cache(&self.k_proj.forward(x)?)?;
        let v_conv = self.v_conv.forward_no_cache(&self.v_proj.forward(x)?)?;

        let mut q = q_conv.reshape(&[b, l, self.num_heads, self.head_dim])?;
        let mut k = k_conv.reshape(&[b, l, self.num_heads, self.head_dim])?;
        let v = v_conv.reshape(&[b, l, self.num_heads, self.head_dim])?;

        q = unit_rms_norm(&q, 1e-6)?.multiply(&array!(self.scale * self.scale))?;
        k = unit_rms_norm(&k, 1e-6)?.multiply(&array!(self.scale))?;

        let a_logits = self
            .f_b_proj
            .forward(&self.f_a_proj.forward(x)?)?
            .reshape(&[b, l, self.num_heads, self.head_dim])?;
        let b_logits = self
            .b_proj
            .forward(x)?
            .reshape(&[b, l, self.num_heads, 1])?;
        let out = self.gated_delta_update(&q, &k, &v, &a_logits, &b_logits)?;
        let gate = self
            .g_b_proj
            .forward(&self.g_a_proj.forward(x)?)?
            .reshape(&[b, l, self.num_heads, self.head_dim])?;
        let out = self
            .o_norm
            .forward(&out)?
            .multiply(&mlx_rs::ops::sigmoid(&gate)?)?
            .reshape(&[b, l, self.num_heads * self.head_dim])?;
        self.o_proj.forward(&out)
    }
}
