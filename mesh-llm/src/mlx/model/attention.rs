use super::*;
use mlx_rs::array;

// ── Attention ──

pub(crate) struct Attention {
    pub(super) q_proj: QuantizedLinear,
    pub(super) k_proj: QuantizedLinear,
    pub(super) v_proj: QuantizedLinear,
    pub(super) o_proj: QuantizedLinear,
    pub(super) q_norm: Option<RMSNorm>,
    pub(super) k_norm: Option<RMSNorm>,
    pub(super) v_norm: Option<RMSNorm>,
    pub(super) num_heads: i32,
    pub(super) num_kv_heads: i32,
    pub(super) head_dim: i32,
    pub(super) scale: f32,
    pub(super) attn_logit_softcapping: Option<f32>,
    pub(super) rope_dim: i32,
    pub(super) rope_theta: f32,
    pub(super) rope_traditional: bool,
    pub(super) window_size: Option<i32>,
    pub(super) kv_shared_source: Option<usize>,
}

impl Attention {
    pub(super) fn apply_qk_norm(
        x: Array,
        norm: Option<&RMSNorm>,
        b: i32,
        l: i32,
        num_heads: i32,
        head_dim: i32,
    ) -> Result<Array> {
        let Some(norm) = norm else {
            return Ok(x.reshape(&[b, l, num_heads, head_dim])?);
        };
        let norm_width = norm.weight.shape()[0];
        if norm_width == num_heads * head_dim {
            return Ok(norm.forward(&x)?.reshape(&[b, l, num_heads, head_dim])?);
        }
        norm.forward(&x.reshape(&[b, l, num_heads, head_dim])?)
    }

    pub(super) fn forward_no_cache(&self, x: &Array) -> Result<Array> {
        let shape = x.shape();
        let (b, l) = (shape[0], shape[1]);

        let q = self.q_proj.forward(x)?;
        let q = Self::apply_qk_norm(q, self.q_norm.as_ref(), b, l, self.num_heads, self.head_dim)?
            .transpose_axes(&[0, 2, 1, 3])?;
        let q = apply_rope(
            &q,
            self.rope_dim,
            self.head_dim,
            self.rope_theta,
            self.rope_traditional,
            0,
        )?;

        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;
        let k = Self::apply_qk_norm(
            k,
            self.k_norm.as_ref(),
            b,
            l,
            self.num_kv_heads,
            self.head_dim,
        )?
        .transpose_axes(&[0, 2, 1, 3])?;
        let v = v.reshape(&[b, l, self.num_kv_heads, self.head_dim])?;
        let v = if let Some(norm) = &self.v_norm {
            norm.forward(&v)?
        } else {
            v
        }
        .transpose_axes(&[0, 2, 1, 3])?;
        let k = apply_rope(
            &k,
            self.rope_dim,
            self.head_dim,
            self.rope_theta,
            self.rope_traditional,
            0,
        )?;

        let mask = if self.window_size.is_some() {
            attention_mask(l, l, 0, 0, self.window_size)?
        } else {
            None
        };
        let attn = if self.attn_logit_softcapping.is_some() || mask.is_some() {
            manual_scaled_dot_product_attention_with_mask(
                &q,
                &k,
                &v,
                self.scale,
                self.attn_logit_softcapping,
                mask.as_ref(),
            )?
        } else {
            let mask = if l > 1 {
                Some(mlx_rs::fast::ScaledDotProductAttentionMask::Causal)
            } else {
                None
            };
            mlx_rs::fast::scaled_dot_product_attention(&q, &k, &v, self.scale, mask)?
        };

        let attn =
            attn.transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[b, l, self.num_heads * self.head_dim])?;
        self.o_proj.forward(&attn)
    }

    pub(super) fn forward(
        &self,
        x: &Array,
        cache: &mut KVCache,
        shared_cache: Option<&KVCache>,
    ) -> Result<Array> {
        let shape = x.shape();
        let (b, l) = (shape[0], shape[1]);

        let q = self.q_proj.forward(x)?;
        let q = Self::apply_qk_norm(q, self.q_norm.as_ref(), b, l, self.num_heads, self.head_dim)?
            .transpose_axes(&[0, 2, 1, 3])?;

        let offset = shared_cache.unwrap_or(&*cache).offset() as i32;
        let q = apply_rope(
            &q,
            self.rope_dim,
            self.head_dim,
            self.rope_theta,
            self.rope_traditional,
            offset,
        )?;
        let (cache_entries, key_start) = if let Some(shared_cache) = shared_cache {
            let (k, v) = shared_cache
                .views()
                .context("Gemma4 shared KV cache was empty")?;
            let key_start =
                shared_cache.key_start_for_attention(l as usize, k.shape()[2] as usize) as i32;
            (CachedKv::Dense { keys: k, values: v }, key_start)
        } else {
            let k = self.k_proj.forward(x)?;
            let v = self.v_proj.forward(x)?;
            let k = Self::apply_qk_norm(
                k,
                self.k_norm.as_ref(),
                b,
                l,
                self.num_kv_heads,
                self.head_dim,
            )?
            .transpose_axes(&[0, 2, 1, 3])?;
            let v = v.reshape(&[b, l, self.num_kv_heads, self.head_dim])?;
            let v = if let Some(norm) = &self.v_norm {
                norm.forward(&v)?
            } else {
                v
            }
            .transpose_axes(&[0, 2, 1, 3])?;
            let k = apply_rope(
                &k,
                self.rope_dim,
                self.head_dim,
                self.rope_theta,
                self.rope_traditional,
                offset,
            )?;
            let entries = cache.update_cached(k, v)?;
            let key_start =
                (offset as usize + l as usize).saturating_sub(entries.key_len() as usize) as i32;
            (entries, key_start)
        };

        // Causal mask for prefill (multi-token). Decode (l=1) needs no mask.
        let mask = if self.window_size.is_some() {
            attention_mask(
                l,
                cache_entries.key_len(),
                key_start,
                offset,
                self.window_size,
            )?
        } else {
            None
        };
        let attn = match cache_entries {
            CachedKv::Dense { keys, values } => {
                if self.attn_logit_softcapping.is_some() || mask.is_some() {
                    manual_scaled_dot_product_attention_with_mask(
                        &q,
                        &keys,
                        &values,
                        self.scale,
                        self.attn_logit_softcapping,
                        mask.as_ref(),
                    )?
                } else {
                    let mask = if l > 1 {
                        Some(mlx_rs::fast::ScaledDotProductAttentionMask::Causal)
                    } else {
                        None
                    };
                    mlx_rs::fast::scaled_dot_product_attention(
                        &q, &keys, &values, self.scale, mask,
                    )?
                }
            }
            CachedKv::Quantized {
                keys,
                values,
                group_size,
                bits,
            } => {
                anyhow::ensure!(
                    self.attn_logit_softcapping.is_none(),
                    "quantized KV cache does not support attention softcapping yet"
                );
                quantized_scaled_dot_product_attention_with_mask(
                    &q,
                    &keys,
                    &values,
                    self.scale,
                    mask.as_ref(),
                    group_size,
                    bits,
                )?
            }
        };

        let attn =
            attn.transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[b, l, self.num_heads * self.head_dim])?;

        self.o_proj.forward(&attn)
    }
}

pub(crate) struct DeepseekV3Attention {
    pub(super) q_proj: Option<QuantizedLinear>,
    pub(super) q_a_proj: Option<QuantizedLinear>,
    pub(super) q_a_layernorm: Option<RMSNorm>,
    pub(super) q_b_proj: Option<QuantizedLinear>,
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
    pub(super) rope_theta: f32,
}

impl DeepseekV3Attention {
    fn build_q(&self, x: &Array) -> Result<Array> {
        let q = if let Some(q_proj) = &self.q_proj {
            q_proj.forward(x)?
        } else {
            self.q_b_proj
                .as_ref()
                .context("missing q_b_proj for DeepSeekV3 attention")?
                .forward(
                    &self
                        .q_a_layernorm
                        .as_ref()
                        .context("missing q_a_layernorm for DeepSeekV3 attention")?
                        .forward(
                            &self
                                .q_a_proj
                                .as_ref()
                                .context("missing q_a_proj for DeepSeekV3 attention")?
                                .forward(x)?,
                        )?,
                )?
        };
        Ok(q)
    }

    fn attention_mask(&self, q_pe: &Array, k_pe: &Array, causal: bool) -> Result<Array> {
        let mut pe_scores = mlx_rs::ops::matmul(
            &q_pe.multiply(&array!(self.scale))?,
            &k_pe.transpose_axes(&[0, 1, 3, 2])?,
        )?;
        if causal {
            let mask = attention_mask(q_pe.shape()[2], k_pe.shape()[2], 0, 0, None)?
                .context("expected causal mask")?;
            let fill = array!(pe_scores.dtype().finfo_min()? as f32).as_dtype(pe_scores.dtype())?;
            pe_scores = mlx_rs::ops::r#where(&mask, &pe_scores, &fill)?;
        }
        Ok(pe_scores)
    }

    fn forward_impl(&self, x: &Array, cache: Option<&mut KVCache>) -> Result<Array> {
        let shape = x.shape();
        let (b, l) = (shape[0], shape[1]);

        let q = self
            .build_q(x)?
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

        let offset = cache
            .as_ref()
            .map(|cache| cache.offset() as i32)
            .unwrap_or(0);
        let q_pe = apply_rope(
            &q_pe,
            self.qk_rope_head_dim,
            self.qk_rope_head_dim,
            self.rope_theta,
            false,
            offset,
        )?;
        let k_pe = apply_rope(
            &k_pe,
            self.qk_rope_head_dim,
            self.qk_rope_head_dim,
            self.rope_theta,
            false,
            offset,
        )?;

        let (kv_latent, k_pe) = if let Some(cache) = cache {
            cache.update(kv_latent, k_pe)?
        } else {
            (kv_latent, k_pe)
        };

        let mask = self.attention_mask(&q_pe, &k_pe, l > 1)?;
        let output = if l == 1 {
            let q_nope = self.embed_q.forward(&q_nope, true)?;
            let output = mlx_rs::fast::scaled_dot_product_attention(
                &q_nope,
                &kv_latent,
                &kv_latent,
                self.scale,
                Some((&mask).into()),
            )?;
            self.unembed_out.forward(&output, true)?
        } else {
            let k = self.embed_q.forward(&kv_latent, false)?;
            let v = self.unembed_out.forward(&kv_latent, true)?;
            mlx_rs::fast::scaled_dot_product_attention(
                &q_nope,
                &k,
                &v,
                self.scale,
                Some((&mask).into()),
            )?
        };

        let output = output.transpose_axes(&[0, 2, 1, 3])?.reshape(&[
            b,
            l,
            self.num_heads * self.v_head_dim,
        ])?;
        self.o_proj.forward(&output)
    }

    pub(super) fn forward_no_cache(&self, x: &Array) -> Result<Array> {
        self.forward_impl(x, None)
    }

    pub(super) fn forward(&self, x: &Array, cache: &mut KVCache) -> Result<Array> {
        self.forward_impl(x, Some(cache))
    }
}

pub(super) fn attention_mask(
    query_len: i32,
    key_len: i32,
    key_start: i32,
    query_start: i32,
    window_size: Option<i32>,
) -> Result<Option<Array>> {
    if query_len == 1 && window_size.is_none() {
        return Ok(None);
    }

    let key_positions = mlx_rs::ops::arange::<_, i32>(key_start, key_start + key_len, 1)?;
    let query_positions = mlx_rs::ops::arange::<_, i32>(query_start, query_start + query_len, 1)?;
    let left = query_positions.expand_dims(1)?;
    let right = key_positions.expand_dims(0)?;
    let mut mask = left.ge(&right)?;
    if let Some(window_size) = window_size {
        let upper_bound = right.add(&array!(window_size))?;
        mask = mask.logical_and(&left.lt(&upper_bound)?)?;
    }
    Ok(Some(mask))
}

fn manual_scaled_dot_product_attention_with_mask(
    q: &Array,
    k: &Array,
    v: &Array,
    scale: f32,
    softcap: Option<f32>,
    mask: Option<&Array>,
) -> Result<Array> {
    let num_heads = q.shape()[1];
    let num_kv_heads = k.shape()[1];
    anyhow::ensure!(
        num_heads % num_kv_heads == 0,
        "cannot align attention heads: q_heads={}, kv_heads={}",
        num_heads,
        num_kv_heads
    );
    let repeats = num_heads / num_kv_heads;
    let batch = q.shape()[0];
    let query_len = q.shape()[2];
    let head_dim = q.shape()[3];

    let mut queries = q.clone();
    if scale != 1.0 {
        queries = queries.multiply(&array!(scale))?;
    }

    let (queries, keys, values) = if repeats > 1 {
        (
            queries.reshape(&[batch, num_kv_heads, repeats, query_len, head_dim])?,
            k.expand_dims(2)?,
            v.expand_dims(2)?,
        )
    } else {
        (queries, k.clone(), v.clone())
    };

    let key_t = if repeats > 1 {
        keys.transpose_axes(&[0, 1, 2, 4, 3])?
    } else {
        keys.transpose_axes(&[0, 1, 3, 2])?
    };
    let mut scores = mlx_rs::ops::matmul(&queries, &key_t)?;
    if let Some(softcap) = softcap {
        scores = scores.divide(&array!(softcap))?;
        scores = mlx_rs::ops::tanh(&scores)?.multiply(&array!(softcap))?;
    }
    if let Some(mask) = mask {
        let fill = array!(scores.dtype().finfo_min()? as f32).as_dtype(scores.dtype())?;
        scores = mlx_rs::ops::r#where(mask, &scores, &fill)?;
    }
    let probs = mlx_rs::ops::softmax_axis(&scores, -1, true)?;
    let mut output = mlx_rs::ops::matmul(&probs, &values)?;
    if repeats > 1 {
        output = output.reshape(&[batch, num_heads, query_len, head_dim])?;
    }
    Ok(output)
}

fn quantized_scaled_dot_product_attention_with_mask(
    q: &Array,
    k: &QuantizedCacheArrays,
    v: &QuantizedCacheArrays,
    scale: f32,
    mask: Option<&Array>,
    group_size: i32,
    bits: i32,
) -> Result<Array> {
    let num_heads = q.shape()[1];
    let num_kv_heads = k.data.shape()[1];
    anyhow::ensure!(
        num_heads % num_kv_heads == 0,
        "cannot align quantized attention heads: q_heads={}, kv_heads={}",
        num_heads,
        num_kv_heads
    );
    let repeats = num_heads / num_kv_heads;
    let batch = q.shape()[0];
    let query_len = q.shape()[2];
    let head_dim = q.shape()[3];

    let mut queries = q.clone();
    if scale != 1.0 {
        queries = queries.multiply(&array!(scale))?;
    }

    let (queries, keys, values) = if repeats > 1 {
        (
            queries.reshape(&[batch, num_kv_heads, repeats, query_len, head_dim])?,
            QuantizedCacheArrays {
                data: k.data.expand_dims(2)?,
                scales: k.scales.expand_dims(2)?,
                biases: k.biases.expand_dims(2)?,
            },
            QuantizedCacheArrays {
                data: v.data.expand_dims(2)?,
                scales: v.scales.expand_dims(2)?,
                biases: v.biases.expand_dims(2)?,
            },
        )
    } else {
        (
            queries,
            QuantizedCacheArrays {
                data: k.data.clone(),
                scales: k.scales.clone(),
                biases: k.biases.clone(),
            },
            QuantizedCacheArrays {
                data: v.data.clone(),
                scales: v.scales.clone(),
                biases: v.biases.clone(),
            },
        )
    };

    let mut scores = mlx_rs::ops::quantized_matmul(
        &queries,
        &keys.data,
        &keys.scales,
        &keys.biases,
        true,
        group_size,
        bits,
    )?;

    if let Some(mask) = mask {
        let fill = array!(scores.dtype().finfo_min()? as f32).as_dtype(scores.dtype())?;
        scores = mlx_rs::ops::r#where(mask, &scores, &fill)?;
    }

    let probs = mlx_rs::ops::softmax_axis(&scores, -1, true)?;
    let mut output = mlx_rs::ops::quantized_matmul(
        &probs,
        &values.data,
        &values.scales,
        &values.biases,
        false,
        group_size,
        bits,
    )?;

    if repeats > 1 {
        output = output.reshape(&[batch, num_heads, query_len, head_dim])?;
    }

    Ok(output)
}

pub(super) fn apply_rope(
    x: &Array,
    rope_dim: i32,
    head_dim: i32,
    rope_theta: f32,
    rope_traditional: bool,
    offset: i32,
) -> Result<Array> {
    if rope_dim == head_dim {
        return Ok(mlx_rs::fast::rope(
            x,
            head_dim,
            rope_traditional,
            Some(rope_theta),
            1.0,
            offset,
            None::<&Array>,
        )?);
    }

    let rotated = x.index((
        std::ops::RangeFull,
        std::ops::RangeFull,
        std::ops::RangeFull,
        ..rope_dim,
    ));
    let rotated = mlx_rs::fast::rope(
        &rotated,
        rope_dim,
        rope_traditional,
        Some(rope_theta),
        1.0,
        offset,
        None::<&Array>,
    )?;
    let tail = x.index((
        std::ops::RangeFull,
        std::ops::RangeFull,
        std::ops::RangeFull,
        rope_dim..,
    ));
    Ok(mlx_rs::ops::concatenate_axis(&[&rotated, &tail], 3)?)
}
