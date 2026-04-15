use super::*;

const KV_CACHE_STEP: usize = 256;

enum KVCacheMode {
    Standard,
    Rotating {
        max_size: usize,
        keep: usize,
    },
    Quantized {
        group_size: i32,
        bits: i32,
        min_dense_tokens: usize,
    },
}

pub(super) struct QuantizedCacheArrays {
    pub(super) data: Array,
    pub(super) scales: Array,
    pub(super) biases: Array,
}

impl QuantizedCacheArrays {
    pub(super) fn arrays(&self) -> [&Array; 3] {
        [&self.data, &self.scales, &self.biases]
    }

    pub(super) fn prefix(&self, end: i32) -> Self {
        Self {
            data: self.data.index((
                std::ops::RangeFull,
                std::ops::RangeFull,
                ..end,
                std::ops::RangeFull,
            )),
            scales: self.scales.index((
                std::ops::RangeFull,
                std::ops::RangeFull,
                ..end,
                std::ops::RangeFull,
            )),
            biases: self.biases.index((
                std::ops::RangeFull,
                std::ops::RangeFull,
                ..end,
                std::ops::RangeFull,
            )),
        }
    }

    pub(super) fn trim_to(&self, end: i32) -> Result<Self> {
        Ok(Self {
            data: materialize_cache_prefix(&self.data, end)?,
            scales: materialize_cache_prefix(&self.scales, end)?,
            biases: materialize_cache_prefix(&self.biases, end)?,
        })
    }

    pub(super) fn expand(&self, extra_steps: i32) -> Result<Self> {
        let base_shape = self.data.shape();
        let scale_shape = self.scales.shape();
        let bias_shape = self.biases.shape();
        let extra_data = mlx_rs::ops::zeros_dtype(
            &[base_shape[0], base_shape[1], extra_steps, base_shape[3]],
            self.data.dtype(),
        )?;
        let extra_scales = mlx_rs::ops::zeros_dtype(
            &[scale_shape[0], scale_shape[1], extra_steps, scale_shape[3]],
            self.scales.dtype(),
        )?;
        let extra_biases = mlx_rs::ops::zeros_dtype(
            &[bias_shape[0], bias_shape[1], extra_steps, bias_shape[3]],
            self.biases.dtype(),
        )?;
        Ok(Self {
            data: mlx_rs::ops::concatenate_axis(&[&self.data, &extra_data], 2)?,
            scales: mlx_rs::ops::concatenate_axis(&[&self.scales, &extra_scales], 2)?,
            biases: mlx_rs::ops::concatenate_axis(&[&self.biases, &extra_biases], 2)?,
        })
    }
}

fn materialize_cache_prefix(array: &Array, end: i32) -> Result<Array> {
    use std::ops::RangeFull;

    let end = end.max(0);
    let shape = array.shape();
    let mut owned = mlx_rs::ops::zeros_dtype(&[shape[0], shape[1], end, shape[3]], array.dtype())?;
    if end > 0 {
        let prefix = array.index((RangeFull, RangeFull, ..end, RangeFull));
        owned.try_index_mut((RangeFull, RangeFull, ..end, RangeFull), &prefix)?;
    }
    Ok(owned)
}

pub(super) enum CachedKv {
    Dense {
        keys: Array,
        values: Array,
    },
    Quantized {
        keys: QuantizedCacheArrays,
        values: QuantizedCacheArrays,
        group_size: i32,
        bits: i32,
    },
}

impl CachedKv {
    pub(super) fn key_len(&self) -> i32 {
        match self {
            CachedKv::Dense { keys, .. } => keys.shape()[2],
            CachedKv::Quantized { keys, .. } => keys.data.shape()[2],
        }
    }
}

pub struct KVCache {
    pub(super) keys: Option<Array>,
    pub(super) values: Option<Array>,
    pub(super) qkeys: Option<QuantizedCacheArrays>,
    pub(super) qvalues: Option<QuantizedCacheArrays>,
    start_offset: usize,
    offset: usize,
    idx: usize,
    mode: KVCacheMode,
}

impl KVCache {
    pub fn new() -> Self {
        KVCache {
            keys: None,
            values: None,
            qkeys: None,
            qvalues: None,
            start_offset: 0,
            offset: 0,
            idx: 0,
            mode: KVCacheMode::Standard,
        }
    }

    pub fn new_rotating(max_size: usize, keep: usize) -> Self {
        KVCache {
            keys: None,
            values: None,
            qkeys: None,
            qvalues: None,
            start_offset: 0,
            offset: 0,
            idx: 0,
            mode: KVCacheMode::Rotating { max_size, keep },
        }
    }

    pub fn new_quantized(group_size: i32, bits: i32, min_dense_tokens: usize) -> Self {
        KVCache {
            keys: None,
            values: None,
            qkeys: None,
            qvalues: None,
            start_offset: 0,
            offset: 0,
            idx: 0,
            mode: KVCacheMode::Quantized {
                group_size,
                bits,
                min_dense_tokens,
            },
        }
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    fn current_len(&self) -> usize {
        self.offset.saturating_sub(self.start_offset)
    }

    pub(super) fn retained_start(&self) -> usize {
        self.start_offset
    }

    pub fn can_trim_to(&self, n: usize) -> bool {
        n <= self.offset && n >= self.retained_start()
    }

    /// Return references to cached arrays (for eval/materialization).
    pub fn arrays(&self) -> Vec<&Array> {
        let mut out = Vec::new();
        match self.mode {
            KVCacheMode::Quantized { .. } => {
                if let Some(ref k) = self.qkeys {
                    out.extend(k.arrays());
                }
                if let Some(ref v) = self.qvalues {
                    out.extend(v.arrays());
                }
            }
            _ => {
                if let Some(ref k) = self.keys {
                    out.push(k);
                }
                if let Some(ref v) = self.values {
                    out.push(v);
                }
            }
        }
        out
    }

    pub fn views(&self) -> Option<(Array, Array)> {
        if self.current_len() == 0 {
            return None;
        }
        let keys = self.temporal_order(self.keys.as_ref()?).ok()?;
        let values = self.temporal_order(self.values.as_ref()?).ok()?;
        Some((keys, values))
    }

    fn temporal_order(&self, array: &Array) -> Result<Array> {
        use std::ops::RangeFull;

        match self.mode {
            KVCacheMode::Standard => {
                let end_i = self.offset as i32;
                Ok(array.index((RangeFull, RangeFull, ..end_i, RangeFull)))
            }
            KVCacheMode::Quantized { .. } => {
                let end_i = self.offset as i32;
                Ok(array.index((RangeFull, RangeFull, ..end_i, RangeFull)))
            }
            KVCacheMode::Rotating { keep, .. } => {
                let len = array.shape()[2] as usize;
                if self.idx == len {
                    return Ok(array.clone());
                }
                if self.idx < self.current_len() {
                    let mut parts = Vec::new();
                    if keep > 0 {
                        parts.push(array.index((RangeFull, RangeFull, ..(keep as i32), RangeFull)));
                    }
                    parts.push(array.index((RangeFull, RangeFull, self.idx as i32.., RangeFull)));
                    if self.idx > keep {
                        parts.push(array.index((
                            RangeFull,
                            RangeFull,
                            keep as i32..self.idx as i32,
                            RangeFull,
                        )));
                    }
                    let refs: Vec<&Array> = parts.iter().collect();
                    Ok(mlx_rs::ops::concatenate_axis(&refs, 2)?)
                } else {
                    Ok(array.index((RangeFull, RangeFull, ..(self.idx as i32), RangeFull)))
                }
            }
        }
    }

    pub(super) fn key_start_for_attention(&self, query_len: usize, key_len: usize) -> usize {
        (self.offset + query_len).saturating_sub(key_len)
    }

    pub fn update(&mut self, k: Array, v: Array) -> Result<(Array, Array)> {
        match self.update_cached(k, v)? {
            CachedKv::Dense { keys, values } => Ok((keys, values)),
            CachedKv::Quantized { .. } => bail!("quantized KV cache does not expose dense views"),
        }
    }

    pub(super) fn update_cached(&mut self, k: Array, v: Array) -> Result<CachedKv> {
        match self.mode {
            KVCacheMode::Standard => self
                .update_standard(k, v)
                .map(|(keys, values)| CachedKv::Dense { keys, values }),
            KVCacheMode::Rotating { max_size, keep } => self.update_rotating(k, v, max_size, keep),
            KVCacheMode::Quantized {
                group_size,
                bits,
                min_dense_tokens,
            } => self.update_quantized(k, v, group_size, bits, min_dense_tokens),
        }
    }

    fn update_standard(&mut self, k: Array, v: Array) -> Result<(Array, Array)> {
        use std::ops::RangeFull;

        let seq_len = k.shape()[2] as usize;
        let prev = self.offset;

        if self.keys.is_none() || (prev + seq_len) > self.keys.as_ref().unwrap().shape()[2] as usize
        {
            // Grow: pre-allocate in steps, matching the incoming dtype
            let [b, n_kv_heads, _, k_head_dim] = k.shape()[..4] else {
                bail!("unexpected k shape");
            };
            let v_head_dim = v.shape()[3];
            let k_dtype = k.dtype();
            let v_dtype = v.dtype();

            let n_steps = ((KV_CACHE_STEP + seq_len - 1) / KV_CACHE_STEP) * KV_CACHE_STEP;
            let k_shape = &[b, n_kv_heads, n_steps as i32, k_head_dim];
            let v_shape = &[b, n_kv_heads, n_steps as i32, v_head_dim];

            let new_k = mlx_rs::ops::zeros_dtype(k_shape, k_dtype)?;
            let new_v = mlx_rs::ops::zeros_dtype(v_shape, v_dtype)?;

            if let (Some(ref mut old_k), Some(ref mut old_v)) = (&mut self.keys, &mut self.values) {
                if prev % KV_CACHE_STEP != 0 {
                    *old_k = old_k.index((RangeFull, RangeFull, ..(prev as i32), RangeFull));
                    *old_v = old_v.index((RangeFull, RangeFull, ..(prev as i32), RangeFull));
                }
                self.keys = Some(mlx_rs::ops::concatenate_axis(
                    &[old_k as &Array, &new_k],
                    2,
                )?);
                self.values = Some(mlx_rs::ops::concatenate_axis(
                    &[old_v as &Array, &new_v],
                    2,
                )?);
            } else {
                self.keys = Some(new_k);
                self.values = Some(new_v);
            }
        }

        self.offset = prev + seq_len;
        self.start_offset = 0;
        let prev_i = prev as i32;
        let end_i = self.offset as i32;

        // Slice-assign into pre-allocated buffer (no copy of existing data)
        self.keys
            .as_mut()
            .unwrap()
            .try_index_mut((RangeFull, RangeFull, prev_i..end_i, RangeFull), &k)?;
        self.values
            .as_mut()
            .unwrap()
            .try_index_mut((RangeFull, RangeFull, prev_i..end_i, RangeFull), &v)?;

        // Return views up to current offset
        let k_out = self
            .keys
            .as_ref()
            .unwrap()
            .index((RangeFull, RangeFull, ..end_i, RangeFull));
        let v_out = self
            .values
            .as_ref()
            .unwrap()
            .index((RangeFull, RangeFull, ..end_i, RangeFull));

        Ok((k_out, v_out))
    }

    fn update_rotating(
        &mut self,
        k: Array,
        v: Array,
        max_size: usize,
        keep: usize,
    ) -> Result<CachedKv> {
        let seq_len = k.shape()[2] as usize;
        if seq_len == 1 {
            return self.update_rotating_in_place(k, v, max_size, keep);
        }
        self.update_rotating_concat(k, v, max_size, keep)
    }

    fn update_rotating_concat(
        &mut self,
        k: Array,
        v: Array,
        max_size: usize,
        keep: usize,
    ) -> Result<CachedKv> {
        let seq_len = k.shape()[2] as usize;
        if self.keys.is_none() {
            self.keys = Some(k);
            self.values = Some(v);
        } else {
            let ordered_k = self.temporal_order(self.keys.as_ref().unwrap())?;
            let ordered_v = self.temporal_order(self.values.as_ref().unwrap())?;
            self.idx = ordered_k.shape()[2] as usize;
            let current_len = ordered_k.shape()[2] as usize;
            let trim_size = (current_len + seq_len).saturating_sub(max_size);
            self.keys = Some(trim_rotating_cache(&ordered_k, trim_size, keep, Some(&k))?);
            self.values = Some(trim_rotating_cache(&ordered_v, trim_size, keep, Some(&v))?);
            self.start_offset += trim_size;
        }

        self.offset += seq_len;
        self.idx = self.keys.as_ref().unwrap().shape()[2] as usize;
        let (keys, values) = self
            .views()
            .context("rotating KV cache was empty after concat update")?;
        Ok(CachedKv::Dense { keys, values })
    }

    fn update_rotating_in_place(
        &mut self,
        k: Array,
        v: Array,
        max_size: usize,
        keep: usize,
    ) -> Result<CachedKv> {
        use std::ops::RangeFull;

        let seq_len = k.shape()[2] as usize;
        debug_assert_eq!(seq_len, 1);
        let prev = self.offset;

        let current_capacity = self
            .keys
            .as_ref()
            .map(|keys| keys.shape()[2] as usize)
            .unwrap_or(0);
        if self.keys.is_none() || (prev >= current_capacity && current_capacity < max_size) {
            let [b, n_kv_heads, _, k_head_dim] = k.shape()[..4] else {
                bail!("unexpected k shape");
            };
            let v_head_dim = v.shape()[3];
            let new_size = KV_CACHE_STEP
                .min(max_size.saturating_sub(prev))
                .max(seq_len);
            let new_k =
                mlx_rs::ops::zeros_dtype(&[b, n_kv_heads, new_size as i32, k_head_dim], k.dtype())?;
            let new_v =
                mlx_rs::ops::zeros_dtype(&[b, n_kv_heads, new_size as i32, v_head_dim], v.dtype())?;
            if let (Some(old_k), Some(old_v)) = (&self.keys, &self.values) {
                self.keys = Some(mlx_rs::ops::concatenate_axis(&[old_k, &new_k], 2)?);
                self.values = Some(mlx_rs::ops::concatenate_axis(&[old_v, &new_v], 2)?);
            } else {
                self.keys = Some(new_k);
                self.values = Some(new_v);
            }
            self.idx = prev;
        }

        let current_capacity = self.keys.as_ref().unwrap().shape()[2] as usize;
        let trim_size = current_capacity.saturating_sub(max_size);
        if trim_size > 0 {
            let ordered_k = self.temporal_order(self.keys.as_ref().unwrap())?;
            let ordered_v = self.temporal_order(self.values.as_ref().unwrap())?;
            self.keys = Some(trim_rotating_cache(&ordered_k, trim_size, keep, None)?);
            self.values = Some(trim_rotating_cache(&ordered_v, trim_size, keep, None)?);
            self.idx = max_size;
            self.start_offset += trim_size;
        }

        let evicted = usize::from(self.current_len() == max_size);
        if self.idx == max_size {
            self.idx = keep;
        }

        let start = self.idx as i32;
        let end = (self.idx + seq_len) as i32;
        self.keys
            .as_mut()
            .unwrap()
            .try_index_mut((RangeFull, RangeFull, start..end, RangeFull), &k)?;
        self.values
            .as_mut()
            .unwrap()
            .try_index_mut((RangeFull, RangeFull, start..end, RangeFull), &v)?;

        self.offset += seq_len;
        self.start_offset += evicted;
        self.idx += seq_len;
        let (keys, values) = self
            .views()
            .context("rotating KV cache was empty after in-place update")?;
        Ok(CachedKv::Dense { keys, values })
    }

    fn quantize_dense_prefix(
        &mut self,
        group_size: i32,
        bits: i32,
        dense_len: usize,
    ) -> Result<()> {
        if dense_len == 0 {
            return Ok(());
        }

        let dense_end = dense_len as i32;
        let keys = self
            .keys
            .as_ref()
            .context("missing dense keys while migrating to quantized KV")?
            .index((
                std::ops::RangeFull,
                std::ops::RangeFull,
                ..dense_end,
                std::ops::RangeFull,
            ));
        let values = self
            .values
            .as_ref()
            .context("missing dense values while migrating to quantized KV")?
            .index((
                std::ops::RangeFull,
                std::ops::RangeFull,
                ..dense_end,
                std::ops::RangeFull,
            ));
        let [b, n_kv_heads, _, k_head_dim] = keys.shape()[..4] else {
            bail!("unexpected dense key shape while migrating to quantized KV");
        };
        let v_head_dim = values.shape()[3];
        let el_per_int = 32 / bits;

        let (kq, ks, kb) = mlx_rs::ops::quantize(&keys, group_size, bits)?;
        let (vq, vs, vb) = mlx_rs::ops::quantize(&values, group_size, bits)?;
        self.qkeys = Some(QuantizedCacheArrays {
            data: kq,
            scales: ks,
            biases: kb,
        });
        self.qvalues = Some(QuantizedCacheArrays {
            data: vq,
            scales: vs,
            biases: vb,
        });

        let qkeys = self.qkeys.as_mut().unwrap();
        if qkeys.data.shape()[2] != dense_end {
            qkeys.data =
                qkeys
                    .data
                    .reshape(&[b, n_kv_heads, dense_end, k_head_dim / el_per_int])?;
            qkeys.scales =
                qkeys
                    .scales
                    .reshape(&[b, n_kv_heads, dense_end, k_head_dim / group_size])?;
            qkeys.biases =
                qkeys
                    .biases
                    .reshape(&[b, n_kv_heads, dense_end, k_head_dim / group_size])?;
        }

        let qvalues = self.qvalues.as_mut().unwrap();
        if qvalues.data.shape()[2] != dense_end {
            qvalues.data =
                qvalues
                    .data
                    .reshape(&[b, n_kv_heads, dense_end, v_head_dim / el_per_int])?;
            qvalues.scales =
                qvalues
                    .scales
                    .reshape(&[b, n_kv_heads, dense_end, v_head_dim / group_size])?;
            qvalues.biases =
                qvalues
                    .biases
                    .reshape(&[b, n_kv_heads, dense_end, v_head_dim / group_size])?;
        }

        self.keys = None;
        self.values = None;
        Ok(())
    }

    fn update_quantized(
        &mut self,
        k: Array,
        v: Array,
        group_size: i32,
        bits: i32,
        min_dense_tokens: usize,
    ) -> Result<CachedKv> {
        let seq_len = k.shape()[2] as usize;
        let prev = self.offset;

        if self.qkeys.is_none() && (prev + seq_len) <= min_dense_tokens {
            return self
                .update_standard(k, v)
                .map(|(keys, values)| CachedKv::Dense { keys, values });
        }

        if self.qkeys.is_none() {
            self.quantize_dense_prefix(group_size, bits, prev)?;
        }

        if self.qkeys.is_none()
            || (prev + seq_len) > self.qkeys.as_ref().unwrap().data.shape()[2] as usize
        {
            let [b, n_kv_heads, _, k_head_dim] = k.shape()[..4] else {
                bail!("unexpected quantized k shape");
            };
            let v_head_dim = v.shape()[3];
            let el_per_int = 32 / bits;
            let n_steps = ((KV_CACHE_STEP + seq_len - 1) / KV_CACHE_STEP) * KV_CACHE_STEP;

            let init_quant = |head_dim: i32, dtype: Dtype| -> Result<QuantizedCacheArrays> {
                Ok(QuantizedCacheArrays {
                    data: mlx_rs::ops::zeros_dtype(
                        &[b, n_kv_heads, n_steps as i32, head_dim / el_per_int],
                        Dtype::Uint32,
                    )?,
                    scales: mlx_rs::ops::zeros_dtype(
                        &[b, n_kv_heads, n_steps as i32, head_dim / group_size],
                        dtype,
                    )?,
                    biases: mlx_rs::ops::zeros_dtype(
                        &[b, n_kv_heads, n_steps as i32, head_dim / group_size],
                        dtype,
                    )?,
                })
            };

            match (&self.qkeys, &self.qvalues) {
                (Some(existing_k), Some(existing_v)) => {
                    let (mut existing_k, mut existing_v) = (existing_k, existing_v);
                    if prev % KV_CACHE_STEP != 0 {
                        let end = prev as i32;
                        self.qkeys = Some(existing_k.trim_to(end)?);
                        self.qvalues = Some(existing_v.trim_to(end)?);
                        existing_k = self.qkeys.as_ref().unwrap();
                        existing_v = self.qvalues.as_ref().unwrap();
                    }
                    self.qkeys = Some(existing_k.expand(n_steps as i32)?);
                    self.qvalues = Some(existing_v.expand(n_steps as i32)?);
                }
                _ => {
                    self.qkeys = Some(init_quant(k_head_dim, k.dtype())?);
                    self.qvalues = Some(init_quant(v_head_dim, v.dtype())?);
                }
            }
        }

        self.offset = prev + seq_len;
        self.start_offset = 0;
        let prev_i = prev as i32;
        let end_i = self.offset as i32;

        let (kq, ks, kb) = mlx_rs::ops::quantize(&k, group_size, bits)?;
        let (vq, vs, vb) = mlx_rs::ops::quantize(&v, group_size, bits)?;
        let qkeys = self.qkeys.as_mut().unwrap();
        let qvalues = self.qvalues.as_mut().unwrap();
        qkeys.data.try_index_mut(
            (
                std::ops::RangeFull,
                std::ops::RangeFull,
                prev_i..end_i,
                std::ops::RangeFull,
            ),
            &kq,
        )?;
        qkeys.scales.try_index_mut(
            (
                std::ops::RangeFull,
                std::ops::RangeFull,
                prev_i..end_i,
                std::ops::RangeFull,
            ),
            &ks,
        )?;
        qkeys.biases.try_index_mut(
            (
                std::ops::RangeFull,
                std::ops::RangeFull,
                prev_i..end_i,
                std::ops::RangeFull,
            ),
            &kb,
        )?;
        qvalues.data.try_index_mut(
            (
                std::ops::RangeFull,
                std::ops::RangeFull,
                prev_i..end_i,
                std::ops::RangeFull,
            ),
            &vq,
        )?;
        qvalues.scales.try_index_mut(
            (
                std::ops::RangeFull,
                std::ops::RangeFull,
                prev_i..end_i,
                std::ops::RangeFull,
            ),
            &vs,
        )?;
        qvalues.biases.try_index_mut(
            (
                std::ops::RangeFull,
                std::ops::RangeFull,
                prev_i..end_i,
                std::ops::RangeFull,
            ),
            &vb,
        )?;

        Ok(CachedKv::Quantized {
            keys: qkeys.prefix(end_i),
            values: qvalues.prefix(end_i),
            group_size,
            bits,
        })
    }

    /// Rewind the cache to `n` tokens if the requested prefix is still retained.
    pub fn trim_to(&mut self, n: usize) -> Result<bool> {
        if !self.can_trim_to(n) {
            return Ok(false);
        }
        let retained_len = n.saturating_sub(self.start_offset) as i32;
        match self.mode {
            KVCacheMode::Standard => {
                if n != self.offset {
                    if let Some(keys) = &self.keys {
                        self.keys = Some(materialize_cache_prefix(keys, retained_len)?);
                    }
                    if let Some(values) = &self.values {
                        self.values = Some(materialize_cache_prefix(values, retained_len)?);
                    }
                }
            }
            KVCacheMode::Quantized { .. } => {
                if n != self.offset {
                    if let Some(keys) = &self.qkeys {
                        self.qkeys = Some(keys.trim_to(retained_len)?);
                    }
                    if let Some(values) = &self.qvalues {
                        self.qvalues = Some(values.trim_to(retained_len)?);
                    }
                }
            }
            KVCacheMode::Rotating { .. } => {}
        }
        if matches!(self.mode, KVCacheMode::Rotating { .. }) && n != self.offset {
            if let (Some(keys), Some(values)) = (&self.keys, &self.values) {
                self.keys = Some(self.temporal_order(keys)?);
                self.values = Some(self.temporal_order(values)?);
            }
        }
        self.offset = n;
        if matches!(self.mode, KVCacheMode::Rotating { .. }) {
            self.idx = self.current_len();
        }
        Ok(true)
    }
}

pub(super) fn trim_rotating_cache(
    array: &Array,
    trim_size: usize,
    keep: usize,
    append: Option<&Array>,
) -> Result<Array> {
    use std::ops::RangeFull;

    let mut parts = Vec::new();
    if trim_size > 0 {
        if keep > 0 {
            parts.push(array.index((RangeFull, RangeFull, ..(keep as i32), RangeFull)));
        }
        parts.push(array.index((RangeFull, RangeFull, (trim_size + keep) as i32.., RangeFull)));
    } else {
        parts.push(array.clone());
    }
    if let Some(append) = append {
        parts.push(append.clone());
    }
    let refs: Vec<&Array> = parts.iter().collect();
    Ok(mlx_rs::ops::concatenate_axis(&refs, 2)?)
}
