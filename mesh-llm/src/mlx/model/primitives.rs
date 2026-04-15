use anyhow::{bail, Result};
use mlx_rs::array;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{dequantize_device, quantize};
use mlx_rs::Array;
use mlx_rs::{Dtype, StreamOrDevice};

pub struct QuantizedLinear {
    pub(super) weight: Array,
    pub(super) scales: Array,
    pub(super) biases: Array,
    pub(super) bias: Option<Array>,
    pub(super) group_size: i32,
    pub(super) bits: i32,
    pub(super) dense_weight_t: Option<Array>,
}

impl QuantizedLinear {
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let out = if let Some(dense_weight_t) = &self.dense_weight_t {
            mlx_rs::ops::matmul(x, dense_weight_t)?
        } else {
            mlx_rs::ops::quantized_matmul(
                x,
                &self.weight,
                &self.scales,
                &self.biases,
                true,
                self.group_size,
                self.bits,
            )?
        };
        Ok(if let Some(ref bias) = self.bias {
            &out + bias
        } else {
            out
        })
    }
}

pub(super) fn cpu_dense_weight_t(
    weight: &Array,
    scales: &Array,
    biases: &Array,
    group_size: i32,
    bits: i32,
) -> Result<Array> {
    let dense_cpu = dequantize_device(
        weight,
        scales,
        biases,
        group_size,
        bits,
        StreamOrDevice::cpu(),
    )?;
    let dense_cpu = if dense_cpu.dtype() == Dtype::Float32 {
        dense_cpu
    } else if matches!(dense_cpu.dtype(), Dtype::Bfloat16 | Dtype::Float16) {
        dense_cpu.as_dtype(Dtype::Float32)?
    } else {
        bail!(
            "unsupported dense dequantized dtype for CPU fallback: {:?}",
            dense_cpu.dtype()
        );
    };
    let dense = Array::from_slice(dense_cpu.as_slice::<f32>(), dense_cpu.shape());

    Ok(dense.transpose_axes(&[1, 0])?)
}

pub struct RMSNorm {
    pub(super) weight: Array,
    pub(super) eps: f32,
    pub(super) add_unit_offset: bool,
}

impl RMSNorm {
    pub fn forward(&self, x: &Array) -> Result<Array> {
        if self.add_unit_offset {
            let one = array!(1.0f32).as_dtype(self.weight.dtype())?;
            let weight = self.weight.add(&one)?;
            Ok(mlx_rs::fast::rms_norm(x, &weight, self.eps)?)
        } else {
            Ok(mlx_rs::fast::rms_norm(x, &self.weight, self.eps)?)
        }
    }
}

pub(super) fn unit_rms_norm(x: &Array, eps: f32) -> Result<Array> {
    let width = x.shape()[x.shape().len() - 1];
    let weight = mlx_rs::ops::ones::<f32>(&[width])?.as_dtype(x.dtype())?;
    Ok(mlx_rs::fast::rms_norm(x, &weight, eps)?)
}

pub struct LayerNorm {
    eps: f32,
}

impl LayerNorm {
    pub fn forward(&self, x: &Array) -> Result<Array> {
        Ok(mlx_rs::fast::layer_norm(
            x,
            None::<&Array>,
            None::<&Array>,
            self.eps,
        )?)
    }
}

pub enum NormKind {
    Rms(RMSNorm),
    Layer(LayerNorm),
}

impl NormKind {
    pub fn forward(&self, x: &Array) -> Result<Array> {
        match self {
            Self::Rms(norm) => norm.forward(x),
            Self::Layer(norm) => norm.forward(x),
        }
    }
}

impl From<RMSNorm> for NormKind {
    fn from(value: RMSNorm) -> Self {
        Self::Rms(value)
    }
}

pub(super) fn rms_norm_kind(weight: Array, eps: f32, add_unit_offset: bool) -> NormKind {
    NormKind::Rms(RMSNorm {
        weight,
        eps,
        add_unit_offset,
    })
}

pub(super) fn layer_norm_kind(eps: f32) -> NormKind {
    NormKind::Layer(LayerNorm { eps })
}

pub struct QuantizedMultiLinear {
    pub(super) weight: Array,
    pub(super) scales: Array,
    pub(super) biases: Array,
    pub(super) group_size: i32,
    pub(super) bits: i32,
}

impl QuantizedMultiLinear {
    pub(super) fn forward(&self, x: &Array, transpose: bool) -> Result<Array> {
        let num_heads = self.weight.shape()[0];
        let mut outputs = Vec::with_capacity(num_heads as usize);
        for head in 0..num_heads {
            let idx = Array::from_int(head);
            let w = self
                .weight
                .take_axis(&idx, 0)?
                .reshape(&[self.weight.shape()[1], self.weight.shape()[2]])?;
            let s = self
                .scales
                .take_axis(&idx, 0)?
                .reshape(&[self.scales.shape()[1], self.scales.shape()[2]])?;
            let b = self
                .biases
                .take_axis(&idx, 0)?
                .reshape(&[self.biases.shape()[1], self.biases.shape()[2]])?;
            let xh = x.index((
                std::ops::RangeFull,
                head,
                std::ops::RangeFull,
                std::ops::RangeFull,
            ));
            let out = mlx_rs::ops::quantized_matmul(
                &xh,
                &w,
                &s,
                &b,
                transpose,
                self.group_size,
                self.bits,
            )?;
            outputs.push(out.expand_dims(1)?);
        }
        let output_refs: Vec<&Array> = outputs.iter().collect();
        Ok(mlx_rs::ops::concatenate_axis(&output_refs, 1)?)
    }
}

pub(super) fn quantize_stacked_weights(
    dense: &Array,
    group_size: i32,
    bits: i32,
) -> Result<(Array, Array, Array)> {
    let num_heads = dense.shape()[0];
    let mut q_weights = Vec::with_capacity(num_heads as usize);
    let mut q_scales = Vec::with_capacity(num_heads as usize);
    let mut q_biases = Vec::with_capacity(num_heads as usize);
    for head in 0..num_heads {
        let slice = dense
            .index((head, std::ops::RangeFull, std::ops::RangeFull))
            .reshape(&[dense.shape()[1], dense.shape()[2]])?;
        let (w, s, b) = quantize(&slice, group_size, bits)?;
        q_weights.push(w.expand_dims(0)?);
        q_scales.push(s.expand_dims(0)?);
        q_biases.push(b.expand_dims(0)?);
    }
    let q_weight_refs: Vec<&Array> = q_weights.iter().collect();
    let q_scale_refs: Vec<&Array> = q_scales.iter().collect();
    let q_bias_refs: Vec<&Array> = q_biases.iter().collect();
    Ok((
        mlx_rs::ops::concatenate_axis(&q_weight_refs, 0)?,
        mlx_rs::ops::concatenate_axis(&q_scale_refs, 0)?,
        mlx_rs::ops::concatenate_axis(&q_bias_refs, 0)?,
    ))
}
