use super::*;
use serde_json::Value;

pub(super) struct QuantizedEmbedding {
    pub(super) weight: Array,
    pub(super) scales: Array,
    pub(super) biases: Array,
    pub(super) group_size: i32,
    pub(super) bits: i32,
    pub(super) dense_weight: Option<Array>,
    pub(super) dense_weight_t: Option<Array>,
}

impl QuantizedEmbedding {
    pub(super) fn forward(&self, indices: &Array) -> Result<Array> {
        if let Some(dense_weight) = &self.dense_weight {
            return Ok(dense_weight.take_axis(indices, 0)?);
        }
        let w = self.weight.take_axis(indices, 0)?;
        let s = self.scales.take_axis(indices, 0)?;
        let b = self.biases.take_axis(indices, 0)?;
        Ok(mlx_rs::ops::dequantize(
            &w,
            &s,
            &b,
            self.group_size,
            self.bits,
        )?)
    }

    pub(super) fn as_linear(&self) -> QuantizedLinear {
        QuantizedLinear {
            weight: self.weight.clone(),
            scales: self.scales.clone(),
            biases: self.biases.clone(),
            bias: None,
            group_size: self.group_size,
            bits: self.bits,
            dense_weight_t: self.dense_weight_t.clone(),
        }
    }
}

pub(super) fn quant_params_for(
    config: &Value,
    prefix: &str,
    default_group_size: i32,
    default_bits: i32,
) -> (i32, i32) {
    let override_cfg = config
        .get("quantization")
        .and_then(Value::as_object)
        .and_then(|q| q.get(prefix))
        .cloned()
        .and_then(|value| serde_json::from_value::<QuantOverride>(value).ok());

    (
        override_cfg
            .as_ref()
            .and_then(|cfg| cfg.group_size)
            .unwrap_or(default_group_size),
        override_cfg
            .as_ref()
            .and_then(|cfg| cfg.bits)
            .unwrap_or(default_bits),
    )
}
