use super::*;
use mlx_rs::array;

fn expert_slice_2d(array: &Array, expert: i32) -> Result<Array> {
    Ok(array
        .take_axis(&Array::from_int(expert), 0)?
        .reshape(&[array.shape()[1], array.shape()[2]])?)
}

pub(crate) struct QuantizedSwitchLinear {
    pub(super) weight: Array,
    pub(super) scales: Array,
    pub(super) biases: Array,
    pub(super) bias: Option<Array>,
    pub(super) group_size: i32,
    pub(super) bits: i32,
}

impl QuantizedSwitchLinear {
    pub(super) fn forward_single(&self, x: &Array, expert: i32) -> Result<Array> {
        let out = mlx_rs::ops::quantized_matmul(
            x,
            &expert_slice_2d(&self.weight, expert)?,
            &expert_slice_2d(&self.scales, expert)?,
            &expert_slice_2d(&self.biases, expert)?,
            true,
            self.group_size,
            self.bits,
        )?;
        Ok(if let Some(bias) = &self.bias {
            let bias = bias
                .take_axis(&Array::from_int(expert), 0)?
                .reshape(&[1, bias.shape()[1]])?;
            out.add(&bias)?
        } else {
            out
        })
    }
}

pub(crate) struct MLP {
    pub(super) gate_up_proj: Option<QuantizedLinear>,
    pub(super) gate_proj: Option<QuantizedLinear>,
    pub(super) up_proj: Option<QuantizedLinear>,
    pub(super) down_proj: QuantizedLinear,
    pub(super) activation: Activation,
}

#[derive(Clone, Copy)]
pub(crate) enum Activation {
    Silu,
    GeluApproximate,
}

impl MLP {
    pub(super) fn forward(&self, x: &Array) -> Result<Array> {
        let (gate, up) = if let Some(gate_up_proj) = &self.gate_up_proj {
            let gate_up = gate_up_proj.forward(x)?;
            let hidden = gate_up.shape()[gate_up.shape().len() - 1] / 2;
            let gate = gate_up.index((std::ops::RangeFull, std::ops::RangeFull, 0..hidden));
            let up = gate_up.index((
                std::ops::RangeFull,
                std::ops::RangeFull,
                hidden..(hidden * 2),
            ));
            (gate, up)
        } else {
            (
                self.gate_proj
                    .as_ref()
                    .context("missing gate_proj for unfused MLP")?
                    .forward(x)?,
                self.up_proj
                    .as_ref()
                    .context("missing up_proj for unfused MLP")?
                    .forward(x)?,
            )
        };
        let gate = match self.activation {
            Activation::Silu => &mlx_rs::ops::sigmoid(&gate)? * &gate,
            Activation::GeluApproximate => mlx_rs::nn::gelu_approximate(&gate)?,
        };
        self.down_proj.forward(&(&gate * &up))
    }
}

pub(crate) struct DeepseekV3MoE {
    pub(super) switch_gate_proj: QuantizedSwitchLinear,
    pub(super) switch_up_proj: QuantizedSwitchLinear,
    pub(super) switch_down_proj: QuantizedSwitchLinear,
    pub(super) gate_weight: Array,
    pub(super) gate_bias: Array,
    pub(super) top_k: i32,
    pub(super) n_group: i32,
    pub(super) topk_group: i32,
    pub(super) routed_scaling_factor: f32,
    pub(super) norm_topk_prob: bool,
    pub(super) shared_experts: Option<MLP>,
}

impl DeepseekV3MoE {
    fn gate(&self, x: &Array) -> Result<(Array, Array)> {
        let mut scores = mlx_rs::ops::matmul(x, &self.gate_weight.transpose_axes(&[1, 0])?)?;
        scores = mlx_rs::ops::sigmoid(&scores.as_dtype(Dtype::Float32)?)?;
        let orig_scores = scores.clone();
        scores = scores.add(&self.gate_bias)?;

        if self.n_group > 1 {
            let experts_per_group = scores.shape()[scores.shape().len() - 1] / self.n_group;
            let scores_grouped = scores.reshape(&[-1, self.n_group, experts_per_group])?;
            let top2 = mlx_rs::ops::indexing::topk_axis(&scores_grouped, 2, -1)?;
            let group_scores = top2.sum_axes(&[-1], true)?;
            let k = self.n_group - self.topk_group;
            let group_idx = mlx_rs::ops::argpartition_axis(&group_scores, k - 1, -2)?.index((
                std::ops::RangeFull,
                ..k,
                std::ops::RangeFull,
            ));
            let scores_grouped = mlx_rs::ops::indexing::put_along_axis(
                &scores_grouped,
                &group_idx,
                &array!(0.0f32),
                -2,
            )?;
            scores = scores_grouped.reshape(&[-1, self.gate_weight.shape()[0]])?;
        }

        let inds = mlx_rs::ops::argpartition_axis(
            &scores.multiply(&array!(-1.0f32))?,
            self.top_k - 1,
            -1,
        )?
        .index((std::ops::RangeFull, ..self.top_k));
        let mut probs = mlx_rs::ops::indexing::take_along_axis(&orig_scores, &inds, -1)?
            .as_dtype(Dtype::Float32)?;
        if self.top_k > 1 && self.norm_topk_prob {
            probs = probs.divide(&probs.sum_axes(&[-1], true)?)?;
        }
        probs = probs.multiply(&array!(self.routed_scaling_factor))?;
        Ok((inds, probs))
    }

    fn switch_forward_single(&self, x: &Array, expert: i32) -> Result<Array> {
        let x_up = self.switch_up_proj.forward_single(x, expert)?;
        let x_gate = self.switch_gate_proj.forward_single(x, expert)?;
        let activated = &mlx_rs::ops::sigmoid(&x_gate)? * &x_gate;
        self.switch_down_proj
            .forward_single(&activated.multiply(&x_up)?, expert)
    }

    pub(super) fn forward(&self, x: &Array) -> Result<Array> {
        let b = x.shape()[0];
        let l = x.shape()[1];
        let hidden = x.shape()[2];
        let flat = x.reshape(&[b * l, hidden])?;
        let (inds, scores) = self.gate(&flat)?;
        mlx_rs::transforms::eval([&inds, &scores])?;
        let inds_slice = inds.as_slice::<u32>();
        let scores_slice = scores.as_slice::<f32>();
        let mut outputs = Vec::with_capacity((b * l) as usize);
        for token_idx in 0..(b * l) {
            let x_tok = flat.index((token_idx..token_idx + 1, std::ops::RangeFull));
            let mut token_out: Option<Array> = None;
            for expert_slot in 0..self.top_k {
                let offset = (token_idx * self.top_k + expert_slot) as usize;
                let expert = inds_slice[offset] as i32;
                let score = scores_slice[offset];
                let routed = self
                    .switch_forward_single(&x_tok, expert)?
                    .multiply(&array!(score))?;
                token_out = Some(match token_out {
                    Some(acc) => acc.add(&routed)?,
                    None => routed,
                });
            }
            let token_out = if let Some(shared) = &self.shared_experts {
                token_out.unwrap().add(&shared.forward(&x_tok)?)?
            } else {
                token_out.unwrap()
            };
            outputs.push(token_out);
        }
        let output_refs: Vec<&Array> = outputs.iter().collect();
        Ok(mlx_rs::ops::concatenate_axis(&output_refs, 0)?.reshape(&[b, l, hidden])?)
    }
}

pub(crate) struct GptOssMoE {
    pub(super) switch_gate_proj: QuantizedSwitchLinear,
    pub(super) switch_up_proj: QuantizedSwitchLinear,
    pub(super) switch_down_proj: QuantizedSwitchLinear,
    pub(super) router: QuantizedLinear,
    pub(super) top_k: i32,
}

impl GptOssMoE {
    fn switch_forward_single(&self, x: &Array, expert: i32) -> Result<Array> {
        let x_linear = self.switch_up_proj.forward_single(x, expert)?;
        let x_glu = self.switch_gate_proj.forward_single(x, expert)?;
        let x_glu = mlx_rs::ops::clip(&x_glu, ((), 7.0f32))?;
        let x_linear = mlx_rs::ops::clip(&x_linear, (-7.0f32, 7.0f32))?;
        let out_glu =
            x_glu.multiply(&mlx_rs::ops::sigmoid(&x_glu.multiply(&array!(1.702f32))?)?)?;
        let activated = out_glu.multiply(&x_linear.add(&array!(1.0f32))?)?;
        self.switch_down_proj.forward_single(&activated, expert)
    }

    pub(super) fn forward(&self, x: &Array) -> Result<Array> {
        let b = x.shape()[0];
        let l = x.shape()[1];
        let hidden = x.shape()[2];
        let flat = x.reshape(&[b * l, hidden])?;
        let router_logits = self.router.forward(&flat)?.as_dtype(Dtype::Float32)?;
        let inds = mlx_rs::ops::argpartition_axis(
            &router_logits.multiply(&array!(-1.0f32))?,
            self.top_k - 1,
            -1,
        )?
        .index((std::ops::RangeFull, ..self.top_k));
        let weights = mlx_rs::ops::indexing::take_along_axis(&router_logits, &inds, -1)?;
        let weights = mlx_rs::ops::softmax_axis(&weights, -1, true)?;
        mlx_rs::transforms::eval([&inds, &weights])?;
        let inds_slice = inds.as_slice::<u32>();
        let weights_slice = weights.as_slice::<f32>();
        let mut outputs = Vec::with_capacity((b * l) as usize);
        for token_idx in 0..(b * l) {
            let x_tok = flat.index((token_idx..token_idx + 1, std::ops::RangeFull));
            let mut token_out: Option<Array> = None;
            for expert_slot in 0..self.top_k {
                let offset = (token_idx * self.top_k + expert_slot) as usize;
                let expert = inds_slice[offset] as i32;
                let weight = weights_slice[offset];
                let routed = self
                    .switch_forward_single(&x_tok, expert)?
                    .multiply(&array!(weight))?;
                token_out = Some(match token_out {
                    Some(acc) => acc.add(&routed)?,
                    None => routed,
                });
            }
            outputs.push(token_out.context("gpt-oss moe produced no experts")?);
        }
        let output_refs: Vec<&Array> = outputs.iter().collect();
        Ok(mlx_rs::ops::concatenate_axis(&output_refs, 0)?.reshape(&[b, l, hidden])?)
    }
}

pub(crate) enum MlpKind {
    Dense(MLP),
    DeepseekV3MoE(DeepseekV3MoE),
    GptOssMoE(GptOssMoE),
}

impl MlpKind {
    pub(super) fn forward(&self, x: &Array) -> Result<Array> {
        match self {
            Self::Dense(mlp) => mlp.forward(x),
            Self::DeepseekV3MoE(moe) => moe.forward(x),
            Self::GptOssMoE(moe) => moe.forward(x),
        }
    }
}
