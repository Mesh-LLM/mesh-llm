//! Nemotron-H checkpoint rewriting and bounded routed-expert conversion.

use std::{collections::BTreeMap, path::Path};

use anyhow::{Context, Result, ensure};
use half::bf16;
use safemlx::{
    Array, Device, DeviceType, Stream,
    memory::{active_memory, cache_memory, peak_memory, reset_peak_memory},
    module::Module,
    transforms::eval,
};
use safemlx_lm::{
    models::nemotron_h::{BlockInput, TransformerBlock, get_nemotron_h_model_args},
    quantization::{WeightQuantization, quantize_tensor},
    weights::{StrictLoadConfig, StrictLoadReport, load_safetensors_dir_strict},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::{OwnedTensor, expert_bank::AffineExpertBankAssembler};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Projection {
    Up,
    Down,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LayerKind {
    Mamba,
    Attention,
    Mlp,
    Moe,
}

impl LayerKind {
    fn from_marker(marker: char) -> Result<Self> {
        match marker {
            'M' => Ok(Self::Mamba),
            '*' => Ok(Self::Attention),
            '-' => Ok(Self::Mlp),
            'E' => Ok(Self::Moe),
            other => {
                anyhow::bail!("Nemotron-H layer pattern contains unsupported marker {other:?}")
            }
        }
    }

    const fn parameter_field(self) -> &'static str {
        match self {
            Self::Mamba => "mamba",
            Self::Attention => "attention",
            Self::Mlp => "mlp",
            Self::Moe => "moe",
        }
    }
}

#[derive(Debug, Deserialize)]
struct SourceConfig {
    hidden_size: i32,
    num_hidden_layers: usize,
    hybrid_override_pattern: String,
    n_routed_experts: i32,
    moe_intermediate_size: i32,
}

pub(super) struct NemotronHDerivation {
    hidden_size: i32,
    intermediate_size: i32,
    expert_count: usize,
    layer_kinds: Vec<LayerKind>,
    banks: BTreeMap<(usize, Projection), AffineExpertBankAssembler>,
}

/// Evidence that one derived Nemotron-H MoE layer strict-loads and executes.
#[derive(Clone, Debug, Serialize)]
pub struct MlxNemotronHValidationReport {
    pub model_dir: std::path::PathBuf,
    pub layer: usize,
    pub input_shape: Vec<i32>,
    pub output_shape: Vec<i32>,
    pub output_is_finite: bool,
    pub output_f32_sha256: String,
    pub mlx_active_memory_bytes: usize,
    pub mlx_cache_memory_bytes: usize,
    pub mlx_peak_memory_bytes: usize,
}

/// Strict-loads a derived affine Nemotron-H MoE block and runs a deterministic
/// nonzero hidden state through it.
pub fn validate_nemotron_h_moe_stage(
    model_dir: impl AsRef<Path>,
    layer: usize,
) -> Result<MlxNemotronHValidationReport> {
    Ok(validate_nemotron_h_moe_stage_output(model_dir, layer)?.0)
}

pub(crate) fn validate_nemotron_h_moe_stage_output(
    model_dir: impl AsRef<Path>,
    layer: usize,
) -> Result<(MlxNemotronHValidationReport, Vec<f32>)> {
    validate_nemotron_h_moe_stage_output_for_tokens(model_dir, layer, 1)
}

pub(crate) fn validate_nemotron_h_moe_stage_output_for_tokens(
    model_dir: impl AsRef<Path>,
    layer: usize,
    token_count: usize,
) -> Result<(MlxNemotronHValidationReport, Vec<f32>)> {
    let model_dir = model_dir.as_ref();
    ensure!(token_count > 0, "Nemotron-H validation needs tokens");
    let token_count_i32 = i32::try_from(token_count).context("token count exceeds i32")?;
    let args = get_nemotron_h_model_args(model_dir)?;
    ensure!(
        args.hybrid_override_pattern
            .chars()
            .nth(layer)
            .is_some_and(|marker| marker == 'E'),
        "Nemotron-H layer {layer} is not an MoE layer"
    );
    reset_peak_memory()?;
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let mut block = TransformerBlock::new(&args, layer, &stream)?;
    let load_config = StrictLoadConfig::default().strip_prefix(format!("model.layers.{layer}."));
    let mut load_report = StrictLoadReport::default();
    load_safetensors_dir_strict(
        &mut block,
        model_dir,
        &stream,
        &load_config,
        &mut load_report,
    )?;
    load_report.finish(&block, &load_config)?;

    let values = nemotron_h_validation_values(args.hidden_size, token_count)?;
    let input = Array::from_slice(&values, &[1, token_count_i32, args.hidden_size])
        .as_dtype(safemlx::Dtype::Bfloat16, &stream)?;
    let output = block.forward(
        BlockInput {
            x: &input,
            mask: None,
            cache: None,
        },
        &stream,
    )?;
    let output_f32 = output.as_dtype(safemlx::Dtype::Float32, &stream)?;
    let finite = output_f32.is_finite(&stream)?.all(None, &stream)?;
    eval([&output_f32, &finite])?;
    stream.synchronize()?;
    let output_is_finite = finite.try_item::<bool>(&stream)?;
    ensure!(
        output_is_finite,
        "Nemotron-H layer output contains non-finite values"
    );
    let evaluated_output = output_f32.evaluated()?;
    let output_values = evaluated_output.as_slice::<f32>().to_vec();
    let output_f32_sha256 = f32_sha256(&output_values);

    Ok((
        MlxNemotronHValidationReport {
            model_dir: model_dir.to_path_buf(),
            layer,
            input_shape: input.shape().to_vec(),
            output_shape: output.shape().to_vec(),
            output_is_finite,
            output_f32_sha256,
            mlx_active_memory_bytes: active_memory()?,
            mlx_cache_memory_bytes: cache_memory()?,
            mlx_peak_memory_bytes: peak_memory()?,
        },
        output_values,
    ))
}

pub(crate) fn nemotron_h_validation_values(
    hidden_size: i32,
    token_count: usize,
) -> Result<Vec<f32>> {
    let element_count = usize::try_from(hidden_size)?
        .checked_mul(token_count)
        .context("Nemotron-H validation input size overflow")?;
    Ok((0..element_count)
        .map(|index| bf16::from_f32(((index % 31) as f32 - 15.0) / 32.0).to_f32())
        .collect())
}

fn f32_sha256(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

impl NemotronHDerivation {
    pub(super) fn new(config: &Value, layer_start: u32, layer_end: u32) -> Result<Self> {
        ensure!(
            config.get("layers_block_type").is_none() && config.get("moe_latent_size").is_none(),
            "Nemotron-H latent-MoE/Ultra configs require a separate runtime family"
        );
        let source: SourceConfig =
            serde_json::from_value(config.clone()).context("parse Nemotron-H derivation config")?;
        ensure!(
            source.hidden_size > 0,
            "Nemotron-H hidden_size must be non-zero"
        );
        ensure!(
            source.moe_intermediate_size > 0,
            "Nemotron-H moe_intermediate_size must be non-zero"
        );
        ensure!(
            source.n_routed_experts > 0 && source.n_routed_experts <= i32::try_from(u128::BITS)?,
            "Nemotron-H n_routed_experts must be in 1..={} for bounded assembly",
            u128::BITS
        );
        let layer_kinds = source
            .hybrid_override_pattern
            .chars()
            .map(LayerKind::from_marker)
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            layer_kinds.len() == source.num_hidden_layers,
            "Nemotron-H layer pattern has {} entries, expected {}",
            layer_kinds.len(),
            source.num_hidden_layers
        );
        let start = usize::try_from(layer_start)?;
        let end = usize::try_from(layer_end)?;
        ensure!(
            end <= layer_kinds.len(),
            "Nemotron-H stage exceeds layer pattern"
        );
        ensure!(
            end == start + 1 && layer_kinds[start] == LayerKind::Moe,
            "bounded Nemotron-H derivation currently requires exactly one MoE layer"
        );
        let expert_count = usize::try_from(source.n_routed_experts)?;

        let mut banks = BTreeMap::new();
        for (layer, kind) in layer_kinds.iter().enumerate().take(end).skip(start) {
            if *kind != LayerKind::Moe {
                continue;
            }
            for projection in [Projection::Up, Projection::Down] {
                let suffix = match projection {
                    Projection::Up => "up_proj",
                    Projection::Down => "down_proj",
                };
                banks.insert(
                    (layer, projection),
                    AffineExpertBankAssembler::new(
                        format!("model.layers.{layer}.moe.experts.{suffix}"),
                        expert_count,
                    )?,
                );
            }
        }
        Ok(Self {
            hidden_size: source.hidden_size,
            intermediate_size: source.moe_intermediate_size,
            expert_count,
            layer_kinds,
            banks,
        })
    }

    /// Consumes a split routed-expert matrix when `source_name` names one.
    pub(super) fn consume_expert(
        &mut self,
        source_name: &str,
        dense: &Array,
        quantization: WeightQuantization,
        stream: &Stream,
    ) -> Result<bool> {
        let Some(expert) = parse_expert_name(source_name)? else {
            return Ok(false);
        };
        ensure!(
            expert.expert < self.expert_count,
            "Nemotron-H expert {} exceeds configured count {}",
            expert.expert,
            self.expert_count
        );
        let expected_shape = match expert.projection {
            Projection::Up => [self.intermediate_size, self.hidden_size],
            Projection::Down => [self.hidden_size, self.intermediate_size],
        };
        ensure!(
            dense.shape() == expected_shape,
            "Nemotron-H expert source {source_name} has shape {:?}, expected {:?}",
            dense.shape(),
            expected_shape
        );
        let bank = self
            .banks
            .get_mut(&(expert.layer, expert.projection))
            .with_context(|| {
                format!("Nemotron-H expert {source_name} does not belong to a selected MoE layer")
            })?;
        bank.insert(
            expert.expert,
            quantize_tensor(dense, quantization, stream)?,
            stream,
        )?;
        Ok(true)
    }

    pub(super) fn rewrite_name(&self, source_name: &str) -> Result<String> {
        let layer_root = source_name
            .strip_prefix("backbone.layers.")
            .or_else(|| source_name.strip_prefix("model.backbone.layers."));
        if let Some(rest) = layer_root {
            let (layer, suffix) = rest
                .split_once('.')
                .with_context(|| format!("invalid Nemotron-H layer tensor {source_name}"))?;
            let layer = layer.parse::<usize>().with_context(|| {
                format!("invalid Nemotron-H layer index in tensor {source_name}")
            })?;
            let kind = self
                .layer_kinds
                .get(layer)
                .with_context(|| format!("Nemotron-H tensor layer {layer} is out of range"))?;
            if let Some(mixer_suffix) = suffix.strip_prefix("mixer.") {
                return Ok(format!(
                    "model.layers.{layer}.{}.{mixer_suffix}",
                    kind.parameter_field()
                ));
            }
            return Ok(format!("model.layers.{layer}.{suffix}"));
        }
        if let Some(rest) = source_name.strip_prefix("model.backbone.") {
            return Ok(format!("model.{rest}"));
        }
        if let Some(rest) = source_name.strip_prefix("backbone.") {
            return Ok(format!("model.{rest}"));
        }
        Ok(source_name.to_string())
    }

    pub(super) fn keep_dense(output_name: &str) -> bool {
        output_name.ends_with(".moe.gate.weight")
    }

    pub(super) fn finish(self) -> Result<Vec<(String, OwnedTensor)>> {
        self.banks
            .into_values()
            .try_fold(Vec::new(), |mut output, bank| {
                output.extend(bank.finish()?);
                Ok(output)
            })
    }
}

struct ExpertSource {
    layer: usize,
    expert: usize,
    projection: Projection,
}

fn parse_expert_name(name: &str) -> Result<Option<ExpertSource>> {
    let root = name
        .strip_prefix("backbone.layers.")
        .or_else(|| name.strip_prefix("model.backbone.layers."));
    let Some(rest) = root else {
        return Ok(None);
    };
    let Some((layer, rest)) = rest.split_once(".mixer.experts.") else {
        return Ok(None);
    };
    let (expert, projection) = rest
        .split_once('.')
        .with_context(|| format!("invalid split Nemotron-H expert tensor {name}"))?;
    let projection = match projection {
        "up_proj.weight" => Projection::Up,
        "down_proj.weight" => Projection::Down,
        other => anyhow::bail!("unsupported split Nemotron-H expert projection {other:?}"),
    };
    Ok(Some(ExpertSource {
        layer: layer
            .parse()
            .with_context(|| format!("invalid Nemotron-H expert layer in {name}"))?,
        expert: expert
            .parse()
            .with_context(|| format!("invalid Nemotron-H expert index in {name}"))?,
        projection,
    }))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn config() -> Value {
        json!({
            "model_type": "nemotron_h",
            "hidden_size": 64,
            "num_hidden_layers": 4,
            "hybrid_override_pattern": "ME-*",
            "n_routed_experts": 2,
            "moe_intermediate_size": 32
        })
    }

    #[test]
    fn validation_input_covers_every_token() {
        let values = nemotron_h_validation_values(3, 2).unwrap();
        assert_eq!(values.len(), 6);
        assert_eq!(values[0], -15.0 / 32.0);
        assert_eq!(values[3], -12.0 / 32.0);
    }

    #[test]
    fn rewrites_public_layer_keys_to_runtime_fields() {
        let derivation = NemotronHDerivation::new(&config(), 1, 2).unwrap();
        assert_eq!(
            derivation
                .rewrite_name("backbone.layers.0.mixer.in_proj.weight")
                .unwrap(),
            "model.layers.0.mamba.in_proj.weight"
        );
        assert_eq!(
            derivation
                .rewrite_name("backbone.layers.1.mixer.gate.weight")
                .unwrap(),
            "model.layers.1.moe.gate.weight"
        );
        assert_eq!(
            derivation
                .rewrite_name("backbone.layers.2.mixer.up_proj.weight")
                .unwrap(),
            "model.layers.2.mlp.up_proj.weight"
        );
        assert_eq!(
            derivation
                .rewrite_name("backbone.layers.3.mixer.q_proj.weight")
                .unwrap(),
            "model.layers.3.attention.q_proj.weight"
        );
        assert_eq!(
            derivation
                .rewrite_name("backbone.embeddings.weight")
                .unwrap(),
            "model.embeddings.weight"
        );
    }

    #[test]
    fn parses_split_experts_and_rejects_ultra_schema() {
        let expert = parse_expert_name("backbone.layers.1.mixer.experts.127.down_proj.weight")
            .unwrap()
            .unwrap();
        assert_eq!(expert.layer, 1);
        assert_eq!(expert.expert, 127);
        assert_eq!(expert.projection, Projection::Down);

        let mut ultra = config();
        ultra["layers_block_type"] = json!(["mamba"]);
        let error = NemotronHDerivation::new(&ultra, 0, 1)
            .err()
            .expect("Ultra schema should fail");
        assert!(error.to_string().contains("separate runtime family"));
    }
}
