//! Incremental affine expert-bank assembly without retaining dense experts.

use anyhow::{Context, Result, ensure};
use safemlx::{Stream, transforms::eval};
use safemlx_lm::quantization::QuantizedTensor;

use super::{OwnedTensor, owned_tensor};

const MAX_EXPERTS: usize = u128::BITS as usize;

/// Builds one rank-3 affine expert projection from independently quantized
/// rank-2 experts.
///
/// The final packed payloads are allocated once. Each expert is copied into its
/// own leading-dimension slice, so callers never need to retain every dense or
/// MLX expert array at the same time.
pub(super) struct AffineExpertBankAssembler {
    output_prefix: String,
    expert_count: usize,
    completed: u128,
    weights: Option<BankTensor>,
    scales: Option<BankTensor>,
    biases: Option<BankTensor>,
}

impl AffineExpertBankAssembler {
    pub(super) fn new(output_prefix: impl Into<String>, expert_count: usize) -> Result<Self> {
        ensure!(
            expert_count > 0,
            "expert bank must contain at least one expert"
        );
        ensure!(
            expert_count <= MAX_EXPERTS,
            "expert bank completion bitmap supports at most {MAX_EXPERTS} experts"
        );
        Ok(Self {
            output_prefix: output_prefix.into(),
            expert_count,
            completed: 0,
            weights: None,
            scales: None,
            biases: None,
        })
    }

    pub(super) fn insert(
        &mut self,
        expert_index: usize,
        quantized: QuantizedTensor,
        stream: &Stream,
    ) -> Result<()> {
        ensure!(
            expert_index < self.expert_count,
            "expert index {expert_index} exceeds bank size {}",
            self.expert_count
        );
        let bit = 1_u128 << expert_index;
        ensure!(
            self.completed & bit == 0,
            "expert {expert_index} was inserted more than once"
        );
        let biases = quantized
            .biases
            .context("affine expert quantization did not produce biases")?;
        eval([&quantized.weight, &quantized.scales, &biases])?;
        stream.synchronize()?;

        insert_tensor(
            &mut self.weights,
            expert_index,
            self.expert_count,
            owned_tensor(&quantized.weight)?,
        )?;
        insert_tensor(
            &mut self.scales,
            expert_index,
            self.expert_count,
            owned_tensor(&quantized.scales)?,
        )?;
        insert_tensor(
            &mut self.biases,
            expert_index,
            self.expert_count,
            owned_tensor(&biases)?,
        )?;
        self.completed |= bit;
        Ok(())
    }

    pub(super) fn finish(self) -> Result<Vec<(String, OwnedTensor)>> {
        ensure!(
            self.completed == completion_mask(self.expert_count),
            "expert bank is incomplete: received {}/{} experts",
            self.completed.count_ones(),
            self.expert_count
        );
        Ok(vec![
            (
                self.output_prefix.clone(),
                self.weights
                    .context("expert bank has no packed weights")?
                    .finish(),
            ),
            (
                format!("{}_scales", self.output_prefix),
                self.scales.context("expert bank has no scales")?.finish(),
            ),
            (
                format!("{}_biases", self.output_prefix),
                self.biases.context("expert bank has no biases")?.finish(),
            ),
        ])
    }
}

struct BankTensor {
    tensor: OwnedTensor,
    expert_shape: Vec<usize>,
    expert_bytes: usize,
}

impl BankTensor {
    fn new(expert_count: usize, expert: &OwnedTensor) -> Result<Self> {
        ensure!(
            expert.shape.len() == 2,
            "expert projection output must be rank 2, got shape {:?}",
            expert.shape
        );
        let total_bytes = expert
            .data
            .len()
            .checked_mul(expert_count)
            .context("expert bank byte count overflow")?;
        let mut shape = Vec::with_capacity(3);
        shape.push(expert_count);
        shape.extend_from_slice(&expert.shape);
        Ok(Self {
            tensor: OwnedTensor {
                dtype: expert.dtype,
                shape,
                data: vec![0_u8; total_bytes],
            },
            expert_shape: expert.shape.clone(),
            expert_bytes: expert.data.len(),
        })
    }

    fn insert(&mut self, expert_index: usize, expert: OwnedTensor) -> Result<()> {
        ensure!(
            expert.dtype == self.tensor.dtype,
            "expert projection dtype changed from {:?} to {:?}",
            self.tensor.dtype,
            expert.dtype
        );
        ensure!(
            expert.shape == self.expert_shape,
            "expert projection shape changed from {:?} to {:?}",
            self.expert_shape,
            expert.shape
        );
        ensure!(
            expert.data.len() == self.expert_bytes,
            "expert projection byte count changed from {} to {}",
            self.expert_bytes,
            expert.data.len()
        );
        let start = expert_index
            .checked_mul(self.expert_bytes)
            .context("expert bank slice offset overflow")?;
        let end = start
            .checked_add(self.expert_bytes)
            .context("expert bank slice end overflow")?;
        self.tensor.data[start..end].copy_from_slice(&expert.data);
        Ok(())
    }

    fn finish(self) -> OwnedTensor {
        self.tensor
    }
}

fn insert_tensor(
    bank: &mut Option<BankTensor>,
    expert_index: usize,
    expert_count: usize,
    expert: OwnedTensor,
) -> Result<()> {
    if bank.is_none() {
        *bank = Some(BankTensor::new(expert_count, &expert)?);
    }
    bank.as_mut()
        .expect("expert bank initialized above")
        .insert(expert_index, expert)
}

const fn completion_mask(expert_count: usize) -> u128 {
    if expert_count == MAX_EXPERTS {
        u128::MAX
    } else {
        (1_u128 << expert_count) - 1
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use half::bf16;
    use safemlx::{Array, Device, DeviceType, Stream, ops::stack_axis, transforms::eval};
    use safemlx_lm::quantization::{AffineQuantization, quantize_tensor};

    use super::*;

    #[test]
    fn incremental_bank_matches_quantizing_the_stacked_experts() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let quantization = AffineQuantization::new(64, 4).unwrap();
        let experts = (0..2)
            .map(|expert| {
                let values = (0..(7 * 64))
                    .map(|index| {
                        let value = (expert * 449 + index) as f32;
                        bf16::from_f32(value.mul_add(0.03125, -9.0))
                    })
                    .collect::<Vec<_>>();
                Array::from_slice(&values, &[7, 64])
            })
            .collect::<Vec<_>>();

        let mut actual = AffineExpertBankAssembler::new("experts.up_proj", 2).unwrap();
        for expert_index in [1, 0] {
            let quantized = quantize_tensor(&experts[expert_index], quantization, &stream).unwrap();
            actual.insert(expert_index, quantized, &stream).unwrap();
        }
        let actual = actual
            .finish()
            .unwrap()
            .into_iter()
            .collect::<BTreeMap<_, _>>();

        let stacked = stack_axis(&experts, 0, &stream).unwrap();
        let expected = quantize_tensor(&stacked, quantization, &stream).unwrap();
        let biases = expected.biases.as_ref().unwrap();
        eval([&expected.weight, &expected.scales, biases]).unwrap();
        stream.synchronize().unwrap();
        let expected = BTreeMap::from([
            ("experts.up_proj", owned_tensor(&expected.weight).unwrap()),
            (
                "experts.up_proj_scales",
                owned_tensor(&expected.scales).unwrap(),
            ),
            ("experts.up_proj_biases", owned_tensor(biases).unwrap()),
        ]);

        assert_eq!(
            actual.keys().collect::<Vec<_>>(),
            expected.keys().collect::<Vec<_>>()
        );
        for (name, expected) in expected {
            let actual = &actual[name];
            assert_eq!(actual.dtype, expected.dtype, "{name} dtype");
            assert_eq!(actual.shape, expected.shape, "{name} shape");
            assert_eq!(actual.data, expected.data, "{name} bytes");
        }
    }

    #[test]
    fn incomplete_and_duplicate_banks_fail_closed() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let quantization = AffineQuantization::new(64, 4).unwrap();
        let dense = Array::from_slice(&vec![bf16::from_f32(0.5); 3 * 64], &[3, 64]);
        let mut bank = AffineExpertBankAssembler::new("experts.down_proj", 2).unwrap();
        bank.insert(
            0,
            quantize_tensor(&dense, quantization, &stream).unwrap(),
            &stream,
        )
        .unwrap();
        let duplicate = bank
            .insert(
                0,
                quantize_tensor(&dense, quantization, &stream).unwrap(),
                &stream,
            )
            .unwrap_err();
        assert!(duplicate.to_string().contains("more than once"));
        let incomplete = bank.finish().err().expect("bank should be incomplete");
        assert!(incomplete.to_string().contains("incomplete"));
    }
}
