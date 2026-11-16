//! Float16 constants and per-output-channel INT8 Core ML weights.

use std::path::Path;

use anyhow::{Context, Result, bail};
use half::f16;
use mil_spec::{BlobWriter, Block, DType};

use super::gguf::Pair;

const BLOB_PATH: &str = "@model_path/weights/weight.bin";

pub(super) struct Weights<'a> {
    pair: &'a Pair,
    writer: BlobWriter,
}

impl<'a> Weights<'a> {
    pub(super) fn create(pair: &'a Pair, path: &Path) -> Result<Self> {
        let writer = BlobWriter::create(path)
            .with_context(|| format!("create Core ML weights {}", path.display()))?;
        Ok(Self { pair, writer })
    }

    pub(super) fn draft_matrix(
        &mut self,
        graph: &mut Block,
        tensor: &str,
        output: usize,
        input: usize,
    ) -> Result<String> {
        let values = self.pair.draft_tensor(tensor, &[output, input])?;
        self.matrix(graph, tensor, output, input, &values)
    }

    pub(super) fn optional_draft_matrix(
        &mut self,
        graph: &mut Block,
        tensor: &str,
        input: usize,
        allowed_outputs: &[usize],
    ) -> Result<Option<(String, usize)>> {
        let Some(info) = self.pair.draft.tensor(tensor) else {
            return Ok(None);
        };
        let output = usize::try_from(*info.dims.get(1).context("attention gate is not a matrix")?)?;
        if !allowed_outputs.contains(&output) {
            bail!("{tensor} output width {output} does not match the draft attention layout");
        }
        let value = self.draft_matrix(graph, tensor, output, input)?;
        Ok(Some((value, output)))
    }

    pub(super) fn head(
        &mut self,
        graph: &mut Block,
        output: usize,
        input: usize,
    ) -> Result<Vec<(String, usize)>> {
        let values = self.pair.output_tensor()?;
        if values.len() != output * input {
            bail!("DFlash output head does not match the paired vocabulary and hidden width");
        }
        let chunk = output.div_ceil(8);
        (0..output.div_ceil(chunk))
            .map(|part| {
                let start = part * chunk * input;
                let count = chunk.min(output - part * chunk);
                let weight = self.matrix(
                    graph,
                    &format!("output.weight.{part}"),
                    count,
                    input,
                    &values[start..start + count * input],
                )?;
                Ok((weight, count))
            })
            .collect()
    }

    fn matrix(
        &mut self,
        graph: &mut Block,
        name: &str,
        output: usize,
        input: usize,
        values: &[f32],
    ) -> Result<String> {
        if values.len() != output * input {
            bail!("{name} matrix has the wrong number of elements");
        }
        let mut quantized = Vec::with_capacity(values.len());
        let mut scales = Vec::with_capacity(output * 2);
        for row in values.chunks_exact(input) {
            let maximum = row.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
            if !maximum.is_finite() {
                bail!("{name} contains nonfinite weights");
            }
            let scale = if maximum == 0.0 { 1.0 } else { maximum / 127.0 };
            scales.extend_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
            quantized.extend(
                row.iter().map(|&value| {
                    (value / scale).round_ties_even().clamp(-127.0, 127.0) as i8 as u8
                }),
            );
        }
        let data_offset = self.writer.append(DType::Int8, &quantized)?;
        let scale_offset = self.writer.append(DType::Fp16, &scales)?;
        let ident = name.replace('.', "_");
        Ok(graph.konst_q8(
            &ident,
            BLOB_PATH,
            data_offset,
            scale_offset,
            &[output as i64, input as i64, 1, 1],
        ))
    }

    pub(super) fn draft_vector(
        &mut self,
        graph: &mut Block,
        tensor: &str,
        width: usize,
        shape: &[i64],
    ) -> Result<String> {
        let values = self.pair.draft_tensor(tensor, &[width])?;
        self.vector(graph, tensor, &values, shape)
    }

    pub(super) fn draft_array(
        &mut self,
        graph: &mut Block,
        tensor: &str,
        dims: &[usize],
        shape: &[i64],
    ) -> Result<String> {
        let values = self.pair.draft_tensor(tensor, dims)?;
        self.vector(graph, tensor, &values, shape)
    }

    pub(super) fn vector(
        &mut self,
        graph: &mut Block,
        name: &str,
        values: &[f32],
        shape: &[i64],
    ) -> Result<String> {
        if shape.iter().product::<i64>() as usize != values.len() {
            bail!("{name} constant has the wrong shape");
        }
        let bytes = values
            .iter()
            .flat_map(|&value| f16::from_f32(value).to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        let offset = self.writer.append(DType::Fp16, &bytes)?;
        Ok(graph.konst_blob(
            &name.replace('.', "_"),
            BLOB_PATH,
            offset,
            DType::Fp16,
            shape,
        ))
    }

    pub(super) fn finish(self) -> Result<()> {
        self.writer.finish().context("finish Core ML weight blob")
    }
}
