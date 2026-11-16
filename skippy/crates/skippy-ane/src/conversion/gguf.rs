//! Extraction and shape checks for the paired GGUF tensors.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};
use half::f16;
use runtimed_gguf::{GgufFile, MetaValue};

pub(super) const WIDTH: usize = 2_048;
pub(super) const TAPS: [u32; 8] = [2, 7, 12, 17, 23, 28, 33, 38];
pub(super) const MASK_TOKEN: usize = 248_077;
pub(super) const VOCAB: usize = 248_320;

pub(super) struct Pair {
    pub draft: GgufFile,
    pub target: GgufFile,
}

impl Pair {
    pub(super) fn open(draft: &Path, target: &Path) -> Result<Self> {
        let draft = GgufFile::open(draft).context("open paired DFlash GGUF")?;
        let target = GgufFile::open(target).context("open target GGUF")?;
        let expected = [
            ("dflash.block_count", 6),
            ("dflash.embedding_length", WIDTH as u32),
            ("dflash.attention.head_count", 32),
            ("dflash.attention.head_count_kv", 8),
            ("dflash.attention.key_length", 128),
            ("dflash.attention.value_length", 128),
            ("tokenizer.ggml.mask_token_id", MASK_TOKEN as u32),
        ];
        for (name, value) in expected {
            if draft.metadata.get(name) != Some(&MetaValue::U32(value)) {
                bail!("{name} does not match the paired Qwen3.6 DFlash checkpoint");
            }
        }
        let taps = TAPS
            .iter()
            .map(|&value| MetaValue::I32(value as i32))
            .collect();
        if draft.metadata.get("dflash.target_layers") != Some(&MetaValue::Arr(taps)) {
            bail!("DFlash target feature taps differ from the paired checkpoint");
        }
        if draft.meta_str("general.architecture") != Some("dflash")
            || target.meta_str("general.architecture") != Some("qwen35moe")
            || draft.tensor("markov_w1.weight").is_some()
        {
            bail!("GGUF architecture is not the paired Qwen3.6 DFlash model");
        }
        Ok(Self { draft, target })
    }

    pub(super) fn draft_tensor(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        load_tensor(&self.draft, name, shape)
    }

    pub(super) fn target_tensor(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        load_tensor(&self.target, name, shape)
    }

    pub(super) fn target_embeddings(&self, path: &Path) -> Result<Vec<f32>> {
        let values = self.target_tensor("token_embd.weight", &[VOCAB, WIDTH])?;
        let mut writer = BufWriter::new(
            File::create(path).with_context(|| format!("create {}", path.display()))?,
        );
        for &value in &values {
            writer.write_all(&f16::from_f32(value).to_bits().to_le_bytes())?;
        }
        writer.flush()?;
        Ok(values[MASK_TOKEN * WIDTH..(MASK_TOKEN + 1) * WIDTH].to_vec())
    }
}

fn load_tensor(file: &GgufFile, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let info = file
        .tensor(name)
        .with_context(|| format!("missing GGUF tensor {name}"))?;
    let expected = shape
        .iter()
        .rev()
        .map(|&dim| dim as u64)
        .collect::<Vec<_>>();
    if info.dims != expected {
        bail!("GGUF tensor {name} shape {:?} != {:?}", info.dims, expected);
    }
    file.tensor_f32(info)
        .with_context(|| format!("dequantize GGUF tensor {name}"))
}
