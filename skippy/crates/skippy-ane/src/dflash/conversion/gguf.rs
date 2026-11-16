//! Extraction and shape checks for the paired GGUF tensors.

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};
use half::f16;
use runtimed_gguf::GgufFile;

use crate::dflash::profile::DFlashProfile;

pub(super) struct Pair {
    pub draft: GgufFile,
    pub target: GgufFile,
    pub profile: DFlashProfile,
}

impl Pair {
    pub(super) fn open(draft: &Path, target: &Path) -> Result<Self> {
        let draft = GgufFile::open(draft).context("open paired DFlash GGUF")?;
        let target = GgufFile::open(target).context("open target GGUF")?;
        let profile = DFlashProfile::from_gguf(&draft, &target)?;
        validate_draft_tensors(&draft, &profile)?;
        Ok(Self {
            draft,
            target,
            profile,
        })
    }

    pub(super) fn draft_tensor(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        load_tensor(&self.draft, name, shape)
    }

    pub(super) fn target_tensor(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        load_tensor(&self.target, name, shape)
    }

    pub(super) fn embedding_tensor(&self) -> Result<Vec<f32>> {
        let shape = &[self.profile.vocabulary, self.profile.width];
        if self.draft.tensor("token_embd.weight").is_some() {
            self.draft_tensor("token_embd.weight", shape)
        } else {
            self.target_tensor("token_embd.weight", shape)
        }
    }

    pub(super) fn output_tensor(&self) -> Result<Vec<f32>> {
        let shape = &[self.profile.vocabulary, self.profile.width];
        if self.draft.tensor("output.weight").is_some() {
            self.draft_tensor("output.weight", shape)
        } else if self.target.tensor("output.weight").is_some() {
            self.target_tensor("output.weight", shape)
        } else {
            self.embedding_tensor()
        }
    }

    pub(super) fn target_embeddings(&self, path: &Path) -> Result<Vec<f32>> {
        let profile = &self.profile;
        let mut values = self.embedding_tensor()?;
        if profile.embedding_scale != 1.0 {
            for value in &mut values {
                *value *= profile.embedding_scale;
            }
        }
        let mut writer = BufWriter::new(
            File::create(path).with_context(|| format!("create {}", path.display()))?,
        );
        for &value in &values {
            writer.write_all(&f16::from_f32(value).to_bits().to_le_bytes())?;
        }
        writer.flush()?;
        let start = profile.mask_token * profile.width;
        Ok(values[start..start + profile.width].to_vec())
    }
}

fn validate_draft_tensors(draft: &GgufFile, profile: &DFlashProfile) -> Result<()> {
    let mut allowed = HashSet::from([
        "fc.weight".to_string(),
        "enc.output_norm.weight".to_string(),
        "output_norm.weight".to_string(),
        "token_embd.weight".to_string(),
        "output.weight".to_string(),
    ]);
    if profile.decoder == crate::dflash::profile::DecoderContract::Laguna {
        allowed.insert("enc.aux_norm.weight".to_string());
    }
    for index in 0..profile.layers() {
        for suffix in [
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
            "attn_norm.weight",
            "attn_q_norm.weight",
            "attn_k_norm.weight",
            "ffn_norm.weight",
        ] {
            allowed.insert(format!("blk.{index}.{suffix}"));
        }
        if profile.decoder == crate::dflash::profile::DecoderContract::Laguna {
            allowed.insert(format!("blk.{index}.attn_gate.weight"));
        }
    }
    for tensor in &draft.tensors {
        if !allowed.contains(&tensor.name) {
            bail!(
                "ANE DFlash graph does not implement draft tensor {}",
                tensor.name
            );
        }
    }
    Ok(())
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
