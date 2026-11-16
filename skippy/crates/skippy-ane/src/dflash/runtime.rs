//! Stateful draft execution and lane state for a compiled DFlash profile.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use half::f16;
use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::profile::DFlashProfile;

#[path = "coreml.rs"]
mod coreml;

use coreml::CoreMlDraft;

struct FeatureRow {
    position: i32,
    values: Vec<f32>,
}

struct Lane {
    pending: Vec<FeatureRow>,
    slots: Vec<Option<i32>>,
}

#[derive(Deserialize)]
struct PackageMetadata {
    format_version: u32,
    draft_sha256: String,
    embeddings_sha256: String,
    context: usize,
    rows: usize,
    token_output: bool,
    taps: Vec<u32>,
    profile: DFlashProfile,
}

/// One compiled model and one independent Core ML state per Skippy lane.
pub struct AneDraft {
    coreml: CoreMlDraft,
    embeddings: File,
    #[cfg(feature = "conversion")]
    _package_dir: Option<tempfile::TempDir>,
    context: usize,
    profile: DFlashProfile,
    rows: usize,
    lanes: Vec<Lane>,
}

/// Local GGUF pair and Core ML graph dimensions for load-time conversion.
#[cfg(feature = "conversion")]
pub struct PairedGguf<'a> {
    pub draft: &'a Path,
    pub target: &'a Path,
    pub context: usize,
    pub rows: usize,
}

impl AneDraft {
    /// Converts one paired GGUF into a private package and loads it on ANE.
    #[cfg(feature = "conversion")]
    pub fn from_pair(
        lanes: usize,
        n_vocab: usize,
        mask_token: i32,
        pair: PairedGguf<'_>,
    ) -> Result<Self> {
        let package_dir = tempfile::tempdir().context("create ANE draft package directory")?;
        let model = package_dir.path().join("draft.mlpackage");
        let embeddings = package_dir.path().join("embeddings.f16");
        super::conversion::convert(&super::conversion::ConversionOptions {
            draft: pair.draft.to_path_buf(),
            target: pair.target.to_path_buf(),
            out: model.clone(),
            embeddings_out: embeddings.clone(),
            context: pair.context,
            rows: pair.rows,
        })?;
        let compiled = model.with_extension("mlmodelc");
        let mut draft = Self::open(
            &compiled,
            &embeddings,
            lanes,
            n_vocab,
            mask_token,
            pair.draft,
        )?;
        draft._package_dir = Some(package_dir);
        Ok(draft)
    }

    pub fn from_env(
        lanes: usize,
        n_vocab: usize,
        mask_token: i32,
        draft_path: &Path,
    ) -> Result<Option<Self>> {
        let Ok(model) = std::env::var("SKIPPY_DFLASH_ANE_MODEL") else {
            return Ok(None);
        };
        let embeddings = std::env::var("SKIPPY_DFLASH_ANE_EMBEDDINGS")
            .context("SKIPPY_DFLASH_ANE_EMBEDDINGS must name the paired target embeddings")?;
        Self::open(
            Path::new(&model),
            Path::new(&embeddings),
            lanes,
            n_vocab,
            mask_token,
            draft_path,
        )
        .map(Some)
    }

    fn open(
        model: &Path,
        embeddings_path: &Path,
        lanes: usize,
        n_vocab: usize,
        mask_token: i32,
        draft_path: &Path,
    ) -> Result<Self> {
        let metadata_path = Path::new(&model).with_extension("json");
        let metadata: PackageMetadata = serde_json::from_slice(
            &std::fs::read(&metadata_path)
                .with_context(|| format!("read ANE draft metadata {}", metadata_path.display()))?,
        )
        .with_context(|| format!("parse ANE draft metadata {}", metadata_path.display()))?;
        if metadata.format_version != 2 {
            bail!("ANE DFlash package format is incompatible with this runtime");
        }
        let context = metadata.context;
        let rows = metadata.rows;
        let profile = metadata.profile;
        if n_vocab != profile.vocabulary
            || usize::try_from(mask_token).ok() != Some(profile.mask_token)
            || context < rows
            || rows < 2
            || lanes == 0
        {
            bail!("ANE DFlash package dimensions do not match the loaded draft");
        }
        if !metadata.token_output || metadata.taps != profile.target_layers {
            bail!("ANE DFlash package metadata does not match the paired draft graph");
        }
        if file_sha256(draft_path)? != metadata.draft_sha256 {
            bail!("ANE DFlash package was generated from a different draft GGUF");
        }
        if file_sha256(embeddings_path)? != metadata.embeddings_sha256 {
            bail!("ANE DFlash target embeddings do not match the package metadata");
        }
        let embeddings = File::open(embeddings_path)
            .with_context(|| format!("open ANE target embeddings {}", embeddings_path.display()))?;
        if embeddings.metadata()?.len() != (profile.vocabulary * profile.width * 2) as u64 {
            bail!("ANE target embeddings have the wrong size");
        }
        let coreml = CoreMlDraft::open(model, lanes, context, rows, &profile)?;
        Ok(Self {
            coreml,
            embeddings,
            #[cfg(feature = "conversion")]
            _package_dir: None,
            context,
            profile,
            rows,
            lanes: (0..lanes)
                .map(|_| Lane {
                    pending: Vec::new(),
                    slots: vec![None; context],
                })
                .collect(),
        })
    }

    pub fn max_proposals(&self) -> usize {
        self.rows - 1
    }

    fn lane(&mut self, seq_id: i32) -> Result<usize> {
        usize::try_from(seq_id)
            .ok()
            .filter(|&index| index < self.lanes.len())
            .ok_or_else(|| anyhow!("ANE DFlash lane {seq_id} is invalid"))
    }

    pub fn ingest(&mut self, seq_id: i32, features: &[f32], positions: &[i32]) -> Result<()> {
        if positions.is_empty() {
            return Ok(());
        }
        let width = self
            .profile
            .feature_width()
            .context("invalid DFlash feature width")?;
        if features.len() != positions.len() * width {
            bail!("ANE DFlash feature rows have the wrong width");
        }
        let lane = self.lane(seq_id)?;
        for (&position, values) in positions.iter().zip(features.chunks_exact(width)) {
            if position < 0 {
                bail!("ANE DFlash feature position is negative");
            }
            self.lanes[lane].pending.push(FeatureRow {
                position,
                values: values.to_vec(),
            });
        }
        while self.lanes[lane].pending.len() > self.rows {
            let rows = self.lanes[lane]
                .pending
                .drain(..self.rows)
                .collect::<Vec<_>>();
            let after = rows.last().expect("nonempty chunk").position + 1;
            self.run(lane, after, self.profile.mask_token as i32, &rows)?;
        }
        Ok(())
    }

    pub fn propose(
        &mut self,
        seq_id: i32,
        anchor_position: i32,
        anchor: i32,
        max_tokens: usize,
    ) -> Result<Vec<i32>> {
        if max_tokens > self.rows - 1 {
            bail!("ANE DFlash proposal count exceeds the compiled block width");
        }
        if anchor_position < 0 {
            bail!("ANE DFlash anchor position is negative");
        }
        let lane = self.lane(seq_id)?;
        let rows = std::mem::take(&mut self.lanes[lane].pending);
        let mut tokens = self.run(lane, anchor_position, anchor, &rows)?;
        tokens.truncate(max_tokens);
        if tokens
            .iter()
            .any(|&token| token < 0 || token as usize >= self.profile.vocabulary)
        {
            bail!("ANE DFlash returned a token outside the target vocabulary");
        }
        Ok(tokens)
    }

    fn run(
        &mut self,
        lane: usize,
        anchor_position: i32,
        anchor: i32,
        rows: &[FeatureRow],
    ) -> Result<Vec<i32>> {
        let context = self.context;
        let block_rows = self.rows;
        let mut features = vec![
            0u16;
            self.profile
                .feature_width()
                .context("invalid DFlash feature width")?
                * block_rows
        ];
        let mut write = vec![0u16; block_rows * context];
        let mut ctx_positions = vec![0i32; block_rows];
        let mut slots = self.lanes[lane].slots.clone();
        for (row, feature) in rows.iter().enumerate() {
            let position =
                usize::try_from(feature.position).context("negative ANE feature position")?;
            let slot = position % context;
            ctx_positions[row] = feature.position;
            write[row * context + slot] = f16::ONE.to_bits();
            slots[slot] = Some(feature.position);
            for (channel, &value) in feature.values.iter().enumerate() {
                features[channel * block_rows + row] = f16::from_f32(value).to_bits();
            }
        }
        let (ctx_cos, ctx_sin) = rope(
            &ctx_positions,
            self.profile.rotary_width,
            self.profile.rope_base,
        );
        let query_positions = (0..block_rows)
            .map(|index| anchor_position + index as i32)
            .collect::<Vec<_>>();
        let (q_cos, q_sin) = rope(
            &query_positions,
            self.profile.rotary_width,
            self.profile.rope_base,
        );
        let mut mask = vec![f16::MIN.to_bits(); block_rows * (context + block_rows)];
        for row in 0..block_rows {
            let offset = row * (context + block_rows);
            for (slot, position) in slots.iter().enumerate() {
                if position.is_some_and(|position| {
                    position < anchor_position && position >= anchor_position - context as i32
                }) {
                    mask[offset + slot] = f16::ZERO.to_bits();
                }
            }
            let visible = if self.profile.causal {
                row + 1
            } else {
                block_rows
            };
            mask[offset + context..offset + context + visible].fill(f16::ZERO.to_bits());
        }
        let anchor = self.embedding(anchor)?;
        let tokens = self.coreml.predict(
            lane, &features, &write, &ctx_cos, &ctx_sin, &anchor, &q_cos, &q_sin, &mask,
        )?;
        self.lanes[lane].slots = slots;
        Ok(tokens)
    }

    fn embedding(&mut self, token: i32) -> Result<Vec<u16>> {
        let token = usize::try_from(token)
            .ok()
            .filter(|&token| token < self.profile.vocabulary)
            .ok_or_else(|| anyhow!("ANE DFlash anchor token {token} is invalid"))?;
        self.embeddings
            .seek(SeekFrom::Start((token * self.profile.width * 2) as u64))?;
        let mut bytes = vec![0u8; self.profile.width * 2];
        self.embeddings.read_exact(&mut bytes)?;
        Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|chunk| u16::from_le_bytes(*chunk))
            .collect())
    }

    pub fn truncate(&mut self, seq_id: i32, position: i32) {
        if let Ok(lane) = self.lane(seq_id) {
            self.lanes[lane]
                .pending
                .retain(|row| row.position < position);
            for slot in &mut self.lanes[lane].slots {
                if slot.is_some_and(|value| value >= position) {
                    *slot = None;
                }
            }
        }
    }

    pub fn clear(&mut self, seq_id: i32) {
        if let Ok(lane) = self.lane(seq_id) {
            let _ = self.coreml.reset(lane);
            self.lanes[lane].pending.clear();
            self.lanes[lane].slots.fill(None);
        }
    }

    pub fn clear_all(&mut self) {
        for lane in 0..self.lanes.len() {
            self.clear(lane as i32);
        }
    }
}

fn rope(positions: &[i32], head_width: usize, base: f32) -> (Vec<u16>, Vec<u16>) {
    let mut cosine = Vec::with_capacity(positions.len() * head_width);
    let mut sine = Vec::with_capacity(positions.len() * head_width);
    for &position in positions {
        for axis in 0..head_width {
            let exponent = -(((axis % (head_width / 2)) * 2) as f32) / head_width as f32;
            let frequency = base.powf(exponent);
            let angle = position as f32 * frequency;
            cosine.push(f16::from_f32(angle.cos()).to_bits());
            sine.push(f16::from_f32(angle.sin()).to_bits());
        }
    }
    (cosine, sine)
}

fn file_sha256(path: &Path) -> Result<String> {
    let mut file =
        File::open(path).with_context(|| format!("open {} for SHA-256", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(hex::encode(digest.finalize()))
}
