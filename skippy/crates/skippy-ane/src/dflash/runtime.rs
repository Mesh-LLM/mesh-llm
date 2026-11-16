//! Stateful draft execution and lane state for the paired DFlash graph.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use half::f16;
use serde::Deserialize;
use sha2::{Digest, Sha256};

const WIDTH: usize = 2048;
const TAPS: usize = 8;
const HEAD: usize = 128;
const VOCAB: usize = 248_320;
const ROWS: usize = 7;
const MASK_TOKEN: i32 = 248_077;
const ROPE_BASE: f32 = 10_000_000.0;

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
    draft_sha256: String,
    embeddings_sha256: String,
    context: usize,
    rows: usize,
    token_output: bool,
    taps: Vec<u32>,
}

/// One compiled model and one independent Core ML state per Skippy lane.
pub struct AneDraft {
    coreml: CoreMlDraft,
    embeddings: File,
    context: usize,
    lanes: Vec<Lane>,
}

impl AneDraft {
    pub fn from_env(lanes: usize, n_vocab: usize, draft_path: &Path) -> Result<Option<Self>> {
        let Ok(model) = std::env::var("SKIPPY_DFLASH_ANE_MODEL") else {
            return Ok(None);
        };
        let embeddings = std::env::var("SKIPPY_DFLASH_ANE_EMBEDDINGS")
            .context("SKIPPY_DFLASH_ANE_EMBEDDINGS must name the paired target embeddings")?;
        let metadata_path = Path::new(&model).with_extension("json");
        let metadata: PackageMetadata = serde_json::from_slice(
            &std::fs::read(&metadata_path)
                .with_context(|| format!("read ANE draft metadata {}", metadata_path.display()))?,
        )
        .with_context(|| format!("parse ANE draft metadata {}", metadata_path.display()))?;
        let context = metadata.context;
        if n_vocab != VOCAB || context < ROWS || lanes == 0 {
            bail!("ANE DFlash package requires the paired Qwen3.6 vocabulary and context");
        }
        if metadata.rows != ROWS
            || !metadata.token_output
            || metadata.taps != [2, 7, 12, 17, 23, 28, 33, 38]
        {
            bail!("ANE DFlash package metadata does not match the paired draft graph");
        }
        if file_sha256(draft_path)? != metadata.draft_sha256 {
            bail!("ANE DFlash package was generated from a different draft GGUF");
        }
        if file_sha256(Path::new(&embeddings))? != metadata.embeddings_sha256 {
            bail!("ANE DFlash target embeddings do not match the package metadata");
        }
        let embeddings = File::open(&embeddings)
            .with_context(|| format!("open ANE target embeddings {embeddings}"))?;
        if embeddings.metadata()?.len() != (VOCAB * WIDTH * 2) as u64 {
            bail!("ANE target embeddings have the wrong size");
        }
        let coreml = CoreMlDraft::open(Path::new(&model), lanes, context, ROWS)?;
        Ok(Some(Self {
            coreml,
            embeddings,
            context,
            lanes: (0..lanes)
                .map(|_| Lane {
                    pending: Vec::new(),
                    slots: vec![None; context],
                })
                .collect(),
        }))
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
        let width = TAPS * WIDTH;
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
        while self.lanes[lane].pending.len() > ROWS {
            let rows = self.lanes[lane].pending.drain(..ROWS).collect::<Vec<_>>();
            let after = rows.last().expect("nonempty chunk").position + 1;
            self.run(lane, after, MASK_TOKEN, &rows)?;
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
        if max_tokens > ROWS - 1 {
            bail!("ANE DFlash package supports at most six proposals");
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
            .any(|&token| token < 0 || token as usize >= VOCAB)
        {
            bail!("ANE DFlash returned a token outside the paired vocabulary");
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
        let mut features = vec![0u16; TAPS * WIDTH * ROWS];
        let mut write = vec![0u16; ROWS * context];
        let mut ctx_positions = [0i32; ROWS];
        let mut slots = self.lanes[lane].slots.clone();
        for (row, feature) in rows.iter().enumerate() {
            let position =
                usize::try_from(feature.position).context("negative ANE feature position")?;
            let slot = position % context;
            ctx_positions[row] = feature.position;
            write[row * context + slot] = f16::ONE.to_bits();
            slots[slot] = Some(feature.position);
            for (channel, &value) in feature.values.iter().enumerate() {
                features[channel * ROWS + row] = f16::from_f32(value).to_bits();
            }
        }
        let (ctx_cos, ctx_sin) = rope(&ctx_positions);
        let query_positions = std::array::from_fn(|index| anchor_position + index as i32);
        let (q_cos, q_sin) = rope(&query_positions);
        let mut mask = vec![f16::MIN.to_bits(); ROWS * (context + ROWS)];
        for row in 0..ROWS {
            let offset = row * (context + ROWS);
            for (slot, position) in slots.iter().enumerate() {
                if position.is_some_and(|position| {
                    position < anchor_position && position >= anchor_position - context as i32
                }) {
                    mask[offset + slot] = f16::ZERO.to_bits();
                }
            }
            mask[offset + context..offset + context + ROWS].fill(f16::ZERO.to_bits());
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
            .filter(|&token| token < VOCAB)
            .ok_or_else(|| anyhow!("ANE DFlash anchor token {token} is invalid"))?;
        self.embeddings
            .seek(SeekFrom::Start((token * WIDTH * 2) as u64))?;
        let mut bytes = [0u8; WIDTH * 2];
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

fn rope(positions: &[i32; ROWS]) -> (Vec<u16>, Vec<u16>) {
    let mut cosine = Vec::with_capacity(ROWS * HEAD);
    let mut sine = Vec::with_capacity(ROWS * HEAD);
    for &position in positions {
        for axis in 0..HEAD {
            let exponent = -(((axis % (HEAD / 2)) * 2) as f32) / HEAD as f32;
            let frequency = ROPE_BASE.powf(exponent);
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
