//! DFlash target features across a split target.
//!
//! In a split, the draft lives on the final stage, which owns the output head
//! the draft borrows. The draft still reads the inputs of target layers that
//! earlier stages own, so every stage that owns one of the draft's target
//! layers captures that layer's input rows and appends them to its output
//! activation frame as an optional part, one part per layer. Native stage
//! execution carries unplanned optional parts through to its own output, so a
//! stage without this feature still forwards them unchanged; the final stage
//! reads them from its input frame and injects them beside the rows it
//! captured itself.
//!
//! The input of a stage's first layer is the activation it receives, which
//! its own graph never produces, so no stage captures it there. The stage that
//! ends on that layer forwards its own output hidden states as that layer's
//! rows instead; they are the same values. For a hyper-connection target,
//! whose boundary carries several residual streams, it forwards their mean,
//! which is what those graphs publish as the layer input.
//!
//! Rows travel as BF16: hidden states keep their range, and the draft only
//! proposes tokens the target then verifies, so feature precision can change
//! acceptance but never output.

use std::sync::{Arc, Mutex, Weak};

use anyhow::{Context, Result, anyhow, bail, ensure};
use skippy_ffi::Opaque;
use skippy_ffi::llama_draft::LlamaDraftApi;

use crate::{
    ACTIVATION_IDENTITY_BYTES, ACTIVATION_MAX_PARTS, ACTIVATION_PART_OPTIONAL, ActivationFrame,
    ActivationPartDesc, StageModel, StageSession,
};

use super::lock_recovering;

/// ggml's BF16 type id.
const GGML_TYPE_BF16: u32 = 30;
/// Identity prefix of a DFlash feature part; the layer id follows in decimal.
const FEATURE_IDENTITY_PREFIX: &[u8] = b"skippy.dflash.layer_inp.";

/// Activation part identity carrying the input rows of target layer `layer`.
pub fn dflash_feature_identity(layer: u32) -> [u8; ACTIVATION_IDENTITY_BYTES] {
    let mut identity = [0_u8; ACTIVATION_IDENTITY_BYTES];
    let suffix = layer.to_string();
    identity[..FEATURE_IDENTITY_PREFIX.len()].copy_from_slice(FEATURE_IDENTITY_PREFIX);
    identity[FEATURE_IDENTITY_PREFIX.len()..][..suffix.len()].copy_from_slice(suffix.as_bytes());
    identity
}

/// The target layer a DFlash feature part carries, or `None` for any other
/// part.
pub fn dflash_feature_layer(identity: &[u8; ACTIVATION_IDENTITY_BYTES]) -> Option<u32> {
    let suffix = identity.strip_prefix(FEATURE_IDENTITY_PREFIX)?;
    let end = suffix
        .iter()
        .position(|&byte| byte == 0)
        .unwrap_or(suffix.len());
    if end == 0 || suffix[end..].iter().any(|&byte| byte != 0) {
        return None;
    }
    std::str::from_utf8(&suffix[..end]).ok()?.parse().ok()
}

/// Whether a stage over `layer_start..layer_end` produces the input of
/// `layer` inside its own graph. A stage's first layer reads the activation it
/// received, except on stage 0, whose graph computes the token embeddings.
pub(crate) fn stage_captures_layer_input(layer: u32, layer_start: u32, layer_end: u32) -> bool {
    (layer_start..layer_end).contains(&layer) && (layer != layer_start || layer_start == 0)
}

/// The target layer whose input a stage over `layer_start..layer_end` sends
/// as its own output: the next stage's first layer, when the draft reads it.
pub(crate) fn stage_forwards_output_as(
    target_layers: &[u32],
    layer_start: u32,
    layer_end: u32,
) -> Option<u32> {
    (layer_end > layer_start && target_layers.contains(&layer_end)).then_some(layer_end)
}

/// The hidden-state rows of `frame`'s primary part, which are the input of
/// the layer after the producing stage's range.
///
/// A plain boundary is `[n_embd, tokens]` rows. A hyper-connection boundary
/// (glm5-next, deepseek4) is `[n_embd, streams, tokens]`, and the layer input
/// those families publish is the mean of the streams, computed here in the
/// order their graphs use: the streams summed in turn, then scaled by
/// `1 / streams`.
fn frame_hidden_rows(frame: &ActivationFrame, n_embd: usize) -> Result<Vec<f32>> {
    let part = frame
        .desc
        .parts()?
        .first()
        .copied()
        .context("activation frame has no primary part")?;
    let tokens = usize::try_from(frame.desc.token_count).context("frame tokens")?;
    let element = match part.ggml_type {
        crate::GGML_TYPE_F32 => 4,
        crate::GGML_TYPE_F16 | GGML_TYPE_BF16 => 2,
        other => bail!("primary activation part has ggml type {other}"),
    };
    let streams = match (part.rank, part.token_axis) {
        (2, 1) => 1,
        (3, 2) => usize::try_from(part.dimensions[1]).unwrap_or(0),
        _ => 0,
    };
    let packed = [element, n_embd * element, n_embd * streams * element].map(|v| v as u64);
    let rank = usize::try_from(part.rank).unwrap_or(0).min(packed.len());
    let contiguous = part.byte_strides[..rank].iter().all(|&stride| stride == 0)
        || part.byte_strides[..rank] == packed[..rank];
    ensure!(
        dflash_feature_layer(&part.identity).is_none()
            && streams > 0
            && usize::try_from(part.dimensions[0]).ok() == Some(n_embd)
            && contiguous,
        "primary activation part is not {n_embd}-wide contiguous hidden-state rows"
    );
    let start = usize::try_from(part.payload_offset).context("primary offset")?;
    let bytes = tokens * streams * n_embd * element;
    let payload = frame
        .payload
        .get(start..start.saturating_add(bytes))
        .filter(|payload| payload.len() == bytes)
        .ok_or_else(|| anyhow!("primary activation part is truncated"))?;
    let values: Vec<f32> = match part.ggml_type {
        crate::GGML_TYPE_F32 => payload
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes))
            .collect(),
        GGML_TYPE_BF16 => decode_bf16(payload),
        _ => payload
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| f16_to_f32(u16::from_le_bytes(*pair)))
            .collect(),
    };
    if streams == 1 {
        return Ok(values);
    }
    let scale = 1.0 / streams as f32;
    Ok(values
        .chunks_exact(streams * n_embd)
        .flat_map(|token| {
            let mut sum = token[..n_embd].to_vec();
            for stream in token.chunks_exact(n_embd).skip(1) {
                for (acc, value) in sum.iter_mut().zip(stream) {
                    *acc += value;
                }
            }
            sum.into_iter().map(move |value| value * scale)
        })
        .collect())
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let mantissa = u32::from(bits & 0x3ff);
    let magnitude = match (exponent, mantissa) {
        (0, 0) => 0,
        // Subnormal: value is mantissa * 2^-24.
        (0, _) => {
            let value = mantissa as f32 * f32::powi(2.0, -24);
            return if sign == 0 { value } else { -value };
        }
        (0x1f, _) => 0x7f80_0000 | (mantissa << 13),
        _ => ((exponent + 112) << 23) | (mantissa << 13),
    };
    f32::from_bits(sign | magnitude)
}

/// Whether `frame` carries DFlash feature rows for a later stage.
pub fn frame_has_dflash_features(frame: &ActivationFrame) -> bool {
    frame.desc.parts().is_ok_and(|parts| {
        parts
            .iter()
            .any(|part| dflash_feature_layer(&part.identity).is_some())
    })
}

/// Input rows of some target layers for consecutive token positions.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DFlashFeatureRows {
    n_embd: usize,
    rows: usize,
    /// `(layer, rows * n_embd values)`, in ascending layer order.
    layers: Vec<(u32, Vec<f32>)>,
}

impl DFlashFeatureRows {
    pub fn new(n_embd: usize, rows: usize) -> Self {
        Self {
            n_embd,
            rows,
            layers: Vec::new(),
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    pub fn layer_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.layers.iter().map(|(layer, _)| *layer)
    }

    /// Adds one layer's rows, replacing any earlier rows for that layer.
    pub fn insert(&mut self, layer: u32, values: Vec<f32>) -> Result<()> {
        ensure!(
            values.len() == self.rows * self.n_embd,
            "layer {layer} carries {} values, expected {} rows of {}",
            values.len(),
            self.rows,
            self.n_embd
        );
        match self.layers.binary_search_by_key(&layer, |(id, _)| *id) {
            Ok(index) => self.layers[index].1 = values,
            Err(index) => self.layers.insert(index, (layer, values)),
        }
        Ok(())
    }

    /// `rows` rows of `layer` starting at row `start`.
    pub(crate) fn rows_of(&self, layer: u32, start: usize, rows: usize) -> Option<&[f32]> {
        if start.checked_add(rows)? > self.rows {
            return None;
        }
        let (_, values) = self.layers.iter().find(|(id, _)| *id == layer)?;
        Some(&values[start * self.n_embd..(start + rows) * self.n_embd])
    }

    /// Keeps only the last `rows` rows of every layer.
    fn keep_trailing(&mut self, rows: usize) {
        if rows >= self.rows {
            return;
        }
        let drop_values = (self.rows - rows) * self.n_embd;
        for (_, values) in &mut self.layers {
            values.drain(..drop_values);
        }
        self.rows = rows;
    }

    /// Appends rows decoded after these, layer by layer. Layers missing from
    /// either side are dropped, since their rows no longer line up.
    fn extend(&mut self, later: DFlashFeatureRows) {
        if self.layers.is_empty() {
            *self = later;
            return;
        }
        if later.n_embd != self.n_embd {
            *self = later;
            return;
        }
        self.layers
            .retain(|(layer, _)| later.layer_ids().any(|id| id == *layer));
        for (layer, values) in &mut self.layers {
            if let Some((_, more)) = later.layers.iter().find(|(id, _)| id == layer) {
                values.extend_from_slice(more);
            }
        }
        self.rows += later.rows;
    }

    /// Reads every DFlash feature part of `frame`.
    pub fn from_frame(frame: &ActivationFrame) -> Result<Option<Self>> {
        let mut rows: Option<Self> = None;
        for part in frame.desc.parts()? {
            let Some(layer) = dflash_feature_layer(&part.identity) else {
                continue;
            };
            ensure!(
                part.ggml_type == GGML_TYPE_BF16 && part.rank == 2 && part.token_axis == 1,
                "DFlash feature part for layer {layer} has an unexpected layout"
            );
            let n_embd = usize::try_from(part.dimensions[0]).context("feature width")?;
            let count = usize::try_from(part.dimensions[1]).context("feature rows")?;
            let start = usize::try_from(part.payload_offset).context("feature offset")?;
            let bytes = usize::try_from(part.payload_bytes).context("feature bytes")?;
            let payload = frame
                .payload
                .get(start..start.saturating_add(bytes))
                .filter(|payload| payload.len() == n_embd * count * 2)
                .ok_or_else(|| anyhow!("DFlash feature part for layer {layer} is truncated"))?;
            let rows = rows.get_or_insert_with(|| Self::new(n_embd, count));
            ensure!(
                rows.n_embd == n_embd && rows.rows == count,
                "DFlash feature parts disagree on their shape"
            );
            rows.insert(layer, decode_bf16(payload))?;
        }
        Ok(rows)
    }

    /// Appends these rows to `frame` as optional parts. Layers the frame
    /// already carries are left as they are.
    pub fn append_to_frame(&self, frame: &mut ActivationFrame) -> Result<()> {
        if self.layers.is_empty() {
            return Ok(());
        }
        ensure!(
            u32::try_from(self.rows).ok() == Some(frame.desc.token_count),
            "{} DFlash feature rows do not match a {}-token activation frame",
            self.rows,
            frame.desc.token_count
        );
        let existing = frame.desc.parts()?.to_vec();
        let mut count = existing.len();
        for (layer, values) in &self.layers {
            let identity = dflash_feature_identity(*layer);
            if existing.iter().any(|part| part.identity == identity) {
                continue;
            }
            if count >= ACTIVATION_MAX_PARTS {
                bail!("no activation part left for DFlash layer {layer}");
            }
            let offset = frame.payload.len();
            encode_bf16(values, &mut frame.payload);
            let mut part = ActivationPartDesc {
                identity,
                ggml_type: GGML_TYPE_BF16,
                rank: 2,
                token_axis: 1,
                flags: ACTIVATION_PART_OPTIONAL,
                payload_offset: u64::try_from(offset).context("feature offset")?,
                payload_bytes: u64::try_from(values.len() * 2).context("feature bytes")?,
                ..ActivationPartDesc::default()
            };
            part.dimensions[0] = i64::try_from(self.n_embd).context("feature width")?;
            part.dimensions[1] = i64::try_from(self.rows).context("feature rows")?;
            part.byte_strides[0] = 2;
            part.byte_strides[1] = u64::try_from(self.n_embd * 2).context("feature stride")?;
            frame.desc.parts[count] = part;
            count += 1;
        }
        frame.desc.part_count = u32::try_from(count).context("part count")?;
        frame.desc.payload_bytes = u64::try_from(frame.payload.len()).context("payload bytes")?;
        Ok(())
    }
}

fn encode_bf16(values: &[f32], out: &mut Vec<u8>) {
    out.reserve(values.len() * 2);
    for value in values {
        let bits = value.to_bits();
        // Round to nearest even; NaN keeps a quiet payload.
        let rounded = if value.is_nan() {
            (bits >> 16) | 0x40
        } else {
            (bits + 0x7fff + ((bits >> 16) & 1)) >> 16
        };
        out.extend_from_slice(&(rounded as u16).to_le_bytes());
    }
}

fn decode_bf16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| f32::from_bits(u32::from(u16::from_le_bytes(*pair)) << 16))
        .collect()
}

/// Captures the inputs of this stage's DFlash target layers on a stage that
/// carries no draft.
pub(crate) struct DFlashCapture {
    api: &'static LlamaDraftApi,
    target_ctx: *mut Opaque,
    /// Layers whose input this stage's graph produces.
    layers: Vec<u32>,
    /// The layer whose input is this stage's output, when the draft reads it.
    output_layer: Option<u32>,
    n_embd: usize,
    n_batch: usize,
}

// Only touched behind the owning Mutex, and Skippy serializes target-context
// use on the runtime worker.
unsafe impl Send for DFlashCapture {}

impl DFlashCapture {
    fn read(&self, first_row: usize, rows: usize) -> Result<DFlashFeatureRows> {
        if self.layers.is_empty() {
            return Ok(DFlashFeatureRows::new(self.n_embd, rows));
        }
        ensure!(
            first_row + rows <= self.n_batch,
            "{rows} rows from row {first_row} exceed the capture buffer of {} rows",
            self.n_batch
        );
        let mut captured = DFlashFeatureRows::new(self.n_embd, rows);
        for &layer in &self.layers {
            let data = unsafe { (self.api.get_embeddings_layer_inp)(self.target_ctx, layer) };
            if data.is_null() {
                bail!("target layer {layer} input was not captured");
            }
            let all = unsafe {
                std::slice::from_raw_parts(data.cast_const(), (first_row + rows) * self.n_embd)
            };
            captured.insert(layer, all[first_row * self.n_embd..].to_vec())?;
        }
        Ok(captured)
    }
}

impl Drop for DFlashCapture {
    fn drop(&mut self) {
        for &layer in &self.layers {
            unsafe { (self.api.set_embeddings_layer_inp)(self.target_ctx, layer, false) };
        }
    }
}

pub(crate) type SharedDFlashCapture = Arc<Mutex<DFlashCapture>>;

/// A session's split-DFlash state: rows captured for the next stage, and rows
/// that arrived from earlier stages for the draft.
#[derive(Default)]
pub(crate) struct DFlashSplitSession {
    capture: Option<Weak<Mutex<DFlashCapture>>>,
    captured: Option<DFlashFeatureRows>,
    /// The layer, and its width, whose input this stage's output frames carry.
    output_layer: Option<(u32, usize)>,
    /// Whether this stage's graph captures any target layer itself.
    captures_layers: bool,
    /// Rows from earlier stages, and how many of them decodes have consumed.
    pub(crate) remote: Option<(DFlashFeatureRows, usize)>,
    /// Token rows whose captured features this stage could not forward.
    pub(crate) rows_dropped: u64,
}

impl DFlashSplitSession {
    pub(crate) fn capturing(capture: &SharedDFlashCapture) -> Self {
        let (output_layer, captures_layers) = {
            let capture = lock_recovering(capture, |_| {});
            (
                capture.output_layer.map(|layer| (layer, capture.n_embd)),
                !capture.layers.is_empty(),
            )
        };
        Self {
            capture: Some(Arc::downgrade(capture)),
            output_layer,
            captures_layers,
            ..Self::default()
        }
    }

    /// A session holding `rows` captured but not yet forwarded.
    #[cfg(test)]
    pub(crate) fn with_captured(rows: DFlashFeatureRows) -> Self {
        Self {
            captured: Some(rows),
            ..Self::default()
        }
    }

    /// Records this session's rows of the last target decode.
    pub(crate) fn after_decode(&mut self, first_row: usize, rows: usize) {
        let Some(capture) = self.capture.as_ref().and_then(Weak::upgrade) else {
            return;
        };
        let read = lock_recovering(&capture, |_| {}).read(first_row, rows);
        match read {
            Ok(read) => match self.captured.as_mut() {
                Some(captured) => captured.extend(read),
                None => self.captured = Some(read),
            },
            Err(error) => {
                crate::write_native_log_note(format!(
                    "dflash feature capture failed, {rows} rows not forwarded: {error:#}"
                ));
                self.rows_dropped += u64::try_from(rows).unwrap_or(u64::MAX);
                self.captured = None;
            }
        }
    }

    pub(crate) fn clear(&mut self) {
        self.captured = None;
        self.remote = None;
    }
}

impl StageSession {
    /// Hands the draft the feature rows an earlier stage sent with the input
    /// of this session's next decode.
    pub fn dflash_set_remote_features(&mut self, rows: Option<DFlashFeatureRows>) {
        if self.dflash.is_some() {
            self.dflash_split
                .get_or_insert_with(Default::default)
                .remote = rows.map(|rows| (rows, 0));
        }
    }

    /// Appends the feature rows this stage captured for the decode that
    /// produced `frame` to it, for the final stage's draft. Rows captured by
    /// an earlier decode whose frame left without them are dropped: the frame
    /// carries the rows of its own tokens, which are the last ones captured.
    /// When the next stage starts on a target layer, the frame's own hidden
    /// states go along as that layer's rows.
    pub fn dflash_forward_captured(&mut self, frame: &mut ActivationFrame) -> Result<()> {
        let Some(split) = self.dflash_split.as_mut() else {
            return Ok(());
        };
        let tokens = usize::try_from(frame.desc.token_count).context("frame tokens")?;
        let mut rows = split.captured.take();
        if let Some(rows) = rows.as_mut() {
            rows.keep_trailing(tokens);
        }
        // Rows a failed capture never recorded leave without this stage's
        // layers; the final stage's draft finds them missing.
        if split.captures_layers {
            let carried = rows
                .as_ref()
                .map_or(0, |rows| if rows.is_empty() { 0 } else { rows.rows() });
            split.rows_dropped += u64::try_from(tokens.saturating_sub(carried)).unwrap_or(0);
        }
        let mut output_error = None;
        if let Some((layer, n_embd)) = split.output_layer {
            let inserted = frame_hidden_rows(frame, n_embd).and_then(|values| {
                rows.get_or_insert_with(|| DFlashFeatureRows::new(n_embd, tokens))
                    .insert(layer, values)
            });
            if let Err(error) = inserted {
                output_error = Some(error.context(format!("forward layer {layer} input")));
            }
        }
        if let Some(rows) = rows {
            rows.append_to_frame(frame)?;
        }
        output_error.map_or(Ok(()), Err)
    }

    /// Token rows whose captured DFlash features this stage could not forward
    /// since the session was last activated.
    pub fn dflash_rows_dropped(&self) -> u64 {
        self.dflash_split
            .as_ref()
            .map_or(0, |split| split.rows_dropped)
    }

    /// Whether this session feeds a split DFlash pipeline, as a draft or as a
    /// capturing stage.
    pub fn dflash_split_active(&self) -> bool {
        self.dflash.is_some() || self.dflash_split.is_some()
    }
}

impl StageModel {
    /// Captures the inputs of the DFlash target layers this stage owns so its
    /// sessions can forward them to the final stage's draft, plus its output
    /// when the next stage starts on a target layer. Returns the forwarded
    /// layers. Attach before creating serving sessions.
    pub fn attach_dflash_capture(
        &mut self,
        target_layers: &[u32],
        layer_start: u32,
        layer_end: u32,
    ) -> Result<Vec<u32>> {
        if self.dflash_draft().is_some() || self.dflash_capture().is_some() {
            bail!("a DFlash draft or capture is already attached");
        }
        let layers = target_layers
            .iter()
            .copied()
            .filter(|&layer| stage_captures_layer_input(layer, layer_start, layer_end))
            .collect::<Vec<_>>();
        let output_layer = stage_forwards_output_as(target_layers, layer_start, layer_end);
        if layers.is_empty() && output_layer.is_none() {
            return Ok(layers);
        }
        let api = skippy_ffi::llama_draft::llama_draft_api()
            .ok_or_else(|| anyhow!("native runtime does not export the draft-context API"))?;
        let probe = self.create_session()?;
        let target_ctx = unsafe { skippy_ffi::skippy_session_llama_context(probe.raw) };
        if target_ctx.is_null() {
            bail!("target session has no llama context");
        }
        let target_model = unsafe { (api.get_model)(target_ctx) };
        let n_embd = usize::try_from(unsafe { (api.model_n_embd)(target_model) })
            .ok()
            .filter(|n| *n > 0)
            .context("target n_embd")?;
        let n_batch = usize::try_from(unsafe { (api.n_batch)(target_ctx) }).context("n_batch")?;
        let n_layer = u32::try_from(unsafe { (api.model_n_layer)(target_model) }).unwrap_or(0);
        if let Some(&layer) = layers.iter().find(|&&layer| layer >= n_layer) {
            bail!("DFlash target layer {layer} is outside the target's {n_layer} layers");
        }
        for &layer in &layers {
            unsafe { (api.set_embeddings_layer_inp)(target_ctx, layer, true) };
        }
        let capture = DFlashCapture {
            api,
            target_ctx,
            layers: layers.clone(),
            output_layer,
            n_embd,
            n_batch,
        };
        if !layers.is_empty() {
            super::refresh_target_program(&probe)?;
        }
        drop(probe);
        self.install_dflash_capture(Arc::new(Mutex::new(capture)))?;
        crate::write_native_log_note(format!(
            "dflash split capture attached layers={layers:?} output_layer={output_layer:?}"
        ));
        Ok(layers.into_iter().chain(output_layer).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ACTIVATION_FRAME_VERSION, ActivationDesc};

    fn frame(tokens: u32) -> ActivationFrame {
        let mut part = ActivationPartDesc {
            identity: [7; ACTIVATION_IDENTITY_BYTES],
            ggml_type: crate::GGML_TYPE_F32,
            rank: 2,
            token_axis: 1,
            payload_bytes: u64::from(tokens) * 8,
            ..ActivationPartDesc::default()
        };
        part.dimensions[0] = 2;
        part.dimensions[1] = i64::from(tokens);
        let mut parts = [ActivationPartDesc::default(); ACTIVATION_MAX_PARTS];
        parts[0] = part;
        ActivationFrame {
            desc: ActivationDesc {
                version: ACTIVATION_FRAME_VERSION,
                producer_stage_index: 0,
                layer_start: 0,
                layer_end: 4,
                token_count: tokens,
                sequence_count: 1,
                part_count: 1,
                payload_bytes: u64::from(tokens) * 8,
                frontier_identity: [0; ACTIVATION_IDENTITY_BYTES],
                parts,
            },
            payload: vec![1; tokens as usize * 8],
        }
    }

    #[test]
    fn feature_identities_round_trip_and_ignore_other_parts() {
        for layer in [0, 6, 43, 4095] {
            assert_eq!(
                dflash_feature_layer(&dflash_feature_identity(layer)),
                Some(layer)
            );
        }
        assert_eq!(dflash_feature_layer(&[7; ACTIVATION_IDENTITY_BYTES]), None);
        let mut bare = [0; ACTIVATION_IDENTITY_BYTES];
        bare[..FEATURE_IDENTITY_PREFIX.len()].copy_from_slice(FEATURE_IDENTITY_PREFIX);
        assert_eq!(dflash_feature_layer(&bare), None);
    }

    #[test]
    fn feature_rows_ride_a_frame_as_optional_bf16_parts() {
        let mut rows = DFlashFeatureRows::new(2, 3);
        rows.insert(15, vec![1.0, -2.0, 0.5, 3.0, 1024.0, -0.25])
            .unwrap();
        rows.insert(6, vec![0.0; 6]).unwrap();
        let mut frame = frame(3);
        let original = frame.payload.clone();

        rows.append_to_frame(&mut frame).unwrap();

        assert_eq!(frame.desc.part_count, 3);
        assert_eq!(&frame.payload[..original.len()], &original[..]);
        let parts = frame.desc.parts().unwrap();
        assert!(parts[1..].iter().all(|part| part.is_optional()));
        assert_eq!(
            parts[1..]
                .iter()
                .filter_map(|part| dflash_feature_layer(&part.identity))
                .collect::<Vec<_>>(),
            vec![6, 15]
        );
        // These values are exact in BF16.
        assert_eq!(DFlashFeatureRows::from_frame(&frame).unwrap(), Some(rows));
    }

    #[test]
    fn a_frame_without_feature_parts_reads_as_none() {
        assert_eq!(DFlashFeatureRows::from_frame(&frame(2)).unwrap(), None);
    }

    #[test]
    fn appending_twice_keeps_the_first_rows_of_a_layer() {
        let mut first = DFlashFeatureRows::new(2, 1);
        first.insert(3, vec![1.0, 2.0]).unwrap();
        let mut second = DFlashFeatureRows::new(2, 1);
        second.insert(3, vec![5.0, 6.0]).unwrap();
        let mut frame = frame(1);

        first.append_to_frame(&mut frame).unwrap();
        second.append_to_frame(&mut frame).unwrap();

        assert_eq!(frame.desc.part_count, 2);
        assert_eq!(DFlashFeatureRows::from_frame(&frame).unwrap(), Some(first));
    }

    #[test]
    fn rows_must_match_the_frame_tokens() {
        let mut rows = DFlashFeatureRows::new(2, 2);
        rows.insert(1, vec![0.0; 4]).unwrap();
        assert!(rows.append_to_frame(&mut frame(3)).is_err());
    }

    #[test]
    fn bf16_rounds_to_nearest_even() {
        let mut bytes = Vec::new();
        let values = [1.0_f32, 1.0 + f32::EPSILON, -3.140625, 65504.0, 1.0e30];
        encode_bf16(&values, &mut bytes);
        let decoded = decode_bf16(&bytes);
        assert_eq!(decoded[0], 1.0);
        assert_eq!(decoded[1], 1.0);
        assert_eq!(decoded[2], -3.140625);
        assert!((decoded[3] - 65504.0).abs() / 65504.0 < 0.01);
        assert!((decoded[4] - 1.0e30).abs() / 1.0e30 < 0.01);
    }

    fn session_with_captured(rows: DFlashFeatureRows) -> StageSession {
        StageSession {
            raw: std::ptr::null_mut(),
            token_count: 0,
            terminal_stage: false,
            batched_activation_exports: true,
            dflash: None,
            dflash_split: Some(Box::new(DFlashSplitSession::with_captured(rows))),
        }
    }

    #[test]
    fn a_frame_carries_the_rows_of_its_own_tokens() {
        // A one-token decode whose frame left without its row, then a
        // three-token verify window: the window's frame must carry the last
        // three rows, not fail on four.
        let mut rows = DFlashFeatureRows::new(1, 4);
        rows.insert(2, vec![9.0, 1.0, 2.0, 3.0]).unwrap();
        let mut session = session_with_captured(rows);
        let mut window = frame(3);

        session.dflash_forward_captured(&mut window).unwrap();

        let forwarded = DFlashFeatureRows::from_frame(&window).unwrap().unwrap();
        assert_eq!(forwarded.rows(), 3);
        assert_eq!(forwarded.rows_of(2, 0, 3), Some(&[1.0, 2.0, 3.0][..]));
        // The rows were handed off; the next frame starts empty.
        let mut next = frame(1);
        session.dflash_forward_captured(&mut next).unwrap();
        assert_eq!(DFlashFeatureRows::from_frame(&next).unwrap(), None);
    }

    #[test]
    fn later_rows_extend_each_shared_layer() {
        let mut rows = DFlashFeatureRows::new(1, 1);
        rows.insert(2, vec![1.0]).unwrap();
        rows.insert(4, vec![2.0]).unwrap();
        let mut later = DFlashFeatureRows::new(1, 2);
        later.insert(2, vec![3.0, 5.0]).unwrap();

        rows.extend(later);

        assert_eq!(rows.rows(), 3);
        assert_eq!(rows.layer_ids().collect::<Vec<_>>(), vec![2]);
        assert_eq!(rows.rows_of(2, 1, 2), Some(&[3.0, 5.0][..]));
        assert_eq!(rows.rows_of(2, 2, 2), None);
    }

    /// The Qwen3.6-27B DFlash draft's target layers.
    const TARGETS: [u32; 5] = [2, 17, 32, 47, 62];

    fn captured_by(layer_start: u32, layer_end: u32) -> Vec<u32> {
        TARGETS
            .iter()
            .copied()
            .filter(|&layer| stage_captures_layer_input(layer, layer_start, layer_end))
            .collect()
    }

    #[test]
    fn a_cut_on_a_target_layer_moves_it_to_the_stage_before() {
        // 64 layers cut at 32: the final stage's first layer is a target, so
        // stage 0 sends its output as that layer's input.
        assert_eq!(captured_by(0, 32), vec![2, 17]);
        assert_eq!(stage_forwards_output_as(&TARGETS, 0, 32), Some(32));
        assert_eq!(captured_by(32, 64), vec![47, 62]);
        assert_eq!(stage_forwards_output_as(&TARGETS, 32, 64), None);
    }

    #[test]
    fn a_middle_stage_starting_on_a_target_layer_leaves_it_to_its_upstream() {
        assert_eq!(captured_by(0, 17), vec![2]);
        assert_eq!(stage_forwards_output_as(&TARGETS, 0, 17), Some(17));
        assert_eq!(captured_by(17, 40), vec![32]);
        assert_eq!(stage_forwards_output_as(&TARGETS, 17, 40), None);
        assert_eq!(captured_by(40, 64), vec![47, 62]);
    }

    #[test]
    fn every_target_layer_has_exactly_one_source_for_any_cut() {
        for cut in 1..64 {
            for layer in TARGETS {
                let sources = [
                    stage_captures_layer_input(layer, 0, cut),
                    stage_forwards_output_as(&TARGETS, 0, cut) == Some(layer),
                    stage_captures_layer_input(layer, cut, 64),
                ];
                assert_eq!(
                    sources.iter().filter(|&&source| source).count(),
                    1,
                    "layer {layer} at cut {cut}"
                );
            }
        }
    }

    #[test]
    fn stage_zero_captures_the_embedding_input_of_layer_zero() {
        assert!(stage_captures_layer_input(0, 0, 8));
        assert!(!stage_captures_layer_input(8, 8, 16));
    }

    fn hidden_frame(ggml_type: u32, rows: &[f32]) -> ActivationFrame {
        let mut frame = frame(2);
        let part = &mut frame.desc.parts[0];
        part.ggml_type = ggml_type;
        let element = if ggml_type == crate::GGML_TYPE_F32 {
            4
        } else {
            2
        };
        part.byte_strides = [element, 2 * element, 0, 0];
        frame.payload = match ggml_type {
            crate::GGML_TYPE_F32 => rows.iter().flat_map(|value| value.to_le_bytes()).collect(),
            GGML_TYPE_BF16 => {
                let mut bytes = Vec::new();
                encode_bf16(rows, &mut bytes);
                bytes
            }
            // 1.0, -2.0, 0.5, 65504.0 in F16.
            _ => [0x3c00_u16, 0xc000, 0x3800, 0x7bff]
                .iter()
                .flat_map(|bits| bits.to_le_bytes())
                .collect(),
        };
        frame.desc.parts[0].payload_bytes = frame.payload.len() as u64;
        frame.desc.payload_bytes = frame.payload.len() as u64;
        frame
    }

    #[test]
    fn a_stage_ending_on_a_target_layer_forwards_its_output_rows() {
        let hidden = [1.0, -2.0, 0.5, 65504.0];
        for ggml_type in [crate::GGML_TYPE_F32, crate::GGML_TYPE_F16, GGML_TYPE_BF16] {
            let mut captured = DFlashFeatureRows::new(2, 2);
            captured.insert(17, vec![3.0; 4]).unwrap();
            let mut split = DFlashSplitSession::with_captured(captured);
            split.output_layer = Some((32, 2));
            let mut session = session_with_captured(DFlashFeatureRows::default());
            session.dflash_split = Some(Box::new(split));
            let mut output = hidden_frame(ggml_type, &hidden);

            session.dflash_forward_captured(&mut output).unwrap();

            let forwarded = DFlashFeatureRows::from_frame(&output).unwrap().unwrap();
            assert_eq!(forwarded.layer_ids().collect::<Vec<_>>(), vec![17, 32]);
            let layer = forwarded.rows_of(32, 0, 2).unwrap();
            for (got, want) in layer.iter().zip(hidden) {
                assert!(
                    (got - want).abs() <= want.abs() / 128.0,
                    "{ggml_type}: {got} vs {want}"
                );
            }
        }
    }

    #[test]
    fn a_stage_without_native_captures_still_forwards_its_output() {
        let split = DFlashSplitSession {
            output_layer: Some((32, 2)),
            ..DFlashSplitSession::default()
        };
        let mut session = session_with_captured(DFlashFeatureRows::default());
        session.dflash_split = Some(Box::new(split));
        let mut output = hidden_frame(crate::GGML_TYPE_F32, &[1.0, 2.0, 3.0, 4.0]);

        session.dflash_forward_captured(&mut output).unwrap();

        let forwarded = DFlashFeatureRows::from_frame(&output).unwrap().unwrap();
        assert_eq!(forwarded.rows_of(32, 0, 2), Some(&[1.0, 2.0, 3.0, 4.0][..]));
    }

    #[test]
    fn a_hyper_connection_boundary_forwards_the_mean_of_its_streams() {
        // Two tokens of two 2-wide streams: [n_embd, streams, tokens].
        let streams = [1.0_f32, 2.0, 3.0, 6.0, -1.0, 0.5, 5.0, 1.5];
        let mut output = frame(2);
        let part = &mut output.desc.parts[0];
        part.rank = 3;
        part.token_axis = 2;
        part.dimensions = [2, 2, 2, 0];
        part.byte_strides = [4, 8, 16, 0];
        output.payload = streams.iter().flat_map(|v| v.to_le_bytes()).collect();
        output.desc.parts[0].payload_bytes = output.payload.len() as u64;
        output.desc.payload_bytes = output.payload.len() as u64;

        assert_eq!(
            frame_hidden_rows(&output, 2).unwrap(),
            vec![2.0, 4.0, 2.0, 1.0]
        );
        // Strides that are not packed streams are refused.
        output.desc.parts[0].byte_strides = [4, 16, 8, 0];
        assert!(frame_hidden_rows(&output, 2).is_err());
    }

    #[test]
    fn a_malformed_output_is_an_error_not_a_panic() {
        let split = DFlashSplitSession {
            output_layer: Some((32, 3)),
            ..DFlashSplitSession::default()
        };
        let mut session = session_with_captured(DFlashFeatureRows::default());
        session.dflash_split = Some(Box::new(split));
        let mut output = hidden_frame(crate::GGML_TYPE_F32, &[1.0, 2.0, 3.0, 4.0]);

        assert!(session.dflash_forward_captured(&mut output).is_err());
        assert_eq!(DFlashFeatureRows::from_frame(&output).unwrap(), None);
    }

    #[test]
    fn f16_decodes_normals_subnormals_and_specials() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0001), f32::powi(2.0, -24));
        assert_eq!(f16_to_f32(0x8001), -f32::powi(2.0, -24));
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert!(f16_to_f32(0x7e00).is_nan());
    }
}
