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
//! its own graph never produces, so no stage captures it there. The stage
//! that receives it already holds those rows: the final stage hands its input
//! activation to the draft as that layer's rows, and a middle stage forwards
//! it as a feature part. The boundary layer therefore never crosses a link
//! twice, once as the activation and again as a feature.
//!
//! Rows travel in a compact encoding; see `feature_wire`.

use std::sync::{Arc, Mutex, Weak};

use anyhow::{Context, Result, anyhow, bail, ensure};
use skippy_ffi::Opaque;
use skippy_ffi::llama_draft::LlamaDraftApi;

use crate::{
    ACTIVATION_MAX_PARTS, ACTIVATION_PART_OPTIONAL, ActivationFrame, ActivationPartDesc,
    StageModel, StageSession,
};

use super::feature_wire::{
    DFlashFeatureEncoding, GGML_TYPE_BF16, decode_bf16, decode_q8_rows, dflash_feature_identity,
    dflash_feature_layer, dflash_feature_part, encode_bf16, encode_q8_rows, f16_to_f32,
    q8_feature_identity,
};
use super::lock_recovering;

/// Whether a stage over `layer_start..layer_end` produces the input of
/// `layer` inside its own graph. A stage's first layer reads the activation it
/// received, except on stage 0, whose graph computes the token embeddings.
pub(crate) fn stage_captures_layer_input(layer: u32, layer_start: u32, layer_end: u32) -> bool {
    (layer_start..layer_end).contains(&layer) && (layer != layer_start || layer_start == 0)
}

/// The target layer whose input is the activation a stage starting at
/// `layer_start` receives, when the draft reads it.
pub(crate) fn stage_receives_layer_input(target_layers: &[u32], layer_start: u32) -> Option<u32> {
    (layer_start > 0 && target_layers.contains(&layer_start)).then_some(layer_start)
}

/// The hidden-state rows of `frame`'s primary part: the activation a stage
/// receives, which is the input of its first layer.
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
    let contiguous = part.byte_strides[..2] == [0, 0]
        || part.byte_strides[..2] == [element as u64, (n_embd * element) as u64];
    ensure!(
        dflash_feature_layer(&part.identity).is_none()
            && part.rank == 2
            && part.token_axis == 1
            && usize::try_from(part.dimensions[0]).ok() == Some(n_embd)
            && contiguous,
        "primary activation part is not {n_embd}-wide contiguous hidden-state rows"
    );
    let start = usize::try_from(part.payload_offset).context("primary offset")?;
    let bytes = tokens * n_embd * element;
    let payload = frame
        .payload
        .get(start..start.saturating_add(bytes))
        .filter(|payload| payload.len() == bytes)
        .ok_or_else(|| anyhow!("primary activation part is truncated"))?;
    Ok(match part.ggml_type {
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
    })
}

/// Whether `frame` carries DFlash feature rows for a later stage.
pub fn frame_has_dflash_features(frame: &ActivationFrame) -> bool {
    frame.desc.parts().is_ok_and(|parts| {
        parts
            .iter()
            .any(|part| dflash_feature_layer(&part.identity).is_some())
    })
}

/// Payload bytes of the DFlash feature parts `frame` carries. Activation
/// codecs pass these parts through, so they cross the link at this size.
pub fn dflash_feature_payload_bytes(frame: &ActivationFrame) -> usize {
    frame.desc.parts().map_or(0, |parts| {
        parts
            .iter()
            .filter(|part| dflash_feature_layer(&part.identity).is_some())
            .map(|part| usize::try_from(part.payload_bytes).unwrap_or(usize::MAX))
            .fold(0, usize::saturating_add)
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

    /// Reads every DFlash feature part of `frame`, in either encoding.
    pub fn from_frame(frame: &ActivationFrame) -> Result<Option<Self>> {
        let mut rows: Option<Self> = None;
        for part in frame.desc.parts()? {
            let Some((layer, encoding, width)) = dflash_feature_part(&part.identity) else {
                continue;
            };
            let count = usize::try_from(part.dimensions[1]).context("feature rows")?;
            let (expected_type, n_embd) = match encoding {
                DFlashFeatureEncoding::Bf16 => (
                    GGML_TYPE_BF16,
                    usize::try_from(part.dimensions[0]).context("feature width")?,
                ),
                DFlashFeatureEncoding::Q8 => {
                    let n_embd = width.context("Q8 feature part names no width")?;
                    let words = usize::try_from(part.dimensions[0]).context("feature words")?;
                    ensure!(
                        words * 4 == encoding.row_bytes(n_embd),
                        "Q8 feature part for layer {layer} has {words} words a row, \
                         not the {} bytes of {n_embd} values",
                        encoding.row_bytes(n_embd)
                    );
                    (crate::GGML_TYPE_I32, n_embd)
                }
            };
            ensure!(
                part.ggml_type == expected_type && part.rank == 2 && part.token_axis == 1,
                "DFlash feature part for layer {layer} has an unexpected layout"
            );
            let start = usize::try_from(part.payload_offset).context("feature offset")?;
            let bytes = usize::try_from(part.payload_bytes).context("feature bytes")?;
            let payload = frame
                .payload
                .get(start..start.saturating_add(bytes))
                .filter(|payload| payload.len() == encoding.row_bytes(n_embd) * count)
                .ok_or_else(|| anyhow!("DFlash feature part for layer {layer} is truncated"))?;
            let rows = rows.get_or_insert_with(|| Self::new(n_embd, count));
            ensure!(
                rows.n_embd == n_embd && rows.rows == count,
                "DFlash feature parts disagree on their shape"
            );
            let values = match encoding {
                DFlashFeatureEncoding::Bf16 => decode_bf16(payload),
                DFlashFeatureEncoding::Q8 => decode_q8_rows(payload, n_embd),
            };
            rows.insert(layer, values)?;
        }
        Ok(rows)
    }

    /// Appends these rows to `frame` as optional parts in this process's
    /// configured encoding. Layers the frame already carries are left as they
    /// are.
    pub fn append_to_frame(&self, frame: &mut ActivationFrame) -> Result<()> {
        self.append_to_frame_as(frame, DFlashFeatureEncoding::configured())
    }

    /// Appends these rows to `frame` as optional parts in `encoding`.
    pub fn append_to_frame_as(
        &self,
        frame: &mut ActivationFrame,
        encoding: DFlashFeatureEncoding,
    ) -> Result<()> {
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
        let new_layers: Vec<_> = self
            .layers
            .iter()
            .filter(|(layer, _)| {
                !existing
                    .iter()
                    .any(|part| dflash_feature_layer(&part.identity) == Some(*layer))
            })
            .collect();
        // Check capacity before touching the frame, so a full part directory
        // leaves the frame exactly as it was rather than half-extended.
        if count + new_layers.len() > ACTIVATION_MAX_PARTS {
            bail!(
                "no activation parts left for {} DFlash feature layers ({count} of {ACTIVATION_MAX_PARTS} used)",
                new_layers.len()
            );
        }
        for (layer, values) in new_layers {
            let offset = frame.payload.len();
            let row_bytes = encoding.row_bytes(self.n_embd);
            let (identity, ggml_type, element, width) = match encoding {
                DFlashFeatureEncoding::Bf16 => {
                    encode_bf16(values, &mut frame.payload);
                    (
                        dflash_feature_identity(*layer),
                        GGML_TYPE_BF16,
                        2,
                        self.n_embd,
                    )
                }
                DFlashFeatureEncoding::Q8 => {
                    encode_q8_rows(values, self.n_embd, &mut frame.payload);
                    (
                        q8_feature_identity(*layer, self.n_embd)?,
                        crate::GGML_TYPE_I32,
                        4,
                        row_bytes / 4,
                    )
                }
            };
            let mut part = ActivationPartDesc {
                identity,
                ggml_type,
                rank: 2,
                token_axis: 1,
                flags: ACTIVATION_PART_OPTIONAL,
                payload_offset: u64::try_from(offset).context("feature offset")?,
                payload_bytes: u64::try_from(frame.payload.len() - offset)
                    .context("feature bytes")?,
                ..ActivationPartDesc::default()
            };
            part.dimensions[0] = i64::try_from(width).context("feature width")?;
            part.dimensions[1] = i64::try_from(self.rows).context("feature rows")?;
            part.byte_strides[0] = element;
            part.byte_strides[1] = u64::try_from(row_bytes).context("feature stride")?;
            frame.desc.parts[count] = part;
            count += 1;
        }
        frame.desc.part_count = u32::try_from(count).context("part count")?;
        frame.desc.payload_bytes = u64::try_from(frame.payload.len()).context("payload bytes")?;
        Ok(())
    }
}

/// Captures the inputs of this stage's DFlash target layers on a stage that
/// carries no draft.
pub(crate) struct DFlashCapture {
    api: &'static LlamaDraftApi,
    target_ctx: *mut Opaque,
    /// Layers whose input this stage's graph produces.
    layers: Vec<u32>,
    /// The layer whose input is the activation this stage receives, when the
    /// draft reads it. This stage forwards those rows.
    input_layer: Option<u32>,
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
    /// The layer, and its width, whose input is the activation this stage
    /// receives and forwards.
    input_layer: Option<(u32, usize)>,
    /// That layer's rows from the input of the message being executed.
    received_input: Option<Vec<f32>>,
    /// Whether this stage's graph captures any target layer itself.
    captures_layers: bool,
    /// Rows from earlier stages, and how many of them decodes have consumed.
    pub(crate) remote: Option<(DFlashFeatureRows, usize)>,
    /// Token rows whose captured features this stage could not forward.
    pub(crate) rows_dropped: u64,
}

impl DFlashSplitSession {
    pub(crate) fn capturing(capture: &SharedDFlashCapture) -> Self {
        let (input_layer, captures_layers) = {
            let capture = lock_recovering(capture, |_| {});
            (
                capture.input_layer.map(|layer| (layer, capture.n_embd)),
                !capture.layers.is_empty(),
            )
        };
        Self {
            capture: Some(Arc::downgrade(capture)),
            input_layer,
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
        self.received_input = None;
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

    /// Whether this session reads anything from its input frames for split
    /// DFlash: feature rows for its draft, or the activation it forwards as
    /// its first layer's rows.
    pub fn dflash_reads_input(&self) -> bool {
        self.dflash.is_some()
            || self
                .dflash_split
                .as_ref()
                .is_some_and(|split| split.input_layer.is_some())
    }

    /// Takes what split DFlash needs from `input`, the frame of this
    /// session's next decode. A drafting final stage hands its draft the
    /// earlier stages' feature rows, plus the activation itself as the rows
    /// of its first layer when the draft reads that layer. A capturing stage
    /// that starts on a target layer keeps the activation to forward as that
    /// layer's rows. Malformed rows are dropped and reported; the draft then
    /// finds them missing.
    pub fn dflash_receive_input(&mut self, input: Option<&ActivationFrame>) -> Result<()> {
        if let Some(link) = self.dflash.as_ref() {
            let input_layer = link.input_layer;
            let (rows, error) = match input.map(DFlashFeatureRows::from_frame).transpose() {
                Ok(rows) => (rows.flatten(), None),
                Err(error) => (None, Some(error.context("read DFlash feature parts"))),
            };
            let mut rows = rows;
            let mut error = error;
            if let (Some((layer, n_embd)), Some(frame)) = (input_layer, input)
                && !rows
                    .as_ref()
                    .is_some_and(|rows| rows.layer_ids().any(|id| id == layer))
                && error.is_none()
            {
                let tokens = usize::try_from(frame.desc.token_count).unwrap_or(0);
                let inserted = frame_hidden_rows(frame, n_embd).and_then(|values| {
                    rows.get_or_insert_with(|| DFlashFeatureRows::new(n_embd, tokens))
                        .insert(layer, values)
                });
                if let Err(inserted) = inserted {
                    error = Some(inserted.context(format!("read layer {layer} input")));
                    rows = None;
                }
            }
            self.dflash_set_remote_features(rows);
            return error.map_or(Ok(()), Err);
        }
        let Some(split) = self.dflash_split.as_mut() else {
            return Ok(());
        };
        split.received_input = None;
        let (Some((layer, n_embd)), Some(frame)) = (split.input_layer, input) else {
            return Ok(());
        };
        split.received_input = Some(
            frame_hidden_rows(frame, n_embd)
                .with_context(|| format!("read layer {layer} input"))?,
        );
        Ok(())
    }

    /// Appends the feature rows this stage captured for the decode that
    /// produced `frame` to it, for the final stage's draft. Rows captured by
    /// an earlier decode whose frame left without them are dropped: the frame
    /// carries the rows of its own tokens, which are the last ones captured.
    /// When this stage starts on a target layer, the activation it received
    /// for this decode goes along as that layer's rows.
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
        let received = split.received_input.take();
        let mut input_error = None;
        if let Some((layer, n_embd)) = split.input_layer {
            let inserted = received
                .ok_or_else(|| anyhow!("no input rows were received for this frame"))
                .and_then(|values| {
                    rows.get_or_insert_with(|| DFlashFeatureRows::new(n_embd, tokens))
                        .insert(layer, values)
                });
            if let Err(error) = inserted {
                split.rows_dropped += u64::try_from(tokens).unwrap_or(u64::MAX);
                input_error = Some(error.context(format!("forward layer {layer} input")));
            }
        }
        if let Some(rows) = rows {
            rows.append_to_frame(frame)?;
        }
        input_error.map_or(Ok(()), Err)
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
    /// Whether this stage forwards the activation it receives as the rows of
    /// a draft target layer, so its sessions must see their input frames.
    pub fn dflash_forwards_input(&self) -> bool {
        self.dflash_capture()
            .is_some_and(|capture| lock_recovering(capture, |_| {}).input_layer.is_some())
    }

    /// Captures the inputs of the DFlash target layers this stage owns so its
    /// sessions can forward them to the final stage's draft, plus its input
    /// activation when this stage starts on a target layer. Returns the
    /// forwarded layers. Attach before creating serving sessions.
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
        let input_layer = stage_receives_layer_input(target_layers, layer_start);
        if layers.is_empty() && input_layer.is_none() {
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
            input_layer,
            n_embd,
            n_batch,
        };
        if !layers.is_empty() {
            super::refresh_target_program(&probe)?;
        }
        drop(probe);
        self.install_dflash_capture(Arc::new(Mutex::new(capture)))?;
        crate::write_native_log_note(format!(
            "dflash split capture attached layers={layers:?} input_layer={input_layer:?} encoding={:?}",
            DFlashFeatureEncoding::configured()
        ));
        Ok(layers.into_iter().chain(input_layer).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ACTIVATION_FRAME_VERSION, ACTIVATION_IDENTITY_BYTES, ActivationDesc};

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
    fn feature_rows_ride_a_frame_as_optional_bf16_parts() {
        let mut rows = DFlashFeatureRows::new(2, 3);
        rows.insert(15, vec![1.0, -2.0, 0.5, 3.0, 1024.0, -0.25])
            .unwrap();
        rows.insert(6, vec![0.0; 6]).unwrap();
        let mut frame = frame(3);
        let original = frame.payload.clone();

        rows.append_to_frame_as(&mut frame, DFlashFeatureEncoding::Bf16)
            .unwrap();

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

        first
            .append_to_frame_as(&mut frame, DFlashFeatureEncoding::Bf16)
            .unwrap();
        second
            .append_to_frame_as(&mut frame, DFlashFeatureEncoding::Q8)
            .unwrap();

        assert_eq!(frame.desc.part_count, 2);
        assert_eq!(DFlashFeatureRows::from_frame(&frame).unwrap(), Some(first));
    }

    #[test]
    fn a_full_part_directory_leaves_the_frame_unchanged() {
        let mut first = DFlashFeatureRows::new(2, 1);
        for layer in 0..u32::try_from(ACTIVATION_MAX_PARTS - 2).unwrap() {
            first.insert(layer, vec![1.0, 2.0]).unwrap();
        }
        let mut frame = frame(1);
        first.append_to_frame(&mut frame).unwrap();
        assert_eq!(frame.desc.part_count as usize, ACTIVATION_MAX_PARTS - 1);
        let before = frame.clone();

        let mut second = DFlashFeatureRows::new(2, 1);
        second.insert(100, vec![3.0, 4.0]).unwrap();
        second.insert(101, vec![5.0, 6.0]).unwrap();
        assert!(second.append_to_frame(&mut frame).is_err());

        assert_eq!(frame.payload, before.payload);
        assert_eq!(frame.desc.part_count, before.desc.part_count);
        assert_eq!(frame.desc.payload_bytes, before.desc.payload_bytes);
        assert_eq!(DFlashFeatureRows::from_frame(&frame).unwrap(), Some(first));
    }

    #[test]
    fn rows_must_match_the_frame_tokens() {
        let mut rows = DFlashFeatureRows::new(2, 2);
        rows.insert(1, vec![0.0; 4]).unwrap();
        assert!(rows.append_to_frame(&mut frame(3)).is_err());
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
        assert_close(forwarded.rows_of(2, 0, 3).unwrap(), &[1.0, 2.0, 3.0]);
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
    fn a_cut_on_a_target_layer_leaves_it_to_the_stage_that_receives_it() {
        // 64 layers cut at 32: the final stage's first layer is a target, and
        // its input is the activation that stage receives.
        assert_eq!(captured_by(0, 32), vec![2, 17]);
        assert_eq!(stage_receives_layer_input(&TARGETS, 0), None);
        assert_eq!(captured_by(32, 64), vec![47, 62]);
        assert_eq!(stage_receives_layer_input(&TARGETS, 32), Some(32));
    }

    #[test]
    fn a_middle_stage_starting_on_a_target_layer_forwards_its_input() {
        assert_eq!(captured_by(0, 17), vec![2]);
        assert_eq!(captured_by(17, 40), vec![32]);
        assert_eq!(stage_receives_layer_input(&TARGETS, 17), Some(17));
        assert_eq!(captured_by(40, 64), vec![47, 62]);
        assert_eq!(stage_receives_layer_input(&TARGETS, 40), None);
    }

    #[test]
    fn every_target_layer_has_exactly_one_source_for_any_cut() {
        for cut in 1..64 {
            for layer in TARGETS {
                let sources = [
                    stage_captures_layer_input(layer, 0, cut),
                    stage_receives_layer_input(&TARGETS, cut) == Some(layer),
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

    fn assert_close(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len());
        for (got, want) in got.iter().zip(want) {
            assert!((got - want).abs() <= want.abs() / 64.0, "{got} vs {want}");
        }
    }

    #[test]
    fn feature_rows_ride_a_frame_as_optional_q8_parts() {
        let values = [1.0, -2.0, 0.5, 3.0, 1024.0, -512.0];
        let mut rows = DFlashFeatureRows::new(2, 3);
        rows.insert(15, values.to_vec()).unwrap();
        rows.insert(6, vec![0.0; 6]).unwrap();
        let mut frame = frame(3);

        rows.append_to_frame_as(&mut frame, DFlashFeatureEncoding::Q8)
            .unwrap();

        let parts = frame.desc.parts().unwrap();
        assert_eq!(parts.len(), 3);
        for part in &parts[1..] {
            assert!(part.is_optional());
            assert_eq!(part.ggml_type, crate::GGML_TYPE_I32);
            // A row is one word: a two-byte scale and two values.
            assert_eq!(part.dimensions[..2], [1, 3]);
            assert_eq!(part.byte_strides[..2], [4, 4]);
            assert_eq!(part.payload_bytes, 12);
        }
        assert_eq!(dflash_feature_payload_bytes(&frame), 24);
        let read = DFlashFeatureRows::from_frame(&frame).unwrap().unwrap();
        assert_eq!(read.layer_ids().collect::<Vec<_>>(), vec![6, 15]);
        assert_close(read.rows_of(15, 0, 3).unwrap(), &values);
        assert_eq!(read.rows_of(6, 0, 3), Some(&[0.0; 6][..]));

        // A width that disagrees with the packed rows is refused.
        let mut wrong = frame.clone();
        wrong.desc.parts[1].identity = q8_feature_identity(6, 64).unwrap();
        assert!(DFlashFeatureRows::from_frame(&wrong).is_err());
    }

    fn drafting_session(input_layer: Option<(u32, usize)>) -> StageSession {
        let mut link = super::super::DFlashSessionLink::detached(0);
        link.input_layer = input_layer;
        StageSession {
            raw: std::ptr::null_mut(),
            token_count: 0,
            terminal_stage: true,
            batched_activation_exports: true,
            dflash: Some(link),
            dflash_split: None,
        }
    }

    #[test]
    fn a_final_stage_reads_its_first_layer_from_its_input_activation() {
        let hidden = [1.0, -2.0, 0.5, 65504.0];
        let mut input = hidden_frame(crate::GGML_TYPE_F32, &hidden);
        let mut upstream = DFlashFeatureRows::new(2, 2);
        upstream.insert(17, vec![3.0; 4]).unwrap();
        upstream.append_to_frame(&mut input).unwrap();
        let mut session = drafting_session(Some((32, 2)));
        assert!(session.dflash_reads_input());

        session.dflash_receive_input(Some(&input)).unwrap();

        let remote = session.dflash_remote_features_for_test().unwrap();
        assert_eq!(remote.layer_ids().collect::<Vec<_>>(), vec![17, 32]);
        // The activation never went through a feature encoding.
        assert_eq!(remote.rows_of(32, 0, 2), Some(&hidden[..]));
        assert_close(remote.rows_of(17, 0, 2).unwrap(), &[3.0; 4]);
    }

    #[test]
    fn first_layer_rows_an_upstream_stage_still_sends_are_kept() {
        let mut input = hidden_frame(crate::GGML_TYPE_F32, &[1.0, 2.0, 3.0, 4.0]);
        let mut sent = DFlashFeatureRows::new(2, 2);
        sent.insert(32, vec![5.0; 4]).unwrap();
        sent.append_to_frame_as(&mut input, DFlashFeatureEncoding::Bf16)
            .unwrap();
        let mut session = drafting_session(Some((32, 2)));

        session.dflash_receive_input(Some(&input)).unwrap();

        let remote = session.dflash_remote_features_for_test().unwrap();
        assert_eq!(remote.rows_of(32, 0, 2), Some(&[5.0; 4][..]));
    }

    #[test]
    fn a_malformed_input_activation_leaves_the_draft_without_rows() {
        let input = hidden_frame(crate::GGML_TYPE_F32, &[1.0, 2.0, 3.0, 4.0]);
        let mut session = drafting_session(Some((32, 3)));

        assert!(session.dflash_receive_input(Some(&input)).is_err());

        assert_eq!(session.dflash_remote_features_for_test(), None);
    }

    fn middle_session(captured: Option<DFlashFeatureRows>) -> StageSession {
        let split = DFlashSplitSession {
            input_layer: Some((17, 2)),
            captured,
            ..DFlashSplitSession::default()
        };
        let mut session = session_with_captured(DFlashFeatureRows::default());
        session.dflash_split = Some(Box::new(split));
        session
    }

    #[test]
    fn a_middle_stage_forwards_its_input_as_its_first_layer() {
        let hidden = [1.0, -2.0, 0.5, 4.0];
        let mut captured = DFlashFeatureRows::new(2, 2);
        captured.insert(32, vec![3.0; 4]).unwrap();
        let mut session = middle_session(Some(captured));
        assert!(session.dflash_reads_input());

        session
            .dflash_receive_input(Some(&hidden_frame(crate::GGML_TYPE_F32, &hidden)))
            .unwrap();
        let mut output = hidden_frame(crate::GGML_TYPE_F32, &[9.0; 4]);
        session.dflash_forward_captured(&mut output).unwrap();

        let forwarded = DFlashFeatureRows::from_frame(&output).unwrap().unwrap();
        assert_eq!(forwarded.layer_ids().collect::<Vec<_>>(), vec![17, 32]);
        assert_close(forwarded.rows_of(17, 0, 2).unwrap(), &hidden);
        // The received rows belonged to that frame alone.
        let mut next = hidden_frame(crate::GGML_TYPE_F32, &[9.0; 4]);
        assert!(session.dflash_forward_captured(&mut next).is_err());
        assert_eq!(session.dflash_rows_dropped(), 2);
    }

    #[test]
    fn a_malformed_middle_input_is_an_error_not_a_panic() {
        let mut session = middle_session(None);
        if let Some(split) = session.dflash_split.as_mut() {
            split.input_layer = Some((17, 3));
        }

        let input = hidden_frame(crate::GGML_TYPE_F32, &[1.0; 4]);
        assert!(session.dflash_receive_input(Some(&input)).is_err());
        let mut output = hidden_frame(crate::GGML_TYPE_F32, &[1.0; 4]);
        assert!(session.dflash_forward_captured(&mut output).is_err());

        assert_eq!(DFlashFeatureRows::from_frame(&output).unwrap(), None);
    }
}
