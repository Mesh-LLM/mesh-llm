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
    layers: Vec<u32>,
    n_embd: usize,
    n_batch: usize,
}

// Only touched behind the owning Mutex, and Skippy serializes target-context
// use on the runtime worker.
unsafe impl Send for DFlashCapture {}

impl DFlashCapture {
    fn read(&self, first_row: usize, rows: usize) -> Result<DFlashFeatureRows> {
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
    /// Rows from earlier stages, and how many of them decodes have consumed.
    pub(crate) remote: Option<(DFlashFeatureRows, usize)>,
}

impl DFlashSplitSession {
    pub(crate) fn capturing(capture: &SharedDFlashCapture) -> Self {
        Self {
            capture: Some(Arc::downgrade(capture)),
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
                crate::write_native_log_note(format!("dflash feature capture failed: {error:#}"));
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
    pub fn dflash_forward_captured(&mut self, frame: &mut ActivationFrame) -> Result<()> {
        let Some(mut rows) = self
            .dflash_split
            .as_mut()
            .and_then(|split| split.captured.take())
        else {
            return Ok(());
        };
        rows.keep_trailing(usize::try_from(frame.desc.token_count).context("frame tokens")?);
        rows.append_to_frame(frame)
    }

    /// Whether this session feeds a split DFlash pipeline, as a draft or as a
    /// capturing stage.
    pub fn dflash_split_active(&self) -> bool {
        self.dflash.is_some() || self.dflash_split.is_some()
    }
}

impl StageModel {
    /// Captures the inputs of the DFlash target layers this stage owns so its
    /// sessions can forward them to the final stage's draft. Returns the
    /// captured layers. Attach before creating serving sessions.
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
            .filter(|layer| (layer_start..layer_end).contains(layer))
            .collect::<Vec<_>>();
        if layers.is_empty() {
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
        for &layer in &layers {
            unsafe { (api.set_embeddings_layer_inp)(target_ctx, layer, true) };
        }
        let capture = DFlashCapture {
            api,
            target_ctx,
            layers: layers.clone(),
            n_embd,
            n_batch,
        };
        super::refresh_target_program(&probe)?;
        drop(probe);
        self.install_dflash_capture(Arc::new(Mutex::new(capture)))?;
        crate::write_native_log_note(format!("dflash split capture attached layers={layers:?}"));
        Ok(layers)
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
}
