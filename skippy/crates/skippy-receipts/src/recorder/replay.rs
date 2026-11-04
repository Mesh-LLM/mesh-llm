//! What a replay of a recorded turn is given and what it reports: the steps
//! and the cells it starts from, the values it recomputes per position and
//! layer, and the [`Replayer`] a stage implements over its native runtime.

use std::sync::Arc;

use anyhow::{Context, Result, bail};

use super::{KvCell, PrefixCell, UbatchRecord};
use crate::attention::AttentionDescriptor;
use crate::layer_tensors::{FLAG_ROUTER, GGML_F32, MIXING_KINDS, TensorKind, UbatchColumns};
use crate::wire::{MatrixOpening, MixingKind, SelectionOpening, TensorOpening, encode_bytes};

/// One step of a replay, columns ready for the native side. Other
/// sequences' columns are dummies at their recorded cells.
#[derive(Clone, Debug, PartialEq)]
pub enum ReplayStep {
    Decode {
        record: Arc<UbatchRecord>,
        /// Positions of the replayed sequence to capture at this step.
        capture: Vec<u32>,
        /// Positions of the replayed sequence whose token-mixing values
        /// alone to capture at this step, besides `capture`.
        mixing: Vec<u32>,
    },
    /// Removes positions `[p0, p1)` of the replayed sequence; `p1 < 0` has no end.
    SeqRm { p0: i32, p1: i32 },
}

/// One position of the challenged layer, as the replay recomputed it.
#[derive(Clone, Debug, PartialEq)]
pub struct ReplayedLayer {
    /// Little-endian f32 bytes.
    pub residual_in: Vec<u8>,
    pub residual_out: Option<Vec<u8>>,
    /// The logits a final stage's last layer leads to, little-endian f32,
    /// where the replay reported them.
    pub logits: Option<Vec<u8>>,
    /// The mixing leaf's values, in leaf order.
    pub mixing: Vec<ReplayedMixing>,
    pub matmuls: Vec<ReplayedMatmul>,
    pub selections: Vec<SelectionOpening>,
    /// The layer's attention ops, as the replay's graph describes them.
    pub attention: Vec<AttentionDescriptor>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReplayedMixing {
    pub kind: MixingKind,
    pub index: u32,
    /// ggml type and bytes; `None` where the layer kept no row.
    pub value: Option<(i32, Vec<u8>)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReplayedMatmul {
    pub index: u32,
    pub weight: String,
    pub router: bool,
    pub experts: Option<Vec<i32>>,
    /// Little-endian f32 bytes.
    pub input: Vec<u8>,
    pub output: Vec<u8>,
}

impl ReplayedLayer {
    /// The token-mixing values alone of column `col` of a captured replay
    /// ubatch for `layer`.
    pub fn from_mixing_columns(layer: u32, columns: &dyn UbatchColumns, col: usize) -> Self {
        Self {
            residual_in: Vec::new(),
            residual_out: None,
            logits: None,
            mixing: mixing_columns(layer, columns, col),
            matmuls: Vec::new(),
            selections: Vec::new(),
            attention: Vec::new(),
        }
    }

    /// Collects column `col` of a captured replay ubatch for `layer`.
    pub fn from_columns(layer: u32, columns: &dyn UbatchColumns, col: usize) -> Result<Self> {
        let f32_column = |l: u32, kind: TensorKind, index: u32| -> Result<Option<Vec<u8>>> {
            match columns.column(l, kind, index, col) {
                Some(c) if c.ggml_type != GGML_F32 => {
                    bail!(
                        "replay reported {} {index} of layer {l} as type {}",
                        kind.name(),
                        c.ggml_type
                    )
                }
                c => Ok(c.map(|c| c.bytes.to_vec())),
            }
        };
        let residual_in = f32_column(layer, TensorKind::Residual, 0)?
            .with_context(|| format!("replay did not capture the residual of layer {layer}"))?;
        let residual_out = f32_column(layer + 1, TensorKind::Residual, 0)?;
        let logits = f32_column(layer + 1, TensorKind::Logits, 0)?;
        let mixing = mixing_columns(layer, columns, col);
        let mut matmuls = Vec::new();
        for index in 0..columns.count(layer, TensorKind::MatmulInput) {
            let (Some(input), Some(output)) = (
                columns.column(layer, TensorKind::MatmulInput, index, col),
                f32_column(layer, TensorKind::MatmulOutput, index)?,
            ) else {
                // the layer dropped this position's row before the matmul
                continue;
            };
            if input.ggml_type != GGML_F32 {
                bail!(
                    "replay reported matmul input {index} of layer {layer} as type {}",
                    input.ggml_type
                );
            }
            let experts = columns
                .column(layer, TensorKind::ExpertIds, index, col)
                .map(|c| c.i32_values().context("expert ids are not i32"))
                .transpose()?;
            matmuls.push(ReplayedMatmul {
                index,
                weight: input.name.to_owned(),
                router: input.flags & FLAG_ROUTER != 0,
                experts,
                input: input.bytes.to_vec(),
                output,
            });
        }
        let selections = (0..columns.count(layer, TensorKind::Selection))
            .filter_map(|index| {
                let column = columns.column(layer, TensorKind::Selection, index, col)?;
                Some(
                    column
                        .i32_values()
                        .context("a selection is not i32")
                        .map(|ids| SelectionOpening { index, ids }),
                )
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            residual_in,
            residual_out,
            logits,
            mixing,
            matmuls,
            selections,
            attention: Vec::new(),
        })
    }

    pub(super) fn mixing_openings(&self) -> Vec<TensorOpening> {
        self.mixing
            .iter()
            .map(|m| {
                TensorOpening::new(
                    m.kind,
                    m.index,
                    m.value.as_ref().map(|(t, bytes)| (*t, bytes.as_slice())),
                )
            })
            .collect()
    }

    pub(super) fn matrices(&self) -> Vec<MatrixOpening> {
        self.matmuls
            .iter()
            .map(|m| MatrixOpening {
                index: m.index,
                weight: m.weight.clone(),
                router: m.router,
                experts: m.experts.clone(),
                input: encode_bytes(&m.input),
                output: encode_bytes(&m.output),
            })
            .collect()
    }
}

/// The mixing leaf's values of `layer` in column `col`, in leaf order.
fn mixing_columns(layer: u32, columns: &dyn UbatchColumns, col: usize) -> Vec<ReplayedMixing> {
    MIXING_KINDS
        .iter()
        .flat_map(|&kind| (0..columns.count(layer, kind)).map(move |index| (kind, index)))
        .map(|(kind, index)| ReplayedMixing {
            kind: MixingKind::from_tensor_kind(kind).expect("a mixing kind"),
            index,
            value: columns
                .column(layer, kind, index, col)
                .map(|c| (c.ggml_type, c.bytes.to_vec())),
        })
        .collect()
}

/// One position of one layer, as a replay recomputed it.
pub type ReplayedPosition = (u32, u32, ReplayedLayer);

/// The cells of the stage's KV cache a replay starts from, copied into the
/// replay context before its steps: the prefix the turn attended to, and
/// the turn's own cells when the replay computes only some of its ubatches
/// (`turn_seed`).
#[derive(Clone, Debug, PartialEq)]
pub struct ReplaySeed {
    pub cells: Vec<PrefixCell>,
    /// KV cells the replay context needs: the seeded ones and every one
    /// the steps use.
    pub n_cells: u32,
    /// On a stage with hybrid memory, the recurrent state the turn started
    /// from, which the replay imports after the cells.
    pub state: Option<Arc<[u8]>>,
}

impl ReplaySeed {
    pub fn indices(&self) -> Vec<u32> {
        self.cells.iter().map(|c| c.cell).collect()
    }

    pub fn positions(&self) -> Vec<i32> {
        self.cells.iter().map(|c| c.pos).collect()
    }
}

/// Which tensors a replay reports, and from which values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ReplayScope<'a> {
    /// The committed values and the matmuls and selections an opening
    /// shows besides them.
    Full,
    /// The committed values alone, as the live recorder marks them. Marking
    /// more can change which kernels a backend fuses (CUDA's fused MoE
    /// routing and gated matmuls, for two), so a full replay may not
    /// reproduce the committed values bit for bit while this one does.
    Committed,
    /// As [`Self::Full`], with each anchor's committed value written into
    /// its token's column as soon as the replay computes it, so the matmuls
    /// and selections of the anchored tokens follow from the committed
    /// values.
    Anchored(&'a [ReplayAnchor]),
}

/// A committed value an anchored replay writes into one token's column.
#[derive(Clone, Debug, PartialEq)]
pub struct ReplayAnchor {
    /// The decode step, as an index into the replay's steps.
    pub step: usize,
    pub seq: i32,
    pub pos: i32,
    pub layer: u32,
    pub value: AnchorValue,
    pub values: Vec<f32>,
}

/// What an anchor replaces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnchorValue {
    /// The residual entering the layer.
    Residual,
    /// A token-mixing output, by index.
    MixingOutput(u32),
}

/// Replays a sequence's recorded ubatches; implemented over the native runtime.
pub trait Replayer {
    /// Runs `steps` for `seq`, from the cells of `seed` when there is one,
    /// and returns, for every captured position and every one of `layers`,
    /// the position, the layer and its values. A seed is copied with the
    /// stage idle, after [`super::Recorder::check_seed`] confirms its cells still
    /// hold what the turn read or wrote. `scope` says which tensors it
    /// reports.
    fn replay(
        &self,
        seq: i32,
        layers: &[u32],
        scope: ReplayScope<'_>,
        seed: Option<&ReplaySeed>,
        steps: &[ReplayStep],
    ) -> Result<Vec<ReplayedPosition>>;

    /// The rows the stage stored for `layer` in `cells`, read from its KV
    /// cache with the stage idle, after [`super::Recorder::check_cells`] confirms
    /// they still hold what the turn read or wrote. One [`StoredRows`] per
    /// stored row of the layer, by index, each with a row per cell.
    fn read_rows(&self, layer: u32, cells: &[PrefixCell]) -> Result<Vec<StoredRows>>;

    /// [`Self::read_rows`] of cells whose rows are about to be checked
    /// against their digests ([`super::Recorder::verify_exposed`]): a cell
    /// the recorder marked exposed is read too, after
    /// [`super::Recorder::check_unseen_overwrite`].
    fn read_rows_to_check(&self, layer: u32, cells: &[PrefixCell]) -> Result<Vec<StoredRows>> {
        self.read_rows(layer, cells)
    }

    /// Pins `cells` of the stage's KV cache in place of those pinned before
    /// ([`super::Recorder::pinned_cells`]): while one is empty, the stage writes
    /// other cells as long as a ubatch fits there. It does not wait for the
    /// stage to be idle.
    fn pin_cells(&self, cells: &[KvCell]) -> Result<()>;
}

/// One stored row of a layer in several KV cells, in the cache's type.
#[derive(Clone, Debug, PartialEq)]
pub struct StoredRows {
    pub ggml_type: i32,
    pub row_bytes: usize,
    /// Row `i` at `values[i * row_bytes..]`.
    pub values: Vec<u8>,
}

impl StoredRows {
    pub fn row(&self, i: usize) -> &[u8] {
        self.values
            .get(i * self.row_bytes..(i + 1) * self.row_bytes)
            .unwrap_or(&[])
    }
}
