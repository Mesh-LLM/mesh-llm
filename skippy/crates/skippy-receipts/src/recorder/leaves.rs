//! Hashes the committed leaves of a ubatch.
//!
//! One BLAKE3 call runs on one core, and a ubatch holds two leaves per layer
//! for every column, so the work is split across cores. A unit is one layer
//! of one column (its residual and mixing leaves), or the stage output
//! residual of one column.

use std::num::NonZero;
use std::ops::Range;
use std::sync::OnceLock;

use anyhow::{Context, Result};

use crate::commitment::{
    Hash, MixingDigest, ValueDigest, cell_digest, mixing_leaf_from, mixing_rest, residual_leaf_of,
    stored_leaf,
};
use crate::layer_tensors::{GGML_F32, MIXING_KINDS, TensorKind, UbatchColumns};

/// Hashing threads per ubatch, at most.
const MAX_THREADS: usize = 8;
/// A thread is spawned only for at least this many units.
const MIN_UNITS_PER_THREAD: usize = 4;

/// A unit's residual leaf, and for a layer its mixing leaf's halves.
type UnitLeaves = (Hash, Option<MixingHalves>);

#[derive(Clone, Copy)]
struct MixingHalves {
    rest: Hash,
    stored: Hash,
}

/// What a stage committed at one position.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct PositionLeaves {
    /// The leaves, in slot order.
    pub leaves: Vec<Hash>,
    /// Per layer, the half of its mixing leaf that is not stored rows
    /// ([`mixing_rest`]), which opens the stored rows.
    pub rests: Vec<Hash>,
    /// The digest of the cell the position was written to.
    pub cell: Hash,
}

struct Ubatch<'a> {
    layers: Range<u32>,
    pos: &'a [i32],
    /// The ubatch's tokens, on the stage that holds layer 0.
    tokens: Option<&'a [i32]>,
    columns: &'a dyn UbatchColumns,
}

impl Ubatch<'_> {
    fn units_per_column(&self) -> usize {
        self.layers.len() + 1
    }

    fn unit(&self, unit: usize) -> Result<UnitLeaves> {
        let per_column = self.units_per_column();
        let col = unit / per_column;
        let layer = self.layers.start + u32::try_from(unit % per_column)?;
        let position = u32::try_from(self.pos[col]).context("negative position")?;
        if layer == self.layers.end {
            let output = self.residual(layer, col)?;
            return Ok((residual_leaf_of(position, layer, None, output), None));
        }
        let residual = self
            .residual(layer, col)?
            .with_context(|| format!("layer {layer} reported no residual column"))?;
        let token = match (layer, self.tokens) {
            (0, Some(tokens)) => Some(tokens[col]),
            _ => None,
        };
        let residual = residual_leaf_of(position, layer, token, Some(residual));
        let (stored, rest): (Vec<MixingDigest>, Vec<MixingDigest>) =
            mixing_digests(self.columns, layer, col)
                .into_iter()
                .partition(|d| d.kind == TensorKind::MixingStored);
        let halves = MixingHalves {
            rest: mixing_rest(&rest),
            stored: stored_leaf(position, layer, &stored),
        };
        Ok((residual, Some(halves)))
    }

    fn residual(&self, layer: u32, col: usize) -> Result<Option<ValueDigest>> {
        match self.columns.column(layer, TensorKind::Residual, 0, col) {
            Some(column) if column.ggml_type != GGML_F32 => {
                anyhow::bail!("layer {layer} reported a residual that is not f32")
            }
            column => Ok(column.map(|c| c.value_digest())),
        }
    }

    fn units(&self, range: Range<usize>) -> Vec<Result<UnitLeaves>> {
        range.map(|unit| self.unit(unit)).collect()
    }

    fn hash_all(&self) -> Vec<Result<UnitLeaves>> {
        let units = self.pos.len() * self.units_per_column();
        // the count reads cgroup files on Linux, too slow for every ubatch
        static CPUS: OnceLock<usize> = OnceLock::new();
        let cpus =
            *CPUS.get_or_init(|| std::thread::available_parallelism().map_or(1, NonZero::get));
        let threads = cpus.min(MAX_THREADS).min(units / MIN_UNITS_PER_THREAD);
        if threads <= 1 {
            return self.units(0..units);
        }
        let chunk = units.div_ceil(threads);
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..units)
                .step_by(chunk)
                .map(|start| scope.spawn(move || self.units(start..(start + chunk).min(units))))
                .collect();
            handles
                .into_iter()
                .flat_map(|handle| handle.join().expect("a leaf hashing thread panicked"))
                .collect()
        })
    }
}

/// The leaves of every column of a ubatch, in slot order, with the column's
/// position. A column with a missing tensor is an error. On the stage that
/// holds layer 0, the residual leaf of layer 0 binds each column's token.
pub(super) fn ubatch_leaves(
    layers: Range<u32>,
    pos: &[i32],
    tokens: &[i32],
    columns: &dyn UbatchColumns,
) -> Vec<Result<(u32, PositionLeaves)>> {
    let tokens = (layers.start == 0).then_some(tokens);
    let ubatch = Ubatch {
        layers,
        pos,
        tokens,
        columns,
    };
    let layers_start = ubatch.layers.start;
    let per_column = ubatch.units_per_column();
    let mut hashed = ubatch.hash_all().into_iter();
    pos.iter()
        .map(|&pos| {
            // take every unit of the column, so the next column starts at its own
            let units: Vec<Result<UnitLeaves>> = hashed.by_ref().take(per_column).collect();
            let units = units.into_iter().collect::<Result<Vec<_>>>()?;
            let position = u32::try_from(pos).context("negative position")?;
            Ok((position, position_leaves(position, layers_start, units)?))
        })
        .collect()
}

/// The digests of a layer's committed mixing values at one column, in leaf
/// order.
fn mixing_digests(columns: &dyn UbatchColumns, layer: u32, col: usize) -> Vec<MixingDigest> {
    MIXING_KINDS
        .iter()
        .flat_map(|&kind| {
            (0..columns.count(layer, kind)).map(move |index| MixingDigest {
                kind,
                index,
                value: columns
                    .column(layer, kind, index, col)
                    .map(|c| (c.ggml_type, c.value_digest())),
            })
        })
        .collect()
}

/// A position's leaves in slot order, its rests by layer and its cell
/// digest, from its units.
fn position_leaves(
    position: u32,
    layer_start: u32,
    units: Vec<UnitLeaves>,
) -> Result<PositionLeaves> {
    let mut leaves = Vec::with_capacity(2 * units.len());
    let mut rests = Vec::with_capacity(units.len());
    let mut stored = Vec::with_capacity(units.len());
    for (residual, halves) in units {
        leaves.push(residual);
        if let Some(halves) = halves {
            let layer = layer_start + u32::try_from(rests.len())?;
            leaves.push(mixing_leaf_from(
                position,
                layer,
                &halves.rest,
                &halves.stored,
            ));
            rests.push(halves.rest);
            stored.push(halves.stored);
        }
    }
    Ok(PositionLeaves {
        leaves,
        rests,
        cell: cell_digest(stored)?,
    })
}
