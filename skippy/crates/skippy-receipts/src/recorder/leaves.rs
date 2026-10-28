//! Hashes the committed leaves of a ubatch.
//!
//! One BLAKE3 call runs on one core, and a ubatch holds two leaves per layer
//! for every column, so the work is split across cores. A unit is one layer
//! of one column (its residual and attention leaves), or the stage output
//! residual of one column.

use std::num::NonZero;
use std::ops::Range;

use anyhow::{Context, Result};

use super::{TensorKind, UbatchColumns};
use crate::commitment::{Hash, attention_leaf, residual_leaf};

/// Hashing threads per ubatch, at most.
const MAX_THREADS: usize = 8;
/// A thread is spawned only for at least this many units.
const MIN_UNITS_PER_THREAD: usize = 4;

type UnitLeaves = (Hash, Option<Hash>);

struct Ubatch<'a> {
    layers: Range<u32>,
    pos: &'a [i32],
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
            let output = self.columns.column(layer, TensorKind::Residual, col);
            return Ok((residual_leaf(position, layer, output), None));
        }
        let column = |kind: TensorKind| {
            self.columns
                .column(layer, kind, col)
                .with_context(|| format!("layer {layer} reported no {kind:?} column"))
        };
        let residual = residual_leaf(position, layer, Some(column(TensorKind::Residual)?));
        let attention = attention_leaf(
            position,
            layer,
            column(TensorKind::AttnQ)?,
            column(TensorKind::AttnK)?,
            column(TensorKind::AttnV)?,
            column(TensorKind::AttnOut)?,
        );
        Ok((residual, Some(attention)))
    }

    fn units(&self, range: Range<usize>) -> Vec<Result<UnitLeaves>> {
        range.map(|unit| self.unit(unit)).collect()
    }

    fn hash_all(&self) -> Vec<Result<UnitLeaves>> {
        let units = self.pos.len() * self.units_per_column();
        let threads = std::thread::available_parallelism()
            .map_or(1, NonZero::get)
            .min(MAX_THREADS)
            .min(units / MIN_UNITS_PER_THREAD);
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
/// position. A column with a missing tensor is an error.
pub(super) fn ubatch_leaves(
    layers: Range<u32>,
    pos: &[i32],
    columns: &dyn UbatchColumns,
) -> Vec<Result<(u32, Vec<Hash>)>> {
    let ubatch = Ubatch {
        layers,
        pos,
        columns,
    };
    let per_column = ubatch.units_per_column();
    let mut hashed = ubatch.hash_all().into_iter();
    pos.iter()
        .map(|&pos| {
            let mut leaves = Vec::with_capacity(2 * per_column - 1);
            let mut error = None;
            for unit in hashed.by_ref().take(per_column) {
                match unit {
                    Ok((residual, attention)) => {
                        leaves.push(residual);
                        leaves.extend(attention);
                    }
                    Err(e) => {
                        error.get_or_insert(e);
                    }
                }
            }
            match error {
                Some(e) => Err(e),
                None => Ok((u32::try_from(pos).context("negative position")?, leaves)),
            }
        })
        .collect()
}
