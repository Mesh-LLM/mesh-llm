//! The opening against the layer the auditor built: its plan.
//!
//! The auditor's evaluation reports the layer's weight matmuls, mixing
//! values and selections by index. An opening must describe that layer and
//! no other: the same matmuls, each once, reading the same weights with
//! the same router flags; the same mixing values; the same selections. A
//! value the auditor does not get it computes as zeros, so a node that
//! left one out would skip the work behind it; and a matmul opened twice
//! would count twice toward the row check's allowance for outliers.

use std::collections::{BTreeMap, BTreeSet};

use super::{Computed, Verdicts};
use crate::layer_tensors::{FLAG_ROUTER, TensorKind};
use crate::wire::{LayerOpening, LeafOpening, MixingKind};

/// One weight matmul of a layer, as the auditor's graph has it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedMatmul {
    /// The GGUF tensor the matmul reads.
    pub weight: String,
    /// Its output feeds a selection, so every row of it is checked.
    pub router: bool,
}

/// The layer's weight matmuls by index.
pub(super) fn matmuls(computed: &Computed<'_>) -> BTreeMap<u32, PlannedMatmul> {
    computed
        .indices(TensorKind::MatmulInput)
        .filter_map(|index| {
            let tensor = computed.get(TensorKind::MatmulInput, index)?;
            let planned = PlannedMatmul {
                weight: tensor.name.clone(),
                router: tensor.flags & FLAG_ROUTER != 0,
            };
            Some((index, planned))
        })
        .collect()
}

/// Fails the "layer plan" verdict for everything in `opening` that is not
/// the auditor's layer.
pub(super) fn check(opening: &LayerOpening, computed: &Computed<'_>, verdicts: &mut Verdicts) {
    let planned = matmuls(computed);
    for leaf in &opening.leaves {
        check_matmuls(leaf, &planned, verdicts);
        check_mixing(leaf, computed, verdicts);
        check_selections(leaf, computed, verdicts);
    }
}

/// A final layer drops the rows it does not output, with their later
/// matmuls and selections; the committed residual out records which rows
/// those are.
fn kept(leaf: &LeafOpening) -> bool {
    leaf.residual_out.is_some()
}

fn check_matmuls(
    leaf: &LeafOpening,
    planned: &BTreeMap<u32, PlannedMatmul>,
    verdicts: &mut Verdicts,
) {
    let position = leaf.position;
    let mut opened = BTreeSet::new();
    for m in &leaf.matrices {
        if !opened.insert(m.index) {
            verdicts.inconsistent(format!(
                "position {position}: matmul {} ({}) is opened more than once",
                m.index, m.weight
            ));
        }
        match planned.get(&m.index) {
            None => verdicts.inconsistent(format!(
                "position {position}: the layer has no matmul {} ({})",
                m.index, m.weight
            )),
            Some(plan) if plan.weight != m.weight => verdicts.inconsistent(format!(
                "position {position}: matmul {} reads {}, opened as {}",
                m.index, plan.weight, m.weight
            )),
            Some(plan) if plan.router != m.router => verdicts.inconsistent(format!(
                "position {position}: {} is opened with the wrong router flag",
                m.weight
            )),
            Some(_) => {}
        }
    }
    // a kept row must open every matmul: the auditor computes an unopened
    // one as zeros, which would skip its work
    if kept(leaf) {
        for (index, plan) in planned.iter().filter(|(index, _)| !opened.contains(index)) {
            verdicts.inconsistent(format!(
                "position {position}: matmul {index} ({}) is not opened",
                plan.weight
            ));
        }
    }
}

fn check_mixing(leaf: &LeafOpening, computed: &Computed<'_>, verdicts: &mut Verdicts) {
    let position = leaf.position;
    if let Err(error) = crate::binding::check_mixing_order(&leaf.mixing) {
        verdicts.inconsistent(format!("position {position}: {error:#}"));
    }
    for (kind, mixing) in [
        (TensorKind::MixingInput, MixingKind::Input),
        (TensorKind::MixingStored, MixingKind::Stored),
        (TensorKind::MixingState, MixingKind::State),
    ] {
        let opened = leaf.mixing.iter().filter(|t| t.kind == mixing).count();
        if opened != computed.count(kind) {
            verdicts.inconsistent(format!(
                "position {position}: {opened} opened {}s, the layer has {}",
                kind.name(),
                computed.count(kind)
            ));
        }
    }
    if !kept(leaf) {
        return;
    }
    // a row the layer kept has every value its token mixing reads, stores
    // and gives; an absent one would go uncompared. A state is opened only
    // where a ubatch starts.
    let absent = leaf
        .mixing
        .iter()
        .filter(|t| t.kind != MixingKind::State && t.values.is_none());
    for value in absent {
        verdicts.inconsistent(format!(
            "position {position}: {} {} is absent on a row the layer kept",
            value.kind.tensor_kind().name(),
            value.index
        ));
    }
}

fn check_selections(leaf: &LeafOpening, computed: &Computed<'_>, verdicts: &mut Verdicts) {
    let dropped = !kept(leaf) && leaf.selections.is_empty();
    let opened: Vec<u32> = leaf.selections.iter().map(|s| s.index).collect();
    let planned: Vec<u32> = computed.indices(TensorKind::SelectionScores).collect();
    if opened != planned && !dropped {
        verdicts.inconsistent(format!(
            "position {}: selections {opened:?} are opened, the layer has {planned:?}",
            leaf.position
        ));
    }
}
