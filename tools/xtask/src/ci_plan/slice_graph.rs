//! Slice dependency edges: every dependency names a known slice and the
//! graph is acyclic.

use crate::ci_plan::diagnostics::{PlanResult, fail, repr, repr_list, sorted_unknown};
use crate::ci_plan::slice_catalog::SliceDefinition;
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn check_dependencies(
    definitions: &[SliceDefinition],
    known: &BTreeSet<&str>,
) -> PlanResult<()> {
    for slice in definitions {
        let unknown = sorted_unknown(&slice.depends_on, known);
        if !unknown.is_empty() {
            return fail(format!(
                "slice {} depends on unknown slices {}",
                slice.id,
                repr_list(&unknown)
            ));
        }
    }
    let edges = definitions
        .iter()
        .map(|slice| (slice.id.as_str(), slice.depends_on.as_slice()))
        .collect::<BTreeMap<_, _>>();
    let mut visiting = BTreeSet::new();
    let mut visited = BTreeSet::new();
    for slice in definitions {
        visit(&slice.id, &edges, &mut visiting, &mut visited)?;
    }
    Ok(())
}

/// Depth-first `_assert_acyclic`, reporting the node first seen twice.
fn visit<'a>(
    node: &'a str,
    edges: &BTreeMap<&'a str, &'a [String]>,
    visiting: &mut BTreeSet<&'a str>,
    visited: &mut BTreeSet<&'a str>,
) -> PlanResult<()> {
    if visiting.contains(node) {
        return fail(format!("slice dependency cycle includes {}", repr(node)));
    }
    if visited.contains(node) {
        return Ok(());
    }
    visiting.insert(node);
    for dependency in edges.get(node).copied().unwrap_or_default() {
        visit(dependency, edges, visiting, visited)?;
    }
    visiting.remove(node);
    visited.insert(node);
    Ok(())
}
