use super::super::ledger::BoundaryDisposition;
use super::boundaries::Boundaries;
use super::{GraphBuilder, recipe_sources};
use crate::command::DynResult;
use std::collections::BTreeSet;
use std::path::Path;

impl Boundaries {
    pub(super) fn walk_gated(
        &mut self,
        root: &Path,
        builder: &mut GraphBuilder<'_>,
        visited: &mut BTreeSet<(String, &'static str)>,
    ) -> DynResult<()> {
        loop {
            let mut targets = Vec::new();
            for index in 0..builder.edges.len() {
                let occurrence = self.occurrence(root, &builder.edges, index);
                let edge = &builder.edges[index];
                let Some(record) = self.find(edge, occurrence) else {
                    continue;
                };
                let target = match record.disposition {
                    BoundaryDisposition::PlatformConditional
                    | BoundaryDisposition::BoundedSelector
                    | BoundaryDisposition::FiniteTarget => {
                        record.bound_child.clone().or_else(|| edge.child.clone())
                    }
                    BoundaryDisposition::RuntimeSelectedBoundary => record.bound_child.clone(),
                    BoundaryDisposition::ExternalTrustBoundary => None,
                };
                if let Some(bound) = &record.bound_child
                    && !builder.known.contains(bound.as_str())
                {
                    return Err(
                        format!("boundary record {}: unknown child {bound}", record.id).into(),
                    );
                }
                targets.extend(target.filter(|path| {
                    builder.known.contains(path.as_str())
                        && !visited.contains(&(path.clone(), "same_commit"))
                }));
            }
            builder
                .queue
                .extend(targets.into_iter().map(|path| (path, "same_commit")));
            if builder.queue.is_empty() {
                return Ok(());
            }
            recipe_sources::expand(root, builder, visited)?;
        }
    }
}
