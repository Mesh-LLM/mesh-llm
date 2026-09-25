use super::super::ledger::{BoundaryDisposition, BoundaryLedger, BoundaryRecord};
use super::super::scan::Candidate;
use super::Edge;
use crate::command::DynResult;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::ops::Deref;
use std::path::Path;

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub(in super::super) enum EdgeDisposition {
    Contract,
    Traversed,
    Nonexecution,
    ExternalTrustBoundary,
    PlatformConditional,
    RuntimeSelectedBoundary,
    BoundedSelector,
    FiniteTarget,
    Unresolved,
}

#[derive(Debug, Serialize)]
pub(in super::super) struct ClassifiedEdge {
    #[serde(flatten)]
    pub(in super::super) edge: Edge,
    pub(in super::super) occurrence: usize,
    pub(in super::super) candidate_id: Option<String>,
    pub(in super::super) disposition: EdgeDisposition,
    pub(in super::super) boundary: Option<BoundaryRecord>,
}

impl Deref for ClassifiedEdge {
    type Target = Edge;
    fn deref(&self) -> &Edge {
        &self.edge
    }
}

pub(super) struct Classified {
    pub(super) edges: Vec<ClassifiedEdge>,
    pub(super) counts: BTreeMap<EdgeDisposition, usize>,
    pub(super) unresolved: usize,
    pub(super) unbound_records: Vec<String>,
}

pub(super) struct Boundaries {
    pub(super) records: Vec<BoundaryRecord>,
    files: BTreeMap<String, Vec<String>>,
}

impl Boundaries {
    pub(super) fn load(root: &Path) -> DynResult<Self> {
        let file = root.join("ci/automation-migration/invocations.json");
        let records = if file.is_file() {
            serde_json::from_slice::<BoundaryLedger>(&fs::read(file)?)?.boundary_records
        } else {
            Vec::new()
        };
        let mut boundaries = Self {
            records,
            files: BTreeMap::new(),
        };
        let mut ids = BTreeSet::new();
        for record in boundaries.records.clone() {
            if !ids.insert(record.id.clone()) {
                return Err(format!("boundary record: duplicate {}", record.id).into());
            }
            boundaries.complete(root, &record)?;
        }
        Ok(boundaries)
    }

    pub(super) fn lines(&mut self, root: &Path, path: &str) -> &[String] {
        self.files.entry(path.to_owned()).or_insert_with(|| {
            fs::read_to_string(root.join(path))
                .map(|text| text.lines().map(|line| line.trim().to_owned()).collect())
                .unwrap_or_default()
        })
    }

    fn complete(&mut self, root: &Path, record: &BoundaryRecord) -> DynResult<()> {
        if [
            &record.selector,
            &record.reachable_bytes,
            &record.replacement_owner,
            &record.deletion_condition,
            &record.rationale,
        ]
        .iter()
        .any(|field| field.trim().is_empty())
            || record.occurrence == 0
            || record.evidence.is_empty()
        {
            return Err(
                format!("boundary record {}: missing owner or rationale", record.id).into(),
            );
        }
        for evidence in &record.evidence {
            let line = evidence
                .line
                .checked_sub(1)
                .and_then(|index| self.lines(root, &evidence.path).get(index).cloned());
            if line.as_deref() != Some(evidence.text.trim()) {
                return Err(format!(
                    "boundary record {}: stale evidence {}:{}",
                    record.id, evidence.path, evidence.line
                )
                .into());
            }
        }
        Ok(())
    }

    pub(super) fn occurrence(&mut self, root: &Path, edges: &[Edge], index: usize) -> usize {
        let edge = &edges[index];
        if edge.line == 0 {
            return edges[..=index]
                .iter()
                .filter(|row| {
                    row.parent == edge.parent
                        && row.source_block == edge.source_block
                        && row.child == edge.child
                })
                .count();
        }
        self.lines(root, &edge.parent)
            .iter()
            .take(edge.line)
            .filter(|line| **line == edge.source_block)
            .count()
    }

    pub(super) fn find(&self, edge: &Edge, occurrence: usize) -> Option<&BoundaryRecord> {
        self.records.iter().find(|record| {
            record.caller == edge.parent
                && record.line == edge.line
                && record.source_block == edge.source_block
                && record.occurrence == occurrence
                && record.child == edge.child
        })
    }

    pub(super) fn classify(
        &mut self,
        root: &Path,
        edges: Vec<Edge>,
        observed: &[Candidate],
        validated: &BTreeSet<String>,
    ) -> DynResult<Classified> {
        let occurrences = (0..edges.len())
            .map(|index| self.occurrence(root, &edges, index))
            .collect::<Vec<_>>();
        let mut used = BTreeSet::new();
        let mut result = Classified {
            edges: Vec::with_capacity(edges.len()),
            counts: BTreeMap::new(),
            unresolved: 0,
            unbound_records: Vec::new(),
        };
        for (edge, occurrence) in edges.into_iter().zip(occurrences) {
            let candidate_id = (edge.line != 0)
                .then(|| {
                    observed
                        .iter()
                        .filter(|row| {
                            row.path == edge.parent && row.source_block == edge.source_block
                        })
                        .nth(occurrence.saturating_sub(1))
                        .map(|row| row.id.clone())
                })
                .flatten();
            let record = self.find(&edge, occurrence).cloned();
            if let Some(record) = &record {
                bind(record, &edge, candidate_id.as_deref(), validated)?;
                used.insert(record.id.clone());
            }
            let disposition = match record.as_ref().map(|row| row.disposition) {
                Some(BoundaryDisposition::ExternalTrustBoundary) => {
                    EdgeDisposition::ExternalTrustBoundary
                }
                Some(BoundaryDisposition::PlatformConditional) => {
                    EdgeDisposition::PlatformConditional
                }
                Some(BoundaryDisposition::RuntimeSelectedBoundary) => {
                    EdgeDisposition::RuntimeSelectedBoundary
                }
                Some(BoundaryDisposition::BoundedSelector) => EdgeDisposition::BoundedSelector,
                Some(BoundaryDisposition::FiniteTarget) => EdgeDisposition::FiniteTarget,
                None if edge.unresolved_reason.is_some() => EdgeDisposition::Unresolved,
                None if matches!(edge.status, "reference_only" | "provisioning_selection") => {
                    EdgeDisposition::Nonexecution
                }
                None if edge.contract_source.is_some() => EdgeDisposition::Contract,
                None => EdgeDisposition::Traversed,
            };
            *result.counts.entry(disposition).or_default() += 1;
            result.unresolved += usize::from(disposition == EdgeDisposition::Unresolved);
            result.edges.push(ClassifiedEdge {
                edge,
                occurrence,
                candidate_id,
                disposition,
                boundary: record,
            });
        }
        result.unbound_records = self
            .records
            .iter()
            .filter(|record| !used.contains(&record.id))
            .map(|record| format!("{} ({}:{})", record.id, record.caller, record.line))
            .collect();
        Ok(result)
    }
}

fn bind(
    record: &BoundaryRecord,
    edge: &Edge,
    expected: Option<&str>,
    validated: &BTreeSet<String>,
) -> DynResult<()> {
    if record.candidate_id.as_deref() != expected
        || record
            .candidate_id
            .as_ref()
            .is_some_and(|id| !validated.contains(id))
    {
        return Err(format!("boundary record {}: candidate ID does not match", record.id).into());
    }
    let compatible = edge.unresolved_reason.is_some()
        && match record.disposition {
            BoundaryDisposition::ExternalTrustBoundary => {
                edge.trust_revision == "protected_main_external"
            }
            BoundaryDisposition::PlatformConditional => {
                matches!(edge.status, "optional_branch" | "unknown_selection")
            }
            BoundaryDisposition::RuntimeSelectedBoundary
            | BoundaryDisposition::BoundedSelector
            | BoundaryDisposition::FiniteTarget => edge.status == "unknown_selection",
        };
    if !compatible {
        return Err(format!(
            "boundary record {}: disposition does not fit edge status {}",
            record.id, edge.status
        )
        .into());
    }
    Ok(())
}
