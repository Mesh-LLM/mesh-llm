//! Comparison records, their per-group counts and the evidence summary.

use crate::command::DynResult;
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Outcome {
    /// Every compared byte, stream and status agreed.
    Identical,
    /// A difference with a recorded, bounded explanation.
    Explained,
    /// Only the Rust side ran (no interpreter was requested); it succeeded.
    RustOnly,
    /// An unexplained difference or failure.
    Different,
}

#[derive(Debug, Serialize)]
pub(super) struct Comparison {
    pub(super) group: &'static str,
    pub(super) label: String,
    pub(super) outcome: Outcome,
    pub(super) detail: String,
}

/// Collects every comparison of one suite run.
#[derive(Default)]
pub(super) struct Ledger {
    pub(super) rows: Vec<Comparison>,
}

impl Ledger {
    pub(super) fn record(
        &mut self,
        group: &'static str,
        label: impl Into<String>,
        difference: Option<String>,
    ) {
        let (outcome, detail) = match difference {
            None => (Outcome::Identical, String::new()),
            Some(detail) => (Outcome::Different, detail),
        };
        self.push(group, label, outcome, detail);
    }

    pub(super) fn push(
        &mut self,
        group: &'static str,
        label: impl Into<String>,
        outcome: Outcome,
        detail: impl Into<String>,
    ) {
        self.rows.push(Comparison {
            group,
            label: label.into(),
            outcome,
            detail: detail.into(),
        });
    }

    pub(super) fn unexplained(&self) -> usize {
        self.rows
            .iter()
            .filter(|row| row.outcome == Outcome::Different)
            .count()
    }

    fn counts(&self) -> BTreeMap<&'static str, BTreeMap<Outcome, usize>> {
        let mut counts: BTreeMap<&'static str, BTreeMap<Outcome, usize>> = BTreeMap::new();
        for row in &self.rows {
            *counts
                .entry(row.group)
                .or_default()
                .entry(row.outcome)
                .or_default() += 1;
        }
        counts
    }

    /// The summary document: run context plus counts and every comparison.
    pub(super) fn summary(&self, context: Value) -> Value {
        let counts = self
            .counts()
            .into_iter()
            .map(|(group, outcomes)| {
                let outcomes = outcomes
                    .into_iter()
                    .map(|(outcome, count)| (outcome_name(outcome).to_owned(), json!(count)))
                    .collect::<serde_json::Map<_, _>>();
                (group.to_owned(), Value::Object(outcomes))
            })
            .collect::<serde_json::Map<_, _>>();
        let mut summary = json!({
            "suite": "ci",
            "counts": counts,
            "unexplained": self.unexplained(),
            "comparisons": self.rows,
        });
        if let (Value::Object(target), Value::Object(extra)) = (&mut summary, context) {
            target.extend(extra);
        }
        summary
    }

    /// One line per group for the terminal.
    pub(super) fn lines(&self) -> Vec<String> {
        let mut lines = self
            .counts()
            .into_iter()
            .map(|(group, outcomes)| {
                let parts = outcomes
                    .into_iter()
                    .map(|(outcome, count)| format!("{}={count}", outcome_name(outcome)))
                    .collect::<Vec<_>>();
                format!("{group}: {}", parts.join(" "))
            })
            .collect::<Vec<_>>();
        lines.extend(
            self.rows
                .iter()
                .filter(|row| row.outcome == Outcome::Different)
                .map(|row| format!("DIFFERENT {} {}: {}", row.group, row.label, row.detail)),
        );
        lines
    }
}

impl PartialOrd for Outcome {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Outcome {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        outcome_name(*self).cmp(outcome_name(*other))
    }
}

fn outcome_name(outcome: Outcome) -> &'static str {
    match outcome {
        Outcome::Identical => "identical",
        Outcome::Explained => "explained",
        Outcome::RustOnly => "rust_only",
        Outcome::Different => "different",
    }
}

pub(super) fn write_summary(directory: &Path, summary: &Value) -> DynResult<()> {
    crate::command::write_json_file(&directory.join("summary.json"), summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migration_ci_shadow_ledger_counts_only_differences_as_unexplained() {
        let mut ledger = Ledger::default();
        ledger.record("frozen-cases", "a", None);
        ledger.push("command-surface", "b", Outcome::Explained, "wording");
        ledger.record("frozen-cases", "c", Some("byte".to_owned()));
        assert_eq!(ledger.unexplained(), 1);
        let summary = ledger.summary(json!({"legacy": {"enabled": false}}));
        assert_eq!(summary["counts"]["frozen-cases"]["identical"], 1);
        assert_eq!(summary["counts"]["frozen-cases"]["different"], 1);
        assert_eq!(summary["legacy"]["enabled"], false);
    }
}
