//! Complete alternating old/new schedules and exact cell admission.
use super::{acceptance::Version, telemetry::Cell};
use crate::command::DynResult;
use std::collections::BTreeSet;

pub(super) fn schedule(rounds: u64) -> DynResult<Vec<(u64, Version)>> {
    if !(1..=1000).contains(&rounds) {
        return Err("A/B rounds must be in 1..=1000".into());
    }
    let mut cells = Vec::new();
    for round in 1..=rounds {
        let versions = if round % 2 == 1 {
            [Version::Old, Version::New]
        } else {
            [Version::New, Version::Old]
        };
        for version in versions {
            cells.push((round, version));
        }
    }
    Ok(cells)
}

pub(super) fn complete(cells: &[Cell], rounds: u64, requests_per_round: u64) -> DynResult<()> {
    if requests_per_round == 0 {
        return Err("A/B cells require a positive request count".into());
    }
    let expected: BTreeSet<_> = schedule(rounds)?.into_iter().collect();
    let mut observed = BTreeSet::new();
    for cell in cells {
        if !observed.insert((cell.round, cell.version)) {
            return Err("duplicate measured A/B cell identity".into());
        }
        if cell.summary.requests != requests_per_round
            || cell.summary.successful != requests_per_round
            || !cell.summary.errors.is_empty()
        {
            return Err("A/B cell did not complete every admitted request".into());
        }
    }
    if observed != expected {
        return Err("A/B comparison requires every old/new cell for every round".into());
    }
    Ok(())
}

#[cfg(test)]
#[path = "rounds_tests.rs"]
mod tests;
