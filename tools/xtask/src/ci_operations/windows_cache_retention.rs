//! Pure dry-run retention planning. No GitHub transport or deletion authority.

#[path = "windows_cache_retention_identity.rs"]
mod identity;
#[path = "windows_cache_retention_input.rs"]
mod input;

use crate::{command::DynResult, repository::check_report::CheckReport};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

pub(super) use identity::{Family, compatibility, family};
pub(super) use input::Policy;
use input::{Cache, Candidate, Inventory};

#[derive(Debug, Serialize)]
pub(super) struct Plan {
    pub(super) schema_version: u32,
    pub(super) dry_run: bool,
    pub(super) target_ref: String,
    pub(super) max_bytes: u64,
    pub(super) inventory_bytes: u64,
    pub(super) retained_bytes: u64,
    pub(super) deleted_bytes: u64,
    pub(super) untouched_bytes: u64,
    pub(super) missing_families: Vec<Family>,
    pub(super) retain: Vec<Cache>,
    pub(super) delete: Vec<Cache>,
    pub(super) untouched: Vec<Cache>,
}

pub(super) fn run(args: &[String]) -> CheckReport {
    const USAGE: &str = "usage: ci-ops windows-cache-retention --inventory FILE --policy FILE\n";
    if args == ["--help"] {
        return CheckReport::success(USAGE.into());
    }
    let execute = || -> DynResult<String> {
        let [inventory_flag, inventory, policy_flag, policy] = args else {
            return Err(USAGE.into());
        };
        if inventory_flag != "--inventory" || policy_flag != "--policy" {
            return Err(USAGE.into());
        }
        let plan = plan_json(
            &input::read_bounded(inventory)?,
            &input::read_bounded(policy)?,
        )?;
        Ok(format!("{}\n", serde_json::to_string_pretty(&plan)?))
    };
    match execute() {
        Ok(json) => CheckReport::success(json),
        Err(error) => {
            CheckReport::failure(String::new(), format!("Windows cache retention: {error}\n"))
        }
    }
}

pub(super) fn plan_json(inventory: &[u8], policy: &[u8]) -> DynResult<Plan> {
    input::check_size(inventory)?;
    input::check_size(policy)?;
    let policy: Policy = serde_json::from_slice(policy)?;
    let inventory: Inventory = serde_json::from_slice(inventory)?;
    plan(inventory, policy)
}

fn plan(inventory: Inventory, policy: Policy) -> DynResult<Plan> {
    let active = input::active(&policy)?;
    let (candidates, mut untouched, inventory_bytes) = input::candidates(inventory, &policy)?;
    let mut groups = BTreeMap::<Family, Vec<Candidate>>::new();
    for candidate in candidates {
        groups
            .entry(candidate.family.clone())
            .or_default()
            .push(candidate);
    }
    let mut retained = BTreeSet::new();
    let mut retained_bytes = 0_u64;
    let mut history = Vec::new();
    let mut missing_families = Vec::new();
    for entries in groups.values_mut() {
        entries.sort_by(|a, b| {
            b.created
                .cmp(&a.created)
                .then_with(|| b.cache.id.cmp(&a.cache.id))
        });
    }
    for (family, current) in &active {
        let Some(entries) = groups.get(family) else {
            missing_families.push(family.clone());
            continue;
        };
        let useful = entries
            .iter()
            .position(|entry| current.contains(&entry.cache.key))
            .unwrap_or(0);
        let mandatory = &entries[useful];
        retained_bytes = retained_bytes
            .checked_add(mandatory.cache.size_in_bytes)
            .ok_or("retained bytes overflow")?;
        retained.insert(mandatory.cache.id);
        history.extend(
            entries
                .iter()
                .filter(|entry| entry.cache.id != mandatory.cache.id),
        );
    }
    if retained_bytes > policy.max_bytes {
        return Err(
            "minimum useful active families exceed byte budget; no deletion plan produced".into(),
        );
    }
    retain_history(&history, &policy, &mut retained, &mut retained_bytes)?;
    let mut retain = Vec::new();
    let mut delete = Vec::new();
    for candidate in groups.into_values().flatten() {
        if retained.contains(&candidate.cache.id) {
            retain.push(candidate.cache);
        } else {
            delete.push(candidate.cache);
        }
    }
    retain.sort_by_key(|cache| cache.id);
    delete.sort_by_key(|cache| cache.id);
    untouched.sort_by_key(|cache| cache.id);
    Ok(Plan {
        schema_version: 1,
        dry_run: true,
        target_ref: policy.target_ref,
        max_bytes: policy.max_bytes,
        inventory_bytes,
        retained_bytes,
        deleted_bytes: byte_sum(&delete)?,
        untouched_bytes: byte_sum(&untouched)?,
        missing_families,
        retain,
        delete,
        untouched,
    })
}

fn retain_history(
    history: &[&Candidate],
    policy: &Policy,
    retained: &mut BTreeSet<u64>,
    bytes: &mut u64,
) -> DynResult<()> {
    let mut history = history.to_vec();
    history.sort_by(|a, b| {
        b.created
            .cmp(&a.created)
            .then_with(|| b.cache.id.cmp(&a.cache.id))
    });
    let mut counts = BTreeMap::<&Family, usize>::new();
    for candidate in history {
        let count = counts.entry(&candidate.family).or_insert(1);
        let next_bytes = bytes
            .checked_add(candidate.cache.size_in_bytes)
            .ok_or("retained bytes overflow")?;
        if *count < policy.history_per_family && next_bytes <= policy.max_bytes {
            retained.insert(candidate.cache.id);
            *bytes = next_bytes;
            *count = count.checked_add(1).ok_or("history count overflow")?;
        }
    }
    Ok(())
}

fn byte_sum(caches: &[Cache]) -> DynResult<u64> {
    caches.iter().try_fold(0_u64, |total, cache| {
        total
            .checked_add(cache.size_in_bytes)
            .ok_or_else(|| "cache byte total overflow".into())
    })
}

#[cfg(test)]
#[path = "windows_cache_retention_tests.rs"]
mod tests;
