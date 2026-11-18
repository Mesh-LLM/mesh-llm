use super::super::ci_metrics_time::{Instant, timestamp};
use super::identity::NAMESPACE;
use super::{Family, family};
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;

pub(super) const INPUT_LIMIT: usize = 16 * 1024 * 1024;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::ci_operations) struct Policy {
    pub(super) schema_version: u32,
    pub(super) target_ref: String,
    /// Must cover all current CI, release and warmer consumers, not only warmed rows.
    pub(super) current_keys: Vec<String>,
    pub(super) active_families_complete: bool,
    pub(super) max_bytes: u64,
    /// Total retained entries per family, including its mandatory useful member.
    pub(super) history_per_family: usize,
}

#[derive(Debug, Deserialize)]
pub(super) struct Inventory {
    pub(super) total_count: usize,
    pub(super) actions_caches: Vec<Cache>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(in crate::ci_operations) struct Cache {
    pub(in crate::ci_operations) id: u64,
    pub(super) key: String,
    pub(super) version: String,
    #[serde(rename = "ref")]
    pub(super) reference: String,
    pub(super) size_in_bytes: u64,
    pub(super) created_at: String,
    pub(super) last_accessed_at: String,
}

pub(super) struct Candidate {
    pub(super) cache: Cache,
    pub(super) family: Family,
    pub(super) created: Instant,
}

pub(super) fn read_bounded(path: &str) -> DynResult<Vec<u8>> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take(
            u64::try_from(INPUT_LIMIT)?
                .checked_add(1)
                .ok_or("input limit overflow")?,
        )
        .read_to_end(&mut bytes)?;
    check_size(&bytes)?;
    Ok(bytes)
}

pub(super) fn check_size(bytes: &[u8]) -> DynResult<()> {
    if bytes.len() > INPUT_LIMIT {
        return Err("Windows cache input exceeds 16 MiB".into());
    }
    Ok(())
}

pub(super) fn active(policy: &Policy) -> DynResult<BTreeMap<Family, BTreeSet<String>>> {
    if policy.schema_version != 1
        || policy.target_ref != "refs/heads/main"
        || !policy.active_families_complete
        || policy.current_keys.is_empty()
        || policy.max_bytes == 0
        || policy.history_per_family == 0
    {
        return Err("retention requires schema 1, exact main ref, complete active keys and positive budgets".into());
    }
    let mut result = BTreeMap::<Family, BTreeSet<String>>::new();
    for key in &policy.current_keys {
        if !result.entry(family(key)?).or_default().insert(key.clone()) {
            return Err("duplicate current cache key".into());
        }
    }
    Ok(result)
}

pub(super) fn candidates(
    inventory: Inventory,
    policy: &Policy,
) -> DynResult<(Vec<Candidate>, Vec<Cache>, u64)> {
    if inventory.total_count != inventory.actions_caches.len() {
        return Err("incomplete cache inventory: total_count differs from supplied entries".into());
    }
    let mut seen = BTreeSet::new();
    let mut owned = Vec::new();
    let mut untouched = Vec::new();
    let mut total = 0_u64;
    for cache in inventory.actions_caches {
        if cache.id == 0 || !seen.insert(cache.id) || cache.version.is_empty() {
            return Err("cache inventory has zero/duplicate id or missing version".into());
        }
        total = total
            .checked_add(cache.size_in_bytes)
            .ok_or("cache byte total overflow")?;
        let created = cache_time(&cache.created_at)?;
        let accessed = cache_time(&cache.last_accessed_at)?;
        if accessed < created {
            return Err("cache access timestamp predates creation".into());
        }
        if cache.reference == policy.target_ref && cache.key.starts_with(NAMESPACE) {
            owned.push(Candidate {
                family: family(&cache.key)?,
                cache,
                created,
            });
        } else {
            untouched.push(cache);
        }
    }
    Ok((owned, untouched, total))
}

fn cache_time(value: &str) -> DynResult<Instant> {
    if !value.ends_with('Z') && !value.contains('+') {
        return Err("cache timestamp requires an explicit timezone".into());
    }
    timestamp(value)
        .map_err(|_| "invalid cache timestamp")?
        .ok_or_else(|| "cache timestamp predates GitHub Actions".into())
}
