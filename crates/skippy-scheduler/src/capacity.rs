use std::cmp::Ordering;

/// One releasable cache entry considered by capacity-aware admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvictableCacheEntry {
    pub id: String,
    pub units: u64,
    pub recompute_cost: u64,
    pub last_used: u64,
}

/// Current occupancy for one independently constrained memory component.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComponentCapacitySnapshot {
    pub component: String,
    pub capacity_units: u64,
    pub active_units: u64,
    pub pinned_cache_units: u64,
    pub evictable_entries: Vec<EvictableCacheEntry>,
}

/// Capacity required before a request may enter execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapacityDemand {
    pub request_units: u64,
    /// Admission must leave at least this much free space after eviction.
    pub minimum_free_units: u64,
    /// Healthy operation prefers this much free space when entries are
    /// releasable. This must be greater than or equal to the minimum.
    pub target_free_units: u64,
}

/// Deterministic admission and eviction decision for one component.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapacityPlan {
    pub component: String,
    pub admitted: bool,
    pub victim_ids: Vec<String>,
    pub required_eviction_units: u64,
    pub evicted_units: u64,
    pub predicted_recompute_cost: u64,
    pub projected_free_units: u64,
    pub admission_deficit_units: u64,
}

/// Admit against pinned-versus-evictable capacity and select the cheapest
/// cache entries to recompute until the target free-space watermark is met.
///
/// Pinned cache units are never candidates. Entries are ordered by
/// recomputation cost per released unit, then recency, total cost, and stable
/// identity. Recency precedes total cost when density ties because uniform
/// per-token stage costs would otherwise prefer small, recently used prefixes
/// over cold entries that are less likely to be restored again. The plan
/// rejects only when active + pinned + request + minimum
/// headroom cannot fit even after every evictable entry is released.
pub fn plan_component_capacity(
    snapshot: &ComponentCapacitySnapshot,
    demand: CapacityDemand,
) -> CapacityPlan {
    let minimum_free = demand.minimum_free_units.min(snapshot.capacity_units);
    let target_free = demand
        .target_free_units
        .max(minimum_free)
        .min(snapshot.capacity_units);
    let protected_units = snapshot
        .active_units
        .saturating_add(snapshot.pinned_cache_units)
        .saturating_add(demand.request_units);
    let protected_with_minimum = protected_units.saturating_add(minimum_free);
    if protected_with_minimum > snapshot.capacity_units {
        return CapacityPlan {
            component: snapshot.component.clone(),
            admitted: false,
            victim_ids: Vec::new(),
            required_eviction_units: 0,
            evicted_units: 0,
            predicted_recompute_cost: 0,
            projected_free_units: snapshot.capacity_units.saturating_sub(protected_units),
            admission_deficit_units: protected_with_minimum.saturating_sub(snapshot.capacity_units),
        };
    }

    let evictable_units = snapshot
        .evictable_entries
        .iter()
        .map(|entry| entry.units)
        .fold(0u64, u64::saturating_add);
    let projected_occupancy = protected_units.saturating_add(evictable_units);
    let eviction_target = projected_occupancy
        .saturating_add(target_free)
        .saturating_sub(snapshot.capacity_units);
    let mut candidates = snapshot.evictable_entries.clone();
    candidates.sort_by(compare_eviction_value);
    let mut victim_ids = Vec::new();
    let mut evicted_units = 0u64;
    let mut predicted_recompute_cost = 0u64;
    for entry in candidates {
        if evicted_units >= eviction_target {
            break;
        }
        if entry.units == 0 {
            continue;
        }
        victim_ids.push(entry.id);
        evicted_units = evicted_units.saturating_add(entry.units);
        predicted_recompute_cost = predicted_recompute_cost.saturating_add(entry.recompute_cost);
    }
    CapacityPlan {
        component: snapshot.component.clone(),
        admitted: true,
        victim_ids,
        required_eviction_units: eviction_target,
        evicted_units,
        predicted_recompute_cost,
        projected_free_units: snapshot
            .capacity_units
            .saturating_sub(projected_occupancy.saturating_sub(evicted_units)),
        admission_deficit_units: 0,
    }
}

fn compare_eviction_value(left: &EvictableCacheEntry, right: &EvictableCacheEntry) -> Ordering {
    let left_units = left.units.max(1);
    let right_units = right.units.max(1);
    u128::from(left.recompute_cost)
        .saturating_mul(u128::from(right_units))
        .cmp(&u128::from(right.recompute_cost).saturating_mul(u128::from(left_units)))
        .then_with(|| left.last_used.cmp(&right.last_used))
        .then_with(|| left.recompute_cost.cmp(&right.recompute_cost))
        .then_with(|| left.id.cmp(&right.id))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: &str, units: u64, recompute_cost: u64, last_used: u64) -> EvictableCacheEntry {
        EvictableCacheEntry {
            id: id.to_string(),
            units,
            recompute_cost,
            last_used,
        }
    }

    #[test]
    fn rejects_when_pinned_capacity_cannot_leave_the_minimum_watermark() {
        let plan = plan_component_capacity(
            &ComponentCapacitySnapshot {
                component: "kv-cells".into(),
                capacity_units: 100,
                active_units: 50,
                pinned_cache_units: 40,
                evictable_entries: vec![entry("evictable", 10, 10, 0)],
            },
            CapacityDemand {
                request_units: 5,
                minimum_free_units: 10,
                target_free_units: 20,
            },
        );

        assert!(!plan.admitted);
        assert_eq!(plan.admission_deficit_units, 5);
        assert!(plan.victim_ids.is_empty());
    }

    #[test]
    fn evicts_low_recompute_cost_before_older_expensive_entries() {
        let plan = plan_component_capacity(
            &ComponentCapacitySnapshot {
                component: "kv-cells".into(),
                capacity_units: 100,
                active_units: 30,
                pinned_cache_units: 0,
                evictable_entries: vec![
                    entry("old-expensive", 30, 300, 1),
                    entry("new-cheap", 30, 30, 2),
                ],
            },
            CapacityDemand {
                request_units: 20,
                minimum_free_units: 10,
                target_free_units: 20,
            },
        );

        assert!(plan.admitted);
        assert_eq!(plan.victim_ids, ["new-cheap"]);
        assert_eq!(plan.evicted_units, 30);
        assert_eq!(plan.predicted_recompute_cost, 30);
        assert_eq!(plan.projected_free_units, 20);
    }

    #[test]
    fn equal_cost_density_prefers_cold_entries_before_smaller_hot_entries() {
        let plan = plan_component_capacity(
            &ComponentCapacitySnapshot {
                component: "kv-cells".into(),
                capacity_units: 100,
                active_units: 30,
                pinned_cache_units: 0,
                evictable_entries: vec![
                    entry("cold-large", 30, 300, 1),
                    entry("hot-small", 10, 100, 2),
                ],
            },
            CapacityDemand {
                request_units: 20,
                minimum_free_units: 10,
                target_free_units: 20,
            },
        );

        assert!(plan.admitted);
        assert_eq!(plan.victim_ids, ["cold-large"]);
        assert_eq!(plan.evicted_units, 30);
        assert_eq!(plan.predicted_recompute_cost, 300);
    }

    #[test]
    fn target_watermark_evicts_when_the_hard_minimum_was_already_met() {
        let plan = plan_component_capacity(
            &ComponentCapacitySnapshot {
                component: "kv-cells".into(),
                capacity_units: 100,
                active_units: 30,
                pinned_cache_units: 0,
                evictable_entries: vec![entry("cheap", 20, 20, 0)],
            },
            CapacityDemand {
                request_units: 20,
                minimum_free_units: 10,
                target_free_units: 40,
            },
        );

        assert!(plan.admitted);
        assert_eq!(plan.victim_ids, ["cheap"]);
        assert_eq!(plan.projected_free_units, 50);
    }

    #[test]
    fn keeps_every_entry_when_the_target_watermark_is_already_met() {
        let plan = plan_component_capacity(
            &ComponentCapacitySnapshot {
                component: "kv-cells".into(),
                capacity_units: 100,
                active_units: 20,
                pinned_cache_units: 0,
                evictable_entries: vec![entry("keep", 20, 20, 0)],
            },
            CapacityDemand {
                request_units: 10,
                minimum_free_units: 10,
                target_free_units: 30,
            },
        );

        assert!(plan.admitted);
        assert!(plan.victim_ids.is_empty());
        assert_eq!(plan.projected_free_units, 50);
    }
}
