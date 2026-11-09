//! This stage's cache tiers as the resource sampler reads them: weak handles,
//! so the sampler never keeps a stage alive, and a non-blocking radix lock,
//! so it never waits behind the scheduler. Cache decisions also become the
//! `skippy.kv.*` events.

use std::sync::{
    Arc, Mutex, PoisonError, Weak,
    atomic::{AtomicU64, AtomicUsize, Ordering},
};

use skippy_cache::{L2Tier, L3Tier, UnifiedRadixCache, UnifiedRadixCacheStats};

use super::{KvLifecycleEvent, KvStageIntegration, RadixExactEntry, RadixResidentEntry};
use crate::telemetry::log_events::{self, KvAdmissionShortfall, KvEviction, KvLookup, KvTier};
use crate::telemetry::resource_metrics::kv_tiers::{
    self, KvTierReading, KvTierSource, L1Reading, L2Reading, L3Reading, RecordQueueReading,
};

type Radix = Mutex<UnifiedRadixCache<RadixResidentEntry, RadixExactEntry>>;

struct StageTierSource {
    radix: Weak<Radix>,
    recurrent_limit_bytes: u64,
    l2: Option<Weak<L2Tier>>,
    l3: Option<Weak<L3Tier>>,
    records_pending: Weak<AtomicUsize>,
    record_queue_bytes: Weak<AtomicU64>,
    records_dropped: Weak<AtomicU64>,
    /// The latest L1 reading, reused while the radix lock is held elsewhere.
    last_l1: Mutex<L1Reading>,
}

/// Make `integration`'s tiers visible to the resource sampler.
pub(super) fn register(integration: &KvStageIntegration) {
    kv_tiers::register(Box::new(StageTierSource {
        radix: Arc::downgrade(&integration.radix),
        recurrent_limit_bytes: integration.resident_checkpoint_max_bytes,
        l2: integration.l2.as_ref().map(|l2| Arc::downgrade(l2.tier())),
        l3: integration.l3.as_ref().map(Arc::downgrade),
        records_pending: Arc::downgrade(&integration.exact_state_records_pending),
        record_queue_bytes: Arc::downgrade(&integration.exact_state_record_queue_bytes),
        records_dropped: Arc::downgrade(&integration.exact_state_records_dropped),
        last_l1: Mutex::new(L1Reading::default()),
    }));
}

/// Count an L1 decision for the tier gauges.
pub(super) fn record_lifecycle(event: KvLifecycleEvent) {
    match event {
        KvLifecycleEvent::CacheLookupHit { matched_tokens, .. } => {
            kv_tiers::record_l1_hit(matched_tokens as u64);
            log_events::record(&KvLookup {
                tier: KvTier::L1,
                matched_tokens: Some(matched_tokens as u64),
            });
        }
        KvLifecycleEvent::CacheLookupMiss => {
            kv_tiers::record_l1_miss();
            log_events::record(&KvLookup {
                tier: KvTier::L1,
                matched_tokens: None,
            });
        }
        KvLifecycleEvent::PrefixRestored {
            restored_tokens, ..
        } => kv_tiers::record_l1_restore(restored_tokens as u64),
        KvLifecycleEvent::CacheEviction {
            evicted_entries,
            evicted_tokens,
        } => {
            kv_tiers::record_l1_eviction(evicted_tokens);
            log_events::record(&KvEviction {
                tier: KvTier::L1,
                entries: evicted_entries as u64,
                tokens: evicted_tokens,
            });
        }
        KvLifecycleEvent::CapacityApproachingLimit {
            admission_deficit_tokens,
        } => {
            kv_tiers::record_l1_admission_shortfall();
            log_events::record(&KvAdmissionShortfall {
                deficit_tokens: admission_deficit_tokens,
            });
        }
        _ => {}
    }
}

/// A lookup in L2 or L3, which serving decides outside the L1 lifecycle.
pub(super) fn record_lookup(tier: KvTier, matched_tokens: Option<u64>) {
    log_events::record(&KvLookup {
        tier,
        matched_tokens,
    });
}

impl KvTierSource for StageTierSource {
    fn read(&self) -> Option<KvTierReading> {
        let radix = self.radix.upgrade()?;
        Some(KvTierReading {
            l1: self.l1(&radix),
            l2: self
                .l2
                .as_ref()
                .and_then(Weak::upgrade)
                .map(|tier| l2(&tier)),
            l3: self
                .l3
                .as_ref()
                .and_then(Weak::upgrade)
                .map(|tier| l3(&tier)),
            record_queue: RecordQueueReading {
                records: self
                    .records_pending
                    .upgrade()
                    .map_or(0, |pending| pending.load(Ordering::Relaxed) as u64),
                bytes: load(&self.record_queue_bytes),
                dropped: load(&self.records_dropped),
            },
        })
    }
}

impl StageTierSource {
    fn l1(&self, radix: &Radix) -> L1Reading {
        let mut last = self.last_l1.lock().unwrap_or_else(PoisonError::into_inner);
        if let Ok(radix) = radix.try_lock() {
            *last = l1(&radix.stats(), self.recurrent_limit_bytes);
        }
        *last
    }
}

fn load(counter: &Weak<AtomicU64>) -> u64 {
    counter
        .upgrade()
        .map_or(0, |counter| counter.load(Ordering::Relaxed))
}

fn l1(stats: &UnifiedRadixCacheStats, recurrent_limit_bytes: u64) -> L1Reading {
    L1Reading {
        entries: stats.resident_entries as u64,
        tokens: stats.resident_tokens,
        pinned_tokens: stats.resident_pinned_tokens,
        recurrent_bytes: stats.recurrent_logical_bytes,
        recurrent_limit_bytes,
        attention_evictions: stats.resident_evictions,
        recurrent_evictions: stats.recurrent_evictions,
    }
}

fn l2(tier: &L2Tier) -> L2Reading {
    let stats = tier.stats();
    L2Reading {
        entries: stats.entries,
        bytes: stats.bytes,
        budget_bytes: stats.budget_bytes,
        hits: stats.hits,
        misses: stats.misses,
        inserts: stats.inserts,
        evictions: stats.evictions,
        refusals: stats.admission_rejects,
    }
}

fn l3(tier: &L3Tier) -> L3Reading {
    let store = tier.store();
    let pressure = store.pressure();
    let activity = tier.manager().activity_without_inventory();
    L3Reading {
        store: std::ptr::from_ref(store) as usize,
        used_bytes: pressure.used_bytes,
        budget_bytes: pressure.budget_bytes,
        reserved_bytes: pressure.reserved_inflight_bytes,
        filesystem_available_bytes: pressure.filesystem_available_bytes,
        hits: activity.hits,
        misses: activity.misses,
        fills: activity.fills,
        writes: activity.writes,
        bytes_read: activity.bytes_read,
        bytes_written: activity.bytes_written,
        evictions: activity.evictions,
        corrupt: activity.corrupt_entries,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l1_readings_come_from_the_radix_stats_and_the_checkpoint_budget() {
        let stats = UnifiedRadixCacheStats {
            resident_entries: 3,
            resident_tokens: 96,
            resident_pinned_tokens: 32,
            recurrent_logical_bytes: 512,
            resident_evictions: 2,
            recurrent_evictions: 1,
            ..UnifiedRadixCacheStats::default()
        };
        let reading = l1(&stats, 4096);
        assert_eq!(
            reading,
            L1Reading {
                entries: 3,
                tokens: 96,
                pinned_tokens: 32,
                recurrent_bytes: 512,
                recurrent_limit_bytes: 4096,
                attention_evictions: 2,
                recurrent_evictions: 1,
            }
        );
    }

    #[test]
    fn a_dropped_stage_stops_reporting_and_a_held_lock_reuses_the_last_reading() {
        let radix: Arc<Radix> = Arc::new(Mutex::new(UnifiedRadixCache::new()));
        let source = StageTierSource {
            radix: Arc::downgrade(&radix),
            recurrent_limit_bytes: 64,
            l2: None,
            l3: None,
            records_pending: Weak::new(),
            record_queue_bytes: Weak::new(),
            records_dropped: Weak::new(),
            last_l1: Mutex::new(L1Reading::default()),
        };
        let first = source.read().expect("live stage");
        assert_eq!(first.l1.recurrent_limit_bytes, 64);
        assert!(first.l2.is_none() && first.l3.is_none());

        let held = radix.lock().expect("radix lock");
        assert_eq!(source.read().map(|reading| reading.l1), Some(first.l1));
        drop(held);

        drop(radix);
        assert!(source.read().is_none());
    }
}
