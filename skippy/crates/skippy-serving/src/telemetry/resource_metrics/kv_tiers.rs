//! KV cache tier gauges. L1 is the resident radix cache: prefixes kept as
//! shared cells of the native attention pool, plus recurrent checkpoints in
//! host RAM. L2 is the optional host-RAM mirror of L3 fills. L3 is the
//! durable store. Each tier reports its pressure (how full it is against its
//! limit, and how often it evicts or refuses) and its lookups, split into
//! hits and misses.
//!
//! Each cache-enabled stage registers a source the sampler reads without
//! keeping the stage alive. L1 lookups are counted as the cache decides them;
//! the native pool's size comes from the scheduler's step sampling.

use std::{
    collections::BTreeMap,
    sync::{
        Mutex, PoisonError,
        atomic::{AtomicU64, Ordering},
    },
};

use opentelemetry_proto::tonic::metrics::v1::Metric;
use serde_json::{Value, json};

use super::{
    UNSET, current,
    otlp_metrics::{gauge, point},
    rate,
};

static L1_HITS: AtomicU64 = AtomicU64::new(0);
static L1_MISSES: AtomicU64 = AtomicU64::new(0);
static L1_HIT_TOKENS: AtomicU64 = AtomicU64::new(0);
static L1_RESTORED_TOKENS: AtomicU64 = AtomicU64::new(0);
static L1_EVICTED_TOKENS: AtomicU64 = AtomicU64::new(0);
static L1_SHORTFALLS: AtomicU64 = AtomicU64::new(0);
static POOL_CELLS: AtomicU64 = AtomicU64::new(UNSET);

/// An L1 lookup found a usable resident prefix of `matched_tokens`.
pub(crate) fn record_l1_hit(matched_tokens: u64) {
    L1_HITS.fetch_add(1, Ordering::Relaxed);
    L1_HIT_TOKENS.fetch_add(matched_tokens, Ordering::Relaxed);
}

pub(crate) fn record_l1_miss() {
    L1_MISSES.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_l1_restore(tokens: u64) {
    L1_RESTORED_TOKENS.fetch_add(tokens, Ordering::Relaxed);
}

pub(crate) fn record_l1_eviction(tokens: u64) {
    L1_EVICTED_TOKENS.fetch_add(tokens, Ordering::Relaxed);
}

/// Admission could not keep the pool's free-space watermark even after
/// evicting every releasable prefix. Each shortfall's size belongs to its
/// `skippy.kv.admission_shortfall` event; the gauges count how often.
pub(crate) fn record_l1_admission_shortfall() {
    L1_SHORTFALLS.fetch_add(1, Ordering::Relaxed);
}

/// Cells in the shared native KV pool; zero is an unknown size.
pub(crate) fn record_kv_pool_cells(cells: u64) {
    if cells > 0 {
        POOL_CELLS.store(cells, Ordering::Relaxed);
    }
}

/// One stage's resident cache. Counters are cumulative since it started.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct L1Reading {
    pub(crate) entries: u64,
    /// Tokens of the retained prefixes, counting a shared prefix once per
    /// entry that extends it; the pool's used cells count it once.
    pub(crate) tokens: u64,
    /// Retained tokens a restore holds, which eviction cannot release.
    pub(crate) pinned_tokens: u64,
    pub(crate) recurrent_bytes: u64,
    pub(crate) recurrent_limit_bytes: u64,
    pub(crate) attention_evictions: u64,
    pub(crate) recurrent_evictions: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct L2Reading {
    pub(crate) entries: u64,
    pub(crate) bytes: u64,
    pub(crate) budget_bytes: u64,
    pub(crate) hits: u64,
    pub(crate) misses: u64,
    pub(crate) inserts: u64,
    pub(crate) evictions: u64,
    pub(crate) refusals: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct L3Reading {
    /// Identifies the store. Stages on one node share it, so it counts once.
    pub(crate) store: usize,
    pub(crate) used_bytes: Option<u64>,
    pub(crate) budget_bytes: u64,
    pub(crate) reserved_bytes: u64,
    pub(crate) filesystem_available_bytes: Option<u64>,
    pub(crate) hits: u64,
    pub(crate) misses: u64,
    pub(crate) fills: u64,
    pub(crate) writes: u64,
    pub(crate) bytes_read: u64,
    pub(crate) bytes_written: u64,
    pub(crate) evictions: u64,
    pub(crate) corrupt: u64,
}

/// Exact-state records waiting for the worker that publishes them to L1 and
/// writes them through to L3.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct RecordQueueReading {
    pub(crate) records: u64,
    pub(crate) bytes: u64,
    pub(crate) dropped: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct KvTierReading {
    pub(crate) l1: L1Reading,
    pub(crate) l2: Option<L2Reading>,
    pub(crate) l3: Option<L3Reading>,
    pub(crate) record_queue: RecordQueueReading,
}

pub(crate) trait KvTierSource: Send + Sync {
    /// The stage's tiers now; `None` once the stage is gone.
    fn read(&self) -> Option<KvTierReading>;
}

static SOURCES: Mutex<Vec<Box<dyn KvTierSource>>> = Mutex::new(Vec::new());

pub(crate) fn register(source: Box<dyn KvTierSource>) {
    SOURCES
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .push(source);
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct L1Lookups {
    hits: u64,
    misses: u64,
    hit_tokens: u64,
    restored_tokens: u64,
    evicted_tokens: u64,
    shortfalls: u64,
}

impl L1Lookups {
    fn now() -> Self {
        Self {
            hits: L1_HITS.load(Ordering::Relaxed),
            misses: L1_MISSES.load(Ordering::Relaxed),
            hit_tokens: L1_HIT_TOKENS.load(Ordering::Relaxed),
            restored_tokens: L1_RESTORED_TOKENS.load(Ordering::Relaxed),
            evicted_tokens: L1_EVICTED_TOKENS.load(Ordering::Relaxed),
            shortfalls: L1_SHORTFALLS.load(Ordering::Relaxed),
        }
    }
}

/// Every stage's tiers summed, with the process-wide L1 lookup counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct KvTiers {
    stages: u64,
    l1: L1Reading,
    lookups: L1Lookups,
    l2: Option<L2Reading>,
    l3: Option<L3Reading>,
    record_queue: RecordQueueReading,
    pool_cells: Option<u64>,
}

impl KvTiers {
    pub(super) fn now() -> Self {
        let mut readings = Vec::new();
        SOURCES
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .retain(|source| match source.read() {
                Some(reading) => {
                    readings.push(reading);
                    true
                }
                None => false,
            });
        Self {
            lookups: L1Lookups::now(),
            pool_cells: current(&POOL_CELLS),
            ..Self::sum(readings)
        }
    }

    fn sum(readings: impl IntoIterator<Item = KvTierReading>) -> Self {
        readings
            .into_iter()
            .fold(Self::default(), |mut tiers, reading| {
                tiers.stages += 1;
                tiers.l1 = tiers.l1.plus(reading.l1);
                if let Some(l2) = reading.l2 {
                    tiers.l2 = Some(tiers.l2.unwrap_or_default().plus(l2));
                }
                if let Some(l3) = reading.l3 {
                    tiers.l3 = Some(match tiers.l3 {
                        Some(sum) if sum.store == l3.store => sum,
                        Some(sum) => sum.plus(l3),
                        None => l3,
                    });
                }
                tiers.record_queue = tiers.record_queue.plus(reading.record_queue);
                tiers
            })
    }
}

impl L1Reading {
    fn plus(self, other: Self) -> Self {
        Self {
            entries: self.entries + other.entries,
            tokens: self.tokens + other.tokens,
            pinned_tokens: self.pinned_tokens + other.pinned_tokens,
            recurrent_bytes: self.recurrent_bytes + other.recurrent_bytes,
            recurrent_limit_bytes: self.recurrent_limit_bytes + other.recurrent_limit_bytes,
            attention_evictions: self.attention_evictions + other.attention_evictions,
            recurrent_evictions: self.recurrent_evictions + other.recurrent_evictions,
        }
    }
}

impl L2Reading {
    fn plus(self, other: Self) -> Self {
        Self {
            entries: self.entries + other.entries,
            bytes: self.bytes + other.bytes,
            budget_bytes: self.budget_bytes + other.budget_bytes,
            hits: self.hits + other.hits,
            misses: self.misses + other.misses,
            inserts: self.inserts + other.inserts,
            evictions: self.evictions + other.evictions,
            refusals: self.refusals + other.refusals,
        }
    }
}

impl L3Reading {
    /// Two different stores; the free space is the tighter filesystem's.
    fn plus(self, other: Self) -> Self {
        Self {
            store: self.store,
            used_bytes: self.used_bytes.zip(other.used_bytes).map(|(a, b)| a + b),
            budget_bytes: self.budget_bytes + other.budget_bytes,
            reserved_bytes: self.reserved_bytes + other.reserved_bytes,
            filesystem_available_bytes: match (
                self.filesystem_available_bytes,
                other.filesystem_available_bytes,
            ) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            },
            hits: self.hits + other.hits,
            misses: self.misses + other.misses,
            fills: self.fills + other.fills,
            writes: self.writes + other.writes,
            bytes_read: self.bytes_read + other.bytes_read,
            bytes_written: self.bytes_written + other.bytes_written,
            evictions: self.evictions + other.evictions,
            corrupt: self.corrupt + other.corrupt,
        }
    }
}

impl RecordQueueReading {
    fn plus(self, other: Self) -> Self {
        Self {
            records: self.records + other.records,
            bytes: self.bytes + other.bytes,
            dropped: self.dropped + other.dropped,
        }
    }
}

/// Appends gauges for one sampling interval.
struct Gauges<'a> {
    metrics: &'a mut Vec<Metric>,
    seconds: f64,
    time: u64,
}

impl Gauges<'_> {
    fn level(&mut self, name: &str, unit: &str, description: &str, value: u64) {
        self.metrics.push(gauge(
            name,
            unit,
            description,
            vec![point(self.time, value as f64, &[])],
        ));
    }

    /// `used / limit`, omitted for an unknown or unlimited `limit`.
    fn utilization(&mut self, name: &str, description: &str, used: u64, limit: u64) {
        if limit > 0 {
            self.metrics.push(gauge(
                name,
                "1",
                description,
                vec![point(self.time, used as f64 / limit as f64, &[])],
            ));
        }
    }

    fn per_second(&mut self, name: &str, unit: &str, description: &str, was: u64, now: u64) {
        self.metrics.push(gauge(
            name,
            unit,
            description,
            vec![point(self.time, rate(was, now, self.seconds), &[])],
        ));
    }

    /// One rate per `(attribute value, was, now)`, under attribute `key`.
    fn per_second_by(
        &mut self,
        name: &str,
        unit: &str,
        description: &str,
        key: &str,
        values: &[(&str, u64, u64)],
    ) {
        let points = values
            .iter()
            .map(|&(value, was, now)| {
                point(self.time, rate(was, now, self.seconds), &[(key, value)])
            })
            .collect();
        self.metrics.push(gauge(name, unit, description, points));
    }

    /// Lookups per second split by result, and the share that missed when
    /// there were any.
    fn lookups(&mut self, tier: &str, hits: (u64, u64), misses: (u64, u64)) {
        self.per_second_by(
            &format!("skippy.kv.{tier}.lookups"),
            "{lookup}/s",
            "Cache lookups at this tier, by result",
            "result",
            &[("hit", hits.0, hits.1), ("miss", misses.0, misses.1)],
        );
        let hit = hits.1.saturating_sub(hits.0);
        let missed = misses.1.saturating_sub(misses.0);
        if hit + missed > 0 {
            self.metrics.push(gauge(
                &format!("skippy.kv.{tier}.miss_ratio"),
                "1",
                "Share of this interval's lookups at this tier that missed",
                vec![point(self.time, missed as f64 / (hit + missed) as f64, &[])],
            ));
        }
    }
}

/// The tier gauges for the interval from `before` to `after`.
pub(super) fn gauges(
    metrics: &mut Vec<Metric>,
    before: &KvTiers,
    after: &KvTiers,
    kv_used_cells: Option<u64>,
    seconds: f64,
    time: u64,
) {
    if after.stages == 0 {
        return;
    }
    let mut out = Gauges {
        metrics,
        seconds,
        time,
    };
    l1_gauges(&mut out, before, after, kv_used_cells);
    if let Some(l2) = after.l2 {
        l2_gauges(&mut out, before.l2, l2);
    }
    if let Some(l3) = after.l3 {
        l3_gauges(&mut out, before.l3, l3);
    }
    record_queue_gauges(&mut out, before.record_queue, after.record_queue);
}

fn l1_gauges(out: &mut Gauges<'_>, before: &KvTiers, after: &KvTiers, kv_used_cells: Option<u64>) {
    if let Some(limit) = after.pool_cells {
        out.level(
            "skippy.kv.l1.pool.limit",
            "{cell}",
            "Cells in the shared native KV pool",
            limit,
        );
        if let Some(used) = kv_used_cells {
            out.utilization(
                "skippy.kv.l1.pool.utilization",
                "Share of the native KV pool in use by requests and retained prefixes",
                used,
                limit,
            );
        }
    }
    let (was, now) = (before.l1, after.l1);
    out.level(
        "skippy.kv.l1.entries",
        "{entry}",
        "Prefixes the resident cache retains",
        now.entries,
    );
    out.level(
        "skippy.kv.l1.tokens",
        "{token}",
        "Tokens of retained prefixes, a shared prefix once per entry",
        now.tokens,
    );
    out.level(
        "skippy.kv.l1.pinned_tokens",
        "{token}",
        "Retained tokens a restore holds, which eviction cannot release",
        now.pinned_tokens,
    );
    out.level(
        "skippy.kv.l1.recurrent.usage",
        "By",
        "Host RAM held by recurrent checkpoints",
        now.recurrent_bytes,
    );
    out.level(
        "skippy.kv.l1.recurrent.limit",
        "By",
        "Recurrent checkpoint budget",
        now.recurrent_limit_bytes,
    );
    out.utilization(
        "skippy.kv.l1.recurrent.utilization",
        "Share of the recurrent checkpoint budget in use",
        now.recurrent_bytes,
        now.recurrent_limit_bytes,
    );
    out.per_second_by(
        "skippy.kv.l1.evictions",
        "{entry}/s",
        "Resident cache evictions, by component",
        "component",
        &[
            (
                "attention",
                was.attention_evictions,
                now.attention_evictions,
            ),
            (
                "recurrent",
                was.recurrent_evictions,
                now.recurrent_evictions,
            ),
        ],
    );
    l1_lookup_gauges(out, before, after);
}

fn l1_lookup_gauges(out: &mut Gauges<'_>, before: &KvTiers, after: &KvTiers) {
    let (was, now) = (before.lookups, after.lookups);
    out.lookups("l1", (was.hits, now.hits), (was.misses, now.misses));
    out.per_second(
        "skippy.kv.l1.hit_tokens",
        "{token}/s",
        "Prompt tokens L1 hits matched",
        was.hit_tokens,
        now.hit_tokens,
    );
    out.per_second(
        "skippy.kv.l1.restored_tokens",
        "{token}/s",
        "Tokens restored from retained prefixes onto requests",
        was.restored_tokens,
        now.restored_tokens,
    );
    out.per_second(
        "skippy.kv.l1.evicted_tokens",
        "{token}/s",
        "Tokens released by resident cache evictions",
        was.evicted_tokens,
        now.evicted_tokens,
    );
    out.per_second(
        "skippy.kv.l1.admission_shortfalls",
        "{request}/s",
        "Admissions left below the pool's free-space watermark after evicting all it could",
        was.shortfalls,
        now.shortfalls,
    );
}

fn l2_gauges(out: &mut Gauges<'_>, before: Option<L2Reading>, now: L2Reading) {
    out.level(
        "skippy.kv.l2.entries",
        "{entry}",
        "Entries in host-RAM L2",
        now.entries,
    );
    out.level(
        "skippy.kv.l2.usage",
        "By",
        "Distinct segment bytes held by L2",
        now.bytes,
    );
    out.level(
        "skippy.kv.l2.limit",
        "By",
        "L2 byte budget",
        now.budget_bytes,
    );
    out.utilization(
        "skippy.kv.l2.utilization",
        "Share of the L2 budget in use",
        now.bytes,
        now.budget_bytes,
    );
    let Some(was) = before else {
        return;
    };
    out.lookups("l2", (was.hits, now.hits), (was.misses, now.misses));
    out.per_second(
        "skippy.kv.l2.inserts",
        "{entry}/s",
        "L2 admissions",
        was.inserts,
        now.inserts,
    );
    out.per_second(
        "skippy.kv.l2.evictions",
        "{entry}/s",
        "L2 evictions to stay within budget",
        was.evictions,
        now.evictions,
    );
    out.per_second(
        "skippy.kv.l2.refusals",
        "{entry}/s",
        "L2 admissions refused for failing verification",
        was.refusals,
        now.refusals,
    );
}

fn l3_gauges(out: &mut Gauges<'_>, before: Option<L3Reading>, now: L3Reading) {
    if let Some(used) = now.used_bytes {
        out.level(
            "skippy.kv.l3.usage",
            "By",
            "Bytes the L3 store manages",
            used,
        );
        out.utilization(
            "skippy.kv.l3.utilization",
            "Share of the L3 budget in use",
            used,
            now.budget_bytes,
        );
    }
    out.level(
        "skippy.kv.l3.limit",
        "By",
        "L3 byte budget; zero is unlimited",
        now.budget_bytes,
    );
    out.level(
        "skippy.kv.l3.reserved",
        "By",
        "Bytes reserved by L3 writes in flight",
        now.reserved_bytes,
    );
    if let Some(available) = now.filesystem_available_bytes {
        out.level(
            "skippy.kv.l3.filesystem.available",
            "By",
            "Free space on the L3 store's filesystem",
            available,
        );
    }
    let Some(was) = before else {
        return;
    };
    out.lookups("l3", (was.hits, now.hits), (was.misses, now.misses));
    out.per_second(
        "skippy.kv.l3.fills",
        "{fill}/s",
        "Prefixes restored from L3",
        was.fills,
        now.fills,
    );
    out.per_second(
        "skippy.kv.l3.writes",
        "{write}/s",
        "Prefixes written to L3",
        was.writes,
        now.writes,
    );
    out.per_second_by(
        "skippy.kv.l3.io",
        "By/s",
        "L3 payload bytes, by direction",
        "direction",
        &[
            ("read", was.bytes_read, now.bytes_read),
            ("write", was.bytes_written, now.bytes_written),
        ],
    );
    out.per_second(
        "skippy.kv.l3.evictions",
        "{manifest}/s",
        "L3 entries removed to stay within budget",
        was.evictions,
        now.evictions,
    );
    out.per_second(
        "skippy.kv.l3.corrupt",
        "{object}/s",
        "L3 objects quarantined after failing verification",
        was.corrupt,
        now.corrupt,
    );
}

fn record_queue_gauges(out: &mut Gauges<'_>, was: RecordQueueReading, now: RecordQueueReading) {
    out.level(
        "skippy.kv.record_queue.records",
        "{record}",
        "Exact-state records waiting to publish to L1 and write through to L3",
        now.records,
    );
    out.level(
        "skippy.kv.record_queue.bytes",
        "By",
        "Payload bytes of waiting exact-state records",
        now.bytes,
    );
    out.per_second(
        "skippy.kv.record_queue.dropped",
        "{record}/s",
        "Exact-state records dropped because the queue was full",
        was.dropped,
        now.dropped,
    );
}

/// Tier readings for a resource snapshot, under `key`'s naming.
pub(super) fn snapshot(attrs: &mut BTreeMap<String, Value>, key: impl Fn(&str) -> String) {
    let tiers = KvTiers::now();
    if tiers.stages == 0 {
        return;
    }
    let mut put = |name: &str, value: u64| {
        attrs.insert(key(name), json!(value));
    };
    put("kv.l1.entries", tiers.l1.entries);
    put("kv.l1.tokens", tiers.l1.tokens);
    put("kv.l1.recurrent_bytes", tiers.l1.recurrent_bytes);
    put("kv.l1.hits", tiers.lookups.hits);
    put("kv.l1.misses", tiers.lookups.misses);
    if let Some(cells) = tiers.pool_cells {
        put("kv.l1.pool_cells", cells);
    }
    if let Some(l2) = tiers.l2 {
        put("kv.l2.bytes", l2.bytes);
        put("kv.l2.hits", l2.hits);
        put("kv.l2.misses", l2.misses);
    }
    if let Some(l3) = tiers.l3 {
        if let Some(used) = l3.used_bytes {
            put("kv.l3.used_bytes", used);
        }
        put("kv.l3.hits", l3.hits);
        put("kv.l3.misses", l3.misses);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opentelemetry_proto::tonic::metrics::v1::{metric::Data, number_data_point};

    fn values(metrics: &[Metric], name: &str) -> Vec<(Vec<String>, f64)> {
        let Some(Data::Gauge(gauge)) = metrics
            .iter()
            .find(|metric| metric.name == name)
            .and_then(|metric| metric.data.clone())
        else {
            return Vec::new();
        };
        gauge
            .data_points
            .iter()
            .map(|point| {
                let attributes = point
                    .attributes
                    .iter()
                    .filter_map(|attribute| attribute.value.as_ref())
                    .map(|value| format!("{:?}", value.value))
                    .collect();
                let value = match point.value {
                    Some(number_data_point::Value::AsDouble(value)) => value,
                    _ => f64::NAN,
                };
                (attributes, value)
            })
            .collect()
    }

    fn only(metrics: &[Metric], name: &str) -> Option<f64> {
        match values(metrics, name).as_slice() {
            [(_, value)] => Some(*value),
            _ => None,
        }
    }

    fn l3(store: usize, hits: u64) -> L3Reading {
        L3Reading {
            store,
            used_bytes: Some(100),
            budget_bytes: 1000,
            filesystem_available_bytes: Some(store as u64 * 10),
            hits,
            ..L3Reading::default()
        }
    }

    #[test]
    fn stages_sum_but_a_shared_l3_store_counts_once() {
        let stage = |store, entries| KvTierReading {
            l1: L1Reading {
                entries,
                ..L1Reading::default()
            },
            l2: Some(L2Reading {
                bytes: 10,
                ..L2Reading::default()
            }),
            l3: Some(l3(store, 5)),
            ..KvTierReading::default()
        };
        let shared = KvTiers::sum([stage(1, 2), stage(1, 3)]);
        assert_eq!(shared.stages, 2);
        assert_eq!(shared.l1.entries, 5);
        assert_eq!(shared.l2.map(|l2| l2.bytes), Some(20));
        assert_eq!(shared.l3.map(|l3| l3.hits), Some(5));

        let separate = KvTiers::sum([stage(1, 2), stage(2, 3)]);
        let l3 = separate.l3.expect("l3");
        assert_eq!((l3.hits, l3.used_bytes), (10, Some(200)));
        assert_eq!(l3.filesystem_available_bytes, Some(10));
    }

    #[test]
    fn pressure_and_misses_come_from_levels_and_the_interval() {
        let before = KvTiers {
            stages: 1,
            lookups: L1Lookups {
                hits: 10,
                misses: 2,
                ..L1Lookups::default()
            },
            l2: Some(L2Reading {
                hits: 1,
                misses: 1,
                ..L2Reading::default()
            }),
            l3: Some(l3(1, 4)),
            ..KvTiers::default()
        };
        let after = KvTiers {
            l1: L1Reading {
                recurrent_bytes: 256,
                recurrent_limit_bytes: 1024,
                attention_evictions: 3,
                ..L1Reading::default()
            },
            lookups: L1Lookups {
                hits: 13,
                misses: 3,
                ..L1Lookups::default()
            },
            l2: Some(L2Reading {
                bytes: 50,
                budget_bytes: 200,
                hits: 1,
                misses: 3,
                ..L2Reading::default()
            }),
            l3: Some(L3Reading {
                misses: 1,
                ..l3(1, 4)
            }),
            pool_cells: Some(4096),
            ..before
        };
        let mut metrics = Vec::new();
        gauges(&mut metrics, &before, &after, Some(1024), 0.5, 9);

        assert_eq!(only(&metrics, "skippy.kv.l1.pool.utilization"), Some(0.25));
        assert_eq!(
            only(&metrics, "skippy.kv.l1.recurrent.utilization"),
            Some(0.25)
        );
        assert_eq!(only(&metrics, "skippy.kv.l1.miss_ratio"), Some(0.25));
        let lookups = values(&metrics, "skippy.kv.l1.lookups");
        assert_eq!(
            lookups.iter().map(|(_, rate)| *rate).collect::<Vec<_>>(),
            [6.0, 2.0]
        );
        let evictions = values(&metrics, "skippy.kv.l1.evictions");
        assert_eq!(evictions[0].1, 6.0);
        assert_eq!(only(&metrics, "skippy.kv.l2.utilization"), Some(0.25));
        assert_eq!(only(&metrics, "skippy.kv.l2.miss_ratio"), Some(1.0));
        assert_eq!(only(&metrics, "skippy.kv.l3.utilization"), Some(0.1));
        assert_eq!(only(&metrics, "skippy.kv.l3.miss_ratio"), Some(1.0));
    }

    #[test]
    fn no_lookups_means_no_miss_ratio_and_no_stage_means_no_gauges() {
        let tiers = KvTiers {
            stages: 1,
            ..KvTiers::default()
        };
        let mut metrics = Vec::new();
        gauges(&mut metrics, &tiers, &tiers, None, 1.0, 1);
        assert!(only(&metrics, "skippy.kv.l1.miss_ratio").is_none());
        assert!(only(&metrics, "skippy.kv.l1.pool.utilization").is_none());
        assert!(only(&metrics, "skippy.kv.l1.entries").is_some());

        let mut metrics = Vec::new();
        gauges(
            &mut metrics,
            &KvTiers::default(),
            &KvTiers::default(),
            None,
            1.0,
            1,
        );
        assert!(metrics.is_empty());
    }
}
