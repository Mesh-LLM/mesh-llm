//! Matched same-capacity LRU baseline for policy comparison (#1650 first
//! slice). Capacity is in exclusive bytes so the comparison is apples-to-
//! apples against `BenefitPolicy` under a hard byte budget.

use crate::policy::traces::TraceAccess;

pub struct LruCache {
    capacity_bytes: u64,
    used_bytes: u64,
    /// Most-recent first.
    order: Vec<u64>,
    sizes: std::collections::HashMap<u64, u64>,
    pub hits: u64,
    pub misses: u64,
    /// Saved cold-prefill cost (higher is better) over the trace.
    pub saved_cost: f64,
    /// Bytes written into the cache (lower is better).
    pub bytes_written: u64,
}

impl LruCache {
    pub fn new(capacity_bytes: u64) -> Self {
        Self {
            capacity_bytes,
            used_bytes: 0,
            order: Vec::new(),
            sizes: std::collections::HashMap::new(),
            hits: 0,
            misses: 0,
            saved_cost: 0.0,
            bytes_written: 0,
        }
    }

    pub fn access(&mut self, access: &TraceAccess) -> bool {
        if self.order.contains(&access.entry) {
            self.hits += 1;
            self.saved_cost += (access.cold_prefill_cost - access.restore_cost).max(0.0);
            self.order.retain(|e| *e != access.entry);
            self.order.insert(0, access.entry);
            return true;
        }
        self.misses += 1;
        while self.used_bytes + access.exclusive_bytes > self.capacity_bytes {
            let Some(victim) = self.order.pop() else {
                break;
            };
            if let Some(size) = self.sizes.remove(&victim) {
                self.used_bytes = self.used_bytes.saturating_sub(size);
            }
        }
        if access.exclusive_bytes <= self.capacity_bytes {
            self.order.insert(0, access.entry);
            self.sizes.insert(access.entry, access.exclusive_bytes);
            self.used_bytes += access.exclusive_bytes;
            self.bytes_written += access.exclusive_bytes;
        }
        false
    }
}
