//! Disposable observation pacing. Durable debts remain in the ledger.
use std::{
    collections::{BTreeMap, BTreeSet},
    time::{Duration, Instant},
};

const BATCH: usize = 8;
const MAX_BACKOFF: Duration = Duration::from_secs(60);

#[derive(Default)]
pub(crate) struct ReceiptSchedule {
    attempts: BTreeMap<String, (Instant, Duration)>,
}

impl ReceiptSchedule {
    /// Reserve attempts before I/O, so cancellation cannot hot-loop an invoice.
    /// Oldest due first prevents a slow first lookup starving later invoices.
    pub(crate) fn reserve(
        &mut self,
        hashes: impl Iterator<Item = String>,
        now: Instant,
    ) -> Vec<String> {
        let outstanding: BTreeSet<_> = hashes.collect();
        self.attempts.retain(|hash, _| outstanding.contains(hash));
        for hash in outstanding {
            self.attempts
                .entry(hash)
                .or_insert((now, Duration::from_millis(250)));
        }
        let mut due: Vec<_> = self
            .attempts
            .iter()
            .filter(|(_, (next, _))| *next <= now)
            .map(|(hash, (next, _))| (*next, hash.clone()))
            .collect();
        due.sort();
        due.truncate(BATCH);
        due.into_iter()
            .map(|(_, hash)| {
                let entry = self.attempts.get_mut(&hash).expect("scheduled invoice");
                entry.0 = now + entry.1;
                entry.1 = (entry.1 * 2).min(MAX_BACKOFF);
                hash
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounds_batches_and_gives_old_invoices_a_turn() {
        let mut schedule = ReceiptSchedule::default();
        let now = Instant::now();
        let hashes = || (0..20).map(|i| format!("hash-{i:02}"));
        assert_eq!(schedule.reserve(hashes(), now).len(), 8);
        let next = schedule.reserve(hashes(), now);
        assert_eq!(next.len(), 8);
        assert_eq!(next[0], "hash-08");
        assert_eq!(schedule.reserve(hashes(), now).len(), 4);
        assert!(schedule.reserve(hashes(), now).is_empty());
    }
    #[test]
    fn backoff_caps_without_forgetting_debt_and_paid_entries_are_removed() {
        let mut schedule = ReceiptSchedule::default();
        let mut now = Instant::now();
        for _ in 0..20 {
            assert_eq!(
                schedule.reserve(std::iter::once("debt".into()), now),
                vec!["debt"]
            );
            now += MAX_BACKOFF;
        }
        assert_eq!(schedule.attempts["debt"].1, MAX_BACKOFF);
        schedule.reserve(std::iter::empty(), now);
        assert!(schedule.attempts.is_empty());
    }
}
