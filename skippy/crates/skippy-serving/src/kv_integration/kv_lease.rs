//! In-place KV page records: the lane's own cells are the payload.
//!
//! On a host-visible backend a parked lane's attention KV need not be copied
//! out to be recorded. The lane hands the recorder spans into its cells plus
//! a [`LanePin`]; the recorder worker copies the spans into its own buffer
//! once, off the runtime lock, and drops the pin. Until every pin on a lane
//! is dropped, the lane's cells are read-shared: nothing may reset, trim
//! below the leased page, import state into, or free that lane. The
//! [`LeaseLedger`] is how the lane side enforces that window.
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use skippy_runtime::KvPageSpans;

use super::{LEASED_RECORD_SLOTS, record_fits_queue};

/// Why a leased record could not reserve its slot and bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LeasedRecordRefusal {
    QueueFull,
    ByteFull,
}

/// Admission is claimed before cloning a resident recurrent snapshot: a
/// leased slot, and every byte the record will occupy once the worker has
/// copied the page out of the lane (attention KV plus the snapshot). The copy
/// happens inside this reservation, never on top of a full queue; a page the
/// budget cannot hold stays on the lane. Until enqueue transfers it to the
/// worker, dropping the reservation releases both counters, including on a
/// disconnected recorder or a failed enqueue.
#[derive(Debug)]
pub(crate) struct LeasedRecordReservation {
    pending: Arc<AtomicUsize>,
    queue_bytes: Arc<AtomicU64>,
    bytes: u64,
    transferred: bool,
}

impl LeasedRecordReservation {
    /// `bytes` is the stored size of the record: the leased page plus the
    /// recurrent snapshot that will be cloned once this succeeds.
    pub(crate) fn acquire(
        pending: &Arc<AtomicUsize>,
        queue_bytes: &Arc<AtomicU64>,
        bytes: u64,
        byte_limit: u64,
    ) -> Result<Self, LeasedRecordRefusal> {
        pending
            .try_update(Ordering::AcqRel, Ordering::Acquire, |count| {
                (count < LEASED_RECORD_SLOTS).then_some(count + 1)
            })
            .map_err(|_| LeasedRecordRefusal::QueueFull)?;
        let mut held = queue_bytes.load(Ordering::Acquire);
        loop {
            if !record_fits_queue(held, bytes, byte_limit) {
                pending.fetch_sub(1, Ordering::AcqRel);
                return Err(LeasedRecordRefusal::ByteFull);
            }
            let Some(next) = held.checked_add(bytes) else {
                pending.fetch_sub(1, Ordering::AcqRel);
                return Err(LeasedRecordRefusal::ByteFull);
            };
            match queue_bytes.compare_exchange_weak(held, next, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => break,
                Err(actual) => held = actual,
            }
        }
        Ok(Self {
            pending: Arc::clone(pending),
            queue_bytes: Arc::clone(queue_bytes),
            bytes,
            transferred: false,
        })
    }

    /// The enqueue/worker completion path now owns the counters. Matching
    /// handles and size are an internal invariant, not a fresh admission.
    pub(crate) fn transfer(&mut self, pending: &AtomicUsize, queue_bytes: &AtomicU64, bytes: u64) {
        assert!(std::ptr::eq(self.pending.as_ref(), pending));
        assert!(std::ptr::eq(self.queue_bytes.as_ref(), queue_bytes));
        assert_eq!(self.bytes, bytes);
        self.transferred = true;
    }
}

impl Drop for LeasedRecordReservation {
    fn drop(&mut self) {
        if !self.transferred {
            self.pending.fetch_sub(1, Ordering::AcqRel);
            self.queue_bytes.fetch_sub(self.bytes, Ordering::AcqRel);
        }
    }
}

/// Per-lane count of in-flight in-place records, plus the highest token
/// position any of them covers. Cloned into every [`LanePin`].
#[derive(Debug, Clone, Default)]
pub(crate) struct LeaseLedger {
    inner: Arc<LeaseLedgerInner>,
}

#[derive(Debug, Default)]
struct LeaseLedgerInner {
    outstanding: AtomicUsize,
    /// Highest `token_count` among in-flight leases. Only meaningful while
    /// `outstanding > 0`; a trim below it would invalidate a reader.
    frontier: AtomicU64,
}

impl LeaseLedger {
    /// Register a reader of this lane's cells up to `token_count`.
    pub(crate) fn pin(&self, token_count: u64) -> LanePin {
        self.inner.outstanding.fetch_add(1, Ordering::AcqRel);
        self.inner.frontier.fetch_max(token_count, Ordering::AcqRel);
        LanePin {
            ledger: self.clone(),
        }
    }

    pub(crate) fn in_flight(&self) -> bool {
        self.inner.outstanding.load(Ordering::Acquire) > 0
    }

    /// Whether a trim of the lane to `token_count` tokens would cut into a
    /// page a reader is still copying.
    pub(crate) fn blocks_trim_to(&self, token_count: u64) -> bool {
        self.in_flight() && self.inner.frontier.load(Ordering::Acquire) > token_count
    }

    /// Block until every reader has dropped its pin. Readers only memcpy, so
    /// this is bounded by one page copy per outstanding pin; a longer wait
    /// is a stuck recorder worker and is logged rather than hidden.
    pub(crate) fn wait_idle(&self) {
        let started = Instant::now();
        let mut warned = Duration::from_secs(5);
        while self.in_flight() {
            std::thread::sleep(Duration::from_millis(1));
            if started.elapsed() >= warned {
                tracing::warn!(
                    outstanding = self.inner.outstanding.load(Ordering::Acquire),
                    waited_ms = started.elapsed().as_millis() as u64,
                    "lane cells still leased to the exact-state recorder"
                );
                warned *= 2;
            }
        }
    }
}

/// One reader's hold on a lane's cells. Dropping it is the release.
#[derive(Debug)]
pub(crate) struct LanePin {
    ledger: LeaseLedger,
}

impl Drop for LanePin {
    fn drop(&mut self) {
        self.ledger.inner.outstanding.fetch_sub(1, Ordering::AcqRel);
    }
}

/// A KV page the recorder will read straight out of a lane's cells.
#[derive(Debug)]
pub(crate) struct LeasedKvPage {
    pub(crate) spans: KvPageSpans,
    /// Held until the worker has copied the spans out. Never read; the Drop
    /// impl is the release.
    #[allow(dead_code)]
    pub(crate) pin: LanePin,
}

impl LeasedKvPage {
    pub(crate) fn byte_len(&self) -> u64 {
        self.spans.byte_len() as u64
    }

    /// Copy the page out of the cells. The returned bytes are owned; the
    /// lease, and with it the lane's pin, ends when `self` is dropped.
    pub(crate) fn materialise(&self) -> Result<Vec<u8>, skippy_runtime::KvPageSpanError> {
        let mut kv = vec![0_u8; self.spans.byte_len()];
        // SAFETY: the pin keeps the lane from resetting, trimming below the
        // page, importing state, or freeing its cells while we hold it; see
        // `LeaseLedger`. The spans were produced by the native walk for a
        // page that is still resident under that guarantee.
        unsafe { self.spans.read(&mut kv) }?;
        Ok(kv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stored_bytes_refuse_before_a_clone_and_release_the_slot() {
        let pending = Arc::new(AtomicUsize::new(0));
        let bytes = Arc::new(AtomicU64::new(8));
        // A two-byte snapshot would fit; the page it rides with does not.
        assert_eq!(
            LeasedRecordReservation::acquire(&pending, &bytes, 2 + 4, 10).unwrap_err(),
            LeasedRecordRefusal::ByteFull
        );
        assert_eq!(pending.load(Ordering::Acquire), 0);
        assert_eq!(bytes.load(Ordering::Acquire), 8);

        let reservation = LeasedRecordReservation::acquire(&pending, &bytes, 2, 10).unwrap();
        assert_eq!(pending.load(Ordering::Acquire), 1);
        assert_eq!(bytes.load(Ordering::Acquire), 10);
        drop(reservation);
        assert_eq!(pending.load(Ordering::Acquire), 0);
        assert_eq!(bytes.load(Ordering::Acquire), 8);
    }

    #[test]
    fn pins_count_and_release_exactly_once() {
        let ledger = LeaseLedger::default();
        assert!(!ledger.in_flight());
        let first = ledger.pin(100);
        let second = ledger.pin(40);
        assert!(ledger.in_flight());
        assert!(ledger.blocks_trim_to(99));
        assert!(!ledger.blocks_trim_to(100));
        drop(second);
        assert!(ledger.in_flight());
        drop(first);
        assert!(!ledger.in_flight());
        assert!(!ledger.blocks_trim_to(0));
    }

    #[test]
    fn wait_idle_returns_once_the_reader_releases() {
        let ledger = LeaseLedger::default();
        let pin = ledger.pin(8);
        let reader = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            drop(pin);
        });
        ledger.wait_idle();
        assert!(!ledger.in_flight());
        reader.join().unwrap();
    }
}
