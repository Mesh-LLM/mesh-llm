//! Reserve recorder bytes before native export, then transfer accounting to
//! the existing queue/worker completion boundary. Refusal never allocates KV.
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

#[derive(Debug)]
pub(super) struct ExportByteReservation {
    counter: Arc<AtomicU64>,
    bytes: u64,
    transferred: bool,
}

impl ExportByteReservation {
    pub(super) fn acquire(counter: &Arc<AtomicU64>, bytes: u64, cap: u64) -> Option<Self> {
        let mut held = counter.load(Ordering::Acquire);
        loop {
            if !super::super::record_fits_queue(held, bytes, cap) {
                return None;
            }
            let next = held.checked_add(bytes)?;
            match counter.compare_exchange_weak(held, next, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => {
                    return Some(Self {
                        counter: counter.clone(),
                        bytes,
                        transferred: false,
                    });
                }
                Err(actual) => held = actual,
            }
        }
    }

    pub(super) fn transfer(mut self, counter: &AtomicU64, bytes: u64) -> bool {
        self.transferred = std::ptr::eq(self.counter.as_ref(), counter) && self.bytes == bytes;
        self.transferred
    }
}

impl Drop for ExportByteReservation {
    fn drop(&mut self) {
        if !self.transferred {
            self.counter.fetch_sub(self.bytes, Ordering::AcqRel);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_hybrid_pair_is_refused_before_export_and_cancellation_releases_bytes() {
        let counter = Arc::new(AtomicU64::new(0));
        let cap = 4 * 1024 * 1024 * 1024;
        let shared = 64_111 * 32_768 + 52_690_944;
        let continuation = 64_367 * 32_768 + 52_690_944;
        let first = ExportByteReservation::acquire(&counter, shared, cap).unwrap();
        assert!(ExportByteReservation::acquire(&counter, continuation, cap).is_none());
        drop(first);
        assert_eq!(counter.load(Ordering::Acquire), 0);
        assert!(ExportByteReservation::acquire(&counter, continuation, cap).is_some());
    }

    #[test]
    fn transfer_keeps_bytes_until_worker_release_and_mismatch_releases_reservation() {
        let counter = Arc::new(AtomicU64::new(0));
        let credit = ExportByteReservation::acquire(&counter, 10, 10).unwrap();
        assert!(credit.transfer(&counter, 10));
        assert_eq!(counter.fetch_sub(10, Ordering::AcqRel), 10);
        let credit = ExportByteReservation::acquire(&counter, 10, 10).unwrap();
        assert!(!credit.transfer(&counter, 9));
        assert_eq!(counter.load(Ordering::Acquire), 0);
    }
}
