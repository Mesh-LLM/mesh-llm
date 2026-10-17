use std::sync::Arc;

/// Admission class for an exact-state capture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureAdmission {
    /// Same-session continuation boundary (post-decode). Re-derivable only by
    /// replaying generation; competes for the reserved recorder slot.
    Continuation,
    /// Prefill-ladder boundary (shared checkpoint / final prefill state).
    /// Re-derivable by re-prefill; best-effort admission.
    BestEffort,
}

/// Total in-flight exact-state payloads per stage — exporting, queued in the
/// recorder channel, or held by the recorder worker. Sized as channel
/// capacity (1) plus one worker-held record.
pub const EXACT_STATE_ADMISSION_TOTAL: usize = 2;
/// Slots reserved for [`CaptureAdmission::Continuation`]: BestEffort captures
/// can never occupy more than `TOTAL - RESERVED` slots, so a continuation
/// boundary always finds a slot unless the whole budget is exhausted.
pub const EXACT_STATE_ADMISSION_RESERVED: usize = 1;

/// Move-only credit for one in-flight exact-state payload. Construction
/// reserves a slot atomically (CAS, so competing producers cannot
/// oversubscribe the budget); Drop releases it exactly once on every terminal
/// path — payload stored, skipped, errored, command dropped unexecuted, or
/// channel receiver dropped. The credit moves into
/// [`PendingExactStateRecord`] so the worker's completion releases it.
#[derive(Debug)]
pub struct ExactStateAdmissionCredit {
    class: CaptureAdmission,
    total: Arc<std::sync::atomic::AtomicUsize>,
    best_effort: Arc<std::sync::atomic::AtomicUsize>,
}

impl ExactStateAdmissionCredit {
    pub(crate) fn acquire(
        total: &Arc<std::sync::atomic::AtomicUsize>,
        best_effort: &Arc<std::sync::atomic::AtomicUsize>,
        class: CaptureAdmission,
    ) -> Option<Self> {
        use std::sync::atomic::Ordering::{AcqRel, Acquire};
        match class {
            CaptureAdmission::Continuation => {
                let mut observed = total.load(Acquire);
                loop {
                    if observed >= EXACT_STATE_ADMISSION_TOTAL {
                        return None;
                    }
                    match total.compare_exchange_weak(observed, observed + 1, AcqRel, Acquire) {
                        Ok(_) => {
                            return Some(Self {
                                class,
                                total: Arc::clone(total),
                                best_effort: Arc::clone(best_effort),
                            });
                        }
                        Err(actual) => observed = actual,
                    }
                }
            }
            CaptureAdmission::BestEffort => {
                let reserved_cap = EXACT_STATE_ADMISSION_TOTAL - EXACT_STATE_ADMISSION_RESERVED;
                let mut observed_best_effort = best_effort.load(Acquire);
                loop {
                    if observed_best_effort >= reserved_cap {
                        return None;
                    }
                    match best_effort.compare_exchange_weak(
                        observed_best_effort,
                        observed_best_effort + 1,
                        AcqRel,
                        Acquire,
                    ) {
                        Ok(_) => break,
                        Err(actual) => observed_best_effort = actual,
                    }
                }
                // Reserve the total slot; roll the BestEffort reservation back
                // if the total budget was exhausted concurrently.
                let mut observed_total = total.load(Acquire);
                loop {
                    if observed_total >= EXACT_STATE_ADMISSION_TOTAL {
                        best_effort.fetch_sub(1, AcqRel);
                        return None;
                    }
                    match total.compare_exchange_weak(
                        observed_total,
                        observed_total + 1,
                        AcqRel,
                        Acquire,
                    ) {
                        Ok(_) => {
                            return Some(Self {
                                class,
                                total: Arc::clone(total),
                                best_effort: Arc::clone(best_effort),
                            });
                        }
                        Err(actual) => observed_total = actual,
                    }
                }
            }
        }
    }
}

impl Drop for ExactStateAdmissionCredit {
    fn drop(&mut self) {
        let _ = self.total.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
        if self.class == CaptureAdmission::BestEffort {
            let _ = self
                .best_effort
                .fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
        }
    }
}

#[cfg(test)]
mod admission_credit_tests {
    use super::{
        CaptureAdmission, EXACT_STATE_ADMISSION_RESERVED, EXACT_STATE_ADMISSION_TOTAL,
        ExactStateAdmissionCredit,
    };
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    fn counters() -> (Arc<AtomicUsize>, Arc<AtomicUsize>) {
        (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)))
    }

    #[test]
    fn construction_and_drop_account_exactly_once() {
        let (total, best_effort) = counters();
        let guard = ExactStateAdmissionCredit::acquire(
            &total,
            &best_effort,
            CaptureAdmission::Continuation,
        );
        assert!(guard.is_some(), "fresh budget admits Continuation");
        assert_eq!(total.load(Ordering::Acquire), 1);
        drop(guard);
        assert_eq!(total.load(Ordering::Acquire), 0);
    }

    #[test]
    fn continuation_declines_when_total_budget_exhausted() {
        let (total, best_effort) = counters();
        let first = ExactStateAdmissionCredit::acquire(
            &total,
            &best_effort,
            CaptureAdmission::Continuation,
        );
        let second = ExactStateAdmissionCredit::acquire(
            &total,
            &best_effort,
            CaptureAdmission::Continuation,
        );
        assert!(first.is_some() && second.is_some(), "two slots for TOTAL=2");
        let third = ExactStateAdmissionCredit::acquire(
            &total,
            &best_effort,
            CaptureAdmission::Continuation,
        );
        assert!(
            third.is_none(),
            "third Continuation must decline at TOTAL=2"
        );
        assert_eq!(total.load(Ordering::Acquire), 2);
    }

    #[test]
    fn best_effort_declines_at_reserved_cap_even_with_budget_headroom() {
        let (total, best_effort) = counters();
        let first =
            ExactStateAdmissionCredit::acquire(&total, &best_effort, CaptureAdmission::BestEffort);
        assert!(first.is_some());
        let second =
            ExactStateAdmissionCredit::acquire(&total, &best_effort, CaptureAdmission::BestEffort);
        assert!(
            second.is_none(),
            "BestEffort cap is TOTAL - RESERVED = 1; second must decline"
        );
        assert_eq!(best_effort.load(Ordering::Acquire), 1);
    }

    #[test]
    fn continuation_admits_after_best_effort_under_reserved_share() {
        let (total, best_effort) = counters();
        let best =
            ExactStateAdmissionCredit::acquire(&total, &best_effort, CaptureAdmission::BestEffort);
        assert!(best.is_some(), "BestEffort takes its single slot");
        let cont = ExactStateAdmissionCredit::acquire(
            &total,
            &best_effort,
            CaptureAdmission::Continuation,
        );
        assert!(
            cont.is_some(),
            "Continuation must admit into the reserved share"
        );
    }

    #[test]
    fn concurrent_producers_never_oversubscribe_the_budget() {
        use std::sync::Barrier;
        const THREADS: usize = 4;
        const ATTEMPTS: usize = 25;
        // ONE shared budget for every producer: racing acquisitions hit the
        // SAME counters, classes are mixed per attempt (both CAS paths,
        // including the BestEffort rollback, race concurrently), credits are
        // RETAINED across a barrier, and rollback/drain are asserted exactly.
        let (total, best_effort) = counters();
        let start = Arc::new(Barrier::new(THREADS));
        let handles: Vec<_> = (0..THREADS)
            .map(|thread| {
                let total = Arc::clone(&total);
                let best_effort = Arc::clone(&best_effort);
                let start = Arc::clone(&start);
                std::thread::spawn(move || {
                    let mut held = Vec::new();
                    start.wait();
                    for attempt in 0..ATTEMPTS {
                        let class = if (thread + attempt) % 2 == 0 {
                            CaptureAdmission::Continuation
                        } else {
                            CaptureAdmission::BestEffort
                        };
                        if let Some(credit) =
                            ExactStateAdmissionCredit::acquire(&total, &best_effort, class)
                        {
                            held.push((class, credit));
                        }
                    }
                    held
                })
            })
            .collect();
        let mut held: Vec<_> = handles
            .into_iter()
            .flat_map(|handle| handle.join().unwrap())
            .collect();
        let held_best_effort = held
            .iter()
            .filter(|(class, _)| *class == CaptureAdmission::BestEffort)
            .count();
        // Retention bounds with mixed classes: Continuation fills at most
        // TOTAL slots, BestEffort at most TOTAL - RESERVED, and together they
        // cannot exceed the total budget.
        assert!(held.len() <= EXACT_STATE_ADMISSION_TOTAL);
        assert!(
            held_best_effort <= EXACT_STATE_ADMISSION_TOTAL - EXACT_STATE_ADMISSION_RESERVED,
            "BestEffort holds must respect the reserved share"
        );
        assert_eq!(
            total.load(Ordering::Acquire),
            held.len(),
            "retained credits must account for every outstanding slot"
        );
        assert_eq!(
            best_effort.load(Ordering::Acquire),
            held_best_effort,
            "retained credits must account for the BestEffort counter"
        );

        // Top the budget up to exactly TOTAL held credits so the hammer below
        // runs against a deterministically exhausted budget.
        while held.len() < EXACT_STATE_ADMISSION_TOTAL {
            let credit = ExactStateAdmissionCredit::acquire(
                &total,
                &best_effort,
                CaptureAdmission::Continuation,
            )
            .expect("topping up an under-subscribed budget must succeed");
            held.push((CaptureAdmission::Continuation, credit));
        }

        // Rollback hammer: the total budget is exhausted by the held credits,
        // so every concurrent BestEffort attempt must decline, and the
        // rollback path must not leak a BestEffort slot (a missing rollback
        // would drift the counter above the held count).
        let hammer = Arc::new(Barrier::new(THREADS));
        let hammers: Vec<_> = (0..THREADS)
            .map(|_| {
                let total = Arc::clone(&total);
                let best_effort = Arc::clone(&best_effort);
                let hammer = Arc::clone(&hammer);
                std::thread::spawn(move || {
                    hammer.wait();
                    for _ in 0..ATTEMPTS {
                        assert!(
                            ExactStateAdmissionCredit::acquire(
                                &total,
                                &best_effort,
                                CaptureAdmission::BestEffort
                            )
                            .is_none(),
                            "an exhausted budget must decline every BestEffort attempt"
                        );
                    }
                })
            })
            .collect();
        for handle in hammers {
            handle.join().unwrap();
        }
        assert_eq!(
            total.load(Ordering::Acquire),
            held.len(),
            "the rollback hammer must not change the total"
        );
        assert_eq!(
            best_effort.load(Ordering::Acquire),
            held_best_effort,
            "the rollback hammer must not leak a BestEffort slot"
        );

        // Drain: dropping every held credit releases each slot exactly once,
        // and the budget recovers for new admissions.
        drop(held);
        assert_eq!(total.load(Ordering::Acquire), 0);
        assert_eq!(best_effort.load(Ordering::Acquire), 0);
        let recovery_continuation = ExactStateAdmissionCredit::acquire(
            &total,
            &best_effort,
            CaptureAdmission::Continuation,
        );
        let recovery_best_effort =
            ExactStateAdmissionCredit::acquire(&total, &best_effort, CaptureAdmission::BestEffort);
        assert!(
            recovery_continuation.is_some() && recovery_best_effort.is_some(),
            "budget must recover after the drain"
        );
    }

    #[test]
    fn rollback_is_exercised_when_the_budget_is_held_by_continuations() {
        use std::sync::Barrier;
        const THREADS: usize = 4;
        const ATTEMPTS: usize = 50;
        // Deterministic rollback exercise: the budget STARTS held by exactly
        // TWO Continuation credits (total=2, best_effort=0), so the first
        // BestEffort attempts must pass the class cap and fail the total CAS
        // — the rollback path. Overlapping contenders can still decline at
        // the class cap while a rollback is in flight; the exact 2/0
        // assertions afterwards are what prove no rollback leaked. The
        // mixed-contention test above can retain a BestEffort credit, in
        // which case its hammer exits at the class cap throughout.
        let (total, best_effort) = counters();
        let first = ExactStateAdmissionCredit::acquire(
            &total,
            &best_effort,
            CaptureAdmission::Continuation,
        )
        .expect("first Continuation admitted");
        let second = ExactStateAdmissionCredit::acquire(
            &total,
            &best_effort,
            CaptureAdmission::Continuation,
        )
        .expect("second Continuation admitted");
        assert_eq!(total.load(Ordering::Acquire), 2);
        assert_eq!(best_effort.load(Ordering::Acquire), 0);

        let barrier = Arc::new(Barrier::new(THREADS));
        let hammers: Vec<_> = (0..THREADS)
            .map(|_| {
                let total = Arc::clone(&total);
                let best_effort = Arc::clone(&best_effort);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    for _ in 0..ATTEMPTS {
                        assert!(
                            ExactStateAdmissionCredit::acquire(
                                &total,
                                &best_effort,
                                CaptureAdmission::BestEffort
                            )
                            .is_none(),
                            "a continuation-exhausted budget must decline every BestEffort via rollback"
                        );
                    }
                })
            })
            .collect();
        for handle in hammers {
            handle.join().unwrap();
        }
        assert_eq!(
            total.load(Ordering::Acquire),
            2,
            "rollback leaves the total at the held credits"
        );
        assert_eq!(
            best_effort.load(Ordering::Acquire),
            0,
            "rollback must not leak a BestEffort slot"
        );

        // Drain and recovery.
        drop((first, second));
        assert_eq!(total.load(Ordering::Acquire), 0);
        assert_eq!(best_effort.load(Ordering::Acquire), 0);
        assert!(
            ExactStateAdmissionCredit::acquire(&total, &best_effort, CaptureAdmission::BestEffort)
                .is_some(),
            "budget recovers after the drain"
        );
    }
}
