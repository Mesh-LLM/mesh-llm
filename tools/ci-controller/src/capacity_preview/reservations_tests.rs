use super::*;
use crate::capacity_preview::{Role, RunnerClass, WorkKind};
use std::collections::{BTreeMap, BTreeSet};
use std::convert::Infallible;
use std::sync::{Barrier, Mutex};

fn class() -> RunnerClass {
    RunnerClass {
        provider: "github".into(),
        os: "macos".into(),
        architecture: "arm64".into(),
        label: "macos-15".into(),
        group: None,
    }
}

fn work(id: &str, kind: WorkKind, workers: u64) -> Work {
    Work {
        id: id.into(),
        kind,
        role: "mac".into(),
        workers,
    }
}

fn main(revision: char) -> WorkKind {
    WorkKind::Main {
        repository_id: 42,
        revision: revision.to_string().repeat(40),
    }
}

fn pending(work: Work) -> Pending {
    Pending {
        work,
        supersedes: None,
    }
}

fn model() -> Model {
    let role = Role {
        runner: class(),
        pools: BTreeSet::from(["mac".into(), "org".into()]),
    };
    Model {
        revision: 4,
        snapshot: Snapshot {
            observed_at: 100,
            max_age: 10,
            pools: BTreeMap::from([("mac".into(), 2), ("org".into(), 2)]),
            roles: BTreeMap::from([("mac".into(), role)]),
            active: Vec::new(),
            pending: Vec::new(),
        },
        reserved: Vec::new(),
    }
}

fn reserved(transition: Transition) -> (Work, Model) {
    let Transition::Reserve { work, next } = transition else {
        panic!("expected a proposed reservation")
    };
    (work, *next)
}

#[test]
fn proposal_charges_active_and_reserved_work_before_later_main_and_release() {
    let mut state = model();
    state.snapshot.active.push(Active {
        work: work("main-1", main('a'), 1),
        observed_class: Some(class()),
    });
    state.snapshot.pending = vec![
        pending(work("main-2", main('b'), 1)),
        pending(work("main-3", main('c'), 1)),
        pending(work("release", WorkKind::Release { repository_id: 42 }, 1)),
    ];

    let (selected, next) = reserved(propose(&state, 4, 105).unwrap());
    assert_eq!(selected.id, "main-2");
    assert_eq!(next.revision, 5);
    assert_eq!(next.reserved, [selected]);
    assert_eq!(next.snapshot.pending.len(), 2);
    assert_eq!(next.snapshot.pending[0].work.id, "main-3");
    assert_eq!(propose(&next, 5, 105), Ok(Transition::Wait));
    assert_eq!(state.snapshot.pending.len(), 3, "input is not mutated");
}

#[test]
fn concurrent_proposals_need_an_external_atomic_expected_revision_commit() {
    let mut state = model();
    state.snapshot.pending = vec![
        pending(work("main-1", main('a'), 2)),
        pending(work("main-2", main('b'), 1)),
    ];
    let first = propose(&state, 4, 105).unwrap();
    let second = propose(&state, 4, 105).unwrap();
    assert_eq!(first, second, "a pure model cannot serialize callers");
    let (_, committed_example) = reserved(first);
    assert_eq!(
        propose(&committed_example, 4, 105),
        Err(ModelError::RevisionMismatch)
    );
    assert_eq!(propose(&committed_example, 5, 105), Ok(Transition::Wait));
}

#[test]
fn supplied_start_transfers_the_charge_without_proving_or_freeing_capacity() {
    let mut state = model();
    state.snapshot.pending = vec![
        pending(work("main-1", main('a'), 2)),
        pending(work("main-2", main('b'), 1)),
    ];
    let (selected, reserved_state) = reserved(propose(&state, 4, 105).unwrap());
    let observation = Active {
        work: selected.clone(),
        observed_class: Some(class()),
    };
    let started = observe_start(&reserved_state, 5, 105, observation).unwrap();
    assert_eq!(started.revision, 6);
    assert!(started.reserved.is_empty());
    assert_eq!(started.snapshot.active[0].work, selected);
    assert_eq!(propose(&started, 6, 105), Ok(Transition::Wait));
    assert_eq!(reserved_state.reserved.len(), 1, "input is not mutated");
}

#[test]
fn missing_mismatched_or_stale_start_observations_fail_closed() {
    let mut state = model();
    state
        .snapshot
        .pending
        .push(pending(work("main", main('a'), 1)));
    let (selected, next) = reserved(propose(&state, 4, 105).unwrap());
    let mut observed = Active {
        work: selected.clone(),
        observed_class: None,
    };
    assert_eq!(
        observe_start(&next, 5, 105, observed.clone()),
        Err(ModelError::Preview(PreviewError::MissingRunnerClass))
    );
    observed.observed_class = Some(class());
    observed.work.workers = 2;
    assert_eq!(
        observe_start(&next, 5, 105, observed.clone()),
        Err(ModelError::ReservationMismatch)
    );
    observed.work = selected.clone();
    observed.observed_class.as_mut().unwrap().provider = "depot".into();
    assert_eq!(
        observe_start(&next, 5, 105, observed.clone()),
        Err(ModelError::Preview(PreviewError::RunnerClassMismatch))
    );
    observed.observed_class = Some(class());
    observed.work.id = "unknown".into();
    assert_eq!(
        observe_start(&next, 5, 105, observed.clone()),
        Err(ModelError::UnknownReservation)
    );
    observed.work = selected;
    assert_eq!(
        observe_start(&next, 5, 111, observed),
        Err(ModelError::Preview(PreviewError::StaleSnapshot))
    );
    assert_eq!(next.reserved.len(), 1);
}

#[test]
fn unresolved_supersession_and_over_capacity_reservations_are_not_discarded() {
    let mut state = model();
    let pr = WorkKind::PullRequest {
        repository_id: 42,
        number: 7,
    };
    state.snapshot.pending = vec![
        pending(work("old", pr.clone(), 1)),
        Pending {
            work: work("new", pr, 1),
            supersedes: Some("old".into()),
        },
    ];
    assert_eq!(
        propose(&state, 4, 105),
        Err(ModelError::UnreconciledSupersession)
    );
    assert_eq!(state.snapshot.pending.len(), 2);

    state.snapshot.pending.clear();
    state.reserved.push(work("too-large", main('a'), 3));
    assert_eq!(
        propose(&state, 4, 105),
        Err(ModelError::ReservationOverCapacity)
    );
    assert_eq!(state.reserved.len(), 1);
}

#[test]
fn duplicate_work_and_revision_overflow_fail_without_a_transition() {
    let mut state = model();
    let duplicate = work("main", main('a'), 1);
    state.reserved.push(duplicate.clone());
    state.snapshot.pending.push(pending(duplicate));
    assert_eq!(
        propose(&state, 4, 105),
        Err(ModelError::Preview(PreviewError::DuplicateWork))
    );
    state.revision = u64::MAX;
    assert_eq!(
        propose(&state, u64::MAX, 105),
        Err(ModelError::RevisionOverflow)
    );
}

// This process-local fake exercises the adapter contract only. It is not a
// persistent store and supplies no authentication or cross-run coordination.
struct FakeStore {
    state: Mutex<Model>,
    after_read: Option<Barrier>,
}

impl AtomicModelStore for FakeStore {
    type Error = Infallible;

    fn load(&self) -> Result<Model, Self::Error> {
        let current = self.state.lock().unwrap().clone();
        if let Some(barrier) = &self.after_read {
            barrier.wait();
        }
        Ok(current)
    }

    fn compare_exchange(&self, expected: &Model, replacement: &Model) -> Result<bool, Self::Error> {
        let mut current = self.state.lock().unwrap();
        if *current != *expected {
            return Ok(false);
        }
        *current = replacement.clone();
        Ok(true)
    }
}

#[test]
fn injected_atomic_store_refuses_a_competing_whole_state_commit() {
    let mut state = model();
    state.snapshot.pending = vec![
        pending(work("main-1", main('a'), 2)),
        pending(work("main-2", main('b'), 1)),
    ];
    let store = FakeStore {
        state: Mutex::new(state),
        after_read: Some(Barrier::new(2)),
    };
    let (first, second) = std::thread::scope(|scope| {
        let first = scope.spawn(|| try_reserve(&store, 105).unwrap());
        let second = scope.spawn(|| try_reserve(&store, 105).unwrap());
        (first.join().unwrap(), second.join().unwrap())
    });
    assert!(matches!(
        (&first, &second),
        (
            StoredTransition::Reserved { .. },
            StoredTransition::Contended
        ) | (
            StoredTransition::Contended,
            StoredTransition::Reserved { .. }
        )
    ));
    let stored = store.state.lock().unwrap();
    assert_eq!(stored.revision, 5);
    assert_eq!(stored.reserved.len(), 1);
    assert_eq!(stored.reserved[0].id, "main-1");
    assert_eq!(stored.snapshot.pending[0].work.id, "main-2");
    assert_eq!(propose(&stored, 5, 105), Ok(Transition::Wait));
}

#[test]
fn stored_start_claim_remains_charged_and_invalid_claim_does_not_write() {
    let mut state = model();
    state.snapshot.pending = vec![
        pending(work("main-1", main('a'), 2)),
        pending(work("main-2", main('b'), 1)),
    ];
    let store = FakeStore {
        state: Mutex::new(state),
        after_read: None,
    };
    let StoredTransition::Reserved { work, revision } = try_reserve(&store, 105).unwrap() else {
        panic!("expected stored reservation")
    };
    assert_eq!(revision, 5);
    let invalid = Active {
        work: work.clone(),
        observed_class: None,
    };
    assert_eq!(
        try_record_start(&store, 105, invalid),
        Err(StoreError::Model(ModelError::Preview(
            PreviewError::MissingRunnerClass
        )))
    );
    assert_eq!(store.state.lock().unwrap().revision, 5);
    let supplied = Active {
        work,
        observed_class: Some(class()),
    };
    assert_eq!(
        try_record_start(&store, 105, supplied),
        Ok(StoredTransition::StartRecorded { revision: 6 })
    );
    assert_eq!(try_reserve(&store, 105), Ok(StoredTransition::Wait));
    let stored = store.state.lock().unwrap();
    assert!(stored.reserved.is_empty());
    assert_eq!(stored.snapshot.active.len(), 1);
    assert_eq!(stored.snapshot.pending.len(), 1);
}

#[test]
fn lost_commit_acknowledgement_remains_an_uncertain_outcome() {
    struct LostAcknowledgement(Mutex<Model>);
    impl AtomicModelStore for LostAcknowledgement {
        type Error = &'static str;

        fn load(&self) -> Result<Model, Self::Error> {
            Ok(self.0.lock().unwrap().clone())
        }

        fn compare_exchange(
            &self,
            expected: &Model,
            replacement: &Model,
        ) -> Result<bool, Self::Error> {
            let mut state = self.0.lock().unwrap();
            assert_eq!(*state, *expected);
            *state = replacement.clone();
            Err("acknowledgement lost")
        }
    }

    let mut state = model();
    state
        .snapshot
        .pending
        .push(pending(work("main", main('a'), 1)));
    let store = LostAcknowledgement(Mutex::new(state));
    assert_eq!(
        try_reserve(&store, 105),
        Err(StoreError::CommitUnknown("acknowledgement lost"))
    );
    let stored = store.0.lock().unwrap();
    assert_eq!(stored.revision, 5);
    assert_eq!(stored.reserved[0].id, "main");
    assert!(stored.snapshot.pending.is_empty());
}

#[test]
fn pending_pr_supersession_preserves_every_main_and_queue_order() {
    let mut state = model();
    let pr = WorkKind::PullRequest {
        repository_id: 42,
        number: 7,
    };
    state.snapshot.pending = vec![
        pending(work("old-1", pr.clone(), 1)),
        pending(work("main-1", main('a'), 1)),
        Pending {
            work: work("old-2", pr.clone(), 1),
            supersedes: Some("old-1".into()),
        },
        pending(work("release", WorkKind::Release { repository_id: 42 }, 1)),
        Pending {
            work: work("new", pr, 1),
            supersedes: Some("old-2".into()),
        },
        pending(work("main-2", main('b'), 1)),
    ];
    let proposal = propose_pending_supersession(&state, 4, 105)
        .unwrap()
        .unwrap();
    assert_eq!(
        proposal
            .removed
            .iter()
            .map(|work| work.id.as_str())
            .collect::<Vec<_>>(),
        ["old-1", "old-2"]
    );
    assert_eq!(proposal.next.revision, 5);
    assert_eq!(
        proposal
            .next
            .snapshot
            .pending
            .iter()
            .map(|entry| entry.work.id.as_str())
            .collect::<Vec<_>>(),
        ["main-1", "release", "new", "main-2"]
    );
    assert!(
        proposal
            .next
            .snapshot
            .pending
            .iter()
            .all(|entry| entry.supersedes.is_none())
    );
    let (selected, _) = reserved(propose(&proposal.next, 5, 105).unwrap());
    assert_eq!(selected.id, "main-1");
    assert_eq!(state.snapshot.pending.len(), 6, "input is not mutated");
}

#[test]
fn active_reserved_or_cross_pr_predecessors_are_never_pruned() {
    let pr = WorkKind::PullRequest {
        repository_id: 42,
        number: 7,
    };
    let mut state = model();
    state.snapshot.active.push(Active {
        work: work("old", pr.clone(), 1),
        observed_class: Some(class()),
    });
    state.snapshot.pending.push(Pending {
        work: work("new", pr.clone(), 1),
        supersedes: Some("old".into()),
    });
    assert_eq!(
        propose_pending_supersession(&state, 4, 105),
        Err(ModelError::UnreconciledSupersession)
    );
    state.snapshot.active.clear();
    state.reserved.push(work("old", pr, 1));
    assert_eq!(
        propose_pending_supersession(&state, 4, 105),
        Err(ModelError::UnreconciledSupersession)
    );
    assert_eq!(state.reserved.len(), 1);

    state.reserved.clear();
    state
        .snapshot
        .pending
        .insert(0, pending(work("old", main('a'), 1)));
    assert_eq!(
        propose_pending_supersession(&state, 4, 105),
        Err(ModelError::Preview(PreviewError::InvalidSupersession))
    );
    assert_eq!(state.snapshot.pending.len(), 2);
}

#[test]
fn competing_pending_supersession_proposals_commit_at_most_once() {
    let mut state = model();
    let pr = WorkKind::PullRequest {
        repository_id: 42,
        number: 7,
    };
    state.snapshot.pending = vec![
        pending(work("old", pr.clone(), 1)),
        Pending {
            work: work("new", pr, 1),
            supersedes: Some("old".into()),
        },
    ];
    let store = FakeStore {
        state: Mutex::new(state),
        after_read: Some(Barrier::new(2)),
    };
    let (first, second) = std::thread::scope(|scope| {
        let first = scope.spawn(|| try_supersede_pending(&store, 105).unwrap());
        let second = scope.spawn(|| try_supersede_pending(&store, 105).unwrap());
        (first.join().unwrap(), second.join().unwrap())
    });
    assert!(matches!(
        (&first, &second),
        (
            StoredTransition::PendingSuperseded { .. },
            StoredTransition::Contended
        ) | (
            StoredTransition::Contended,
            StoredTransition::PendingSuperseded { .. }
        )
    ));
    let stored = store.state.lock().unwrap();
    assert_eq!(stored.revision, 5);
    assert_eq!(stored.snapshot.pending.len(), 1);
    assert_eq!(stored.snapshot.pending[0].work.id, "new");
    assert_eq!(stored.snapshot.pending[0].supersedes, None);
}
