use super::*;
use crate::capacity_preview::{Pending, Role, RunnerClass, Snapshot, WorkKind};
use std::collections::{BTreeMap, BTreeSet};
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

fn work(id: &str, kind: WorkKind) -> Work {
    Work {
        id: id.into(),
        kind,
        role: "mac".into(),
        workers: 1,
    }
}

fn main(revision: char) -> WorkKind {
    WorkKind::Main {
        repository_id: 42,
        revision: revision.to_string().repeat(40),
    }
}

fn journal() -> Journal {
    Journal {
        model: Model {
            revision: 4,
            snapshot: Snapshot {
                observed_at: 100,
                max_age: 10,
                pools: BTreeMap::from([("mac".into(), 1)]),
                roles: BTreeMap::from([(
                    "mac".into(),
                    Role {
                        runner: class(),
                        pools: BTreeSet::from(["mac".into()]),
                    },
                )]),
                active: Vec::new(),
                pending: Vec::new(),
            },
            reserved: Vec::new(),
        },
        max_receipts: 128,
        receipts: BTreeMap::new(),
    }
}

fn pending(work: Work) -> Pending {
    Pending {
        work,
        supersedes: None,
    }
}

#[derive(Clone, Copy)]
enum Failure {
    None,
    BeforeCommit,
    AfterCommit,
}

// This process-local fake demonstrates the required contract, not a shared
// store, authentic state or a production recovery mechanism.
struct FakeStore {
    state: Mutex<(Journal, Failure)>,
    after_load: Option<Barrier>,
}

impl AtomicJournalStore for FakeStore {
    type Error = &'static str;

    fn load(&self) -> Result<Journal, Self::Error> {
        let journal = self.state.lock().unwrap().0.clone();
        if let Some(barrier) = &self.after_load {
            barrier.wait();
        }
        Ok(journal)
    }

    fn compare_exchange(
        &self,
        expected: &Journal,
        replacement: &Journal,
    ) -> Result<bool, Self::Error> {
        let mut state = self.state.lock().unwrap();
        if &state.0 != expected {
            return Ok(false);
        }
        let failure = state.1;
        state.1 = Failure::None;
        if matches!(failure, Failure::BeforeCommit) {
            return Err("acknowledgement unavailable");
        }
        state.0 = replacement.clone();
        if matches!(failure, Failure::AfterCommit) {
            return Err("acknowledgement unavailable");
        }
        Ok(true)
    }
}

fn store(state: Journal, failure: Failure) -> FakeStore {
    FakeStore {
        state: Mutex::new((state, failure)),
        after_load: None,
    }
}

#[test]
fn lost_acknowledgement_retries_the_identical_operation_without_double_reserving() {
    for failure in [Failure::BeforeCommit, Failure::AfterCommit] {
        let mut state = journal();
        state.model.snapshot.pending = vec![
            pending(work("main-1", main('a'))),
            pending(work("main-2", main('b'))),
        ];
        let store = store(state, failure);
        let request = Request::Reserve { now: 105 };
        assert_eq!(
            apply(&store, "reserve-1", request.clone()),
            Err(JournalError::CommitUnknown("acknowledgement unavailable"))
        );
        let expected = Receipt::Reserved {
            work: work("main-1", main('a')),
            revision: 5,
        };
        let recovered = apply(&store, "reserve-1", request.clone()).unwrap();
        assert_eq!(
            recovered,
            if matches!(failure, Failure::AfterCommit) {
                Outcome::Replayed(expected.clone())
            } else {
                Outcome::Applied(expected.clone())
            }
        );
        assert_eq!(
            apply(&store, "reserve-1", request),
            Ok(Outcome::Replayed(expected))
        );
        let state = &store.state.lock().unwrap().0;
        assert_eq!(state.model.reserved.len(), 1);
        assert_eq!(state.model.snapshot.pending[0].work.id, "main-2");
        assert_eq!(state.receipts.len(), 1);
    }
}

#[test]
fn receipts_survive_later_transitions_and_bind_the_complete_request() {
    let mut state = journal();
    let selected = work("main", main('a'));
    state.model.snapshot.pending.push(pending(selected.clone()));
    let store = store(state, Failure::None);
    let reserve = Request::Reserve { now: 105 };
    let receipt = Receipt::Reserved {
        work: selected.clone(),
        revision: 5,
    };
    assert_eq!(
        apply(&store, "reserve", reserve.clone()),
        Ok(Outcome::Applied(receipt.clone()))
    );
    let start = Request::RecordStart {
        now: 105,
        observed: Box::new(Active {
            work: selected,
            observed_class: Some(class()),
        }),
    };
    store.state.lock().unwrap().1 = Failure::AfterCommit;
    assert_eq!(
        apply(&store, "start", start.clone()),
        Err(JournalError::CommitUnknown("acknowledgement unavailable"))
    );
    assert_eq!(
        apply(&store, "reserve", reserve),
        Ok(Outcome::Replayed(receipt))
    );
    assert_eq!(
        apply(&store, "start", start),
        Ok(Outcome::Replayed(Receipt::StartRecorded { revision: 6 }))
    );
    assert_eq!(
        apply(&store, "reserve", Request::Reserve { now: 106 }),
        Err(JournalError::ReusedOperationId)
    );
    assert_eq!(
        apply(&store, "reserve", Request::SupersedePending { now: 105 }),
        Err(JournalError::ReusedOperationId)
    );
    let state = &store.state.lock().unwrap().0;
    assert_eq!(state.model.revision, 6);
    assert_eq!(state.model.snapshot.active.len(), 1);
    assert_eq!(state.receipts.len(), 2);
}

#[test]
fn pending_supersession_replays_its_removed_work_without_removing_main() {
    let mut state = journal();
    let pr = WorkKind::PullRequest {
        repository_id: 42,
        number: 7,
    };
    let old = work("old", pr.clone());
    state.model.snapshot.pending = vec![
        pending(work("main", main('a'))),
        pending(old.clone()),
        Pending {
            work: work("new", pr),
            supersedes: Some("old".into()),
        },
    ];
    let store = store(state, Failure::AfterCommit);
    let request = Request::SupersedePending { now: 105 };
    assert!(matches!(
        apply(&store, "supersede", request.clone()),
        Err(JournalError::CommitUnknown(_))
    ));
    assert_eq!(
        apply(&store, "supersede", request),
        Ok(Outcome::Replayed(Receipt::PendingSuperseded {
            removed: vec![old],
            revision: 5,
        }))
    );
    let state = &store.state.lock().unwrap().0;
    assert_eq!(state.model.snapshot.pending[0].work.id, "main");
    assert_eq!(state.model.snapshot.pending[1].work.id, "new");
    assert_eq!(state.receipts.len(), 1);
}

#[test]
fn concurrent_operations_commit_at_most_one_reservation_and_receipt() {
    for (first_id, second_id) in [("one", "two"), ("one", "one")] {
        let mut state = journal();
        state.model.snapshot.pools = BTreeMap::from([("mac".into(), 2), ("org".into(), 2)]);
        state
            .model
            .snapshot
            .roles
            .get_mut("mac")
            .unwrap()
            .pools
            .insert("org".into());
        let mut first = work("main-1", main('a'));
        first.workers = 2;
        state.model.snapshot.pending = vec![pending(first), pending(work("main-2", main('b')))];
        let store = FakeStore {
            state: Mutex::new((state, Failure::None)),
            after_load: Some(Barrier::new(2)),
        };
        let (a, b) = std::thread::scope(|scope| {
            let a = scope.spawn(|| apply(&store, first_id, Request::Reserve { now: 105 }).unwrap());
            let b =
                scope.spawn(|| apply(&store, second_id, Request::Reserve { now: 105 }).unwrap());
            (a.join().unwrap(), b.join().unwrap())
        });
        assert!(matches!(
            (&a, &b),
            (Outcome::Applied(_), Outcome::Contended) | (Outcome::Contended, Outcome::Applied(_))
        ));
        let state = &store.state.lock().unwrap().0;
        assert_eq!(state.model.revision, 5);
        assert_eq!(state.model.reserved.len(), 1);
        assert_eq!(state.model.reserved[0].id, "main-1");
        assert_eq!(state.model.reserved[0].workers, 2);
        assert_eq!(state.model.snapshot.pending.len(), 1);
        assert_eq!(state.model.snapshot.pending[0].work.id, "main-2");
        assert_eq!(propose(&state.model, 5, 105), Ok(Transition::Wait));
        assert_eq!(state.receipts.len(), 1);
    }
}

#[test]
fn invalid_start_writes_nothing_and_valid_start_keeps_capacity_charged() {
    let mut state = journal();
    state.model.snapshot.pools = BTreeMap::from([("mac".into(), 2), ("org".into(), 2)]);
    state
        .model
        .snapshot
        .roles
        .get_mut("mac")
        .unwrap()
        .pools
        .insert("org".into());
    let mut selected = work("main-1", main('a'));
    selected.workers = 2;
    state.model.snapshot.pending = vec![
        pending(selected.clone()),
        pending(work("main-2", main('b'))),
    ];
    let store = store(state, Failure::None);
    assert_eq!(
        apply(&store, "reserve", Request::Reserve { now: 105 }),
        Ok(Outcome::Applied(Receipt::Reserved {
            work: selected.clone(),
            revision: 5,
        }))
    );
    let before = store.state.lock().unwrap().0.clone();
    let invalid = Request::RecordStart {
        now: 105,
        observed: Box::new(Active {
            work: selected.clone(),
            observed_class: None,
        }),
    };
    assert_eq!(
        apply(&store, "start", invalid),
        Err(JournalError::Model(ModelError::Preview(
            crate::capacity_preview::PreviewError::MissingRunnerClass
        )))
    );
    assert_eq!(store.state.lock().unwrap().0, before);
    let valid = Request::RecordStart {
        now: 105,
        observed: Box::new(Active {
            work: selected,
            observed_class: Some(class()),
        }),
    };
    assert_eq!(
        apply(&store, "start", valid),
        Ok(Outcome::Applied(Receipt::StartRecorded { revision: 6 }))
    );
    assert_eq!(
        apply(&store, "next", Request::Reserve { now: 105 }),
        Ok(Outcome::Wait)
    );
    let state = &store.state.lock().unwrap().0;
    assert!(state.model.reserved.is_empty());
    assert_eq!(state.model.snapshot.active.len(), 1);
    assert_eq!(state.model.snapshot.pending.len(), 1);
    assert_eq!(state.receipts.len(), 2);
}

#[test]
fn competing_supersession_commits_at_most_one_receipt() {
    let mut state = journal();
    let pr = WorkKind::PullRequest {
        repository_id: 42,
        number: 7,
    };
    let old = work("old", pr.clone());
    state.model.snapshot.pending = vec![
        pending(old.clone()),
        Pending {
            work: work("new", pr),
            supersedes: Some("old".into()),
        },
    ];
    let store = FakeStore {
        state: Mutex::new((state, Failure::None)),
        after_load: Some(Barrier::new(2)),
    };
    let (first, second) = std::thread::scope(|scope| {
        let first =
            scope.spawn(|| apply(&store, "first", Request::SupersedePending { now: 105 }).unwrap());
        let second = scope
            .spawn(|| apply(&store, "second", Request::SupersedePending { now: 105 }).unwrap());
        (first.join().unwrap(), second.join().unwrap())
    });
    let receipt = Outcome::Applied(Receipt::PendingSuperseded {
        removed: vec![old],
        revision: 5,
    });
    assert!(
        (first == receipt && second == Outcome::Contended)
            || (first == Outcome::Contended && second == receipt)
    );
    let state = &store.state.lock().unwrap().0;
    assert_eq!(state.model.revision, 5);
    assert_eq!(state.model.snapshot.pending.len(), 1);
    assert_eq!(state.model.snapshot.pending[0].work.id, "new");
    assert_eq!(state.model.snapshot.pending[0].supersedes, None);
    assert_eq!(state.receipts.len(), 1);
}

#[test]
fn invalid_ids_and_noop_results_do_not_change_the_supplied_journal() {
    let state = journal();
    let store = store(state.clone(), Failure::None);
    for id in ["", "has spaces", "a\nb"] {
        assert_eq!(
            apply(&store, id, Request::Reserve { now: 105 }),
            Err(JournalError::InvalidOperationId)
        );
    }
    assert_eq!(
        apply(&store, "wait", Request::Reserve { now: 105 }),
        Ok(Outcome::Wait)
    );
    assert_eq!(
        apply(
            &store,
            "no-supersession",
            Request::SupersedePending { now: 105 }
        ),
        Ok(Outcome::NoSupersession)
    );
    assert_eq!(store.state.lock().unwrap().0, state);
}

#[test]
fn full_journal_replays_but_never_evicts_or_commits_a_new_operation() {
    let mut state = journal();
    state.max_receipts = 1;
    state.model.snapshot.pending = vec![
        pending(work("main-1", main('a'))),
        pending(work("main-2", main('b'))),
    ];
    let store = store(state, Failure::None);
    let request = Request::Reserve { now: 105 };
    let receipt = match apply(&store, "reserve-1", request.clone()).unwrap() {
        Outcome::Applied(receipt) => receipt,
        outcome => panic!("unexpected outcome: {outcome:?}"),
    };
    let committed = store.state.lock().unwrap().0.clone();
    assert_eq!(
        apply(&store, "reserve-1", request),
        Ok(Outcome::Replayed(receipt))
    );
    assert_eq!(
        apply(&store, "reserve-1", Request::Reserve { now: 106 }),
        Err(JournalError::ReusedOperationId)
    );
    for request in [
        Request::Reserve { now: 105 },
        Request::SupersedePending { now: 105 },
        Request::RecordStart {
            now: 105,
            observed: Box::new(Active {
                work: work("main-1", main('a')),
                observed_class: Some(class()),
            }),
        },
    ] {
        assert_eq!(
            apply(&store, "new-operation", request),
            Err(JournalError::ReceiptCapacityExhausted)
        );
    }
    assert_eq!(store.state.lock().unwrap().0, committed);
}

#[test]
fn zero_or_already_exceeded_receipt_limit_is_rejected_without_writes() {
    let mut state = journal();
    state.max_receipts = 0;
    let zero_store = store(state.clone(), Failure::None);
    assert_eq!(
        apply(&zero_store, "reserve", Request::Reserve { now: 105 }),
        Err(JournalError::InvalidReceiptLimit)
    );
    assert_eq!(zero_store.state.lock().unwrap().0, state);

    state.max_receipts = 1;
    let entry = Entry {
        request: Request::Reserve { now: 105 },
        receipt: Receipt::StartRecorded { revision: 5 },
    };
    state.receipts.insert("one".into(), entry.clone());
    state.receipts.insert("two".into(), entry);
    let overfull_store = store(state.clone(), Failure::None);
    assert_eq!(
        apply(&overfull_store, "one", Request::Reserve { now: 105 }),
        Err(JournalError::InvalidReceiptLimit)
    );
    assert_eq!(overfull_store.state.lock().unwrap().0, state);
}

#[test]
fn last_slot_with_lost_acknowledgement_replays_the_committed_receipt() {
    let mut state = journal();
    state.max_receipts = 1;
    let selected = work("main", main('a'));
    state.model.snapshot.pending.push(pending(selected.clone()));
    let store = store(state, Failure::AfterCommit);
    let request = Request::Reserve { now: 105 };
    assert_eq!(
        apply(&store, "reserve", request.clone()),
        Err(JournalError::CommitUnknown("acknowledgement unavailable"))
    );
    assert_eq!(
        apply(&store, "reserve", request),
        Ok(Outcome::Replayed(Receipt::Reserved {
            work: selected,
            revision: 5,
        }))
    );
    assert_eq!(
        apply(&store, "another", Request::Reserve { now: 105 }),
        Err(JournalError::ReceiptCapacityExhausted)
    );
    let state = &store.state.lock().unwrap().0;
    assert_eq!(state.receipts.len(), 1);
    assert_eq!(state.model.reserved.len(), 1);
}

#[test]
fn racing_for_last_receipt_slot_commits_at_most_one_operation() {
    let mut state = journal();
    state.max_receipts = 1;
    state.model.snapshot.pending = vec![
        pending(work("main-1", main('a'))),
        pending(work("main-2", main('b'))),
    ];
    let store = FakeStore {
        state: Mutex::new((state, Failure::None)),
        after_load: Some(Barrier::new(2)),
    };
    let (a, b) = std::thread::scope(|scope| {
        let a = scope.spawn(|| apply(&store, "one", Request::Reserve { now: 105 }).unwrap());
        let b = scope.spawn(|| apply(&store, "two", Request::Reserve { now: 105 }).unwrap());
        (a.join().unwrap(), b.join().unwrap())
    });
    assert!(matches!(
        (&a, &b),
        (Outcome::Applied(_), Outcome::Contended) | (Outcome::Contended, Outcome::Applied(_))
    ));
    let state = &store.state.lock().unwrap().0;
    assert_eq!(state.max_receipts, 1);
    assert_eq!(state.receipts.len(), 1);
    assert_eq!(state.model.reserved.len(), 1);
}
