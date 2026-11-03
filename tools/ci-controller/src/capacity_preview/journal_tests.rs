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
        state.model.snapshot.pending = vec![
            pending(work("main-1", main('a'))),
            pending(work("main-2", main('b'))),
        ];
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
        assert_eq!(state.model.reserved.len(), 1);
        assert_eq!(state.model.snapshot.pending.len(), 1);
        assert_eq!(state.receipts.len(), 1);
    }
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
