use super::*;
fn topology(run: &str, nodes: &[&str]) -> Topology {
    Topology {
        run_id: run.into(),
        stages: nodes
            .iter()
            .enumerate()
            .map(|(index, node)| Stage {
                index: u32::try_from(index).unwrap(),
                node_id: (*node).into(),
            })
            .collect(),
    }
}
#[test]
fn replacement_requires_new_run_and_absent_killed_node() {
    let mut observation = Observation {
        topologies: vec![topology("old", &["seed", "one"])],
        model_count: 1,
    };
    assert!(observation.startup_ready());
    assert_eq!(observation.recovery("two", "old"), Recovery::Pending);
    observation.topologies[0].run_id = "new".into();
    assert_eq!(observation.recovery("two", "old"), Recovery::Replacement);
    assert_eq!(observation.recovery("one", "old"), Recovery::Pending);
}
#[test]
fn fallback_and_withdraw_match_shell_precedence() {
    let mut observation = Observation {
        topologies: vec![topology("old", &["seed"])],
        model_count: 1,
    };
    assert_eq!(observation.recovery("one", "old"), Recovery::LocalFallback);
    observation.model_count = 0;
    assert_eq!(observation.recovery("one", "old"), Recovery::Withdraw);
    observation.topologies[0].stages.push(Stage {
        index: 1,
        node_id: "one-full".into(),
    });
    assert_eq!(observation.recovery("one", "old"), Recovery::Pending);
}
#[test]
fn downstream_selection_prefers_worker_other_than_driver() {
    let observation = Observation {
        topologies: vec![topology("old", &["seed", "one-full", "two-full"])],
        model_count: 1,
    };
    let workers = [
        ("one".into(), MemberId::WorkerOne),
        ("two".into(), MemberId::WorkerTwo),
    ];
    assert_eq!(
        observation.downstream_worker(&workers, MemberId::WorkerOne),
        Some(MemberId::WorkerTwo)
    );
    let fallback = Observation {
        topologies: vec![topology("old", &["seed", "one-full"])],
        model_count: 1,
    };
    assert_eq!(
        fallback.downstream_worker(&workers, MemberId::WorkerOne),
        Some(MemberId::WorkerOne)
    );
}
#[test]
fn downstream_selection_supports_a_third_worker() {
    let third = MemberId::new("worker-three", 0).unwrap();
    let workers = [
        ("one".into(), MemberId::WorkerOne),
        ("two".into(), MemberId::WorkerTwo),
        ("three".into(), third),
    ];
    let observation = Observation {
        topologies: vec![topology("old", &["seed", "three-full"])],
        model_count: 1,
    };
    assert_eq!(
        observation.downstream_worker(&workers, MemberId::WorkerOne),
        Some(third)
    );
}
#[test]
fn recovery_sequence_waits_for_all_configured_workers() {
    let mut sequence = Sequence::with_worker_count(false, 3).unwrap();
    sequence.advance(Facts {
        invite_present: true,
        ..Facts::default()
    });
    sequence.advance(Facts {
        workers_ready: true,
        ..Facts::default()
    });
    assert_eq!(
        sequence.advance(Facts {
            seed_peers: 2,
            ..Facts::default()
        }),
        Step::AwaitPeers
    );
    assert_eq!(
        sequence.advance(Facts {
            seed_peers: 3,
            ..Facts::default()
        }),
        Step::AwaitSplit
    );
}
#[test]
fn recovery_worker_count_rejects_outside_session_capacity() {
    for count in [0, 1, 16, usize::MAX] {
        assert!(Sequence::with_worker_count(false, count).is_err());
    }
}
#[test]
fn stable_sweeps_reset_on_pending_and_accept_any_outcome() {
    let mut stability = Stability::new(NonZeroUsize::new(2).unwrap(), Expected::Any);
    assert_eq!(stability.sweep(&[Recovery::Replacement]), None);
    assert_eq!(stability.sweep(&[Recovery::Pending]), None);
    assert_eq!(
        stability.sweep(&[Recovery::Withdraw, Recovery::Replacement]),
        None
    );
    assert_eq!(
        stability.sweep(&[Recovery::LocalFallback]),
        Some(Recovery::LocalFallback)
    );
}
#[test]
fn recovery_sequence_requires_stop_receipt_before_recovery() {
    let mut sequence = Sequence::new(true);
    assert_eq!(
        sequence.advance(Facts {
            stable_recovery: true,
            ..Facts::default()
        }),
        Step::AwaitInvite
    );
    assert_eq!(
        sequence.advance(Facts {
            invite_present: true,
            ..Facts::default()
        }),
        Step::StartWorkers
    );
    assert_eq!(
        sequence.advance(Facts {
            workers_ready: true,
            ..Facts::default()
        }),
        Step::AwaitPeers
    );
    assert_eq!(
        sequence.advance(Facts {
            seed_peers: 2,
            ..Facts::default()
        }),
        Step::AwaitSplit
    );
    assert_eq!(
        sequence.advance(Facts {
            split_ready: true,
            ..Facts::default()
        }),
        Step::StartupChat
    );
    assert_eq!(
        sequence.advance(Facts {
            chat_valid: chat_valid("chat.completion", 1),
            ..Facts::default()
        }),
        Step::SelectDownstream
    );
    assert_eq!(
        sequence.advance(Facts {
            downstream: Some(MemberId::WorkerTwo),
            ..Facts::default()
        }),
        Step::StopWorker(MemberId::WorkerTwo)
    );
    assert_eq!(
        sequence.advance(Facts {
            stopped: Some(MemberId::WorkerOne),
            ..Facts::default()
        }),
        Step::StopWorker(MemberId::WorkerTwo)
    );
    assert_eq!(
        sequence.advance(Facts {
            stopped: Some(MemberId::WorkerTwo),
            ..Facts::default()
        }),
        Step::AwaitRecovery
    );
    assert_eq!(
        sequence.advance(Facts {
            stable_recovery: true,
            ..Facts::default()
        }),
        Step::RecoveryChat
    );
    assert_eq!(
        sequence.advance(Facts {
            chat_valid: true,
            ..Facts::default()
        }),
        Step::Complete
    );
}
#[test]
fn recovery_sequence_without_inference_skips_chat_gates() {
    let mut sequence = Sequence::new(false);
    let facts = || Facts {
        invite_present: true,
        workers_ready: true,
        seed_peers: 2,
        split_ready: true,
        downstream: Some(MemberId::WorkerOne),
        stopped: Some(MemberId::WorkerOne),
        stable_recovery: true,
        ..Facts::default()
    };
    for expected in [
        Step::StartWorkers,
        Step::AwaitPeers,
        Step::AwaitSplit,
        Step::SelectDownstream,
        Step::StopWorker(MemberId::WorkerOne),
        Step::AwaitRecovery,
        Step::Complete,
    ] {
        assert_eq!(sequence.advance(facts()), expected);
    }
}
#[test]
fn startup_requires_distinct_stage_nodes_and_advertised_model() {
    for (nodes, models) in [
        (&["seed", "seed"][..], 1),
        (&["seed", "worker"][..], 0),
        (&["seed"][..], 1),
    ] {
        let observation = Observation {
            topologies: vec![topology("run", nodes)],
            model_count: models,
        };
        assert!(!observation.startup_ready());
    }
    assert!(!chat_valid("chat.completion", 0));
    assert!(!chat_valid("error", 1));
}
