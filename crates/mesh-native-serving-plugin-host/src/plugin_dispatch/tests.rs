//! Deadline, ordering, and shutdown behavior of the plugin dispatch workers.

use std::{
    sync::{Arc, atomic::Ordering, mpsc::sync_channel},
    time::{Duration, Instant},
};

use skippy_server::frontend::{
    GenerationAbort, GenerationCommit, GenerationLifecycleIngress, GenerationLifecycleObservation,
    GenerationStart, LinearProposalDiscardReason, LinearProposalQuery, LinearProposalSourceOutcome,
};

use super::{PluginCommand, PluginCommandQueue, PluginCommandQueueError, PluginDriver};
use crate::{
    NativeLifecycleIngress,
    test_support::{
        fake_active, fake_active_with_events, fake_active_with_failing_proposal,
        fake_active_with_late_candidate, fake_active_with_observations, fake_active_with_options,
        fake_active_with_timing, fake_observations, proposal_query, wait_for_event,
    },
};

#[test]
fn blocking_plugin_cannot_extend_the_decode_deadline() {
    let active = fake_active(Duration::from_millis(250));
    let observations = fake_observations(&active);
    let driver = PluginDriver::spawn(active).unwrap();
    let started = Instant::now();
    let result = driver
        .propose(LinearProposalQuery::new(
            1,
            2,
            16,
            16,
            0,
            8_192,
            started + Duration::from_millis(5),
        ))
        .unwrap();
    let elapsed = started.elapsed();

    assert!(result.proposal.unwrap().is_none());
    assert_eq!(
        result.telemetry.outcome,
        LinearProposalSourceOutcome::HostDeadlineExceeded
    );
    assert!(
        elapsed < Duration::from_millis(150),
        "decode waited {elapsed:?} for a blocking plugin"
    );
    drop(driver);
    assert_eq!(observations.cancel_count.load(Ordering::SeqCst), 1);
}

#[test]
fn slow_commit_abstains_before_plugin_dispatch_and_later_positions_recover() {
    let (active, events, _) = fake_active_with_timing(
        Duration::ZERO,
        Duration::from_millis(40),
        Duration::ZERO,
        false,
    );
    let driver = Arc::new(PluginDriver::spawn(active).unwrap());
    let ingress = NativeLifecycleIngress {
        driver: Arc::clone(&driver),
    };
    ingress
        .try_submit(GenerationLifecycleObservation::Committed(
            GenerationCommit {
                request_id: 1,
                session_id: 2,
                generated_token_count: 1,
                token_ids: vec![4].into_boxed_slice(),
            },
        ))
        .unwrap();
    wait_for_event(events.as_ref(), "commit");

    let started = Instant::now();
    let missed = driver
        .propose(proposal_query(started + Duration::from_millis(5)))
        .unwrap();
    assert!(missed.proposal.unwrap().is_none());
    assert_eq!(
        missed.telemetry.outcome,
        LinearProposalSourceOutcome::HostDeadlineExceeded
    );
    assert!(
        started.elapsed() < Duration::from_millis(30),
        "proposal wait exceeded its deadline: {:?}",
        started.elapsed()
    );

    let recovered = driver
        .propose(proposal_query(Instant::now() + Duration::from_millis(100)))
        .unwrap();
    assert!(recovered.proposal.unwrap().is_none());
    assert_eq!(
        recovered.telemetry.outcome,
        LinearProposalSourceOutcome::Abstained
    );
    assert_eq!(*events.lock().unwrap(), ["commit", "proposal"]);
}

#[test]
fn running_passive_discard_cannot_delay_the_next_proposal() {
    let (active, events, _) = fake_active_with_timing(
        Duration::ZERO,
        Duration::ZERO,
        Duration::from_millis(100),
        false,
    );
    let driver = Arc::new(PluginDriver::spawn(active).unwrap());
    driver
        .enqueue(PluginCommand::Discard(
            vec![1],
            LinearProposalDiscardReason::PositionMismatch,
        ))
        .unwrap();
    wait_for_event(events.as_ref(), "discard");

    let started = Instant::now();
    let response = driver
        .propose(proposal_query(started + Duration::from_millis(20)))
        .unwrap();
    assert!(response.proposal.unwrap().is_none());
    assert_eq!(
        response.telemetry.outcome,
        LinearProposalSourceOutcome::Abstained
    );
    assert!(started.elapsed() < Duration::from_millis(60));
    assert_eq!(*events.lock().unwrap(), ["discard", "proposal"]);
}

#[test]
fn worker_reports_pre_dispatch_deadlines_without_running_the_callback() {
    let (active, events, _) = fake_active_with_timing(
        Duration::ZERO,
        Duration::from_millis(40),
        Duration::ZERO,
        false,
    );
    let driver = PluginDriver::spawn(active).unwrap();
    driver
        .enqueue(PluginCommand::Committed(GenerationCommit {
            request_id: 1,
            session_id: 2,
            generated_token_count: 1,
            token_ids: vec![4].into_boxed_slice(),
        }))
        .unwrap();
    wait_for_event(events.as_ref(), "commit");
    let (reply, response) = sync_channel(1);
    driver
        .queue
        .try_enqueue(PluginCommand::Proposal(
            proposal_query(Instant::now() + Duration::from_millis(5)),
            reply,
        ))
        .unwrap();

    let response = response.recv_timeout(Duration::from_millis(100)).unwrap();
    assert!(response.proposal.unwrap().is_none());
    assert_eq!(
        response.telemetry.outcome,
        LinearProposalSourceOutcome::DeadlineExceededBeforeDispatch
    );
    assert_eq!(*events.lock().unwrap(), ["commit"]);
}

#[test]
fn late_candidate_is_reported_and_not_forwarded_to_the_decode() {
    let driver =
        PluginDriver::spawn(fake_active_with_late_candidate(Duration::from_millis(20))).unwrap();
    let (reply, response) = sync_channel(1);
    driver
        .queue
        .try_enqueue(PluginCommand::Proposal(
            proposal_query(Instant::now() + Duration::from_millis(5)),
            reply,
        ))
        .unwrap();

    let response = response.recv_timeout(Duration::from_millis(100)).unwrap();
    assert!(response.proposal.unwrap().is_none());
    assert_eq!(
        response.telemetry.outcome,
        LinearProposalSourceOutcome::CandidateReturnedTooLate
    );
}

#[test]
fn stopped_worker_rejects_lifecycle_delivery() {
    let queue = PluginCommandQueue::new();
    queue.close();

    assert!(matches!(
        queue.try_enqueue(PluginCommand::Abort(GenerationAbort {
            request_id: 1,
            session_id: 2,
        })),
        Err(PluginCommandQueueError::Stopped)
    ));
}

#[test]
fn lifecycle_ingress_shares_plugin_queue_order_with_proposals() {
    let (active, events) = fake_active_with_events(Duration::ZERO);
    let driver = Arc::new(PluginDriver::spawn(active).unwrap());
    let ingress = NativeLifecycleIngress {
        driver: Arc::clone(&driver),
    };
    ingress
        .try_submit(GenerationLifecycleObservation::Started(GenerationStart {
            request_id: 1,
            session_id: 2,
            agent_session_id: None,
            prompt_token_ids: Arc::from([3]),
        }))
        .unwrap();
    ingress
        .try_submit(GenerationLifecycleObservation::Committed(
            GenerationCommit {
                request_id: 1,
                session_id: 2,
                generated_token_count: 1,
                token_ids: vec![4].into_boxed_slice(),
            },
        ))
        .unwrap();
    driver
        .propose(LinearProposalQuery::new(
            1,
            2,
            1,
            1,
            0,
            8,
            Instant::now() + Duration::from_millis(100),
        ))
        .unwrap();

    assert_eq!(*events.lock().unwrap(), ["begin", "commit", "proposal"]);
}

#[test]
fn proposal_applies_pending_tokens_before_lookup() {
    let (active, events) = fake_active_with_events(Duration::ZERO);
    active
        .begin(&GenerationStart {
            request_id: 1,
            session_id: 2,
            agent_session_id: None,
            prompt_token_ids: Arc::from([3]),
        })
        .unwrap();
    let result = active
        .propose(
            LinearProposalQuery::new(
                1,
                2,
                1,
                3,
                2,
                8,
                Instant::now() + Duration::from_millis(100),
            )
            .with_pending_token_ids(vec![4, 5].into_boxed_slice()),
        )
        .unwrap();

    assert!(result.is_none());
    assert_eq!(*events.lock().unwrap(), ["begin", "commit", "proposal"]);
}

#[test]
fn blocking_commit_cannot_extend_the_proposal_deadline() {
    let (active, _, _) = fake_active_with_timing(
        Duration::ZERO,
        Duration::from_millis(250),
        Duration::ZERO,
        false,
    );
    let driver = Arc::new(PluginDriver::spawn(active).unwrap());
    let ingress = NativeLifecycleIngress {
        driver: Arc::clone(&driver),
    };
    ingress
        .try_submit(GenerationLifecycleObservation::Started(GenerationStart {
            request_id: 1,
            session_id: 2,
            agent_session_id: None,
            prompt_token_ids: Arc::from([3]),
        }))
        .unwrap();

    let started = Instant::now();
    let result = driver
        .propose(
            LinearProposalQuery::new(1, 2, 1, 2, 1, 8, started + Duration::from_millis(5))
                .with_pending_token_ids(vec![4].into_boxed_slice()),
        )
        .unwrap();

    assert!(result.proposal.unwrap().is_none());
    assert!(
        started.elapsed() < Duration::from_millis(150),
        "proposal waited {:?} for a blocking commit",
        started.elapsed()
    );
}

#[test]
fn lifecycle_callback_failure_is_observed_without_poisoning_the_driver() {
    let (active, _, _) = fake_active_with_options(Duration::ZERO, true);
    let driver = Arc::new(PluginDriver::spawn(active).unwrap());
    let ingress = NativeLifecycleIngress {
        driver: Arc::clone(&driver),
    };
    ingress
        .try_submit(GenerationLifecycleObservation::Started(GenerationStart {
            request_id: 7,
            session_id: 9,
            agent_session_id: None,
            prompt_token_ids: Arc::from([3]),
        }))
        .unwrap();

    driver
        .propose(LinearProposalQuery::new(
            7,
            9,
            1,
            1,
            0,
            8,
            Instant::now() + Duration::from_millis(100),
        ))
        .unwrap();

    assert_eq!(driver.lifecycle_delivery_failures(), 1);
    assert!(driver.ensure_healthy().is_ok());
}

#[test]
fn generation_abort_bypasses_unhealthy_driver_gate() {
    let (active, _, abort_count) = fake_active_with_observations(Duration::ZERO);
    let driver = Arc::new(PluginDriver::spawn(active).unwrap());
    *driver.fatal_error.lock().unwrap() = Some("report proposal failed".to_string());
    let ingress = NativeLifecycleIngress {
        driver: Arc::clone(&driver),
    };

    ingress
        .try_submit(GenerationLifecycleObservation::Aborted(GenerationAbort {
            request_id: 7,
            session_id: 9,
        }))
        .unwrap();

    drop(ingress);
    drop(driver);
    assert_eq!(abort_count.load(Ordering::SeqCst), 1);
}

#[test]
fn a_full_queue_can_still_be_closed_and_drained() {
    // Closing takes no queue capacity, so a saturated queue can never strand
    // its worker the way a queued shutdown command could.
    let queue = PluginCommandQueue::new();
    let mut queued = 0_usize;
    while queue
        .try_enqueue(PluginCommand::Abort(GenerationAbort {
            request_id: 1,
            session_id: 2,
        }))
        .is_ok()
    {
        queued += 1;
        assert!(queued < 10_000, "queue never reported full");
    }

    queue.close();
    for _ in 0..queued {
        assert!(queue.next().is_some(), "close must not discard queued work");
    }
    assert!(
        queue.next().is_none(),
        "a drained, closed queue must stop its worker"
    );
}

#[test]
fn dropping_the_driver_drains_its_backlog_and_shuts_the_plugin_down() {
    let (active, events, _) = fake_active_with_timing(
        Duration::ZERO,
        Duration::from_millis(2),
        Duration::ZERO,
        false,
    );
    let observations = fake_observations(&active);
    let driver = PluginDriver::spawn(active).unwrap();
    for generated_token_count in 1..=20 {
        driver
            .enqueue(PluginCommand::Committed(GenerationCommit {
                request_id: 1,
                session_id: 2,
                generated_token_count,
                token_ids: vec![4].into_boxed_slice(),
            }))
            .unwrap();
    }

    drop(driver);
    assert_eq!(
        events.lock().unwrap().len(),
        20,
        "backlog must be delivered"
    );
    assert_eq!(observations.shutdown_count.load(Ordering::SeqCst), 1);
}

#[test]
fn a_full_passive_queue_still_accepts_the_terminal_discard() {
    // Reserved headroom keeps a deadline discard deliverable, so the plugin
    // always learns the fate of a decision the host withheld.
    let queue = PluginCommandQueue::new();
    while queue
        .try_enqueue(PluginCommand::Discard(
            vec![1],
            LinearProposalDiscardReason::PositionMismatch,
        ))
        .is_ok()
    {}

    assert!(matches!(
        queue.try_enqueue(PluginCommand::Discard(
            vec![2],
            LinearProposalDiscardReason::PositionMismatch,
        )),
        Err(PluginCommandQueueError::Full)
    ));
    queue
        .try_enqueue_terminal(PluginCommand::Discard(
            vec![3],
            LinearProposalDiscardReason::DeadlineExceeded,
        ))
        .expect("terminal discard must use the reserved headroom");
}

#[test]
fn a_late_candidate_delivers_its_discard_to_the_plugin() {
    // The host withholds a late candidate, so `discard` is the only way the
    // plugin can learn that decision's fate.
    let active = fake_active_with_late_candidate(Duration::from_millis(20));
    let observations = fake_observations(&active);
    let driver = PluginDriver::spawn(active).unwrap();
    let (reply, response) = sync_channel(1);
    driver
        .queue
        .try_enqueue(PluginCommand::Proposal(
            proposal_query(Instant::now() + Duration::from_millis(5)),
            reply,
        ))
        .unwrap();

    let response = response.recv_timeout(Duration::from_millis(500)).unwrap();
    assert!(response.proposal.unwrap().is_none());
    assert_eq!(
        response.telemetry.outcome,
        LinearProposalSourceOutcome::CandidateReturnedTooLate
    );
    wait_for_event(observations.events.as_ref(), "discard");
}

#[test]
fn a_failing_proposal_callback_reports_source_error_instead_of_abstaining() {
    // Fail open for decode, but keep the plugin's failure distinguishable
    // from a deliberate abstention.
    let driver = PluginDriver::spawn(fake_active_with_failing_proposal()).unwrap();
    let response = driver
        .propose(proposal_query(Instant::now() + Duration::from_millis(200)))
        .unwrap();

    assert!(response.proposal.is_err());
    assert_eq!(
        response.telemetry.outcome,
        LinearProposalSourceOutcome::SourceError
    );
}
