use super::super::tests::direct_iteration;
use super::*;

#[test]
fn selected_sessions_have_the_same_order_for_every_arrival_permutation() {
    for arrival in [
        ["a", "b", "c"],
        ["a", "c", "b"],
        ["b", "a", "c"],
        ["b", "c", "a"],
        ["c", "a", "b"],
        ["c", "b", "a"],
    ] {
        let mut queue = arrival.map(|session| direct_iteration(session, 1)).into();

        let batch = take_direct_iteration_batch(&mut queue, 3, 3);

        assert_eq!(session_ids(&batch), ["a", "b", "c"]);
        assert!(queue.is_empty());
    }
}

#[test]
fn canonical_execution_order_does_not_change_fifo_admission_at_the_batch_cap() {
    let mut queue = VecDeque::from([
        direct_iteration("z-oldest", 1),
        direct_iteration("b-next", 1),
        direct_iteration("a-later", 1),
    ]);

    let batch = take_direct_iteration_batch(&mut queue, 2, 8);

    assert_eq!(session_ids(&batch), ["b-next", "z-oldest"]);
    assert_eq!(queue.pop_front().unwrap().session_id, "a-later");
    assert!(queue.is_empty());
}

#[test]
fn canonical_execution_preserves_budget_and_deferred_session_fifo_across_waves() {
    let mut queue = VecDeque::from([
        direct_iteration("z", 2),
        direct_iteration("b", 4),
        direct_iteration("z", 1),
        direct_iteration("a", 1),
        direct_iteration("c", 1),
    ]);

    let first = take_direct_iteration_batch(&mut queue, 3, 3);

    assert_eq!(session_ids(&first), ["a", "z"]);
    assert_eq!(first.iter().map(|r| r.token_ids.len()).sum::<usize>(), 3);
    assert_eq!(
        queue
            .iter()
            .map(|r| (r.session_id.as_str(), r.token_ids.len()))
            .collect::<Vec<_>>(),
        [("b", 4), ("z", 1), ("c", 1)]
    );
    let second = take_direct_iteration_batch(&mut queue, 3, 4);
    assert_eq!(session_ids(&second), ["b"]);
    let third = take_direct_iteration_batch(&mut queue, 3, 4);
    assert_eq!(session_ids(&third), ["c", "z"]);
    assert!(queue.is_empty());
}

fn session_ids(batch: &[DirectIteration]) -> Vec<&str> {
    batch
        .iter()
        .map(|request| request.session_id.as_str())
        .collect()
}

#[test]
fn direct_iteration_batch_defers_duplicate_sessions() {
    let mut queue = VecDeque::from([
        direct_iteration("same", 1),
        direct_iteration("same", 1),
        direct_iteration("other", 1),
    ]);

    let batch = take_direct_iteration_batch(&mut queue, 3, 8);

    assert_eq!(
        batch
            .iter()
            .map(|request| request.session_id.as_str())
            .collect::<Vec<_>>(),
        ["other", "same"]
    );
    assert_eq!(queue.len(), 1);
    assert_eq!(queue.front().unwrap().session_id, "same");
}

#[test]
fn direct_prefill_wave_never_exceeds_configured_native_batch_tokens() {
    let mut queue = (0..8)
        .map(|index| direct_iteration(&format!("session-{index}"), 128))
        .collect::<VecDeque<_>>();

    let batch = take_direct_iteration_batch(&mut queue, 8, 512);

    assert_eq!(batch.len(), 4);
    assert_eq!(
        batch
            .iter()
            .map(|request| request.token_ids.len())
            .sum::<usize>(),
        512
    );
    assert_eq!(queue.len(), 4);
}

#[test]
fn direct_wave_packs_decode_rows_queued_behind_an_oversized_prefill_chunk() {
    let mut queue = VecDeque::from([
        direct_iteration("prefill-a", 384),
        direct_iteration("prefill-b", 256),
        direct_iteration("decode-a", 1),
        direct_iteration("decode-b", 1),
        direct_iteration("late-arrival", 1),
    ]);

    let batch = take_direct_iteration_batch(&mut queue, 4, 512);

    assert_eq!(
        batch
            .iter()
            .map(|request| request.session_id.as_str())
            .collect::<Vec<_>>(),
        ["decode-a", "decode-b", "late-arrival", "prefill-a"]
    );
    assert_eq!(queue.len(), 1);
    assert_eq!(queue.front().unwrap().session_id, "prefill-b");
}

#[test]
fn deferred_prefill_chunk_stays_ahead_of_requests_it_did_not_reach() {
    let mut queue = VecDeque::from([
        direct_iteration("prefill-a", 384),
        direct_iteration("prefill-b", 256),
        direct_iteration("decode-a", 1),
        direct_iteration("decode-b", 1),
    ]);

    let batch = take_direct_iteration_batch(&mut queue, 2, 512);

    assert_eq!(batch.len(), 2);
    assert_eq!(
        queue
            .iter()
            .map(|request| request.session_id.as_str())
            .collect::<Vec<_>>(),
        ["prefill-b", "decode-b"]
    );
}

#[test]
fn deferred_prefill_chunk_is_not_overtaken_by_its_own_session() {
    let mut queue = VecDeque::from([
        direct_iteration("prefill-a", 384),
        direct_iteration("same", 256),
        direct_iteration("same", 1),
        direct_iteration("decode-a", 1),
    ]);

    let batch = take_direct_iteration_batch(&mut queue, 4, 512);

    assert_eq!(
        batch
            .iter()
            .map(|request| request.session_id.as_str())
            .collect::<Vec<_>>(),
        ["decode-a", "prefill-a"]
    );
    assert_eq!(
        queue
            .iter()
            .map(|request| request.token_ids.len())
            .collect::<Vec<_>>(),
        [256, 1]
    );
}
