use super::*;
use crate::kv_integration::RadixResidentEntry;

#[test]
fn exact_state_retention_floor_survives_a_hard_cap_blind_to_recurrent_state() {
    // Regression: the byte limits are derived from attention KV metadata,
    // so hybrid snapshots (recurrent state the estimate cannot see) can
    // push the retained working set past the hard cap. The retention floor
    // must keep every concurrent session's entry; collapsing to a single
    // entry made sessions evict each other on every request.
    let radix = Mutex::new(UnifiedRadixCache::new());
    let blobs = Mutex::new(CacheBlobStore::new(4));
    let budget = StorageBudget::new();
    // Each snapshot is 32 bytes; four of them (128) exceed the hard cap
    // of 80 even though each individually fits the 8x headroom.
    let oversized_for_attention_estimate = limits(10, 80);
    for index in 0..4 {
        store_exact_radix_record(
            &radix,
            &blobs,
            512,
            oversized_for_attention_estimate,
            None,
            None,
            pending(
                &format!("session-{index}"),
                &[1, index + 2],
                &distinct_payload(index as u8, 32),
                &budget,
            ),
        )
        .unwrap();
    }
    // Distinct blocks, so the retained set really is past the hard cap and
    // the pre-floor policy would have evicted down to one entry.
    assert!(blobs.lock().unwrap().physical_bytes() > 80);
    let mut radix = radix.lock().unwrap();
    assert_eq!(radix.stats().recurrent_entries, 4);
    for index in 0..4 {
        assert!(
            radix.lookup_recurrent("model", &[1, index + 2]).is_some(),
            "session {index} must retain its exact-state entry"
        );
    }
}

/// Payload bytes that dedupe into distinct 4-byte blocks per `seed`.
fn distinct_payload(seed: u8, len: usize) -> Vec<u8> {
    (0..len)
        .map(|offset| seed.wrapping_mul(64).wrapping_add(offset as u8))
        .collect()
}

fn store_sessions(
    radix: &Mutex<UnifiedRadixCache<RadixResidentEntry, RadixExactEntry>>,
    blobs: &Mutex<CacheBlobStore>,
    limits: ExactStateByteLimits,
    sessions: &[(&str, &[i32], u8)],
) {
    let budget = StorageBudget::new();
    for (page, tokens, seed) in sessions {
        store_exact_radix_record(
            radix,
            blobs,
            512,
            limits,
            None,
            None,
            pending(page, tokens, &distinct_payload(*seed, 32), &budget),
        )
        .unwrap();
    }
}

#[test]
fn explicit_exact_budget_stays_a_strict_ceiling_past_the_retention_floor() {
    let radix = Mutex::new(UnifiedRadixCache::new());
    let blobs = Mutex::new(CacheBlobStore::new(4));
    let explicit_ceiling = ExactStateByteLimits {
        soft_bytes: 10,
        hard_bytes: 40,
        hard_is_explicit: true,
    };
    store_sessions(
        &radix,
        &blobs,
        explicit_ceiling,
        &[
            ("a", &[1, 2], 0),
            ("b", &[1, 3], 1),
            ("c", &[1, 4], 2),
            ("d", &[1, 5], 3),
        ],
    );
    // One indivisible 32-byte snapshot fits the ceiling; a second does not.
    assert_eq!(radix.lock().unwrap().stats().recurrent_entries, 1);
    assert!(blobs.lock().unwrap().physical_bytes() <= 40);
}

#[test]
fn explicit_exact_budget_evicts_a_continuation_when_its_parent_is_leased() {
    let radix = Mutex::new(UnifiedRadixCache::new());
    let blobs = Mutex::new(CacheBlobStore::new(4));
    let explicit_ceiling = ExactStateByteLimits {
        soft_bytes: 10,
        hard_bytes: 40,
        hard_is_explicit: true,
    };
    let checkpoint = [1, 2];
    let continuation = [1, 2, 3];
    store_sessions(&radix, &blobs, explicit_ceiling, &[("a", &checkpoint, 0)]);
    radix
        .lock()
        .unwrap()
        .acquire_recurrent("model", &checkpoint)
        .unwrap();
    store_sessions(
        &radix,
        &blobs,
        explicit_ceiling,
        &[("a-continued", &continuation, 1)],
    );

    let mut radix = radix.lock().unwrap();
    assert_eq!(radix.stats().recurrent_entries, 1);
    assert!(radix.recurrent_exact("model", &continuation).is_none());
    assert!(radix.release_recurrent("model", &checkpoint));
    assert_eq!(blobs.lock().unwrap().physical_bytes(), 32);
}

#[test]
fn unbounded_byte_limits_keep_every_entry_the_entry_limit_allows() {
    let radix = Mutex::new(UnifiedRadixCache::new());
    let blobs = Mutex::new(CacheBlobStore::new(4));
    store_sessions(
        &radix,
        &blobs,
        limits(0, 0),
        &[
            ("a", &[1, 2], 0),
            ("b", &[1, 3], 1),
            ("c", &[1, 4], 2),
            ("d", &[1, 5], 3),
            ("e", &[1, 6], 4),
        ],
    );
    assert_eq!(radix.lock().unwrap().stats().recurrent_entries, 5);
}

#[test]
fn retention_floor_keeps_each_sessions_latest_continuation() {
    let radix = Mutex::new(UnifiedRadixCache::new());
    let blobs = Mutex::new(CacheBlobStore::new(4));
    // Session A stores a prefill and a post-decode checkpoint between the
    // other sessions' prefills. Five snapshots exceed the derived hard
    // cap; the superseded prefill must go, not another session's only one.
    store_sessions(
        &radix,
        &blobs,
        limits(10, 80),
        &[
            ("a-prefill", &[1, 2], 0),
            ("b", &[1, 3], 1),
            ("a-decode", &[1, 2, 9, 9], 2),
            ("c", &[1, 4], 3),
            ("d", &[1, 5], 4),
        ],
    );
    let mut radix = radix.lock().unwrap();
    assert!(radix.lookup_recurrent("model", &[1, 2]).is_none());
    for tokens in [&[1, 2, 9, 9][..], &[1, 3], &[1, 4], &[1, 5]] {
        assert!(
            radix.lookup_recurrent("model", tokens).is_some(),
            "latest continuation {tokens:?} must survive"
        );
    }
}

#[test]
fn entry_limit_prefers_superseded_checkpoints_to_other_conversations() {
    let radix = Mutex::new(UnifiedRadixCache::new());
    let blobs = Mutex::new(CacheBlobStore::new(4));
    let budget = StorageBudget::new();
    let limits = limits(0, 0);
    let store = |page, tokens, seed| {
        store_exact_radix_record(
            &radix,
            &blobs,
            4,
            limits,
            None,
            None,
            pending(page, tokens, &distinct_payload(seed, 32), &budget),
        )
        .unwrap();
    };
    store("a-prefill", &[1, 2], 0);
    store("b", &[1, 3], 1);
    store("c", &[1, 4], 2);
    store("d", &[1, 5], 3);
    radix
        .lock()
        .unwrap()
        .lookup_recurrent("model", &[1, 2])
        .unwrap();
    store("a-continued", &[1, 2, 9], 4);

    let mut radix = radix.lock().unwrap();
    assert!(radix.recurrent_exact("model", &[1, 2]).is_none());
    for tokens in [&[1, 2, 9][..], &[1, 3], &[1, 4], &[1, 5]] {
        assert!(radix.recurrent_exact("model", tokens).is_some());
    }
}
