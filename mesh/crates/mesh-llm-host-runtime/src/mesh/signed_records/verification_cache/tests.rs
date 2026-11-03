use super::*;
use ed25519_dalek::SigningKey;
use mesh_llm_membership::signed_record::{
    CACHE_AFFINITY_RECORD, HeldRecords, NODE_RECORD, RecordAdmission, RecordSkip,
};
use std::sync::atomic::Ordering;

fn signed(kind: RecordKind, seq: u64) -> VerifiedRecord {
    VerifiedRecord::sign(
        kind,
        &SigningKey::from_bytes(&[1; 32]),
        seq,
        1_000_000,
        b"body",
    )
}

fn verify(cache: &VerificationCache, kind: RecordKind, record: &VerifiedRecord) -> VerifiedRecord {
    cache
        .verify(kind, record.signed_bytes(), record.signature())
        .unwrap()
}

#[test]
fn repeated_records_of_both_kinds_reuse_signature_verification() {
    let cache = VerificationCache::default();
    for kind in [NODE_RECORD, CACHE_AFFINITY_RECORD] {
        let record = signed(kind, 1);
        for _ in 0..10 {
            assert_eq!(verify(&cache, kind, &record), record);
        }
    }
    assert_eq!(cache.verifications.load(Ordering::Relaxed), 2);
}

#[test]
fn hits_require_exact_signed_bytes_signature_and_kind() {
    let cache = VerificationCache::default();
    let record = signed(NODE_RECORD, 1);
    verify(&cache, NODE_RECORD, &record);
    let mut bytes = record.signed_bytes().to_vec();
    *bytes.last_mut().unwrap() ^= 1;
    assert_eq!(
        cache.verify(NODE_RECORD, &bytes, record.signature()),
        Err(RecordError::BadSignature)
    );
    let mut signature = *record.signature();
    signature[0] ^= 1;
    assert_eq!(
        cache.verify(NODE_RECORD, record.signed_bytes(), &signature),
        Err(RecordError::BadSignature)
    );
    assert_eq!(
        cache.verify(
            CACHE_AFFINITY_RECORD,
            record.signed_bytes(),
            record.signature()
        ),
        Err(RecordError::BadSignature)
    );
    let limited = RecordKind {
        max_signed_bytes: 1,
        ..NODE_RECORD
    };
    assert_eq!(
        cache.verify(limited, record.signed_bytes(), record.signature()),
        Err(RecordError::Oversize)
    );
    assert_eq!(
        cache.verify(NODE_RECORD, record.signed_bytes(), &[0; 63]),
        Err(RecordError::InvalidSignatureLength)
    );
    assert_eq!(verify(&cache, NODE_RECORD, &record), record);
    assert_eq!(cache.entries.lock().unwrap().records.len(), 1);
    assert_eq!(cache.verifications.load(Ordering::Relaxed), 5);
}

#[test]
fn failed_signatures_are_never_cached() {
    let cache = VerificationCache::default();
    let record = signed(NODE_RECORD, 1);
    for _ in 0..2 {
        assert!(
            cache
                .verify(NODE_RECORD, record.signed_bytes(), &[0; 64])
                .is_err()
        );
    }
    assert_eq!(cache.verifications.load(Ordering::Relaxed), 2);
    let entries = cache.entries.lock().unwrap();
    assert!(entries.records.is_empty());
    assert!(entries.insertion_order.is_empty());
    assert_eq!(entries.signed_bytes, 0);
}

#[test]
fn record_and_byte_limits_evict_oldest_entries() {
    let first = signed(NODE_RECORD, 1);
    let second = signed(NODE_RECORD, 2);
    for (max_entries, max_signed_bytes) in [(1, usize::MAX), (10, first.signed_bytes().len())] {
        let cache = VerificationCache {
            max_entries,
            max_signed_bytes,
            ..Default::default()
        };
        verify(&cache, NODE_RECORD, &first);
        verify(&cache, NODE_RECORD, &second);
        verify(&cache, NODE_RECORD, &second);
        assert_eq!(cache.verifications.load(Ordering::Relaxed), 2);
        {
            let entries = cache.entries.lock().unwrap();
            assert_eq!(entries.records.len(), 1);
            assert_eq!(entries.insertion_order.len(), 1);
            assert_eq!(entries.signed_bytes, second.signed_bytes().len());
        }
        verify(&cache, NODE_RECORD, &first);
        assert_eq!(cache.verifications.load(Ordering::Relaxed), 3);
    }
}

#[test]
fn oversized_entry_does_not_evict_a_retained_record() {
    let record = signed(NODE_RECORD, 1);
    let cache = VerificationCache {
        max_signed_bytes: record.signed_bytes().len(),
        ..Default::default()
    };
    verify(&cache, NODE_RECORD, &record);
    let large = VerifiedRecord::sign(
        NODE_RECORD,
        &SigningKey::from_bytes(&[1; 32]),
        2,
        1_000_000,
        b"larger body",
    );
    for _ in 0..2 {
        verify(&cache, NODE_RECORD, &large);
    }
    verify(&cache, NODE_RECORD, &record);
    assert_eq!(cache.verifications.load(Ordering::Relaxed), 3);
    assert_eq!(cache.entries.lock().unwrap().records.len(), 1);
}

#[test]
fn cached_records_still_obey_current_admission_time_and_sequence() {
    let local = iroh::SecretKey::from_bytes(&[2; 32]).public();
    let relay = iroh::SecretKey::from_bytes(&[3; 32]).public();
    for kind in [NODE_RECORD, CACHE_AFFINITY_RECORD] {
        let cache = VerificationCache::default();
        let old = signed(kind, 1);
        let newer = signed(kind, 2);
        let now = old.header().issued_at_unix_ms;
        let mut held = HeldRecords::new(kind);
        let admit = |held: &mut HeldRecords, record: &VerifiedRecord, now| {
            held.admit_verified(local, relay, verify(&cache, kind, record), (), now)
        };
        assert!(matches!(
            admit(&mut held, &old, now),
            RecordAdmission::Accepted { .. }
        ));
        assert!(matches!(
            admit(&mut held, &newer, now),
            RecordAdmission::Accepted { .. }
        ));
        assert!(matches!(
            admit(&mut held, &old, now),
            RecordAdmission::Skipped {
                reason: RecordSkip::Superseded,
                ..
            }
        ));
        assert!(matches!(
            admit(&mut held, &newer, now + kind.max_age_ms + 1),
            RecordAdmission::Skipped {
                reason: RecordSkip::Expired,
                ..
            }
        ));
        assert_eq!(cache.verifications.load(Ordering::Relaxed), 2);
        // A self-originating restart may roll back its sequence, even on a hit.
        assert!(matches!(
            held.admit_verified(
                local,
                old.endpoint_id(),
                verify(&cache, kind, &old),
                (),
                now
            ),
            RecordAdmission::Accepted { .. }
        ));
        // A clock moving backwards must re-evaluate even a cached self-record.
        assert!(matches!(
            held.admit_verified(local, old.endpoint_id(), verify(&cache, kind, &old), (), 0),
            RecordAdmission::Skipped {
                reason: RecordSkip::Expired,
                ..
            }
        ));
    }
}

#[test]
fn concurrent_misses_do_not_duplicate_cache_accounting() {
    let cache = VerificationCache::default();
    let record = signed(NODE_RECORD, 1);
    let barrier = std::sync::Barrier::new(4);
    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(|| {
                barrier.wait();
                verify(&cache, NODE_RECORD, &record);
            });
        }
    });
    let before = cache.verifications.load(Ordering::Relaxed);
    verify(&cache, NODE_RECORD, &record);
    assert_eq!(cache.verifications.load(Ordering::Relaxed), before);
    let entries = cache.entries.lock().unwrap();
    assert_eq!(entries.records.len(), 1);
    assert_eq!(entries.insertion_order.len(), 1);
    assert_eq!(entries.signed_bytes, record.signed_bytes().len());
}
