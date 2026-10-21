use super::*;
use crate::kv_proto::PageIdentity;
use skippy_protocol::{StageConfig, StageKvCacheConfig, StageKvCacheMode};
use skippy_runtime::ModelStateKind;

fn checkpoint(namespace: &str, tokens: &[i32], durable: bool) -> ResidentRecurrentCheckpoint {
    let config = StageConfig {
        kv_cache: Some(StageKvCacheConfig {
            mode: StageKvCacheMode::LookupRecord,
            exact_max_bytes: None,
            max_entries: 8,
            max_bytes: 0,
            l2_max_bytes: 0,
            codec: skippy_protocol::StageKvCacheCodec::Native,
            min_tokens: 1,
            shared_prefix_stride_tokens: 1,
            shared_prefix_record_limit: 0,
        }),
        ..Default::default()
    };
    ResidentRecurrentCheckpoint {
        identity: PrefillKvIdentity {
            identity: PageIdentity {
                token_count: tokens.len() as u64,
                ..Default::default()
            },
            page_id: format!("{namespace}-{tokens:?}"),
            namespace: namespace.into(),
            token_ids: tokens.to_vec(),
        },
        recurrent: vec![0; 8],
        cache: KvStageIntegration::from_test_state(&config, ModelStateKind::Recurrent)
            .unwrap()
            .unwrap(),
        policy: ResidentRecurrentPolicy {
            write_through_l3: durable,
            l3_cost: None,
        },
    }
}

fn lengths(set: &RecurrentCheckpoints) -> Vec<usize> {
    set.iter()
        .map(|entry| entry.identity.token_ids.len())
        .collect()
}

#[test]
fn bounds_preserve_shared_boundary_and_return_displaced_checkpoints() {
    for (limit, bytes) in [(2, 0), (4, 16)] {
        let mut set = RecurrentCheckpoints::default();
        assert!(
            set.insert(checkpoint("a", &[1], true), limit, bytes)
                .is_empty()
        );
        assert!(
            set.insert(checkpoint("a", &[1, 2], false), limit, bytes)
                .is_empty()
        );
        let evicted = set.insert(checkpoint("a", &[1, 2, 3], false), limit, bytes);
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0].identity.token_ids, [1, 2]);
        assert_eq!(lengths(&set), [1, 3]);
        assert_eq!(set.retained_bytes(), 16);
    }
}

#[test]
fn one_entry_limit_can_evict_the_shared_boundary_without_looping() {
    let mut set = RecurrentCheckpoints::default();
    set.insert(checkpoint("a", &[1], true), 1, 8);
    let displaced = set.insert(checkpoint("a", &[1, 2], false), 1, 8);
    assert_eq!(displaced.len(), 1);
    assert!(displaced[0].policy.write_through_l3);
    assert_eq!(lengths(&set), [2]);
}

#[test]
fn shortened_tail_boundary_survives_prefill_and_decode_captures_within_existing_limits() {
    let mut set = RecurrentCheckpoints::default();
    let prompt = (1..=192).collect::<Vec<_>>();
    // Include an older anchor to exercise eviction rather than just insertion.
    for (count, shared) in [(100, true), (128, true), (188, true), (192, false)] {
        set.insert(checkpoint("a", &prompt[..count], shared), 4, 32);
    }
    let mut generated = prompt;
    generated.extend(193..=199);
    set.insert(checkpoint("a", &generated, false), 4, 32);
    assert_eq!(lengths(&set), [100, 128, 188, 199]);
    assert_eq!(set.retained_bytes(), 32);

    // A shortened prompt can restore the nearby boundary; all later states
    // become invalid and must be removed from the lane.
    assert_eq!(set.restore(1).identity.token_ids.len(), 128);
    assert_eq!(lengths(&set), [100, 128]);
}

#[test]
fn replacement_preserves_durability_but_branch_and_namespace_changes_do_not() {
    let mut set = RecurrentCheckpoints::default();
    set.insert(checkpoint("a", &[1], true), 4, 0);
    set.insert(checkpoint("a", &[1], false), 4, 0);
    assert_eq!(set.len(), 1);
    assert!(set.iter().next().unwrap().policy.write_through_l3);
    set.insert(checkpoint("a", &[1, 2], false), 4, 0);
    set.insert(checkpoint("a", &[1, 3], false), 4, 0);
    assert_eq!(lengths(&set), [1, 2]);
    assert_eq!(set.iter().last().unwrap().identity.token_ids, [1, 3]);
    set.insert(checkpoint("b", &[1], false), 4, 0);
    assert_eq!(set.len(), 1);
    assert!(!set.iter().next().unwrap().policy.write_through_l3);
}

#[test]
fn restore_drops_invalid_suffixes_and_demotion_prioritizes_latest() {
    let mut set = RecurrentCheckpoints::default();
    for count in 1..=4 {
        let tokens = (1..=count).collect::<Vec<_>>();
        set.insert(checkpoint("a", &tokens, count == 2), 4, 0);
    }
    assert_eq!(set.restore(2).identity.token_ids, [1, 2, 3]);
    assert_eq!(lengths(&set), [1, 2, 3]);
    let demoted = set.take_for_demotion();
    assert_eq!(demoted.len(), 2);
    assert_eq!(demoted[0].identity.token_ids, [1, 2, 3]);
    assert_eq!(demoted[1].identity.token_ids, [1]);
    assert!(set.is_empty());
    assert_eq!(set.retained_bytes(), 0);
    assert!(set.take_for_demotion().is_empty());
    set.insert(checkpoint("a", &[1], false), 4, 0);
    assert_eq!(set.take_for_demotion().len(), 1);
}
