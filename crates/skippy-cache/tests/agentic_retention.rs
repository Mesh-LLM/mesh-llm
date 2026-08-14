//! End-to-end retention behaviour for the agentic workload the feature targets.
//!
//! These tests compose the pieces that individually look fine but only deliver
//! value together, which is the trap the plan calls out: the candidate ladder
//! decides whether a reusable page exists at a shareable length, and the disk
//! tier decides whether it survived the gap. Either alone yields nothing for
//! the motivating scenario.

use skippy_cache::{
    ExactStateCache, ExactStatePayload, ExactStatePayloadKind, PrefixCandidatePolicy,
    PrefixDiskTier, PrefixMissReason, prefix_identity,
};
use skippy_protocol::{FlashAttentionType, LoadMode, StageConfig};

/// The shipped agentic cache shape: 256-token floor, 128-token stride.
fn policy(record_limit: u64) -> PrefixCandidatePolicy {
    PrefixCandidatePolicy {
        min_tokens: 256,
        stride_tokens: 128,
        record_limit,
        page_size_tokens: 256,
        max_resident_tokens_hint: 0,
    }
}

fn stage_config() -> StageConfig {
    StageConfig {
        run_id: "run".to_string(),
        topology_id: "topology".to_string(),
        model_id: "org/model:Q4_K_M".to_string(),
        package_ref: None,
        manifest_sha256: None,
        source_model_path: None,
        source_model_sha256: None,
        source_model_bytes: None,
        materialized_path: None,
        materialized_pinned: false,
        model_path: None,
        projector_path: None,
        stage_id: "stage-0".to_string(),
        stage_index: 0,
        layer_start: 0,
        layer_end: 32,
        ctx_size: 131_072,
        lane_count: 4,
        n_batch: None,
        n_ubatch: None,
        n_gpu_layers: 0,
        mmap: None,
        mlock: false,
        cache_type_k: "f16".to_string(),
        cache_type_v: "f16".to_string(),
        flash_attn_type: FlashAttentionType::Auto,
        filter_tensors_on_load: false,
        selected_device: None,
        kv_cache: None,
        native_mtp_enabled: true,
        load_mode: LoadMode::RuntimeSlice,
        bind_addr: "127.0.0.1:0".to_string(),
        upstream: None,
        downstream: None,
    }
}

/// A realistic agent prompt: a stable system prompt plus tool schemas, then a
/// request-specific tail.
fn agent_prompt(shared_prefix_tokens: usize, tail_seed: i32, tail_tokens: usize) -> Vec<i32> {
    let mut tokens: Vec<i32> = (0..shared_prefix_tokens as i32).collect();
    tokens.extend((0..tail_tokens as i32).map(|index| 900_000 + tail_seed * 10_000 + index));
    tokens
}

/// Two different agent sessions sharing a system prompt must land on the same
/// `page_id` for the shared region. This is the property everything else
/// depends on, and it holds because identity contains no `session_id`.
#[test]
fn distinct_sessions_agree_on_the_shared_prefix_page_id() {
    let config = stage_config();
    let shared = 2048;
    let session_a = agent_prompt(shared, 1, 700);
    let session_b = agent_prompt(shared, 2, 1500);

    let a = prefix_identity(&config, 0, &session_a[..shared]);
    let b = prefix_identity(&config, 0, &session_b[..shared]);

    assert_eq!(a.page_id, b.page_id);
    // ...while the full prompts differ, as they must.
    let a_full = prefix_identity(&config, 0, &session_a);
    let b_full = prefix_identity(&config, 0, &session_b);
    assert_ne!(a_full.page_id, b_full.page_id);
}

/// The full motivating scenario. Session A serves a large prompt. Its prefix
/// is evicted from RAM. Session B arrives later with the same system prompt
/// and a different tail, and must still get a warm prefix.
#[test]
fn shared_prefix_survives_eviction_and_serves_a_later_session() {
    let dir = tempfile::tempdir().unwrap();
    let config = stage_config();
    let shared_tokens = 2048;

    let disk = PrefixDiskTier::open(dir.path(), 64 << 20).unwrap();
    // A single-entry RAM cache guarantees the prefix is evicted.
    let mut cache = ExactStateCache::<()>::new(1, 0).with_disk_tier(disk);

    // --- Session A ---
    let session_a = agent_prompt(shared_tokens, 1, 700);
    let recorded = policy(6).record_candidate_token_counts(session_a.len() as u64);
    // The ladder must reach the shared region for any of this to work.
    let shared_candidate = recorded
        .iter()
        .copied()
        .find(|candidate| *candidate <= shared_tokens as u64)
        .expect("ladder must record a candidate inside the shared prefix");

    let shared_id = prefix_identity(&config, 0, &session_a[..shared_candidate as usize]);
    let kv_bytes = vec![0xABu8; 64 * 1024];
    cache.record(
        shared_id.page_id.clone(),
        shared_candidate,
        ExactStatePayload::full_state(kv_bytes.clone()),
        (),
    );

    // Evict it by recording an unrelated prompt.
    let other = prefix_identity(&config, 0, &agent_prompt(512, 9, 100));
    cache.record(
        other.page_id,
        612,
        ExactStatePayload::full_state(vec![1u8; 1024]),
        (),
    );
    assert!(
        cache.lookup(&shared_id.page_id).is_none(),
        "prefix should have left RAM"
    );

    // --- Session B, later, same system prompt and a different tail ---
    let session_b = agent_prompt(shared_tokens, 2, 1500);
    let probed = policy(6).candidate_token_counts(session_b.len() as u64);
    assert!(
        probed.contains(&shared_candidate),
        "lookup must probe the recorded shared length"
    );

    let lookup_id = prefix_identity(&config, 0, &session_b[..shared_candidate as usize]);
    assert_eq!(
        lookup_id.page_id, shared_id.page_id,
        "sessions must agree on the shared page id"
    );

    let restored = cache
        .lookup_with_disk(&lookup_id.page_id, ExactStatePayloadKind::FullState, || ())
        .unwrap()
        .expect("shared prefix should be served from the disk tier");

    assert!(restored.from_disk);
    assert_eq!(restored.token_count, shared_candidate);
    assert_eq!(
        restored
            .payload
            .full_state_bytes_timed()
            .unwrap()
            .0
            .as_ref(),
        &kv_bytes[..]
    );
    // The reuse must be substantial to be worth anything.
    assert!(
        restored.token_count >= 1024,
        "reused only {} tokens",
        restored.token_count
    );
}

/// Restarting the process must not throw away retention. This is the
/// difference between a cache and a warm start.
#[test]
fn retention_survives_a_process_restart() {
    let dir = tempfile::tempdir().unwrap();
    let config = stage_config();
    let tokens = agent_prompt(2048, 1, 0);
    let identity = prefix_identity(&config, 0, &tokens);
    let payload = vec![0x5Au8; 128 * 1024];

    {
        let disk = PrefixDiskTier::open(dir.path(), 64 << 20).unwrap();
        let mut cache = ExactStateCache::<()>::new(1, 0).with_disk_tier(disk);
        cache.record(
            identity.page_id.clone(),
            2048,
            ExactStatePayload::full_state(payload.clone()),
            (),
        );
        // Evict so it is demoted to disk.
        cache.record(
            "unrelated".to_string(),
            256,
            ExactStatePayload::full_state(vec![0u8; 64]),
            (),
        );
    }

    // New process, same cache directory and the same stage configuration.
    let disk = PrefixDiskTier::open(dir.path(), 64 << 20).unwrap();
    let mut cache = ExactStateCache::<()>::new(1, 0).with_disk_tier(disk);
    let restored = cache
        .lookup_with_disk(&identity.page_id, ExactStatePayloadKind::FullState, || ())
        .unwrap()
        .expect("prefix should survive a restart");

    assert_eq!(
        restored
            .payload
            .full_state_bytes_timed()
            .unwrap()
            .0
            .as_ref(),
        &payload[..]
    );
}

/// A page written under one KV dtype must never be served to a stage running
/// another. Before identity covered `cache_type_k`/`_v` this collided and the
/// q8_0 bytes would have been imported as f16 — silent numerical corruption,
/// which is precisely the failure a persistent tier makes reachable.
#[test]
fn a_page_written_under_a_different_kv_dtype_is_never_served() {
    let dir = tempfile::tempdir().unwrap();
    let quality = stage_config();
    let saver = StageConfig {
        cache_type_k: "q8_0".to_string(),
        cache_type_v: "q8_0".to_string(),
        ..stage_config()
    };
    let tokens = agent_prompt(2048, 1, 0);

    let disk = PrefixDiskTier::open(dir.path(), 64 << 20).unwrap();
    let mut cache = ExactStateCache::<()>::new(1, 0).with_disk_tier(disk);

    let quality_id = prefix_identity(&quality, 0, &tokens);
    cache.record(
        quality_id.page_id.clone(),
        2048,
        ExactStatePayload::full_state(vec![0xF1; 4096]),
        (),
    );
    cache.record(
        "evictor".to_string(),
        256,
        ExactStatePayload::full_state(vec![0u8; 64]),
        (),
    );

    // The saver-configured stage computes a different page id and misses.
    let saver_id = prefix_identity(&saver, 0, &tokens);
    assert_ne!(saver_id.page_id, quality_id.page_id);
    assert!(
        cache
            .lookup_with_disk(&saver_id.page_id, ExactStatePayloadKind::FullState, || ())
            .unwrap()
            .is_none(),
        "a q8_0 stage must not read an f16 page"
    );
}

/// Miss classification must tell an operator whether a bigger/slower tier
/// would have helped. That is the decision this whole feature turns on.
#[test]
fn miss_reasons_report_whether_retention_would_have_helped() {
    let config = stage_config();
    let mut cache = ExactStateCache::<()>::new(1, 0);

    let held = prefix_identity(&config, 0, &agent_prompt(2048, 1, 0));
    cache.record(
        held.page_id.clone(),
        2048,
        ExactStatePayload::full_state(vec![1u8; 1024]),
        (),
    );
    // Evict it.
    cache.record(
        "evictor".to_string(),
        256,
        ExactStatePayload::full_state(vec![0u8; 64]),
        (),
    );

    // One miss on a prefix we held, two on prefixes we never saw. The unseen
    // prompts need distinct tails, or they are literally the same tokens and
    // therefore the same page.
    cache.lookup(&held.page_id);
    cache.lookup(&prefix_identity(&config, 0, &agent_prompt(2048, 7, 64)).page_id);
    cache.lookup(&prefix_identity(&config, 0, &agent_prompt(2048, 8, 64)).page_id);

    let stats = cache.miss_stats();
    assert_eq!(stats.misses_for(PrefixMissReason::EvictedRecently), 1);
    assert_eq!(stats.misses_for(PrefixMissReason::NeverSeen), 2);
    // A third of misses are recoverable by retaining longer.
    assert!((stats.recoverable_miss_ratio() - 1.0 / 3.0).abs() < 1e-9);
}

/// Split serving caches per stage, so two stages of one model must not share
/// or overwrite each other's pages.
#[test]
fn split_stages_keep_independent_pages_for_the_same_tokens() {
    let stage0 = StageConfig {
        stage_id: "stage-0".to_string(),
        stage_index: 0,
        layer_start: 0,
        layer_end: 20,
        ..stage_config()
    };
    let stage1 = StageConfig {
        stage_id: "stage-1".to_string(),
        stage_index: 1,
        layer_start: 20,
        layer_end: 40,
        ..stage_config()
    };
    let tokens = agent_prompt(2048, 1, 0);

    assert_ne!(
        prefix_identity(&stage0, 0, &tokens).page_id,
        prefix_identity(&stage1, 0, &tokens).page_id
    );
}

/// Regression test for a gap found while exercising a live dense model.
///
/// KV page bytes are meaningless without their descriptor (row strides,
/// element types, layer range). An earlier revision kept the descriptor only
/// in RAM, so after a restart the archived bytes survived and were silently
/// unusable — the disk tier looked populated while never serving a hit. The
/// descriptor must round-trip with the payload.
#[test]
fn archived_kv_page_metadata_survives_a_restart() {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    struct PageDesc {
        k_row_bytes: u32,
        v_row_bytes: u32,
        layer_start: i32,
        layer_end: i32,
    }

    let dir = tempfile::tempdir().unwrap();
    let desc = PageDesc {
        k_row_bytes: 512,
        v_row_bytes: 512,
        layer_start: 0,
        layer_end: 32,
    };
    let kv_bytes = vec![0x3Cu8; 32 * 1024];

    {
        let disk = PrefixDiskTier::open(dir.path(), 64 << 20).unwrap();
        let mut cache = ExactStateCache::<PageDesc>::new(4, 0).with_disk_tier(disk);
        assert!(cache.store_on_disk(
            "page-a",
            2048,
            ExactStatePayloadKind::KvRecurrent,
            &[&kv_bytes, &[]],
            desc.clone(),
        ));
    }

    // A fresh process has no in-RAM descriptor and must recover it from disk.
    let disk = PrefixDiskTier::open(dir.path(), 64 << 20).unwrap();
    let mut cache = ExactStateCache::<PageDesc>::new(4, 0).with_disk_tier(disk);
    let restored = cache
        .lookup_disk_only("page-a", ExactStatePayloadKind::KvRecurrent)
        .unwrap()
        .expect("archived KV page should be importable after a restart");

    assert_eq!(restored.extra, desc, "descriptor must round-trip");
    assert_eq!(
        restored.payload.kv_bytes().unwrap().unwrap().as_ref(),
        &kv_bytes[..]
    );
}
