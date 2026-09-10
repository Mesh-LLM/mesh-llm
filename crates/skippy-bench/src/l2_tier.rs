//! `l2-tier` benchmark: cold L3 fill versus warm L2 lookup on identical
//! packed entries (#1651).
//!
//! Builds a temporary L3 store, spills a synthetic multi-turn prompt at a
//! recorded prefix length, then measures two restore paths in-process:
//!
//! - **L3 cold fill**: `L3Tier::fill_longest` — index probe + segment
//!   assembly + digest verification from disk.
//! - **L2 warm lookup**: `L2Tier::get` + `to_payload` — the entry was
//!   admitted from an identical verified L3 fill, so the lookup is a
//!   digest-keyed handle assembly (no re-hash: admission verified the
//!   wire once; segments are immutable afterward).
//!
//! Admission hashing (the one-time wire BLAKE3) is measured separately
//! and reported as its own metric, never inside the timed lookup.
//!
//! Both paths produce payloads with identical bytes; the harness asserts
//! that before timing so a correctness regression cannot hide behind a
//! speedup. Output goes to stdout as JSON lines.
use std::time::Instant;

use anyhow::{Context, Result};

use crate::cli::L2TierArgs;
use skippy_cache::{
    ExactStatePayload, ExactStatePayloadMirror, L2Origin, L2Tier, l2_cache_key, l3_prefix_key,
};

fn percentile(samples_ns: &mut [u128], pct: f64) -> f64 {
    samples_ns.sort_unstable();
    let index = ((pct / 100.0) * (samples_ns.len() as f64 - 1.0)).round() as usize;
    samples_ns[index.min(samples_ns.len() - 1)] as f64
}

pub fn l2_tier(args: L2TierArgs) -> Result<()> {
    // Validation: reject degenerate configurations up front instead of
    // dividing by zero or allocating nothing below.
    if args.pairs == 0 {
        anyhow::bail!("--pairs must be at least 1");
    }
    if args.tokens == 0 {
        anyhow::bail!("--tokens must be at least 1");
    }
    if args.kv_bytes_per_token == 0 {
        anyhow::bail!("--kv-bytes-per-token must be at least 1");
    }

    let namespace = "bench-namespace";
    let state_identity = args.model_identity.clone();
    let token_ids: Vec<i32> = (0..args.tokens).map(|i| (i % 128_000) as i32).collect();

    // Deterministic synthetic KV payload: content matters only for digests,
    // size matters for timing.
    let payload_len = args
        .tokens
        .checked_mul(args.kv_bytes_per_token)
        .context("--tokens * --kv-bytes-per-token overflows")?;
    let payload_bytes: Vec<u8> = (0..payload_len).map(|i| (i % 251) as u8).collect();
    let payload = ExactStatePayload::full_state(payload_bytes);

    let _ = std::fs::remove_dir_all(&args.store_root);
    let tier = skippy_cache::L3Tier::open(
        args.store_root.clone(),
        (payload_len as u64) * 8,
        state_identity.clone(),
        64 * 1024,
    )
    .context("failed to open bench L3 tier")?;

    // Spill once: this is the population path, not the measured path.
    let manifest_key = tier
        .spill(namespace, &token_ids, &payload, None, None)
        .context("bench spill failed")?;
    let _ = manifest_key;

    // Locate once to learn the recorded prefix key/digest used by both paths.
    let location = tier
        .locate_longest(namespace, &token_ids, 8)
        .context("bench locate failed")?
        .context("bench spill was not locatable")?;
    let manifest = tier.store().load_manifest(&location.manifest_key)?;
    let payload_digest = manifest.payload_digest.clone();
    let recorded_tokens = manifest.token_count;

    let l2_budget_bytes = args
        .l2_budget_mib
        .map(|mib| {
            mib.checked_mul(1024 * 1024)
                .context("--l2-budget-mib overflows")
        })
        .transpose()?
        .unwrap_or(payload_len as u64 * 4);
    let l2 = L2Tier::new(l2_budget_bytes);

    let cache_key = l2_cache_key(&args.model_identity, &state_identity, namespace, &token_ids);

    // Warmup: one L3 fill, then admit it into L2 from the verified wire.
    // The wire check inside `admit` is the one-time admission hash; it is
    // timed separately below.
    let warm_fill = tier
        .fill_longest(namespace, &token_ids, 8)
        .context("bench warmup L3 fill failed")?
        .context("bench warmup L3 fill missed")?;
    let (warm_wire, _) = warm_fill.payload.full_state_bytes_timed().context("wire")?;
    let admission_started = Instant::now();
    l2.admit(
        cache_key.clone(),
        warm_fill.token_count,
        payload_digest.clone(),
        &warm_wire,
        ExactStatePayloadMirror::from_manifest(&manifest)
            .map_err(|refusal| anyhow::anyhow!(refusal.reason()))?,
        L2Origin::FromL3,
    )
    .map_err(|refusal| anyhow::anyhow!("bench warmup L2 admit refused: {}", refusal.reason()))?;
    let admission_hash_ns = admission_started.elapsed().as_nanos();

    let mut l3_samples: Vec<u128> = Vec::with_capacity(args.pairs);
    let mut l2_samples: Vec<u128> = Vec::with_capacity(args.pairs);

    for pair in 0..args.pairs {
        // Cold-ish L3 fill: the OS page cache will help after warmup, which
        // matches the production comparison — both paths run on the same
        // machine state, the delta is the tier delta.
        let start = Instant::now();
        let fill = tier
            .fill_longest(namespace, &token_ids, 8)
            .context("bench L3 fill failed")?
            .context("bench L3 fill missed")?;
        let l3_ns = start.elapsed().as_nanos();

        let start = Instant::now();
        let hit = l2.get(&cache_key);
        let l2_payload = hit.as_ref().map(|hit| hit.to_payload());
        let l2_ns = start.elapsed().as_nanos();

        // Correctness gate, outside the timer: L2 must return
        // byte-identical state to the L3 fill, or the speedup is
        // meaningless.
        let hit = hit.context("bench L2 lookup missed")?;
        let l2_payload = l2_payload.context("bench L2 payload missing")?;
        anyhow::ensure!(hit.token_count == fill.token_count);
        anyhow::ensure!(hit.payload_digest == payload_digest);
        let (l3_bytes, _) = fill.payload.full_state_bytes_timed().context("l3 bytes")?;
        let (l2_bytes, _) = l2_payload.full_state_bytes_timed().context("l2 bytes")?;
        anyhow::ensure!(
            l3_bytes == l2_bytes,
            "pair {pair}: L2 payload diverged from L3 fill"
        );

        l3_samples.push(l3_ns);
        l2_samples.push(l2_ns);
    }

    let stats = l2.stats();
    let mut l3_sorted = l3_samples.clone();
    let mut l2_sorted = l2_samples.clone();
    let summary = serde_json::json!({
        "bench": "l2-tier",
        "pairs": args.pairs,
        "tokens": recorded_tokens,
        "payload_bytes": payload_len,
        "l2_budget_bytes": l2_budget_bytes,
        "model_identity": args.model_identity,
        "l3_fill_ns": {
            "p50": percentile(&mut l3_sorted, 50.0),
            "p99": percentile(&mut l3_sorted, 99.0),
        },
        "l2_lookup_assembly_handle_ns": {
            "p50": percentile(&mut l2_sorted, 50.0),
            "p99": percentile(&mut l2_sorted, 99.0),
        },
        "l2_admission_hash_ns_one_time": admission_hash_ns,
        "speedup_p50": percentile(&mut l3_sorted, 50.0) / percentile(&mut l2_sorted, 50.0).max(1.0),
        "l2_stats": {
            "hits": stats.hits,
            "misses": stats.misses,
            "evictions": stats.evictions,
            "bytes": stats.bytes,
            "segments": stats.segments,
            "shared_bytes_admitted": stats.shared_bytes_admitted,
        },
        "l3_prefix_key": l3_prefix_key(namespace, &token_ids),
    });
    println!("{summary}");

    if !args.keep_store {
        let _ = std::fs::remove_dir_all(&args.store_root);
    }
    Ok(())
}
