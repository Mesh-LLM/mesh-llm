//! Measure what an L3 restore costs, and how the cost moves with pack layout.
//!
//! Restore latency is bounded by physical reads and per-object work, not by
//! bytes. This profile builds a synthetic store at a realistic KV geometry and
//! compares two layouts of the same payload:
//!
//! - `single`: every segment published in one pack (one contiguous read)
//! - `turns`: the payload grown turn by turn, so each spill publishes its own
//!   pack and the final manifest references one pack per turn
//!
//! Both layouts are produced the way `L3Tier::spill` produces them:
//! content-addressed segments committed under a manifest, with reused
//! segments deduped rather than rewritten.
//!
//! ```text
//! cargo run -p skippy-cache --release --example l3_restore_profile -- \
//!     --mib 1024 --turns 32 --tokens-per-turn 300
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use skippy_cache::{
    GeometryBlock, GeometryKind, HandoffManifest, HandoffSegmentRef, HandoffSegmentStore,
    PayloadGeometry, SegmentCodecIdentity, segment_digest,
};

const MIB: u64 = 1024 * 1024;
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

struct Options {
    mib: u64,
    turns: u32,
    tokens_per_turn: u64,
    layers: u32,
    kv_heads: u32,
    head_dim: u32,
    dtype_bytes: u32,
    window_rows: u64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            mib: 1024,
            turns: 32,
            tokens_per_turn: 300,
            layers: 32,
            kv_heads: 8,
            head_dim: 128,
            dtype_bytes: 2,
            window_rows: 64,
        }
    }
}

#[derive(Clone, Copy)]
enum Layout {
    Single,
    Turns,
}

fn main() -> Result<()> {
    let options = Options::parse()?;
    let blocks = row_layout(&options);
    let bytes_per_token = blocks.iter().map(|block| block.stride).sum::<u64>();
    let total_tokens = (options.mib * MIB) / bytes_per_token.max(1);
    anyhow::ensure!(total_tokens > options.window_rows, "payload is too small");

    println!(
        "payload {:.3} GiB, {} tokens, {:.0} KiB/token, {} rows/window",
        (total_tokens * bytes_per_token) as f64 / GIB,
        total_tokens,
        bytes_per_token as f64 / 1024.0,
        options.window_rows,
    );

    hash_benchmark(options.mib.min(1024))?;
    for layout in [Layout::Single, Layout::Turns] {
        run_layout(&options, &blocks, total_tokens, layout)?;
    }
    Ok(())
}

/// The runtime's export order: every layer's K run, then every layer's V run,
/// one token-row per token.
fn row_layout(options: &Options) -> Vec<GeometryBlock> {
    let row =
        u64::from(options.kv_heads) * u64::from(options.head_dim) * u64::from(options.dtype_bytes);
    let mut blocks = Vec::with_capacity(options.layers as usize * 2);
    for layer in 0..options.layers {
        blocks.push(GeometryBlock {
            stride: row,
            kind: GeometryKind::Key,
            layer,
            column: 0,
        });
        blocks.push(GeometryBlock {
            stride: row,
            kind: GeometryKind::Value,
            layer,
            column: 0,
        });
    }
    blocks
}

/// One logical segment the geometry plan would cut for `tokens` rows.
struct SegmentPlan {
    block: usize,
    window_start: u64,
    offset: u64,
    bytes: usize,
}

fn segment_plan(blocks: &[GeometryBlock], tokens: u64, window_rows: u64) -> Vec<SegmentPlan> {
    let mut plan = Vec::new();
    let mut offset = 0u64;
    for (block_index, block) in blocks.iter().enumerate() {
        let mut row = 0u64;
        while row < tokens {
            let rows = window_rows.min(tokens - row);
            let bytes = rows.saturating_mul(block.stride);
            plan.push(SegmentPlan {
                block: block_index,
                window_start: row,
                offset,
                bytes: bytes as usize,
            });
            offset = offset.saturating_add(bytes);
            row += rows;
        }
    }
    plan
}

#[derive(Default)]
struct SegmentCache {
    segments: HashMap<(usize, u64, usize), Vec<u8>>,
}

impl SegmentCache {
    fn ensure(&mut self, plan: &SegmentPlan) {
        self.segments
            .entry((plan.block, plan.window_start, plan.bytes))
            .or_insert_with(|| {
                let mut bytes = vec![0u8; plan.bytes];
                fill_segment(&mut bytes, plan.block as u64, plan.window_start);
                bytes
            });
    }

    fn get(&self, plan: &SegmentPlan) -> &[u8] {
        self.segments
            .get(&(plan.block, plan.window_start, plan.bytes))
            .expect("segment generated before read")
    }
}

fn run_layout(
    options: &Options,
    blocks: &[GeometryBlock],
    total_tokens: u64,
    layout: Layout,
) -> Result<()> {
    let root = scratch_root(layout);
    let _ = std::fs::remove_dir_all(&root);
    let store = HandoffSegmentStore::open(&root, 0).context("open profile store")?;
    let mut cache = SegmentCache::default();

    let started = Instant::now();
    let mut turns = 0u32;
    let mut committed = 0u64;
    let manifest = match layout {
        Layout::Single => spill(&store, blocks, total_tokens, options, &mut cache)?,
        Layout::Turns => {
            let base = (total_tokens / 4).max(options.window_rows);
            let mut produced = base;
            let mut last = None;
            while produced < total_tokens && turns < options.turns {
                let before = store.managed_usage_bytes()?;
                last = Some(spill(&store, blocks, produced, options, &mut cache)?);
                committed = committed.saturating_add(store.managed_usage_bytes()? - before);
                produced = (produced + options.tokens_per_turn).min(total_tokens);
                turns += 1;
            }
            let manifest = last.expect("at least one turn");
            let before = store.managed_usage_bytes()?;
            let final_manifest = spill(&store, blocks, total_tokens, options, &mut cache)?;
            committed = committed.saturating_add(store.managed_usage_bytes()? - before);
            turns += 1;
            let _ = manifest;
            final_manifest
        }
    };
    let write_elapsed = started.elapsed();

    let loaded = store.load_manifest(&manifest.payload_digest)?;
    let restore_started = Instant::now();
    let (payload, stats) = store.assemble_traced(&loaded)?;
    let restore_elapsed = restore_started.elapsed();

    anyhow::ensure!(
        payload.len() == loaded.total_bytes as usize,
        "assembled payload has the wrong length"
    );
    anyhow::ensure!(
        segment_digest(&payload) == loaded.payload_digest,
        "assembled payload does not match its manifest"
    );

    let amplification = committed as f64 / manifest.total_bytes.max(1) as f64;
    println!(
        "\n== {} ({} spills)",
        match layout {
            Layout::Single => "single pack",
            Layout::Turns => "turn-by-turn packs",
        },
        turns
    );
    println!(
        "  write {:.2}s, committed {:.3} GiB, write amplification {amplification:.3}x",
        write_elapsed.as_secs_f64(),
        committed as f64 / GIB,
    );
    println!(
        "  restore {:.3}s: {:.3} GiB over {} segments in {} packs, {} physical reads / \
         {} pack opens, read {:.3}s, longest read {:.3}ms",
        restore_elapsed.as_secs_f64(),
        payload.len() as f64 / GIB,
        stats.segments,
        stats.packs,
        stats.physical_reads,
        stats.packs_opened,
        stats.read_nanos as f64 / 1e9,
        stats.max_read_nanos as f64 / 1e6,
    );
    let verify_gbs = stats.bytes as f64 / GIB / (stats.verify_nanos as f64 / 1e9).max(1e-9);
    let hash_gbs = stats.hashed_bytes as f64 / GIB / (stats.hash_nanos as f64 / 1e9).max(1e-9);
    let cpu_nanos = stats.read_nanos + stats.verify_nanos + stats.hash_nanos;
    println!(
        "  hash: {:.3}s verifying segments ({verify_gbs:.1} GiB/s CPU), \
         {:.3}s on the payload digest ({hash_gbs:.1} GiB/s); hash CPU is {:.0}% of wall",
        stats.verify_nanos as f64 / 1e9,
        stats.hash_nanos as f64 / 1e9,
        cpu_nanos as f64 / stats.total_nanos.max(1) as f64 * 100.0,
    );
    println!(
        "  read {:.3} GiB ({:.1}% over payload), {:.0}% of wall time inside pack reads, \
         {} loose segments, {:.0} MiB/s",
        stats.bytes_read as f64 / GIB,
        (stats.bytes_read as f64 / stats.bytes.max(1) as f64 - 1.0) * 100.0,
        stats.read_nanos as f64 / stats.total_nanos.max(1) as f64 * 100.0,
        stats.loose_segment_reads,
        payload.len() as f64 / (1024.0 * 1024.0) / restore_elapsed.as_secs_f64().max(1e-9),
    );

    // Budget enforcement is the other O(state) path on the commit critical
    // path: it rebuilds the object reference map from every manifest.
    for fraction in [95u64, 85, 70] {
        let usage = store.managed_usage_bytes()?;
        let target = usage * fraction / 100;
        let prune_started = Instant::now();
        let freed = store.enforce_budget_to(target)?;
        println!(
            "  prune to {fraction}% of usage: {:.3}s, freed {:.3} GiB",
            prune_started.elapsed().as_secs_f64(),
            freed as f64 / GIB,
        );
    }

    drop(store);
    let _ = std::fs::remove_dir_all(&root);
    Ok(())
}

fn spill(
    store: &HandoffSegmentStore,
    blocks: &[GeometryBlock],
    tokens: u64,
    options: &Options,
    cache: &mut SegmentCache,
) -> Result<HandoffManifest> {
    let plan = segment_plan(blocks, tokens, options.window_rows);
    for segment in &plan {
        cache.ensure(segment);
    }
    let segments = plan
        .iter()
        .map(|segment| cache.get(segment))
        .collect::<Vec<_>>();
    let held = store
        .try_put_segments(&segments)?
        .map_err(|refusal| anyhow::anyhow!("packed put refused: {}", refusal.reason()))?;
    let mut manifest = HandoffManifest::new("blake3:profile".into(), "full-state".into());
    let mut hasher = blake3::Hasher::new();
    for (index, (stored, bytes)) in held.iter().zip(&segments).enumerate() {
        let bytes_len = bytes.len() as u64;
        manifest.segments.push(HandoffSegmentRef {
            index: index as u32,
            offset: plan[index].offset,
            bytes: bytes_len,
            digest: stored.digest.clone(),
            codec_identity: Some(SegmentCodecIdentity::raw(bytes_len)),
            meta_json: None,
        });
        hasher.update(bytes);
    }
    manifest.total_bytes = plan
        .last()
        .map_or(0, |last| last.offset + last.bytes as u64);
    manifest.kv_bytes = manifest.total_bytes;
    manifest.kv_decoded_bytes = manifest.total_bytes;
    manifest.token_count = tokens;
    manifest.payload_digest = hasher.finalize().to_hex().to_string();
    store.commit(&manifest)?;
    drop(held);
    Ok(manifest)
}

/// Peak BLAKE3 throughput for the two shapes this store asks for: many
/// 128 KiB segment digests, and one large payload digest.
fn hash_benchmark(mib: u64) -> Result<()> {
    let bytes = vec![0x5au8; (mib * MIB) as usize];
    let chunk = 128 * 1024;
    let rounds = 3;
    let mut best = f64::MAX;
    for _ in 0..rounds {
        let started = Instant::now();
        for segment in bytes.chunks(chunk) {
            let _ = blake3::hash(segment);
        }
        best = best.min(started.elapsed().as_secs_f64());
    }
    println!(
        "blake3: {:.2} GiB/s on 128 KiB segments, single thread",
        mib as f64 / 1024.0 / best,
    );
    let mut best = f64::MAX;
    for _ in 0..rounds {
        let started = Instant::now();
        let _ = blake3::hash(&bytes);
        best = best.min(started.elapsed().as_secs_f64());
    }
    println!(
        "blake3: {:.2} GiB/s on one {mib} MiB buffer, single thread",
        mib as f64 / 1024.0 / best,
    );
    let workers = std::thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(1);
    let mut best = f64::MAX;
    for _ in 0..rounds {
        let started = Instant::now();
        let slice_len = (mib as usize * MIB as usize).div_ceil(workers);
        std::thread::scope(|scope| {
            for shard in bytes.chunks(slice_len.max(1)) {
                scope.spawn(move || {
                    let _ = blake3::hash(shard);
                });
            }
        });
        best = best.min(started.elapsed().as_secs_f64());
    }
    println!(
        "blake3: {:.2} GiB/s across {workers} threads",
        mib as f64 / 1024.0 / best,
    );
    Ok(())
}

fn fill_segment(bytes: &mut [u8], block: u64, window: u64) {
    let mut state = block
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(window.wrapping_mul(0xD1B5_4A32_D192_ED03))
        | 1;
    for chunk in bytes.chunks_mut(8) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let value = state.to_le_bytes();
        let take = chunk.len().min(8);
        chunk[..take].copy_from_slice(&value[..take]);
    }
}

fn scratch_root(layout: Layout) -> PathBuf {
    let name = match layout {
        Layout::Single => "single",
        Layout::Turns => "turns",
    };
    std::env::temp_dir().join(format!(
        "skippy-l3-restore-profile-{name}-{}",
        std::process::id()
    ))
}

impl Options {
    fn parse() -> Result<Self> {
        let mut options = Options::default();
        let mut args = std::env::args().skip(1);
        while let Some(flag) = args.next() {
            let value = args
                .next()
                .with_context(|| format!("{flag} needs a value"))?;
            match flag.as_str() {
                "--mib" => options.mib = value.parse()?,
                "--turns" => options.turns = value.parse()?,
                "--tokens-per-turn" => options.tokens_per_turn = value.parse()?,
                "--layers" => options.layers = value.parse()?,
                "--kv-heads" => options.kv_heads = value.parse()?,
                "--head-dim" => options.head_dim = value.parse()?,
                "--dtype-bytes" => options.dtype_bytes = value.parse()?,
                "--window-rows" => options.window_rows = value.parse()?,
                other => anyhow::bail!("unknown flag {other}"),
            }
        }
        Ok(options)
    }
}

#[allow(dead_code)]
fn _assert_payload_geometry_shape(blocks: &[GeometryBlock], tokens: u64, window: u64) {
    let geometry = PayloadGeometry {
        blocks: blocks.to_vec(),
        rows: tokens,
        window_rows: window,
        tail_bytes: 0,
    };
    debug_assert!(geometry.matches(geometry.total_bytes()));
}
