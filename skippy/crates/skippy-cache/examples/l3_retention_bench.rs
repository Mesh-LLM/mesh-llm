//! Synthetic L3 bookkeeping and retention screen; no model inference is measured.

use std::{fs, hint::black_box, path::Path, time::Instant};

use anyhow::{Context, Result, ensure};
use serde::Serialize;
use skippy_cache::{ExactStatePayload, L3CacheManager, L3Tier, StoreLimits};

const PAYLOAD_BYTES: usize = 64 * 1024;
const SEGMENT_BYTES: usize = 1024;
const NAMESPACE: &str = "max-11-synthetic-v1";

#[derive(Serialize)]
struct Observation {
    case: String,
    sample: usize,
    operation: String,
    elapsed_ns: u128,
    hit: bool,
    used_bytes: u64,
    bytes_written: u64,
    evictions: u64,
}

#[derive(Serialize)]
struct Evidence {
    schema: u32,
    commit: String,
    arm: String,
    repeat: usize,
    payload_bytes: usize,
    segment_bytes: usize,
    observations: Vec<Observation>,
}

fn payload(id: usize) -> Vec<u8> {
    let mut state = (id as u64 + 1).wrapping_mul(0x9e3779b97f4a7c15);
    (0..PAYLOAD_BYTES)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as u8
        })
        .collect()
}

fn tokens(id: usize) -> [i32; 2] {
    [1, id as i32 + 2]
}

fn spill(tier: &L3Tier, id: usize) -> Result<()> {
    tier.spill(
        NAMESPACE,
        &tokens(id),
        &ExactStatePayload::full_state(payload(id)),
        None,
        None,
    )?;
    Ok(())
}

fn open(root: &Path, budget: u64) -> Result<L3Tier> {
    Ok(L3CacheManager::acquire(root, StoreLimits::new(budget, 0))?
        .tier("max-11-exact-state-v1".to_string(), SEGMENT_BYTES))
}

fn observe(
    tier: &L3Tier,
    case: &str,
    sample: usize,
    operation: &str,
    elapsed_ns: u128,
    hit: bool,
) -> Result<Observation> {
    let usage = tier.store().usage()?;
    let activity = tier.activity();
    Ok(Observation {
        case: case.to_string(),
        sample,
        operation: operation.to_string(),
        elapsed_ns,
        hit,
        used_bytes: usage.used_bytes,
        bytes_written: activity.bytes_written,
        evictions: usage.evicted_manifests,
    })
}

fn scaling(root: &Path, count: usize, rows: &mut Vec<Observation>) -> Result<()> {
    let case = format!("manifests-{count}");
    let tier = open(&root.join(&case), 1024 * 1024 * 1024)?;
    for id in 0..count {
        spill(&tier, id)?;
    }
    for sample in 0..16 {
        let id = count + sample;
        let start = Instant::now();
        spill(&tier, id)?;
        let elapsed = start.elapsed().as_nanos();
        rows.push(observe(&tier, &case, sample, "spill", elapsed, false)?);
        let location = tier
            .locate_longest(NAMESPACE, &tokens(id), 1)?
            .context("lost spill")?;
        let fill = tier.load(&location)?;
        ensure!(fill.payload.full_state_bytes_timed()?.0.as_ref() == payload(id));
    }
    Ok(())
}

fn retention_ids() -> Vec<usize> {
    (0..192)
        .chain((0..3).flat_map(|_| 160..192))
        .chain(192..320)
        .chain(160..192)
        .chain(0..192)
        .collect()
}

fn retention(root: &Path, mib: u64, rows: &mut Vec<Observation>) -> Result<()> {
    let case = format!("budget-{mib}MiB");
    let path = root.join(&case);
    let budget = mib * 1024 * 1024;
    let tier = open(&path, budget)?;
    for (sample, id) in retention_ids().into_iter().enumerate() {
        let start = Instant::now();
        let location = tier.locate_longest(NAMESPACE, &tokens(id), 1)?;
        let hit = location.is_some();
        if let Some(location) = location {
            let fill = tier.load(&location)?;
            ensure!(fill.payload.full_state_bytes_timed()?.0.as_ref() == payload(id));
        } else {
            spill(&tier, id)?;
        }
        let elapsed = start.elapsed().as_nanos();
        let row = observe(&tier, &case, sample, "request", elapsed, hit)?;
        ensure!(row.used_bytes <= budget, "managed bytes exceed budget");
        rows.push(row);
    }
    // Reopening checks persistence and reconciliation without touching an existing user cache.
    let mut survivors = Vec::new();
    for id in 160..192 {
        if tier.locate_longest(NAMESPACE, &tokens(id), 1)?.is_some() {
            survivors.push(id);
        }
    }
    ensure!(
        !survivors.is_empty(),
        "no live entries remained to verify across reopen"
    );
    drop(tier);
    let tier = open(&path, budget)?;
    let mut found_after_reopen = Vec::new();
    for id in 160..192 {
        let start = Instant::now();
        let location = tier.locate_longest(NAMESPACE, &tokens(id), 1)?;
        let hit = location.is_some();
        if let Some(location) = location {
            let fill = tier.load(&location)?;
            ensure!(fill.payload.full_state_bytes_timed()?.0.as_ref() == payload(id));
            found_after_reopen.push(id);
        }
        rows.push(observe(
            &tier,
            &case,
            id,
            "reopen",
            start.elapsed().as_nanos(),
            hit,
        )?);
    }
    ensure!(
        survivors.iter().all(|id| found_after_reopen.contains(id)),
        "reopen lost a live entry"
    );
    Ok(())
}

fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    ensure!(
        args.len() == 6,
        "usage: l3_retention_bench ROOT OUTPUT ARM REPEAT COMMIT"
    );
    let root_arg = Path::new(&args[1]);
    ensure!(!root_arg.exists(), "benchmark needs a fresh root");
    // Tier acquisition expects an absolute cache root, and cleanup below
    // removes it recursively, so pin both paths before any store is opened.
    let root = std::path::absolute(root_arg)?;
    let output = std::path::absolute(&args[2])?;
    ensure!(
        !output.starts_with(&root),
        "OUTPUT must live outside ROOT: cleanup removes ROOT recursively"
    );
    fs::create_dir_all(&root)?;
    let mut observations = Vec::new();
    for count in [16, 128, 512] {
        scaling(&root, count, &mut observations)?;
    }
    for mib in [4, 8, 32] {
        retention(&root, mib, &mut observations)?;
    }
    let evidence = Evidence {
        schema: 1,
        commit: args[5].clone(),
        arm: args[3].clone(),
        repeat: args[4].parse()?,
        payload_bytes: PAYLOAD_BYTES,
        segment_bytes: SEGMENT_BYTES,
        observations,
    };
    fs::write(&output, serde_json::to_vec_pretty(&evidence)?)?;
    fs::remove_dir_all(root)?;
    black_box(evidence);
    Ok(())
}
