use super::runtime_seed_io::{files, read, save};
use super::runtime_seed_stats::Snapshot;
use super::runtime_seed_types::{Arm, EPOCH, IMAGE, ResultEvidence};
use crate::command::DynResult;
use std::path::Path;

pub(super) fn run(directory: &Path) -> DynResult<()> {
    let paths = files(directory, Some("result.json"))?;
    if paths.len() != 6 {
        return Err("inconclusive: require all six results".into());
    }
    let mut results = Vec::new();
    for path in paths {
        let result: ResultEvidence = read(&path)?;
        let raw: Snapshot = read(&path.with_file_name("raw-stats.json"))?;
        validate(&result, &raw)?;
        results.push(result);
    }
    let first = results.first().ok_or("missing samples")?;
    for result in &results {
        let context = &result.context;
        if context.source != first.context.source
            || context.cache.id != first.context.cache.id
            || context.cache.version != first.context.cache.version
            || context.run_id != first.context.run_id
            || context.run_attempt != first.context.run_attempt
        {
            return Err("inconclusive/mismatched evidence".into());
        }
    }
    let mut deltas = Vec::new();
    let mut coverage = true;
    for pair in 1..=3 {
        let cold = sample(&results, pair, Arm::Cold)?;
        let warm = sample(&results, pair, Arm::Warm)?;
        if cold.native_cacheable_requests == 0
            || cold.native_cacheable_requests != warm.native_cacheable_requests
        {
            return Err("incomparable C/C++ workloads".into());
        }
        if cold.context.host_cpu.is_empty()
            || cold.context.host_cpu != warm.context.host_cpu
            || cold.context.runner_class.is_empty()
            || cold.context.runner_class != warm.context.runner_class
            || cold.context.kernel.is_empty()
            || cold.context.kernel != warm.context.kernel
            || cold.context.runner_image_os != warm.context.runner_image_os
            || cold.context.runner_image_version != warm.context.runner_image_version
        {
            return Err("incomparable host".into());
        }
        deltas.push(cold.total_seconds - warm.total_seconds);
        coverage &= warm.native_hits > cold.native_hits;
    }
    let benefit = deltas.iter().all(|delta| *delta > 0.0);
    let mut ordered = deltas.clone();
    ordered.sort_by(f64::total_cmp);
    let mut reasons = Vec::new();
    if results.iter().any(|result| !result.warm_floor_passed) {
        reasons.push("warm-floor-failure");
    }
    if !coverage {
        reasons.push("no-incremental-c-cpp-coverage");
    }
    if !benefit {
        reasons.push("no-consistent-total-time-benefit");
    }
    let summary = serde_json::json!({"schema":1, "paired_seconds_saved":deltas,
        "median_seconds_saved":ordered[1], "native_coverage_observed":coverage,
        "total_benefit_observed":benefit, "eligibility_changed":false,
        "classification":if reasons.is_empty() { "observed-benefit" } else { "not-qualified" },
        "reasons":reasons});
    save(&directory.join("summary.json"), &summary)?;
    crate::command::print_json(&summary)
}

fn sample(results: &[ResultEvidence], pair: u8, arm: Arm) -> DynResult<&ResultEvidence> {
    let mut matches = results
        .iter()
        .filter(|result| result.context.pair == pair && result.context.arm == arm);
    let result = matches.next().ok_or("duplicate/missing sample")?;
    if matches.next().is_some() {
        return Err("duplicate/missing sample".into());
    }
    Ok(result)
}

fn validate(result: &ResultEvidence, raw: &Snapshot) -> DynResult<()> {
    let context = &result.context;
    context.cache.validate()?;
    if context.image != IMAGE || context.epoch != EPOCH {
        return Err("image mismatch".into());
    }
    if !result.verified || result.eligibility_changed {
        return Err("missing verification".into());
    }
    for seconds in [
        result.action_seconds,
        result.total_seconds,
        context.restore_seconds.ok_or("missing restore timing")?,
    ] {
        if !seconds.is_finite() || seconds < 0.0 {
            return Err("invalid timing".into());
        }
    }
    let measurement = raw.measurement()?;
    let floor = context.arm == Arm::Cold || measurement.hit_rate >= 0.01;
    if result.native_hits != measurement.native_hits
        || result.native_cacheable_requests != measurement.native_requests
        || (result.hit_rate - measurement.hit_rate).abs() > f64::EPSILON
        || result.warm_floor_passed != floor
        || result.classification
            != if floor {
                "measured"
            } else {
                "warm-floor-failure"
            }
        || result.language_hits != raw.stats.cache_hits.counts
        || result.language_misses != raw.stats.cache_misses.counts
        || result.assembler_hits
            != raw
                .stats
                .cache_hits
                .counts
                .get("Assembler")
                .copied()
                .unwrap_or(0)
        || result.assembler_misses
            != raw
                .stats
                .cache_misses
                .counts
                .get("Assembler")
                .copied()
                .unwrap_or(0)
    {
        return Err("derived counter mismatch".into());
    }
    Ok(())
}
