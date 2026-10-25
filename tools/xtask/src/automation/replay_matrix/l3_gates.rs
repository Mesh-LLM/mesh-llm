use super::l3_contract::{Check, Gates, Phase, Request, Run};
use crate::command::DynResult;
use std::collections::BTreeMap;

fn check(checks: &mut Vec<Check>, name: &str, passed: bool, detail: impl Into<String>) {
    checks.push(Check {
        name: name.into(),
        passed,
        detail: detail.into(),
    });
}
fn median(requests: &[Request]) -> Option<f64> {
    let mut values = requests
        .iter()
        .map(|r| r.ttft_seconds.filter(|v| v.is_finite() && *v > 0.0))
        .collect::<Option<Vec<_>>>()?;
    values.sort_by(f64::total_cmp);
    let length = values.len();
    if length == 0 {
        None
    } else if length % 2 == 1 {
        Some(values[length / 2])
    } else {
        Some(values[length / 2 - 1] / 2.0 + values[length / 2] / 2.0)
    }
}
fn identity(request: &Request) -> (&str, usize) {
    (&request.session_id, request.assistant_turn)
}
fn samples(phase: &Phase, run: &Run, repeats: usize) -> bool {
    let ids = phase
        .requests
        .iter()
        .map(|request| &request.request_id)
        .collect::<std::collections::BTreeSet<_>>();
    ids.len() == phase.requests.len()
        && run.config.required_sources.iter().all(|source| {
            phase
                .requests
                .iter()
                .filter(|request| &request.source_dataset == source)
                .count()
                == repeats
        })
}
fn requests(run: &Run, checks: &mut Vec<Check>, cold: &Phase, restart: &Phase) {
    let all = run
        .phases
        .values()
        .flat_map(|phase| &phase.requests)
        .collect::<Vec<_>>();
    check(
        checks,
        "all_requests_succeed",
        !all.is_empty() && all.iter().all(|r| r.error.is_none()),
        "every recorded request must succeed",
    );
    let mut baseline = BTreeMap::new();
    let mut baseline_valid = !cold.requests.is_empty();
    for request in &cold.requests {
        if let Some(hash) = &request.content_sha256 {
            baseline_valid &= !hash.is_empty();
            if let Some(previous) = baseline.insert(identity(request), hash) {
                baseline_valid &= previous == hash;
            }
        } else {
            baseline_valid = false;
        }
    }
    let matches = baseline_valid
        && all.iter().all(|request| {
            if request.error.is_some() {
                return false;
            }
            // High-load cohorts are disjoint and compared by their own paired gates.
            baseline
                .get(&identity(request))
                .is_none_or(|hash| request.content_sha256.as_ref() == Some(*hash))
        });
    check(
        checks,
        "output_identity",
        matches,
        "greedy seeded output matches every repeated disk-off checkpoint",
    );
    check(
        checks,
        "prompt_token_range",
        cold.requests.iter().chain(&restart.requests).all(|r| {
            r.prompt_tokens
                .is_some_and(|n| (run.config.prompt_min..=run.config.prompt_max).contains(&n))
        }),
        "cold and restart token counts must be measured within the configured range",
    );
    let samples = run.config.restart_samples;
    check(
        checks,
        "every_restart_reads_l3",
        restart.activity_deltas.len() == samples
            && restart
                .activity_deltas
                .iter()
                .all(|d| d.fills >= 1 && d.bytes_read > 0),
        "each owned restart physically reads L3",
    );
}
fn disk(run: &Run, checks: &mut Vec<Check>) -> DynResult<()> {
    let phase = |name| {
        run.phases
            .get(name)
            .ok_or_else(|| format!("missing L3 phase {name}"))
    };
    let delta = |name| {
        phase(name)?
            .activity_delta
            .as_ref()
            .ok_or_else(|| format!("missing delta for {name}"))
    };
    let growth = phase("multi_turn_growth")?;
    let sources = growth
        .requests
        .iter()
        .map(|r| &r.source_dataset)
        .collect::<std::collections::BTreeSet<_>>();
    check(
        checks,
        "multi_turn_growth",
        delta("multi_turn_growth")?.writes > 0
            && run
                .config
                .required_sources
                .iter()
                .all(|s| sources.contains(s)),
        "captured multi-turn growth commits across required sources",
    );
    let l1 = delta("same_process_l1")?;
    check(
        checks,
        "same_process_l1_avoids_disk",
        l1.fills == 0 && l1.bytes_read == 0,
        "same-process L1 avoids physical disk reads",
    );
    let fill = phase("concurrent_fill")?;
    check(
        checks,
        "single_physical_fill",
        delta("concurrent_fill")?.fills == 1 && fill.requests.len() == run.config.identical_repeats,
        "identical-prefix concurrent wave performs one physical fill",
    );
    let record = delta("concurrent_record")?;
    let initial = delta("disk_on_empty")?;
    let allowed =
        (initial.bytes_written as f64 * run.config.max_payload_write_amplification).ceil();
    check(
        checks,
        "single_physical_write",
        record.writes == 1
            && initial.bytes_written > 0
            && (record.bytes_written as f64) <= allowed
            && phase("concurrent_record")?.requests.len() == run.config.identical_repeats,
        "one committed record within payload amplification limit",
    );
    let low = phase("low_space")?;
    check(
        checks,
        "low_space_falls_back_cold",
        !low.requests.is_empty()
            && low.requests.iter().all(|r| r.error.is_none())
            && low
                .status_after
                .as_ref()
                .is_some_and(|s| s.effective.state == "read_only_low_space")
            && delta("low_space")?.writes == 0,
        "low-space read-only inference succeeds without writes",
    );
    let traffic = phase("lifecycle_under_traffic")?;
    check(
        checks,
        "lifecycle_under_traffic",
        !traffic.requests.is_empty()
            && traffic.requests.iter().all(|r| r.error.is_none())
            && traffic.prune.is_some()
            && traffic.clear.is_some()
            && traffic
                .final_clear
                .as_ref()
                .and_then(|o| o.status.usage.as_ref())
                .is_some_and(|u| u.manifests == 0 && u.reserved_inflight_bytes == 0),
        "prune/clear under traffic ends with stable empty cache and no in-flight writer",
    );
    Ok(())
}
fn high_load(run: &Run, checks: &mut Vec<Check>) -> DynResult<()> {
    for concurrency in &run.config.concurrency {
        let summary = |disk| {
            run.phases
                .get(&format!("high_load_{disk}_c{concurrency}"))
                .and_then(|p| p.summary.as_ref())
                .ok_or("missing high-load paired summary")
        };
        let off = summary("off")?;
        let on = summary("on")?;
        let ratio = off
            .decode_inter_token_p99_seconds
            .zip(on.decode_inter_token_p99_seconds)
            .filter(|(a, b)| a.is_finite() && b.is_finite() && *a > 0.0 && *b >= 0.0)
            .map(|(a, b)| 100.0 * (b / a - 1.0));
        check(
            checks,
            &format!("high_load_c{concurrency}"),
            off.failed_requests == 0
                && on.failed_requests == 0
                && !off.content_sha256_by_request.is_empty()
                && off.content_sha256_by_request == on.content_sha256_by_request
                && ratio.is_some_and(|r| r <= run.config.max_decode_p99_regression_pct),
            format!("paired output equality; decode p99 regression {ratio:?}%"),
        );
    }
    Ok(())
}
pub(super) fn evaluate(run: &Run) -> DynResult<Gates> {
    run.config.validate(true)?;
    if run.schema_version != 1 || run.kind != "disk-l3-lifecycle" {
        return Err("unsupported disk-L3 artifact".into());
    }
    let cold = run
        .phases
        .get("disk_off_cold")
        .ok_or("missing disk-off cold samples")?;
    let restart = run
        .phases
        .get("restart_l3")
        .ok_or("missing restart samples")?;
    let mut checks = Vec::new();
    requests(run, &mut checks, cold, restart);
    let cold_p50 = median(&cold.requests);
    let restart_p50 = median(&restart.requests);
    let ratio = cold_p50.zip(restart_p50).map(|(a, b)| b / a);
    check(
        &mut checks,
        "post_restart_l3_ttft",
        ratio.is_some_and(|r| r <= run.config.max_l3_ttft_ratio),
        format!("restart/cold p50 ratio {ratio:?}"),
    );
    disk(run, &mut checks)?;
    high_load(run, &mut checks)?;
    check(
        &mut checks,
        "complete_samples",
        cold.requests.len() == run.config.cold_samples * run.config.required_sources.len()
            && restart.requests.len()
                == run.config.restart_samples * run.config.required_sources.len()
            && samples(cold, run, run.config.cold_samples)
            && samples(restart, run, run.config.restart_samples)
            && run.completed_at.as_ref().is_some_and(|s| !s.is_empty()),
        "all configured cold/restart samples and completion receipt retained",
    );
    Ok(Gates {
        evaluated: true,
        passed: checks.iter().all(|c| c.passed),
        checks,
        cold_ttft_p50_seconds: cold_p50,
        restart_l3_ttft_p50_seconds: restart_p50,
        restart_l3_ttft_ratio: ratio,
    })
}
