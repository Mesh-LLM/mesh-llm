use super::pooled_metrics::Cell;
use crate::command::DynResult;
use crate::repository::{check_args::Grammar, check_report::CheckReport};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Deserialize)]
pub(super) struct ArmPass {
    pub(super) label: String,
    #[serde(rename = "ref")]
    reference: String,
    commit: String,
    cells: Vec<Cell>,
}

pub(super) fn pool(results: &[ArmPass]) -> DynResult<Vec<serde_json::Value>> {
    let mut groups = BTreeMap::<(&str, usize), Vec<&Cell>>::new();
    let mut metadata = BTreeMap::new();
    for result in results {
        metadata.insert(result.label.as_str(), (&result.reference, &result.commit));
        for cell in &result.cells {
            groups
                .entry((&result.label, cell.concurrency))
                .or_default()
                .push(cell);
        }
    }
    let mut rows = Vec::new();
    for ((label, concurrency), cells) in groups {
        let (reference, commit) = metadata[label];
        let mut row = super::pooled_metrics::aggregate(&cells)?;
        let sum = |select: fn(&Cell) -> u64| {
            cells.iter().try_fold(0_u64, |total, cell| {
                total
                    .checked_add(select(cell))
                    .ok_or("pooled count overflow")
            })
        };
        let requests = sum(|cell| cell.requests)?;
        let success = sum(|cell| cell.successful_requests)?;
        let failures = requests
            .checked_sub(success)
            .ok_or("success count exceeds request count")?;
        let mut failed = cells
            .iter()
            .flat_map(|cell| cell.failed_request_ids.iter().flatten().cloned())
            .collect::<Vec<_>>();
        failed.sort();
        let mut hashes = BTreeMap::<&str, BTreeSet<&str>>::new();
        let mut hash_records = 0_usize;
        let mut successful_records = 0_usize;
        let mut successful_ids = BTreeSet::new();
        for cell in &cells {
            hash_records = hash_records
                .checked_add(cell.content_sha256_by_request.len())
                .ok_or("hash count overflow")?;
            successful_records = successful_records
                .checked_add(cell.successful_request_ids.len())
                .ok_or("request identity count overflow")?;
            successful_ids.extend(cell.successful_request_ids.iter().map(String::as_str));
            for (request, digest) in &cell.content_sha256_by_request {
                hashes.entry(request).or_default().insert(digest);
            }
        }
        row["label"] = label.into();
        row["ref"] = reference.clone().into();
        row["commit"] = commit.clone().into();
        row["concurrency"] = concurrency.into();
        row["passes"] = cells.len().into();
        row["trajectories_per_pass"] = cells[0].trajectories.into();
        row["trajectory_replays"] = sum(|cell| cell.trajectories)?.into();
        row["requests"] = requests.into();
        row["successful_requests"] = success.into();
        row["failed_requests"] = failures.into();
        row["failed_request_ids"] = serde_json::to_value(failed)?;
        row["failure_identity_known"] =
            (failures == 0 || cells.iter().all(|cell| cell.failed_request_ids.is_some())).into();
        row["content_identity_known"] = (successful_records > 0
            && hash_records == successful_records
            && hashes.keys().copied().collect::<BTreeSet<_>>() == successful_ids)
            .into();
        row["content_stable_across_passes"] =
            hashes.values().all(|digests| digests.len() == 1).into();
        row["content_identity"] = serde_json::to_value(hashes)?;
        row["prompt_tokens_min"] =
            serde_json::json!(cells.iter().filter_map(|cell| cell.prompt_tokens_min).min());
        row["prompt_tokens_max"] =
            serde_json::json!(cells.iter().filter_map(|cell| cell.prompt_tokens_max).max());
        rows.push(row);
    }
    compare(
        &mut rows,
        results.first().map(|result| result.label.as_str()),
    );
    Ok(rows)
}

fn compare(rows: &mut [serde_json::Value], baseline: Option<&str>) {
    let baseline: BTreeMap<_, _> = rows
        .iter()
        .filter(|row| row["label"].as_str() == baseline)
        .filter_map(|row| Some((row["concurrency"].as_u64()?, row.clone())))
        .collect();
    for row in rows {
        let reference = row["concurrency"]
            .as_u64()
            .and_then(|concurrency| baseline.get(&concurrency));
        let comparable = reference.is_some_and(|reference| {
            let output = (row["content_identity_known"] == true
                && reference["content_identity_known"] == true
                && row["content_stable_across_passes"] == true
                && reference["content_stable_across_passes"] == true
                && row["content_identity"] == reference["content_identity"])
                || (row["content_identity_known"] == false
                    && reference["content_identity_known"] == false);
            output
                && row["requests"] == reference["requests"]
                && row["failure_identity_known"] == true
                && reference["failure_identity_known"] == true
                && row["failed_request_ids"] == reference["failed_request_ids"]
        });
        row["delta_comparable"] = comparable.into();
        for metric in [
            "agent_steps_per_second",
            "workload_output_tokens_per_second",
            "decode_tokens_per_second",
            "ttft_p50_seconds",
        ] {
            let delta = reference
                .and_then(|reference| row[metric].as_f64().zip(reference[metric].as_f64()))
                .filter(|(_, base)| comparable && *base != 0.0)
                .map(|(value, base)| 100.0 * (value / base - 1.0));
            row[format!("{metric}_delta_pct")] = serde_json::json!(delta);
        }
    }
}

pub(in crate::automation) fn run(args: &[String]) -> DynResult<()> {
    const GRAMMAR: Grammar = Grammar {
        usage: "cargo xtool automation replay-matrix pooled-rows --input PATH --output PATH",
        values: &["--input", "--output"],
        flags: &["--help"],
    };
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    if parsed.flag("--help") {
        return CheckReport::success(format!("{}\n", GRAMMAR.usage)).emit();
    }
    if !parsed.positionals.is_empty() {
        return GRAMMAR.error("unexpected positional arguments").emit();
    }
    let results: Vec<ArmPass> = serde_json::from_slice(&std::fs::read(
        parsed.last("--input").ok_or("missing --input")?,
    )?)?;
    crate::command::write_json_file(
        std::path::Path::new(parsed.last("--output").ok_or("missing --output")?),
        &pool(&results)?,
    )
}
