use crate::command::DynResult;
use serde_json::Value;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

pub(super) fn load(root: &Path) -> DynResult<BTreeMap<String, Vec<Value>>> {
    let mut paths = Vec::new();
    collect(root, &mut paths)?;
    paths.sort();
    let mut baseline = BTreeMap::<String, Vec<Value>>::new();
    for path in paths {
        for row in super::history_artifacts::records(&path)? {
            let key = key(&row)?;
            baseline.entry(key).or_default().push(row);
        }
    }
    Ok(baseline)
}
pub(super) fn key(row: &Value) -> DynResult<String> {
    Ok(format!(
        "{}|c{}",
        row["cohort"]["model"]
            .as_str()
            .ok_or("history row missing model")?,
        row["replay"]["concurrency"]
            .as_u64()
            .ok_or("history row missing concurrency")?
    ))
}
pub(super) fn compare(row: &Value, prior: &[Value]) -> DynResult<Vec<String>> {
    if row["complete"] != true {
        return Ok(vec![format!("{}: incomplete run", key(row)?)]);
    }
    let matching = prior
        .iter()
        .filter(|prior| {
            prior["complete"] == true
                && prior["model"]["sha256"] == row["model"]["sha256"]
                && prior["hardware_fingerprint"] == row["hardware_fingerprint"]
                && prior["replay"] == row["replay"]
                && prior["session_cohort_sha256"] == row["session_cohort_sha256"]
        })
        .collect::<Vec<_>>();
    if matching.len() < 3 {
        return Ok(Vec::new());
    }
    let mut problems = Vec::new();
    for (metric, direction) in [
        ("decode_tokens_per_second", 1.0),
        ("end_to_end_tokens_per_second", 1.0),
        ("ttft_ms_mean", -1.0),
        ("ttft_ms_p90", -1.0),
        ("finish_reason_length_pct", -1.0),
    ] {
        let values = matching[matching.len() - 3..]
            .iter()
            .filter_map(|prior| prior[metric].as_f64())
            .collect::<Vec<_>>();
        let Some(candidate) = row[metric].as_f64() else {
            continue;
        };
        if values.len() != 3 {
            continue;
        }
        if !candidate.is_finite()
            || candidate < 0.0
            || values
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err("invalid history baseline metric".into());
        }
        let mut values = values;
        values.sort_by(f64::total_cmp);
        let median = values[1];
        let failed = if median == 0.0 {
            candidate.abs() > 0.0
        } else {
            100.0 * (candidate - median) / median * direction < -5.0
                && (candidate - median).abs() > 0.0
        };
        if failed {
            problems.push(format!(
                "{}: {metric} regressed against median {median:.2}, candidate {candidate:.2}",
                key(row)?
            ));
        }
    }
    Ok(problems)
}
fn collect(root: &Path, files: &mut Vec<PathBuf>) -> DynResult<()> {
    if !std::fs::symlink_metadata(root)?.file_type().is_dir() {
        return Err("baseline root must be a regular directory".into());
    }
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        if kind.is_symlink() {
            return Err("baseline shards must not be symlinked".into());
        }
        if kind.is_dir() {
            collect(&entry.path(), files)?;
        } else if kind.is_file()
            && entry
                .path()
                .extension()
                .is_some_and(|extension| extension == "jsonl")
        {
            files.push(entry.path());
        }
    }
    Ok(())
}
