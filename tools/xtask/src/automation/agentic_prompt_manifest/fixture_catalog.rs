//! Admission of pinned scheduler fixture catalogs before materialization.
use serde_json::Value;

use super::selection::Selection;
use crate::command::DynResult;

const WORKLOAD_FIELDS: [&str; 10] = [
    "rounds",
    "families",
    "requests_per_family",
    "prefix_blocks",
    "output_tokens",
    "ctx_size",
    "lanes",
    "admission_concurrency",
    "cache_entries",
    "stagger_ms",
];

pub(super) fn validate(catalog: &Value) -> DynResult<()> {
    if catalog.get("schema_version").and_then(Value::as_u64) != Some(1) {
        return Err("scheduler fixture schema_version must be 1".into());
    }
    let datasets = catalog
        .get("datasets")
        .and_then(Value::as_object)
        .ok_or("scheduler fixture datasets must be an object")?;
    let profiles = catalog
        .get("profiles")
        .and_then(Value::as_object)
        .filter(|profiles| !profiles.is_empty())
        .ok_or("scheduler fixture profiles must be nonempty")?;
    for dataset in datasets.values() {
        validate_dataset(dataset)?;
    }
    for profile in profiles.values() {
        validate_profile(profile, datasets)?;
    }
    Ok(())
}

fn validate_dataset(dataset: &Value) -> DynResult<()> {
    text(dataset, "repo_id")?;
    if text(dataset, "repo_type")? != "dataset" {
        return Err("fixture repo_type must be dataset".into());
    }
    hex_identity(text(dataset, "revision")?, 40)?;
    let parquet = text(dataset, "parquet_file")?;
    let files = dataset
        .get("files")
        .and_then(Value::as_array)
        .ok_or("fixture dataset files must be an array")?;
    if !files.iter().any(|file| file.as_str() == Some(parquet)) {
        return Err("dataset files must include parquet_file".into());
    }
    if !dataset.get("provenance").is_some_and(Value::is_object) {
        return Err("fixture dataset provenance missing".into());
    }
    Ok(())
}

fn validate_profile(profile: &Value, datasets: &serde_json::Map<String, Value>) -> DynResult<()> {
    text(profile, "description")?;
    let model = object(profile, "model")?;
    for field in ["id", "repo", "filename"] {
        text(model, field)?;
    }
    hex_identity(text(model, "revision")?, 40)?;
    hex_identity(text(model, "sha256")?, 64)?;
    let workload = object(profile, "workload")?;
    for field in WORKLOAD_FIELDS {
        let value = workload
            .get(field)
            .and_then(Value::as_f64)
            .filter(|value| value.is_finite() && *value > 0.0);
        if value.is_none() {
            return Err(format!("fixture workload {field} must be positive").into());
        }
    }
    let families = integer(workload, "families")?;
    let requests = families
        .checked_mul(integer(workload, "requests_per_family")?)
        .ok_or("fixture request count overflow")?;
    if integer(workload, "admission_concurrency")? != requests
        || integer(workload, "lanes")? < requests
    {
        return Err("fixture must admit its complete workload with one lane per request".into());
    }
    validate_trace(object(profile, "ci_trace")?, families, requests)?;
    object(profile, "hardware_acceptance")?;
    let corpus = object(profile, "corpus")?;
    match text(corpus, "kind")? {
        "synthetic" => Ok(()),
        "hf" => validate_hf_corpus(corpus, workload, datasets),
        _ => Err("unsupported scheduler fixture corpus kind".into()),
    }
}

fn validate_trace(trace: &Value, families: u64, requests: u64) -> DynResult<()> {
    for field in [
        "prompt_tokens",
        "expected_fcfs_switches",
        "expected_dfs_switches",
    ] {
        integer(trace, field)?;
    }
    let order = trace
        .get("family_order")
        .and_then(Value::as_array)
        .ok_or("fixture family_order must be an array")?;
    let order = order
        .iter()
        .map(|value| {
            value
                .as_u64()
                .ok_or("fixture family must be a nonnegative integer")
        })
        .collect::<Result<Vec<_>, _>>()?;
    if u64::try_from(order.len())? != requests || order.iter().any(|family| *family >= families) {
        return Err("fixture trace must cover every request and valid family".into());
    }
    let unique = order.into_iter().collect::<std::collections::BTreeSet<_>>();
    if u64::try_from(unique.len())? != families {
        return Err("fixture trace must cover every family".into());
    }
    Ok(())
}

fn validate_hf_corpus(
    corpus: &Value,
    workload: &Value,
    datasets: &serde_json::Map<String, Value>,
) -> DynResult<()> {
    if !datasets.contains_key(text(corpus, "dataset")?) {
        return Err("fixture references unknown dataset".into());
    }
    let selection = selection(object(corpus, "selection")?)?;
    if u64::try_from(selection.families)? != integer(workload, "families")? {
        return Err("fixture family count drifted".into());
    }
    hex_identity(text(corpus, "prompt_manifest_sha256")?, 64)?;
    let rows = corpus
        .get("rows")
        .and_then(Value::as_array)
        .ok_or("fixture rows must be an array")?;
    if rows.len() != selection.families {
        return Err("fixture must pin one row per family".into());
    }
    let mut tokens = 0_u64;
    for row in rows {
        text(row, "session_id")?;
        text(row, "source_dataset")?;
        integer(row, "n_turns")?;
        integer(row, "max_isl")?;
        let count = integer(row, "total_tokens")?;
        if count == 0 {
            return Err("fixture total_tokens must be positive".into());
        }
        tokens = tokens
            .checked_add(count)
            .ok_or("fixture token count overflow")?;
    }
    let requests = integer(workload, "families")?
        .checked_mul(integer(workload, "requests_per_family")?)
        .ok_or("fixture request count overflow")?;
    let prompt = tokens
        .checked_mul(integer(workload, "requests_per_family")?)
        .ok_or("fixture prompt count overflow")?;
    let output = requests
        .checked_mul(integer(workload, "output_tokens")?)
        .ok_or("fixture output count overflow")?;
    let required = prompt
        .checked_add(output)
        .and_then(u64::checked_next_power_of_two)
        .ok_or("fixture context count overflow")?;
    if integer(workload, "ctx_size")? < required {
        return Err(format!(
            "fixture ctx_size must cover pinned row totals, need at least {required}"
        )
        .into());
    }
    Ok(())
}

pub(super) fn selection(value: &Value) -> DynResult<Selection> {
    if text(value, "order")? != "md5(session_id)" {
        return Err("fixture selection order must be md5(session_id)".into());
    }
    let sources = value
        .get("sources")
        .and_then(Value::as_array)
        .ok_or("fixture sources must be an array")?
        .iter()
        .map(|source| {
            source
                .as_str()
                .map(String::from)
                .ok_or("fixture source must be a string")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let selection = Selection {
        sources,
        families: usize::try_from(integer(value, "families")?)?,
        min_isl: integer(value, "min_isl")?,
        max_isl_exclusive: integer(value, "max_isl_exclusive")?,
        min_turns: integer(value, "min_turns")?,
    };
    selection.validate()?;
    Ok(selection)
}

pub(super) fn text<'a>(value: &'a Value, field: &str) -> DynResult<&'a str> {
    value
        .get(field)
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .ok_or_else(|| format!("fixture {field} must be nonempty text").into())
}

pub(super) fn integer(value: &Value, field: &str) -> DynResult<u64> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("fixture {field} must be a nonnegative integer").into())
}

pub(super) fn object<'a>(value: &'a Value, field: &str) -> DynResult<&'a Value> {
    value
        .get(field)
        .filter(|value| value.is_object())
        .ok_or_else(|| format!("fixture {field} must be an object").into())
}

fn hex_identity(text: &str, length: usize) -> DynResult<()> {
    if text.len() != length || !text.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("fixture identity must have {length} hexadecimal characters").into());
    }
    Ok(())
}
