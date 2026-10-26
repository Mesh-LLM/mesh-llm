//! Narrow allowlisted observed producer rows consumed by the existing renderer.
use crate::command::DynResult;
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::Path};
pub(super) fn row(
    cell: &Value,
    correctness: Option<&Value>,
    observations: &[Value],
    complete: bool,
) -> Value {
    let mut skippy = json!({"status":"unmeasured"});
    if let Some(correctness) = correctness
        && correctness["status"] == "completed"
    {
        let report = &correctness["rows"][0]["evidence"]["skippy"];
        for field in [
            "status",
            "cache_storage_bytes",
            "cache_hit_import_ms",
            "cache_hit_decode_ms",
            "cache_hit_total_ms",
            "recompute_total_ms",
        ] {
            skippy[field] = report[field].clone();
        }
    }
    let serial = observations.iter().find(|r| r["cohort"] == "native-serial");
    let llama = match serial {
        Some(r) if r["status"] == "completed" => {
            json!({"status":"ok","warm_mean_ms":r["warm_statistics"]["warm_mean_ms"],"warm_median_ms":r["warm_statistics"]["warm_median_ms"]})
        }
        Some(r) => json!({"status":r["status"],"reason":r["reason"]}),
        None => json!({"status":"unmeasured"}),
    };
    json!({"family":cell["case"]["family"],"model_id":cell["case"]["model_id"],"payload":cell["case"]["payload"],"stage_load_mode":cell["case"]["stage_load_mode"],"use_case":cell["use_case"]["key"],"use_case_label":cell["use_case"]["label"],"prefix_tokens":cell["case"]["prefix_tokens"],"benchmark_prompt_token_count":correctness.map(|r|r["rows"][0]["evidence"]["skippy"]["benchmark_prompt_token_count"].clone()),"notes":if complete{"Native owned producer observations; supplied source/build provenance, no automatic promotion"}else{"Incomplete/unmeasured producer row; see typed cell evidence; no promotion"},"case":{"resident_kv_bytes_per_token":cell["case"]["resident_kv_bytes_per_token"]},"skippy":skippy,"llama_server":llama})
}
pub(super) fn publish_report(rows: &[Value], cells: &[Value], directory: &Path) -> DynResult<()> {
    let input = json!(rows);
    let bytes = serde_json::to_vec_pretty(&input)?;
    if bytes.len() > 16 * 1024 * 1024 {
        return Err("cache producer report input exceeds16MiB".into());
    }
    let mut use_cases = BTreeMap::new();
    for cell in cells {
        let item = &cell["use_case"];
        if let Some(key) = item["key"].as_str() {
            use_cases.entry(key).or_insert_with(||json!({"key":key,"label":item["label"],"source":{"dataset":item["source"]["dataset"].as_str().unwrap_or_default(),"config":item["source"]["config"].as_str().unwrap_or_default(),"split":item["source"]["split"].as_str().unwrap_or_default(),"row_idx":item["source"]["row_idx"]}}));
        }
    }
    let text = crate::automation::cache_family_report::producer(
        &input,
        &json!({"use_cases":use_cases.into_values().collect::<Vec<_>>()}),
    )?;
    if text.len() > 16 * 1024 * 1024 {
        return Err("cache producer report text exceeds16MiB".into());
    }
    crate::automation::receipt_files::fresh(
        &directory.join("production-cache-bench.json"),
        &bytes,
    )?;
    crate::automation::receipt_files::fresh(
        &directory.join("production-cache-bench.md"),
        text.as_bytes(),
    )
}
