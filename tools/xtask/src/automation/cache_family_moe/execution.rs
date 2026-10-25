use super::{
    contract::{Case, Input},
    evidence, processes,
};
use crate::{automation::cache_family_run::child, command::DynResult, process::Cancellation};
use serde_json::{Value, json};
use std::{
    path::Path,
    time::{Duration, Instant},
};
fn one(
    input: &Input,
    case: &Case,
    directory: &Path,
    until: Instant,
    cancel: &Cancellation,
) -> DynResult<Value> {
    let request = input.admission(case)?;
    let (before, clean) = child::run(
        "cache-family-cell-admission",
        &request,
        &directory.join("before"),
        until,
        cancel,
    )?;
    if !clean || before["admitted"].is_null() {
        return Err("MoE inspector/model/native build identity refused".into());
    }
    let inspection = processes::inspect(&before["admitted"], directory, until, cancel)?;
    let mut correctness = case.correctness.clone();
    correctness["model"] = before["admitted"]["model"].clone();
    correctness["native_build"] = before["admitted"]["native_build"].clone();
    correctness["borrow_resident_hits"] = json!(true);
    correctness["cache_decoded_result_hits"] = json!(false);
    if correctness["topologies"].is_null() {
        correctness["topologies"] = json!(["one-stage", "split-middle", "split-final"]);
    }
    let (stage, clean) = child::run(
        "cache-family-correctness",
        &correctness,
        &directory.join("correctness"),
        until,
        cancel,
    )?;
    let (after, after_clean) = child::run(
        "cache-family-cell-admission",
        &request,
        &directory.join("after"),
        until,
        cancel,
    )?;
    if !after_clean || before != after {
        return Err("MoE inspector/model/build bytes changed".into());
    }
    let mut rows = Vec::new();
    let mut complete = clean && stage["status"] == "completed";
    if let Some(trials) = stage["rows"].as_array() {
        for trial in trials {
            let report = &trial["evidence"]["skippy"];
            if trial["evidence"]["status"] != "pass" {
                complete = false;
                rows.push(json!({"status":"fail","topology":trial["topology"],"reason":"owned_cache_correctness_failed"}));
                continue;
            }
            let start = report["layer_start"]
                .as_u64()
                .ok_or("MoE layer start absent")?
                .try_into()?;
            let end = report["layer_end"]
                .as_u64()
                .ok_or("MoE layer end absent")?
                .try_into()?;
            let expert = evidence::experts(&inspection, start, end)?;
            let row = evidence::row(trial, &trial["topology"], report, &expert);
            complete &= row["status"] == "pass";
            rows.push(row);
        }
    }
    let expected = correctness["topologies"]
        .as_array()
        .ok_or("MoE requested topology roster absent")?
        .len();
    complete &= rows.len() == expected && expected > 0;
    Ok(
        json!({"status":if complete{"completed"}else{"incomplete"},"rows":rows,"before_identity":before,"after_identity":after,"correctness_evidence":"correctness/output/cache-correctness-stage.json","scope":"provided_pinned_inspector_tensor_presence_plus_correctness_no_sequence_ID_or_route_attestation"}),
    )
}
pub(super) fn execute(input: &Input, directory: &Path, cancel: &Cancellation) -> DynResult<Value> {
    let until = Instant::now() + Duration::from_secs(input.execution_seconds);
    let mut cases = Vec::new();
    let mut complete = true;
    for (index, case) in input.cases.iter().enumerate() {
        let value = if cancel.is_cancelled() || Instant::now() >= until {
            json!({"status":"not-launched","rows":[]})
        } else {
            let root = directory.join(format!("case-{index:02}"));
            std::fs::create_dir(&root)?;
            let result=one(input,case,&root,until,cancel).unwrap_or_else(|_|json!({"status":"refused","reason":"owned_identity_inspection_correctness_or_deadline_failed","rows":[]}));
            super::publish(&root.join("case.json"), &result)?;
            result
        };
        complete &= value["status"] == "completed";
        cases.push(json!({"case_key":case.correctness["case_key"],"evidence":format!("case-{index:02}/case.json"),"observation":value}));
    }
    Ok(
        json!({"schema_version":1,"status":if complete&&!cancel.is_cancelled()&&Instant::now()<until{"completed"}else{"incomplete"},"cases":cases,"inspector_source_provenance":"caller_declared_not_build_attested","promotion":null}),
    )
}
