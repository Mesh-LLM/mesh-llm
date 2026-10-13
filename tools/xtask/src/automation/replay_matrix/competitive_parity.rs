//! Deterministic offered-concurrency probe on the same owned synthetic server.
use crate::{command::DynResult, process::Cancellation};
use serde_json::{Value, json};
use std::time::{Duration, Instant};
pub(super) async fn probe(
    base: &str,
    model: &str,
    concurrency: usize,
    deadline: Instant,
    cancellation: Cancellation,
) -> DynResult<Value> {
    let mut tasks = tokio::task::JoinSet::new();
    for index in 0..concurrency {
        let base = base.to_owned();
        let model = model.to_owned();
        let cancellation = cancellation.clone();
        tasks.spawn(async move {
        let body=json!({"model":model,"messages":[{"role":"user","content":format!("Reply with exactly one short sentence about scheduler parity. Case {index}.")}],"max_tokens":32,"min_tokens":32,"ignore_eos":true,"temperature":0,"seed":42,"stream":true,"stream_options":{"include_usage":true}});
        let cancelled=async{while !cancellation.is_cancelled(){tokio::time::sleep(Duration::from_millis(5)).await;}};
        let budget=deadline.saturating_duration_since(Instant::now());
        let result=tokio::select!{()=cancelled=>Err("parity request interrupted".to_owned()),result=tokio::time::timeout(budget,crate::automation::openai_exchange::request(&base,&body,false))=>match result{Ok(Ok(evidence))if evidence.completion_tokens==32=>Ok(evidence),Ok(Ok(_))=>Err("parity output shorter than32 tokens".into()),Ok(Err(error))=>Err(error.to_string()),Err(_)=>Err("parity deadline exceeded".into())}};
        match result{Ok(evidence)=>json!({"request_index":index,"valid":true,"requested_completion_tokens":32,"completion_tokens":evidence.completion_tokens,"content_sha256":evidence.content_sha256}),Err(error)=>json!({"request_index":index,"valid":false,"error":error})}
    });
    }
    let mut results = std::collections::BTreeMap::new();
    let mut failures = Vec::new();
    while let Some(result) = tasks.join_next().await {
        match result {
            Ok(row) => {
                results.insert(
                    row["request_index"]
                        .as_u64()
                        .ok_or("parity request index")?,
                    row,
                );
            }
            Err(error) => failures.push(error.to_string()),
        }
    }
    let rows: Vec<_> = results.into_values().collect();
    let passed = failures.is_empty()
        && rows.len() == concurrency
        && rows.iter().all(|row| row["valid"] == true);
    Ok(
        json!({"schema_version":1,"scope":"competitive_scheduler_parity_probe","concurrency":concurrency,"passed":passed,"results":rows,"join_failures":failures}),
    )
}
