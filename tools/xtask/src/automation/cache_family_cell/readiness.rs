//! HTTP/model readiness is an owned finite worker, separate from log admission.
use super::{
    admission::hash,
    contract::{Host, Input},
};
use crate::{command::DynResult, process::Cancellation};
use serde_json::{Value, json};
use std::{
    path::Path,
    time::{Duration, Instant},
};
async fn cancelled(cancel: &Cancellation) {
    while !cancel.is_cancelled() {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}
async fn wait(input: &Input, cancel: &Cancellation) -> bool {
    let until = Instant::now() + Duration::from_secs(input.startup_timeout_secs);
    let url = if input.host == Host::NativeBaseline {
        format!("{}health", input.worker.base_url)
    } else {
        format!("{}/models", input.worker.base_url)
    };
    while Instant::now() < until && !cancel.is_cancelled() {
        let cap = until
            .saturating_duration_since(Instant::now())
            .min(Duration::from_secs(2));
        let bytes = tokio::select! {biased;()=cancelled(cancel)=>return false,
        result=tokio::time::timeout(cap,crate::automation::openai_exchange::get(&url))=>result.ok().and_then(Result::ok)};
        if let Some(bytes) = bytes
            && let Ok(value) = serde_json::from_slice::<Value>(&bytes)
        {
            let ready = if input.host == Host::NativeBaseline {
                value["status"] == "ok"
            } else {
                value["data"]
                    .as_array()
                    .is_some_and(|models| models.iter().any(|model| model["id"] == input.model_id))
            };
            if ready {
                return true;
            }
        }
        tokio::select! {biased;()=cancelled(cancel)=>return false,
        ()=tokio::time::sleep(until.saturating_duration_since(Instant::now()).min(Duration::from_millis(50)))=>{}}
    }
    false
}
pub(super) fn run(input: &Path, output: &Path) -> DynResult<()> {
    let bytes = crate::automation::waiting_prefix::adaptive_identity::bounded(input, 1024 * 1024)?;
    let input: Input = serde_json::from_slice(&bytes)?;
    input.validate()?;
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let cancel = interrupt.cancellation();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let started = Instant::now();
    let ready = runtime.block_on(wait(&input, &cancel));
    let finish = interrupt.finish();
    crate::automation::waiting_prefix::adaptive_identity::fresh(
        output,
        &serde_json::to_vec(
            &json!({"schema_version":1,"request_sha256":hash(&bytes),"ready":ready,
        "elapsed_seconds":started.elapsed().as_secs_f64(),"base_url":input.worker.base_url,"model_id":if input.host==Host::NativeBaseline{None}else{Some(input.model_id)}}),
        )?,
    )?;
    finish?;
    if !ready {
        return Err("cache cell HTTP/model readiness failed".into());
    }
    Ok(())
}
