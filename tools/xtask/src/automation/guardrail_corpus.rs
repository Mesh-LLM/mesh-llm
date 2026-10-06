//! Native local guardrail corpus orchestration; fake rows are synthetic evidence.
#[path = "guardrail_corpus/corpus.rs"]
mod corpus;
#[path = "guardrail_corpus/curl_transport.rs"]
mod curl_transport;
#[cfg(test)]
#[path = "guardrail_corpus/tests.rs"]
mod tests;
#[path = "guardrail_corpus/transport.rs"]
pub(in crate::automation) mod transport;
use crate::{command::DynResult, process::Cancellation};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    io::Write,
    path::PathBuf,
    time::{Duration, Instant},
};

pub(crate) const USAGE: &str = "cargo xtool automation guardrail-corpus --base-url HTTP_HTTPS_OR_FAKE --model MODEL [--guardrail-mode off|metrics|enforce] [--trials 1..1000] --out PATH [--timeout-secs 1..3600]";
struct Options {
    base: String,
    model: String,
    mode: String,
    trials: u64,
    out: PathBuf,
    seconds: u64,
}
impl Options {
    fn parse(args: &[String]) -> DynResult<Self> {
        if !args.len().is_multiple_of(2) {
            return Err(USAGE.into());
        }
        let mut flags = BTreeMap::new();
        for pair in args.as_chunks::<2>().0 {
            if ![
                "--base-url",
                "--model",
                "--guardrail-mode",
                "--trials",
                "--out",
                "--timeout-secs",
            ]
            .contains(&pair[0].as_str())
                || flags.insert(pair[0].as_str(), pair[1].as_str()).is_some()
            {
                return Err("unknown or duplicate guardrail option".into());
            }
        }
        let base = flags.get("--base-url").ok_or(USAGE)?.to_string();
        if !base.starts_with("fake://") {
            let uri: hyper::Uri = base.parse()?;
            if !matches!(uri.scheme_str(), Some("http" | "https"))
                || uri.host().is_none()
                || uri.authority().is_none_or(|a| a.as_str().contains('@'))
                || uri.path().trim_end_matches('/') != "/v1"
                || uri.query().is_some()
            {
                return Err(
                    "guardrail corpus requires http(s)://HOST:PORT/v1 or fake://LABEL".into(),
                );
            }
        }
        let model = flags.get("--model").ok_or(USAGE)?.to_string();
        let mode = flags.get("--guardrail-mode").unwrap_or(&"off").to_string();
        let trials = flags.get("--trials").unwrap_or(&"20").parse()?;
        let seconds = flags.get("--timeout-secs").unwrap_or(&"3600").parse()?;
        if model.is_empty()
            || model.len() > 4096
            || model.contains(['\0', '\r', '\n'])
            || !["off", "metrics", "enforce"].contains(&mode.as_str())
            || !(1..=1000).contains(&trials)
            || !(1..=3600).contains(&seconds)
        {
            return Err("invalid guardrail mode/model/count/budget".into());
        }
        let requested = std::path::absolute(flags.get("--out").ok_or(USAGE)?)?;
        let out = prepare_output(&requested)?;
        Ok(Self {
            base,
            model,
            mode,
            trials,
            out,
            seconds,
        })
    }
}
// Canonicalize the nearest existing directory, then admit every new child entry.
// This is local output custody, not a hostile concurrent-filesystem sandbox.
fn prepare_output(requested: &std::path::Path) -> DynResult<PathBuf> {
    fresh_output(requested)?;
    let name = requested.file_name().ok_or("output filename")?;
    let mut anchor = requested.parent().ok_or("output parent")?;
    let mut missing = Vec::new();
    loop {
        match std::fs::symlink_metadata(anchor) {
            Ok(metadata) => {
                if !metadata.is_dir() || metadata.is_symlink() {
                    return Err("guardrail output parent must be a regular directory".into());
                }
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                missing.push(
                    anchor
                        .file_name()
                        .ok_or("output directory component")?
                        .to_owned(),
                );
                anchor = anchor.parent().ok_or("output ancestor")?;
            }
            Err(error) => return Err(error.into()),
        }
    }
    let mut parent = anchor.canonicalize()?;
    for component in missing.into_iter().rev() {
        parent.push(component);
        match std::fs::create_dir(&parent) {
            Ok(()) => (),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => (),
            Err(error) => return Err(error.into()),
        }
        let metadata = std::fs::symlink_metadata(&parent)?;
        if !metadata.is_dir() || metadata.is_symlink() {
            return Err("guardrail output directory boundary changed".into());
        }
    }
    let out = parent.join(name);
    fresh_output(&out)?;
    Ok(out)
}
fn fresh_output(out: &std::path::Path) -> DynResult<()> {
    match std::fs::symlink_metadata(out) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
        Ok(_) => Err("guardrail output must be fresh".into()),
    }
}
pub(crate) fn run(args: &[String]) -> DynResult<()> {
    if args == ["--help"] || args == ["-h"] {
        println!("{USAGE}");
        return Ok(());
    }
    let input = Options::parse(args)?;
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let cancellation = interrupt.cancellation();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let result = runtime.block_on(execute(&input, &cancellation));
    let restored = interrupt.finish();
    let (report, failed) = result?;
    let mut file = tempfile::NamedTempFile::new_in(input.out.parent().ok_or("output parent")?)?;
    file.write_all(&serde_json::to_vec_pretty(&report)?)?;
    file.write_all(b"\n")?;
    file.as_file().sync_all()?;
    file.persist_noclobber(&input.out)?;
    println!(
        "{}",
        json!({"out":input.out,"backend_mode":report["backend_mode"],"prompt_count":5,"trials":input.trials,"total_requests":report["total_requests"]})
    );
    restored?;
    if failed {
        Err("guardrail corpus transport incomplete; partial evidence retained".into())
    } else {
        Ok(())
    }
}
async fn bounded<T>(
    future: impl std::future::Future<Output = DynResult<T>>,
    deadline: Instant,
    cancellation: &Cancellation,
) -> DynResult<T> {
    let remaining = deadline.saturating_duration_since(Instant::now());
    if remaining.is_zero() || cancellation.is_cancelled() {
        return Err("guardrail deadline/cancellation".into());
    }
    let cancelled = async {
        loop {
            if cancellation.is_cancelled() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    };
    tokio::select! {result=tokio::time::timeout(remaining.min(Duration::from_secs(60)),future)=>result.map_err(|_|"guardrail HTTP deadline")?,()=cancelled=>Err("guardrail cancelled".into())}
}
fn synthetic(case: &corpus::Case, trial: u64, mode: &str) -> Value {
    let hash = Sha256::digest(format!("{mode}:{}:{trial}", case.id).as_bytes());
    let sample = u16::from_be_bytes([hash[0], hash[1]]);
    json!({"ok":case.supported(),"status":if case.supported(){200}else{400},"response_kind":case.expected,"latency_ms":4.0+f64::from(sample%2400)/100.0,"latency_origin":"deterministic_synthetic_not_measured"})
}
async fn execute(input: &Options, cancellation: &Cancellation) -> DynResult<(Value, bool)> {
    let deadline = Instant::now() + Duration::from_secs(input.seconds);
    let probe = if input.base.starts_with("fake://") {
        Ok(false)
    } else {
        bounded(
            transport::exchange(
                &input.base,
                "models",
                None,
                deadline.min(Instant::now() + Duration::from_secs(10)),
                cancellation,
            ),
            deadline.min(Instant::now() + Duration::from_secs(10)),
            cancellation,
        )
        .await
        .and_then(|r| {
            let v: Value = serde_json::from_slice(&r.body)?;
            Ok(r.status == 200 && v["data"].as_array().is_some_and(|a| !a.is_empty()))
        })
    };
    let live = probe.as_ref().is_ok_and(|v| *v);
    let fallback_reason = if input.base.starts_with("fake://") {
        Some("explicit_fake_scheme".to_string())
    } else if !live {
        Some("availability_probe_failed_or_empty".to_string())
    } else {
        None
    };
    let mut results = Vec::new();
    let mut incomplete = false;
    'trials: for trial in 0..input.trials {
        for case in corpus::cases() {
            if cancellation.is_cancelled() || Instant::now() >= deadline {
                incomplete = true;
                break 'trials;
            }
            let body = case.request(&input.model, &input.mode);
            let started = Instant::now();
            let mut observed_status = None;
            let result = if live {
                bounded(transport::exchange(&input.base,"chat/completions",Some(&body),deadline.min(Instant::now()+Duration::from_secs(60)),cancellation),deadline,cancellation).await
                    .and_then(|r| {observed_status=Some(r.status);let stream=body["stream"]==true;let ok=if r.status==200{transport::successful(&r.body,stream)?&&case.supported()}else{false};Ok(json!({"ok":ok,"status":r.status,"response_kind":if r.status==200 {if stream{"stream"}else{"chat"}}else{"error"},"latency_ms":started.elapsed().as_secs_f64()*1000.0,"latency_origin":"observed_http"}))})
            } else {
                Ok(synthetic(&case, trial, &input.mode))
            };
            let failed = result.is_err();
            let mut result=result.unwrap_or_else(|_|json!({"ok":false,"status":observed_status,"response_kind":"transport_failure","error_kind":if observed_status==Some(200){"invalid_response_protocol"}else{"connection_or_deadline"},"latency_ms":started.elapsed().as_secs_f64()*1000.0,"latency_origin":"observed_failed_request"}));
            result.as_object_mut().ok_or("case result")?.extend(json!({"trial":trial+1,"case_id":case.id,"category":case.category,"expected_outcome":case.expected,"retry_count":case.declared_retries,"artifact_path":format!(".sisyphus/evidence/openai-guardrail-corpus/{}.json",case.id)}).as_object().ok_or("row metadata")?.clone());
            results.push(result);
            if failed {
                incomplete = true;
                break 'trials;
            }
        }
    }
    Ok((
        report(input, results, live, fallback_reason, incomplete),
        incomplete,
    ))
}
fn report(
    input: &Options,
    results: Vec<Value>,
    live: bool,
    fallback: Option<String>,
    incomplete: bool,
) -> Value {
    let success = results.iter().filter(|r| r["ok"] == true).count();
    let latencies = results
        .iter()
        .filter_map(|r| r["latency_ms"].as_f64())
        .collect();
    let server = match input.mode.as_str() {
        "off" => "disabled",
        "metrics" => "metrics",
        _ => "enforce",
    };
    let metadata = json!({"model":input.model,"guardrail_mode":input.mode,"mesh_guardrails":input.mode!="off","expected_server_mode":server,"api_path":"/v1/chat/completions","backend_mode":if live{"live"}else{"fake"},"trials_per_prompt":input.trials,"corpus_name":"openai-guardrail-corpus-v1"});
    json!({"schema_version":1,"status":if incomplete{"incomplete"}else{"completed"},"backend_mode":if live{"live"}else{"fake"},"fallback_reason":fallback,"expected_server_mode":server,"request_metadata_used":metadata,"prompt_count":5,"trials":input.trials,"planned_requests":5*input.trials,"total_requests":results.len(),"success_count":success,"failure_count":results.len()-success,"retry_count":input.trials,"retry_count_scope":"declared_corpus_metadata_no_physical_retry","physical_retries":0,"latency_ms":latency_summary(latencies),"corpus":corpus::cases().into_iter().map(|c|json!({"case_id":c.id,"category":c.category,"prompt":c.prompt,"request_overrides":c.overrides,"expected_outcome":c.expected,"retry_count":c.declared_retries,"artifact_path":format!(".sisyphus/evidence/openai-guardrail-corpus/{}.json",c.id)})).collect::<Vec<_>>(),"results":results})
}
fn latency_summary(mut values: Vec<f64>) -> Value {
    values.sort_by(f64::total_cmp);
    if values.is_empty() {
        return json!({"min":null,"mean":null,"p50":null,"p95":null,"max":null});
    }
    let percentile = |p: f64| {
        let position = p * (values.len() - 1) as f64;
        let low = position.floor() as usize;
        let high = position.ceil() as usize;
        values[low] + (values[high] - values[low]) * (position - low as f64)
    };
    json!({"min":values[0],"mean":values.iter().sum::<f64>()/values.len()as f64,"p50":percentile(0.5),"p95":percentile(0.95),"max":values[values.len()-1]})
}
