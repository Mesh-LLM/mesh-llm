//! Competitive cache-trace request worker; launch ownership stays with the retained server owners.
use crate::{command::DynResult, process::Cancellation};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::OpenOptions,
    io::{Read, Write},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    config: PathBuf,
    config_sha256: String,
    manifest: PathBuf,
    cell: Value,
    base_url: String,
    served_model: String,
    launch_provenance: Value,
    output: PathBuf,
    timeout_seconds: u64,
    request_timeout_seconds: u64,
}
struct Admitted {
    config: Value,
    prompts: Vec<Value>,
    concurrency: usize,
    output_tokens: u64,
}
pub(in crate::automation) fn run(args: &[String]) -> DynResult<()> {
    use crate::repository::{check_args::Grammar, check_report::CheckReport};
    const GRAMMAR: Grammar = Grammar {
        usage: "cargo xtool automation replay-matrix competitive-cell --input PATH",
        values: &["--input"],
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
    let started = Instant::now();
    let input: Input = serde_json::from_slice(&read(
        Path::new(parsed.last("--input").ok_or("missing --input")?),
        8 * 1024 * 1024,
    )?)?;
    let admitted = admit(&input)?;
    let deadline = started + Duration::from_secs(input.timeout_seconds);
    if !input.output.is_absolute() {
        return Err("competitive output must be absolute".into());
    }
    // Refuse existing evidence rather than overwrite or silently discard a prior cell.
    std::fs::create_dir(&input.output)?;
    let mut records = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(input.output.join("requests.jsonl"))?;
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let measured = runtime.block_on(measure(
        &input,
        &admitted,
        deadline,
        interrupt.cancellation(),
        &mut records,
    ));
    let finish = interrupt.finish();
    records.flush()?;
    let mut summary = match &measured {
        Ok(summary) => summary.clone(),
        Err(error) => {
            json!({"schema_version":1,"scope":"competitive_request_worker","completed":false,"error":error.to_string(),"cell":input.cell,"config_sha256":input.config_sha256,"launch_provenance":input.launch_provenance})
        }
    };
    summary["requests_sha256"] =
        crate::product::digest::file_sha256(&input.output.join("requests.jsonl"))
            .map_err(|error| error.error)?
            .into();
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(input.output.join("worker-summary.json"))?;
    file.write_all(&serde_json::to_vec_pretty(&summary)?)?;
    file.write_all(b"\n")?;
    finish?;
    let summary = measured?;
    if summary["passed"] != true {
        return Err("competitive requests failed; worker evidence retained".into());
    }
    Ok(())
}
pub(super) fn read(path: &Path, maximum: u64) -> DynResult<Vec<u8>> {
    if !std::fs::symlink_metadata(path)?.is_file() {
        return Err("competitive input must be regular".into());
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
    }
    let file = options.open(path)?;
    if !file.metadata()?.is_file() {
        return Err("opened competitive input must be regular".into());
    }
    let mut bytes = Vec::new();
    file.take(maximum + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > maximum {
        return Err("competitive input exceeds bounded size".into());
    }
    Ok(bytes)
}
fn digest(value: &Value) -> bool {
    value.as_str().is_some_and(|text| {
        text.len() == 64
            && text
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    })
}
fn admit_provenance(provenance: &Value, config: &Value, cell: &Value) -> DynResult<()> {
    let launch = provenance
        .as_object()
        .ok_or("launch provenance object required")?;
    let allowed = [
        "binary_sha256",
        "model_sha256",
        "runtime_directory_sha256",
        "backend_version_sha256",
        "comparison_input_sha256",
        "tensor_equivalence_sha256",
    ];
    if launch.keys().any(|key| !allowed.contains(&key.as_str()))
        || launch.values().any(|value| !digest(value))
    {
        return Err("launch provenance must contain only bounded hash identities".into());
    }
    if !launch.contains_key("binary_sha256") {
        return Err("launch binary identity required".into());
    }
    let model = config["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|model| model["key"] == cell["model"])
        .ok_or("competitive model missing")?;
    if launch.get("model_sha256") != Some(&model["sha256"]) {
        return Err("launch model identity differs from selected pin".into());
    }
    Ok(())
}
fn admit(input: &Input) -> DynResult<Admitted> {
    if !(1..=86400).contains(&input.timeout_seconds)
        || input.request_timeout_seconds == 0
        || input.request_timeout_seconds > input.timeout_seconds
    {
        return Err("invalid competitive deadline budget".into());
    }
    let uri: hyper::Uri = input.base_url.parse()?;
    if uri.scheme_str() != Some("http")
        || uri.host() != Some("127.0.0.1")
        || uri.port_u16().is_none_or(|port| port == 0)
        || uri.path() != "/v1"
        || uri.query().is_some()
    {
        return Err("competitive worker requires owned loopback http://127.0.0.1:<port>/v1".into());
    }
    if input.served_model.is_empty() || input.served_model.chars().any(char::is_control) {
        return Err("competitive served model required".into());
    }
    let config_bytes = read(&input.config, 8 * 1024 * 1024)?;
    if hex::encode(Sha256::digest(&config_bytes)) != input.config_sha256 {
        return Err("competitive config source identity changed".into());
    }
    let config: Value = serde_json::from_slice(&config_bytes)?;
    super::competitive_plan::admit_cell(&config, &config_bytes, &input.cell)?;
    if input.cell["workload"] != "thoughtworks" {
        return Err("competitive worker currently requires cache-trace workload".into());
    }
    admit_provenance(&input.launch_provenance, &config, &input.cell)?;
    let bytes = read(&input.manifest, 64 * 1024 * 1024)?;
    let selection = &config["thoughtworks"]["selection"];
    if hex::encode(Sha256::digest(&bytes))
        != selection["manifest_sha256"]
            .as_str()
            .ok_or("prompt manifest hash missing")?
    {
        return Err("competitive prompt bytes changed".into());
    }
    let manifest: Value = serde_json::from_slice(&bytes)?;
    if manifest["metadata"]["rows"] != selection["rows"]
        || manifest["metadata"]["dataset_revision"] != config["thoughtworks"]["dataset"]["revision"]
    {
        return Err("competitive prompt provenance changed".into());
    }
    let prompts = manifest["prompts"].as_array().ok_or("prompts missing")?;
    let expected = selection["families"]
        .as_u64()
        .unwrap()
        .checked_mul(selection["requests_per_family"].as_u64().unwrap())
        .ok_or("prompt count overflow")?;
    if prompts.len() as u64 != expected
        || prompts
            .iter()
            .any(|row| row["prompt"].as_str().is_none_or(str::is_empty))
    {
        return Err("competitive prompt cohort changed or empty".into());
    }
    for fraction in config["thoughtworks"]["warm_fractions"]
        .as_array()
        .filter(|fractions| !fractions.is_empty())
        .ok_or("warm prefix fractions required")?
    {
        if !fraction
            .as_f64()
            .is_some_and(|value| value.is_finite() && value > 0.0 && value < 1.0)
        {
            return Err("warm prefix fractions must be finite between zero and one".into());
        }
    }
    let count = usize::try_from(
        input.cell["prompt_count"]
            .as_u64()
            .ok_or("prompt count missing")?,
    )?;
    let concurrency = usize::try_from(
        input.cell["concurrency"]
            .as_u64()
            .ok_or("concurrency missing")?,
    )?;
    if count > prompts.len() {
        return Err("planned prompt wave exceeds manifest".into());
    }
    Ok(Admitted {
        output_tokens: input.cell["output_tokens"]
            .as_u64()
            .ok_or("output budget missing")?,
        config,
        prompts: prompts[..count].to_vec(),
        concurrency,
    })
}
async fn measure(
    input: &Input,
    admitted: &Admitted,
    deadline: Instant,
    cancellation: Cancellation,
    file: &mut std::fs::File,
) -> DynResult<Value> {
    readiness(input, deadline, &cancellation).await?;
    let peer = Exchange {
        base: &input.base_url,
        model: &input.served_model,
        tokens: admitted.output_tokens,
        deadline,
        seconds: input.request_timeout_seconds,
        cancellation: &cancellation,
    };
    let mut measured = Vec::new();
    let mut warm_passed = true;
    let mut measured_seconds = 0.0;
    for (group, prompts) in admitted.prompts.chunks(admitted.concurrency).enumerate() {
        for fraction in admitted.config["thoughtworks"]["warm_fractions"]
            .as_array()
            .unwrap()
        {
            for (local, item) in prompts.iter().enumerate() {
                let prompt = prefix(item["prompt"].as_str().unwrap(), fraction.as_f64().unwrap());
                let row = exchange(
                    &peer,
                    item,
                    &prompt,
                    group * admitted.concurrency + local,
                    &format!("warm-{:.0}", fraction.as_f64().unwrap() * 100.0),
                )
                .await;
                warm_passed &= row["error"].is_null();
                append(file, &row)?;
                check(deadline, &cancellation)?;
            }
        }
        let started = Instant::now();
        let mut tasks = tokio::task::JoinSet::new();
        for (local, item) in prompts.iter().enumerate() {
            let base = input.base_url.clone();
            let served = input.served_model.clone();
            let item = item.clone();
            let cancellation = cancellation.clone();
            let index = group * admitted.concurrency + local;
            let tokens = admitted.output_tokens;
            let seconds = input.request_timeout_seconds;
            tasks.spawn(async move {
                let peer = Exchange {
                    base: &base,
                    model: &served,
                    tokens,
                    deadline,
                    seconds,
                    cancellation: &cancellation,
                };
                exchange(
                    &peer,
                    &item,
                    item["prompt"].as_str().unwrap(),
                    index,
                    "measured",
                )
                .await
            });
        }
        let mut rows = std::collections::BTreeMap::new();
        let mut join_failure = None;
        while let Some(result) = tasks.join_next().await {
            match result {
                Ok(row) => {
                    rows.insert(row["request_index"].as_u64().unwrap(), row);
                }
                Err(error) => join_failure = Some(error),
            }
        }
        if let Some(error) = join_failure {
            return Err(error.into());
        }
        measured_seconds += started.elapsed().as_secs_f64();
        for row in rows.into_values() {
            append(file, &row)?;
            measured.push(row);
        }
        check(deadline, &cancellation)?;
    }
    let passed = warm_passed
        && measured.len() == admitted.prompts.len()
        && measured.iter().all(|row| row["error"].is_null());
    let tokens: u64 = measured
        .iter()
        .filter(|row| row["error"].is_null())
        .map(|row| row["completion_tokens"].as_u64().unwrap_or(0))
        .sum();
    Ok(
        json!({"schema_version":1,"scope":"competitive_request_worker","completed":true,"passed":passed,"cell":input.cell,"config_sha256":input.config_sha256,"launch_provenance":input.launch_provenance,"launch_provenance_scope":"declarations_from_launch_owner_not_attestation","requests":measured.len(),"successful_requests":measured.iter().filter(|row|row["error"].is_null()).count(),"measured_wall_seconds":measured_seconds,"completion_tokens":tokens,"prompt_manifest_sha256":admitted.config["thoughtworks"]["selection"]["manifest_sha256"]}),
    )
}
fn prefix(prompt: &str, fraction: f64) -> String {
    let length = prompt.chars().count();
    let take = ((length as f64 * fraction).floor() as usize).max(1);
    prompt.chars().take(take).collect()
}
fn append(file: &mut std::fs::File, row: &Value) -> DynResult<()> {
    serde_json::to_writer(&mut *file, row)?;
    file.write_all(b"\n")?;
    file.flush()?;
    Ok(())
}
fn check(deadline: Instant, cancellation: &Cancellation) -> DynResult<()> {
    if cancellation.is_cancelled() {
        return Err("competitive worker interrupted".into());
    }
    if Instant::now() >= deadline {
        return Err("competitive cell deadline exceeded".into());
    }
    Ok(())
}
async fn readiness(input: &Input, deadline: Instant, cancellation: &Cancellation) -> DynResult<()> {
    loop {
        check(deadline, cancellation)?;
        let budget = Duration::from_secs(2).min(deadline.saturating_duration_since(Instant::now()));
        if let Some(bytes) =
            ready_get(&format!("{}/models", input.base_url), budget, cancellation).await?
        {
            let document: Value = serde_json::from_slice(&bytes)?;
            if document["data"]
                .as_array()
                .is_some_and(|rows| rows.iter().any(|row| row["id"] == input.served_model))
            {
                return Ok(());
            }
        }
        tokio::time::sleep(
            Duration::from_millis(50).min(deadline.saturating_duration_since(Instant::now())),
        )
        .await;
    }
}
struct Exchange<'a> {
    base: &'a str,
    model: &'a str,
    tokens: u64,
    deadline: Instant,
    seconds: u64,
    cancellation: &'a Cancellation,
}
async fn exchange(
    peer: &Exchange<'_>,
    item: &Value,
    prompt: &str,
    index: usize,
    phase: &str,
) -> Value {
    let Exchange {
        base,
        model,
        tokens,
        deadline,
        seconds,
        cancellation,
    } = peer;
    let body = json!({"model":model,"messages":[{"role":"user","content":prompt}],"max_tokens":tokens,"min_tokens":tokens,"ignore_eos":true,"temperature":0,"seed":42,"stream":true,"stream_options":{"include_usage":true}});
    let budget =
        Duration::from_secs(*seconds).min(deadline.saturating_duration_since(Instant::now()));
    let cancelled = async {
        while !cancellation.is_cancelled() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    };
    let result = tokio::select! { ()=cancelled=>Err("competitive request interrupted".to_owned()), result=tokio::time::timeout(budget,crate::automation::openai_exchange::request(base,&body,false))=>match result { Ok(Ok(evidence)) if evidence.completion_tokens==*tokens=>Ok(evidence), Ok(Ok(_))=>Err("completion token count differs from request".into()), Ok(Err(error))=>Err(error.to_string()), Err(_)=>Err("competitive request deadline exceeded".into()) } };
    let mut row = json!({"request_index":index,"phase":phase,"family":item.get("family"),"prompt_sha256":hex::encode(Sha256::digest(prompt.as_bytes())),"requested_completion_tokens":tokens});
    match result {
        Ok(evidence) => {
            row["prompt_tokens"] = json!(evidence.prompt_tokens);
            row["cached_prompt_tokens"] = json!(evidence.cached_tokens);
            row["new_prompt_tokens"] = json!(evidence.prompt_tokens - evidence.cached_tokens);
            row["completion_tokens"] = json!(evidence.completion_tokens);
            row["content_sha256"] = json!(evidence.content_sha256);
            row["ttft_seconds"] = json!(evidence.ttft_seconds);
            row["elapsed_seconds"] = json!(evidence.elapsed_seconds);
            row["error"] = Value::Null;
        }
        Err(error) => row["error"] = json!(error),
    }
    row
}
// Both competitive request workers race the actual readiness GET against interruption.
// The losing HTTP future is dropped in this owning current-thread runtime.
pub(super) async fn ready_get(
    url: &str,
    budget: Duration,
    cancellation: &Cancellation,
) -> DynResult<Option<Vec<u8>>> {
    let cancelled = async {
        while !cancellation.is_cancelled() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    };
    tokio::select! {
        ()=cancelled=>Err("competitive readiness interrupted".into()),
        result=tokio::time::timeout(budget,crate::automation::openai_exchange::get(url))=>Ok(match result{Ok(Ok(bytes))=>Some(bytes),Ok(Err(_))|Err(_)=>None})
    }
}
