//! Attach-only cache matrix. Endpoints and runtime/cache configuration remain operator-owned.
mod observation;
mod transport;
use super::command_interrupt::Interrupt;
use crate::{command::DynResult, process::Cancellation, repository::check_args::Grammar};
use observation::{Observation, Row};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    io::Write,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
const GRAMMAR: Grammar = Grammar {
    usage: "cargo xtool automation openai-cache-matrix --llama-base-url URL|--llama-cold-base-url URL --llama-warm-base-url URL --skippy-cold-base-url URL --skippy-warm-base-url URL --model ID [--api-key KEY] [--pattern exact|shared-prefix] [--prefix-repetitions N] [--max-tokens N] [--repeats N] [--timeout SECONDS] [--allow-missing-warm-cache] [--output-dir PATH]",
    values: &[
        "--llama-base-url",
        "--llama-cold-base-url",
        "--llama-warm-base-url",
        "--skippy-cold-base-url",
        "--skippy-warm-base-url",
        "--model",
        "--api-key",
        "--pattern",
        "--prefix-repetitions",
        "--max-tokens",
        "--repeats",
        "--timeout",
        "--output-dir",
    ],
    flags: &["--allow-missing-warm-cache", "--help"],
};
struct Options {
    urls: [String; 4],
    model: String,
    api_key: Option<String>,
    prefix: String,
    measured: String,
    warmup: String,
    pattern: String,
    repeats: u64,
    max_tokens: u64,
    timeout: Duration,
    whole_budget: Duration,
    prefix_repetitions: u64,
    allow_missing: bool,
    output: PathBuf,
}
fn endpoint(value: &str) -> DynResult<String> {
    let url = url::Url::parse(value)?;
    if !matches!(url.scheme(), "http" | "https")
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || url.host().is_none()
    {
        return Err(
            "cache matrix requires credential-free HTTP/HTTPS endpoints without query or fragment"
                .into(),
        );
    }
    Ok(value.trim_end_matches('/').to_owned())
}
fn authorization(value: Option<&str>) -> DynResult<Option<String>> {
    let Some(value) = value.filter(|value| !value.is_empty()) else {
        return Ok(None);
    };
    if value.len() > 4096 || value.chars().any(char::is_control) {
        return Err("cache matrix authorization must be bounded single-line text".into());
    }
    Ok(Some(value.to_owned()))
}
fn options(args: &[String]) -> DynResult<Option<Options>> {
    let parsed = GRAMMAR.parse(args).map_err(|error| format!("{error:?}"))?;
    if parsed.flag("--help") {
        println!("{}", GRAMMAR.usage);
        return Ok(None);
    }
    if !parsed.positionals.is_empty() {
        return Err("cache matrix accepts named arguments".into());
    }
    let required = |name| parsed.last(name).ok_or_else(|| format!("missing {name}"));
    let baseline = parsed.last("--llama-base-url");
    let cold = parsed
        .last("--llama-cold-base-url")
        .or(baseline)
        .ok_or("missing cold native endpoint")?;
    let warm = parsed
        .last("--llama-warm-base-url")
        .or(baseline)
        .ok_or("missing warm native endpoint")?;
    let urls = [
        endpoint(cold)?,
        endpoint(required("--skippy-cold-base-url")?)?,
        endpoint(warm)?,
        endpoint(required("--skippy-warm-base-url")?)?,
    ];
    let model = required("--model")?.to_owned();
    if model.is_empty() || model.len() > 4096 {
        return Err("model id must be nonempty/bounded".into());
    }
    let prefix_repetitions: u64 = parsed
        .last("--prefix-repetitions")
        .unwrap_or("256")
        .parse()?;
    let repeats: u64 = parsed.last("--repeats").unwrap_or("3").parse()?;
    let max_tokens: u64 = parsed.last("--max-tokens").unwrap_or("16").parse()?;
    let timeout: f64 = parsed.last("--timeout").unwrap_or("120").parse()?;
    if !(1..=1024).contains(&prefix_repetitions)
        || !(1..=100).contains(&repeats)
        || !(1..=4096).contains(&max_tokens)
        || !timeout.is_finite()
        || !(0.01..=3600.0).contains(&timeout)
        || (repeats * 4 + 2) as f64 * timeout > 86400.0
    {
        return Err("cache matrix request/count/deadline bounds refused".into());
    }
    let pattern = parsed.last("--pattern").unwrap_or("exact").to_owned();
    if !["exact", "shared-prefix"].contains(&pattern.as_str()) {
        return Err("invalid pattern".into());
    }
    let measured = "Measure the reusable prefix and answer with the word measured.".to_owned();
    let warmup = if pattern == "exact" {
        measured.clone()
    } else {
        "Warm the reusable prefix and answer with the word warmup.".into()
    };
    let api_key = authorization(parsed.last("--api-key"))?;
    let output = Path::new(
        parsed
            .last("--output-dir")
            .unwrap_or("target/skippy-openai-cache-matrix"),
    );
    let output = std::env::current_dir()?.join(output);
    if output
        .components()
        .any(|component| matches!(component, std::path::Component::ParentDir))
        || output.file_name().is_none()
    {
        return Err("output must have a regular nontraversing leaf".into());
    }
    let output = canonical_output(output)?;
    Ok(Some(Options {
        urls,
        model,
        api_key,
        prefix: "Skippy prompt cache benchmark shared prefix. "
            .repeat(usize::try_from(prefix_repetitions)?),
        measured,
        warmup,
        pattern,
        repeats,
        max_tokens,
        timeout: Duration::from_secs_f64(timeout),
        whole_budget: Duration::from_secs_f64((repeats * 4 + 2) as f64 * timeout + 1.0),
        prefix_repetitions,
        allow_missing: parsed.flag("--allow-missing-warm-cache"),
        output,
    }))
}
fn canonical_output(path: PathBuf) -> DynResult<PathBuf> {
    let leaf = path.file_name().ok_or("output leaf absent")?.to_os_string();
    let mut parent = path.parent().ok_or("output parent absent")?.to_path_buf();
    let mut missing = Vec::new();
    loop {
        match std::fs::symlink_metadata(&parent) {
            Ok(metadata) if metadata.is_dir() => break,
            Ok(_) => return Err("existing output ancestor must be a regular directory".into()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                missing.push(
                    parent
                        .file_name()
                        .ok_or("output ancestor absent")?
                        .to_os_string(),
                );
                if !parent.pop() {
                    return Err("output ancestor absent".into());
                }
            }
            Err(error) => return Err(error.into()),
        }
    }
    let mut output = parent.canonicalize()?;
    for component in missing.into_iter().rev() {
        output.push(component);
    }
    output.push(leaf);
    Ok(output)
}
async fn one(
    opts: &Options,
    index: usize,
    tail: &str,
    run: Option<u64>,
    token: &Cancellation,
    matrix_deadline: Instant,
) -> Result<Observation, String> {
    let deadline = matrix_deadline.min(Instant::now() + opts.timeout);
    transport::observe(opts, index, tail, run, token, deadline).await
}
fn atomic(root: &Path, name: &str, bytes: &[u8]) -> DynResult<()> {
    let output = root.join(name);
    if std::fs::symlink_metadata(&output).is_ok_and(|m| !m.is_file()) {
        return Err("cache matrix output leaf must be regular".into());
    }
    let mut file = tempfile::NamedTempFile::new_in(root)?;
    file.write_all(bytes)?;
    file.as_file().sync_all()?;
    file.persist(output)?;
    Ok(())
}
fn publish(
    opts: &Options,
    rows: &[Row],
    current: Value,
    status: &str,
    error: Option<&str>,
) -> DynResult<()> {
    let encoded = json!({"schema_version":1,"model":opts.model,"pattern":opts.pattern,"prefix_repetitions":opts.prefix_repetitions,"max_tokens":opts.max_tokens,"repeats":opts.repeats,"status":status,"error":error,"rows":rows,"incomplete_row":current,"endpoint_custody":"declared operator-owned, configuration not attested","endpoints":opts.urls,"shared_prefix_sha256":hex::encode(Sha256::digest(opts.prefix.as_bytes())),"warmup_tail_sha256":hex::encode(Sha256::digest(opts.warmup.as_bytes())),"measured_tail_sha256":hex::encode(Sha256::digest(opts.measured.as_bytes()))});
    atomic(
        &opts.output,
        "cache-matrix.json",
        &serde_json::to_vec_pretty(&encoded)?,
    )?;
    atomic(
        &opts.output,
        "cache-matrix.md",
        observation::markdown(rows).as_bytes(),
    )
}
async fn execute(opts: &Options, token: &Cancellation) -> DynResult<()> {
    let matrix_deadline = Instant::now() + opts.whole_budget;
    let names = ["cold native", "cold Skippy", "warm native", "warm Skippy"];
    let mut rows = Vec::new();
    publish(opts, &rows, Value::Null, "running", None)?;
    for (index, name) in names.into_iter().enumerate() {
        let mut warmup = None;
        let mut runs = Vec::new();
        let result = async {
            if index >= 2 {
                warmup = Some(one(opts, index, &opts.warmup, None, token, matrix_deadline).await?);
                publish(
                    opts,
                    &rows,
                    json!({"name":name,"warmup":warmup,"runs":runs}),
                    "running",
                    None,
                )
                .map_err(|_| "partial evidence publication failed")?;
            }
            for count in 1..=opts.repeats {
                runs.push(
                    one(
                        opts,
                        index,
                        &opts.measured,
                        Some(count),
                        token,
                        matrix_deadline,
                    )
                    .await?,
                );
                publish(
                    opts,
                    &rows,
                    json!({"name":name,"warmup":warmup,"runs":runs}),
                    "running",
                    None,
                )
                .map_err(|_| "partial evidence publication failed")?;
            }
            Ok::<_, String>(())
        }
        .await;
        if let Err(error) = result {
            publish(
                opts,
                &rows,
                json!({"name":name,"warmup":warmup,"runs":runs}),
                if token.is_cancelled() {
                    "cancelled"
                } else {
                    "failed"
                },
                Some(&error),
            )?;
            return Err(error.into());
        }
        rows.push(Row::summarize(
            name,
            index % 2 == 1,
            index >= 2,
            warmup,
            runs,
        ));
    }
    let missing = rows
        .iter()
        .skip(2)
        .any(|row| row.max_cached_tokens.is_none_or(|n| n == 0));
    let status = if missing && !opts.allow_missing {
        "warm-cache-proof-failed"
    } else {
        "completed"
    };
    publish(opts, &rows, Value::Null, status, None)?;
    println!("{}", observation::markdown(&rows));
    println!(
        "Wrote {}\nWrote {}",
        opts.output.join("cache-matrix.json").display(),
        opts.output.join("cache-matrix.md").display()
    );
    if missing && !opts.allow_missing {
        Err("warm cache proof failed; observed results retained".into())
    } else {
        Ok(())
    }
}
fn prepare_parent(path: &Path) -> DynResult<()> {
    let mut current = PathBuf::new();
    for component in path.components() {
        current.push(component.as_os_str());
        match std::fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.is_dir() => (),
            Ok(_) => return Err("output parent components must be regular directories".into()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                std::fs::create_dir(&current)?
            }
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}
pub(crate) fn run(args: &[String]) -> DynResult<()> {
    let Some(opts) = options(args)? else {
        return Ok(());
    };
    prepare_parent(opts.output.parent().ok_or("output parent absent")?)?;
    std::fs::create_dir(&opts.output)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let interrupt = Interrupt::install()?;
    let result = runtime.block_on(execute(&opts, &interrupt.cancellation()));
    let finished = interrupt.finish();
    if finished.is_err() {
        let mut report: Value =
            serde_json::from_slice(&std::fs::read(opts.output.join("cache-matrix.json"))?)?;
        report["status"] = json!("cancelled");
        report["error"] = json!("command interruption observed at finalization");
        atomic(
            &opts.output,
            "cache-matrix.json",
            &serde_json::to_vec_pretty(&report)?,
        )?;
        return Err("cache matrix cancelled; endpoint processes remain operator-owned".into());
    }
    result
}

#[cfg(test)]
mod authorization_tests {
    use super::*;
    #[test]
    fn matrix_authorization_normalizes_empty_and_refuses_controls_without_echo() {
        assert!(authorization(None).unwrap().is_none());
        assert!(authorization(Some("")).unwrap().is_none());
        for control in ['\r', '\n', '\0', '\t'] {
            let value = format!("owned{control}value");
            assert_eq!(
                authorization(Some(&value)).unwrap_err().to_string(),
                "cache matrix authorization must be bounded single-line text"
            );
        }
        assert!(authorization(Some(&"a".repeat(4097))).is_err());
        assert_eq!(
            authorization(Some("bounded-value")).unwrap().as_deref(),
            Some("bounded-value")
        );
    }
}
