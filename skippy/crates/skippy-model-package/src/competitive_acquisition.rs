//! Immutable competitive inputs acquisition and native tokenizer semantic export.
#[path = "competitive_acquisition/acquire.rs"]
pub(crate) mod acquire;
#[path = "competitive_acquisition/contract.rs"]
pub(crate) mod contract;
#[cfg(test)]
#[path = "competitive_acquisition/download_tests.rs"]
mod download_tests;
#[path = "competitive_acquisition/export.rs"]
mod export;
#[cfg(all(test, unix))]
#[path = "competitive_acquisition/full_chain.rs"]
pub(crate) mod full_chain;
#[cfg(all(test, unix))]
#[path = "competitive_acquisition/full_frontend.rs"]
mod full_frontend;
#[path = "competitive_acquisition/json.rs"]
pub(crate) mod json;
#[path = "competitive_acquisition/listing.rs"]
pub(crate) mod listing;
#[path = "competitive_acquisition/local.rs"]
mod local;
#[path = "competitive_acquisition/read.rs"]
mod read;
#[cfg(test)]
#[path = "competitive_acquisition/tests.rs"]
mod tests;
#[path = "competitive_acquisition/verify.rs"]
mod verify;
use anyhow::{Result, bail};
pub use contract::{Case, Request};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    io::Write as _,
    path::Path,
    time::{Duration, Instant},
};
pub(super) fn publish(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = tempfile::NamedTempFile::new_in(
        path.parent()
            .ok_or_else(|| anyhow::anyhow!("publication parent"))?,
    )?;
    file.write_all(bytes)?;
    file.as_file().sync_all()?;
    file.persist_noclobber(path)
        .map_err(|_| anyhow::anyhow!("fresh output publication refused"))?;
    Ok(())
}
pub async fn execute(input: &Request, deadline: Instant, evidence: &mut Value) -> Result<()> {
    execute_using(input, deadline, evidence, None).await
}
struct NativeClients {
    client: hf_hub::HFClient,
    listing: listing::Listing,
}
async fn execute_using(
    input: &Request,
    deadline: Instant,
    evidence: &mut Value,
    supplied: Option<NativeClients>,
) -> Result<()> {
    input.validate()?;
    contract::check(deadline)?;
    let mut fd = read::open(&input.config, 1024 * 1024, false)?;
    let bytes = read::read(&mut fd, 1024 * 1024)?;
    if contract::digest(&bytes) != input.config_sha256 {
        bail!("config content pin mismatch");
    }
    let config: Value = json::unique(&bytes)?;
    let selected = select(&config, input)?;
    let token = match &input.credential_file {
        Some(p) => {
            let mut f = read::open(p, 4096, true)?;
            let raw = read::read(&mut f, 4096)?;
            let value = std::str::from_utf8(&raw)?.trim().to_string();
            if value.is_empty() || value.chars().any(char::is_control) {
                bail!("credential grammar refused");
            }
            value
        }
        None => String::new(),
    };
    // Own the fresh root only after complete selection and pin admission.
    std::fs::create_dir(&input.output_directory)?;
    evidence["output_owned"] = json!(true);
    let NativeClients { client, listing } = match supplied {
        Some(c) => c,
        None => NativeClients {
            listing: listing::Listing::new(token.clone())?,
            client: acquire::client(token, &input.output_directory.join("private-cache"))?,
        },
    };
    let source_root = input.output_directory.join("sources");
    std::fs::create_dir(&source_root)?;
    let tokenizers = input.output_directory.join("tokenizers");
    std::fs::create_dir(&tokenizers)?;
    for row in selected {
        contract::check(deadline)?;
        let key = row["key"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("model key"))?;
        let model_root = input.output_directory.join("models").join(key);
        let artifact = acquire::artifact(
            &client,
            row,
            false,
            &model_root,
            input.maximum_bytes,
            deadline,
        )
        .await?;
        let mut family = json!({"key":key,"model":artifact,"source":null,"vllm_config":null,"export":null,"tokenizer_skipped":input.skip_tokenizers,"vllm_config_skipped":input.skip_vllm_configs});
        let source = &row["vllm_hf_config"];
        if !input.skip_tokenizers {
            let root = source_root.join(key);
            let snapshot = acquire::snapshot(
                &client,
                &listing,
                acquire::Snapshot {
                    repo: source["repo"].as_str().unwrap(),
                    revision: source["revision"].as_str().unwrap(),
                    output: &root,
                    complete: key == "granite-h1-hybrid",
                    maximum: input.maximum_bytes,
                },
                deadline,
            )
            .await?;
            let pins: BTreeMap<String, String> = serde_json::from_value(snapshot["files"].clone())?;
            let accepted = input
                .export_sha256
                .get(key)
                .ok_or_else(|| anyhow::anyhow!("explicit derived export pin required"))?;
            let export = if key == "granite-h1-hybrid" {
                export::granite(&root, &tokenizers.join(key), &pins, accepted, deadline)?
            } else {
                export::fast(
                    &root,
                    &tokenizers.join(key),
                    accepted,
                    &input.semantic_cases[key],
                    deadline,
                )?
            };
            acquire::recheck(&root, &pins, deadline)?;
            if local::pins(&root, deadline)? != pins {
                bail!("snapshot roster drift");
            }
            family["source"] = snapshot;
            family["export"] = export;
            family["benchmark_config_tree_sha256"] = row["tokenizer_sha256"].clone();
            family["benchmark_pin_matches"] =
                json!(row["tokenizer_sha256"].as_str() == Some(accepted));
        }
        if !input.skip_vllm_configs {
            let configuration = json!({"repo":source["repo"],"revision":source["revision"],"filename":"config.json","sha256":source["sha256"]});
            let configuration = acquire::artifact(
                &client,
                &configuration,
                false,
                &input.output_directory.join("vllm-configs").join(key),
                input.maximum_bytes,
                deadline,
            )
            .await?;
            family["vllm_config"] = configuration;
        }
        evidence["families"]
            .as_array_mut()
            .ok_or_else(|| anyhow::anyhow!("receipt roster"))?
            .push(family);
        checkpoint(&input.output_directory.join("progress.json"), evidence)?;
    }
    let mut derived_config = config.clone();
    for row in evidence["families"].as_array().unwrap() {
        for model in derived_config["models"].as_array_mut().unwrap() {
            if model["key"] == row["key"] && !row["export"].is_null() {
                model["tokenizer_sha256"] = row["export"]["tree_sha256"].clone();
            }
        }
    }
    let derived_bytes = serde_json::to_vec_pretty(&derived_config)?;
    publish(
        &input.output_directory.join("derived-benchmark-config.json"),
        &derived_bytes,
    )?;
    evidence["derived_config_sha256"] = json!(contract::digest(&derived_bytes));
    evidence["trajectory_manifest_created"] = json!(false);
    if !input.skip_dataset {
        let dataset = acquire::artifact(
            &client,
            &config["thoughtworks"]["dataset"],
            true,
            &input.output_directory.join("thoughtworks"),
            input.maximum_bytes,
            deadline,
        )
        .await?;
        evidence["dataset"] = dataset;
    }
    evidence["dataset_skipped"] = json!(input.skip_dataset);
    if read::read(&mut fd, 1024 * 1024)? != bytes {
        bail!("config changed during acquisition");
    }
    contract::check(deadline)
}
fn select<'a>(config: &'a Value, input: &Request) -> Result<Vec<&'a Value>> {
    let models = config["models"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("models required"))?;
    let mut wanted = std::collections::BTreeSet::new();
    for k in &input.model_keys {
        if !wanted.insert(k.as_str()) || !models.iter().any(|r| r["key"].as_str() == Some(k)) {
            bail!("unknown/duplicate family selection");
        }
    }
    let rows: Vec<_> = models
        .iter()
        .filter(|r| wanted.is_empty() || wanted.contains(r["key"].as_str().unwrap_or("")))
        .collect();
    let mut found = std::collections::BTreeSet::new();
    for row in &rows {
        let key = row["key"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("model key"))?;
        if ![
            "llama32-dense",
            "deepseek-v2-moe",
            "falcon-h1-recurrent",
            "granite-h1-hybrid",
        ]
        .contains(&key)
            || !found.insert(key)
        {
            bail!("configured family refused");
        }
        let source = &row["vllm_hf_config"];
        for field in ["repo", "revision", "sha256"] {
            if source[field].as_str().is_none() {
                bail!("pinned tokenizer snapshot required");
            }
        }
        if !contract::pin(source["revision"].as_str().unwrap(), 40)
            || !contract::pin(source["sha256"].as_str().unwrap(), 64)
            || (!input.skip_tokenizers
                && input
                    .export_sha256
                    .get(key)
                    .is_none_or(|p| !contract::pin(p, 64)))
        {
            bail!("source/derived pin refused");
        }
        if !input.skip_tokenizers
            && key != "granite-h1-hybrid"
            && input.semantic_cases.get(key).is_none_or(Vec::is_empty)
        {
            bail!("native semantic case roster required");
        }
    }
    if rows.is_empty() {
        bail!("empty family roster");
    }
    Ok(rows)
}
pub fn run(args: &[String]) -> Result<()> {
    run_using(args, None)
}
fn run_using(args: &[String], supplied: Option<NativeClients>) -> Result<()> {
    if let [verb, input, path, phase, label] = args
        && verb == "verify-acquired"
        && input == "--input"
        && phase == "--phase"
    {
        return verify::run(Path::new(path), label);
    }
    if args.len() == 3 && args[0] == "export-local" && args[1] == "--input" {
        return local::run(Path::new(&args[2]));
    }
    if args.len() != 2 || args[0] != "--input" {
        bail!("usage: model-package-competitive-inputs --input ABS");
    }
    let path = Path::new(&args[1]);
    let mut file = read::open(path, 1024 * 1024, false)?;
    let bytes = read::read(&mut file, 1024 * 1024)?;
    let input: Request = serde_json::from_slice(&bytes)?;
    input.validate()?;
    #[cfg(unix)]
    let signal = crate::snapshot_promotion::local_publisher::SignalLatch::install()?;
    let deadline = Instant::now()
        .checked_add(Duration::from_secs(input.timeout_seconds))
        .ok_or_else(|| anyhow::anyhow!("deadline overflow"))?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let mut evidence = json!({"schema_version":1,"request_sha256":contract::digest(&serde_json::to_vec(&input)?),"request_transport_sha256":contract::digest(&bytes),"status":"FAILED","output_owned":false,"families":[],"dataset":null,"error":null,"real_family_qualified":false});
    let result = runtime.block_on(async {
        let work = execute_using(&input, deadline, &mut evidence, supplied);
        let timer = tokio::time::sleep_until(tokio::time::Instant::from_std(deadline));
        let cancellation = cancelled();
        futures::pin_mut!(work, timer, cancellation);
        match futures::future::select(work, futures::future::select(timer, cancellation)).await {
            futures::future::Either::Left((r, remaining)) => {
                use futures::FutureExt as _;
                let r = r.and_then(|()| {
                    if read::read(&mut file, 1024 * 1024)? != bytes {
                        bail!("input custody changed");
                    }
                    contract::check(deadline)
                });
                if remaining.now_or_never().is_some() {
                    Err(anyhow::anyhow!("terminal deadline/cancellation refusal"))
                } else {
                    r.and_then(|()| contract::check(deadline))
                }
            }
            futures::future::Either::Right(_) => {
                Err(anyhow::anyhow!("acquisition deadline/cancellation"))
            }
        }
    });
    #[cfg(unix)]
    let result = terminal(result, deadline, || signal.cancelled());
    evidence["status"] = json!(if result.is_ok() {
        "ACQUIRED_EXPORTED"
    } else {
        "FAILED"
    });
    evidence["error"] = result
        .as_ref()
        .err()
        .map_or(Value::Null, |e| json!(e.to_string()));
    if evidence["output_owned"] == json!(true) {
        publish(
            &input.output_directory.join("acquisition.json"),
            &serde_json::to_vec_pretty(&evidence)?,
        )?;
    }
    result
}

pub(crate) async fn cancelled() -> Result<()> {
    #[cfg(unix)]
    {
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
        let control = tokio::signal::ctrl_c();
        futures::pin_mut!(control);
        match futures::future::select(control, Box::pin(term.recv())).await {
            futures::future::Either::Left((r, _)) => r?,
            futures::future::Either::Right((None, _)) => bail!("signal observation failed"),
            futures::future::Either::Right((Some(()), _)) => (),
        };
    }
    #[cfg(not(unix))]
    tokio::signal::ctrl_c().await?;
    Ok(())
}

fn checkpoint(path: &Path, value: &Value) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value)?;
    if bytes.len() > 8 * 1024 * 1024 {
        bail!("acquisition progress bound");
    }
    let mut file = tempfile::NamedTempFile::new_in(path.parent().unwrap())?;
    file.write_all(&bytes)?;
    file.as_file().sync_all()?;
    file.persist(path)
        .map_err(|_| anyhow::anyhow!("owned progress publication refused"))?;
    Ok(())
}

#[cfg(unix)]
fn terminal(result: Result<()>, deadline: Instant, cancelled: impl FnOnce() -> bool) -> Result<()> {
    result?;
    if cancelled() {
        bail!("terminal cancellation refused");
    }
    contract::check(deadline)
}
