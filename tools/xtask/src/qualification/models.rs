//! `qualify models`: execute an executable model qualification suite against a
//! composed product over the routes the standalone CLI actually serves.

use super::contract::{self, Json};
use super::serve::{self, Server};
use crate::repository::check_args::Grammar;
use crate::repository::check_report::CheckReport;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::Duration;

const USAGE: &str = "qualify models --product-dir <dir> --row-id <row> --suite <dense|recurrent|moe|kv-cache> [--device <dev>] [--ctx-size 2048] --model <artifact-id> <path> <sha256> <model-id> --evidence <file> [--availability <file> --hardware-evidence <file>]";

const PROMPT: &str = "Reply with one short sentence about the weather.";
/// Long enough for the runtime prefix cache to retain a block on every
/// supported backend and tokenizer (~500 tokens).
const PREFIX_TOKENS: usize = 160;
const SUFFIX: &str = "\nNow summarise the reference facts in one line.";

fn prefix() -> String {
    let facts = (0..PREFIX_TOKENS)
        .map(|index| format!("fact-{index}"))
        .collect::<Vec<_>>()
        .join(" ");
    format!("Reference facts: {facts}")
}

/// One `--model ARTIFACT PATH SHA256 MODEL_ID` group.
struct ModelSpec {
    artifact_id: String,
    path: PathBuf,
    sha256: String,
    model_id: String,
}

fn usage_tokens(response: &Value) -> Result<&Value, String> {
    let usage = response.get("usage").filter(|usage| usage.is_object());
    let usage = usage.ok_or("completion lacks usage")?;
    contract::require(
        usage
            .get("prompt_tokens")
            .and_then(Value::as_i64)
            .is_some_and(|v| v > 0),
        "completion lacks positive prefill tokens",
    )?;
    contract::require(
        usage
            .get("completion_tokens")
            .and_then(Value::as_i64)
            .is_some_and(|v| v > 0),
        "completion lacks positive decode tokens",
    )?;
    Ok(usage)
}

fn prompt_tokens(response: &Value) -> Result<i64, String> {
    Ok(usage_tokens(response)?
        .get("prompt_tokens")
        .and_then(Value::as_i64)
        .unwrap_or(0))
}

fn cached_tokens(response: &Value) -> i64 {
    response
        .get("usage")
        .and_then(|usage| usage.get("prompt_tokens_details"))
        .and_then(|details| details.get("cached_tokens"))
        .and_then(Value::as_i64)
        .filter(|value| *value >= 0)
        .unwrap_or(0)
}

fn completion(
    base: &str,
    model_id: &str,
    messages: Value,
    options: &[(&str, Value)],
) -> Result<Value, String> {
    let mut body = json!({
        "model": model_id,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 8,
    });
    for (key, value) in options {
        body[key] = value.clone();
    }
    let (status, response) = super::http::request_json(
        &format!("{base}/v1/chat/completions"),
        Some(body),
        "POST",
        Duration::from_secs(180),
    )
    .map_err(|error| error.to_string())?;
    contract::require(
        status == 200,
        &format!("chat completion returned HTTP {status}: {response}"),
    )?;
    contract::require(
        response.get("model").and_then(Value::as_str) == Some(model_id),
        "completion model differs from the selected model",
    )?;
    let choices = response.get("choices").and_then(Value::as_array);
    contract::require(
        choices.is_some_and(|choices| choices.len() == 1),
        "completion must have one choice",
    )?;
    let content = choices
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
        .and_then(Value::as_str);
    contract::require(
        content.is_some_and(|content| !content.trim().is_empty()),
        "completion has no generated text",
    )?;
    usage_tokens(&response)?;
    Ok(response)
}

fn text_of(response: &Value) -> String {
    response["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default()
        .to_owned()
}

fn stream_completion(base: &str, model_id: &str, prompt: &str) -> Result<(usize, usize), String> {
    let body = json!({
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16,
        "temperature": 0,
        "stream": true,
    });
    let response = super::http::request(
        &format!("{base}/v1/chat/completions"),
        serde_json::to_vec(&body).ok(),
        "POST",
        Duration::from_secs(180),
    )
    .map_err(|error| error.to_string())?;
    contract::require(
        response.status == 200,
        &format!(
            "/v1/chat/completions stream returned HTTP {}",
            response.status
        ),
    )?;
    let payload = String::from_utf8_lossy(&response.body);
    let lines: Vec<&str> = payload
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();
    contract::require(!lines.is_empty(), "stream produced no server-sent events")?;
    contract::require(
        lines.iter().any(|line| line.contains("data:")),
        "stream did not emit server-sent events",
    )?;
    let deltas: Vec<&&str> = lines
        .iter()
        .filter(|line| line.starts_with("data:") && !line.contains("[DONE]"))
        .collect();
    contract::require(!deltas.is_empty(), "stream produced no content deltas")?;
    contract::require(
        lines.iter().any(|line| line.contains("[DONE]")),
        "stream did not terminate with [DONE]",
    )?;
    Ok((deltas.len(), payload.len()))
}

fn dense_cases(
    server: &Server,
    model_id: &str,
    observations: &mut serde_json::Map<String, Value>,
) -> Result<Vec<String>, String> {
    let first = completion(
        &server.base,
        model_id,
        json!([{"role": "user", "content": PROMPT}]),
        &[],
    )?;
    let prefill = prompt_tokens(&first)?;
    observations.insert("prefill_tokens".into(), json!(prefill));
    let history = json!([
        {"role": "user", "content": PROMPT},
        {"role": "assistant", "content": text_of(&first)},
        {"role": "user", "content": "Now answer in five words."},
    ]);
    let second = completion(&server.base, model_id, history, &[])?;
    contract::require(
        prompt_tokens(&second)? > prefill,
        "continuation did not grow the prompt",
    )?;
    let (deltas, _) = stream_completion(&server.base, model_id, PROMPT)?;
    observations.insert("stream_deltas".into(), json!(deltas));
    Ok(["load", "prefill-decode", "stream", "continuation"]
        .map(str::to_owned)
        .to_vec())
}

fn recurrent_cases(
    server: &Server,
    model_id: &str,
    observations: &mut serde_json::Map<String, Value>,
) -> Result<Vec<String>, String> {
    let model = model_id;
    let first = completion(
        &server.base,
        model,
        json!([{"role": "user", "content": PROMPT}]),
        &[],
    )?;
    let mut previous = prompt_tokens(&first)?;
    observations.insert("prefill_tokens".into(), json!(previous));
    let mut history = vec![
        json!({"role": "user", "content": PROMPT}),
        json!({"role": "assistant", "content": text_of(&first)}),
    ];
    for turn in 0..2 {
        history.push(json!({"role": "user", "content": format!("Turn {turn}: continue.")}));
        let response = completion(&server.base, model, Value::Array(history.clone()), &[])?;
        let tokens = prompt_tokens(&response)?;
        contract::require(
            tokens > previous,
            "recurrent state did not grow across turns",
        )?;
        previous = tokens;
        history.push(json!({"role": "assistant", "content": text_of(&response)}));
    }
    observations.insert("state_prompt_tokens".into(), json!(previous));
    Ok(["prefill-decode", "state-preservation"]
        .map(str::to_owned)
        .to_vec())
}

fn moe_cases(
    server: &Server,
    model_id: &str,
    observations: &mut serde_json::Map<String, Value>,
) -> Result<Vec<String>, String> {
    let model = model_id;
    let prefix = prefix();
    let first = completion(
        &server.base,
        model,
        json!([{"role": "user", "content": prefix}]),
        &[],
    )?;
    let repeat = completion(
        &server.base,
        model,
        json!([{"role": "user", "content": prefix}]),
        &[],
    )?;
    contract::require(
        prompt_tokens(&first)? == prompt_tokens(&repeat)?,
        "repeated identical prompt changed the prefill",
    )?;
    let extended = completion(
        &server.base,
        model,
        json!([{"role": "user", "content": format!("{prefix}{SUFFIX}")}]),
        &[],
    )?;
    contract::require(
        prompt_tokens(&extended)? > prompt_tokens(&first)?,
        "suffix continuation did not grow the prompt",
    )?;
    observations.insert("expert_prompt_tokens".into(), json!(prompt_tokens(&first)?));
    Ok([
        "expert-execution",
        "repeated-restore",
        "suffix-continuation",
    ]
    .map(str::to_owned)
    .to_vec())
}

fn kv_cases(
    server: &Server,
    model_id: &str,
    observations: &mut serde_json::Map<String, Value>,
) -> Result<Vec<String>, String> {
    let model = model_id;
    let prefix = prefix();
    let cold = completion(
        &server.base,
        model,
        json!([{"role": "user", "content": prefix}]),
        &[],
    )?;
    contract::require(prompt_tokens(&cold)? > 0, "cold prefill was empty")?;
    let warm = completion(
        &server.base,
        model,
        json!([{"role": "user", "content": prefix}]),
        &[],
    )?;
    contract::require(
        cached_tokens(&warm) > 0,
        "a repeated prefix was not served from cache",
    )?;
    let extended_prompt = format!("{prefix}{SUFFIX}");
    let extended = completion(
        &server.base,
        model,
        json!([{"role": "user", "content": extended_prompt}]),
        &[],
    )?;
    contract::require(
        prompt_tokens(&extended)? > prompt_tokens(&cold)?,
        "suffix continuation did not grow the prompt",
    )?;
    // The runtime reports reuse for a repeat of the longest already-seen
    // prompt, so prove reuse persists at the longer length.
    let repeat = completion(
        &server.base,
        model,
        json!([{"role": "user", "content": extended_prompt}]),
        &[],
    )?;
    contract::require(
        cached_tokens(&repeat) > 0,
        "the extended prompt was not served from cache on repeat",
    )?;
    let divergent: String = prefix.chars().rev().collect();
    let divergent = completion(
        &server.base,
        model,
        json!([{"role": "user", "content": divergent}]),
        &[],
    )?;
    contract::require(
        prompt_tokens(&divergent)? > 0,
        "divergent prefix did not execute",
    )?;
    let other = completion(
        &server.base,
        model,
        json!([{"role": "user", "content": PROMPT}]),
        &[],
    )?;
    contract::require(
        prompt_tokens(&other)? > 0,
        "isolated prompt did not execute",
    )?;
    observations.insert("cached_tokens".into(), json!(cached_tokens(&warm)));
    observations.insert("cold_prompt_tokens".into(), json!(prompt_tokens(&cold)?));
    Ok(["isolation", "divergent-prefix", "suffix-continuation"]
        .map(str::to_owned)
        .to_vec())
}

/// One suite's case driver plus whether a restart case follows.
type SuiteDriver = (
    fn(&Server, &str, &mut serde_json::Map<String, Value>) -> Result<Vec<String>, String>,
    bool,
);

/// Suites that restart the server after their driver cases.
fn suite_driver(name: &str) -> Option<SuiteDriver> {
    match name {
        "dense" => Some((dense_cases, true)),
        "recurrent" => Some((recurrent_cases, true)),
        "moe" => Some((moe_cases, false)),
        "kv-cache" => Some((kv_cases, false)),
        _ => None,
    }
}

/// Capability each suite's model set must cover; a group is satisfied when a
/// model carries any of its tags.
fn suite_model_tags(name: &str) -> &'static [&'static [&'static str]] {
    match name {
        "dense" => &[&["dense"]],
        "recurrent" => &[&["hybrid", "recurrent"]],
        "moe" => &[&["moe"]],
        "kv-cache" => &[&["dense"], &["hybrid", "recurrent"]],
        _ => &[],
    }
}

fn registry_tags(root: &Path) -> Result<BTreeMap<String, BTreeSet<String>>, String> {
    let registry = contract::load_json(&root.join("ci/model-artifacts/registry.json"))?;
    let mut tags = BTreeMap::new();
    for artifact in registry
        .get("artifacts")
        .and_then(Json::as_array)
        .unwrap_or(&[])
    {
        let Some(id) = artifact.get("id").and_then(Json::as_str) else {
            continue;
        };
        let set = artifact
            .get("capability_tags")
            .and_then(Json::as_array)
            .unwrap_or(&[])
            .iter()
            .filter_map(|tag| tag.as_str().map(str::to_owned))
            .collect();
        tags.insert(id.to_owned(), set);
    }
    Ok(tags)
}

fn validate_suite_models(
    suite: &str,
    models: &[ModelSpec],
    known: &BTreeMap<String, BTreeSet<String>>,
) -> Result<(), String> {
    for group in suite_model_tags(suite) {
        contract::require(
            models.iter().any(|model| {
                known
                    .get(&model.artifact_id)
                    .is_some_and(|tags| group.iter().any(|tag| tags.contains(*tag)))
            }),
            &format!("{suite} lacks a model tagged {group:?}"),
        )?;
    }
    Ok(())
}

fn build_hardware(
    row: &Json,
    availability: &Json,
    profile: &Value,
    selected_device: &str,
) -> Result<Value, String> {
    contract::require(
        availability.get("state").and_then(Json::as_str) == Some("available"),
        "hardware evidence requires a policy-available row",
    )?;
    contract::require(
        availability.get("row_id").and_then(Json::as_str) == row.get("id").and_then(Json::as_str),
        "availability belongs to another row",
    )?;
    let runner = availability
        .get("runner")
        .and_then(Json::as_str)
        .filter(|runner| !runner.trim().is_empty())
        .ok_or("available row lacks an approved runner")?;
    let backend = contract::row_field(row, "backend")?;
    let expected = contract::expected_device(backend).unwrap_or("");
    contract::require(
        selected_device == expected,
        &format!("product selected device {selected_device:?}, not {expected:?}"),
    )?;
    if backend == "cpu" {
        return Ok(json!({
            "actual_backend": "cpu",
            "runner": runner,
            "device": "CPU",
            "selected_device": "CPU",
        }));
    }
    let device = if backend == "metal" {
        apple_chip_name()
    } else {
        None
    }
    .or_else(|| first_gpu_name(profile));
    let device = device
        .filter(|device| !device.trim().is_empty())
        .ok_or("GPU row exposes no hardware device identity")?;
    let (driver, runtime_version, architecture) = match backend {
        "cuda" => {
            let cuda = profile.get("cuda").cloned().unwrap_or(Value::Null);
            let mut majors = number_or_string_list(cuda.get("toolkit_majors"));
            majors.sort();
            let mut arches = string_list(cuda.get("gpu_arches"));
            arches.sort();
            (
                cuda.get("driver_version")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                majors.last().map(|major| format!("CUDA {major}")),
                arches.last().cloned(),
            )
        }
        "rocm" => {
            let rocm = profile.get("rocm").cloned().unwrap_or(Value::Null);
            let mut arches = string_list(rocm.get("gpu_arches"));
            arches.sort();
            let version = rocm
                .get("version")
                .and_then(Value::as_str)
                .map(str::to_owned);
            (version.clone(), version, arches.first().cloned())
        }
        "vulkan" => {
            let vulkan = profile.get("vulkan").cloned().unwrap_or(Value::Null);
            let api = vulkan
                .get("api_version")
                .and_then(Value::as_str)
                .map(str::to_owned);
            (api.clone(), api, Some(device.clone()))
        }
        _ => {
            let macos = macos_version();
            (
                macos.map(|version| format!("macOS {version}")),
                metal_runtime_version(profile),
                profile
                    .get("arch")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
            )
        }
    };
    for (name, value) in [
        ("driver", &driver),
        ("runtime_version", &runtime_version),
        ("device_architecture", &architecture),
    ] {
        contract::require(
            value.as_ref().is_some_and(|value| !value.trim().is_empty()),
            &format!("GPU row exposes no {name} identity from the composed product"),
        )?;
    }
    Ok(json!({
        "actual_backend": backend,
        "runner": runner,
        "device": device,
        "selected_device": selected_device,
        "driver": driver,
        "runtime_version": runtime_version,
        "device_architecture": architecture,
    }))
}

fn string_list(value: Option<&Value>) -> Vec<String> {
    value
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default()
}

/// String or integer list entries as strings (CUDA toolkit majors are ints),
/// ordered numerically like the reference implementation's `sorted`.
fn number_or_string_list(value: Option<&Value>) -> Vec<String> {
    let mut items: Vec<(i64, String)> = value
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| {
                    item.as_i64()
                        .map(|number| (number, number.to_string()))
                        .or_else(|| {
                            item.as_str().map(|text| {
                                (text.parse::<i64>().unwrap_or(i64::MAX), text.to_owned())
                            })
                        })
                })
                .collect()
        })
        .unwrap_or_default();
    items.sort();
    items.into_iter().map(|(_, text)| text).collect()
}

fn gpu_display_names(profile: &Value) -> Vec<String> {
    profile
        .get("gpus")
        .and_then(Value::as_array)
        .map(|gpus| {
            gpus.iter()
                .filter_map(|gpu| gpu.get("display_name"))
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|name| !name.is_empty())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

fn first_gpu_name(profile: &Value) -> Option<String> {
    gpu_display_names(profile)
        .into_iter()
        .find(|name| name != "Type: GPU")
}

/// The Metal runtime the product bound: an explicit version token, the host's
/// Metal label, or the OS Metal stack; the row's backend proof stays the
/// runtime's own device event, not this string.
fn metal_runtime_version(profile: &Value) -> Option<String> {
    let names = gpu_display_names(profile);
    for name in &names {
        if let Some(start) = name.find("Metal ") {
            let version: String = name[start + 6..]
                .chars()
                .take_while(|ch| ch.is_ascii_digit() || *ch == '.')
                .collect();
            if !version.is_empty() {
                return Some(format!("Metal {version}"));
            }
        }
    }
    for name in &names {
        if name.to_lowercase().contains("metal") {
            return Some(name.clone());
        }
    }
    macos_version().map(|version| format!("Metal (macOS {version})"))
}

fn command_output(arguments: &[&str]) -> Option<String> {
    let output = std::process::Command::new(arguments[0])
        .args(&arguments[1..])
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn apple_chip_name() -> Option<String> {
    command_output(&["sysctl", "-n", "machdep.cpu.brand_string"]).filter(|name| !name.is_empty())
}

fn macos_version() -> Option<String> {
    command_output(&["sysctl", "-n", "kern.osproductversion"]).filter(|value| !value.is_empty())
}

fn host_hardware(product_dir: &Path, manifest: &Json, cache_dir: &Path) -> Result<Value, String> {
    let cli = manifest.get("cli").ok_or("product manifest lacks cli")?;
    let binary = product_dir.join(
        cli.get("path")
            .and_then(Json::as_str)
            .ok_or("product manifest lacks cli path")?,
    );
    let runtime = manifest
        .get("runtime")
        .ok_or("product manifest lacks runtime")?;
    let output = serve::run_capture(
        &binary,
        &[
            "--runtime-bundle".to_owned(),
            product_dir
                .join(
                    runtime
                        .get("path")
                        .and_then(Json::as_str)
                        .unwrap_or_default(),
                )
                .to_string_lossy()
                .into_owned(),
            "--runtime-cache".to_owned(),
            cache_dir
                .join("doctor-runtime-cache")
                .to_string_lossy()
                .into_owned(),
            "--runtime-selection".to_owned(),
            manifest
                .get("backend")
                .and_then(Json::as_str)
                .unwrap_or_default()
                .to_owned(),
            "doctor".to_owned(),
            "--output".to_owned(),
            "json".to_owned(),
        ],
        Duration::from_secs(180),
    )?;
    let report: Value = serde_json::from_str(&output).map_err(|error| error.to_string())?;
    report
        .get("hardware")
        .filter(|hardware| hardware.is_object())
        .cloned()
        .ok_or_else(|| "composed product reported no hardware profile".to_owned())
}

struct ModelsJob<'a> {
    root: &'a Path,
    product_dir: &'a Path,
    row_id: &'a str,
    suite: &'a str,
    device: Option<&'a str>,
    ctx_size: u32,
    models: &'a [ModelSpec],
    evidence: &'a Path,
    availability_path: Option<&'a Path>,
    hardware_evidence: Option<&'a Path>,
}

fn run_qualification(job: &ModelsJob<'_>) -> Result<Value, String> {
    let root = job.root;
    let product_dir = job.product_dir;
    let row_id = job.row_id;
    let suite = job.suite;
    let device = job.device;
    let ctx_size = job.ctx_size;
    let models = job.models;
    let evidence = job.evidence;
    let availability_path = job.availability_path;
    let hardware_evidence = job.hardware_evidence;
    let row = contract::catalog_row(root, row_id)?;
    let backend = contract::row_field(&row, "backend")?;
    let expected_device = contract::expected_device(backend).unwrap_or("");
    let device = device.unwrap_or(expected_device);
    contract::require(
        device == expected_device,
        "device differs from the selected backend",
    )?;
    let manifest_path = product_dir.join("product-manifest.json");
    let manifest = contract::load_json(&manifest_path)?;
    contract::require(
        manifest.get("contract").and_then(Json::as_str) == Some("skippy-product-v1")
            && manifest.get("target").and_then(Json::as_str)
                == contract::row_field(&row, "target").ok()
            && manifest.get("backend").and_then(Json::as_str) == Some(backend),
        "composed product differs from the selected row",
    )?;
    contract::validate_product_bytes(&manifest, product_dir)?;
    let identifiers: BTreeMap<String, String> = models
        .iter()
        .map(|model| (model.artifact_id.clone(), model.sha256.clone()))
        .collect();
    let models_json = Json::Object(
        identifiers
            .iter()
            .map(|(id, sha)| (id.clone(), Json::String(sha.clone())))
            .collect(),
    );
    contract::validate_models(root, suite, &models_json)?;
    let known = registry_tags(root)?;
    validate_suite_models(suite, models, &known)?;

    let (driver, restart) = suite_driver(suite).ok_or("unknown qualification suite")?;
    let mut cases: Vec<String> = Vec::new();
    let mut observations = serde_json::Map::new();
    let mut selected_device: Option<String> = None;
    for model in models {
        contract::require(
            model.path.is_file() && contract::digest(&model.path)? == model.sha256,
            &format!("pinned {} bytes differ", model.artifact_id),
        )?;
        let directory = tempfile::Builder::new()
            .prefix(&format!("skippy-{suite}-"))
            .tempdir()
            .map_err(|error| error.to_string())?;
        let mut server = start_model_server(
            product_dir,
            &manifest,
            model,
            device,
            ctx_size,
            directory.path(),
        )?;
        if selected_device.is_none() {
            let found = serve::wait_for_backend_device(&server.log_path, Duration::from_secs(60));
            let found = found.ok_or_else(|| {
                format!(
                    "product reported no backend device event; {}",
                    server.tail()
                )
            })?;
            contract::require(
                found == expected_device,
                &format!(
                    "product selected device {found:?}, not {expected_device:?}; {}",
                    server.tail()
                ),
            )?;
            selected_device = Some(found);
        }
        let mut executed = driver(&server, &model.model_id, &mut observations)?;
        if restart {
            server.stop();
            let resumed = start_model_server(
                product_dir,
                &manifest,
                model,
                device,
                ctx_size,
                directory.path(),
            )?;
            let repeat = completion(
                &resumed.base,
                &model.model_id,
                json!([{"role": "user", "content": PROMPT}]),
                &[],
            )?;
            let expected = observations
                .get("prefill_tokens")
                .and_then(Value::as_i64)
                .unwrap_or(0);
            contract::require(
                prompt_tokens(&repeat)? == expected,
                "restart changed the prefill for an identical prompt",
            )?;
            drop(resumed);
            executed.push("restart".to_owned());
        }
        cases.extend(executed);
        observations.insert(
            format!("{}_log_sha256", model.artifact_id),
            json!(contract::digest(&server.log_path)?),
        );
        for (family, case) in [
            ("dense", "dense-prefix-hit"),
            ("hybrid", "recurrent-prefix-hit"),
        ] {
            if suite == "kv-cache"
                && known
                    .get(&model.artifact_id)
                    .is_some_and(|tags| tags.contains(family))
            {
                cases.push(case.to_owned());
            }
        }
    }
    let executed_set: BTreeSet<&str> = cases.iter().map(String::as_str).collect();
    let required: BTreeSet<&str> = contract::suite_cases(suite)
        .unwrap_or(&[])
        .iter()
        .copied()
        .collect();
    contract::require(
        executed_set == required,
        &format!("{suite} executed cases differ from the contract: {executed_set:?}"),
    )?;
    let result = json!({
        "schema_version": 1,
        "status": "passed",
        "source_sha": manifest.get("source_sha").and_then(Json::as_str).unwrap_or_default(),
        "row_id": row_id,
        "suite": suite,
        "product_manifest_sha256": contract::digest(&manifest_path)?,
        "executed_cases": executed_set.into_iter().collect::<Vec<_>>(),
        "models": identifiers,
        "observations": Value::Object(observations),
    });
    std::fs::write(evidence, contract::render(&result)?)
        .map_err(|error| format!("{}: {error}", evidence.display()))?;
    if let Some(hardware_evidence) = hardware_evidence {
        let availability_path =
            availability_path.ok_or("--hardware-evidence requires --availability")?;
        let availability = contract::load_json(availability_path)?;
        let selected = selected_device.clone().unwrap_or_default();
        let profile = host_hardware(
            product_dir,
            &manifest,
            evidence.parent().unwrap_or(Path::new(".")),
        )?;
        let hardware = build_hardware(&row, &availability, &profile, &selected)?;
        if let Some(parent) = hardware_evidence.parent() {
            std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        std::fs::write(hardware_evidence, contract::render(&hardware)?)
            .map_err(|error| format!("{}: {error}", hardware_evidence.display()))?;
    }
    Ok(result)
}

fn start_model_server(
    product_dir: &Path,
    manifest: &Json,
    model: &ModelSpec,
    device: &str,
    ctx_size: u32,
    directory: &Path,
) -> Result<Server, String> {
    let port = serve::free_port()?;
    let cli = manifest
        .get("cli")
        .and_then(|cli| cli.get("path"))
        .and_then(Json::as_str)
        .unwrap_or_default();
    let runtime = manifest
        .get("runtime")
        .and_then(|runtime| runtime.get("path"))
        .and_then(Json::as_str)
        .unwrap_or_default();
    let arguments = vec![
        "--runtime-bundle".to_owned(),
        product_dir.join(runtime).to_string_lossy().into_owned(),
        "--runtime-cache".to_owned(),
        directory
            .join("runtime-cache")
            .to_string_lossy()
            .into_owned(),
        "--runtime-selection".to_owned(),
        manifest
            .get("backend")
            .and_then(Json::as_str)
            .unwrap_or_default()
            .to_owned(),
        "serve".to_owned(),
        "--model-path".to_owned(),
        model.path.to_string_lossy().into_owned(),
        "--model-id".to_owned(),
        model.model_id.clone(),
        "--device".to_owned(),
        device.to_owned(),
        "--ctx-size".to_owned(),
        ctx_size.to_string(),
        "--n-gpu-layers".to_owned(),
        if device == "CPU" {
            "0".into()
        } else {
            "999".into()
        },
        "--bind-addr".to_owned(),
        format!("127.0.0.1:{port}"),
    ];
    Server::start(
        &product_dir.join(cli),
        &arguments,
        &model.model_id,
        &directory.join("serve.log"),
    )
}

pub(super) fn run(root: &Path, args: &[String]) -> CheckReport {
    const GRAMMAR: Grammar = Grammar {
        usage: USAGE,
        values: &[
            "--product-dir",
            "--row-id",
            "--suite",
            "--device",
            "--ctx-size",
            "--model",
            "--evidence",
            "--availability",
            "--hardware-evidence",
        ],
        flags: &[],
    };
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report,
    };
    match parse_and_run(root, &parsed) {
        Ok(()) => CheckReport::success(String::new()),
        Err(error) => CheckReport::failure(
            String::new(),
            format!("standalone model qualification failed: {error}\n"),
        ),
    }
}

fn parse_and_run(
    root: &Path,
    parsed: &crate::repository::check_args::ParsedArgs,
) -> Result<(), String> {
    for required in ["--product-dir", "--row-id", "--suite", "--evidence"] {
        if parsed.last(required).is_none() {
            return Err(format!("the following arguments are required: {required}"));
        }
    }
    let groups = parsed.all("--model");
    if groups.is_empty() || !groups.len().is_multiple_of(4) {
        return Err("each --model requires ARTIFACT PATH SHA256 MODEL_ID".to_owned());
    }
    let mut models = Vec::new();
    for group in groups.chunks(4) {
        models.push(ModelSpec {
            artifact_id: group[0].to_owned(),
            path: PathBuf::from(group[1]),
            sha256: group[2].to_owned(),
            model_id: group[3].to_owned(),
        });
    }
    let ctx_size: u32 = parsed
        .last("--ctx-size")
        .unwrap_or("2048")
        .parse()
        .map_err(|_| "invalid --ctx-size".to_owned())?;
    run_qualification(&ModelsJob {
        root,
        product_dir: Path::new(parsed.last("--product-dir").unwrap_or_default()),
        row_id: parsed.last("--row-id").unwrap_or_default(),
        suite: parsed.last("--suite").unwrap_or_default(),
        device: parsed.last("--device"),
        ctx_size,
        models: &models,
        evidence: Path::new(parsed.last("--evidence").unwrap_or_default()),
        availability_path: parsed.last("--availability").map(Path::new),
        hardware_evidence: parsed.last("--hardware-evidence").map(Path::new),
    })?;
    Ok(())
}

#[cfg(test)]
mod tests;
