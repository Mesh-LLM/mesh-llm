//! `qualify laya`: execute System One and Decisions against one composed
//! standalone Laya product, reusing the in-crate Laya parity and System One
//! case drivers.

use super::contract::{self, Json};
use super::serve::{self, Server};
use crate::repository::check_args::Grammar;
use crate::repository::check_report::CheckReport;
use serde_json::{Value, json};
use std::path::Path;
use std::time::Duration;

const USAGE: &str = "qualify laya --product-dir <dir> --model <path> --model-sha256 <sha256> --model-id <id> --row-id <row> [--device <dev>] --evidence-dir <dir>";

const MODEL_ARTIFACT: &str = "family-laya-multilingual";
const QUESTION_STATE: &str = "I was charged twice this month. Please refund the second payment.";

fn system_questions() -> Value {
    json!({
        "question_0": {"type": "noul", "instructions": "Does this need action today?"},
        "question_1": {"type": "choice", "instructions": "Which team?",
                       "criteria": {"billing": "Payments and refunds", "support": "Technical help"}},
        "question_2": {"type": "score", "instructions": "How frustrated?",
                       "criteria": ["Calm", "Frustrated"]},
    })
}

fn decision_questions() -> Value {
    json!([
        {"type": "predicate", "name": "urgent", "instructions": "Does this need action today?"},
        {"type": "choice", "name": "team", "instructions": "Which team?",
         "choices": [
             {"value": "billing", "description": "Payments and refunds"},
             {"value": "support", "description": "Technical help"},
         ]},
        {"type": "score", "name": "frustration", "instructions": "How frustrated?",
         "levels": [
             {"label": "Calm", "description": "Calm"},
             {"label": "Frustrated", "description": "Frustrated"},
         ]},
    ])
}

fn system_body(model_id: &str) -> Value {
    json!({"model": model_id, "state": QUESTION_STATE, "questions": system_questions()})
}

fn decisions_body(model_id: &str) -> Value {
    json!({"model": model_id, "input": QUESTION_STATE, "questions": decision_questions()})
}

fn positive(base: &str, route: &str, body: Value) -> Result<Value, String> {
    let (status, response) = super::http::request_json(
        &format!("{base}{route}"),
        Some(body),
        "POST",
        Duration::from_secs(300),
    )
    .map_err(|error| error.to_string())?;
    contract::require(
        status == 200,
        &format!("{route} returned HTTP {status}: {response}"),
    )?;
    Ok(response)
}

fn negative(base: &str, route: &str, body: Option<Value>, expected: u16) -> Result<(), String> {
    let (status, response) = super::http::request_json(
        &format!("{base}{route}"),
        body,
        if expected == 405 { "GET" } else { "POST" },
        Duration::from_secs(300),
    )
    .map_err(|error| error.to_string())?;
    contract::require(
        status == expected,
        &format!("{route} returned HTTP {status}; expected {expected}"),
    )?;
    let message = response
        .get("error")
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str);
    contract::require(
        message.is_some_and(|message| !message.is_empty()),
        &format!("{route} lacked a typed error: {response}"),
    )
}

fn float(value: &Value) -> Option<f64> {
    value.as_f64().filter(|value| value.is_finite())
}

/// `/systemone` vs `/v1/decisions` equivalence: identical answers under both
/// surfaces, normalized probabilities, and zero generated tokens.
pub(crate) fn equivalence(system: &Value, decisions: &Value, model_id: &str) -> Result<(), String> {
    contract::require(
        system.get("model").and_then(Value::as_str) == Some(model_id)
            && decisions.get("model").and_then(Value::as_str) == Some(model_id),
        "System One and Decisions model identities differ",
    )?;
    let answers = decisions
        .get("answers")
        .and_then(Value::as_array)
        .ok_or("Decisions answer order or names differ")?;
    let kinds: Vec<(&str, &str)> = answers
        .iter()
        .map(|answer| {
            (
                answer.get("type").and_then(Value::as_str).unwrap_or(""),
                answer.get("name").and_then(Value::as_str).unwrap_or(""),
            )
        })
        .collect();
    contract::require(
        kinds
            == [
                ("predicate", "urgent"),
                ("choice", "team"),
                ("score", "frustration"),
            ],
        "Decisions answer order or names differ",
    )?;
    let original = system
        .get("answers")
        .and_then(Value::as_object)
        .filter(|answers| {
            answers.len() == 3
                && ["question_0", "question_1", "question_2"]
                    .iter()
                    .all(|key| answers.contains_key(*key))
        })
        .ok_or("System One answer set differs")?;
    let [predicate, choice, score] = &answers[..] else {
        return Err("Decisions answer order or names differ".to_owned());
    };
    let probability = predicate
        .get("probability")
        .and_then(float)
        .filter(|value| (0.0..=1.0).contains(value))
        .ok_or("predicate probability is not finite and bounded")?;
    let noul = original["question_0"].get("noul").and_then(float);
    contract::require(
        noul.is_some_and(|noul| (probability - noul).abs() <= 1e-6),
        "predicate probability differs from System One",
    )?;
    let choice_value = choice.get("choice").and_then(Value::as_str).unwrap_or("");
    contract::require(
        Some(choice_value) == original["question_1"].get("choice").and_then(Value::as_str),
        "choice answer differs from System One",
    )?;
    let choice_probabilities = choice
        .get("probabilities")
        .and_then(Value::as_array)
        .ok_or("choice values or order differ")?;
    contract::require(
        choice_probabilities
            .iter()
            .map(|item| item.get("value").and_then(Value::as_str).unwrap_or(""))
            .collect::<Vec<_>>()
            == ["billing", "support"],
        "choice values or order differ",
    )?;
    let values: Vec<f64> = choice_probabilities
        .iter()
        .filter_map(|item| item.get("probability").and_then(float))
        .collect();
    contract::require(
        values.len() == choice_probabilities.len()
            && values.iter().all(|value| (0.0..=1.0).contains(value))
            && (values.iter().sum::<f64>() - 1.0).abs() <= 1e-3,
        "choice probabilities are not normalized",
    )?;
    let argmax = values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
        .unwrap_or(usize::MAX);
    contract::require(
        Some(choice_value)
            == choice_probabilities
                .get(argmax)
                .and_then(|item| item.get("value"))
                .and_then(Value::as_str),
        "choice does not select the highest probability",
    )?;
    for item in choice_probabilities {
        let value = item.get("value").and_then(Value::as_str).unwrap_or("");
        let expected = original["question_1"]["probabilities"]
            .get(value)
            .and_then(float);
        let actual = item.get("probability").and_then(float);
        contract::require(
            expected
                .zip(actual)
                .is_some_and(|(a, b)| (a - b).abs() <= 1e-6),
            "choice probability differs from System One",
        )?;
    }
    let score_value = score.get("score").and_then(float);
    let expected_score = original["question_2"].get("score").and_then(float);
    contract::require(
        score_value
            .zip(expected_score)
            .is_some_and(|(a, b)| (a - b).abs() <= 1e-6),
        "score expectation differs from System One",
    )?;
    let score_probabilities = score
        .get("probabilities")
        .and_then(Value::as_array)
        .ok_or("score levels or order differ")?;
    contract::require(
        score_probabilities
            .iter()
            .map(|item| {
                (
                    item.get("value").and_then(Value::as_i64).unwrap_or(-1),
                    item.get("label").and_then(Value::as_str).unwrap_or(""),
                )
            })
            .collect::<Vec<_>>()
            == [(0, "Calm"), (1, "Frustrated")],
        "score levels or order differ",
    )?;
    let scores: Vec<f64> = score_probabilities
        .iter()
        .filter_map(|item| item.get("probability").and_then(float))
        .collect();
    contract::require(
        scores.len() == score_probabilities.len()
            && scores.iter().all(|value| (0.0..=1.0).contains(value))
            && (scores.iter().sum::<f64>() - 1.0).abs() <= 1e-3,
        "score probabilities are not normalized",
    )?;
    let expectation: f64 = scores
        .iter()
        .enumerate()
        .map(|(index, value)| index as f64 * value)
        .sum();
    contract::require(
        score_value.is_some_and(|value| (value - expectation).abs() <= 1e-3),
        "score is not the distribution expectation",
    )?;
    for item in score_probabilities {
        let level = item
            .get("value")
            .and_then(Value::as_i64)
            .unwrap_or(-1)
            .to_string();
        let expected = original["question_2"]["probabilities"]
            .get(&level)
            .and_then(float);
        let actual = item.get("probability").and_then(float);
        contract::require(
            expected
                .zip(actual)
                .is_some_and(|(a, b)| (a - b).abs() <= 1e-6),
            "score probability differs from System One",
        )?;
    }
    let usage = decisions.get("usage").filter(|usage| usage.is_object());
    contract::require(
        usage
            .and_then(|usage| usage.get("input_tokens").and_then(Value::as_i64))
            .is_some_and(|tokens| tokens > 0)
            && usage.and_then(|usage| usage.get("output_tokens").and_then(Value::as_i64))
                == Some(0),
        "Decisions must perform a read without generation",
    )
}

fn run_driver(arguments: &[String], output: &Path, root: &Path) -> Result<Value, String> {
    let executable = std::env::current_exe().map_err(|error| error.to_string())?;
    let mut argv = Vec::new();
    argv.extend(arguments.iter().cloned());
    let status = std::process::Command::new(executable)
        .args(&argv)
        .current_dir(root)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .status()
        .map_err(|error| error.to_string())?;
    contract::require(
        status.success(),
        &format!("{} did not pass", arguments.join(" ")),
    )?;
    let report = contract::load_json(output)?;
    let passed = report.get("passed").and_then(|v| match v {
        Json::Bool(flag) => Some(*flag),
        _ => None,
    }) == Some(true)
        || report.get("status").and_then(Json::as_str) == Some("pass");
    contract::require(passed, &format!("{} did not pass", arguments.join(" ")))?;
    Ok(report.to_value())
}

fn start_server(
    product_dir: &Path,
    manifest: &Json,
    model: &Path,
    model_id: &str,
    device: &str,
    evidence_dir: &Path,
    label: &str,
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
        "--runtime-selection".to_owned(),
        manifest
            .get("backend")
            .and_then(Json::as_str)
            .unwrap_or_default()
            .to_owned(),
        "serve".to_owned(),
        "--model-path".to_owned(),
        model.to_string_lossy().into_owned(),
        "--model-id".to_owned(),
        model_id.to_owned(),
        "--device".to_owned(),
        device.to_owned(),
        "--bind-addr".to_owned(),
        format!("127.0.0.1:{port}"),
    ];
    Server::start(
        &product_dir.join(cli),
        &arguments,
        model_id,
        &evidence_dir.join(format!("laya-{label}.log")),
    )
}

fn evidence(
    suite: &str,
    source_sha: &str,
    row_id: &str,
    product_digest: &str,
    model_hash: &str,
    observations: &serde_json::Map<String, Value>,
) -> Value {
    let mut cases: Vec<&str> = contract::suite_cases(suite).unwrap_or(&[]).to_vec();
    cases.sort_unstable();
    json!({
        "schema_version": 1,
        "status": "passed",
        "source_sha": source_sha,
        "row_id": row_id,
        "suite": suite,
        "product_manifest_sha256": product_digest,
        "executed_cases": cases,
        "models": {MODEL_ARTIFACT: model_hash},
        "observations": Value::Object(observations.clone()),
    })
}

struct LayaJob<'a> {
    root: &'a Path,
    product_dir: &'a Path,
    model: &'a Path,
    model_hash: &'a str,
    model_id: &'a str,
    row_id: &'a str,
    device: Option<&'a str>,
    evidence_dir: &'a Path,
}

fn run_qualification(job: &LayaJob<'_>) -> Result<(), String> {
    let root = job.root;
    let product_dir = job.product_dir;
    let model = job.model;
    let model_hash = job.model_hash;
    let model_id = job.model_id;
    let row_id = job.row_id;
    let device = job.device;
    let evidence_dir = job.evidence_dir;
    let manifest_path = product_dir.join("product-manifest.json");
    let manifest = contract::load_json(&manifest_path)?;
    let row = contract::catalog_row(root, row_id)?;
    let backend = contract::row_field(&row, "backend")?;
    contract::require(
        manifest.get("source_sha").and_then(Json::as_str).is_some()
            && manifest.get("target").and_then(Json::as_str)
                == contract::row_field(&row, "target").ok()
            && manifest.get("backend").and_then(Json::as_str) == Some(backend),
        "product differs from selected row",
    )?;
    let expected_device = contract::expected_device(backend).unwrap_or("");
    let device = device.unwrap_or(expected_device);
    contract::require(
        device == expected_device,
        "device differs from selected backend",
    )?;
    contract::validate_product_bytes(&manifest, product_dir)?;
    contract::require(
        model.is_file() && contract::digest(model)? == model_hash,
        "pinned Laya bytes differ",
    )?;
    contract::validate_models(
        root,
        "system-one",
        &Json::Object(vec![(
            MODEL_ARTIFACT.to_owned(),
            Json::String(model_hash.to_owned()),
        )]),
    )?;
    std::fs::create_dir_all(evidence_dir).map_err(|error| error.to_string())?;
    let mut observations = serde_json::Map::new();
    let mut server = start_server(
        product_dir,
        &manifest,
        model,
        model_id,
        device,
        evidence_dir,
        "initial",
    )?;
    let base = server.base.clone();
    let initial_answers: Result<Option<Value>, String> = (|| {
        let parity = evidence_dir.join("laya-goldens.json");
        let report = run_driver(
            &[
                "automation".into(),
                "laya".into(),
                "parity".into(),
                "--base-url".into(),
                base.clone(),
                "--model".into(),
                model_id.to_owned(),
                "--json-out".into(),
                parity.to_string_lossy().into_owned(),
            ],
            &parity,
            root,
        )?;
        let results = report.get("results").and_then(Value::as_array);
        contract::require(
            results.is_some_and(|results| results.len() >= 7),
            "Laya golden battery is incomplete",
        )?;
        observations.insert("goldens_sha256".into(), json!(contract::digest(&parity)?));
        let reader = evidence_dir.join("laya-reader-contract.json");
        let report = run_driver(
            &[
                "automation".into(),
                "system-one-cases".into(),
                "--base-url".into(),
                base.clone(),
                "--model".into(),
                model_id.to_owned(),
                "--mode".into(),
                "full-read".into(),
                "--alias".into(),
                model_id.to_owned(),
                "--json-out".into(),
                reader.to_string_lossy().into_owned(),
            ],
            &reader,
            root,
        )?;
        let cases = report.get("cases").and_then(Value::as_array);
        contract::require(
            cases.is_some_and(|cases| cases.len() >= 6),
            "Laya reader contract is incomplete",
        )?;
        observations.insert("reader_sha256".into(), json!(contract::digest(&reader)?));
        let system = positive(&base, "/systemone", system_body(model_id))?;
        let decisions = positive(&base, "/v1/decisions", decisions_body(model_id))?;
        equivalence(&system, &decisions, model_id)?;
        let concurrent: Vec<Result<Value, String>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..2)
                .map(|_| scope.spawn(|| positive(&base, "/systemone", system_body(model_id))))
                .collect();
            handles
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .unwrap_or_else(|_| Err("concurrent read panicked".to_owned()))
                })
                .collect()
        });
        for result in concurrent {
            contract::require(
                result?.get("answers") == system.get("answers"),
                "concurrent System One reads changed answers",
            )?;
        }
        negative(
            &base,
            "/systemone",
            Some(json!({"model": model_id, "state": QUESTION_STATE, "questions": {}})),
            400,
        )?;
        negative(
            &base,
            "/systemone",
            Some(system_body("missing-laya-model")),
            400,
        )?;
        negative(&base, "/systemone", None, 405)?;
        negative(
            &base,
            "/v1/decisions",
            Some(json!({"model": model_id, "input": QUESTION_STATE, "questions": []})),
            400,
        )?;
        let mut duplicate = decisions_body(model_id);
        let first_choice = duplicate["questions"][1]["choices"][0].clone();
        duplicate["questions"][1]["choices"]
            .as_array_mut()
            .ok_or("duplicate choice fixture is malformed")?
            .push(first_choice);
        negative(&base, "/v1/decisions", Some(duplicate), 400)?;
        Ok(system.get("answers").cloned())
    })();
    let log_digest = contract::digest(&server.log_path);
    server.stop();
    let initial_answers = initial_answers?;
    observations.insert("initial_log_sha256".into(), json!(log_digest?));
    let mut resumed = start_server(
        product_dir,
        &manifest,
        model,
        model_id,
        device,
        evidence_dir,
        "restart",
    )?;
    let restart_result: Result<(), String> = (|| {
        let resumed_system = positive(&resumed.base, "/systemone", system_body(model_id))?;
        let resumed_decisions = positive(&resumed.base, "/v1/decisions", decisions_body(model_id))?;
        equivalence(&resumed_system, &resumed_decisions, model_id)?;
        contract::require(
            resumed_system.get("answers") == initial_answers.as_ref(),
            "System One answers changed after process restart",
        )?;
        Ok(())
    })();
    let restart_log_digest = contract::digest(&resumed.log_path);
    resumed.stop();
    restart_result?;
    observations.insert("restart_log_sha256".into(), json!(restart_log_digest?));
    let source_sha = manifest
        .get("source_sha")
        .and_then(Json::as_str)
        .unwrap_or_default();
    let product_digest = contract::digest(&manifest_path)?;
    for suite in ["system-one", "decisions"] {
        let record = evidence(
            suite,
            source_sha,
            row_id,
            &product_digest,
            model_hash,
            &observations,
        );
        std::fs::write(
            evidence_dir.join(format!("{suite}.json")),
            contract::render(&record)?,
        )
        .map_err(|error| error.to_string())?;
    }
    Ok(())
}

pub(super) fn run(root: &Path, args: &[String]) -> CheckReport {
    const GRAMMAR: Grammar = Grammar {
        usage: USAGE,
        values: &[
            "--product-dir",
            "--model",
            "--model-sha256",
            "--model-id",
            "--row-id",
            "--device",
            "--evidence-dir",
        ],
        flags: &[],
    };
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report,
    };
    let required: Vec<&str> = GRAMMAR
        .values
        .iter()
        .filter(|name| **name != "--device" && parsed.last(name).is_none())
        .copied()
        .collect();
    if !required.is_empty() || !parsed.positionals.is_empty() {
        return GRAMMAR.error(&format!(
            "the following arguments are required: {}",
            required.join(", ")
        ));
    }
    let value = |name: &str| parsed.last(name).unwrap_or_default();
    let result = run_qualification(&LayaJob {
        root,
        product_dir: Path::new(value("--product-dir")),
        model: Path::new(value("--model")),
        model_hash: value("--model-sha256"),
        model_id: value("--model-id"),
        row_id: value("--row-id"),
        device: parsed.last("--device"),
        evidence_dir: Path::new(value("--evidence-dir")),
    });
    match result {
        Ok(()) => CheckReport::success(String::new()),
        Err(error) => CheckReport::failure(
            String::new(),
            format!("standalone Laya qualification failed: {error}\n"),
        ),
    }
}

#[cfg(test)]
mod tests;
