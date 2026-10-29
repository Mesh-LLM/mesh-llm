//! `skippy audit-receipt`: sends one chat request to a node serving with
//! receipts, then challenges the receipt and checks the opening against the
//! claimed GGUF on this machine: weight rows, and the glue between them by
//! computing the challenged layer on the opened values.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use reqwest::StatusCode;
use serde::Serialize;
use skippy_receipts::audit::{self, AuditReport, MatrixVerdict};
use skippy_receipts::challenge;
use skippy_receipts::glue::{self, GlueEvaluator, GlueReport, GlueVerdict};
use skippy_receipts::weights::{LocalFile, WeightFile};
use skippy_receipts::wire::{Challenge, Opening, Receipt};

/// How long to wait for a turn's receipt after its response.
const RECEIPT_WAIT: Duration = Duration::from_secs(10);

pub struct AuditReceiptCommand {
    /// OpenAI base URL of stage 0, for example `http://127.0.0.1:9337`.
    pub api: String,
    /// Receipts API of stage 0, for example `http://127.0.0.1:9501`.
    pub receipts: String,
    /// The GGUF the node claims to serve.
    pub claimed_model: PathBuf,
    pub prompt: String,
    pub max_tokens: u32,
    /// Challenge this layer instead of a random one.
    pub layer: Option<u32>,
    /// Times to compute the challenged layer for the cost report; the
    /// checks use the first.
    pub timing_runs: u32,
}

/// Loads one layer of a GGUF to evaluate its glue.
pub type OpenGlueEvaluator = dyn Fn(&Path, u32) -> Result<Box<dyn GlueEvaluator>>;

#[derive(Serialize)]
struct AuditOutput {
    response: String,
    receipt: Receipt,
    challenge: Challenge,
    paths_ok: bool,
    matrices: Vec<MatrixOutput>,
    glue: Vec<GlueOutput>,
    /// Why the glue could not be checked.
    glue_error: Option<String>,
    /// Weights read inside token-mixing ops, which no check covers.
    unchecked_weights: Vec<String>,
    cost: Option<CostOutput>,
    pass: bool,
}

#[derive(Serialize)]
struct GlueOutput {
    check: String,
    tokens: usize,
    failing_tokens: usize,
    max_error: f64,
    worst: String,
    failures: Vec<String>,
    pass: bool,
}

impl From<&GlueVerdict> for GlueOutput {
    fn from(g: &GlueVerdict) -> Self {
        Self {
            check: g.check.clone(),
            tokens: g.tokens,
            failing_tokens: g.failing_tokens,
            max_error: g.max_error,
            worst: g.worst.clone(),
            failures: g.failures.clone(),
            pass: g.pass,
        }
    }
}

/// What the audit cost this machine, measured.
#[derive(Clone, Serialize)]
struct CostOutput {
    positions: usize,
    /// Loading the challenged layer alone on the CPU.
    layer_load_seconds: f64,
    /// Each evaluation of the layer at every opened position; the first
    /// includes creating the evaluation context.
    glue_eval_seconds: Vec<f64>,
    /// The whole glue check: evaluation and comparisons.
    glue_check_seconds: f64,
    /// Reading weight rows and checking the matmul columns.
    row_check_seconds: f64,
}

impl CostOutput {
    /// The fastest evaluation, per opened position.
    fn eval_ms_per_position(&self) -> f64 {
        let best = self
            .glue_eval_seconds
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        best * 1000.0 / self.positions.max(1) as f64
    }
}

#[derive(Serialize)]
struct MatrixOutput {
    weight: String,
    weight_type: String,
    router: bool,
    tokens: usize,
    failing_tokens: usize,
    max_error: f64,
    best_profile: String,
    pass: bool,
}

impl From<&MatrixVerdict> for MatrixOutput {
    fn from(m: &MatrixVerdict) -> Self {
        Self {
            weight: m.weight.clone(),
            weight_type: m.weight_type.clone(),
            router: m.router,
            tokens: m.tokens,
            failing_tokens: m.failing_tokens,
            max_error: m.max_error,
            best_profile: m.best_profile.to_string(),
            pass: m.pass,
        }
    }
}

pub fn run(command: AuditReceiptCommand, open_glue: &OpenGlueEvaluator) -> Result<()> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()?;
    let receipt_id = uuid::Uuid::new_v4().hyphenated().to_string();
    let response = chat(&client, &command, &receipt_id)?;
    let receipts = command.receipts.trim_end_matches('/');
    let receipt = fetch_receipt(&client, receipts, &receipt_id)?;
    let weights = WeightFile::open(Box::new(LocalFile::open(&command.claimed_model)?))?;
    let mut rng = rand::rng();
    let mut challenge = challenge::draw(&receipt, &weights.layer_bytes(), &mut rng)?;
    if let Some(layer) = command.layer {
        challenge.layer = layer;
    }
    let opening: Opening = get_json(
        client
            .post(format!("{receipts}/receipts/open"))
            .json(&challenge),
    )?;
    let paths = audit::verify_paths(&receipt, &challenge, &opening);
    let checks = match &paths {
        Ok(()) => Some(check_opening(
            &command, open_glue, &weights, &opening, &mut rng,
        )?),
        Err(_) => None,
    };
    let report = checks.as_ref().map(|c| &c.rows);
    let glue = checks.as_ref().map(|c| &c.glue);
    let glue_report = glue.and_then(|g| g.as_ref().ok());
    let pass =
        paths.is_ok() && report.is_some_and(|r| r.pass) && glue_report.is_some_and(|g| g.pass);
    let output = AuditOutput {
        response,
        receipt,
        challenge,
        paths_ok: paths.is_ok(),
        matrices: report
            .map(|r| r.matrices.iter().map(MatrixOutput::from).collect())
            .unwrap_or_default(),
        glue: glue_report
            .map(|g| g.verdicts.iter().map(GlueOutput::from).collect())
            .unwrap_or_default(),
        glue_error: glue
            .and_then(|g| g.as_ref().err())
            .map(|e| format!("{e:#}")),
        unchecked_weights: glue_report
            .map(|g| g.unchecked_weights.clone())
            .unwrap_or_default(),
        cost: checks.as_ref().map(|c| c.cost.clone()),
        pass,
    };
    crate::console::present(&output, |out| {
        render(out, &output, paths.as_ref().err(), report)
    })?;
    if !pass {
        bail!("receipt audit failed");
    }
    Ok(())
}

struct OpeningChecks {
    rows: AuditReport,
    glue: Result<GlueReport>,
    cost: CostOutput,
}

/// Row checks, then the glue check on the challenged layer loaded alone.
fn check_opening(
    command: &AuditReceiptCommand,
    open_glue: &OpenGlueEvaluator,
    weights: &WeightFile,
    opening: &Opening,
    rng: &mut impl rand::Rng,
) -> Result<OpeningChecks> {
    let started = Instant::now();
    let rows = audit::check_rows(weights, opening, rng)?;
    let row_check_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let evaluator = open_glue(&command.claimed_model, opening.layer);
    let layer_load_seconds = started.elapsed().as_secs_f64();
    let mut glue_eval_seconds = Vec::new();
    let mut glue_check_seconds = 0.0;
    let glue = evaluator.and_then(|evaluator| {
        let started = Instant::now();
        let report = glue::check_glue(evaluator.as_ref(), opening)?;
        glue_check_seconds = started.elapsed().as_secs_f64();
        glue_eval_seconds.push(report.eval_seconds);
        let request = glue::glue_request(opening)?;
        for _ in 1..command.timing_runs {
            let started = Instant::now();
            evaluator.evaluate(&request)?;
            glue_eval_seconds.push(started.elapsed().as_secs_f64());
        }
        Ok(report)
    });
    Ok(OpeningChecks {
        rows,
        glue,
        cost: CostOutput {
            positions: opening.leaves.len(),
            layer_load_seconds,
            glue_eval_seconds,
            glue_check_seconds,
            row_check_seconds,
        },
    })
}

fn chat(
    client: &reqwest::blocking::Client,
    command: &AuditReceiptCommand,
    receipt_id: &str,
) -> Result<String> {
    let api = command.api.trim_end_matches('/');
    let models: serde_json::Value = get_json(client.get(format!("{api}/v1/models")))?;
    let model = models["data"][0]["id"]
        .as_str()
        .context("the node lists no model")?;
    let body = serde_json::json!({
        "model": model,
        "messages": [{ "role": "user", "content": command.prompt }],
        "max_tokens": command.max_tokens,
        "temperature": 0,
        "stream": false,
    });
    let value: serde_json::Value = get_json(
        client
            .post(format!("{api}/v1/chat/completions"))
            .header("x-request-id", receipt_id)
            .json(&body),
    )?;
    Ok(value["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default()
        .to_string())
}

/// Stage 0 commits a turn just after its response is sent, so the receipt
/// can be missing for a moment.
fn fetch_receipt(
    client: &reqwest::blocking::Client,
    receipts: &str,
    receipt_id: &str,
) -> Result<Receipt> {
    let deadline = Instant::now() + RECEIPT_WAIT;
    loop {
        let response = client
            .get(format!("{receipts}/receipts/{receipt_id}"))
            .send()?;
        if response.status() != StatusCode::NOT_FOUND || Instant::now() >= deadline {
            return decode(response);
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn get_json<T: serde::de::DeserializeOwned>(
    request: reqwest::blocking::RequestBuilder,
) -> Result<T> {
    decode(request.send()?)
}

fn decode<T: serde::de::DeserializeOwned>(response: reqwest::blocking::Response) -> Result<T> {
    let status = response.status();
    let body = response.text()?;
    if !status.is_success() {
        bail!("{status}: {body}");
    }
    serde_json::from_str(&body).with_context(|| format!("decode reply: {body}"))
}

fn render(
    out: &mut dyn Write,
    output: &AuditOutput,
    path_error: Option<&anyhow::Error>,
    report: Option<&AuditReport>,
) -> std::io::Result<()> {
    let receipt = &output.receipt;
    writeln!(out, "response: {}", output.response.trim())?;
    writeln!(out)?;
    writeln!(
        out,
        "receipt {}  model {}  positions {}..{}",
        receipt.receipt_id, receipt.model, receipt.position_start, receipt.position_end
    )?;
    for stage in &receipt.stages {
        writeln!(
            out,
            "  stage {}  layers {:>2}..{:<2}  root {}…  leaves {}  cache {}  ({})",
            stage.stage_index,
            stage.layer_start,
            stage.layer_end,
            &stage.root[..16.min(stage.root.len())],
            stage.leaf_count,
            stage.cache_types.join(","),
            stage.backend
        )?;
    }
    let stage = receipt
        .stage_for_layer(output.challenge.layer)
        .map_or(0, |s| s.stage_index);
    writeln!(
        out,
        "challenge: layer {} (stage {stage}), positions {:?}",
        output.challenge.layer, output.challenge.positions
    )?;
    match path_error {
        Some(error) => writeln!(out, "merkle paths: FAIL ({error:#})")?,
        None => writeln!(out, "merkle paths: ok")?,
    }
    if let Some(report) = report {
        for m in &report.matrices {
            writeln!(
                out,
                "  {:<30} {:<5} {:>3}/{:<3} pass  max err {:.1e}  {}  {}{}",
                m.weight,
                m.weight_type,
                m.tokens - m.failing_tokens,
                m.tokens,
                m.max_error,
                m.best_profile,
                if m.pass { "ok" } else { "FAIL" },
                if m.router { "  (router, all rows)" } else { "" }
            )?;
        }
    }
    if let Some(error) = &output.glue_error {
        writeln!(out, "glue: cannot check ({error})")?;
    } else if !output.glue.is_empty() {
        writeln!(out, "glue (the layer computed here on the opened values):")?;
    }
    for g in &output.glue {
        writeln!(
            out,
            "  {:<14} {:>3}/{:<3} pass  max err {:.1e}  {}{}",
            g.check,
            g.tokens - g.failing_tokens,
            g.tokens,
            g.max_error,
            if g.pass { "ok" } else { "FAIL" },
            if g.pass {
                String::new()
            } else {
                format!("  worst: {}", g.worst)
            }
        )?;
        for failure in &g.failures {
            writeln!(out, "      {failure}")?;
        }
    }
    if !output.unchecked_weights.is_empty() {
        writeln!(
            out,
            "unchecked weights (inside token mixing): {}",
            output.unchecked_weights.join(", ")
        )?;
    }
    if let Some(cost) = &output.cost {
        writeln!(
            out,
            "client cost: layer load {:.2} s, glue {:.2} ms per position ({} positions, first run {:.0} ms), row checks {:.0} ms",
            cost.layer_load_seconds,
            cost.eval_ms_per_position(),
            cost.positions,
            cost.glue_eval_seconds.first().copied().unwrap_or(0.0) * 1000.0,
            cost.row_check_seconds * 1000.0
        )?;
    }
    writeln!(
        out,
        "{}",
        if output.pass {
            "AUDIT PASS"
        } else {
            "AUDIT FAIL"
        }
    )
}
