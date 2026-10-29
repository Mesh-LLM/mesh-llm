//! `skippy audit-receipt`: sends one chat request to a node serving with
//! receipts, then challenges the receipt and checks the opening against the
//! claimed GGUF on this machine.

use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use reqwest::StatusCode;
use serde::Serialize;
use skippy_receipts::audit::{self, AuditReport, MatrixVerdict};
use skippy_receipts::challenge;
use skippy_receipts::glue::{self, GlueVerdict};
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
}

#[derive(Serialize)]
struct AuditOutput {
    response: String,
    receipt: Receipt,
    challenge: Challenge,
    paths_ok: bool,
    matrices: Vec<MatrixOutput>,
    glue: Vec<GlueOutput>,
    pass: bool,
}

#[derive(Serialize)]
struct GlueOutput {
    check: String,
    tokens: usize,
    failing_tokens: usize,
    max_error: f64,
    pass: bool,
}

impl From<&GlueVerdict> for GlueOutput {
    fn from(g: &GlueVerdict) -> Self {
        Self {
            check: g.check.to_string(),
            tokens: g.tokens,
            failing_tokens: g.failing_tokens,
            max_error: g.max_error,
            pass: g.pass,
        }
    }
}

#[derive(Serialize)]
struct MatrixOutput {
    weight: String,
    weight_type: String,
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
            tokens: m.tokens,
            failing_tokens: m.failing_tokens,
            max_error: m.max_error,
            best_profile: m.best_profile.to_string(),
            pass: m.pass,
        }
    }
}

pub fn run(command: AuditReceiptCommand) -> Result<()> {
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
    let (report, glue) = match &paths {
        Ok(()) => (
            Some(audit::check_rows(&weights, &opening, &mut rng)?),
            glue::check_glue(&weights, &opening)?,
        ),
        Err(_) => (None, Vec::new()),
    };
    let pass =
        paths.is_ok() && report.as_ref().is_some_and(|r| r.pass) && glue.iter().all(|g| g.pass);
    let output = AuditOutput {
        response,
        receipt,
        challenge,
        paths_ok: paths.is_ok(),
        matrices: report
            .as_ref()
            .map(|r| r.matrices.iter().map(MatrixOutput::from).collect())
            .unwrap_or_default(),
        glue: glue.iter().map(GlueOutput::from).collect(),
        pass,
    };
    crate::console::present(&output, |out| {
        render(out, &output, paths.as_ref().err(), report.as_ref())
    })?;
    if !pass {
        bail!("receipt audit failed");
    }
    Ok(())
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
            "  stage {}  layers {:>2}..{:<2}  root {}…  leaves {}  ({})",
            stage.stage_index,
            stage.layer_start,
            stage.layer_end,
            &stage.root[..16.min(stage.root.len())],
            stage.leaf_count,
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
                "  {:<26} {:<5} {:>2}/{:<2} tokens pass  max err {:.1e}  {}  {}",
                m.weight,
                m.weight_type,
                m.tokens - m.failing_tokens,
                m.tokens,
                m.max_error,
                m.best_profile,
                if m.pass { "ok" } else { "FAIL" }
            )?;
        }
    }
    if !output.glue.is_empty() {
        writeln!(out, "glue:")?;
    }
    for g in &output.glue {
        writeln!(
            out,
            "  {:<34} {:>2}/{:<2} tokens pass  max err {:.1e}  {}",
            g.check,
            g.tokens - g.failing_tokens,
            g.tokens,
            g.max_error,
            if g.pass { "ok" } else { "FAIL" }
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
