//! `skippy audit-receipt`: sends one chat request to a node serving with
//! receipts, then challenges the receipt and checks the opening against the
//! claimed GGUF on this machine.
//!
//! The request names a fresh VRF key, so the node binds the receipt to it
//! and signs the turn root. The challenge comes from the VRF of that root,
//! and the checked rows from the VRF of the opening as received, so
//! neither the node nor this client chooses them.
//!
//! The checks, in order: the receipt is bound to this client's key and
//! signed, covers every layer of the claimed model, and does not contradict
//! the request; the opening holds exactly what the challenge asks, every
//! opened leaf is in its stage tree, stages agree where they hand over,
//! and every opened value fits the claimed model; each picked layer's glue
//! computed on the opened values, and its weight rows; the input of layer
//! 0 is what the receipt's tokens make; the token sampled at each head
//! position is the greedy choice of the claimed output head on the opened
//! final residual, or with `--temperature` the token the draw of the
//! request's seed picks; the response is the text of the tokens the receipt says
//! were sampled; and each picked attention head's partials combine to its
//! opened output, and recompute, for blocks this client draws after the
//! opening, from the K and V rows the node opens.
//!
//! The report ends with one of four lines, and the command with the exit
//! code that goes with it ([`Outcome`]): `AUDIT PASS` (0), `AUDIT FAIL`
//! (1), `NO AUDIT` (2) when this machine could not run a check, and
//! `NO ANSWER` (3) when the node gave no usable signed answer to the
//! challenge.
//!
//! The audit itself is [`skippy_receipt_audit`]'s. This module sends the
//! request, opens the challenge over HTTP, keeps the evidence file, and
//! prints the report.
//!
//! [`Outcome`]: skippy_receipt_audit::outcome::Outcome

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use reqwest::StatusCode;
use skippy_receipt_audit::chat_reply::ChatReply;
use skippy_receipt_audit::claimed_file;
use skippy_receipt_audit::claimed_model::LoadClaimedModel;
use skippy_receipt_audit::output::NoAuditOutput;
use skippy_receipt_audit::report_text;
use skippy_receipt_audit::turn_audit::{
    self, AuditOptions, ClaimedSetup, OpenChallenge, Opened, OpenedRows, ReceivedTurn, TurnAudit,
};
use skippy_receipts::attention::opening::BlockChallenge;
use skippy_receipts::turn;
use skippy_receipts::vrf;
use skippy_receipts::wire::{Challenge, Receipt};

/// How long to wait for a turn's receipt after its response.
const RECEIPT_WAIT: Duration = Duration::from_secs(10);

pub struct AuditReceiptCommand {
    /// OpenAI base URL of stage 0, for example `http://127.0.0.1:9337`.
    pub api: String,
    /// Receipts API of stage 0, for example `http://127.0.0.1:9501`. Without
    /// it the audit takes the receipt from the response and opens it on the
    /// OpenAI API (`POST /v1/receipts/open`), as a client of a Mesh node does.
    pub receipts: Option<String>,
    /// The GGUF the node claims to serve.
    pub claimed_model: PathBuf,
    /// The model id the chat request names. Without it, the first model
    /// the node lists.
    pub model: Option<String>,
    pub prompt: String,
    pub max_tokens: u32,
    /// Challenge this layer alone instead of random ones.
    pub layer: Option<u32>,
    /// Times to compute each challenged layer for the cost report; the
    /// checks use the first.
    pub timing_runs: u32,
    /// Where to write the audit's evidence, which `recheck` checks again.
    pub evidence: Option<PathBuf>,
    /// How the request asks the node to choose its tokens.
    pub sampling: AuditSampling,
}

/// Where the audit opens its challenge.
enum OpenRoute {
    /// A stage's receipts API.
    ReceiptsApi(String),
    /// The OpenAI API, for a bound receipt. The body names the model, and
    /// `x-mesh-target` the node key, so a Mesh node routes the challenge to
    /// the peer that signed the receipt; a plain node ignores both.
    OpenAi {
        api: String,
        model: String,
        node_key: Option<String>,
    },
}

/// Audits one turn, prints the report, and ends with the audit's outcome:
/// `Ok` for a pass, else an error that carries the outcome's exit code
/// ([`exit_code`]): 1 for a failed audit, 2 for no audit, 3 for a node
/// that did not answer the challenge.
///
/// Whatever keeps this machine from auditing is no audit: the claimed GGUF
/// cannot be opened, the node cannot be reached for the request, or its
/// response carries no receipt. None of them is a failed audit.
pub fn run(command: AuditReceiptCommand, load: LoadClaimedModel<'_>) -> Result<()> {
    let evidence = command.evidence.clone();
    let audit = audit(command, load)
        .unwrap_or_else(|error| TurnAudit::NoAudit(NoAuditOutput::new("", format!("{error:#}"))));
    conclude(audit, evidence.as_deref())
}

/// Prints an audit's report, keeps its evidence at `evidence`, and ends
/// with its outcome.
fn conclude(audit: TurnAudit, evidence: Option<&std::path::Path>) -> Result<()> {
    let outcome = audit.outcome();
    let reason = match &audit {
        TurnAudit::NoAudit(output) => {
            crate::console::present(output, |out| report_text::render_no_audit(out, output))?;
            output.reason.clone()
        }
        TurnAudit::Unanswered(output) => {
            crate::console::present(output, |out| report_text::render_unanswered(out, output))?;
            output.reason.clone()
        }
        TurnAudit::Audited {
            output,
            evidence: kept,
        } => {
            if let (Some(path), Some(kept)) = (evidence, kept) {
                std::fs::write(path, serde_json::to_vec_pretty(kept)?)
                    .with_context(|| format!("write evidence {}", path.display()))?;
            }
            crate::console::present(&**output, |out| report_text::render(out, output))?;
            let mut reasons = output.failures();
            if reasons.is_empty() {
                reasons = output.unchecked();
            }
            reasons.join("; ")
        }
    };
    AuditExit::result(outcome, reason)
}

/// Sends the request and audits the turn. An error is this machine's or
/// the request's: nothing was audited.
fn audit(command: AuditReceiptCommand, load: LoadClaimedModel<'_>) -> Result<TurnAudit> {
    let setup = ClaimedSetup::open(
        &command.claimed_model,
        &claimed_file::default_cache_root()?,
        load,
    )?;
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()?;
    let receipt_id = uuid::Uuid::new_v4().hyphenated().to_string();
    // a fresh key per audit: the node binds it before the client sees the root
    let key = vrf::SecretKey::generate(&mut rand::rng());
    let sampling = command.sampling.with_seed();
    let (response, carried) = chat(&client, &command, &sampling, &receipt_id, &key)?;
    let receipts = command
        .receipts
        .as_deref()
        .map(|url| url.trim_end_matches('/'));
    // a node that predates the receipt in the response body serves it only
    // from its receipts API
    let receipt = match (carried, receipts) {
        (Some(receipt), _) => receipt,
        (None, Some(receipts)) => fetch_receipt(&client, receipts, &receipt_id)?,
        (None, None) => bail!(
            "the response carried no receipt; pass --receipts to fetch it from the node's receipts API"
        ),
    };
    let route = match receipts {
        Some(receipts) => OpenRoute::ReceiptsApi(receipts.to_owned()),
        None => OpenRoute::OpenAi {
            api: command.api.trim_end_matches('/').to_owned(),
            model: response.model.clone(),
            node_key: receipt.binding.as_ref().map(|b| b.node_key.clone()),
        },
    };
    let opener = HttpOpener {
        client: &client,
        route,
    };
    let turn = ReceivedTurn {
        receipt,
        key: &key,
        response: Some((&command.prompt, &response)),
        // what `chat` sent
        sampling: sampling.requested(),
        asked_id: Some(&receipt_id),
    };
    let options = AuditOptions {
        layer: command.layer,
        timing_runs: command.timing_runs,
    };
    Ok(turn_audit::audit_turn(&setup, turn, &opener, &options))
}

/// Opens challenges over HTTP: on a stage's receipts API, or on the OpenAI
/// API of a node.
pub struct HttpOpener<'a> {
    client: &'a reqwest::blocking::Client,
    route: OpenRoute,
}

impl OpenChallenge for HttpOpener<'_> {
    fn open(&self, challenge: &Challenge) -> Result<Opened> {
        let started = Instant::now();
        let (opening, bytes, signature) = self.post("open", challenge)?;
        Ok(Opened {
            opening,
            bytes,
            signature,
            seconds: started.elapsed().as_secs_f64(),
        })
    }

    fn rows(&self, challenge: &BlockChallenge) -> Result<OpenedRows> {
        let started = Instant::now();
        let (rows, bytes, signature) = self.post("rows", challenge)?;
        Ok(OpenedRows {
            rows,
            bytes,
            signature,
            seconds: started.elapsed().as_secs_f64(),
        })
    }
}

impl HttpOpener<'_> {
    /// Posts `body` to the route's `action` endpoint, and returns the
    /// decoded answer, its bytes, and the node's signature of them. A busy
    /// node is asked again ([`busy_retry`]).
    fn post<B: serde::Serialize, T: serde::de::DeserializeOwned>(
        &self,
        action: &str,
        body: &B,
    ) -> Result<(T, Vec<u8>, Option<String>)> {
        let started = Instant::now();
        let response = loop {
            let response = self.request(action, body)?.send()?;
            let retry_after = response
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|v| v.to_str().ok());
            match busy_retry::pause(response.status(), retry_after, started.elapsed()) {
                Some(pause) => std::thread::sleep(pause),
                None => break response,
            }
        };
        let signature = response
            .headers()
            .get(turn::OPENING_SIGNATURE_HEADER)
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned);
        let (value, bytes) = read_bytes(response)?;
        Ok((value, bytes, signature))
    }

    /// The request that posts `body` to the route's `action` endpoint.
    fn request<B: serde::Serialize>(
        &self,
        action: &str,
        body: &B,
    ) -> Result<reqwest::blocking::RequestBuilder> {
        let request = match &self.route {
            OpenRoute::ReceiptsApi(receipts) => self
                .client
                .post(format!("{receipts}/receipts/{action}"))
                .json(body),
            OpenRoute::OpenAi {
                api,
                model,
                node_key,
            } => {
                let mut body = serde_json::to_value(body)?;
                body["model"] = model.clone().into();
                let request = self
                    .client
                    .post(format!("{api}/v1/receipts/{action}"))
                    .json(&body);
                match node_key {
                    Some(key) => request.header("x-mesh-target", key),
                    None => request,
                }
            }
        };
        Ok(request)
    }
}

fn chat(
    client: &reqwest::blocking::Client,
    command: &AuditReceiptCommand,
    sampling: &AuditSampling,
    receipt_id: &str,
    key: &vrf::SecretKey,
) -> Result<(ChatReply, Option<Receipt>)> {
    let api = command.api.trim_end_matches('/');
    let listed: serde_json::Value;
    let model = match &command.model {
        Some(model) => model.as_str(),
        None => {
            listed = get_json(client.get(format!("{api}/v1/models")))?;
            listed["data"][0]["id"]
                .as_str()
                .context("the node lists no model")?
        }
    };
    let mut body = serde_json::json!({
        "model": model,
        "messages": [{ "role": "user", "content": command.prompt }],
        "max_tokens": command.max_tokens,
        "stream": false,
    });
    sampling.apply(&mut body);
    let value: serde_json::Value = get_json(
        client
            .post(format!("{api}/v1/chat/completions"))
            .header("x-request-id", receipt_id)
            .header("x-receipt-client-key", hex::encode(key.public()))
            .json(&body),
    )?;
    let message = &value["choices"][0]["message"];
    let text = |field: &str| message[field].as_str().unwrap_or_default().to_string();
    let receipt = match value.get("receipt") {
        Some(receipt) => {
            Some(serde_json::from_value(receipt.clone()).context("decode the response's receipt")?)
        }
        None => None,
    };
    let reply = ChatReply {
        model: model.to_owned(),
        content: text("content"),
        reasoning: text("reasoning_content"),
        cut_short: value["choices"][0]["finish_reason"] == "length",
    };
    Ok((reply, receipt))
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
            .send()
            .with_context(|| {
                format!(
                    "no receipts API at {receipts}: a stage that issues no receipts serves none, and says why in a warning when it starts"
                )
            })?;
        if response.status() != StatusCode::NOT_FOUND || Instant::now() >= deadline {
            return decode(response);
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// The reply decoded, and its body as received.
fn read_bytes<T: serde::de::DeserializeOwned>(
    response: reqwest::blocking::Response,
) -> Result<(T, Vec<u8>)> {
    let status = response.status();
    let body = response.bytes()?.to_vec();
    if !status.is_success() {
        bail!("{status}: {}", String::from_utf8_lossy(&body));
    }
    let value = serde_json::from_slice(&body)
        .with_context(|| format!("decode reply: {}", String::from_utf8_lossy(&body)))?;
    Ok((value, body))
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

mod audit_exit;
mod audit_sampling;
mod busy_retry;
mod recheck;

pub use audit_exit::{AuditExit, exit_code};
pub use audit_sampling::AuditSampling;
pub use recheck::{RecheckReceiptCommand, recheck};
