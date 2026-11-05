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
//! signed; it covers every layer of the claimed model; every opened leaf is
//! in its stage tree, and stages agree where they hand over; each picked
//! layer's weight rows, and its glue computed on the opened values; the
//! input of layer 0 is what the receipt's tokens make; the token sampled at
//! each head position is the greedy choice of the claimed output head on
//! the opened final residual; the response is the text of the tokens the
//! receipt says were sampled; and each picked attention head's partials
//! combine to its opened output, and recompute, for blocks this client
//! draws after the opening, from the K and V rows the node opens.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use reqwest::StatusCode;
use skippy_receipts::attention::opening::BlockChallenge;
use skippy_receipts::glue::GlueEvaluator;
use skippy_receipts::turn;
use skippy_receipts::vrf;
use skippy_receipts::wire::{Challenge, Receipt};

use turn_audit::{
    AuditOptions, ClaimedSetup, OpenChallenge, Opened, OpenedRows, ReceivedTurn, TurnAudit,
};

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
    pub prompt: String,
    pub max_tokens: u32,
    /// Challenge this layer alone instead of random ones.
    pub layer: Option<u32>,
    /// Times to compute each challenged layer for the cost report; the
    /// checks use the first.
    pub timing_runs: u32,
    /// Where to write the audit's evidence, which `recheck` checks again.
    pub evidence: Option<PathBuf>,
}

/// Loads the claimed model the native loader reads from a path: the GGUF,
/// or its sparse copy.
pub type LoadClaimedModel<'a> = &'a dyn Fn(PathBuf) -> Box<dyn ClaimedModel>;

/// The ggml types of the KV cache a stage stored one layer's rows in, by
/// the layer's stored rows: K, then V. `None` keeps the default. The
/// auditor builds the layer with them, since llama.cpp builds a layer
/// differently for a quantized cache: it rotates the keys and values.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheTypes {
    pub k: Option<u32>,
    pub v: Option<u32>,
}

/// The claimed GGUF on this machine, as the audit needs it.
pub trait ClaimedModel {
    /// Runs `check` with one layer of the model loaded alone, with a KV
    /// cache of `cache` types.
    fn with_layer(
        &self,
        layer: u32,
        cache: CacheTypes,
        check: &mut dyn FnMut(&dyn GlueEvaluator) -> Result<()>,
    ) -> Result<()>;
    /// The text of `tokens`.
    fn detokenize(&self, tokens: &[i32]) -> Result<String>;
    /// Whether `token` ends generation, so it adds no text to a response.
    fn is_end_of_generation(&self, token: i32) -> Result<bool>;
    /// The runtime the glue evaluation runs, as a receipt names it.
    fn runtime(&self) -> String;
}

/// What the node answered: its content and reasoning text, and the receipt
/// when the response carried it.
pub struct ChatReply {
    /// The model the request named.
    pub model: String,
    pub content: String,
    pub reasoning: String,
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

pub fn run(command: AuditReceiptCommand, load: LoadClaimedModel<'_>) -> Result<()> {
    let setup = ClaimedSetup::open(
        &command.claimed_model,
        &claimed_file::default_cache_root(),
        load,
    )?;
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()?;
    let receipt_id = uuid::Uuid::new_v4().hyphenated().to_string();
    // a fresh key per audit: the node binds it before the client sees the root
    let key = vrf::SecretKey::generate(&mut rand::rng());
    let (response, carried) = chat(&client, &command, &receipt_id, &key)?;
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
        // the audit asks for greedy decoding
        asked_greedy: true,
    };
    let options = AuditOptions {
        layer: command.layer,
        timing_runs: command.timing_runs,
    };
    match turn_audit::audit_turn(&setup, turn, &opener, &options)? {
        TurnAudit::NoAudit(output) => {
            crate::console::present(&output, |out| output::render_no_audit(out, &output))?;
            bail!("the receipt was not audited: {}", output.reason);
        }
        TurnAudit::Audited { output, evidence } => {
            if let (Some(path), Some(evidence)) = (&command.evidence, &evidence) {
                std::fs::write(path, serde_json::to_vec_pretty(evidence)?)
                    .with_context(|| format!("write evidence {}", path.display()))?;
            }
            crate::console::present(&*output, |out| output::render(out, &output))?;
            if !output.pass {
                bail!("receipt audit failed");
            }
            Ok(())
        }
    }
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
    /// decoded answer, its bytes, and the node's signature of them.
    fn post<B: serde::Serialize, T: serde::de::DeserializeOwned>(
        &self,
        action: &str,
        body: &B,
    ) -> Result<(T, Vec<u8>, Option<String>)> {
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
        let response = request.send()?;
        let signature = response
            .headers()
            .get(turn::OPENING_SIGNATURE_HEADER)
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned);
        let (value, bytes) = read_bytes(response)?;
        Ok((value, bytes, signature))
    }
}

fn chat(
    client: &reqwest::blocking::Client,
    command: &AuditReceiptCommand,
    receipt_id: &str,
    key: &vrf::SecretKey,
) -> Result<(ChatReply, Option<Receipt>)> {
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
        // greedy decoding takes the best logit at every token, which a
        // reasoning budget (by default half the tokens, on a thinking
        // model) would not when it closes the thinking
        "reasoning_budget": -1,
        "stream": false,
    });
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
            .send()?;
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

pub mod attention_audit;
pub mod claimed_file;
mod head_key_file;
mod native_model;
mod output;
mod recheck;
pub mod recurrent_audit;
pub mod turn_audit;

pub use native_model::load_claimed_model;
pub use output::{AuditOutput, NoAuditOutput};
pub use recheck::{RecheckReceiptCommand, recheck};
