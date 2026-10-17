//! The live referee call for a twin pair this node asked (the requester).
//!
//! A pair is two of this node's own exchanges a client marked with one twin
//! bracket id, whose texts this node kept (`exchange_text`), each answered by
//! a provider whose signed half this node holds (pushed to it). With the pair
//! complete, [`consider`] applies the rules of `request` (the trigger, the
//! one-call cap) and, when a referee is due:
//!
//! 1. chooses one (`select`): the eligible nodes are read from this node's
//!    own records: the announced keys, the model and weights each peer's held
//!    halves say it serves, the host's blocks, and this node's own counts;
//! 2. asks it to re-answer the twins' request through the host
//!    (`x-mesh-target`, temperature 0, the twins' seed, a `referee-` nonce);
//! 3. sends it the adjudicate request (both halves, its own answer, the tier
//!    it was chosen from), and takes back a verdict only when that referee
//!    signed it at that tier;
//! 4. holds the verdict itself and delivers it to both providers.
//!
//! Without a twin bracket id from the host nothing here runs: no pair is
//! ever guessed from timing, and the page says no client marked them as a pair.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use mesh_llm_plugin::PluginContext;
use serde_json::{json, Value};

use super::bar::{Clock, Recorded, SystemClock};
use super::half::{model_hash, served_by, weights_digest, Half};
use super::request::{self, Book, Called, Pair, Twin};
use super::select::{random_draw, select, Candidate};
use crate::capsule_emit::CapsuleState;

/// The host's OpenAI-compatible API, through which a re-answer is routed to
/// the chosen referee. Unset: [`DEFAULT_OPENAI_API`].
pub const ENV_OPENAI_API: &str = "CAPSULES_OPENAI_API_URL";
pub const DEFAULT_OPENAI_API: &str = "http://127.0.0.1:9337";
/// Routes a request to one named node.
pub const MESH_TARGET_HEADER: &str = "x-mesh-target";
/// The client nonce the referee's plugin seals, so its record of the call is
/// findable; every referee call's nonce starts with [`REFEREE_NONCE_PREFIX`].
pub const CLIENT_NONCE_HEADER: &str = "X-Capsule-Client-Nonce";
pub const REFEREE_NONCE_PREFIX: &str = "referee-";
/// How long a re-answer may take.
const REANSWER_TIMEOUT: Duration = Duration::from_secs(30);

/// One twin as the live call needs it: the provider's signed half and the
/// bodies this node kept.
#[derive(Debug, Clone)]
pub struct LiveTwin {
    pub record: Value,
    pub request_body: Value,
    pub response_body: Value,
    pub twin: Twin,
}

#[derive(Debug, Clone)]
pub struct LivePair {
    pub pair: Pair,
    pub twins: [LiveTwin; 2],
}

fn json_lines(path: &Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .map(|t| {
            t.lines()
                .filter_map(|l| serde_json::from_str(l).ok())
                .collect()
        })
        .unwrap_or_default()
}

fn poc_text<'a>(record: &'a Value, pointer: &str) -> Option<&'a str> {
    record
        .pointer("/model_attestation/compute_attestation/x-mesh-poc-v1")?
        .pointer(pointer)?
        .as_str()
        .filter(|s| !s.is_empty())
}

/// The kept texts of this node's exchanges in `bracket`.
fn kept_texts(ledger_dir: &Path, bracket: &str) -> Vec<Value> {
    let Ok(entries) = std::fs::read_dir(ledger_dir.join("disclosures").join("by-exchange")) else {
        return Vec::new();
    };
    let mut texts: Vec<Value> = entries
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().ends_with(".json"))
        .filter_map(|e| std::fs::read(e.path()).ok())
        .filter_map(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .filter(|doc| doc.get("twin_bracket_id").and_then(Value::as_str) == Some(bracket))
        .collect();
    // In exchange id order, so a pair always reads the same way round.
    texts.sort_by(|a, b| a["exchange_id"].as_str().cmp(&b["exchange_id"].as_str()));
    texts
}

/// The pair in `bracket`, when it is complete here: two kept exchanges, each
/// with this node's own record naming its provider, and that provider's
/// signed half held. `None` until then.
pub fn assemble(ledger_dir: &Path, bracket: &str) -> Option<LivePair> {
    let texts = kept_texts(ledger_dir, bracket);
    let [first, second] = texts.as_slice() else {
        return None;
    };
    let own = crate::evidence_panes::read_capsule_records(ledger_dir);
    let held = json_lines(&ledger_dir.join(crate::record_push_receive::RECEIVED_CAPSULES_FILENAME));
    let twin = |text: &Value| -> Option<(String, LiveTwin)> {
        let exchange = text.get("exchange_id")?.as_str()?;
        let mine = own
            .iter()
            .rev()
            .find(|r| poc_text(r, "/serving_provenance/exchange_id") == Some(exchange))?;
        let provider = poc_text(mine, "/serving_provenance/served_by_node_id")?;
        let request_digest = mine
            .pointer("/effect/request_digest")?
            .as_str()?
            .to_string();
        let record = held
            .iter()
            .rev()
            .find(|r| {
                served_by(r) == Some(provider)
                    && r.pointer("/effect/request_digest").and_then(Value::as_str)
                        == Some(&request_digest)
            })?
            .clone();
        let request_body = text.get("request_body").filter(|b| b.is_object())?.clone();
        let response_body = text.get("response_body").filter(|b| b.is_object())?.clone();
        let half = Half {
            capsule: record.clone(),
            request_body: request_body.clone(),
            response_body: Some(response_body.clone()),
            response_text: None,
            node_id: Some(provider.to_string()),
            weights_digest: weights_digest(&record).map(str::to_string),
        };
        let twin = Twin {
            node_id: provider.to_string(),
            capsule_id: half.capsule_id().to_string(),
            text: half.text().to_string(),
            sampled: half.sampled(),
            model_hash: model_hash(&record).map(str::to_string),
            weights_digest: half.weights_digest.clone(),
        };
        Some((
            request_digest,
            LiveTwin {
                record,
                request_body,
                response_body,
                twin,
            },
        ))
    };
    let (digest_a, a) = twin(first)?;
    let (_, b) = twin(second)?;
    Some(LivePair {
        pair: Pair {
            twin_bracket_id: Some(bracket.to_string()),
            request_digest: digest_a,
            twins: [a.twin.clone(), b.twin.clone()],
        },
        twins: [a, b],
    })
}

/// The nodes this node could ask, as its own records know them: each peer
/// with an announced key, serving the model and weights its latest held half
/// says, blocked when the host says so.
fn candidates(
    ledger_dir: &Path,
    peer_keys: Option<&str>,
    blocked: &[String],
    self_id: &str,
) -> Vec<Candidate> {
    let announced: Vec<String> = peer_keys
        .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
        .and_then(|v| v.as_object().map(|o| o.keys().cloned().collect()))
        .unwrap_or_default();
    let held = json_lines(&ledger_dir.join(crate::record_push_receive::RECEIVED_CAPSULES_FILENAME));
    let mut serving: HashMap<&str, (Option<&str>, Option<&str>)> = HashMap::new();
    for record in &held {
        if let Some(server) = served_by(record) {
            serving.insert(server, (model_hash(record), weights_digest(record)));
        }
    }
    announced
        .iter()
        .filter(|node| node.as_str() != self_id)
        .map(|node| {
            let (model, weights) = serving.get(node.as_str()).copied().unwrap_or((None, None));
            Candidate {
                node_id: node.clone(),
                model_hash: model.map(str::to_string),
                weights_digest: weights.map(str::to_string),
                announced_key: crate::peer_keys::announced_key_in(peer_keys, node).is_some(),
                blocked: blocked.contains(node),
            }
        })
        .collect()
}

/// This node's counted verdicts, per judged node and model, with the time it
/// recorded each.
fn recorded_verdicts(ledger_dir: &Path) -> Vec<Recorded> {
    let records = crate::evidence_panes::read_capsule_records(ledger_dir);
    let by_peer =
        crate::verdict_counts::fold(&records, &crate::verdict_counts::read_requested(ledger_dir));
    by_peer
        .into_iter()
        .flat_map(|(peer, verdicts)| {
            verdicts.into_iter().filter_map(move |v| {
                Some(Recorded {
                    node_id: peer.clone(),
                    model_hash: v.model_hash?,
                    bucket: v.bucket.to_string(),
                    recorded_at: chrono::DateTime::parse_from_rfc3339(v.recorded_at.as_deref()?)
                        .ok()?
                        .into(),
                })
            })
        })
        .collect()
}

/// The peers this node's host has stopped routing to, as the host published
/// them (`crate::peer_blocks_seen`); never read from the host's operator routes.
fn host_blocks(ledger_dir: &Path) -> Vec<String> {
    crate::peer_blocks_seen::blocked_now(ledger_dir)
}

/// The re-answer request: the twins' own request, greedy, far enough to pass
/// the word where they differ.
pub fn reanswer_body(twin: &LiveTwin, divergence: usize) -> Value {
    let seed = twin
        .record
        .pointer("/model_attestation/compute_attestation/decoding/seed")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    json!({
        "model": twin.request_body.get("model").cloned().unwrap_or(Value::Null),
        "messages": twin.request_body.get("messages").cloned().unwrap_or_else(|| json!([])),
        "temperature": 0,
        "max_tokens": (8 * (divergence + 2)).min(256),
        "seed": seed,
    })
}

async fn reanswer(referee: &str, body: &Value) -> Option<Value> {
    let api = crate::settings::var(ENV_OPENAI_API)
        .ok()
        .filter(|u| !u.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_OPENAI_API.to_string());
    let nonce = format!(
        "{REFEREE_NONCE_PREFIX}{}",
        &crate::producer::capsule::fresh_store_nonce()[..32]
    );
    let response = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", api.trim_end_matches('/')))
        .header(MESH_TARGET_HEADER, referee)
        .header(CLIENT_NONCE_HEADER, nonce)
        .timeout(REANSWER_TIMEOUT)
        .json(body)
        .send()
        .await
        .ok()?
        .error_for_status()
        .ok()?;
    response.json::<Value>().await.ok().filter(Value::is_object)
}

/// The adjudicate request sent to the referee.
pub fn adjudicate_request(
    live: &LivePair,
    tier: u64,
    answer_request: &Value,
    answer_response: &Value,
) -> Value {
    let half = |t: &LiveTwin| json!({"capsule": t.record, "request_body": t.request_body, "response_body": t.response_body});
    json!({
        "subject": {"kind": crate::referee::service::ADJUDICATE_SUBJECT_KIND},
        "selection_tier": tier,
        "twin_bracket_id": live.pair.twin_bracket_id,
        "halves": [half(&live.twins[0]), half(&live.twins[1])],
        "referee_answer": {"request_body": answer_request, "response_body": answer_response},
    })
}

/// Ask `referee`, chosen from `tier`: its re-answer, then its signed verdict.
/// The verdict is taken only when that referee signed it at that tier.
async fn ask(
    context: &mut PluginContext<'_>,
    ledger_dir: &Path,
    peer_keys: Option<&str>,
    live: &LivePair,
    referee: &str,
    tier: u64,
) -> (Called, Option<Value>) {
    let [a, b] = &live.twins;
    let divergence = super::verdict::compare_transcripts(&a.twin.text, &b.twin.text)
        .divergence_index
        .unwrap_or(0);
    let body = reanswer_body(a, divergence);
    let Some(response) = reanswer(referee, &body).await else {
        return (Called::Unreachable, None);
    };
    let request = adjudicate_request(live, tier, &body, &response);
    if let Some(asked) = crate::mesh_evidence_bridge::requested_adjudication(referee, &request) {
        if let Err(error) =
            crate::mesh_evidence_bridge::record_requested_adjudication(ledger_dir, &asked)
        {
            tracing::warn!(%error, "the adjudicate request could not be recorded; not sent");
            return (Called::Unreachable, None);
        }
    }
    use crate::mesh_evidence_bridge::SendError;
    let reply = match crate::mesh_evidence_bridge::send_evidence_request(context, referee, &request)
        .await
    {
        Ok((_, reply)) => reply,
        // No plugin stream to open: nothing there can sign a verdict.
        Err(SendError::NotOpened(_)) => return (Called::CannotSign, None),
        Err(SendError::NoAnswer(_) | SendError::Malformed(_)) => {
            return (Called::Unreachable, None)
        }
    };
    let capsule = reply.get("verdict_capsule").cloned().unwrap_or(Value::Null);
    match crate::referee::hold::verdict_facts(&capsule, peer_keys) {
        Some(facts) if facts.referee_node_id == referee && facts.selection_tier == Some(tier) => {
            (Called::Verdict(facts.verdict), Some(capsule))
        }
        // A refusal, or anything not signed by that referee at that tier.
        _ => (Called::CannotSign, None),
    }
}

/// What [`consider`] came to.
#[derive(Debug, Clone, PartialEq)]
pub enum Considered {
    /// The pair's row after this attempt.
    Row(Value),
    /// The pair is not complete here yet.
    NotComplete,
    /// The pair is being decided already (the exchange handler, or the
    /// operator asking again): this attempt did nothing.
    Deciding,
    /// The call could not be recorded before it was made, so it was not made.
    NotRecorded,
}

/// A pair whose decision this attempt holds: the pair, its book, the gate's
/// answer (`None`: a referee is due), and the marker, released on drop.
pub struct Begun {
    pub live: LivePair,
    pub book: Book,
    pub gated: Option<request::Attempted>,
    /// Held to the end of the decision.
    pub deciding: request::Deciding,
}

/// The first step of [`consider`]: assemble the pair, take its marker, and
/// apply the gate. `Err` when this attempt goes no further: the pair is not
/// complete here, or is being decided already. Synchronous and without the
/// host, so a test can hold the marker and see a second attempt return at
/// once.
pub fn begin(
    ledger_dir: &Path,
    bracket: &str,
    on: bool,
    manual: bool,
) -> Result<Begun, Considered> {
    let live = assemble(ledger_dir, bracket).ok_or(Considered::NotComplete)?;
    let deciding = request::Deciding::claim(&live.pair).ok_or(Considered::Deciding)?;
    let book = Book::load(ledger_dir);
    let gated = book.gate(on, &live.pair, manual);
    Ok(Begun {
        live,
        book,
        gated,
        deciding,
    })
}

/// Apply the rules to the pair in `bracket` and, when a referee is due, ask
/// one. `manual`: the operator asked again. One decision per pair at a time,
/// and the pair's one call is recorded as in flight before it is made.
pub async fn consider(
    context: &mut PluginContext<'_>,
    capsules: &Arc<CapsuleState>,
    bracket: &str,
    self_id: &str,
    manual: bool,
) -> Considered {
    let ledger_dir = capsules.ledger_dir().to_path_buf();
    let on = request::adjudicate_differing_twins();
    let Begun {
        live,
        mut book,
        gated,
        deciding: _held_until_decided,
    } = match begin(&ledger_dir, bracket, on, manual) {
        Ok(begun) => begun,
        Err(considered) => return considered,
    };
    let peer_keys = crate::peer_keys::registry();
    let attempted = match gated {
        Some(done) => done,
        None => {
            let [a, _] = &live.pair.twins;
            let blocked = host_blocks(&ledger_dir);
            let pool = select(
                SystemClock.now(),
                super::bar::bar_days(),
                a.model_hash.as_deref().unwrap_or_default(),
                a.weights_digest.as_deref().unwrap_or_default(),
                [&live.pair.twins[0].node_id, &live.pair.twins[1].node_id],
                &candidates(&ledger_dir, peer_keys.as_deref(), &blocked, self_id),
                &recorded_verdicts(&ledger_dir),
            );
            match (pool.pick(&mut random_draw), pool.tier) {
                (Some(referee), Some(tier)) => {
                    let referee = referee.to_string();
                    let at = crate::producer::timestamp::utc_now_iso8601();
                    if let Err(error) =
                        request::record_in_flight(&ledger_dir, &live.pair, &referee, tier, &at)
                    {
                        tracing::warn!(%error, bracket, "the call could not be recorded; not made");
                        return Considered::NotRecorded;
                    }
                    let (called, verdict) = ask(
                        context,
                        &ledger_dir,
                        peer_keys.as_deref(),
                        &live,
                        &referee,
                        tier,
                    )
                    .await;
                    if let Some(verdict) = verdict {
                        hold_and_deliver(context, capsules, &live, &referee, self_id, verdict)
                            .await;
                    }
                    book.called(&live.pair, &referee, tier, called)
                }
                _ => book.nobody_eligible(&live.pair),
            }
        }
    };
    let now = crate::producer::timestamp::utc_now_iso8601();
    if let Err(error) = request::record(&ledger_dir, &live.pair, &attempted, &now) {
        tracing::warn!(%error, bracket, "the pair's outcome could not be recorded");
    }
    Considered::Row(attempted.row)
}

/// Hold the verdict here (this node asked for it) and deliver it to both
/// providers; a refused delivery is sealed on this node's chain.
async fn hold_and_deliver(
    context: &mut PluginContext<'_>,
    capsules: &Arc<CapsuleState>,
    live: &LivePair,
    referee: &str,
    self_id: &str,
    verdict: Value,
) {
    let body = json!({ crate::referee::hold::DELIVERY_MARKER: 1, "verdict_capsule": verdict });
    let own_key = hex::encode(capsules.signing_key().verifying_key().to_bytes());
    let peer_keys = crate::peer_keys::registry();
    let door = crate::referee::hold::Node {
        ledger_dir: capsules.ledger_dir(),
        peer_keys: peer_keys.as_deref(),
        own_key_id: &own_key,
    };
    let now = crate::producer::timestamp::utc_now_iso8601();
    match crate::referee::hold::hold_delivered_verdict(&door, &body, referee, &now) {
        Ok(Ok(reply)) => {
            if let Err(error) =
                crate::record_push_bridge::seal_received_verdict(capsules, &reply["adjudication"])
            {
                tracing::warn!(%error, "a verdict this node asked for was held, but its record did not seal");
            }
        }
        Ok(Err(reason)) => tracing::warn!(reason, "the verdict this node asked for was not held"),
        Err(error) => tracing::warn!(%error, "the verdict this node asked for could not be held"),
    }
    if !crate::share_policy::adjudications_delivered() {
        tracing::info!("share_adjudications is off: the verdict is held here and not delivered");
        crate::routing_rule::mark_due();
        return;
    }
    for twin in &live.twins {
        if let Err(error) = crate::adjudication_records::push_verdict(
            context,
            capsules,
            &twin.twin.node_id,
            self_id,
            &body["verdict_capsule"],
        )
        .await
        {
            tracing::warn!(%error, peer = %twin.twin.node_id, "a verdict was not delivered");
        }
    }
    crate::routing_rule::mark_due();
}

/// The operator's "ask again" tool.
pub const ASK_AGAIN_OPERATION: &str = "referee_ask_again";

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct AskAgainArgs {
    /// The pair's twin bracket id, as a client marked it.
    pub twin_bracket_id: String,
}

/// The operator asks again about one pair (`manual`).
pub async fn ask_again(
    args: AskAgainArgs,
    context: &mut PluginContext<'_>,
    capsules: Arc<CapsuleState>,
    self_id: Option<String>,
) -> mesh_llm_plugin::PluginResult<Value> {
    let Some(self_id) = self_id else {
        return Err(mesh_llm_plugin::PluginError::invalid_request(
            "this node does not know its own peer id yet",
        ));
    };
    match consider(context, &capsules, &args.twin_bracket_id, &self_id, true).await {
        Considered::Row(row) => Ok(row),
        Considered::Deciding => Err(mesh_llm_plugin::PluginError::invalid_request(
            "this pair is being decided right now; ask again once that is done",
        )),
        Considered::NotRecorded => Err(mesh_llm_plugin::PluginError::internal(
            "the call could not be recorded beside the ledger, so no referee was asked",
        )),
        Considered::NotComplete => Err(mesh_llm_plugin::PluginError::invalid_request(
            "no complete twin pair with that bracket id is held here: both twins' kept texts and \
             both providers' signed halves are needed",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record_push_parity::read_json;
    use crate::referee::parity::{referee_dir, CaseNode, NOW};

    /// A requester's ledger holding the corpus pair `contradicts_b`: its kept
    /// texts, its own records naming each provider, and the providers'
    /// signed halves pushed to it.
    fn requester_with_the_pair() -> (tempfile::TempDir, Value) {
        let corpus = read_json(&referee_dir().join("corpus/service.json"));
        let case = corpus["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == json!("contradicts_b"))
            .unwrap()
            .clone();
        let request: Value =
            serde_json::from_str(case["requests"][0]["body"].as_str().unwrap()).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let texts = dir.path().join("disclosures").join("by-exchange");
        std::fs::create_dir_all(&texts).unwrap();
        let (mut own, mut held) = (String::new(), String::new());
        for (i, half) in request["halves"].as_array().unwrap().iter().enumerate() {
            let exchange = format!("exchange-{i}");
            let provider = served_by(&half["capsule"]).unwrap();
            std::fs::write(
                texts.join(format!("{exchange}.json")),
                json!({"v": 1, "exchange_id": exchange, "twin_bracket_id": "bracket-1",
                       "request_body": half["request_body"], "response_body": half["response_body"]})
                .to_string(),
            )
            .unwrap();
            own += &(json!({"capsule_id": format!("own-{i}"), "effect": {"request_digest": half["capsule"]["effect"]["request_digest"]},
                "model_attestation": {"compute_attestation": {"x-mesh-poc-v1": {"serving_provenance": {
                    "exchange_id": exchange, "served_by_node_id": provider, "twin_bracket_id": "bracket-1"}}}}})
            .to_string()
                + "\n");
            held += &(half["capsule"].to_string() + "\n");
        }
        std::fs::write(dir.path().join("capsules.jsonl"), own).unwrap();
        std::fs::write(
            dir.path()
                .join(crate::record_push_receive::RECEIVED_CAPSULES_FILENAME),
            held,
        )
        .unwrap();
        (dir, case)
    }

    #[test]
    fn a_complete_pair_is_assembled_from_what_this_node_holds() {
        let (dir, _) = requester_with_the_pair();
        let live = assemble(dir.path(), "bracket-1").expect("the pair is complete");
        assert_eq!(live.pair.twins[0].node_id, "node-a");
        assert_eq!(live.pair.twins[1].node_id, "node-b");
        assert!(live
            .pair
            .twins
            .iter()
            .all(|t| t.model_hash.is_some() && t.weights_digest.is_some()));
        assert!(
            super::super::request::before_asking(true, &live.pair).is_none(),
            "a referee is due"
        );
        assert!(
            assemble(dir.path(), "bracket-2").is_none(),
            "another bracket holds nothing"
        );
        std::fs::remove_file(
            dir.path()
                .join(crate::record_push_receive::RECEIVED_CAPSULES_FILENAME),
        )
        .unwrap();
        assert!(
            assemble(dir.path(), "bracket-1").is_none(),
            "incomplete until both halves are held"
        );
    }

    /// The request this node would send is one the referee's door signs.
    #[test]
    fn the_adjudicate_request_is_one_the_referee_signs() {
        let (dir, case) = requester_with_the_pair();
        let live = assemble(dir.path(), "bracket-1").unwrap();
        let sent: Value =
            serde_json::from_str(case["requests"][0]["body"].as_str().unwrap()).unwrap();
        let answer = &sent["referee_answer"];
        let request =
            adjudicate_request(&live, 2, &answer["request_body"], &answer["response_body"]);
        let referee = CaseNode::new(&case["node"]);
        let door = crate::referee::service::Referee {
            ledger_dir: referee.ledger_dir(),
            signing_key: &referee.key,
            peer_keys: referee.peer_keys.as_deref(),
        };
        let body = serde_json::to_vec(&request).unwrap();
        let reply = crate::referee::service::handle_adjudicate_request(&door, &body, NOW).unwrap();
        assert_eq!(reply["verdict"], json!("contradicted:node-b"), "{reply}");
        let facts = crate::referee::hold::verdict_facts(
            &reply["verdict_capsule"],
            referee.peer_keys.as_deref(),
        )
        .expect("signed by the referee it names");
        assert_eq!(
            facts.selection_tier,
            Some(2),
            "the tier this node asked at is sealed"
        );
    }

    /// The referee's door checks the requester's signature before it reads
    /// anything: a request signed for another node (relayed), one changed
    /// after signing, one signed for another channel (an Evidence Request, a
    /// `ledger-fetch/1` fetch: ruling u338), or one replayed is refused
    /// `not_authorized`, even about a pair it already ruled on, where it
    /// would otherwise return the held verdict. Unsigned, or signed for this
    /// node under the referee's tag, it is answered.
    #[test]
    fn the_referee_refuses_a_failed_signature_before_it_reads_anything() {
        use crate::referee::service::{
            handle_asked_adjudicate_request, Asked, REASON_NOT_AUTHORIZED, REASON_REQUEST_MALFORMED,
        };
        use crate::requester_auth::{Keys, ReplayCache};
        let (dir, case) = requester_with_the_pair();
        let live = assemble(dir.path(), "bracket-1").unwrap();
        let sent: Value =
            serde_json::from_str(case["requests"][0]["body"].as_str().unwrap()).unwrap();
        let answer = &sent["referee_answer"];
        let request =
            adjudicate_request(&live, 2, &answer["request_body"], &answer["response_body"]);
        let referee = CaseNode::new(&case["node"]);
        let door = crate::referee::service::Referee {
            ledger_dir: referee.ledger_dir(),
            signing_key: &referee.key,
            peer_keys: referee.peer_keys.as_deref(),
        };
        let own_node =
            crate::peer_keys::peer_id_for_key_in(referee.peer_keys.as_deref(), &referee.key_id())
                .expect("the referee is named");
        let asker = crate::evidence_request_parity::asker_key("node-y");
        let at: chrono::DateTime<chrono::Utc> =
            chrono::DateTime::parse_from_rfc3339(NOW).unwrap().into();
        use crate::requester_auth::Channel;
        let signed_on = |channel: Channel, audience: &str| {
            let mut request = request.clone();
            crate::requester_auth::sign(channel, &mut request, &asker, at, audience);
            request
        };
        let signed_for = |audience: &str| signed_on(Channel::RefereeAdjudicate, audience);
        let announced = crate::evidence_request_parity::announced_keys();
        let replay = ReplayCache::default();
        let asked = Asked {
            keys: Keys {
                announced: &announced,
                configured: None,
            },
            replay: &replay,
            own_node: Some(&own_node),
        };
        let ask = |request: &Value| {
            let body = serde_json::to_vec(request).unwrap();
            handle_asked_adjudicate_request(&door, &body, NOW, &asked).unwrap()
        };
        let issued = referee
            .ledger_dir()
            .join(crate::referee::service::ISSUED_ADJUDICATIONS_FILENAME);

        let relayed = signed_for("node-x");
        let as_evidence = signed_on(Channel::EvidenceRequest, &own_node);
        let as_fetch = signed_on(Channel::LedgerFetch, &own_node);
        let mut changed = signed_for(&own_node);
        changed["selection_tier"] = json!(1);
        let mut malformed = request.clone();
        malformed["requester_auth"] = json!("x");
        for (request, reason) in [
            (&relayed, REASON_NOT_AUTHORIZED),
            (&changed, REASON_NOT_AUTHORIZED),
            (&as_evidence, REASON_NOT_AUTHORIZED),
            (&as_fetch, REASON_NOT_AUTHORIZED),
            (&malformed, REASON_REQUEST_MALFORMED),
        ] {
            let reply = ask(request);
            assert_eq!(reply["reason"], json!(reason), "{reply}");
            assert!(!issued.exists(), "nothing was ruled on");
        }

        let signed = signed_for(&own_node);
        let reply = ask(&signed);
        assert_eq!(reply["verdict"], json!("contradicted:node-b"), "{reply}");
        assert!(issued.exists());
        // The pair is ruled on now: each of these would get the held verdict
        // back if the door read the ledger before the signature.
        for request in [&signed, &relayed, &changed, &as_evidence, &as_fetch] {
            let reply = ask(request);
            assert_eq!(reply["reason"], json!(REASON_NOT_AUTHORIZED), "{reply}");
        }
        assert_eq!(
            ask(&request)["verdict"],
            json!("contradicted:node-b"),
            "unsigned, the held verdict"
        );
        let unbound = {
            let mut request = request.clone();
            let key = ed25519_dalek::SigningKey::from_bytes(&[9; 32]);
            crate::requester_auth::sign(
                Channel::RefereeAdjudicate,
                &mut request,
                &key,
                at,
                &own_node,
            );
            request
        };
        assert_eq!(
            ask(&unbound)["verdict"],
            json!("contradicted:node-b"),
            "a key no host binds reads as unsigned"
        );
    }

    /// The operator asking again while the exchange handler is deciding the
    /// same pair returns at once: no second decision, no second call.
    #[test]
    fn asking_again_while_a_decision_is_in_flight_returns_at_once() {
        let (dir, _) = requester_with_the_pair();
        let first = begin(dir.path(), "bracket-1", true, false).expect("free to decide");
        assert!(first.gated.is_none(), "a referee is due");
        let started = std::time::Instant::now();
        let again = begin(dir.path(), "bracket-1", true, true);
        assert!(
            matches!(again, Err(Considered::Deciding)),
            "the second attempt goes no further"
        );
        assert!(
            started.elapsed() < std::time::Duration::from_secs(1),
            "and does not wait"
        );
        drop(first);
        let after =
            begin(dir.path(), "bracket-1", true, true).expect("free once the first is done");
        assert!(after.gated.is_none());
        assert!(matches!(
            begin(dir.path(), "bracket-9", true, true),
            Err(Considered::NotComplete)
        ));
    }

    #[test]
    fn the_reanswer_is_greedy_on_the_twins_request() {
        let (dir, _) = requester_with_the_pair();
        let live = assemble(dir.path(), "bracket-1").unwrap();
        let body = reanswer_body(&live.twins[0], 3);
        assert_eq!(body["temperature"], json!(0));
        assert_eq!(body["max_tokens"], json!(40));
        assert_eq!(body["messages"], live.twins[0].request_body["messages"]);
    }
}
