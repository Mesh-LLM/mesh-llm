//! Replay real recorded open-model traces through the MoA gateway.
//!
//! The fixture in `tests/fixtures/real_traces.json` was recorded from 9
//! open-weight models on OpenRouter (see `evals/moa-openrouter/`). Each case
//! holds every worker's **raw OpenAI-shaped response** — structured
//! `tool_calls`, `content`, and `finish_reason` — plus its observed latency.
//!
//! That matters because the two things this suite is here to protect are
//! exactly the two things you lose if you flatten worker responses to text:
//!
//! 1. `tool_calls` — Together's MoA aggregator reads `.content`, which is
//!    empty when a model returns a tool call, so its synthesis step receives
//!    a list of blank strings on every agentic turn.
//! 2. `finish_reason` — 39 of the recorded responses came back `"length"`,
//!    and 24 of those carry *partial text*. Before truncation was plumbed
//!    through, a half-finished sentence entered arbitration as a normal
//!    answer at the default 0.5 confidence and could be returned verbatim.
//!
//! These replay through the real `HttpBackend`-shaped path: the fixture is
//! served as a JSON body, so `extract_text_from_response` does the parsing
//! under test rather than the test reconstructing its output.
//!
//! Not covered here: the HTTP-400 "Reasoning is mandatory" retry in
//! `HttpBackend`. That needs a live endpoint (verified against
//! minimax-m2.5, which failed 12/12 requests until the thinking flags were
//! dropped) and is not reachable through a `ModelBackend` fake.

use async_trait::async_trait;
use mesh_mixture_of_agents as moa;
use serde_json::{Value, json};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

/// Recorded latencies span 0.5s–30s. Replaying them literally would make
/// this suite take minutes, so compress the timeline while preserving
/// *ordering* — which is what early exit, first-answer grace, and strong
/// patience actually key off.
const LATENCY_DIVISOR: u64 = 40;

// ─── Fixture model ───────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Deserialize)]
struct RecordedWorker {
    model: String,
    elapsed_ms: u64,
    finish_reason: Option<String>,
    content: Option<String>,
    tool_calls: Option<Value>,
    error: Option<String>,
}

impl RecordedWorker {
    /// Rebuild the exact OpenAI-shaped body this model returned.
    fn response_body(&self) -> Value {
        let mut message = json!({ "role": "assistant" });
        let obj = message.as_object_mut().unwrap();
        obj.insert(
            "content".to_string(),
            match &self.content {
                Some(c) => json!(c),
                None => Value::Null,
            },
        );
        if let Some(tcs) = &self.tool_calls {
            obj.insert("tool_calls".to_string(), tcs.clone());
        }
        json!({
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": self.finish_reason.clone().unwrap_or_else(|| "stop".into()),
            }]
        })
    }

    fn truncated(&self) -> bool {
        self.finish_reason.as_deref() == Some("length")
    }

    /// Text a truncated worker would have contributed, if any. Used to assert
    /// it is never echoed back to the caller verbatim.
    fn partial_text(&self) -> Option<&str> {
        if !self.truncated() || self.tool_calls.is_some() {
            return None;
        }
        self.content
            .as_deref()
            .map(str::trim)
            .filter(|t| !t.is_empty())
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
struct RecordedCase {
    id: String,
    scenario: String,
    step: u32,
    has_tools: bool,
    messages: Vec<Value>,
    workers: Vec<RecordedWorker>,
}

#[derive(Debug, serde::Deserialize)]
struct Fixture {
    cases: Vec<RecordedCase>,
}

fn load_fixture() -> Fixture {
    let raw = include_str!("fixtures/real_traces.json");
    serde_json::from_str(raw).expect("real_traces.json should parse")
}

// ─── Replay backend ──────────────────────────────────────────────────

/// Text every replayed reducer returns. Distinct from any recorded worker
/// payload so a test can tell synthesis apart from a verbatim relay.
const SYNTHESIZED: &str = "SYNTHESIZED-BY-REDUCER";

/// Serves one recorded worker response after its recorded (compressed)
/// latency. An `error` row fails the same way a dead peer would, so the
/// durability paths get exercised rather than mocked away.
///
/// A reducer call is a *different* call with different input, so replaying the
/// recorded worker body for it would be wrong — it would hand the reducer's
/// slot back the same truncated text the worker produced, and any assertion
/// about "what the caller received" would be measuring the fixture rather than
/// the gateway. Reducer calls are detected by their packed context and answered
/// with [`SYNTHESIZED`].
struct ReplayBackend {
    worker: RecordedWorker,
    calls: AtomicUsize,
}

impl ReplayBackend {
    fn new(worker: RecordedWorker) -> Arc<Self> {
        Arc::new(Self {
            worker,
            calls: AtomicUsize::new(0),
        })
    }
}

/// Does this call carry reducer context? `pack_for_reducer_selected` always
/// emits a `## Worker outputs` section; worker prompts never do.
fn is_reducer_call(messages: &[Value]) -> bool {
    messages
        .first()
        .and_then(|m| m.get("content"))
        .and_then(Value::as_str)
        .is_some_and(|s| s.contains("## Worker outputs"))
}

#[async_trait]
impl moa::ModelBackend for ReplayBackend {
    async fn chat_completion(
        &self,
        _model: &str,
        messages: &[Value],
        _tools: Option<&Value>,
        _max_tokens: u32,
        _timeout: Duration,
        sampling: moa::SamplingParams,
    ) -> Result<Value, String> {
        self.calls.fetch_add(1, Ordering::SeqCst);

        // MoA policy: workers never think. If this ever regresses, the
        // recorded traces stop being representative of what we replay.
        assert_ne!(
            sampling.enable_thinking,
            Some(true),
            "MoA must not ask a worker to enable thinking"
        );

        // The reducer is a fresh call, not a replay of this model's worker
        // turn. Answer it with synthesized text.
        if is_reducer_call(messages) {
            return Ok(json!({
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": SYNTHESIZED},
                    "finish_reason": "stop",
                }]
            }));
        }

        tokio::time::sleep(Duration::from_millis(
            (self.worker.elapsed_ms / LATENCY_DIVISOR).max(1),
        ))
        .await;

        match &self.worker.error {
            Some(e) => Err(e.clone()),
            None => Ok(self.worker.response_body()),
        }
    }
}

fn config_for(case: &RecordedCase) -> moa::GatewayConfig {
    let mut backends: Vec<Arc<dyn moa::ModelBackend>> = Vec::new();
    let mut models = Vec::new();
    for w in &case.workers {
        models.push(moa::ModelEntry::new(w.model.clone(), backends.len()));
        backends.push(ReplayBackend::new(w.clone()));
    }

    moa::GatewayConfig {
        backends,
        models,
        worker_timeout: Duration::from_secs(10),
        hedge_delay: Duration::from_millis(200),
        reducer_timeout: Duration::from_secs(5),
        first_answer_grace: Duration::ZERO,
        strong_patience: Duration::ZERO,
        // Mirrors `effective_enable_thinking_for_moa`, which is now
        // unconditionally off.
        enable_thinking: Some(false),
        actor_candidates: Vec::new(),
        // This suite exists to verify fan-out accounting (every dispatched
        // worker attributed, aborted stragglers reconciled), so it pins the
        // fan-out on. Production defaults to `Auto`, which skips advisors for
        // a strong actor — covered by `tool_turn`'s gate tests.
        reference_policy: moa::ReferencePolicy::Always,
        refinement_policy: Default::default(),
    }
}

fn request_body(case: &RecordedCase) -> Value {
    let mut body = json!({
        "model": "mesh",
        "messages": case.messages,
        "max_tokens": 512,
    });
    if case.has_tools {
        // Same schemas the traces were recorded against.
        body.as_object_mut().unwrap().insert(
            "tools".to_string(),
            json!([
                tool_schema("list_dir", &[("path", "string")]),
                tool_schema("read_file", &[("path", "string")]),
                tool_schema("search", &[("pattern", "string"), ("path", "string")]),
                tool_schema("run_command", &[("cmd", "string")]),
                tool_schema(
                    "edit_file",
                    &[
                        ("path", "string"),
                        ("before", "string"),
                        ("after", "string")
                    ],
                ),
            ]),
        );
    }
    body
}

fn tool_schema(name: &str, params: &[(&str, &str)]) -> Value {
    let props: serde_json::Map<String, Value> = params
        .iter()
        .map(|(p, ty)| (p.to_string(), json!({"type": ty})))
        .collect();
    json!({
        "type": "function",
        "function": {
            "name": name,
            "description": format!("{name} tool"),
            "parameters": {
                "type": "object",
                "properties": props,
                "required": params.iter().map(|(p, _)| *p).collect::<Vec<_>>(),
            }
        }
    })
}

fn response_text(body: &Value) -> String {
    body.pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string()
}

fn response_tool_calls(body: &Value) -> Vec<(String, String)> {
    body.pointer("/choices/0/message/tool_calls")
        .and_then(Value::as_array)
        .map(|tcs| {
            tcs.iter()
                .filter_map(|tc| {
                    Some((
                        tc.pointer("/function/name")?.as_str()?.to_string(),
                        tc.pointer("/function/arguments")?.as_str()?.to_string(),
                    ))
                })
                .collect()
        })
        .unwrap_or_default()
}

// ─── Tests ───────────────────────────────────────────────────────────

/// The fixture must actually contain the shapes these tests rely on. If a
/// re-record loses them, fail loudly here rather than passing vacuously
/// everywhere else.
#[test]
fn fixture_covers_the_shapes_under_test() {
    let fx = load_fixture();
    assert!(fx.cases.len() >= 40, "expected a substantial corpus");

    let truncated_partial = fx
        .cases
        .iter()
        .flat_map(|c| &c.workers)
        .filter(|w| w.partial_text().is_some())
        .count();
    assert!(
        truncated_partial >= 10,
        "fixture must contain truncated-with-partial-text responses \
         (the shape that could previously be returned verbatim); found {truncated_partial}"
    );

    let with_tools = fx
        .cases
        .iter()
        .filter(|c| c.workers.iter().any(|w| w.tool_calls.is_some()))
        .count();
    assert!(
        with_tools >= 20,
        "fixture must contain agentic tool-call cases; found {with_tools}"
    );

    let models: std::collections::BTreeSet<&str> = fx
        .cases
        .iter()
        .flat_map(|c| &c.workers)
        .map(|w| w.model.as_str())
        .collect();
    assert!(
        models.len() >= 5,
        "fixture should span a heterogeneous pool; found {models:?}"
    );
}

/// Truncated worker text must never reach the caller verbatim.
///
/// A response cut off at the token limit is a half-finished sentence. It may
/// inform synthesis, but shipping it as the final answer is a bug — and it
/// was reachable before `finish_reason` was plumbed through, because such an
/// answer looked normal to the parser and carried the same default 0.5
/// confidence as everyone else.
#[tokio::test(flavor = "multi_thread")]
async fn truncated_worker_text_is_never_returned_verbatim() {
    let fx = load_fixture();
    let mut checked = 0usize;

    for case in &fx.cases {
        let partials: Vec<String> = case
            .workers
            .iter()
            .filter_map(|w| w.partial_text().map(str::to_string))
            .collect();
        if partials.is_empty() {
            continue;
        }

        let result = moa::handle_turn(&config_for(case), &request_body(case)).await;
        let text = response_text(&result.response_body);
        if text.is_empty() {
            continue;
        }

        for partial in &partials {
            assert_ne!(
                text.trim(),
                partial.as_str(),
                "case `{}` returned a truncated worker payload verbatim",
                case.id
            );
            checked += 1;
        }
    }

    assert!(
        checked > 0,
        "no truncated payloads were exercised — fixture or filter is wrong"
    );
}

/// Every recorded case must produce a well-formed response, or a clean
/// structured failure when every worker died. No panics, no empty 200s.
#[tokio::test(flavor = "multi_thread")]
async fn every_recorded_case_produces_a_wellformed_turn() {
    let fx = load_fixture();

    for case in &fx.cases {
        let live = case.workers.iter().filter(|w| w.error.is_none()).count();
        let result = moa::handle_turn(&config_for(case), &request_body(case)).await;
        let body = &result.response_body;

        // Worker accounting: every dispatched worker is attributed, even
        // when early-exit consensus aborted the stragglers.
        if result.turn_kind != moa::TurnKind::ToolResult {
            assert_eq!(
                result.worker_summaries.len(),
                case.workers.len(),
                "case `{}`: every dispatched worker must appear in worker_summaries",
                case.id
            );
        }

        if live == 0 {
            assert_eq!(
                result.turn_kind,
                moa::TurnKind::Failed,
                "case `{}`: all-dead pool must fail cleanly",
                case.id
            );
            continue;
        }

        let text = response_text(body);
        let tools = response_tool_calls(body);
        assert!(
            !text.trim().is_empty() || !tools.is_empty(),
            "case `{}` ({:?}) produced neither text nor a tool call: {body}",
            case.id,
            result.turn_kind
        );

        // Any tool call we emit must carry a parseable JSON object for
        // `arguments` — agent harnesses reject anything else.
        for (name, args) in &tools {
            let parsed: Value = serde_json::from_str(args).unwrap_or_else(|e| {
                panic!(
                    "case `{}`: tool `{name}` args not JSON ({e}): {args:?}",
                    case.id
                )
            });
            assert!(
                parsed.is_object(),
                "case `{}`: tool `{name}` arguments must be a JSON object, got {parsed}",
                case.id
            );
        }
    }
}

/// When workers agree on a tool *name* but not its *arguments*, the winning
/// arguments must be the ones the majority actually proposed.
///
/// This is the `explore_error_handling` step-0 shape: all 9 models call
/// `list_dir`, 8 with `{"path": "src"}` and `mistral-small-3.2-24b` with
/// `{"path": "rust_project/src"}` — a directory that does not exist. Every
/// OpenAI-shape tool call is normalized to a fixed 0.9 confidence, so the
/// old confidence-only tiebreak had no way to prefer the majority.
#[tokio::test(flavor = "multi_thread")]
async fn majority_arguments_win_when_workers_agree_on_the_tool() {
    let fx = load_fixture();

    let cases: Vec<&RecordedCase> = fx
        .cases
        .iter()
        .filter(|c| {
            if c.step != 0 || !c.has_tools {
                return false;
            }
            let live: Vec<&RecordedWorker> =
                c.workers.iter().filter(|w| w.error.is_none()).collect();
            if live.is_empty() || live.iter().any(|w| w.tool_calls.is_none()) {
                return false; // need an all-tool turn
            }
            let names: std::collections::BTreeSet<String> = live
                .iter()
                .filter_map(|w| w.tool_calls.as_ref())
                .flat_map(|tcs| tcs.as_array().cloned().unwrap_or_default())
                .filter_map(|tc| {
                    tc.pointer("/function/name")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                })
                .collect();
            let args: std::collections::BTreeSet<String> = live
                .iter()
                .filter_map(|w| w.tool_calls.as_ref())
                .flat_map(|tcs| tcs.as_array().cloned().unwrap_or_default())
                .filter_map(|tc| {
                    tc.pointer("/function/arguments")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                })
                .collect();
            names.len() == 1 && args.len() > 1
        })
        .collect();

    assert!(
        !cases.is_empty(),
        "fixture must contain a name-unanimous / args-divergent case"
    );

    for case in cases {
        // Majority argument string among live tool proposals.
        let mut counts: std::collections::BTreeMap<String, usize> = Default::default();
        for w in case.workers.iter().filter(|w| w.error.is_none()) {
            for tc in w
                .tool_calls
                .as_ref()
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default()
            {
                if let Some(a) = tc.pointer("/function/arguments").and_then(Value::as_str) {
                    *counts.entry(a.to_string()).or_default() += 1;
                }
            }
        }
        let (majority, majority_n) = counts
            .iter()
            .max_by_key(|(_, n)| **n)
            .map(|(a, n)| (a.clone(), *n))
            .unwrap();
        // Only meaningful when there genuinely is a majority.
        if majority_n < 2 {
            continue;
        }
        let minority: Vec<&String> = counts
            .keys()
            .filter(|a| **a != majority && counts[*a] < majority_n)
            .collect();
        if minority.is_empty() {
            continue;
        }

        let result = moa::handle_turn(&config_for(case), &request_body(case)).await;
        let tools = response_tool_calls(&result.response_body);

        // The reducer may legitimately rewrite the turn into prose; only
        // assert when a tool call was emitted from worker proposals.
        if tools.is_empty() || result.reducer_used {
            continue;
        }

        let majority_val: Value = serde_json::from_str(&majority).unwrap_or(Value::Null);
        for (name, args) in &tools {
            let got: Value = serde_json::from_str(args).unwrap_or(Value::Null);
            assert_eq!(
                got, majority_val,
                "case `{}`: tool `{name}` should use the majority arguments \
                 ({majority_n} workers proposed {majority}), not a minority variant. \
                 Got {args}",
                case.id
            );
        }
    }
}

/// Agentic turns must keep producing structured tool calls.
///
/// This is the property Together's design cannot hold: its aggregator reads
/// `.content`, so on a turn where every worker returns a tool call it
/// synthesizes from empty strings.
#[tokio::test(flavor = "multi_thread")]
async fn unanimous_tool_turns_emit_a_structured_tool_call() {
    let fx = load_fixture();
    let mut asserted = 0usize;

    for case in fx.cases.iter().filter(|c| c.step == 0 && c.has_tools) {
        let live: Vec<&RecordedWorker> =
            case.workers.iter().filter(|w| w.error.is_none()).collect();
        if live.is_empty() || live.iter().any(|w| w.tool_calls.is_none()) {
            continue;
        }
        let names: std::collections::BTreeSet<String> = live
            .iter()
            .filter_map(|w| w.tool_calls.as_ref())
            .flat_map(|tcs| tcs.as_array().cloned().unwrap_or_default())
            .filter_map(|tc| {
                tc.pointer("/function/name")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .collect();
        if names.len() != 1 {
            continue; // divergent tool choice may legitimately go to the reducer
        }
        let expected = names.into_iter().next().unwrap();

        let result = moa::handle_turn(&config_for(case), &request_body(case)).await;
        let tools = response_tool_calls(&result.response_body);

        assert!(
            !tools.is_empty(),
            "case `{}`: {} workers unanimously proposed `{expected}`, but the turn \
             returned no tool call (kind={:?}). Tool calls must survive fan-out.",
            case.id,
            live.len(),
            result.turn_kind,
        );
        for (name, _) in &tools {
            assert_eq!(
                name, &expected,
                "case `{}`: emitted tool `{name}` but every worker proposed `{expected}`",
                case.id
            );
        }
        asserted += 1;
    }

    assert!(
        asserted > 0,
        "no unanimous tool turns exercised — fixture or filter is wrong"
    );
}

/// Recorded scenarios where no tool is warranted must not invent one.
#[tokio::test(flavor = "multi_thread")]
async fn conceptual_questions_do_not_invent_tool_calls() {
    let fx = load_fixture();

    for case in fx
        .cases
        .iter()
        .filter(|c| c.scenario.contains("no_tool_needed"))
    {
        let result = moa::handle_turn(&config_for(case), &request_body(case)).await;
        let tools = response_tool_calls(&result.response_body);
        assert!(
            tools.is_empty(),
            "case `{}`: no worker proposed a tool, so the turn must not emit one; got {tools:?}",
            case.id
        );
        assert!(
            !response_text(&result.response_body).trim().is_empty(),
            "case `{}`: expected a prose answer",
            case.id
        );
    }
}
