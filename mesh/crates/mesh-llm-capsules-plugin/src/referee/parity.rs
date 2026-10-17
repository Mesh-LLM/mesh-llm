//! The referee parity run: the pinned corpus (`vectors/parity/referee/`)
//! through this plugin's referee, every answer held to the expected one
//! (`golden/` for the paths judged against the reference implementation,
//! `rule_answers/` for the paths stated from the rules). The input and answer
//! formats are specified in that corpus's README at its source; see
//! `vectors/parity/README.md`.
//!
//! A build with one of the `mutant-referee-*` features must fail this run on
//! the cases `mutants.json` lists for it; CI checks that it does.
//!
//! Set `REFEREE_PARITY_OUT=<path>` to also write this implementation's
//! answers, to compare them case by case with another implementation's.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};

use ed25519_dalek::SigningKey;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::record_push_parity::{
    appended, body_bytes, canonical, line_counts, normalize_reply, read_json,
};
use crate::referee::half::Half;
use crate::referee::verdict::{self, RefereeAnswer, UnknownWeights, Unreachable};

pub(crate) fn referee_dir() -> PathBuf {
    crate::record_push_parity::parity_dir().join("referee")
}

/// The fixed clock every path runs at.
pub(crate) const NOW: &str = "2026-09-29T00:00:00Z";

/// The corpus's test key for `node_id`: the Ed25519 key whose seed is
/// SHA-256 of this prefix followed by the node's name. For this corpus only.
const KEY_SEED_PREFIX: &str = "referee-parity-corpus/v1/";

pub(crate) fn test_key(node_id: &str) -> SigningKey {
    let seed: [u8; 32] = Sha256::digest(format!("{KEY_SEED_PREFIX}{node_id}")).into();
    SigningKey::from_bytes(&seed)
}

/// A node under test on a fresh ledger directory, with the files the case
/// says it starts with.
pub(crate) struct CaseNode {
    pub dir: tempfile::TempDir,
    pub key: SigningKey,
    pub peer_keys: Option<String>,
    pub record_at_completion_off: bool,
}

impl CaseNode {
    pub fn new(node: &Value) -> Self {
        let dir = tempfile::tempdir().expect("tempdir");
        if let Some(files) = node.get("files").and_then(Value::as_object) {
            for (name, lines) in files {
                let text: String = lines
                    .as_array()
                    .expect("a file is a list of lines")
                    .iter()
                    .map(|line| line.to_string() + "\n")
                    .collect();
                std::fs::write(dir.path().join(name), text).expect("seed a file");
            }
        }
        Self {
            dir,
            key: test_key(node["id"].as_str().expect("node id")),
            peer_keys: node
                .get("peer_keys_env")
                .and_then(Value::as_str)
                .map(str::to_string),
            record_at_completion_off: node.get("record_at_completion").and_then(Value::as_str)
                == Some("off"),
        }
    }

    pub fn ledger_dir(&self) -> &Path {
        self.dir.path()
    }

    pub fn key_id(&self) -> String {
        hex::encode(self.key.verifying_key().to_bytes())
    }

    /// A reply as the harness compares it: a refusal's signature checked,
    /// not copied.
    pub fn normalize(&self, reply: &Value, body: &[u8]) -> Value {
        normalize_reply(reply, body, &self.key.verifying_key(), NOW)
    }
}

// --- adjudicate ---------------------------------------------------------------

fn run_adjudicate(case: &Value) -> Vec<Value> {
    let halves: Vec<Half> = case["halves"]
        .as_array()
        .expect("halves")
        .iter()
        .map(|entry| Half {
            capsule: entry["capsule"].clone(),
            request_body: entry["request_body"].clone(),
            response_body: Some(&entry["response_body"])
                .filter(|b| !b.is_null())
                .cloned(),
            response_text: entry
                .get("response_text")
                .and_then(Value::as_str)
                .map(str::to_string),
            node_id: entry["node_id"].as_str().map(str::to_string),
            weights_digest: entry["weights_digest"].as_str().map(str::to_string),
        })
        .collect();
    let asked = &case["referee"];
    let referee_id = asked.get("node_id").and_then(Value::as_str);
    let mut referee = |a: &Half, b: &Half, comparison: &verdict::Comparison| {
        if asked.get("unreachable") == Some(&json!(true)) {
            return Err(Unreachable);
        }
        Ok(RefereeAnswer {
            verdict: verdict::referee_verdict(
                a,
                b,
                comparison,
                asked["answer_text"].as_str().unwrap_or_default(),
            ),
            capsule_id: None,
            referee_id: referee_id.unwrap_or_default().to_string(),
        })
    };
    let referee: Option<verdict::Referee<'_>> = if asked.is_null() {
        None
    } else {
        Some(&mut referee)
    };
    let answer = match verdict::adjudicate(
        &halves[0],
        &halves[1],
        referee,
        referee_id,
        UnknownWeights::NotComparable,
    ) {
        Ok(outcome) => outcome.to_answer(),
        Err(crate::referee::half::HalfError::Forged) => json!({"error": "forged_half"}),
        Err(crate::referee::half::HalfError::AnswerNotTheSignedBody) => {
            json!({"error": "answer_not_the_signed_body"})
        }
    };
    vec![answer]
}

// --- service (the referee's door) --------------------------------------------

/// Verdict record ids differ between implementations (each seals at its own
/// time), so an answer names them `verdict-1`, `verdict-2`, ... in the order
/// a case first shows them.
#[derive(Default)]
struct VerdictNames(Vec<String>);

impl VerdictNames {
    fn name(&mut self, capsule_id: &str) -> String {
        let at = match self.0.iter().position(|id| id == capsule_id) {
            Some(at) => at,
            None => {
                self.0.push(capsule_id.to_string());
                self.0.len() - 1
            }
        };
        format!("verdict-{}", at + 1)
    }
}

fn held_summary(held: &Value, node_key_id: &str, names: &mut VerdictNames) -> Value {
    let mut out = held.as_object().cloned().unwrap_or_default();
    for key in ["verdict_capsule", "verdict_capsule_id", "issued_at"] {
        out.remove(key);
    }
    let capsule = &held["verdict_capsule"];
    let id = held["verdict_capsule_id"].as_str().unwrap_or_default();
    let attestation = &capsule["model_attestation"]["compute_attestation"];
    let signed = crate::referee::half::record_checks(capsule)
        && crate::referee::half::signer(capsule).as_deref() == Some(node_key_id)
        && capsule.get("key_id").and_then(Value::as_str) == Some(node_key_id);
    out.insert("verdict_capsule_id".into(), json!(names.name(id)));
    out.insert(
        "verdict_capsule_id_is_the_records".into(),
        json!(capsule["capsule_id"].as_str() == Some(id)),
    );
    out.insert(
        "verdict_capsule".into(),
        json!({
            "block": attestation["adjudication"],
            "epistemic_type": attestation.get("epistemic_type"),
            "chain": capsule.get("chain"),
            "provenance": capsule.get("provenance"),
            "signed_by_node": signed,
        }),
    );
    out.insert(
        "issued_at_is_now".into(),
        json!(held.get("issued_at").and_then(Value::as_str) == Some(NOW)),
    );
    Value::Object(out)
}

fn run_service(case: &Value) -> Vec<Value> {
    let node = CaseNode::new(&case["node"]);
    let key_id = node.key_id();
    let referee = crate::referee::service::Referee {
        ledger_dir: node.ledger_dir(),
        signing_key: &node.key,
        peer_keys: node.peer_keys.as_deref(),
    };
    let mut names = VerdictNames::default();
    let issued = crate::referee::service::ISSUED_ADJUDICATIONS_FILENAME;
    case["requests"]
        .as_array()
        .expect("requests")
        .iter()
        .map(|request| {
            let body = body_bytes(request);
            let before = line_counts(node.ledger_dir());
            let reply = crate::referee::service::handle_adjudicate_request(&referee, &body, NOW)
                .expect("local writes succeed");
            let reply = if let Some(reason) = reply.get("reason") {
                let mut normal = node.normalize(&reply, &body);
                if reason == crate::referee::service::REASON_NO_VERDICT {
                    normal["detail"] = reply.get("detail").cloned().unwrap_or(Value::Null);
                }
                normal
            } else {
                held_summary(&reply, &key_id, &mut names)
            };
            let mut added = appended(node.ledger_dir(), &before);
            if let Some(lines) = added.get_mut(issued).and_then(Value::as_array_mut) {
                for line in lines.iter_mut() {
                    *line = held_summary(line, &key_id, &mut names);
                }
            }
            json!({"reply": reply, "appended": added})
        })
        .collect()
}

// --- hold (verdicts over record-push) -----------------------------------------

fn run_hold(case: &Value) -> Vec<Value> {
    let node = CaseNode::new(&case["node"]);
    let receiver = crate::record_push_receive::Receiver {
        ledger_dir: node.ledger_dir(),
        signing_key: &node.key,
        peer_keys: node.peer_keys.as_deref(),
        record_at_completion_off: node.record_at_completion_off,
        rejected_log_limit: crate::record_push_receive::MAX_REJECTED_LOG_BYTES,
    };
    case["pushes"]
        .as_array()
        .expect("pushes")
        .iter()
        .map(|push| {
            let body = body_bytes(push);
            let before = line_counts(node.ledger_dir());
            let reply = crate::record_push_receive::receive(&receiver, &body, push["sender"].as_str(), NOW)
                .expect("local writes succeed");
            json!({"reply": node.normalize(&reply, &body), "appended": appended(node.ledger_dir(), &before)})
        })
        .collect()
}

// --- deliver (verdicts at /evidence/deliver) ----------------------------------

fn run_deliver(case: &Value) -> Vec<Value> {
    let node = CaseNode::new(&case["node"]);
    let key_id = node.key_id();
    let door = crate::referee::hold::Node {
        ledger_dir: node.ledger_dir(),
        peer_keys: node.peer_keys.as_deref(),
        own_key_id: &key_id,
    };
    case["deliveries"]
        .as_array()
        .expect("deliveries")
        .iter()
        .map(|delivery| {
            let body = body_bytes(delivery);
            let before = line_counts(node.ledger_dir());
            let reply = match crate::referee::hold::handle_delivery(&door, &body, NOW)
                .expect("local writes succeed")
            {
                Ok(reply) => reply,
                Err(reason) => crate::record_push_receive::refuse(&node.key, &body, reason, NOW),
            };
            let held = appended(node.ledger_dir(), &before)
                .as_object()
                .is_some_and(|a| !a.is_empty());
            json!({"reply": node.normalize(&reply, &body), "held": held})
        })
        .collect()
}

// --- classify -----------------------------------------------------------------

fn run_classify(case: &Value) -> Vec<Value> {
    let receipts = case["receipts"].as_array().expect("receipts");
    let tally = crate::verdict_counts::classify_reference_receipts(
        receipts,
        case["x"].as_str().expect("x"),
        case["peer_keys_env"].as_str(),
    );
    vec![tally.to_json()]
}

// --- counts -------------------------------------------------------------------

fn time(text: &str) -> chrono::DateTime<chrono::Utc> {
    chrono::DateTime::parse_from_rfc3339(text)
        .expect("an RFC 3339 time")
        .into()
}

fn run_counts(case: &Value) -> Vec<Value> {
    let settings = &case["settings"];
    let rule = crate::routing_rule::rule_for(
        settings["stop_routing_after_contradictions"].as_str(),
        settings["stop_routing_window_days"].as_str(),
    );
    let strings = |v: &Value| -> Vec<String> {
        v.as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect()
    };
    let mut cited: HashSet<String> = strings(&case["cited"]).into_iter().collect();
    let mut blocked: BTreeSet<String> = strings(&case["host"]["blocked"]).into_iter().collect();
    let hook = case["host"]["peer_blocks"].as_bool() == Some(true);
    let dir = tempfile::tempdir().expect("tempdir");
    let mut records: Vec<Value> = Vec::new();
    let mut answers = Vec::new();
    for step in case["steps"].as_array().expect("steps") {
        records.extend(
            step["add_records"]
                .as_array()
                .into_iter()
                .flatten()
                .cloned(),
        );
        let requested: String = step["add_requested"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|line| line.to_string() + "\n")
            .collect();
        let path = dir.path().join(crate::verdict_counts::REQUESTED_FILENAME);
        let mut held = std::fs::read_to_string(&path).unwrap_or_default();
        held.push_str(&requested);
        std::fs::write(&path, held).expect("requested file");
        for peer in strings(&step["unblock"]) {
            blocked.remove(&peer);
        }

        let by_peer = crate::verdict_counts::fold(
            &records,
            &crate::verdict_counts::read_requested(dir.path()),
        );
        let mut outcomes = Vec::new();
        if let Some(rule) = rule {
            let firings = crate::routing_rule::due(
                rule,
                &by_peer,
                &cited,
                time(step["now"].as_str().expect("now")),
            );
            let snapshot = blocked.clone();
            let holds = move |peer: &str| snapshot.contains(peer);
            let holds: &dyn Fn(&str) -> bool = &holds;
            for step in crate::routing_rule::plan(&firings, hook.then_some(holds)) {
                match step {
                    crate::routing_rule::Planned::NoHostHook => {
                        outcomes.push(json!({"outcome": "no_host_hook"}))
                    }
                    crate::routing_rule::Planned::AlreadyBlocked(firing) => {
                        outcomes
                            .push(json!({"outcome": "already_blocked", "peer_id": firing.peer_id}));
                        cited.extend(firing.verdict_capsule_ids.iter().cloned());
                    }
                    crate::routing_rule::Planned::Block(firing) => {
                        blocked.insert(firing.peer_id.clone());
                        outcomes.push(json!({"outcome": "blocked", "peer_id": firing.peer_id,
                            "verdict_capsule_ids": firing.verdict_capsule_ids}));
                        cited.extend(firing.verdict_capsule_ids.iter().cloned());
                    }
                }
            }
        }
        let counts: Map<String, Value> = by_peer
            .iter()
            .map(|(peer, verdicts)| {
                (
                    peer.clone(),
                    crate::verdict_counts::counts_by_model(verdicts),
                )
            })
            .collect();
        let mut spent: Vec<&String> = cited.iter().collect();
        spent.sort();
        answers.push(json!({
            "counts": counts,
            "rule": rule.map(|r| json!({"after": r.after, "window_days": r.window_days})),
            "outcomes": outcomes,
            "cited": spent,
        }));
    }
    answers
}

// --- select -------------------------------------------------------------------

fn run_select(case: &Value) -> Vec<Value> {
    use crate::referee::bar::Recorded;
    use crate::referee::select::{select, Candidate};
    let text = |v: &Value| v.as_str().map(str::to_string);
    let candidates: Vec<Candidate> = case["peers"]
        .as_array()
        .expect("peers")
        .iter()
        .map(|p| Candidate {
            node_id: p["node_id"].as_str().expect("node_id").to_string(),
            model_hash: text(&p["model_hash"]),
            weights_digest: text(&p["weights_digest"]),
            announced_key: p["announced_key"].as_str().is_some_and(|k| !k.is_empty()),
            blocked: p["blocked"] == json!(true),
        })
        .collect();
    let verdicts: Vec<Recorded> = case["verdicts"]
        .as_array()
        .expect("verdicts")
        .iter()
        .map(|v| Recorded {
            node_id: v["node_id"].as_str().expect("node_id").to_string(),
            model_hash: v["model_hash"].as_str().expect("model_hash").to_string(),
            bucket: v["bucket"].as_str().expect("bucket").to_string(),
            recorded_at: time(v["recorded_at"].as_str().expect("recorded_at")),
        })
        .collect();
    let twins: Vec<&str> = case["twins"]
        .as_array()
        .expect("twins")
        .iter()
        .filter_map(Value::as_str)
        .collect();
    let bar_days = case["referee_bar_days"]
        .as_u64()
        .map(|d| u32::try_from(d).expect("days"))
        .unwrap_or(crate::referee::bar::DEFAULT_BAR_DAYS);
    let selection = select(
        time(case["now"].as_str().expect("now")),
        bar_days,
        case["model_hash"].as_str().expect("model_hash"),
        case["weights_digest"].as_str().expect("weights_digest"),
        [twins[0], twins[1]],
        &candidates,
        &verdicts,
    );
    let mut draws = case["draws"]
        .as_array()
        .expect("draws")
        .iter()
        .map(|d| d.as_u64().expect("draw") as usize);
    let asked: Vec<String> = (0..case["draws"].as_array().map_or(0, Vec::len))
        .filter_map(|_| {
            let d = draws.next().expect("a draw");
            selection
                .pick(&mut |n| {
                    assert!(d < n, "a draw inside the pool");
                    d
                })
                .map(str::to_string)
        })
        .collect();
    let answer = if selection.pool.is_empty() {
        json!({"tier": null, "pool": [], "asked": [], "not_adjudicated": crate::referee::select::NO_ELIGIBLE_REFEREE})
    } else {
        json!({"tier": selection.tier, "pool": selection.pool, "asked": asked, "not_adjudicated": null})
    };
    vec![answer]
}

// --- request ------------------------------------------------------------------

fn run_request(case: &Value) -> Vec<Value> {
    use crate::referee::request::{counts_against, Book, Called, Chosen, Pair, Twin};
    let on = case["adjudicate_differing_twins"] != json!(false);
    let mut book = Book::default();
    case["attempts"]
        .as_array()
        .expect("attempts")
        .iter()
        .map(|attempt| {
            let pair = &attempt["pair"];
            let twin = |h: &Value| Twin {
                node_id: h["node_id"].as_str().expect("node_id").to_string(),
                capsule_id: h["capsule_id"].as_str().unwrap_or_default().to_string(),
                text: h["text"].as_str().unwrap_or_default().to_string(),
                sampled: h["temperature"].as_f64().is_some_and(|t| t > 0.0),
                model_hash: h["model_hash"].as_str().map(str::to_string),
                weights_digest: h["weights_digest"].as_str().map(str::to_string),
            };
            let halves = pair["halves"].as_array().expect("halves");
            let pair = Pair {
                twin_bracket_id: pair["twin_bracket_id"].as_str().map(str::to_string),
                request_digest: pair["request_digest"].as_str().expect("request_digest").to_string(),
                twins: [twin(&halves[0]), twin(&halves[1])],
            };
            let selection = &attempt["selection"];
            let choose = || match selection["asked"].as_str() {
                Some(node) => Chosen::Referee { node_id: node.to_string(), tier: selection["tier"].as_u64().expect("tier") },
                None => Chosen::NoEligible,
            };
            let referee = &attempt["referee"];
            let call = |_: &str, _: u64| match referee["reanswer"].as_str() {
                None => Called::Unreachable,
                Some(_) if referee["signs"] != json!(true) => Called::CannotSign,
                Some(reanswer) => {
                    // The ruling the referee signs, by the referee's own rule.
                    let half = |t: &Twin| Half {
                        capsule: json!({}),
                        request_body: json!({"messages": [{"role": "user", "content": "?"}]}),
                        response_body: None,
                        response_text: Some(t.text.clone()),
                        node_id: Some(t.node_id.clone()),
                        weights_digest: t.weights_digest.clone(),
                    };
                    let (a, b) = (half(&pair.twins[0]), half(&pair.twins[1]));
                    let comparison = verdict::compare_transcripts(a.text(), b.text());
                    Called::Verdict(verdict::referee_verdict(&a, &b, &comparison, reanswer))
                }
            };
            let out = book.attempt(on, &pair, attempt["manual"] == json!(true), choose, call);
            json!({"referee_calls": out.calls, "row": out.row, "counts_against": counts_against(&out.row)})
        })
        .collect()
}

// --- the run ------------------------------------------------------------------

type Runner = fn(&Value) -> Vec<Value>;

/// Every path this runner judges, with where its expected answers are.
pub(crate) fn paths() -> Vec<(&'static str, &'static str, Runner)> {
    vec![
        ("adjudicate", "golden", run_adjudicate as Runner),
        ("service", "golden", run_service),
        ("hold", "golden", run_hold),
        ("deliver", "golden", run_deliver),
        ("classify", "golden", run_classify),
        ("select", "rule_answers", run_select),
        ("request", "rule_answers", run_request),
        ("counts", "rule_answers", run_counts),
    ]
}

/// Run the named cases (`path/case`); the ones that differ, with the
/// difference.
pub(crate) fn run_cases(names: &[&str]) -> Vec<(String, String)> {
    let dir = referee_dir();
    let mut differs = Vec::new();
    for (path, expected_dir, runner) in paths() {
        let wanted: Vec<&str> = names
            .iter()
            .filter_map(|n| n.strip_prefix(path).and_then(|rest| rest.strip_prefix('/')))
            .collect();
        if wanted.is_empty() {
            continue;
        }
        let corpus = read_json(&dir.join(format!("corpus/{path}.json")));
        let expected = read_json(&dir.join(format!("{expected_dir}/{path}.json")));
        for name in wanted {
            let case = corpus["cases"]
                .as_array()
                .expect("cases")
                .iter()
                .find(|c| c["name"] == json!(name))
                .unwrap_or_else(|| panic!("no case {path}/{name}"));
            let got = Value::Array(runner(case));
            let want = &expected["answers"][name];
            if canonical(&got) != canonical(want) {
                differs.push((
                    format!("{path}/{name}"),
                    format!("expected {}\ngot      {}", canonical(want), canonical(&got)),
                ));
            }
        }
    }
    differs
}

/// Run every case of every path; `(the cases that differ, with the
/// difference, and this implementation's answers)`.
pub(crate) fn run() -> (Vec<(String, String)>, Value) {
    let dir = referee_dir();
    let mut differs = Vec::new();
    let mut all = Map::new();
    for (path, expected_dir, runner) in paths() {
        let corpus = read_json(&dir.join(format!("corpus/{path}.json")));
        let expected = read_json(&dir.join(format!("{expected_dir}/{path}.json")));
        let mut answers = Map::new();
        for case in corpus["cases"].as_array().expect("cases") {
            let name = case["name"].as_str().expect("name");
            let got = Value::Array(runner(case));
            let want = &expected["answers"][name];
            if canonical(&got) != canonical(want) {
                differs.push((
                    format!("{path}/{name}"),
                    format!(
                        "  expected {}\n  got      {}",
                        canonical(want),
                        canonical(&got)
                    ),
                ));
            }
            answers.insert(name.to_string(), got);
        }
        all.insert(path.to_string(), Value::Object(answers));
    }
    (
        differs,
        json!({"v": 1, "implementation": "rust-plugin", "paths": all}),
    )
}

#[test]
fn the_referee_answers_the_corpus_as_expected() {
    let (differs, answers) = run();
    if let Ok(out) = std::env::var("REFEREE_PARITY_OUT") {
        std::fs::write(&out, answers.to_string() + "\n").expect("write answers");
    }
    let listed: BTreeMap<&str, &str> = differs
        .iter()
        .map(|(c, d)| (c.as_str(), d.as_str()))
        .collect();
    assert!(
        listed.is_empty(),
        "{} case(s) differ:\n{}",
        listed.len(),
        listed
            .iter()
            .map(|(case, diff)| format!("differs: {case}\n{diff}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// The corpus is the pinned copy: every file matches `vectors/SHA256SUMS`.
#[test]
fn the_corpus_is_the_pinned_copy() {
    let vectors = crate::record_push_parity::parity_dir().join("..");
    let sums = std::fs::read_to_string(vectors.join("SHA256SUMS")).expect("SHA256SUMS");
    let mut checked = 0;
    for line in sums.lines().filter(|l| l.contains("parity/referee/")) {
        let (digest, file) = line.split_once("  ").expect("sum  file");
        let bytes = std::fs::read(vectors.join(file)).expect("a pinned file");
        assert_eq!(
            hex::encode(Sha256::digest(&bytes)),
            digest,
            "{file} is not the pinned copy"
        );
        checked += 1;
    }
    assert!(
        checked >= 16,
        "every referee corpus file is pinned ({checked})"
    );
}

/// The port's unit tests, by the names `port_tests.json` gives them: each
/// holds its rule on the corpus cases listed for it.
#[allow(non_snake_case)]
mod port_tests {
    macro_rules! port_test {
        ($name:ident, [$($case:literal),+ $(,)?]) => {
            #[test]
            fn $name() {
                let differs = super::run_cases(&[$($case),+]);
                assert!(differs.is_empty(), "{:?}", differs);
            }
        };
    }

    port_test!(
        differing_pair_triggers_one_call,
        ["request/differing_pair_triggers_one_call"]
    );
    port_test!(
        agreeing_pair_calls_nothing,
        ["request/agreeing_pair_calls_nothing"]
    );
    port_test!(
        sampled_pair_calls_nothing,
        [
            "request/sampled_pair_calls_nothing",
            "request/one_sampled_half_calls_nothing"
        ]
    );
    port_test!(
        default_on_when_setting_unset,
        ["request/default_on_when_setting_unset"]
    );
    port_test!(
        operator_off_calls_nothing_and_says_so,
        ["request/operator_off_calls_nothing_and_says_so"]
    );
    port_test!(
        no_bracket_id_calls_nothing_and_says_so,
        ["request/no_bracket_id_calls_nothing_and_says_so"]
    );
    port_test!(
        second_call_same_pair_refused,
        ["request/second_call_same_pair_refused"]
    );
    port_test!(
        resealed_halves_same_exchange_still_capped,
        ["request/resealed_halves_same_exchange_still_capped"]
    );
    port_test!(
        referee_timeout_is_not_retried,
        [
            "request/referee_timeout_is_not_retried",
            "request/another_referee_is_not_tried"
        ]
    );
    port_test!(
        referee_side_repeat_returns_issued_verdict,
        [
            "service/repeat_returns_the_issued_verdict",
            "service/repeat_with_the_halves_swapped"
        ]
    );
    port_test!(
        same_model_hash_and_digest_only,
        [
            "select/same_model_hash_and_digest_only",
            "select/different_model_hash_never_picked"
        ]
    );
    port_test!(
        different_digest_never_picked,
        ["select/different_digest_never_picked"]
    );
    port_test!(twin_never_picked, ["select/twin_never_picked"]);
    port_test!(
        no_announced_key_never_picked,
        ["select/no_announced_key_never_picked"]
    );
    port_test!(
        blocked_node_never_picked,
        ["select/blocked_node_never_picked"]
    );
    port_test!(
        twins_with_different_digests_not_adjudicated,
        [
            "request/twins_with_different_digests_not_adjudicated",
            "request/twins_with_unknown_digest_not_adjudicated"
        ]
    );
    port_test!(tier1_before_tier2, ["select/tier1_before_tier2"]);
    port_test!(
        cold_picked_only_when_tier1_empty,
        ["select/cold_picked_only_when_tier1_empty"]
    );
    port_test!(
        pick_uniform_within_best_tier,
        ["select/pick_uniform_within_best_tier"]
    );
    port_test!(
        contradiction_for_X_never_picked_for_X_but_ok_for_Y,
        [
            "select/contradiction_for_x_never_picked_for_x",
            "select/contradiction_for_x_ok_for_y"
        ]
    );
    port_test!(
        verdict_carries_selection_tier,
        [
            "service/contradicts_b",
            "service/tier_two_referee",
            "hold/verdict_carries_selection_tier"
        ]
    );
    port_test!(
        altered_tier_fails_verification,
        ["hold/altered_tier_fails_verification"]
    );
    port_test!(
        hold_refuses_tier_other_than_asked,
        ["hold/hold_refuses_tier_other_than_asked"]
    );
    port_test!(no_eligible, ["request/no_eligible_referee"]);
    port_test!(referee_without_plugin, ["request/referee_without_plugin"]);
    port_test!(unreachable, ["request/referee_unreachable"]);
    port_test!(off, ["request/operator_off_calls_nothing_and_says_so"]);
    port_test!(twins_agree, ["request/agreeing_pair_calls_nothing"]);
    port_test!(barred_inside_D, ["select/barred_inside_d"]);
    port_test!(eligible_after_D, ["select/eligible_after_d"]);
    port_test!(
        corroboration_inside_D_does_not_clear,
        ["select/corroboration_inside_d_does_not_clear"]
    );
    port_test!(bar_on_X_leaves_Y, ["select/bar_on_x_leaves_y"]);
    port_test!(
        D_from_setting_default_30,
        [
            "select/d_default_is_30",
            "select/d_from_setting_shorter",
            "select/d_from_setting_longer"
        ]
    );
    port_test!(
        history_still_visible_while_barred,
        ["counts/history_still_visible_while_barred"]
    );
    port_test!(
        received_at_not_the_referees_issued_at,
        ["select/bar_runs_from_this_nodes_clock"]
    );
    port_test!(
        stop_rule_off_by_default,
        ["counts/stop_rule_off_by_default"]
    );
    port_test!(
        fires_at_N_within_window,
        [
            "counts/fires_at_n_within_window",
            "counts/outside_the_window_does_not_count"
        ]
    );
    port_test!(
        one_referee_contributes_at_most_N_minus_1,
        ["counts/one_referee_contributes_at_most_n_minus_1"]
    );
    port_test!(
        only_verified_and_asked_verdicts_count,
        [
            "counts/only_verified_and_asked_verdicts_count",
            "counts/asked_another_referee_about_the_pair"
        ]
    );
    port_test!(
        one_count_per_referee_and_pair,
        ["counts/one_count_per_referee_and_pair"]
    );
    port_test!(
        undo_holds_until_new_contradictions,
        [
            "counts/undo_holds_until_new_contradictions",
            "counts/already_blocked_is_not_blocked_again"
        ]
    );
    port_test!(
        host_without_hook_does_nothing,
        ["counts/host_without_hook_does_nothing"]
    );
    port_test!(
        forged_reference_verdict_counts_nowhere,
        [
            "classify/forged_unsigned",
            "classify/forged_unannounced_key",
            "classify/forged_referee_not_announced",
            "classify/forged_no_referee",
            "classify/forged_copied_block",
            "classify/forged_many"
        ]
    );
    port_test!(
        ack_refusal_counts_only_with_verified_contradiction,
        [
            "classify/ack_refusal_of_a_verified_contradiction",
            "classify/ack_refusal_without_the_verdict",
            "classify/ack_refusal_of_a_forged_verdict",
            "classify/ack_refusal_claiming_a_contradiction"
        ]
    );

    /// Every test `port_tests.json` names is here.
    #[test]
    fn every_named_port_test_is_carried() {
        const CARRIED: &[&str] = &[
            "differing_pair_triggers_one_call",
            "agreeing_pair_calls_nothing",
            "sampled_pair_calls_nothing",
            "default_on_when_setting_unset",
            "operator_off_calls_nothing_and_says_so",
            "no_bracket_id_calls_nothing_and_says_so",
            "second_call_same_pair_refused",
            "resealed_halves_same_exchange_still_capped",
            "referee_timeout_is_not_retried",
            "referee_side_repeat_returns_issued_verdict",
            "same_model_hash_and_digest_only",
            "different_digest_never_picked",
            "twin_never_picked",
            "no_announced_key_never_picked",
            "blocked_node_never_picked",
            "twins_with_different_digests_not_adjudicated",
            "tier1_before_tier2",
            "cold_picked_only_when_tier1_empty",
            "pick_uniform_within_best_tier",
            "contradiction_for_X_never_picked_for_X_but_ok_for_Y",
            "verdict_carries_selection_tier",
            "altered_tier_fails_verification",
            "hold_refuses_tier_other_than_asked",
            "no_eligible",
            "referee_without_plugin",
            "unreachable",
            "off",
            "twins_agree",
            "barred_inside_D",
            "eligible_after_D",
            "corroboration_inside_D_does_not_clear",
            "bar_on_X_leaves_Y",
            "D_from_setting_default_30",
            "history_still_visible_while_barred",
            "received_at_not_the_referees_issued_at",
            "stop_rule_off_by_default",
            "fires_at_N_within_window",
            "one_referee_contributes_at_most_N_minus_1",
            "only_verified_and_asked_verdicts_count",
            "one_count_per_referee_and_pair",
            "undo_holds_until_new_contradictions",
            "host_without_hook_does_nothing",
            "forged_reference_verdict_counts_nowhere",
            "ack_refusal_counts_only_with_verified_contradiction",
        ];
        let listed =
            crate::record_push_parity::read_json(&super::referee_dir().join("port_tests.json"));
        for tests in listed["tests"].as_object().expect("tests").values() {
            for name in tests.as_object().expect("names").keys() {
                assert!(CARRIED.contains(&name.as_str()), "{name} is not carried");
            }
        }
    }
}
