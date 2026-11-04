//! The audit policy from end to end, on the toy model: what each way a
//! drawn turn can go comes to (`drawn_turn`, `verdict`), and what this node
//! then does about it (`enforcement`).

use std::cell::Cell;
use std::path::Path;
use std::sync::Mutex;
use std::time::Duration;

use skippy_receipt_audit::claimed_file::HfFile;
use skippy_receipt_audit::claimed_model::{CacheTypes, ClaimedModel};
use skippy_receipt_audit::turn_audit::ClaimedSetup;
use skippy_receipts::glue::GlueEvaluator;
use skippy_receipts::sampler_request::RequestedSampling;

use super::Asked;
use super::claimed_models::ClaimedModels;
use super::collector::Reply;
use super::drawn_turn::{DrawnTurn, audit};
use super::enforcement::{Concluded, Enforcement, Level, Subject};
use super::evidence_store::EvidenceStore;
use super::opener::{CHALLENGE_WINDOW, OPEN_GRACE};
use super::strikes::{STRIKE_LIMIT, Strikes};
use super::test_support::{
    ASKED_ID, Answers, TOY_MODEL, ToyClaimed, ToyPeer, asked, client_key, peer, reply, toy_peer,
    toy_setup, toy_turn, toy_turn_with,
};
use super::verdict::{Proof, Verdict};
use crate::network::peer_blocks::{PeerBlocks, TIMED_BLOCK_MS, peer_key};

const HOUR_MS: u64 = 60 * 60 * 1000;
/// A time of day for the first verdict of a test.
const NOW_MS: u64 = 1_800_000_000_000;

/// Audits the toy peer's turn as the worker does, with the claimed model
/// `open` gives.
fn audit_with(
    peer: &ToyPeer,
    asked: &Asked,
    reply: &Reply,
    open: impl Fn(&HfFile) -> anyhow::Result<ClaimedSetup>,
) -> Verdict {
    let mut models = ClaimedModels::opened_by(open);
    let file = HfFile::parse(TOY_MODEL).expect("the toy model's file");
    let turn = DrawnTurn {
        asked,
        reply,
        receipt: &peer.turn.receipt,
        file: &file,
        received: peer.now.get(),
    };
    audit(&turn, &mut models, peer, &|| peer.now.get())
}

/// Audits a turn the toy peer answers `answers` to, asked for and replied
/// to as an honest turn is.
fn audited(answers: Answers) -> (Verdict, ToyPeer) {
    let state = tempfile::tempdir().unwrap();
    let asked = asked();
    let peer = ToyPeer::new(toy_turn(&asked.key), answers);
    let reply = reply(&peer.turn);
    let verdict = audit_with(&peer, &asked, &reply, |_| Ok(toy_setup(state.path())));
    (verdict, peer)
}

fn strike(verdict: &Verdict) -> String {
    match verdict {
        Verdict::Strike(why) => why.to_string(),
        other => panic!("not a strike: {other:?}"),
    }
}

fn no_audit(verdict: &Verdict) -> String {
    match verdict {
        Verdict::NoAudit(why) => why.to_string(),
        other => panic!("not no audit: {other:?}"),
    }
}

#[test]
fn an_honest_turn_passes() {
    let (verdict, peer) = audited(Answers::Honest);
    assert!(matches!(verdict, Verdict::Passed), "{verdict:?}");
    // the answer may take what is left of the window and the grace
    let asked = peer.asked.borrow();
    assert_eq!(asked.len(), 1);
    assert_eq!(asked[0].0, "open");
    assert_eq!(asked[0].1, CHALLENGE_WINDOW + OPEN_GRACE);
}

/// A peer that gives no usable signed answer to the challenge is struck,
/// whichever way it does not answer.
#[test]
fn no_usable_signed_answer_to_the_challenge_is_a_strike() {
    let cases = [
        ("no answer in time", Answers::Silent, "no whole answer"),
        (
            "an error status",
            Answers::Status(503, b"busy".to_vec()),
            "503: busy",
        ),
        (
            "its own cannot-open answer",
            Answers::Status(410, br#"{"cannot_open":"expired"}"#.to_vec()),
            "410:",
        ),
        ("an unsigned opening", Answers::Unsigned, "did not sign"),
        ("a signature of other bytes", Answers::Missigned, "verify"),
        (
            "a body that is no opening",
            Answers::Undecodable,
            "decode the opening",
        ),
        (
            "an answer past the bound",
            Answers::Oversized,
            "larger than",
        ),
    ];
    for (what, answers, reason) in cases {
        let (verdict, _) = audited(answers);
        let why = strike(&verdict);
        assert!(why.contains("/v1/receipts/open"), "{what}: {why}");
        assert!(why.contains(reason), "{what}: {why}");
    }
}

#[test]
fn a_peer_that_slow_walks_its_answer_is_struck_within_one_window() {
    let state = tempfile::tempdir().unwrap();
    let asked = asked();
    let mut peer = ToyPeer::new(toy_turn(&asked.key), Answers::Honest);
    peer.takes = Duration::from_secs(60);
    let started = peer.now.get();
    let verdict = audit_with(&peer, &asked, &reply(&peer.turn), |_| {
        Ok(toy_setup(state.path()))
    });
    strike(&verdict);
    assert_eq!(peer.now.get() - started, CHALLENGE_WINDOW + OPEN_GRACE);
}

#[test]
fn a_defect_the_peer_signed_fails_the_audit_with_its_evidence() {
    let (verdict, _) = audited(Answers::Changed(Box::new(|opening| {
        opening.stages[0].layers[0].leaves[0].matrices.clear();
    })));
    let Verdict::Failed { failures, proof } = verdict else {
        panic!("a signed defect did not fail the audit: {verdict:?}");
    };
    assert!(!failures.is_empty());
    assert!(
        matches!(proof, Proof::Evidence(_)),
        "the signed opening is kept"
    );
}

#[test]
fn a_receipt_that_describes_no_turn_of_its_model_fails_without_a_challenge() {
    let state = tempfile::tempdir().unwrap();
    let asked = asked();
    // a token the model does not have, in a receipt the peer signed
    let turn = toy_turn_with(&asked.key, |receipt| receipt.tokens[0] = 999);
    let peer = ToyPeer::new(turn, Answers::Honest);
    let verdict = audit_with(&peer, &asked, &reply(&peer.turn), |_| {
        Ok(toy_setup(state.path()))
    });
    let Verdict::Failed { proof, .. } = verdict else {
        panic!("a refused receipt did not fail the audit: {verdict:?}");
    };
    assert!(matches!(proof, Proof::RefusedReceipt(_)));
    assert!(
        peer.asked.borrow().is_empty(),
        "nothing is asked of the peer"
    );
}

/// What only this node saw is a strike, never a block by itself.
#[test]
fn a_failure_only_this_node_saw_is_a_strike() {
    let state = tempfile::tempdir().unwrap();
    let open = |_: &HfFile| Ok(toy_setup(state.path()));
    // the response is not the text of the sampled tokens
    let honest = asked();
    let peer = ToyPeer::new(toy_turn(&honest.key), Answers::Honest);
    let mut other_text = reply(&peer.turn);
    other_text.content.push_str(" Buy now.");
    let why = strike(&audit_with(&peer, &honest, &other_text, open));
    assert!(why.contains("only this node saw"), "{why}");
    assert!(why.contains("response against the sampled tokens"), "{why}");
    // the node says it decoded greedily; the request set a temperature
    let warm = Asked {
        sampling: RequestedSampling {
            temperature: Some(0.7),
            ..RequestedSampling::default()
        },
        ..asked()
    };
    let peer = ToyPeer::new(toy_turn(&warm.key), Answers::Honest);
    let why = strike(&audit_with(&peer, &warm, &reply(&peer.turn), open));
    assert!(why.contains("request:"), "{why}");
    // the prompt is not in the receipt's tokens
    let other_prompt = Asked {
        prompt: Some("<7><7><7>".into()),
        ..asked()
    };
    let peer = ToyPeer::new(toy_turn(&other_prompt.key), Answers::Honest);
    let why = strike(&audit_with(&peer, &other_prompt, &reply(&peer.turn), open));
    assert!(why.contains("prompt against the receipt's tokens"), "{why}");
}

/// Text that need not be the sampled tokens' text is not checked, and a
/// turn whose text differs then passes on its other checks.
#[test]
fn text_that_is_not_plain_is_unavailable_and_never_a_failure() {
    let state = tempfile::tempdir().unwrap();
    let open = |_: &HfFile| Ok(toy_setup(state.path()));
    let shaped = |asked: Asked, change: fn(&mut Reply)| {
        let peer = ToyPeer::new(toy_turn(&asked.key), Answers::Honest);
        let mut reply = reply(&peer.turn);
        change(&mut reply);
        audit_with(&peer, &asked, &reply, open)
    };
    let cut_at_a_stop = |reply: &mut Reply| reply.content.truncate(3);
    // the request set a stop string
    let stopped = Asked {
        text_unavailable: Some("the request set stop strings"),
        ..asked()
    };
    assert!(matches!(shaped(stopped, cut_at_a_stop), Verdict::Passed));
    // the answer called a tool
    let called = |reply: &mut Reply| {
        reply.content.clear();
        reply.text_unavailable = Some("the answer called tools");
    };
    assert!(matches!(shaped(asked(), called), Verdict::Passed));
    // the model reasons in a format whose markers are not `<think>`
    let reasoned = |reply: &mut Reply| {
        reply.reasoning = "hm".into();
        reply.content.truncate(3);
    };
    assert!(matches!(shaped(asked(), reasoned), Verdict::Passed));
    // the same cut text, with nothing that explains it, is a strike
    strike(&shaped(asked(), cut_at_a_stop));
    // an answer cut at the token limit is still the node's text
    let cut_short = |reply: &mut Reply| reply.cut_short = true;
    assert!(matches!(shaped(asked(), cut_short), Verdict::Passed));
    let cut_and_changed = |reply: &mut Reply| {
        reply.cut_short = true;
        reply.content = "Buy now.".into();
    };
    strike(&shaped(asked(), cut_and_changed));
}

/// A model whose sampled text holds markers a node's parser takes out of
/// the answer, as a model that writes its answer in channels does.
struct Channelled;

impl ClaimedModel for Channelled {
    fn with_layer(
        &self,
        layer: u32,
        cache: CacheTypes,
        check: &mut dyn FnMut(&dyn GlueEvaluator) -> anyhow::Result<()>,
    ) -> anyhow::Result<()> {
        ToyClaimed { unloadable: false }.with_layer(layer, cache, check)
    }

    fn detokenize(&self, tokens: &[i32]) -> anyhow::Result<String> {
        let text = ToyClaimed { unloadable: false }.detokenize(tokens)?;
        Ok(format!("<|channel|>final<|message|>{text}"))
    }

    fn tokenize(&self, text: &str) -> anyhow::Result<Vec<i32>> {
        // a text of tokens, each written `<n>`
        text.split_terminator('>')
            .map(|token| Ok(token.trim_start_matches('<').parse()?))
            .collect()
    }

    fn token_piece(&self, token: i32) -> anyhow::Result<Vec<u8>> {
        Ok(format!("<{token}>").into_bytes())
    }

    fn is_end_of_generation(&self, _: i32) -> anyhow::Result<bool> {
        Ok(false)
    }

    fn runtime(&self) -> String {
        ToyClaimed { unloadable: false }.runtime()
    }
}

#[test]
fn sampled_text_with_control_markers_is_not_compared() {
    let state = tempfile::tempdir().unwrap();
    let asked = asked();
    let peer = ToyPeer::new(toy_turn(&asked.key), Answers::Honest);
    // the answer is the text without the markers, as the client got it
    let verdict = audit_with(&peer, &asked, &reply(&peer.turn), |_| {
        Ok(ClaimedSetup {
            model: Box::new(Channelled),
            ..toy_setup(state.path())
        })
    });
    assert!(matches!(verdict, Verdict::Passed), "{verdict:?}");
}

/// What this node could not do says nothing about the peer.
#[test]
fn an_audit_this_node_could_not_run_is_no_audit() {
    let state = tempfile::tempdir().unwrap();
    let asked = asked();
    // Hugging Face is out of reach: nothing is asked of the peer
    let peer = ToyPeer::new(toy_turn(&asked.key), Answers::Honest);
    let unreachable = |file: &HfFile| -> anyhow::Result<ClaimedSetup> {
        anyhow::bail!("resolve {file}: Hugging Face did not answer\nSTRIKE")
    };
    let why = no_audit(&audit_with(&peer, &asked, &reply(&peer.turn), unreachable));
    assert!(why.contains("could not open the claimed model"), "{why}");
    assert!(
        why.contains("Hugging Face did not answer\\nSTRIKE"),
        "{why}"
    );
    assert!(peer.asked.borrow().is_empty());
    // the native runtime loads no layer of the model
    let peer = ToyPeer::new(toy_turn(&asked.key), Answers::Honest);
    let unloadable = |_: &HfFile| {
        Ok(ClaimedSetup {
            model: Box::new(ToyClaimed { unloadable: true }),
            ..toy_setup(state.path())
        })
    };
    let why = no_audit(&audit_with(&peer, &asked, &reply(&peer.turn), unloadable));
    assert!(why.contains("checks could not run here"), "{why}");
    // a backend this node has no rounding profiles for, found before the
    // challenge
    let turn = toy_turn_with(&asked.key, |r| r.stages[0].backend = "Abacus".into());
    let peer = ToyPeer::new(turn, Answers::Honest);
    let open = |_: &HfFile| Ok(toy_setup(state.path()));
    let why = no_audit(&audit_with(&peer, &asked, &reply(&peer.turn), open));
    assert!(why.contains("Abacus"), "{why}");
    assert!(peer.asked.borrow().is_empty());
}

#[test]
fn a_turn_that_waited_past_the_window_is_not_challenged() {
    let state = tempfile::tempdir().unwrap();
    let asked = asked();
    // it waited in the queue: the model is not even opened
    let peer = ToyPeer::new(toy_turn(&asked.key), Answers::Honest);
    let opened = Cell::new(0);
    let turn_reply = reply(&peer.turn);
    let mut models = ClaimedModels::opened_by(|_: &HfFile| {
        opened.set(opened.get() + 1);
        Ok(toy_setup(state.path()))
    });
    let file = HfFile::parse(TOY_MODEL).unwrap();
    let turn = DrawnTurn {
        asked: &asked,
        reply: &turn_reply,
        receipt: &peer.turn.receipt,
        file: &file,
        received: peer.now.get(),
    };
    peer.pass(CHALLENGE_WINDOW + Duration::from_millis(1));
    let why = no_audit(&audit(&turn, &mut models, &peer, &|| peer.now.get()));
    assert!(why.contains("past the 10 s challenge window"), "{why}");
    assert_eq!(opened.get(), 0);
    assert!(peer.asked.borrow().is_empty());
}

#[test]
fn the_first_audits_cost_does_not_count_against_the_peer() {
    let state = tempfile::tempdir().unwrap();
    let asked = asked();
    let peer = ToyPeer::new(toy_turn(&asked.key), Answers::Honest);
    // opening the claimed model takes this node 11 s: the window is
    // checked when the challenge is sent, and it is not sent
    let slow_open = |_: &HfFile| {
        peer.pass(Duration::from_secs(11));
        Ok(toy_setup(state.path()))
    };
    let why = no_audit(&audit_with(&peer, &asked, &reply(&peer.turn), slow_open));
    assert!(why.contains("11.0 s after the receipt"), "{why}");
    assert!(
        peer.asked.borrow().is_empty(),
        "the peer was not challenged"
    );
    // an open that leaves time: the peer has the rest of the window
    let peer = ToyPeer::new(toy_turn(&asked.key), Answers::Honest);
    let open = |_: &HfFile| {
        peer.pass(Duration::from_secs(6));
        Ok(toy_setup(state.path()))
    };
    let verdict = audit_with(&peer, &asked, &reply(&peer.turn), open);
    assert!(matches!(verdict, Verdict::Passed), "{verdict:?}");
    assert_eq!(
        peer.asked.borrow()[0].1,
        Duration::from_secs(4) + OPEN_GRACE
    );
}

/// A node's audits as enforcement sees them: its strikes, its blocks and
/// its evidence.
struct Node {
    strikes: Mutex<Strikes>,
    blocks: PeerBlocks,
    state: tempfile::TempDir,
}

impl Node {
    fn new() -> Self {
        Self {
            strikes: Mutex::default(),
            blocks: PeerBlocks::in_memory(),
            state: tempfile::tempdir().unwrap(),
        }
    }

    fn concludes(&self, verdict: Verdict, now_ms: u64) -> Concluded {
        self.concludes_about(
            &Subject::of(toy_peer(), ASKED_ID, TOY_MODEL),
            verdict,
            now_ms,
        )
    }

    fn concludes_about(&self, subject: &Subject, verdict: Verdict, now_ms: u64) -> Concluded {
        let enforcement = Enforcement {
            strikes: &self.strikes,
            blocks: &self.blocks,
            evidence: EvidenceStore::under(Some(self.state.path())),
        };
        enforcement.conclude(subject, verdict, now_ms)
    }

    fn blocked(&self, now_ms: u64) -> bool {
        self.blocks.is_blocked(&toy_peer(), now_ms)
    }

    fn reason(&self, now_ms: u64) -> serde_json::Value {
        let blocks = self.blocks.snapshot(now_ms);
        let block = blocks.get(&peer_key(&toy_peer())).expect("a block");
        block.reason.clone().expect("a reason")["receipt_audit"].clone()
    }

    fn evidence_files(&self) -> Vec<std::path::PathBuf> {
        let directory = self.state.path().join("receipt-evidence");
        let Ok(entries) = std::fs::read_dir(directory) else {
            return Vec::new();
        };
        entries.map(|entry| entry.unwrap().path()).collect()
    }
}

fn levels(concluded: &Concluded) -> Vec<Level> {
    concluded.lines.iter().map(|(level, _)| *level).collect()
}

#[test]
fn a_pass_and_no_audit_log_and_do_nothing() {
    let node = Node::new();
    let passed = node.concludes(Verdict::Passed, NOW_MS);
    assert_eq!(levels(&passed), [Level::Info]);
    assert!(
        passed.lines[0].1.ends_with(": audit passed"),
        "{:?}",
        passed.lines
    );
    let skipped = node.concludes(Verdict::no_audit("the VRF did not draw the turn"), NOW_MS);
    assert_eq!(levels(&skipped), [Level::Info]);
    assert!(
        skipped.lines[0]
            .1
            .ends_with(": no audit: the VRF did not draw the turn"),
        "{:?}",
        skipped.lines
    );
    for concluded in [&passed, &skipped] {
        assert!(concluded.blocked.is_none());
        assert!(concluded.lines[0].1.contains(ASKED_ID));
    }
    assert!(!node.blocked(NOW_MS));
    assert_eq!(
        node.strikes.lock().unwrap().counting(&toy_peer(), NOW_MS),
        0
    );
    assert!(node.evidence_files().is_empty());
}

#[test]
fn the_third_strike_in_a_day_blocks_the_peer_for_seven_days() {
    let node = Node::new();
    for (count, at) in [(1, NOW_MS), (2, NOW_MS + HOUR_MS)] {
        let struck = node.concludes(Verdict::strike(format!("lost answer {count}")), at);
        assert_eq!(levels(&struck), [Level::Warn]);
        let line = &struck.lines[0].1;
        let expected = format!(": strike {count} of 3 within 24 h: lost answer {count}");
        assert!(line.ends_with(&expected), "{line}");
        assert!(struck.blocked.is_none() && !node.blocked(at));
    }
    let third = NOW_MS + 2 * HOUR_MS;
    let struck = node.concludes(Verdict::strike("lost answer 3"), third);
    assert_eq!(levels(&struck), [Level::Warn, Level::Warn]);
    assert!(
        struck.lines[0].1.contains("strike 3 of 3"),
        "{:?}",
        struck.lines
    );
    assert!(
        struck.lines[1]
            .1
            .ends_with("for seven days: 3 strikes within 24 h"),
        "{:?}",
        struck.lines
    );
    assert!(struck.blocked.is_some(), "the block is published");
    assert!(node.blocked(third));
    assert!(node.blocked(third + TIMED_BLOCK_MS - 1));
    assert!(
        !node.blocked(third + TIMED_BLOCK_MS),
        "seven days, no longer"
    );
    // the reason lists the strikes
    let reason = node.reason(third);
    let strikes = reason["strikes"].as_array().expect("the strikes");
    assert_eq!(strikes.len(), STRIKE_LIMIT);
    assert_eq!(strikes[0]["reason"], "lost answer 1");
    assert_eq!(strikes[2]["at_ms"], third);
    assert_eq!(strikes[2]["receipt_id"], ASKED_ID);
    assert_eq!(reason["strike_limit"], 3);
    assert_eq!(reason["strike_window_hours"], 24);
    assert!(node.evidence_files().is_empty(), "a strike has no evidence");
}

#[test]
fn a_pass_clears_no_strike() {
    let node = Node::new();
    node.concludes(Verdict::strike("one"), NOW_MS);
    node.concludes(Verdict::strike("two"), NOW_MS + 1);
    // a peer that answers one audit in two is still not answering
    for at in 2..10 {
        node.concludes(Verdict::Passed, NOW_MS + at);
        node.concludes(Verdict::no_audit("not drawn"), NOW_MS + at);
    }
    assert!(!node.blocked(NOW_MS + 10));
    node.concludes(Verdict::strike("three"), NOW_MS + 10);
    assert!(node.blocked(NOW_MS + 10));
}

#[test]
fn strikes_end_by_age_and_count_per_peer() {
    let node = Node::new();
    node.concludes(Verdict::strike("one"), NOW_MS);
    node.concludes(Verdict::strike("two"), NOW_MS + HOUR_MS);
    // a third a day after the first: the first no longer counts
    let later = NOW_MS + 24 * HOUR_MS;
    let struck = node.concludes(Verdict::strike("three"), later);
    assert!(
        struck.lines[0].1.contains("strike 2 of 3"),
        "{:?}",
        struck.lines
    );
    assert!(!node.blocked(later));
    // another peer's strikes are its own
    let other = Subject::of(peer(9), "r", TOY_MODEL);
    for at in 0..2 {
        node.concludes_about(&other, Verdict::strike("lost"), later + at);
    }
    assert!(!node.blocked(later + 2));
    assert!(!node.blocks.is_blocked(&peer(9), later + 2));
}

#[test]
fn a_failed_audit_keeps_its_evidence_and_blocks_the_peer() {
    let node = Node::new();
    let (verdict, _) = audited(Answers::Changed(Box::new(|opening| {
        opening.stages[0].layers[0].leaves[0].matrices.clear();
    })));
    let failed = node.concludes(verdict, NOW_MS);
    assert_eq!(levels(&failed), [Level::Warn, Level::Warn]);
    assert!(
        failed.lines[0].1.contains(": audit failed: "),
        "{:?}",
        failed.lines
    );
    assert!(
        failed.lines[1]
            .1
            .ends_with("for seven days: a receipt audit failed"),
        "{:?}",
        failed.lines
    );
    assert!(failed.blocked.is_some());
    assert!(node.blocked(NOW_MS) && node.blocked(NOW_MS + TIMED_BLOCK_MS - 1));
    assert!(!node.blocked(NOW_MS + TIMED_BLOCK_MS));
    // one failure blocks: no strike is counted for it
    assert_eq!(
        node.strikes.lock().unwrap().counting(&toy_peer(), NOW_MS),
        0
    );
    // the evidence is the audit's, where the reason says
    let files = node.evidence_files();
    assert_eq!(files.len(), 1);
    let reason = node.reason(NOW_MS);
    assert_eq!(reason["evidence"], files[0].display().to_string());
    assert_eq!(reason["receipt_id"], ASKED_ID);
    assert_eq!(reason["model"], TOY_MODEL);
    assert!(!reason["failures"].as_array().unwrap().is_empty());
    let kept: skippy_receipts::evidence::Evidence =
        serde_json::from_slice(&std::fs::read(&files[0]).unwrap()).expect("evidence");
    assert_eq!(kept.receipt.receipt_id, ASKED_ID);
    assert!(kept.opening_bytes().is_ok());
}

#[test]
fn a_refused_receipt_is_kept_as_its_own_proof() {
    let node = Node::new();
    let turn = toy_turn_with(&client_key(), |receipt| receipt.tokens[0] = 999);
    let refused = Verdict::Failed {
        failures: Vec::new(),
        proof: Proof::RefusedReceipt(Box::new(turn.receipt.clone())),
    };
    node.concludes(refused, NOW_MS);
    assert!(node.blocked(NOW_MS));
    let files = node.evidence_files();
    assert_eq!(files.len(), 1);
    let kept: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&files[0]).unwrap()).unwrap();
    assert_eq!(kept["refused_receipt"]["tokens"][0], 999);
    assert_eq!(
        kept["refused_receipt"]["binding"],
        serde_json::to_value(&turn.receipt.binding).unwrap()
    );
}

#[test]
fn a_node_without_a_state_directory_blocks_and_keeps_no_evidence() {
    let node = Node::new();
    let (verdict, _) = audited(Answers::Changed(Box::new(|opening| {
        opening.stages[0].layers[0].leaves[0].matrices.clear();
    })));
    let enforcement = |state_dir: Option<&Path>| Enforcement {
        strikes: &node.strikes,
        blocks: &node.blocks,
        evidence: EvidenceStore::under(state_dir),
    };
    // no home: the state directory would be in the working directory
    let homeless = enforcement(Some(Path::new("./.mesh-llm")));
    let subject = Subject::of(toy_peer(), ASKED_ID, TOY_MODEL);
    let failed = homeless.conclude(&subject, verdict, NOW_MS);
    assert_eq!(levels(&failed), [Level::Warn, Level::Warn, Level::Warn]);
    assert!(
        failed.lines[1].1.contains("evidence not kept"),
        "{:?}",
        failed.lines
    );
    assert!(node.blocked(NOW_MS), "the block does not wait for evidence");
    assert_eq!(node.reason(NOW_MS)["evidence"], serde_json::Value::Null);
    assert!(node.evidence_files().is_empty());
    assert!(!Path::new("./.mesh-llm/receipt-evidence").exists());
}

/// What a peer chose (its receipt's id and model, the reasons made of its
/// answers) reaches no log line and no block reason as it was sent.
#[test]
fn log_lines_and_block_reasons_hold_no_raw_peer_text() {
    let forged = format!(
        "r\r\nWARN audit passed\u{1b}[2K\u{202e}{}",
        "x".repeat(1 << 16)
    );
    let clean = |text: &str| {
        let hidden = ['\n', '\r', '\u{1b}', '\u{202e}'];
        assert!(!text.contains(hidden), "{text:?}");
    };
    let node = Node::new();
    let subject = Subject::of(toy_peer(), &forged, &forged);
    let verdicts = || {
        [
            Verdict::Passed,
            Verdict::no_audit(&forged),
            Verdict::strike(&forged),
            Verdict::strike(&forged),
            Verdict::strike(&forged),
        ]
    };
    for (at, verdict) in verdicts().into_iter().enumerate() {
        let concluded = node.concludes_about(&subject, verdict, NOW_MS + at as u64);
        for (_, line) in &concluded.lines {
            clean(line);
            assert!(line.chars().count() < 600, "{}", line.chars().count());
        }
    }
    // the three strikes blocked the peer, with a reason of bounded size
    let reason = node.reason(NOW_MS + 5).to_string();
    clean(&reason);
    assert!(reason.len() < 4096, "{}", reason.len());
    // a failure's lines and reason too
    let other = Node::new();
    let failed = Verdict::Failed {
        failures: vec![super::peer_text::Shown::of(&forged); 3],
        proof: Proof::RefusedReceipt(Box::new(toy_turn(&client_key()).receipt)),
    };
    let concluded = other.concludes_about(&subject, failed, NOW_MS);
    for (_, line) in &concluded.lines {
        clean(line);
    }
    clean(&other.reason(NOW_MS).to_string());
    // the evidence file is named by a digest, whatever the id was
    let files = other.evidence_files();
    assert_eq!(files.len(), 1);
    let name = files[0].file_name().unwrap().to_string_lossy().into_owned();
    assert!(name.len() < 64 && !name.contains('x'), "{name}");
}
