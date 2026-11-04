//! Whole audits of a toy turn: an honest node, and each way a node or this
//! machine can stop one.

use anyhow::bail;
use skippy_receipts::glue::GlueEvaluator;
use skippy_receipts::toy_turn::{self, LAYERS, RUNTIME, ToyModel, Turn};
use skippy_receipts::wire::encode_f32;

use super::*;
use crate::claimed_model::CacheTypes;
use crate::outcome::Check;
use crate::output::HeadOutput;
use crate::recheck::recheck_evidence;

/// The toy model as the audit loads it. Its token `n` is the text `<n>`.
struct ToyClaimed {
    /// No layer loads, as when the native runtime cannot open the GGUF.
    unloadable: bool,
}

impl ClaimedModel for ToyClaimed {
    fn with_layer(
        &self,
        _: u32,
        _: CacheTypes,
        check: &mut dyn FnMut(&dyn GlueEvaluator) -> Result<()>,
    ) -> Result<()> {
        if self.unloadable {
            bail!("the layer did not load");
        }
        check(&ToyModel)
    }

    fn detokenize(&self, tokens: &[i32]) -> Result<String> {
        Ok(tokens.iter().map(|t| format!("<{t}>")).collect())
    }

    fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
        // a text of tokens, each written `<n>`
        text.split_terminator('>')
            .map(|token| Ok(token.trim_start_matches('<').parse()?))
            .collect()
    }

    fn token_piece(&self, token: i32) -> Result<Vec<u8>> {
        Ok(format!("<{token}>").into_bytes())
    }

    fn is_end_of_generation(&self, _: i32) -> Result<bool> {
        Ok(false)
    }

    fn runtime(&self) -> String {
        RUNTIME.to_owned()
    }
}

/// What a node changes in the opening it committed to.
type Change = Box<dyn Fn(&mut Opening)>;

/// What the toy node does with a challenge.
enum Answer {
    Honest,
    /// The request does not get through.
    Refused,
    /// The opening comes without a signature.
    Unsigned,
    /// The signature is of other bytes.
    Missigned,
    /// The node opens this instead of what it committed, and signs it.
    Changed(Change),
}

struct Node {
    turn: Turn,
    answer: Answer,
}

impl OpenChallenge for Node {
    fn open(&self, challenge: &Challenge) -> Result<Opened> {
        if matches!(self.answer, Answer::Refused) {
            bail!("503 Service Unavailable: the stage cannot replay turns");
        }
        let mut opening = self.turn.open(challenge);
        if let Answer::Changed(change) = &self.answer {
            change(&mut opening);
        }
        let bytes = serde_json::to_vec(&opening)?;
        let signature = match self.answer {
            Answer::Unsigned => None,
            Answer::Missigned => Some(self.turn.sign_opening(challenge, b"another opening")),
            _ => Some(self.turn.sign_opening(challenge, &bytes)),
        };
        Ok(Opened {
            opening,
            bytes,
            signature,
            seconds: 0.25,
        })
    }

    fn rows(&self, _: &BlockChallenge) -> Result<OpenedRows> {
        bail!("the toy model has no attention rows")
    }
}

/// A client of the toy node: its key, its claimed model, and where it
/// keeps its head key.
struct Client {
    key: vrf::SecretKey,
    setup: ClaimedSetup,
    _state: tempfile::TempDir,
}

impl Client {
    fn new() -> Self {
        Self::with(toy_turn::weights(), false)
    }

    fn with(weights: WeightFile, unloadable: bool) -> Self {
        let state = tempfile::tempdir().expect("a temporary directory");
        Self {
            key: vrf::SecretKey::from_seed([9; 32]),
            setup: ClaimedSetup {
                weights,
                model: Box::new(ToyClaimed { unloadable }),
                layer_count: LAYERS,
                head_key: state.path().join("head-key.json"),
            },
            _state: state,
        }
    }

    /// A turn the toy node served this client, greedily.
    fn turn(&self) -> Turn {
        Turn::new().bound(self.key.public(), turn::GREEDY)
    }

    fn audit_with(&self, node: &Node, sampling: RequestedSampling) -> TurnAudit {
        let receipt = node.turn.receipt.clone();
        let reply = ChatReply {
            model: "toy".into(),
            content: ToyClaimed { unloadable: false }
                .detokenize(&receipt.sampled_tokens())
                .expect("text"),
            reasoning: String::new(),
            cut_short: false,
        };
        let asked_id = receipt.receipt_id.clone();
        let received = ReceivedTurn {
            receipt,
            key: &self.key,
            response: Some(("<1><2><3>", &reply)),
            sampling,
            asked_id: Some(&asked_id),
        };
        audit_turn(&self.setup, received, node, &AuditOptions::default())
    }

    fn audit(&self, answer: Answer) -> TurnAudit {
        let node = Node {
            turn: self.turn(),
            answer,
        };
        self.audit_with(&node, RequestedSampling::greedy())
    }
}

fn audited(audit: TurnAudit) -> (AuditOutput, Option<Evidence>) {
    match audit {
        TurnAudit::Audited { output, evidence } => (*output, evidence.map(|e| *e)),
        TurnAudit::NoAudit(output) => panic!("no audit: {}", output.reason),
        TurnAudit::Unanswered(output) => panic!("unanswered: {}", output.reason),
    }
}

#[test]
fn an_honest_turn_passes_and_its_evidence_rechecks() {
    let client = Client::new();
    let audit = client.audit(Answer::Honest);
    assert_eq!(audit.outcome(), Outcome::Pass);
    assert_eq!(audit.outcome().exit_code(), 0);
    let (output, evidence) = audited(audit);
    assert!(output.pass, "{}", output.render_text());
    assert!(output.failures.is_empty() && output.unchecked.is_empty());
    assert_eq!(output.layers.len(), LAYERS as usize);
    assert!(output.layers.iter().all(|l| l.glue_pass));
    // the router is checked whole, by the auditor's plan
    assert!(output.roles.iter().any(|r| r.router && r.pass));
    let head = output.head.as_ref().and_then(Check::verdict).expect("head");
    assert!(head.keyed && head.pass);
    assert!(output.render_text().ends_with("AUDIT PASS\n"));
    // anyone rechecks the evidence, without the client's key
    let evidence = evidence.expect("evidence of a drawn challenge");
    let model = ToyClaimed { unloadable: false };
    let rechecked = recheck_evidence(&evidence, &toy_turn::weights(), &model, LAYERS);
    assert_eq!(rechecked.outcome(), Outcome::Pass);
    let (rechecked, _) = audited(rechecked);
    let head = rechecked
        .head
        .as_ref()
        .and_then(Check::verdict)
        .expect("head");
    // it checks the same logits on every row, and judges the same token
    assert!(!head.keyed && head.every_row && head.pass);
    assert!(head.positions.iter().all(|p| p.logits_ok == Some(true)));
}

#[test]
fn a_node_that_does_not_answer_the_challenge_leaves_the_audit_unanswered() {
    // After it saw the challenge, a node could decline to answer, or
    // answer without a signature it can be held to. That was "no audit",
    // which costs a node nothing; it is its own outcome now.
    let client = Client::new();
    for (answer, reason) in [
        (Answer::Refused, "503 Service Unavailable"),
        (Answer::Unsigned, "did not sign its opening"),
        (Answer::Missigned, "does not verify"),
    ] {
        let audit = client.audit(answer);
        assert_eq!(audit.outcome(), Outcome::Unanswered, "{reason}");
        assert_eq!(audit.outcome().exit_code(), 3);
        let TurnAudit::Unanswered(output) = audit else {
            panic!("{reason}: not unanswered");
        };
        assert_eq!(output.request, "open");
        assert!(output.reason.contains(reason), "{}", output.reason);
        let json = serde_json::to_value(&output).unwrap();
        assert_eq!(json["outcome"], "unanswered");
        assert_eq!(json["pass"], false);
    }
}

/// The failure a signed opening that `change` altered ends in, and its
/// evidence.
fn defect(change: impl Fn(&mut Opening) + 'static) -> (AuditOutput, Evidence) {
    let client = Client::new();
    let audit = client.audit(Answer::Changed(Box::new(change)));
    assert_eq!(audit.outcome(), Outcome::Fail);
    assert_eq!(audit.outcome().exit_code(), 1);
    let (output, evidence) = audited(audit);
    assert!(output.unchecked.is_empty(), "{:?}", output.unchecked);
    (output, evidence.expect("evidence of the signed opening"))
}

/// The outcome a recheck of `evidence` ends in.
fn rechecked(evidence: &Evidence) -> AuditOutput {
    let model = ToyClaimed { unloadable: false };
    audited(recheck_evidence(
        evidence,
        &toy_turn::weights(),
        &model,
        LAYERS,
    ))
    .0
}

#[test]
fn a_decoy_stage_fails_the_audit_and_its_recheck() {
    // every check used to run on the decoy, which no Merkle check read
    let (output, evidence) = defect(|opening| {
        let mut decoy = opening.stages[1].clone();
        decoy.stage_index = 99;
        opening.stages.insert(0, decoy);
    });
    let error = output.opening_error.as_deref().expect("a refused opening");
    assert!(error.contains("stages [99, 0, 1]"), "{error}");
    assert!(output.provable());
    assert!(output.layers.is_empty(), "no check ran on the decoy");
    let again = rechecked(&evidence);
    assert_eq!(again.outcome, Outcome::Fail);
    assert_eq!(again.opening_error, output.opening_error);
}

#[test]
fn a_malformed_signed_opening_is_a_provable_failure_and_not_no_audit() {
    // each of these stopped the audit with an error, which was no audit
    let nan = encode_f32(&[f32::NAN, 0.0, 0.0, 0.0]);
    let short = encode_f32(&[1.0, 2.0]);
    let changes: Vec<(&str, Change)> = vec![
        (
            "no matrices",
            Box::new(|o| o.stages[0].layers[0].leaves[0].matrices.clear()),
        ),
        (
            "a short column",
            Box::new(move |o| o.stages[0].layers[0].leaves[0].matrices[0].output = short.clone()),
        ),
        (
            "a value that is not a number",
            Box::new(move |o| o.stages[1].layers[0].leaves[0].matrices[1].input = nan.clone()),
        ),
        (
            "logits of another length",
            Box::new(|o| o.stages[1].head[0].logits = Some(encode_f32(&[0.5; 3]))),
        ),
        (
            "a tensor the GGUF does not have",
            Box::new(|o| {
                o.stages[0].layers[0].leaves[0].matrices[0].weight = "blk.0.none.weight".into();
            }),
        ),
        (
            "an expert out of range",
            Box::new(|o| {
                o.stages[0].layers[0].leaves[0].matrices[2].experts = Some(vec![0, 9]);
            }),
        ),
        ("no stages", Box::new(|o| o.stages.clear())),
        (
            "a matmul opened twice",
            Box::new(|o| {
                let leaf = &mut o.stages[0].layers[0].leaves[0];
                let copy = leaf.matrices[0].clone();
                leaf.matrices.push(copy);
            }),
        ),
    ];
    for (what, change) in changes {
        let (output, evidence) = defect(change);
        assert!(output.provable(), "{what}: {:?}", output.failures);
        let again = rechecked(&evidence);
        assert_eq!(again.outcome, Outcome::Fail, "{what}: the recheck differs");
        assert!(again.provable(), "{what}");
    }
}

#[test]
fn a_head_key_that_refused_logits_is_not_used_again() {
    // whether a check with the key passes is a bit a node learns about a
    // key the client uses for every node of the model
    let client = Client::new();
    let key_file = client.setup.head_key.clone();
    assert_eq!(client.audit(Answer::Honest).outcome(), Outcome::Pass);
    let kept = std::fs::read(&key_file).expect("the key an audit built");
    assert_eq!(client.audit(Answer::Honest).outcome(), Outcome::Pass);
    assert_eq!(
        std::fs::read(&key_file).unwrap(),
        kept,
        "a passing key is kept"
    );
    // one logit changed, far past any rounding
    let changed = Answer::Changed(Box::new(|opening| {
        let head = &mut opening.stages[1].head[0];
        let mut logits = head.logit_values().unwrap().expect("logits");
        logits[0] += 0.5;
        head.logits = Some(encode_f32(&logits));
    }));
    let (output, evidence) = audited(client.audit(changed));
    assert_eq!(output.outcome, Outcome::Fail);
    let head = output.head.as_ref().and_then(Check::verdict).expect("head");
    assert!(head.positions.iter().any(|p| p.logits_ok == Some(false)));
    assert!(
        !key_file.exists(),
        "the key that refused the logits is kept"
    );
    // the key is the client's alone, so the audit does not claim that the
    // evidence shows this; a recheck reads every row, and it does
    assert!(!output.provable());
    let again = rechecked(&evidence.expect("evidence of the signed opening"));
    assert_eq!(again.outcome, Outcome::Fail);
    assert!(again.provable(), "{:?}", again.failures);
    let head = again.head.as_ref().and_then(Check::verdict).expect("head");
    assert!(head.every_row && head.positions.iter().any(|p| p.logits_ok == Some(false)));
    // the next audit builds another
    assert_eq!(client.audit(Answer::Honest).outcome(), Outcome::Pass);
    assert_ne!(std::fs::read(&key_file).unwrap(), kept);
}

#[test]
fn logits_left_out_fail_the_audit_instead_of_costing_the_client_the_head() {
    let (output, evidence) = defect(|opening| {
        for head in &mut opening.stages[1].head {
            head.logits = None;
        }
    });
    let fault = output.head.as_ref().and_then(Check::node_fault);
    assert!(fault.is_some_and(|f| f.contains("opened no logits")));
    // a recheck judges the token on no other logits, so it finds the same
    // defect in the same opening
    assert!(output.provable());
    let again = rechecked(&evidence);
    assert_eq!(again.outcome, Outcome::Fail);
    assert!(again.provable());
    let fault = again.head.as_ref().and_then(Check::node_fault);
    assert!(fault.is_some_and(|f| f.contains("opened no logits")));
}

#[test]
fn a_cheated_column_fails_its_row_check() {
    let (output, evidence) = defect(|opening| {
        for leaf in &mut opening.stages[0].layers[0].leaves {
            leaf.matrices[0].output = encode_f32(&[9.0, 9.0, 9.0, 9.0]);
        }
    });
    let failed: Vec<&str> = output.failures.iter().map(|f| f.check.as_str()).collect();
    assert!(
        failed
            .iter()
            .any(|f| f.starts_with("weight rows of a.weight")),
        "{failed:?}"
    );
    assert_eq!(rechecked(&evidence).outcome, Outcome::Fail);
}

#[test]
fn a_weight_read_that_fails_is_no_audit() {
    // the claimed GGUF's header is at hand and its weights are not: the
    // node answered honestly, and nothing it sent is at fault
    let client = Client::with(toy_turn::weights_out_of_reach(), false);
    let audit = client.audit(Answer::Honest);
    assert_eq!(audit.outcome(), Outcome::NoAudit);
    assert_eq!(audit.outcome().exit_code(), 2);
    let (output, _) = audited(audit);
    assert!(output.failures.is_empty(), "{:?}", output.failures);
    let out_of_reach = |check: &str| {
        output
            .unchecked
            .iter()
            .any(|u| u.starts_with(check) && u.contains("out of reach"))
    };
    for check in [
        "glue of layer 0",
        "weight rows of layer 1",
        "input of layer 0",
        "output head",
    ] {
        assert!(out_of_reach(check), "{check}: {:?}", output.unchecked);
    }
    assert!(output.render_text().ends_with("NO AUDIT\n"));
    assert_eq!(
        serde_json::to_value(&output).unwrap()["outcome"],
        "no_audit"
    );
}

#[test]
fn a_layer_that_does_not_load_is_no_audit() {
    let client = Client::with(toy_turn::weights(), true);
    let audit = client.audit(Answer::Honest);
    assert_eq!(audit.outcome(), Outcome::NoAudit);
    let (output, _) = audited(audit);
    assert!(output.failures.is_empty(), "{:?}", output.failures);
    assert!(output.unchecked.iter().any(|u| u.contains("did not load")));
    // the head needs no layer, and its check held
    assert!(output.head.as_ref().and_then(Check::verdict).is_some());
    // nor do the weight rows, which are still checked, with no matmul
    // taken for a router since the layer's plan is unknown
    assert!(output.layers.iter().all(|l| l.matrices.len() == 3));
    assert!(output.roles.iter().all(|r| r.pass && !r.router));
}

#[test]
fn a_cheated_column_fails_though_the_layer_does_not_load() {
    // a client that cannot evaluate a layer still sees weights that are
    // not the claimed ones
    let client = Client::with(toy_turn::weights(), true);
    let cheated = Answer::Changed(Box::new(|opening| {
        for leaf in &mut opening.stages[0].layers[0].leaves {
            leaf.matrices[0].output = encode_f32(&[9.0, 9.0, 9.0, 9.0]);
        }
    }));
    let audit = client.audit(cheated);
    assert_eq!(audit.outcome(), Outcome::Fail);
    let (output, _) = audited(audit);
    assert!(output.provable(), "{:?}", output.failures);
}

#[test]
fn a_receipt_this_verifier_cannot_audit_is_no_audit_before_any_challenge() {
    let client = Client::new();
    let mut other_runtime = Turn::new();
    other_runtime.receipt.stages[0].backend = "MTL0".into();
    let node = Node {
        turn: other_runtime.bound(client.key.public(), turn::GREEDY),
        // the node is never asked
        answer: Answer::Refused,
    };
    let audit = client.audit_with(&node, RequestedSampling::greedy());
    assert_eq!(audit.outcome(), Outcome::NoAudit);
    let TurnAudit::NoAudit(output) = audit else {
        panic!("not no audit");
    };
    assert!(output.reason.contains("MTL0"), "{}", output.reason);
    assert_eq!(
        serde_json::to_value(&output).unwrap()["outcome"],
        "no_audit"
    );
}

#[test]
fn a_receipt_from_a_quantized_kv_cache_is_no_audit_and_never_a_failure() {
    // An honest stage with a q8_0 cache fails the loose attention check, so
    // a node that still issues such a receipt, as one built before the
    // stage refused did, is not audited.
    for quantized in ["q8_0", "q4_0"] {
        let client = Client::new();
        let mut turn = Turn::new();
        turn.receipt.stages[0].cache_types = vec![quantized.into()];
        let node = Node {
            turn: turn.bound(client.key.public(), turn::GREEDY),
            // the node is never asked
            answer: Answer::Refused,
        };
        let audit = client.audit_with(&node, RequestedSampling::greedy());
        assert_eq!(audit.outcome(), Outcome::NoAudit, "{quantized}");
        let TurnAudit::NoAudit(output) = audit else {
            panic!("not no audit");
        };
        assert!(output.reason.contains(quantized), "{}", output.reason);
        assert!(
            output.reason.contains("quantized"),
            "{quantized}: {}",
            output.reason
        );
    }
}

#[test]
fn a_sampler_statement_that_contradicts_the_request_fails() {
    // The statement is the node's own text. One that names a grammar, a
    // penalty or a bias leaves its tokens unjudged or moves any token into
    // the candidates; the client asked for none of them.
    let client = Client::new();
    let statement = turn::sampled(&serde_json::json!({
        "seed": 0, "samplers": ["top_k", "temperature"], "temperature": 0.0, "top_k": 40,
        "logit_bias": [[5, 0.0]], "grammar": null, "reasoning_budget": null,
    }));
    let node = Node {
        turn: Turn::new().bound(client.key.public(), &statement),
        answer: Answer::Honest,
    };
    let audit = client.audit_with(&node, RequestedSampling::greedy());
    assert_eq!(audit.outcome(), Outcome::Fail);
    let (output, _) = audited(audit);
    assert_eq!(output.request_errors.len(), 1);
    assert!(
        output.request_errors[0].contains("logit biases"),
        "{:?}",
        output.request_errors
    );
    // the request is not in the evidence: no one else can see this
    assert!(!output.provable());
    // a client that does not know what was sent cannot contradict it
    let audit = client.audit_with(&node, RequestedSampling::default());
    let (output, _) = audited(audit);
    assert!(output.request_errors.is_empty());
}

#[test]
fn a_receipt_under_another_id_than_the_request_fails() {
    let client = Client::new();
    let node = Node {
        turn: client.turn(),
        answer: Answer::Honest,
    };
    let receipt = node.turn.receipt.clone();
    let received = ReceivedTurn {
        receipt,
        key: &client.key,
        response: None,
        sampling: RequestedSampling::greedy(),
        asked_id: Some("the id the client sent"),
    };
    let audit = audit_turn(&client.setup, received, &node, &AuditOptions::default());
    let (output, _) = audited(audit);
    assert_eq!(output.outcome, Outcome::Fail);
    assert!(output.request_errors[0].contains("the id the client sent"));
    assert!(!output.provable());
}

#[test]
fn a_receipt_bound_to_another_key_is_refused_without_a_challenge() {
    let client = Client::new();
    let other = vrf::SecretKey::from_seed([1; 32]);
    let node = Node {
        turn: Turn::new().bound(other.public(), turn::GREEDY),
        answer: Answer::Refused,
    };
    let audit = client.audit_with(&node, RequestedSampling::greedy());
    assert_eq!(audit.outcome(), Outcome::Fail);
    let (output, evidence) = audited(audit);
    assert!(output.binding_error.is_some());
    assert!(output.challenge.picks.is_empty());
    assert!(evidence.is_none() && !output.provable());
}

#[test]
fn a_turn_that_continued_a_cached_prefix_is_audited_with_its_prefix() {
    let client = Client::new();
    let node = Node {
        turn: Turn::starting_at(40).bound(client.key.public(), turn::GREEDY),
        answer: Answer::Honest,
    };
    let audit = client.audit_with(&node, RequestedSampling::greedy());
    let (output, evidence) = audited(audit);
    assert_eq!(output.outcome, Outcome::Pass, "{}", output.render_text());
    // the challenge picks positions of the prefix, in every layer
    let prefix = output.challenge.prefix_positions(40);
    assert_eq!(prefix.len(), skippy_receipts::challenge::PREFIX_POSITIONS);
    for layer in &output.layers {
        assert!(layer.glue_pass, "{}", output.render_text());
    }
    // the prompt is in the tokens of the prefix and the turn together
    let tokens = output.tokens.as_ref().and_then(Check::verdict).unwrap();
    assert!(tokens.prompt_ok);
    assert!(output.notes.is_empty(), "{:?}", output.notes);
    // anyone gets the same verdict from the evidence
    let evidence = evidence.expect("a pass of a bound receipt keeps evidence");
    let rechecked = recheck_evidence(
        &evidence,
        &client.setup.weights,
        client.setup.model.as_ref(),
        LAYERS,
    );
    assert_eq!(rechecked.outcome(), Outcome::Pass);
    // a receipt that names a prefix its stages do not commit is refused
    let mut uncommitted = Turn::starting_at(40);
    uncommitted.receipt.stages[1].prefix = None;
    let node = Node {
        turn: uncommitted.bound(client.key.public(), turn::GREEDY),
        answer: Answer::Refused,
    };
    let audit = client.audit_with(&node, RequestedSampling::greedy());
    let (output, _) = audited(audit);
    assert!(
        output
            .receipt_error
            .is_some_and(|e| e.contains("cached prefix"))
    );
}

#[test]
fn a_challenge_picked_by_hand_keeps_no_evidence() {
    let client = Client::new();
    let node = Node {
        turn: client.turn(),
        answer: Answer::Honest,
    };
    let received = ReceivedTurn {
        receipt: node.turn.receipt.clone(),
        key: &client.key,
        response: None,
        sampling: RequestedSampling::greedy(),
        asked_id: None,
    };
    let options = AuditOptions {
        layer: Some(1),
        timing_runs: 1,
    };
    let (output, evidence) = audited(audit_turn(&client.setup, received, &node, &options));
    assert!(
        output.hand_picked && output.pass,
        "{}",
        output.render_text()
    );
    assert_eq!(output.challenge.picks.len(), 1);
    // shown as evidence it would claim the VRF drew what the client chose
    assert!(evidence.is_none());
    // a layer the model does not have is the client's mistake, not an
    // unanswered challenge
    let received = ReceivedTurn {
        receipt: node.turn.receipt.clone(),
        key: &client.key,
        response: None,
        sampling: RequestedSampling::greedy(),
        asked_id: None,
    };
    let options = AuditOptions {
        layer: Some(LAYERS),
        timing_runs: 1,
    };
    let audit = audit_turn(&client.setup, received, &node, &options);
    assert_eq!(audit.outcome(), Outcome::NoAudit);
}

/// The blocks of one attention pick, as a client asks their rows.
fn blocks(turn: &Turn) -> BlockChallenge {
    BlockChallenge {
        receipt_id: turn.receipt.receipt_id.clone(),
        picks: vec![skippy_receipts::attention::opening::BlockPick {
            position: 3,
            layer: 1,
            blocks: vec![0],
        }],
        proof: Some("ab".into()),
    }
}

/// A node asked for attention rows.
struct RowsNode<'a> {
    turn: &'a Turn,
    /// What it signs: the rows it sends, other bytes, or nothing. `None`
    /// for a node that does not answer.
    signs: Option<&'a str>,
}

const ROWS: &[u8] = br#"{"receipt_id":"toy-turn","stages":[]}"#;

impl OpenChallenge for RowsNode<'_> {
    fn open(&self, _: &Challenge) -> Result<Opened> {
        bail!("not asked")
    }

    fn rows(&self, asked: &BlockChallenge) -> Result<OpenedRows> {
        let signature = match self.signs {
            None => bail!("the peer did not answer /v1/receipts/rows in time"),
            Some("rows") => Some(self.turn.sign_rows(asked, ROWS)),
            Some("other blocks") => {
                let mut other = asked.clone();
                other.picks[0].blocks = vec![1];
                Some(self.turn.sign_rows(&other, ROWS))
            }
            Some(_) => None,
        };
        Ok(OpenedRows {
            rows: serde_json::from_slice(ROWS)?,
            bytes: ROWS.to_vec(),
            signature,
            seconds: 0.0,
        })
    }
}

#[test]
fn attention_rows_a_node_does_not_send_or_sign_are_unanswered_and_not_a_failure() {
    // a failed rows request used to fail the audit while a failed opening
    // was no audit; both are the node not answering
    let client = Client::new();
    let turn = client.turn();
    let bound = turn::verify(&turn.receipt, None).unwrap();
    let blocks = blocks(&turn);
    let answer = |signs| {
        let node = RowsNode { turn: &turn, signs };
        signed_rows(&node, &bound, &blocks)
    };
    assert!(matches!(answer(Some("rows")), RowsAnswer::Signed(_)));
    for (signs, reason) in [
        (None, "did not answer"),
        (Some("nothing"), "did not sign its attention rows"),
        (Some("other blocks"), "does not verify"),
    ] {
        let RowsAnswer::Unanswered(why) = answer(signs) else {
            panic!("{signs:?}: not unanswered");
        };
        assert!(why.contains(reason), "{why}");
    }
}

#[test]
fn evidence_without_rows_for_the_drawn_blocks_is_incomplete() {
    use crate::recheck::evidence_rows;
    let client = Client::new();
    let turn = client.turn();
    let node = Node {
        turn: client.turn(),
        answer: Answer::Honest,
    };
    let (_, evidence) = audited(client.audit_with(&node, RequestedSampling::greedy()));
    let evidence = evidence.expect("evidence");
    let layer_bytes = executable_layer_bytes(&client.setup.weights, LAYERS);
    let mut asked = blocks(&turn);
    asked.proof = None;
    // the rows were removed, or never came
    let provenance = evidence.verify(&layer_bytes).unwrap();
    let RowsAnswer::Missing(reason) = evidence_rows(&provenance, &asked) else {
        panic!("rows from evidence that holds none");
    };
    assert!(reason.contains("holds no attention rows"), "{reason}");
    // rows the node signed for the blocks the seed draws are read
    let mut sent = asked.clone();
    sent.proof = Some(evidence.rows_proof.clone());
    let signed = |blocks: &BlockChallenge, rows: &[u8]| {
        let with_rows = evidence
            .clone()
            .with_rows(rows, Some(turn.sign_rows(blocks, rows)));
        with_rows.verify(&layer_bytes).unwrap()
    };
    let provenance = signed(&sent, ROWS);
    assert!(matches!(
        evidence_rows(&provenance, &asked),
        RowsAnswer::Signed(_)
    ));
    // rows it signed for other blocks are not evidence of these
    let mut other = sent.clone();
    other.picks[0].blocks = vec![1];
    let provenance = signed(&other, ROWS);
    assert!(matches!(
        evidence_rows(&provenance, &asked),
        RowsAnswer::Missing(_)
    ));
    // rows it signed for these blocks that cannot be read are its defect
    let provenance = signed(&sent, b"not rows");
    assert!(matches!(
        evidence_rows(&provenance, &asked),
        RowsAnswer::Defective(_)
    ));
}

#[test]
fn a_file_that_is_not_evidence_is_no_audit() {
    // forged evidence must not read as a failed audit of the node it names
    let client = Client::new();
    let (_, evidence) = audited(client.audit(Answer::Honest));
    let mut forged = evidence.expect("evidence");
    forged.challenge.picks.truncate(1);
    let model = ToyClaimed { unloadable: false };
    let audit = recheck_evidence(&forged, &toy_turn::weights(), &model, LAYERS);
    assert_eq!(audit.outcome(), Outcome::NoAudit);
    let TurnAudit::NoAudit(output) = audit else {
        panic!("not no audit");
    };
    assert!(output.reason.contains("not evidence"), "{}", output.reason);
}

/// The sampler statement of a turn sampled at a temperature of 1 with no
/// filter, drawn by the indexed draw of `seed`.
fn drawn_with(seed: u32) -> String {
    turn::sampled(&serde_json::json!({
        "seed": seed, "draw": turn::INDEXED_DRAW,
        "samplers": ["top_k", "top_p", "min_p", "temperature"],
        "temperature": 1.0, "dynatemp_range": 0.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0,
        "logit_bias": [], "grammar": null, "reasoning_budget": null, "ignore_eos": false,
        "mirostat": { "mode": 0 },
    }))
}

/// What the request of such a turn set.
fn sampled_request(seed: u32) -> RequestedSampling {
    RequestedSampling {
        temperature: Some(1.0),
        seed: Some(u64::from(seed)),
        ..RequestedSampling::greedy()
    }
}

/// The audit of the toy turn when its node says it drew with `seed`.
fn audit_of_seed(client: &Client, seed: u32) -> TurnAudit {
    let node = Node {
        turn: Turn::new().bound(client.key.public(), &drawn_with(seed)),
        answer: Answer::Honest,
    };
    client.audit_with(&node, sampled_request(seed))
}

fn draws(output: &AuditOutput) -> Vec<(bool, bool)> {
    let Some(Check::Verdict(head)) = &output.head else {
        panic!("no head verdict");
    };
    head.positions
        .iter()
        .filter_map(|p| p.draw.as_ref().map(|d| (d.exact, d.pass)))
        .collect()
}

#[test]
fn a_sampled_turn_passes_only_with_the_seed_its_tokens_follow() {
    // The toy turn's tokens are what they are. A seed whose draws pick
    // them at every checked position makes the turn an honest sampled one;
    // with any other seed the node states a draw its tokens do not follow.
    let client = Client::new();
    let mut followed = None;
    let mut broken = None;
    for seed in 1..2000 {
        let audit = audit_of_seed(&client, seed);
        match audit.outcome() {
            Outcome::Pass if followed.is_none() => followed = Some(audited(audit)),
            Outcome::Fail if broken.is_none() => broken = Some(audited(audit)),
            _ => {}
        }
        if followed.is_some() && broken.is_some() {
            break;
        }
    }
    let (output, _) = followed.expect("a seed the tokens follow");
    let replayed = draws(&output);
    assert!(!replayed.is_empty());
    assert!(replayed.iter().all(|&(_, pass)| pass), "{replayed:?}");
    assert!(
        output.notes.iter().all(|n| !n.contains("seed")),
        "{:?}",
        output.notes
    );

    let (output, evidence) = broken.expect("a seed the tokens do not follow");
    assert!(draws(&output).iter().any(|&(_, pass)| !pass));
    assert_eq!(output.failures.len(), 1, "{:?}", output.failures);
    assert!(output.failures[0].check.contains("output head"));
    // the statement, the tokens and the opened logits are all the node's,
    // and signed: anyone who holds the evidence finds the same failure
    assert!(output.provable(), "{:?}", output.failures);
    let model = ToyClaimed { unloadable: false };
    let evidence = evidence.expect("evidence");
    let (rechecked, _) = audited(recheck_evidence(
        &evidence,
        &toy_turn::weights(),
        &model,
        LAYERS,
    ));
    assert_eq!(rechecked.outcome, Outcome::Fail);
    assert_eq!(draws(&rechecked), draws(&output));
}

#[test]
fn a_draw_with_a_seed_the_request_did_not_set_is_noted() {
    let client = Client::new();
    let seed = (1..2000)
        .find(|&seed| audit_of_seed(&client, seed).outcome() == Outcome::Pass)
        .expect("a seed the tokens follow");
    let node = Node {
        turn: Turn::new().bound(client.key.public(), &drawn_with(seed)),
        answer: Answer::Honest,
    };
    // the request named no seed: the node chose this one
    let unseeded = RequestedSampling {
        seed: None,
        ..sampled_request(seed)
    };
    let (output, _) = audited(client.audit_with(&node, unseeded));
    assert_eq!(output.outcome, Outcome::Pass);
    assert!(
        output
            .notes
            .iter()
            .any(|n| n.contains("the seed the node chose")),
        "{:?}",
        output.notes
    );
    // a request that set another seed is contradicted
    let (output, _) = audited(client.audit_with(&node, sampled_request(seed + 1)));
    assert_eq!(output.outcome, Outcome::Fail);
    assert!(
        output.request_errors[0].contains("seed"),
        "{:?}",
        output.request_errors
    );
    assert!(!output.provable());
}

/// The statement of [`drawn_with`] for a turn with a reasoning budget of
/// no tokens, whose sampler was built from the tags `start` and `end`.
fn drawn_with_budget(seed: u32, start: &str, end: &str) -> String {
    let mut parameters = turn::sampler_parameters(&drawn_with(seed)).expect("parameters");
    parameters["reasoning_budget"] = 0.into();
    parameters[turn::REASONING] = serde_json::json!({
        "start": start, "ends": [end], "prefill": "",
    });
    turn::sampled(&parameters)
}

/// The audit of the toy turn when its node says it drew with `seed`, under
/// a reasoning budget whose sampler `statement` names.
fn audit_of_budget(client: &Client, seed: u32, statement: &str) -> TurnAudit {
    let node = Node {
        turn: Turn::new().bound(client.key.public(), statement),
        answer: Answer::Honest,
    };
    let request = RequestedSampling {
        reasoning_budget: Some(0),
        ..sampled_request(seed)
    };
    client.audit_with(&node, request)
}

fn head_of(output: &AuditOutput) -> &HeadOutput {
    let Some(Check::Verdict(head)) = &output.head else {
        panic!("no head verdict");
    };
    head
}

#[test]
fn a_token_a_reasoning_budget_forced_is_not_judged_by_the_draw_and_the_others_are() {
    // The toy turn samples three tokens. A node says the first is the tag
    // that starts the reasoning and the second the tag that ends it, with
    // a budget of no tokens: the sampler forced the second token, and the
    // draw picked the first and the third.
    let client = Client::new();
    let sampled = Turn::new().receipt.sampled_tokens();
    let tag = |token: i32| format!("<{token}>");
    let honest = |seed| drawn_with_budget(seed, &tag(sampled[0]), &tag(sampled[1]));
    // a seed whose draws pick the first and third token and not the second:
    // without the budget the turn fails, with it the turn passes
    let seed = (1..20_000)
        .find(|&seed| {
            audit_of_budget(&client, seed, &honest(seed)).outcome() == Outcome::Pass
                && audit_of_seed(&client, seed).outcome() == Outcome::Fail
        })
        .expect("a seed the free tokens follow and the forced one does not");
    let (output, _) = audited(audit_of_budget(&client, seed, &honest(seed)));
    let head = head_of(&output);
    let forced: Vec<Option<i32>> = head.positions.iter().map(|p| p.forced).collect();
    assert_eq!(forced, [None, Some(sampled[1]), None]);
    // the free tokens were judged by the draw, the forced one was not
    assert!(head.positions[0].draw.as_ref().is_some_and(|d| d.pass));
    assert!(head.positions[1].draw.is_none() && head.positions[1].judged);
    assert!(head.positions[2].draw.as_ref().is_some_and(|d| d.pass));
    let reasoning = head.reasoning.as_ref().expect("the stated sampler");
    assert_eq!((reasoning.budget, reasoning.forced), (0, 1));
    assert_eq!(reasoning.start, tag(sampled[0]));
    assert!(output.notes.iter().all(|n| !n.contains("not checked")));
    let text = output.render_text();
    assert!(
        text.contains("reasoning budget of 0 tokens") && text.contains("forced 1 of the checked"),
        "{text}"
    );

    // A node that sampled another token where its sampler forces one: the
    // statement, the tokens and the tags are signed, so anyone finds it.
    let other = drawn_with_budget(seed, &tag(sampled[0]), &tag(sampled[1] + 1));
    let (output, evidence) = audited(audit_of_budget(&client, seed, &other));
    assert_eq!(output.outcome, Outcome::Fail);
    let head = head_of(&output);
    assert_eq!(head.positions[1].forced, Some(sampled[1] + 1));
    assert!(!head.positions[1].pass);
    assert!(output.provable(), "{:?}", output.failures);
    assert!(
        output
            .render_text()
            .contains("the reasoning budget forces token"),
        "{}",
        output.render_text()
    );
    let model = ToyClaimed { unloadable: false };
    let (rechecked, _) = audited(recheck_evidence(
        &evidence.expect("evidence"),
        &toy_turn::weights(),
        &model,
        LAYERS,
    ));
    assert_eq!(rechecked.outcome, Outcome::Fail);
    assert_eq!(
        head_of(&rechecked).positions[1].forced,
        Some(sampled[1] + 1)
    );
}

#[test]
fn a_reasoning_sampler_stated_outside_the_limits_leaves_the_tokens_unjudged() {
    let client = Client::new();
    let long = "x".repeat(4096);
    let (output, _) = audited(audit_of_budget(
        &client,
        7,
        &drawn_with_budget(7, &long, "<1>"),
    ));
    // the logits are still checked; the tokens are a gap the report names
    assert_eq!(output.outcome, Outcome::Pass);
    let head = head_of(&output);
    assert!(head.positions.iter().all(|p| !p.judged && p.draw.is_none()));
    assert!(head.rule.starts_with("not judged"), "{}", head.rule);
    assert!(head.reasoning.is_none());
    assert!(
        output.notes.iter().any(|n| n.contains("not checked")),
        "{:?}",
        output.notes
    );
}
