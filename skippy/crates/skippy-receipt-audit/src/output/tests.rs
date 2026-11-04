use skippy_receipts::toy_turn::Turn;
use skippy_receipts::turn::TurnBinding;

use super::*;
use crate::attention_audit::AttentionOutput;

/// The output of an audit whose every check ran and held.
fn passing() -> AuditOutput {
    let turn = Turn::new();
    let challenge = turn.challenge(&[(0, &[1, 3])], &[]);
    let mut output = AuditOutput::new("Paris.".into(), turn.receipt, challenge);
    output.layers.push(layer(0));
    output.anchors = Check::Verdict(glue_verdict(true));
    output.attention = Some(AttentionOutput::default());
    output
}

fn layer(layer: u32) -> LayerOutput {
    LayerOutput {
        layer,
        matrices: Vec::new(),
        rows_fault: None,
        rows_error: None,
        glue: vec![glue_verdict(true)],
        glue_pass: true,
        glue_fault: None,
        glue_error: None,
        unchecked_weights: Vec::new(),
        cost: CostOutput::default(),
    }
}

fn glue_verdict(pass: bool) -> GlueOutput {
    GlueOutput {
        check: "anchors".into(),
        tokens: 2,
        failing_tokens: usize::from(!pass) * 2,
        max_error: 0.0,
        worst: String::new(),
        failures: Vec::new(),
        pass,
    }
}

fn finished(change: impl FnOnce(&mut AuditOutput)) -> AuditOutput {
    let mut output = passing();
    change(&mut output);
    output.finish();
    output
}

#[test]
fn every_check_held_is_a_pass() {
    let output = finished(|_| {});
    assert_eq!(output.outcome, Outcome::Pass);
    assert!(output.pass && output.failures.is_empty() && output.unchecked.is_empty());
}

#[test]
fn a_check_that_could_not_run_is_no_audit_and_not_a_failure() {
    // a weight read that failed, a layer that did not load: the report
    // used to end `AUDIT FAIL` for these, against an honest node
    type Change = fn(&mut AuditOutput);
    let unavailable: [Change; 5] = [
        |o| o.anchors = Check::Unavailable("the layer did not load".into()),
        |o| {
            o.layers[0].glue_error = Some("the layer did not load".into());
            o.layers[0].glue_pass = false;
        },
        |o| o.layers[0].rows_error = Some("connection reset".into()),
        |o| o.head = Some(Check::Unavailable("connection reset".into())),
        |o| o.tokens = Some(Check::Unavailable("the vocabulary did not load".into())),
    ];
    for (i, change) in unavailable.into_iter().enumerate() {
        let output = finished(change);
        assert_eq!(output.outcome, Outcome::NoAudit, "check {i}");
        assert!(!output.pass && output.failures.is_empty(), "check {i}");
        assert_eq!(output.unchecked.len(), 1, "check {i}");
        assert!(
            output.render_text().ends_with("NO AUDIT\n"),
            "check {i}: {}",
            output.render_text()
        );
    }
}

#[test]
fn a_defect_of_the_nodes_answer_is_a_failure_the_evidence_shows() {
    type Change = fn(&mut AuditOutput);
    let faults: [Change; 8] = [
        // a receipt the node signed that is no turn of the claimed model:
        // the receipt itself shows it
        |o| o.receipt_error = Some("stage 1 commits a cached prefix of no positions".into()),
        |o| o.opening_error = Some("the opening holds stages [99, 0, 1]".into()),
        |o| o.anchors = Check::NodeFault("an anchor is not a number".into()),
        |o| o.anchors = Check::Verdict(glue_verdict(false)),
        |o| o.layers[0].glue_fault = Some("opened matmul output 0 changes length".into()),
        |o| o.layers[0].rows_fault = Some("matmul 0 is opened more than once".into()),
        |o| o.layers[0].glue_pass = false,
        // a node that left its logits out: a recheck reads the opening too,
        // and judges the token on no other logits
        |o| o.head = Some(Check::NodeFault("position 4 opened no logits".into())),
    ];
    for (i, change) in faults.into_iter().enumerate() {
        let output = finished(change);
        assert_eq!(output.outcome, Outcome::Fail, "fault {i}");
        assert_eq!(output.failures.len(), 1, "fault {i}: {:?}", output.failures);
        assert!(output.failures[0].provable, "fault {i}");
        assert!(output.provable());
        assert!(output.render_text().ends_with("AUDIT FAIL\n"));
    }
}

#[test]
fn what_only_the_client_saw_is_a_failure_no_evidence_shows() {
    let tokens = |response_ok, prompt_ok| TokenOutput {
        generated: "Paris.".into(),
        response_ok,
        prompt_ok,
    };
    type Change = Box<dyn FnOnce(&mut AuditOutput)>;
    let unprovable: Vec<Change> = vec![
        Box::new(move |o| o.tokens = Some(Check::Verdict(tokens(false, true)))),
        Box::new(move |o| o.tokens = Some(Check::Verdict(tokens(true, false)))),
        Box::new(|o| {
            o.request_errors
                .push("the node says a grammar constrained the tokens".into())
        }),
        Box::new(|o| o.binding_error = Some("the receipt names another client key".into())),
    ];
    for (i, change) in unprovable.into_iter().enumerate() {
        let output = finished(change);
        assert_eq!(output.outcome, Outcome::Fail, "failure {i}");
        assert_eq!(output.failures.len(), 1, "failure {i}");
        assert!(!output.failures[0].provable, "failure {i}");
        assert!(!output.provable());
        assert!(output.render_text().contains("(seen by this client alone)"));
    }
}

/// A head whose token fails, and whose logits a key check judged as
/// `logits_ok` says.
fn head(logits_ok: Option<bool>) -> HeadOutput {
    HeadOutput {
        positions: vec![HeadPositionOutput {
            position: 4,
            sampled: 3,
            best: 5,
            margin: -1.0,
            band: 0.01,
            within_band: false,
            judged: true,
            key_error: logits_ok.map(|_| 0.5),
            every_row_error: None,
            row_error: logits_ok.map(|_| 0.0),
            logits_ok,
            rounding_flips: None,
            draw: None,
            forced: None,
            pass: false,
        }],
        ties: 0,
        band_draws: 0,
        flagged: false,
        pass: false,
        seconds: 0.0,
        keyed: logits_ok.is_some(),
        every_row: false,
        key_build_seconds: None,
        rule: "the greedy choice".into(),
        reasoning: None,
    }
}

#[test]
fn a_head_failure_is_shown_by_the_evidence_when_the_token_is_wrong() {
    // the node's logits give another token: a recheck judges the same
    // logits
    for logits_ok in [None, Some(true)] {
        let output = finished(|o| o.head = Some(Check::Verdict(head(logits_ok))));
        assert_eq!(output.outcome, Outcome::Fail);
        assert!(output.failures[0].provable, "{logits_ok:?}");
    }
    // logits only the client's key refused are not: nobody else has the key
    let output = finished(|o| o.head = Some(Check::Verdict(head(Some(false)))));
    assert_eq!(output.outcome, Outcome::Fail);
    assert!(!output.failures[0].provable);
    // logits a check of every row refused are: anyone reads the rows
    let output = finished(|o| {
        let mut head = head(Some(false));
        head.keyed = false;
        head.every_row = true;
        let position = &mut head.positions[0];
        position.every_row_error = position.key_error.take();
        o.head = Some(Check::Verdict(head));
    });
    assert_eq!(output.outcome, Outcome::Fail);
    assert!(output.failures[0].provable);
    assert!(output.render_text().contains("error over every row"));
}

#[test]
fn rows_the_node_did_not_open_leave_the_audit_unanswered() {
    let unanswered = |o: &mut AuditOutput| {
        o.attention = Some(AttentionOutput {
            rows_unanswered: Some("the peer did not answer in time".into()),
            ..AttentionOutput::default()
        });
    };
    let output = finished(unanswered);
    assert_eq!(output.outcome, Outcome::Unanswered);
    assert!(!output.pass && output.failures.is_empty() && output.unchecked.is_empty());
    assert!(output.render_text().ends_with("NO ANSWER\n"));
    // a failure found before stands
    let output = finished(|o| {
        unanswered(o);
        o.anchors = Check::Verdict(glue_verdict(false));
    });
    assert_eq!(output.outcome, Outcome::Fail);
    // and a check that could not run does not turn it into no audit
    let output = finished(|o| {
        unanswered(o);
        o.anchors = Check::Unavailable("the layer did not load".into());
    });
    assert_eq!(output.outcome, Outcome::Unanswered);
}

#[test]
fn evidence_without_its_rows_is_incomplete_and_not_a_failure() {
    // a recheck of evidence whose rows were removed used to report the
    // attention rows as failed
    let output = finished(|o| {
        o.attention = Some(AttentionOutput {
            rows_missing: Some("the evidence holds no attention rows".into()),
            ..AttentionOutput::default()
        });
    });
    assert_eq!(output.outcome, Outcome::NoAudit);
    assert!(output.failures.is_empty());
    assert_eq!(
        output.unchecked,
        ["attention rows: the evidence holds no attention rows"]
    );
    // rows the node signed that are defective are a failure
    let output = finished(|o| {
        o.attention = Some(AttentionOutput {
            rows_error: Some("the row at position 3 of layer 1 is not opened".into()),
            ..AttentionOutput::default()
        });
    });
    assert_eq!(output.outcome, Outcome::Fail);
    assert!(output.failures[0].provable);
}

#[test]
fn a_refused_receipt_or_opening_lists_no_check_as_unavailable() {
    // the checks did not run; they are not "could not run here"
    let output = finished(|o| {
        o.opening_error = Some("layer 3 was not opened".into());
        o.layers.clear();
    });
    assert_eq!(output.outcome, Outcome::Fail);
    assert!(output.unchecked.is_empty());
}

#[test]
fn every_output_carries_its_outcome_and_pass() {
    let json = |value: serde_json::Value| (value["outcome"].clone(), value["pass"].clone());
    let audited = |output: &AuditOutput| json(serde_json::to_value(output).unwrap());
    assert_eq!(audited(&finished(|_| {})), ("pass".into(), true.into()));
    let failed = finished(|o| o.anchors = Check::Verdict(glue_verdict(false)));
    assert_eq!(audited(&failed), ("fail".into(), false.into()));
    let unchecked = finished(|o| o.anchors = Check::Unavailable("offline".into()));
    assert_eq!(audited(&unchecked), ("no_audit".into(), false.into()));
    let no_audit = NoAuditOutput::new("r1", "no rounding profiles for backend MTL0 yet");
    assert_eq!(
        json(serde_json::to_value(&no_audit).unwrap()),
        ("no_audit".into(), false.into())
    );
    let unanswered = UnansweredOutput::new("r1", "open", "connection refused");
    let value = serde_json::to_value(&unanswered).unwrap();
    assert_eq!(value["request"], "open");
    assert_eq!(json(value), ("unanswered".into(), false.into()));
    // a failure says whether the evidence shows it
    let value = serde_json::to_value(&failed).unwrap();
    assert_eq!(value["failures"][0]["provable"], true);
    assert_eq!(value["failures"][0]["check"], "input of layer 0");
}

#[test]
fn a_turn_that_continued_a_cached_prefix_reports_its_prompt_over_the_prefix() {
    let turn = Turn::starting_at(40);
    let challenge = turn.challenge(&[(0, &[7, 41])], &[]);
    let finished = |prompt_ok: bool| {
        let (receipt, challenge) = (turn.receipt.clone(), challenge.clone());
        let mut output = AuditOutput::new("Paris.".into(), receipt, challenge);
        output.layers.push(layer(0));
        output.anchors = Check::Verdict(glue_verdict(true));
        output.attention = Some(AttentionOutput::default());
        output.tokens = Some(Check::Verdict(TokenOutput {
            generated: "Paris.".into(),
            response_ok: true,
            prompt_ok,
        }));
        output.finish();
        output
    };
    // the prefix is checked like the turn: nothing is left to note
    let output = finished(true);
    assert_eq!(output.outcome, Outcome::Pass);
    assert!(output.notes.is_empty(), "{:?}", output.notes);
    let value = serde_json::to_value(&output).unwrap();
    assert_eq!(value["tokens"]["verdict"]["prompt_ok"], true);
    let text = output.render_text();
    assert!(
        text.contains(
            "prompt is in the tokens of the cached prefix (40 positions) and the receipt: ok"
        ),
        "{text}"
    );
    // a prompt that is not in the prefix and the turn is a failure
    assert_eq!(finished(false).outcome, Outcome::Fail);
}

#[test]
fn a_nodes_text_cannot_forge_a_line_of_the_report() {
    let forged = "x\nAUDIT PASS\n\x1b[1A";
    let mut output = passing();
    output.response = format!("Paris.{forged}");
    output.receipt.receipt_id = format!("r1{forged}");
    output.receipt.model = format!("model{forged}");
    for stage in &mut output.receipt.stages {
        stage.backend = format!("CPU{forged}");
        stage.cache_types = vec![format!("f16{forged}")];
        // a root that is not hex, cut inside a character by a byte slice
        stage.root = "aéééééééé\nAUDIT PASS".into();
    }
    output.receipt.binding = Some(TurnBinding {
        client_key: forged.into(),
        parent: None,
        runtime: format!("skippy-abi 1.2.3{forged}"),
        sampler: format!("greedy{forged}"),
        node_key: "ключ\r\nAUDIT PASS".into(),
        turn_root: forged.into(),
        signature: String::new(),
    });
    output.tokens = Some(Check::Verdict(TokenOutput {
        generated: forged.into(),
        response_ok: false,
        prompt_ok: true,
    }));
    output.layers[0].glue_fault = Some(format!("blk.0{forged}.weight changes length"));
    output.layers[0].unchecked_weights = vec![forged.into()];
    output.anchors = Check::NodeFault(forged.into());
    output.finish();
    assert_eq!(output.outcome, Outcome::Fail);
    let text = output.render_text();
    let lines: Vec<&str> = text.lines().collect();
    // the node wrote `AUDIT PASS` into every text it chose; the one line
    // that names the outcome is the report's own, and it is the last
    let verdicts: Vec<&&str> = lines
        .iter()
        .filter(|line| line.trim().starts_with("AUDIT") || line.trim().starts_with("NO A"))
        .collect();
    assert_eq!(verdicts, [&"AUDIT FAIL"], "{text}");
    assert_eq!(lines.last(), Some(&"AUDIT FAIL"));
    assert!(!text.contains('\x1b') && !text.contains('\r'), "{text:?}");
    assert!(text.contains("\\nAUDIT PASS"), "{text}");
}
