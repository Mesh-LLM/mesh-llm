//! `exchange_corroboration/1` and the historical-reference run (PRD T1),
//! against ledgers built in-process.

use super::*;
use crate::evidence_answer::{answer, Outcome as Answer, Responder};
use crate::evidence_request_parity::{
    announced_keys, asker_key, corpus_checkpoints, corpus_ledger, sealed, write_jsonl,
    NODE_KEY_SEED, RESPONDER_NODE,
};
use crate::requester_auth::{Keys, ReplayCache};
use ed25519_dalek::SigningKey;
use evidence_request::envelope::{self, Binding, Envelope, EnvelopeParts};
use evidence_request::jcs;

const NOW: &str = "2026-09-29T00:00:00Z";

fn now() -> chrono::DateTime<chrono::Utc> {
    chrono::DateTime::parse_from_rfc3339(NOW).unwrap().into()
}

fn node_key() -> SigningKey {
    SigningKey::from_bytes(&NODE_KEY_SEED)
}

fn other_node_key() -> SigningKey {
    SigningKey::from_bytes(&[0x7a; 32])
}

fn digest_of(text: &str) -> String {
    hex::encode(Sha256::digest(text.as_bytes()))
}

/// One half of an exchange, as the emitter seals it.
fn half(
    role: &str,
    nonce: Option<&str>,
    source: &str,
    req: &str,
    resp: &str,
    other: &str,
) -> Value {
    let mut poc = json!({"role": role, "client_nonce_source": source});
    if let Some(nonce) = nonce {
        poc["client_nonce"] = json!(nonce);
    }
    poc["serving_provenance"] = if role == "served" {
        json!({"requesting_party": other, "served_by_node_id": RESPONDER_NODE})
    } else {
        json!({"served_by_node_id": other})
    };
    sealed(json!({
        "action_type": "fyi",
        "effect": {"status": "confirmed", "request_digest": req, "response_digest": resp},
        "model_attestation": {"compute_attestation": {"x-mesh-poc-v1": poc}},
    }))
}

/// `record` sealed again after a change.
fn resealed(mut record: Value) -> Value {
    record.as_object_mut().unwrap().remove("capsule_id");
    sealed(record)
}

/// Sorted member names: the order a map keeps is not what is checked.
fn members_of(value: &Value) -> Vec<String> {
    let mut members: Vec<String> = value.as_object().unwrap().keys().cloned().collect();
    members.sort();
    members
}

/// A ledger directory: the corpus with `records` in place of its leaves
/// 3.., all under its first checkpoint, signed by `key`.
fn node_with(records: &[Value], key: &SigningKey) -> tempfile::TempDir {
    let mut ledger = corpus_ledger();
    for (i, record) in records.iter().enumerate() {
        ledger[3 + i] = record.clone();
    }
    let dir = tempfile::tempdir().unwrap();
    write_jsonl(&dir.path().join("capsules.jsonl"), &ledger);
    let cps: Vec<Value> = corpus_checkpoints(&ledger, key)
        .into_iter()
        .map(|cp| cp.line())
        .collect();
    write_jsonl(&dir.path().join("checkpoints.jsonl"), &cps);
    dir
}

fn ask_as(
    dir: &std::path::Path,
    key: &SigningKey,
    bytes: &[u8],
    share: bool,
    share_standard_history: bool,
) -> Answer {
    let announced = announced_keys();
    answer(
        bytes,
        &Responder {
            ledger_dir: dir,
            signing_key: key,
            now: NOW,
            history_segments: "counterparties",
            share_standard_history,
            share_exchange_corroboration: share,
            keys: Keys {
                announced: &announced,
                configured: None,
            },
            replay: &ReplayCache::default(),
            node_id: Some(RESPONDER_NODE),
        },
    )
}

fn ask(dir: &std::path::Path, bytes: &[u8], share: bool) -> Answer {
    ask_as(dir, &node_key(), bytes, share, false)
}

fn unsigned(key: &str) -> Vec<u8> {
    serde_json::to_vec(&request_for(key)).unwrap()
}

fn signed_by(node: &str, key: &str) -> Vec<u8> {
    let mut request = request_for(key);
    crate::requester_auth::sign(
        crate::requester_auth::Channel::EvidenceRequest,
        &mut request,
        &asker_key(node),
        now(),
        RESPONDER_NODE,
    );
    serde_json::to_vec(&request).unwrap()
}

fn refused(outcome: &Answer) -> Option<Reason> {
    match outcome {
        Answer::Refused { reason, .. } => Some(*reason),
        _ => None,
    }
}

fn wire_of(outcome: Answer) -> Value {
    match outcome {
        Answer::Answered { wire, .. } => wire,
        other => panic!("answered, not {other:?}"),
    }
}

const NONCE: &str = "d0e5dfbc-b97e-44e4-ac59-63b8673749d2";
const OTHER_NONCE: &str = "11111111-2222-4333-8444-555555555555";

fn req_digest() -> String {
    "4b".repeat(32)
}
fn resp_digest() -> String {
    "33".repeat(32)
}

fn served_half() -> Value {
    half(
        "served",
        Some(NONCE),
        "client_supplied",
        &req_digest(),
        &resp_digest(),
        "node-a",
    )
}

fn requested_half() -> Value {
    half(
        "requested",
        Some(NONCE),
        "host_forwarded_nonce",
        &req_digest(),
        &resp_digest(),
        RESPONDER_NODE,
    )
}

// ----- responder -----

#[test]
fn a_half_is_answered_by_its_nonce_digest_with_exactly_the_derivation_members() {
    let dir = node_with(&[served_half()], &node_key());
    let key = digest_of(NONCE);
    let by_b = wire_of(ask(dir.path(), &signed_by("node-b", &key), true));
    let by_z = wire_of(ask(dir.path(), &signed_by("node-z", &key), true));
    assert_eq!(
        by_b["artifact"], by_z["artifact"],
        "the same bytes for every requester"
    );
    assert_eq!(
        envelope_of(&by_b).verification_content,
        envelope_of(&by_z).verification_content
    );

    let sent = unsigned(&key);
    let wire = wire_of(ask(dir.path(), &sent, true));
    let Verified::Halves { halves, .. } = verify(&sent, &wire, &node_key().verifying_key()) else {
        panic!("verifies");
    };
    assert_eq!(halves.len(), 1);
    assert_eq!(halves[0].role, "served");
    assert_eq!(halves[0].client_nonce_sha256, key);
    assert_eq!(
        halves[0].request_digest.as_deref(),
        Some(req_digest().as_str())
    );
    assert_eq!(
        halves[0].response_digest.as_deref(),
        Some(resp_digest().as_str())
    );

    let artifact = artifact_of(&wire);
    assert_eq!(members_of(&artifact["entries"][0]), ENTRY_MEMBERS);
}

#[test]
fn a_record_matched_only_by_exchange_id_nonce_or_binding_is_no_such_subject() {
    let by_exchange_id = "ab".repeat(32);
    let by_request_nonce = "cd".repeat(32);
    let mut record = half(
        "served",
        None,
        "host_served_observed",
        &req_digest(),
        &resp_digest(),
        "node-a",
    );
    record["exchange_id"] = json!(by_exchange_id);
    record["nonce"] = json!(by_request_nonce);
    let record = resealed(record);
    let dir = node_with(&[record], &node_key());
    let binding = crate::evidence_log::exchange_binding(&req_digest(), &resp_digest());
    for asked in [
        by_exchange_id.as_str(),
        by_request_nonce.as_str(),
        binding.as_str(),
    ] {
        assert_eq!(
            refused(&ask(dir.path(), &unsigned(asked), true)),
            Some(Reason::NoSuchSubject),
            "{asked} must not join"
        );
    }
}

#[test]
fn the_raw_nonce_and_the_placeholder_find_nothing() {
    let placeholder = half(
        "served",
        Some(HOST_SERVED_NO_NONCE),
        "host_served_observed",
        &req_digest(),
        &resp_digest(),
        "node-a",
    );
    let dir = node_with(&[served_half(), placeholder], &node_key());
    for asked in [
        NONCE.to_string(),
        digest_of(HOST_SERVED_NO_NONCE),
        HOST_SERVED_NO_NONCE.to_string(),
    ] {
        assert_eq!(
            refused(&ask(dir.path(), &unsigned(&asked), true)),
            Some(Reason::NoSuchSubject),
            "{asked}"
        );
    }
}

#[test]
fn by_default_a_stranger_gets_policy_declined_whether_or_not_the_half_exists() {
    let dir = node_with(&[served_half()], &node_key());
    for key in [digest_of(NONCE), digest_of(OTHER_NONCE)] {
        assert_eq!(
            refused(&ask(dir.path(), &unsigned(&key), false)),
            Some(Reason::PolicyDeclined)
        );
        assert_eq!(
            refused(&ask(dir.path(), &signed_by("node-z", &key), false)),
            Some(Reason::PolicyDeclined)
        );
    }
}

#[test]
fn sharing_standard_history_does_not_release_corroboration() {
    let dir = node_with(&[served_half()], &node_key());
    let outcome = ask_as(
        dir.path(),
        &node_key(),
        &unsigned(&digest_of(NONCE)),
        false,
        true,
    );
    assert_eq!(refused(&outcome), Some(Reason::PolicyDeclined));
}

#[test]
fn with_the_setting_on_no_match_is_no_such_subject_never_an_empty_artifact() {
    let dir = node_with(&[served_half()], &node_key());
    assert_eq!(
        refused(&ask(dir.path(), &unsigned(&digest_of(OTHER_NONCE)), true)),
        Some(Reason::NoSuchSubject)
    );
}

#[test]
fn a_counterparty_is_refused_whole_when_one_half_names_another_node() {
    let for_b = half(
        "served",
        Some(NONCE),
        "client_supplied",
        &req_digest(),
        &resp_digest(),
        "node-b",
    );
    let only_a = half(
        "served",
        Some(OTHER_NONCE),
        "client_supplied",
        &req_digest(),
        &resp_digest(),
        "node-a",
    );
    let dir = node_with(&[served_half(), for_b, only_a], &node_key());
    assert_eq!(
        refused(&ask(
            dir.path(),
            &signed_by("node-a", &digest_of(NONCE)),
            false
        )),
        Some(Reason::NotAuthorized),
        "never filtered down to the asker's own half"
    );
    let wire = wire_of(ask(
        dir.path(),
        &signed_by("node-a", &digest_of(OTHER_NONCE)),
        false,
    ));
    let artifact = artifact_of(&wire);
    assert_eq!(
        artifact["entries"].as_array().unwrap().len(),
        1,
        "a party gets its own exchange"
    );
}

#[test]
fn the_artifact_names_no_requester_counterparty_exchange_id_or_body() {
    let mut record = served_half();
    record["exchange_id"] = json!("exchange-7");
    let dir = node_with(&[resealed(record)], &node_key());
    let wire = wire_of(ask(
        dir.path(),
        &signed_by("node-b", &digest_of(NONCE)),
        true,
    ));
    let env = envelope_of(&wire);
    let mut text = String::from_utf8_lossy(&env.artifact).into_owned();
    for item in &env.verification_content {
        text.push_str(&String::from_utf8_lossy(item));
    }
    for absent in [
        "node-a",
        "node-b",
        "exchange-7",
        "exchange_id",
        "client_nonce_source",
        "requesting_party",
        NONCE,
        "confirmed",
    ] {
        assert!(!text.contains(absent), "{absent} leaked: {text}");
    }
}

#[test]
fn a_tampered_inclusion_proof_is_unverifiable() {
    let dir = node_with(&[served_half()], &node_key());
    let sent = unsigned(&digest_of(NONCE));
    let wire = wire_of(ask(dir.path(), &sent, true));
    let mut artifact = artifact_of(&wire);
    // The proof's last octet is the last octet of its path's last hash.
    let mut proof = hex::decode(artifact["entries"][0]["inclusion"].as_str().unwrap()).unwrap();
    *proof.last_mut().unwrap() ^= 1;
    artifact["entries"][0]["inclusion"] = json!(hex::encode(proof));
    let artifact = jcs::to_string(&artifact).unwrap().into_bytes();
    // Re-signed by the responder's own key, so only the proof is wrong.
    let tampered = resigned(&sent, &wire, artifact, &node_key());
    assert!(matches!(
        verify(&sent, &tampered, &node_key().verifying_key()),
        Verified::Failed(why) if why.contains("not included")
    ));
}

fn envelope_of(wire: &Value) -> Envelope {
    match envelope::parse_reply(&serde_json::to_vec(wire).unwrap(), Binding::Json) {
        Ok(envelope::Reply::Envelope(env)) => *env,
        other => panic!("{other:?}"),
    }
}

fn artifact_of(wire: &Value) -> Value {
    serde_json::from_slice(&envelope_of(wire).artifact).unwrap()
}

/// `wire` with `artifact` in place of its own, signed again by `key`.
fn resigned(sent: &[u8], wire: &Value, artifact: Vec<u8>, key: &SigningKey) -> Value {
    let env = envelope_of(wire);
    let parts = EnvelopeParts {
        subject: env.subject.clone(),
        derivation: env.derivation.clone(),
        anchor: env.anchor.clone(),
        artifact,
        verification: env
            .verification
            .iter()
            .zip(&env.verification_content)
            .map(|(v, o)| (v.kind.clone(), o.clone()))
            .collect(),
        issued_at: env.issued_at.clone(),
    };
    let signed = envelope::sign(sent, parts, key).unwrap();
    serde_json::from_slice(&signed.encode(Binding::Json)).unwrap()
}

#[test]
fn an_answer_under_another_key_is_unverifiable() {
    let dir = node_with(&[served_half()], &node_key());
    let sent = unsigned(&digest_of(NONCE));
    let wire = wire_of(ask(dir.path(), &sent, true));
    assert!(matches!(
        verify(&sent, &wire, &other_node_key().verifying_key()),
        Verified::Failed(_)
    ));
}

#[test]
fn the_definition_digest_is_the_digest_of_the_definition_file() {
    let document: Value = serde_json::from_str(DEFINITION_DOCUMENT).unwrap();
    assert_eq!(document["name"], json!(DERIVATION_NAME));
    let bytes = jcs::to_string(&document).unwrap();
    assert_eq!(
        hex::encode(Sha256::digest(bytes.as_bytes())),
        DEFINITION_DIGEST
    );
    let excludes = document["excludes"].as_array().unwrap();
    for named in [
        "exchange_id",
        "client_nonce_source",
        "counterparty identity",
    ] {
        assert!(
            excludes.iter().any(|e| e.as_str() == Some(named)),
            "{named}"
        );
    }
}

// ----- Z's comparison -----

fn x_half() -> Half {
    Half {
        half: "aa".repeat(32),
        client_nonce_sha256: digest_of(NONCE),
        role: "requested".into(),
        request_digest: Some(req_digest()),
        response_digest: Some(resp_digest()),
    }
}

fn y_halves(change: impl FnOnce(&mut Half)) -> Verified {
    let mut y = x_half();
    y.half = "bb".repeat(32);
    y.role = "served".into();
    change(&mut y);
    Verified::Halves {
        anchor: "cc".repeat(32),
        halves: vec![y],
    }
}

#[test]
fn corroborate_maps_each_answer_to_its_own_outcome() {
    let x = x_half();
    assert!(matches!(
        corroborate(&x, &y_halves(|_| {})),
        Outcome::Agree { .. }
    ));
    let Outcome::Disagree { differences } =
        corroborate(&x, &y_halves(|y| y.response_digest = Some("44".repeat(32))))
    else {
        panic!("disagree");
    };
    assert_eq!(differences.len(), 1);
    assert_eq!(differences[0].field, "response_digest");
    let Outcome::Disagree { differences } = corroborate(&x, &y_halves(|y| y.request_digest = None))
    else {
        panic!("a digest Y does not bind never matches");
    };
    assert_eq!(differences[0].field, "request_digest");
    assert_eq!(
        corroborate(&x, &Verified::Refusal(Reason::NoSuchSubject)),
        Outcome::OneSided
    );
    for reason in [
        Reason::NotAuthorized,
        Reason::PolicyDeclined,
        Reason::DeadlineUnmet,
    ] {
        assert_eq!(
            corroborate(&x, &Verified::Refusal(reason)),
            Outcome::SignedRefusal { reason }
        );
    }
    assert!(matches!(
        corroborate(&x, &Verified::Failed("bad".into())),
        Outcome::Unverifiable { .. }
    ));
}

#[test]
fn two_halves_with_the_same_role_disagree_on_role() {
    let mut x = x_half();
    x.role = "served".into();
    let Outcome::Disagree { differences } = corroborate(&x, &y_halves(|_| {})) else {
        panic!("two served halves are not one exchange's two sides");
    };
    assert_eq!(differences[0].field, "role");
}

// ----- Z's run -----

fn y(peer: &str) -> AskY {
    AskY {
        peer_id: peer.into(),
        frame: Frame {
            basis: "x_record_names_y".into(),
            note: None,
        },
        independence_basis: None,
    }
}

fn args_with_key(ys: Vec<AskY>) -> RunArgs {
    RunArgs {
        x_peer: "node-x".into(),
        client_nonce_sha256: Some(digest_of(NONCE)),
        x_record: None,
        ys,
    }
}

/// An answer from a ledger holding `records`, signed by `key`, as the glue
/// would hand it over.
fn reply_from(records: &[Value], key: &SigningKey, share: bool) -> Reply {
    let dir = node_with(records, key);
    let sent = signed_by("node-z", &digest_of(NONCE));
    let answer = match ask_as(dir.path(), key, &sent, share, false) {
        Answer::Answered { wire, .. } | Answer::Refused { wire, .. } => wire,
        Answer::Unanswerable(why) => panic!("{why}"),
    };
    Reply::Answered {
        sent,
        answer,
        key: Some(Box::new(key.verifying_key())),
    }
}

#[test]
fn x_and_y_halves_of_one_exchange_agree_end_to_end() {
    let plan = plan(args_with_key(vec![y("node-y")])).unwrap();
    let x_reply = reply_from(&[requested_half()], &node_key(), true);
    let x = after_x(&plan, &x_reply);
    assert!(x.is_ok(), "{x:?}");
    let y_reply = reply_from(&[served_half()], &other_node_key(), true);
    let result = finish(&plan, x, &[y_reply]);
    assert_eq!(result["x"]["outcome"]["state"], json!("half_obtained"));
    assert_eq!(
        result["ys"][0]["outcome"]["state"],
        json!("agree"),
        "{result}"
    );
}

#[test]
fn a_pair_with_no_shared_nonce_is_not_joinable_and_no_one_is_asked() {
    // The same digests, sealed at the same time, but X's half binds no nonce.
    let x_record = half(
        "served",
        Some(HOST_SERVED_NO_NONCE),
        "host_served_observed",
        &req_digest(),
        &resp_digest(),
        "node-y",
    );
    let fallback = half(
        "served",
        Some(NONCE),
        "plugin_generated_fallback",
        &req_digest(),
        &resp_digest(),
        "node-y",
    );
    for record in [x_record, fallback] {
        let plan = plan(RunArgs {
            x_peer: "node-x".into(),
            client_nonce_sha256: None,
            x_record: Some(record),
            ys: vec![y("node-y")],
        })
        .unwrap();
        assert!(
            matches!(plan.join_key, JoinKey::Unjoinable(_)),
            "nothing to ask by"
        );
        let x = after_x(&plan, &Reply::Nothing("never sent".into()));
        assert!(matches!(x, Err(Outcome::NotJoinable { .. })));
        let result = finish(&plan, x, &[]);
        assert_eq!(result["ys"][0]["outcome"]["state"], json!("not_joinable"));
        assert_eq!(result["join_key"]["kind"], json!("none"));
    }
}

#[test]
fn a_refusal_is_never_an_absence_and_silence_never_a_refusal() {
    let plan = plan(args_with_key(vec![y("node-y"), y("node-w")])).unwrap();
    let x = after_x(&plan, &reply_from(&[requested_half()], &node_key(), true));
    let refusing = reply_from(&[served_half()], &other_node_key(), false);
    let silent = Reply::Nothing("no answer by the deadline".into());
    let result = finish(&plan, x, &[refusing, silent]);
    assert_eq!(
        result["ys"][0]["outcome"],
        json!({"state": "signed_refusal", "reason": "policy_declined"})
    );
    assert_eq!(
        result["ys"][1]["outcome"]["state"],
        json!("recorded_absence")
    );
}

#[test]
fn when_x_refuses_or_is_silent_the_run_stops_and_no_y_is_asked() {
    let plan = plan(args_with_key(vec![y("node-y")])).unwrap();
    let x = after_x(&plan, &reply_from(&[requested_half()], &node_key(), false));
    assert_eq!(
        x,
        Err(Outcome::SignedRefusal {
            reason: Reason::PolicyDeclined
        })
    );
    let result = finish(&plan, x, &[]);
    assert_eq!(result["x"]["outcome"]["state"], json!("signed_refusal"));
    assert_eq!(result["ys"][0]["outcome"]["state"], json!("not_asked"));

    let x = after_x(&plan, &Reply::Nothing("silence".into()));
    assert!(matches!(x, Err(Outcome::RecordedAbsence { .. })));
}

#[test]
fn each_y_states_independence_and_nothing_is_summed_across_ys() {
    let mut declared = y("node-w");
    declared.independence_basis = Some("separate operators, by contract".into());
    let plan = plan(args_with_key(vec![y("node-y"), declared])).unwrap();
    let x = after_x(&plan, &reply_from(&[requested_half()], &node_key(), true));
    let agreeing = reply_from(&[served_half()], &other_node_key(), true);
    let refusing = reply_from(&[served_half()], &other_node_key(), false);
    let result = finish(&plan, x, &[agreeing, refusing]);

    assert_eq!(
        members_of(&result),
        ["join_key", "x", "ys"],
        "no member summarizes across Ys"
    );
    for entry in result["ys"].as_array().unwrap() {
        assert_eq!(
            members_of(entry),
            ["frame", "independence", "outcome", "peer_id"]
        );
    }
    assert_eq!(result["ys"][0]["outcome"]["state"], json!("agree"));
    assert_eq!(
        result["ys"][0]["independence"],
        json!({"state": "not_established"})
    );
    assert_eq!(
        result["ys"][1]["outcome"]["state"],
        json!("signed_refusal"),
        "kept apart, unmerged"
    );
    assert_eq!(result["ys"][1]["independence"]["state"], json!("declared"));
}

#[test]
fn a_run_names_each_y_once_and_never_x() {
    assert!(
        plan(args_with_key(vec![y("node-y"), y("NODE-Y")])).is_err(),
        "named twice"
    );
    assert!(plan(args_with_key(vec![y("node-x")])).is_err(), "X as a Y");
    assert!(plan(args_with_key(vec![])).is_err(), "no Y");
    let mut odd = y("node-y");
    odd.frame.basis = "nearest".into();
    assert!(
        plan(args_with_key(vec![odd])).is_err(),
        "an unknown frame basis"
    );
    let mut both = args_with_key(vec![y("node-y")]);
    both.x_record = Some(requested_half());
    assert!(plan(both).is_err(), "two key sources");
    let mut placeholder = args_with_key(vec![y("node-y")]);
    placeholder.client_nonce_sha256 = Some(digest_of(HOST_SERVED_NO_NONCE));
    assert!(matches!(
        plan(placeholder).unwrap().join_key,
        JoinKey::Unjoinable(_)
    ));
}

#[test]
fn a_run_is_recorded_beside_the_ledger() {
    let dir = tempfile::tempdir().unwrap();
    let result = json!({"join_key": {"kind": "none", "why": "test"}, "x": {}, "ys": []});
    record_run(dir.path(), &result, NOW).unwrap();
    record_run(dir.path(), &result, NOW).unwrap();
    let text = std::fs::read_to_string(dir.path().join(RUNS_FILENAME)).unwrap();
    assert_eq!(text.lines().count(), 2);
    let line: Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
    assert_eq!(line["ran_at"], json!(NOW));
}
