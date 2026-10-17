use super::*;
use crate::evidence_answer::{self, Outcome, Responder};
use crate::evidence_request_parity::{
    announced_keys, corpus_ledger, corpus_ledger_dir, NODE_KEY_SEED, RESPONDER_NODE,
};
use ed25519_dalek::SigningKey;
use evidence_request::request;

const NOW: &str = "2026-09-29T00:00:00Z";

fn key() -> SigningKey {
    SigningKey::from_bytes(&NODE_KEY_SEED)
}

/// A node over the corpus that answers anyone everything.
struct Node {
    _dir: tempfile::TempDir,
    ledger: std::path::PathBuf,
    anchors: Vec<String>,
}

fn node() -> Node {
    let dir = tempfile::tempdir().unwrap();
    let ledger = dir.path().join("ledger");
    let anchors = corpus_ledger_dir(&ledger, &key())
        .iter()
        .map(|c| c.anchor())
        .collect();
    Node {
        _dir: dir,
        ledger,
        anchors,
    }
}

impl Node {
    fn ask(&self, request: &Value) -> (Vec<u8>, Value) {
        let bytes = serde_json::to_vec(request).unwrap();
        let announced = announced_keys();
        let replay = crate::requester_auth::ReplayCache::default();
        let key = key();
        let outcome = evidence_answer::answer_and_log(
            &bytes,
            false,
            &Responder {
                ledger_dir: &self.ledger,
                signing_key: &key,
                now: NOW,
                history_segments: "peers",
                share_standard_history: true,
                share_exchange_corroboration: false,
                keys: crate::requester_auth::Keys {
                    announced: &announced,
                    configured: None,
                },
                replay: &replay,
                node_id: Some(RESPONDER_NODE),
            },
            None,
        );
        let Outcome::Answered { wire, .. } = outcome else {
            panic!("{request}: {outcome:?}");
        };
        (bytes, wire)
    }

    fn pin(&self, i: usize) -> Value {
        json!({"expected_pin": self.anchors[i]})
    }
}

fn check_reply(sent: &[u8], reply: &Value) -> Result<Checked, CheckError> {
    let req = request::parse_json(sent).unwrap();
    check(sent, &req, reply, &key().verifying_key(), &|_| None)
}

fn envelope_of(reply: &Value) -> Envelope {
    match envelope::parse_reply(&serde_json::to_vec(reply).unwrap(), Binding::Json) {
        Ok(Reply::Envelope(env)) => *env,
        other => panic!("{other:?}"),
    }
}

/// The same answer with `change` made to its parts, signed again by the
/// responder: what a responder that lies, with its own key, can send.
fn resigned(sent: &[u8], reply: &Value, change: impl FnOnce(&mut EnvelopeParts)) -> Value {
    resigned_by(&key(), sent, reply, change)
}

/// [`resigned`], by the responder whose key is `key`.
fn resigned_by(
    key: &SigningKey,
    sent: &[u8],
    reply: &Value,
    change: impl FnOnce(&mut EnvelopeParts),
) -> Value {
    let env = envelope_of(reply);
    let mut parts = EnvelopeParts {
        subject: env.subject.clone(),
        derivation: env.derivation.clone(),
        anchor: env.anchor.clone(),
        artifact: env.artifact.clone(),
        verification: env
            .verification
            .iter()
            .zip(&env.verification_content)
            .map(|(v, o)| (v.kind.clone(), o.clone()))
            .collect(),
        issued_at: env.issued_at.clone(),
    };
    change(&mut parts);
    let signed = envelope::sign(sent, parts, key).unwrap();
    let mut out: Value = serde_json::from_slice(&signed.encode(Binding::Json)).unwrap();
    if let Some(r) = reply.get(WITNESS_RECEIPTS) {
        out[WITNESS_RECEIPTS] = r.clone();
    }
    out
}

fn artifact_json(parts: &EnvelopeParts) -> Value {
    serde_json::from_slice(&parts.artifact).unwrap()
}

fn jcs_bytes(v: &Value) -> Vec<u8> {
    evidence_request::jcs::to_string(v).unwrap().into_bytes()
}

/// A record answer carries the record's line and the profile's
/// Checkpointed Local Log material for its `capsule_id`. A profile
/// verifier cannot tie that name to the line and reports `NOT_CHECKED`,
/// never `FAIL`; this node's requester checks it, for every record under
/// either anchor.
#[test]
fn a_record_answer_is_checked_here_and_never_failed_by_the_profile() {
    let node = node();
    let ledger = corpus_ledger();
    for (i, record) in ledger.iter().enumerate().take(300) {
        if record.get("record_type") == Some(&json!("padding")) || i == 21 {
            continue;
        }
        let at = usize::from(i >= 16);
        let id = record["capsule_id"].as_str().unwrap();
        let (sent, reply) = node.ask(&json!({"subject": {"record": id}, "coverage": node.pin(at)}));
        let env = envelope_of(&reply);
        assert_eq!(
            env.artifact,
            record.to_string().into_bytes(),
            "the line, whole"
        );
        let report =
            cll_material::check_anchor_inclusion(&env, &ResponderLogKey(key().verifying_key()));
        assert_eq!(report.status, cll_material::Status::NotChecked, "{i}");
        assert_eq!(
            env.verification
                .iter()
                .map(|v| v.kind.as_str())
                .collect::<Vec<_>>(),
            if at == 0 {
                vec![KIND_CHECKPOINT, KIND_INCLUSION]
            } else {
                vec![KIND_CHECKPOINT, KIND_INCLUSION, KIND_CONSISTENCY]
            }
        );
        let checked = check_reply(&sent, &reply).unwrap();
        assert_eq!(
            (checked.anchor.as_str(), checked.records.as_slice()),
            (node.anchors[at].as_str(), &[i as u64][..])
        );
    }
}

/// The same material proves the record's `capsule_id` preimage, a record
/// that is its own octets: the profile's check passes on it, and this
/// node's requester accepts it too, as it would from evidencebook.
#[test]
fn a_record_that_is_its_own_octets_passes_the_profile_and_is_accepted() {
    let node = node();
    let record = &corpus_ledger()[17];
    let id = record["capsule_id"].as_str().unwrap();
    let (sent, reply) = node.ask(&json!({"subject": {"record": id}, "coverage": node.pin(1)}));
    let mut preimage = record.clone();
    let members = preimage.as_object_mut().unwrap();
    for local in ["capsule_id", "signature", "key_id"] {
        members.remove(local);
    }
    let own = resigned(&sent, &reply, |p| p.artifact = jcs_bytes(&preimage));
    let report = cll_material::check_anchor_inclusion(
        &envelope_of(&own),
        &ResponderLogKey(key().verifying_key()),
    );
    assert_eq!(
        report.status,
        cll_material::Status::Pass,
        "{}",
        report.detail
    );
    assert_eq!(check_reply(&sent, &own).unwrap().records, [17]);
}

/// A record that is its own octets passes the profile's check whenever
/// its path reaches some peak, so a leaf that is itself a peak, sent
/// with an empty path, passes it at any position. This node's requester
/// accepts it only at its own position under the anchor's size (review
/// on Mesh-LLM/mesh-llm#2502: a real leaf 2 accepted as records `[0]`
/// and `[100]`).
#[test]
fn a_record_that_is_its_own_octets_holds_only_at_its_own_position() {
    let (_dir, ledger, key, _) = producer_ledger(3, &[2]);
    let (stated_anchors, _) = anchors_of(&ledger, &key);
    let pin = json!({"expected_pin": stated_anchors[0]});
    let ask = |request: Value| {
        let (checked, wire) = ask_ledger(&ledger, &key, &request);
        checked.unwrap();
        (serde_json::to_vec(&request).unwrap(), wire)
    };
    let check_by = |sent: &[u8], reply: &Value| {
        let req = request::parse_json(sent).unwrap();
        check(sent, &req, reply, &key.verifying_key(), &|_| None)
    };
    // A leaf that is a peak on its own, and its digest.
    let (_, full) = ask(json!({"subject": {"full_history": null}, "coverage": pin}));
    let full = envelope_of(&full);
    let size = stated(&full.verification_content[0], &key.verifying_key())
        .unwrap()
        .cp
        .mmr_size;
    let peak = cll::mmr::peaks(size)
        .unwrap()
        .into_iter()
        .find(|&p| cll::mmr::height_at(p) == 0)
        .expect("the anchor has a one-leaf peak");
    let leaf = cll::mmr::pos_to_leaf_index(peak).unwrap();
    let records: Value = serde_json::from_slice(&full.artifact).unwrap();
    let digest = records["records"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["leaf_index"] == json!(leaf))
        .unwrap()["digest"]
        .clone();
    let (sent, reply) = ask(json!({"subject": {"record": digest}, "coverage": pin}));
    let mut preimage: Value = serde_json::from_slice(&envelope_of(&reply).artifact).unwrap();
    let members = preimage.as_object_mut().unwrap();
    for local in ["capsule_id", "signature", "key_id"] {
        members.remove(local);
    }
    assert_eq!(json!(sha256_hex(&jcs_bytes(&preimage))), digest);
    let own = resigned_by(&key, &sent, &reply, |p| p.artifact = jcs_bytes(&preimage));
    assert_eq!(check_by(&sent, &own).unwrap().records, [leaf]);
    for forged in [0, leaf - 1, 100, leaf_count(size).unwrap()] {
        let bad = resigned_by(&key, &sent, &reply, |p| {
            p.artifact = jcs_bytes(&preimage);
            p.verification[1].1 = encode_inclusion_proof(forged, &[]).unwrap();
        });
        let report = cll_material::check_anchor_inclusion(
            &envelope_of(&bad),
            &ResponderLogKey(key.verifying_key()),
        );
        assert_eq!(report.status, cll_material::Status::Pass, "{forged}");
        assert_eq!(
            check_by(&sent, &bad).unwrap_err(),
            CheckError::Answer("the record's inclusion proof does not hold".into()),
            "{forged}"
        );
    }
}

/// An answer is under the profile the request named. A responder that
/// answers a request for another profile as `evidence-interop/0` is not
/// believed, even signing over that request.
#[test]
fn an_answer_under_another_profile_than_asked_is_refused() {
    let node = node();
    let id = corpus_ledger()[17]["capsule_id"]
        .as_str()
        .unwrap()
        .to_string();
    let (sent, reply) = node.ask(&json!({"subject": {"record": id}, "coverage": node.pin(1)}));
    for profile in [
        json!("evidence-interop/0"),
        json!("evidence-interop/1"),
        json!(7),
    ] {
        let mut asked: Value = serde_json::from_slice(&sent).unwrap();
        asked["profile"] = profile.clone();
        let asked = serde_json::to_vec(&asked).unwrap();
        let answer = resigned(&asked, &reply, |_| {});
        let checked = check_reply(&asked, &answer);
        if profile == json!("evidence-interop/0") {
            assert!(checked.is_ok(), "{profile}");
        } else {
            assert_eq!(
                checked.unwrap_err(),
                CheckError::Answer("the answer is not under the profile the request named".into()),
                "{profile}"
            );
        }
    }
}

/// A record answer whose line, proof or consistency proof is not the one
/// it claims is not evidence, even signed by the responder.
#[test]
fn a_resigned_record_that_does_not_hold_is_refused() {
    let node = node();
    let ledger = corpus_ledger();
    let id = ledger[17]["capsule_id"].as_str().unwrap();
    let (sent, reply) = node.ask(&json!({"subject": {"record": id}, "coverage": node.pin(1)}));
    assert!(check_reply(&sent, &reply).is_ok());
    type Change = Box<dyn Fn(&mut EnvelopeParts)>;
    let other = ledger[18].to_string().into_bytes();
    let tampered: Vec<(&str, Change)> = vec![
        (
            "another record's line",
            Box::new(move |p| p.artifact = other.clone()),
        ),
        (
            "its own line, changed",
            Box::new(|p| {
                let mut line: Value = serde_json::from_slice(&p.artifact).unwrap();
                line["effect"]["status"] = json!("failed");
                p.artifact = line.to_string().into_bytes();
            }),
        ),
        (
            "no inclusion proof",
            Box::new(|p| {
                p.verification.truncate(1);
            }),
        ),
        (
            "a consistency proof that does not hold",
            Box::new(|p| {
                let last = p.verification[2].1.len() - 1;
                p.verification[2].1[last] ^= 1;
            }),
        ),
    ];
    for (what, change) in tampered {
        let bad = resigned(&sent, &reply, |p| change(p));
        assert!(
            matches!(check_reply(&sent, &bad), Err(CheckError::Answer(_))),
            "{what}"
        );
    }
}

/// The material a record set carries beyond the profile is read by no
/// profile verifier: it reports `NOT_CHECKED` before it looks.
#[test]
fn a_record_sets_extra_material_is_not_checked_by_the_profile() {
    let node = node();
    let (sent, reply) = node.ask(&json!({"subject": {"range": [5, 12]}, "coverage": node.pin(0)}));
    let env = envelope_of(&reply);
    let report =
        cll_material::check_anchor_inclusion(&env, &ResponderLogKey(key().verifying_key()));
    assert_eq!(report.status, cll_material::Status::NotChecked);
    assert_eq!(
        check_reply(&sent, &reply).unwrap().records,
        (5..=12).collect::<Vec<_>>()
    );
}

/// The artifact is the same for every requester (-00 §5): an unsigned
/// request and requests signed by two different nodes, differing too in
/// their nonce and coverage form, get byte-identical artifacts.
#[test]
fn the_artifact_is_the_same_whoever_asks() {
    let node = node();
    let now = chrono::DateTime::parse_from_rfc3339(NOW).unwrap().into();
    let signed_by = |asker: &str, mut request: Value| {
        request["requester_id"] = json!(asker);
        crate::requester_auth::sign(
            crate::requester_auth::Channel::EvidenceRequest,
            &mut request,
            &crate::evidence_request_parity::asker_key(asker),
            now,
            RESPONDER_NODE,
        );
        request
    };
    for subject in [
        json!({"range": [16, 20]}),
        json!({"checkpoints": null}),
        json!({"correlation": "n-1"}),
    ] {
        let anonymous = node
            .ask(&json!({"subject": subject, "coverage": node.pin(1), "nonce": "a"}))
            .1;
        let by_a = node
            .ask(&signed_by(
                "node-a",
                json!({"subject": subject, "coverage": {"min_freshness": 300}, "nonce": "b"}),
            ))
            .1;
        let by_b = node
            .ask(&signed_by(
                "node-b",
                json!({"subject": subject, "coverage": node.pin(1), "nonce": "c"}),
            ))
            .1;
        for (who, other) in [("node-a", &by_a), ("node-b", &by_b)] {
            assert!(
                other.get("artifact").is_some(),
                "{subject}: {who} was answered"
            );
            assert_eq!(anonymous["artifact"], other["artifact"], "{subject}: {who}");
            assert_ne!(anonymous["request_digest"], other["request_digest"]);
        }
    }
}

/// What a responder signs with its own key is still checked against its
/// log: a record set whose body, leaf, proof or anchor is not the one it
/// claims is not evidence.
#[test]
fn a_resigned_record_set_that_does_not_hold_is_refused() {
    let node = node();
    let (sent, reply) = node.ask(&json!({"subject": {"range": [16, 20]}, "coverage": node.pin(1)}));
    assert!(check_reply(&sent, &reply).is_ok());
    type Change = Box<dyn Fn(&mut EnvelopeParts)>;
    let tampered: Vec<(&str, Change)> = vec![
        (
            "a changed body",
            Box::new(|p| {
                let mut art = artifact_json(p);
                art["records"][1]["body"] = json!(hex::encode(b"{}"));
                p.artifact = jcs_bytes(&art);
            }),
        ),
        (
            "a body under another record's digest",
            Box::new(|p| {
                let mut art = artifact_json(p);
                art["records"][1]["body"] = art["records"][2]["body"].clone();
                art["records"][1]["digest"] = art["records"][2]["digest"].clone();
                p.artifact = jcs_bytes(&art);
            }),
        ),
        (
            "a record dropped from the range",
            Box::new(|p| {
                let mut art = artifact_json(p);
                art["records"].as_array_mut().unwrap().remove(2);
                p.artifact = jcs_bytes(&art);
                p.verification.remove(3);
            }),
        ),
        ("proofs swapped", Box::new(|p| p.verification.swap(1, 2))),
        (
            "a proof missing",
            Box::new(|p| {
                p.verification.pop();
            }),
        ),
        (
            "another anchor named",
            Box::new(|p| {
                let mut art = artifact_json(p);
                art["anchor"] = json!("e".repeat(64));
                p.artifact = jcs_bytes(&art);
            }),
        ),
        (
            "the anchor checkpoint signed by another key",
            Box::new(|p| {
                let other = SigningKey::from_bytes(&[9; 32]);
                let ledger = corpus_ledger();
                let cps = crate::evidence_request_parity::corpus_checkpoints(&ledger, &other);
                p.verification[0].1 = cps[1].statement.clone();
            }),
        ),
    ];
    for (what, change) in tampered {
        let bad = resigned(&sent, &reply, |p| change(p));
        assert!(
            matches!(check_reply(&sent, &bad), Err(CheckError::Answer(_))),
            "{what}"
        );
    }
}

/// A `correlation` or `exchange` answer may carry only records that
/// carry the identifier or cite the half. A record that does not, slipped
/// in with its own valid proof under the same anchor and signed by the
/// responder, is refused; so is a padding record.
#[test]
fn a_record_set_with_a_record_the_subject_does_not_name_is_refused() {
    let node = node();
    let half = hex::encode(Sha256::digest(b"the counterparty's own half"));
    // The exchange is answered under the earliest checkpoint covering
    // the citing record, checkpoint 0; the others under checkpoint 1.
    for (subject, coverage, anchor, outsider) in [
        (json!({"correlation": "n-1"}), node.pin(1), 1, 1u64),
        (
            json!({"exchange": half}),
            json!({"expected_pin": half}),
            0,
            5u64,
        ),
        (json!({"correlation": "n-1"}), node.pin(1), 1, 8u64),
    ] {
        let (sent, reply) = node.ask(&json!({"subject": subject, "coverage": coverage}));
        let checked = check_reply(&sent, &reply).unwrap();
        assert!(!checked.records.contains(&outsider), "{subject}");
        // The outsider and its proof, from a range answer under the same anchor.
        let (_, other) = node.ask(&json!({
            "subject": {"range": [outsider, outsider]},
            "coverage": node.pin(anchor),
        }));
        let other = envelope_of(&other);
        let other_record =
            serde_json::from_slice::<Value>(&other.artifact).unwrap()["records"][0].clone();
        let other_proof = other.verification_content[1].clone();
        let bad = resigned(&sent, &reply, |p| {
            let mut art = artifact_json(p);
            let records = art["records"].as_array_mut().unwrap();
            let at = records
                .iter()
                .position(|r| r["leaf_index"].as_u64().unwrap() > outsider)
                .unwrap_or(records.len());
            records.insert(at, other_record.clone());
            p.artifact = jcs_bytes(&art);
            p.verification
                .insert(at + 1, (KIND_INCLUSION.to_string(), other_proof.clone()));
        });
        assert!(
            matches!(check_reply(&sent, &bad), Err(CheckError::Answer(_))),
            "{subject} with record {outsider}"
        );
    }
}

/// The index never files padding under a subject, so a padding record
/// that carries the identifier or cites the half still does not fit.
#[test]
fn padding_never_fits_a_correlation_or_exchange() {
    let half = "c".repeat(64);
    let record = |record_type: &str| {
        serde_json::to_vec(&json!({
            "record_type": record_type,
            "nonce": "n-1",
            "references": [{
                "type": "capsule", "digest_alg": "SHA-256", "digest": half,
                "citation_purpose": "counterparty_half",
            }],
        }))
        .unwrap()
    };
    for subject in [
        Subject::Correlation("n-1".into()),
        Subject::Exchange(half.clone()),
    ] {
        assert!(record_fits(&subject, &record("served")), "{subject:?}");
        assert!(!record_fits(&subject, &record("padding")), "{subject:?}");
    }
    assert!(!record_fits(
        &Subject::Correlation("n-1".into()),
        b"not json"
    ));
}

/// A checkpoint list and a history card must start the stream and chain
/// to the anchor, and a history card's consistency proofs must hold.
#[test]
fn a_resigned_checkpoint_list_or_history_that_does_not_chain_is_refused() {
    let node = node();
    let (sent, reply) =
        node.ask(&json!({"subject": {"checkpoints": null}, "coverage": node.pin(1)}));
    assert_eq!(check_reply(&sent, &reply).unwrap().checkpoints, 2);
    let bad = resigned(&sent, &reply, |p| {
        let mut art = artifact_json(p);
        art["checkpoints"].as_array_mut().unwrap().remove(0);
        p.artifact = jcs_bytes(&art);
    });
    assert!(
        check_reply(&sent, &bad).is_err(),
        "a list that does not start the stream"
    );

    let card = json!({"subject": {"full_history": null}, "coverage": node.pin(1), "derivation": "history_card/1"});
    let (sent, reply) = node.ask(&card);
    assert_eq!(check_reply(&sent, &reply).unwrap().checkpoints, 2);
    let bad = resigned(&sent, &reply, |p| {
        let mut art = artifact_json(p);
        let proof = art["history"][1]["consistency_from_previous"]
            .as_str()
            .unwrap()
            .to_string();
        let mut bytes = hex::decode(proof).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 1;
        art["history"][1]["consistency_from_previous"] = json!(hex::encode(bytes));
        p.artifact = jcs_bytes(&art);
    });
    assert!(
        check_reply(&sent, &bad).is_err(),
        "a consistency proof that does not hold"
    );
    let bad = resigned(&sent, &reply, |p| {
        let mut art = artifact_json(p);
        art["history"][1]["consistency_from_previous"] = Value::Null;
        p.artifact = jcs_bytes(&art);
    });
    assert!(
        check_reply(&sent, &bad).is_err(),
        "a missing consistency proof"
    );
}

/// The envelope must answer what was asked: the subject, the pin, and
/// the freshness the anchor attains as the envelope states it.
#[test]
fn an_answer_to_another_question_is_refused() {
    let node = node();
    let ledger = corpus_ledger();
    let id = ledger[0]["capsule_id"].as_str().unwrap();
    let (sent, reply) =
        node.ask(&json!({"subject": {"record": id}, "coverage": {"min_freshness": 300}}));
    assert!(check_reply(&sent, &reply).is_ok());
    let bad = resigned(&sent, &reply, |p| {
        p.anchor.freshness = Some(AnchorFreshness::Size(1))
    });
    assert!(
        check_reply(&sent, &bad).is_err(),
        "a freshness the anchor does not attain"
    );
    // Asked under the other anchor: the same answer does not satisfy it.
    let other =
        serde_json::to_vec(&json!({"subject": {"record": id}, "coverage": node.pin(0)})).unwrap();
    let req = request::parse_json(&other).unwrap();
    assert!(check(&other, &req, &reply, &key().verifying_key(), &|_| None).is_err());
    // Asked about another record, or another subject form: the answer
    // names a subject the request did not.
    let another = ledger[1]["capsule_id"].as_str().unwrap();
    for subject in [json!({"record": another}), json!({"range": [0, 0]})] {
        let asked =
            serde_json::to_vec(&json!({"subject": subject, "coverage": {"min_freshness": 300}}))
                .unwrap();
        let req = request::parse_json(&asked).unwrap();
        // Signed for this request, so only the subject it names differs.
        let answer = resigned(&asked, &reply, |_| {});
        assert!(
            matches!(
                check(&asked, &req, &answer, &key().verifying_key(), &|_| None),
                Err(CheckError::Answer(_))
            ),
            "{subject}"
        );
    }
}

/// A served summary is under the newest anchor, named in its material.
#[test]
fn a_served_summary_names_its_checkpoint() {
    let node = node();
    let (sent, reply) = node.ask(&json!({"subject": {"full_history": null}, "coverage": node.pin(1), "derivation": "served_summary/1"}));
    let checked = check_reply(&sent, &reply).unwrap();
    assert_eq!(checked.anchor, node.anchors[1]);
    let bad = resigned(&sent, &reply, |p| p.verification.clear());
    assert!(check_reply(&sent, &bad).is_err());
}

/// A receipt is for one of the checkpoints the answer carries, by its
/// statement's digest; naming any other is malformed.
#[test]
fn a_receipt_must_name_a_checkpoint_the_answer_carries() {
    let node = node();
    let (sent, mut reply) =
        node.ask(&json!({"subject": {"checkpoints": null}, "coverage": node.pin(1)}));
    let receipts = reply[WITNESS_RECEIPTS].as_array().unwrap().clone();
    assert_eq!(receipts.len(), 1, "the corpus's one stub witness");
    assert_eq!(receipts[0]["checkpoint"], json!(node.anchors[1]));
    let checked = check_reply(&sent, &reply).unwrap();
    assert_eq!(checked.receipts[0].status, ReceiptStatus::Stub);
    reply[WITNESS_RECEIPTS][0]["checkpoint"] = json!("e".repeat(64));
    assert_eq!(
        check_reply(&sent, &reply).unwrap_err(),
        CheckError::ReceiptMalformed
    );
}

/// A ledger the producer itself wrote: `records` sealed exchanges, with
/// capsule-emit's own checkpointer cutting a checkpoint after each
/// record listed in `cuts`. Returns the directory, the ledger and the
/// node key.
fn producer_ledger(
    records: usize,
    cuts: &[usize],
) -> (
    tempfile::TempDir,
    std::path::PathBuf,
    SigningKey,
    Vec<String>,
) {
    use crate::producer::anchor::AnchorClient;
    use crate::producer::checkpoint::{CheckpointCadenceConfig, CheckpointState};
    let dir = tempfile::tempdir().unwrap();
    let state = crate::capsule_emit::CapsuleState::open(dir.path(), "node-under-test").unwrap();
    let ledger = state.ledger_dir().to_path_buf();
    let (mut checkpoints, _) = CheckpointState::load(
        &ledger,
        "node-under-test",
        CheckpointCadenceConfig::default(),
    )
    .unwrap();
    let anchor = AnchorClient::new("http://127.0.0.1:1");
    let mut ids = Vec::new();
    for i in 0..records {
        let id = format!("e-{i}");
        let emitted = state
            .emit_for_exchange(&crate::capsule_emit::ExchangeRecord {
                model: "m",
                client_nonce: None,
                request_bytes: b"{}",
                response_bytes: b"{}",
                latency_ms: 1.0,
                exchange_id: Some(&id),
                requesting_party: Some("party"),
                host_provenance: None,
            })
            .unwrap();
        ids.push(emitted.capsule_id);
        if cuts.contains(&i) {
            checkpoints
                .checkpoint_covering(&ids[i], state.signing_key(), &anchor)
                .unwrap();
        }
    }
    let key = state.signing_key().clone();
    (dir, ledger, key, ids)
}

/// The anchors of `ledger` (stated checkpoints) and its chain length.
fn anchors_of(ledger: &std::path::Path, key: &SigningKey) -> (Vec<String>, usize) {
    let key_id = hex::encode(key.verifying_key().to_bytes());
    crate::evidence_log::with_index(ledger, &key_id, |index| {
        let stated = index.chain.iter().map(|c| c.anchor_digest()).collect();
        (stated, index.chain.len())
    })
    .unwrap()
}

/// `request` answered from `ledger`, openly, and checked as a requester
/// checks it.
fn ask_ledger(
    ledger: &std::path::Path,
    key: &SigningKey,
    request: &Value,
) -> (Result<Checked, CheckError>, Value) {
    let bytes = serde_json::to_vec(request).unwrap();
    let announced = announced_keys();
    let replay = crate::requester_auth::ReplayCache::default();
    let outcome = evidence_answer::answer_and_log(
        &bytes,
        false,
        &Responder {
            ledger_dir: ledger,
            signing_key: key,
            now: NOW,
            history_segments: "peers",
            share_standard_history: true,
            share_exchange_corroboration: false,
            keys: crate::requester_auth::Keys {
                announced: &announced,
                configured: None,
            },
            replay: &replay,
            node_id: Some(RESPONDER_NODE),
        },
        None,
    );
    let Outcome::Answered { wire, .. } = outcome else {
        panic!("{request}: {outcome:?}");
    };
    let req = request::parse_json(&bytes).unwrap();
    (
        check(&bytes, &req, &wire, &key.verifying_key(), &|_| None),
        wire,
    )
}

/// A ledger the producer itself wrote, checkpointed by capsule-emit's
/// own checkpointer: its checkpoints anchor answers, and a record, the
/// full history and a history card served from it verify. A profile
/// verifier reports the record `NOT_CHECKED`, never `FAIL`.
#[test]
fn a_ledger_the_producer_wrote_is_answered_and_verifies() {
    let (_dir, ledger, key, ids) = producer_ledger(5, &[2, 4]);
    let (stated, total) = anchors_of(&ledger, &key);
    assert_eq!(
        (stated.len(), total),
        (2, 2),
        "both checkpoints are stated anchors"
    );
    let (checked, wire) = ask_ledger(
        &ledger,
        &key,
        &json!({"subject": {"record": ids[3]}, "coverage": {"expected_pin": stated[1]}}),
    );
    assert_eq!(checked.unwrap().anchor, stated[1]);
    let report = cll_material::check_anchor_inclusion(
        &envelope_of(&wire),
        &ResponderLogKey(key.verifying_key()),
    );
    assert_eq!(
        report.status,
        cll_material::Status::NotChecked,
        "{}",
        report.detail
    );
    let full = json!({"subject": {"full_history": null}, "coverage": {"min_freshness": 1}});
    assert!(ask_ledger(&ledger, &key, &full).0.unwrap().records.len() >= 5);
    let card = json!({"subject": {"full_history": null}, "coverage": {"expected_pin": stated[1]}, "derivation": "history_card/1"});
    assert_eq!(ask_ledger(&ledger, &key, &card).0.unwrap().checkpoints, 2);
}

/// A checkpoint whose line lost its COSE statement ends the chain, as
/// any checkpoint this index cannot take does: what came before it is
/// still answered and verifies, and nothing after it is served, so every
/// list an answer carries extends checkpoint by checkpoint.
#[test]
fn a_checkpoint_without_its_statement_ends_the_chain() {
    let (_dir, ledger, key, _) = producer_ledger(6, &[1, 3, 5]);
    let path = ledger.join(crate::evidence_log::CHECKPOINTS_FILE);
    let lines: Vec<Value> = std::fs::read_to_string(&path)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(lines.len(), 3);
    let last = hex::encode(Sha256::digest(
        hex::decode(lines[2]["checkpoint_cose"].as_str().unwrap()).unwrap(),
    ));
    let mut stripped = lines.clone();
    stripped[1]
        .as_object_mut()
        .unwrap()
        .remove("checkpoint_cose");
    crate::evidence_request_parity::write_jsonl(&path, &stripped);
    let (anchors, total) = anchors_of(&ledger, &key);
    assert_eq!(total, 1, "the chain ends before the unstated checkpoint");

    let card = json!({"subject": {"full_history": null}, "coverage": {"expected_pin": anchors[0]}, "derivation": "history_card/1"});
    assert_eq!(ask_ledger(&ledger, &key, &card).0.unwrap().checkpoints, 1);
    let bytes = serde_json::to_vec(
        &json!({"subject": {"checkpoints": null}, "coverage": {"expected_pin": last}}),
    )
    .unwrap();
    let announced = announced_keys();
    let replay = crate::requester_auth::ReplayCache::default();
    let outcome = evidence_answer::answer_and_log(
        &bytes,
        false,
        &Responder {
            ledger_dir: &ledger,
            signing_key: &key,
            now: NOW,
            history_segments: "peers",
            share_standard_history: true,
            share_exchange_corroboration: false,
            keys: crate::requester_auth::Keys {
                announced: &announced,
                configured: None,
            },
            replay: &replay,
            node_id: Some(RESPONDER_NODE),
        },
        None,
    );
    assert!(
        matches!(
            outcome,
            Outcome::Refused {
                reason: Reason::CoverageUnsatisfiable,
                ..
            }
        ),
        "{outcome:?}"
    );
}
