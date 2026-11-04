//! What a watched turn's response says before anything is asked of the
//! peer: whether it carried a receipt this node can hold the peer to, and
//! whether the VRF drew the turn for an audit.
//!
//! A peer that advertises receipts owes one for every turn it serves, so a
//! turn without a usable one is a strike, drawn or not: were it no audit, a
//! peer would leave out the receipt of every turn it cheated on. Usable
//! means, in the order checked ([`sort`]):
//!
//! 1. the response carries a receipt;
//! 2. the receipt is signed by the identity key of the peer the request
//!    was sent to, which is the key a serving node signs with, and not by a
//!    key the receipt itself chooses;
//! 3. it decodes, is bound to the VRF key this request named, and its
//!    signature of the turn root verifies;
//! 4. it is for the request id this node sent.
//!
//! A receipt that names another model than the request asked for is a
//! strike whatever the peer advertises: the peer signed that it ran
//! something else. A receipt of another version than this node audits says
//! nothing, so an upgrade on either side blocks no honest peer.
//!
//! Then the VRF decides. A turn it does not draw is no audit, and so is a
//! drawn turn whose model this node cannot tie to a Hugging Face file.
//! Everything here runs on the request path, so it reads no file and waits
//! for nothing; the audit of a drawn turn goes to the worker.

use std::time::Instant;

use skippy_receipt_audit::claimed_file::HfFile;
use skippy_receipt_audit::turn_audit;
use skippy_receipts::turn;
use skippy_receipts::wire::{RECEIPT_VERSION, Receipt};

use super::collector::{Collected, Reply};
use super::drawn_turn::{AuditRunner, Job};
use super::enforcement::{self, Subject};
use super::peer_text::shown;
use super::requested_model::{self, Catalog, Claim, RequestedModel};
use super::verdict::Verdict;
use super::worker::Refused;
use super::{AUDIT_RATE, Asked, WatchedTurn};
use crate::network::peer_blocks::peer_key;

/// What the response of a watched turn came to.
pub(super) enum Sorted {
    /// Nothing more is asked of the peer: the verdict stands.
    Settled(Verdict),
    /// The VRF drew the turn: audit it against `file`.
    Drawn { receipt: Box<Receipt>, file: HfFile },
}

/// The id a receipt as it arrived says it has, for a log line.
fn stated_id(receipt: Option<&serde_json::Value>) -> &str {
    receipt
        .and_then(|receipt| receipt["receipt_id"].as_str())
        .unwrap_or("(none)")
}

/// A receipt the peer owes and did not give: a strike against a peer that
/// advertises receipts, and nothing about one that does not.
fn owed(asked: &Asked, defect: String) -> Sorted {
    Sorted::Settled(match asked.advertised {
        true => Verdict::strike(defect),
        false => Verdict::no_audit(format!(
            "the peer advertises no receipts for this model, and {defect}"
        )),
    })
}

/// The receipt as it arrived, decoded, when it is the peer's own and of
/// the version this node audits; else what it came to.
fn own_receipt(asked: &Asked, value: serde_json::Value) -> Result<Receipt, Sorted> {
    // the signer first: nothing else of a receipt is read before it is the
    // peer's
    let signer = value["binding"]["node_key"].as_str();
    let peer = peer_key(&asked.host);
    if let Some(signer) = signer.filter(|key| !key.eq_ignore_ascii_case(&peer)) {
        return Err(owed(
            asked,
            format!(
                "its receipt is signed by the key {}, which is not the peer's identity key",
                shown(signer)
            ),
        ));
    }
    if let Some(version) = value["version"]
        .as_u64()
        .filter(|v| *v != u64::from(RECEIPT_VERSION))
    {
        return Err(Sorted::Settled(Verdict::no_audit(format!(
            "the receipt is version {version}, and this node audits version {RECEIPT_VERSION}"
        ))));
    }
    let receipt: Receipt = serde_json::from_value(value).map_err(|error| {
        owed(
            asked,
            format!("its receipt does not decode: {}", shown(&error.to_string())),
        )
    })?;
    if receipt.binding.is_none() {
        return Err(owed(
            asked,
            "its receipt is not bound to this request's key and signed".into(),
        ));
    }
    Ok(receipt)
}

/// Sorts the response of a watched turn: `receipt` is what it carried,
/// `unread` whether part of it was too large to read. `rate` is the share
/// of turns the VRF draws.
pub(super) fn sort(
    asked: &Asked,
    catalog: Catalog<'_>,
    receipt: Option<serde_json::Value>,
    unread: bool,
    rate: f64,
) -> Sorted {
    let Some(value) = receipt else {
        let defect = match unread {
            true => "its response held no receipt this node could read: a part of it was too large",
            false => "its response carried no receipt",
        };
        return owed(asked, defect.into());
    };
    let receipt = match own_receipt(asked, value) {
        Ok(receipt) => receipt,
        Err(sorted) => return sorted,
    };
    if let Err(error) = turn::verify(&receipt, Some(&asked.key.public())) {
        let defect = format!(
            "its receipt does not verify: {}",
            shown(&format!("{error:#}"))
        );
        return owed(asked, defect);
    }
    if let Some(id) = asked.id.as_deref().filter(|id| *id != receipt.receipt_id) {
        let defect = format!(
            "its receipt is for the request {}, and this node asked under {id}",
            shown(&receipt.receipt_id)
        );
        return owed(asked, defect);
    }
    // what the request's model name stands for, by this node's own word
    let claim = RequestedModel::of(&asked.model, catalog)
        .map(|requested| requested.claimed(&receipt.model));
    if let Ok(Claim::Another(why)) = &claim {
        return Sorted::Settled(Verdict::strike(why));
    }
    if !turn_audit::audit_drawn(&receipt, &asked.key, rate) {
        return Sorted::Settled(Verdict::no_audit("the VRF did not draw the turn"));
    }
    match claim {
        Ok(Claim::Same(file)) => Sorted::Drawn {
            receipt: Box::new(receipt),
            file,
        },
        Ok(Claim::Unnamed(why) | Claim::Another(why)) => Sorted::Settled(Verdict::no_audit(why)),
        Err(why) => Sorted::Settled(Verdict::no_audit(why)),
    }
}

/// A verdict being acted on off the request path, for whoever waits for it.
pub(super) type Settling = Option<tokio::task::JoinHandle<()>>;

/// Takes the response of a watched turn from the collector: settles it, or
/// queues its audit. Called on the request path.
pub(super) fn receive(turn: WatchedTurn, collected: Collected) -> Settling {
    let received = Instant::now();
    let host = turn.asked.host;
    let mut reply = match collected {
        Collected::Reply(reply) => reply,
        Collected::NotATurn(why) => {
            let subject = Subject::of(host, "(none)", &turn.asked.model);
            return settle(turn, subject, Verdict::no_audit(why));
        }
    };
    let carried = reply.receipt.take();
    let id = stated_id(carried.as_ref()).to_owned();
    match sort(
        &turn.asked,
        &requested_model::node_catalog,
        carried,
        reply.unread,
        AUDIT_RATE,
    ) {
        Sorted::Settled(verdict) => {
            let subject = Subject::of(host, &id, &turn.asked.model);
            settle(turn, subject, verdict)
        }
        Sorted::Drawn { receipt, file } => queue(turn, reply, *receipt, file, received),
    }
}

fn queue(
    turn: WatchedTurn,
    reply: Reply,
    receipt: Receipt,
    file: HfFile,
    received: Instant,
) -> Settling {
    let Ok(runtime) = tokio::runtime::Handle::try_current() else {
        let subject = Subject::of(turn.asked.host, &receipt.receipt_id, &receipt.model);
        let verdict = Verdict::no_audit("the response ended outside the node's runtime");
        return settle(turn, subject, verdict);
    };
    let audits = turn.node.receipt_audits.clone();
    let job = Job {
        turn,
        reply,
        receipt,
        file,
        received,
        runtime,
    };
    let (job, why) = match audits.worker.submit(job, AuditRunner::new) {
        Ok(()) => return None,
        Err(Refused::Busy(job)) => (job, "the audit worker is busy"),
        Err(Refused::ShutDown(job)) => (job, "this node is shutting down"),
        Err(Refused::Disabled(job)) => (job, "this node's audit worker is off"),
    };
    let subject = job.subject();
    settle(job.turn, subject, Verdict::no_audit(why))
}

/// Acts on a verdict reached on the request path. A strike may block the
/// peer, which writes the block store, so it runs on a blocking thread;
/// the other verdicts only log.
fn settle(turn: WatchedTurn, subject: Subject, verdict: Verdict) -> Settling {
    let runtime = tokio::runtime::Handle::try_current().ok();
    match (&verdict, runtime) {
        (Verdict::Strike(_) | Verdict::Failed { .. }, Some(runtime)) => {
            let handle = runtime.clone();
            Some(runtime.spawn_blocking(move || {
                enforcement::conclude_for(&turn.node, &subject, verdict, Some(&handle));
            }))
        }
        (_, runtime) => {
            enforcement::conclude_for(&turn.node, &subject, verdict, runtime.as_ref());
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use ed25519_dalek::SigningKey;
    use skippy_receipts::toy_turn::{RUNTIME, Turn};
    use skippy_receipts::turn::BindingRequest;
    use skippy_receipts::vrf;

    use super::*;
    use crate::network::receipt_audit::test_support::{
        ASKED_ID, COMMIT, TOY_MODEL, asked, client_key, no_catalog, peer, toy_turn, toy_turn_with,
    };

    /// Every turn is drawn, or none is.
    const ALWAYS: f64 = 1.0;
    const NEVER: f64 = 0.0;

    fn carried(turn: &Turn) -> Option<serde_json::Value> {
        Some(serde_json::to_value(&turn.receipt).unwrap())
    }

    fn strike(sorted: Sorted) -> String {
        match sorted {
            Sorted::Settled(Verdict::Strike(why)) => why.to_string(),
            Sorted::Settled(other) => panic!("not a strike: {other:?}"),
            Sorted::Drawn { .. } => panic!("not a strike: the turn goes to an audit"),
        }
    }

    fn no_audit(sorted: Sorted) -> String {
        match sorted {
            Sorted::Settled(Verdict::NoAudit(why)) => why.to_string(),
            Sorted::Settled(other) => panic!("not no audit: {other:?}"),
            Sorted::Drawn { .. } => panic!("not no audit: the turn goes to an audit"),
        }
    }

    #[test]
    fn a_usable_receipt_is_audited_when_the_vrf_draws_it() {
        let asked = asked();
        let turn = toy_turn(&asked.key);
        let Sorted::Drawn { receipt, file } =
            sort(&asked, &no_catalog, carried(&turn), false, ALWAYS)
        else {
            panic!("a drawn turn with a usable receipt is audited");
        };
        assert_eq!(*receipt, turn.receipt);
        assert_eq!(file.to_string(), TOY_MODEL);
        // the same receipt, not drawn, says nothing and asks nothing
        let why = no_audit(sort(&asked, &no_catalog, carried(&turn), false, NEVER));
        assert_eq!(why, "the VRF did not draw the turn");
    }

    /// The receipts a peer that advertises them may not answer with, each
    /// a strike whether the VRF would draw the turn or not.
    #[test]
    fn a_peer_that_advertises_receipts_owes_a_usable_one() {
        let key = client_key();
        let other_key = vrf::SecretKey::from_seed([3; 32]);
        let unbound = {
            let mut turn = Turn::new();
            turn.receipt.model = TOY_MODEL.to_owned();
            turn.receipt.receipt_id = ASKED_ID.to_owned();
            carried(&turn)
        };
        let undecodable = {
            let mut value = carried(&toy_turn(&key)).unwrap();
            value["tokens"] = "not a list of tokens".into();
            Some(value)
        };
        let missigned = {
            // a token changed after the node signed the turn root
            let mut turn = toy_turn(&key);
            turn.receipt.tokens[0] += 1;
            carried(&turn)
        };
        let bad_signature = {
            let mut turn = toy_turn(&key);
            let binding = turn.receipt.binding.as_mut().unwrap();
            binding.signature = "00".repeat(64);
            carried(&turn)
        };
        let wrong_id = carried(&toy_turn_with(&key, |r| r.receipt_id = "another-id".into()));
        let cases: Vec<(&str, Option<serde_json::Value>, bool, &str)> = vec![
            ("no receipt", None, false, "carried no receipt"),
            ("a receipt it could not read", None, true, "too large"),
            (
                "a receipt that is no receipt",
                Some("r".into()),
                false,
                "does not decode",
            ),
            (
                "an undecodable receipt",
                undecodable,
                false,
                "does not decode",
            ),
            ("an unbound receipt", unbound, false, "not bound"),
            (
                "a receipt bound to another key",
                carried(&toy_turn(&other_key)),
                false,
                "another client key",
            ),
            (
                "a changed receipt",
                missigned,
                false,
                "turn root does not match",
            ),
            ("a bad signature", bad_signature, false, "does not verify"),
            (
                "a receipt of another request",
                wrong_id,
                false,
                "asked under",
            ),
        ];
        for (what, receipt, unread, reason) in cases {
            for rate in [ALWAYS, NEVER] {
                let why = strike(sort(&asked(), &no_catalog, receipt.clone(), unread, rate));
                assert!(why.contains(reason), "{what}: {why}");
            }
            // from a peer that advertises none, the same says nothing
            let quiet = Asked {
                advertised: false,
                ..asked()
            };
            let why = no_audit(sort(&quiet, &no_catalog, receipt, unread, ALWAYS));
            assert!(why.contains("advertises no receipts"), "{what}: {why}");
            assert!(why.contains(reason), "{what}: {why}");
        }
    }

    #[test]
    fn the_receipt_must_be_signed_by_the_peer_the_request_was_sent_to() {
        let key = client_key();
        // the toy node's honest receipt, relayed by another peer
        let relayed = Asked {
            host: peer(77),
            ..asked()
        };
        let why = strike(sort(
            &relayed,
            &no_catalog,
            carried(&toy_turn(&key)),
            false,
            ALWAYS,
        ));
        assert!(why.contains("not the peer's identity key"), "{why}");
        // a receipt the peer signed with a key of its own choosing: the
        // signature verifies under the key the receipt names
        let mut forged = Turn::new();
        forged.receipt.model = TOY_MODEL.to_owned();
        forged.receipt.receipt_id = ASKED_ID.to_owned();
        let request = BindingRequest {
            client_key: key.public(),
            parent: None,
            runtime: RUNTIME.to_owned(),
            sampler: turn::GREEDY.to_owned(),
        };
        let own_key = SigningKey::from_bytes(&[42; 32]);
        turn::bind(&mut forged.receipt, &request, &own_key);
        assert!(turn::verify(&forged.receipt, Some(&key.public())).is_ok());
        for rate in [ALWAYS, NEVER] {
            let why = strike(sort(&asked(), &no_catalog, carried(&forged), false, rate));
            assert!(why.contains("not the peer's identity key"), "{why}");
        }
        // the peer's key in upper case is still the peer's key
        let mut upper = serde_json::to_value(&toy_turn(&key).receipt).unwrap();
        let named = upper["binding"]["node_key"]
            .as_str()
            .unwrap()
            .to_uppercase();
        upper["binding"]["node_key"] = named.into();
        let why = strike(sort(&asked(), &no_catalog, Some(upper), false, ALWAYS));
        assert!(!why.contains("identity key"), "{why}");
    }

    #[test]
    fn a_receipt_that_names_another_model_is_a_strike() {
        let key = client_key();
        let small = format!("toy/tiny@{COMMIT}/tiny.gguf");
        let other = toy_turn_with(&key, |r| r.model = small.clone());
        for advertised in [true, false] {
            for rate in [ALWAYS, NEVER] {
                let asked = Asked {
                    advertised,
                    ..asked()
                };
                let why = strike(sort(&asked, &no_catalog, carried(&other), false, rate));
                assert!(why.contains("toy/tiny@"), "{why}");
                assert!(why.contains("the request asked for toy/model@"), "{why}");
            }
        }
    }

    #[test]
    fn a_model_this_node_cannot_identify_is_never_opened() {
        let key = client_key();
        // the request named a runtime's own name: only the peer says what
        // file that is, whatever its receipt names
        let unknown = Asked {
            model: "Qwen3-32B-Q4_K_M".into(),
            ..asked()
        };
        for model in [TOY_MODEL.to_owned(), format!("evil/repo@{COMMIT}/x.gguf")] {
            let turn = toy_turn_with(&key, |r| r.model = model.clone());
            let why = no_audit(sort(&unknown, &no_catalog, carried(&turn), false, ALWAYS));
            assert!(why.contains("cannot tie the model name"), "{model}: {why}");
        }
        // the requested model, named as no file on Hugging Face
        for model in ["toy-model", "/models/toy.gguf", "toy/model@main/toy.gguf"] {
            let turn = toy_turn_with(&key, |r| r.model = model.to_owned());
            let why = no_audit(sort(&asked(), &no_catalog, carried(&turn), false, ALWAYS));
            assert!(why.contains("the receipt names"), "{model}: {why}");
        }
    }

    #[test]
    fn a_name_this_nodes_catalog_lists_is_audited_against_the_catalogs_file() {
        use crate::network::receipt_audit::requested_model::CatalogFile;
        let key = client_key();
        let catalog = |name: &str| {
            (name == "Toy-Q8_0").then(|| CatalogFile {
                repo: "toy/model".into(),
                revision: None,
                file: "toy.gguf".into(),
            })
        };
        let listed = Asked {
            model: "Toy-Q8_0".into(),
            ..asked()
        };
        let honest = toy_turn(&key);
        let sorted = sort(&listed, &catalog, carried(&honest), false, ALWAYS);
        let Sorted::Drawn { file, .. } = sorted else {
            panic!("a catalog name is not audited");
        };
        assert_eq!(file.to_string(), TOY_MODEL);
        // the peer may not say the name stands for another file
        let other = toy_turn_with(&key, |r| r.model = format!("toy/model@{COMMIT}/tiny.gguf"));
        let why = strike(sort(&listed, &catalog, carried(&other), false, ALWAYS));
        assert!(
            why.contains("the request asked for toy/model/toy.gguf"),
            "{why}"
        );
    }

    /// From the collector's hand-off to the node's block store.
    #[tokio::test]
    async fn three_turns_without_a_receipt_block_a_peer_on_the_node() {
        use crate::network::peer_blocks::now_ms;
        let node = crate::mesh::Node::new_for_tests(crate::mesh::NodeRole::Client)
            .await
            .expect("a node");
        let watched = |asked: Asked| WatchedTurn {
            node: node.clone(),
            asked,
        };
        let host = asked().host;
        // what says nothing about the peer is logged, and that is all
        let quiet = || Asked {
            advertised: false,
            ..asked()
        };
        let no_receipt = || Collected::Reply(Reply::default());
        for _ in 0..4 {
            assert!(receive(watched(quiet()), no_receipt()).is_none());
            let broke_off = Collected::NotATurn("the response did not end whole".into());
            assert!(receive(watched(asked()), broke_off).is_none());
        }
        assert!(!node.peer_blocks.is_blocked(&host, now_ms()));
        // a peer that advertises receipts and sends none is struck
        for strikes in 0..3 {
            assert!(
                !node.peer_blocks.is_blocked(&host, now_ms()),
                "blocked after {strikes} strikes"
            );
            let settling = receive(watched(asked()), no_receipt());
            settling
                .expect("a strike is acted on off the request path")
                .await
                .expect("the strike was counted");
        }
        assert!(node.peer_blocks.is_blocked(&host, now_ms()));
        assert!(!node.peer_blocks.is_blocked(&peer(77), now_ms()));
        node.receipt_audits.shut_down().await;
        node.close_endpoint().await;
    }

    #[test]
    fn a_receipt_of_another_version_says_nothing() {
        let key = client_key();
        let mut older = serde_json::to_value(&toy_turn(&key).receipt).unwrap();
        older["version"] = 1.into();
        // fields this version does not have do not make it a strike
        older.as_object_mut().unwrap().remove("tokens");
        let why = no_audit(sort(&asked(), &no_catalog, Some(older), false, ALWAYS));
        assert!(why.contains("the receipt is version 1"), "{why}");
    }

    #[test]
    fn a_reason_holds_no_raw_peer_text() {
        let key = client_key();
        let forged = format!("r\nWARN audit passed\x1b[2K{}", "z".repeat(1 << 16));
        let sorted = [
            sort(
                &asked(),
                &no_catalog,
                carried(&toy_turn_with(&key, |r| r.receipt_id = forged.clone())),
                false,
                ALWAYS,
            ),
            sort(
                &asked(),
                &no_catalog,
                carried(&toy_turn_with(&key, |r| r.model = forged.clone())),
                false,
                ALWAYS,
            ),
            sort(
                &asked(),
                &no_catalog,
                Some(serde_json::json!({"binding": {"node_key": forged}})),
                false,
                ALWAYS,
            ),
            sort(
                &asked(),
                &no_catalog,
                Some(serde_json::json!({"tokens": forged})),
                false,
                ALWAYS,
            ),
        ];
        for sorted in sorted {
            let Sorted::Settled(Verdict::Strike(why) | Verdict::NoAudit(why)) = sorted else {
                panic!("a forged receipt is audited");
            };
            let why = why.to_string();
            assert!(!why.contains('\n') && !why.contains('\x1b'), "{why}");
            assert!(why.chars().count() <= 241, "{}", why.chars().count());
        }
    }
}
