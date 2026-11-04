//! A toy peer, its turn and its claimed model, for the audit's tests: the
//! toy model of `skippy_receipts::toy_turn`, served by a node whose key is
//! the peer's identity, and asked for as this node asks a real peer.

use std::cell::{Cell, RefCell};
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use iroh::EndpointId;
use skippy_receipt_audit::claimed_file::HfFile;
use skippy_receipt_audit::claimed_model::{CacheTypes, ClaimedModel};
use skippy_receipt_audit::turn_audit::ClaimedSetup;
use skippy_receipts::glue::GlueEvaluator;
use skippy_receipts::sampler_request::RequestedSampling;
use skippy_receipts::toy_turn::{self, LAYERS, RUNTIME, ToyModel, Turn};
use skippy_receipts::wire::{Challenge, Opening};
use skippy_receipts::{turn, vrf};

use super::Asked;
use super::collector::Reply;
use super::http_answer::Answer;
use super::opener::Transport;
use super::requested_model::CatalogFile;

pub(super) const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";
/// The model the toy turn's request names, and its receipt.
pub(super) const TOY_MODEL: &str = "toy/model@0123456789abcdef0123456789abcdef01234567/toy.gguf";
/// The receipt id the toy turn's request asks the peer to issue its receipt
/// under (`x-receipt-id`).
pub(super) const ASKED_ID: &str = "0b0e7c1e-58a5-4b3f-9c36-0d2f6f1f8a10";
/// The toy turn's request's own id (`x-request-id`).
pub(super) const REQUEST_ID: &str = "5e1d2f9a-7c34-4b8e-a0f1-3c2d4e5f6a7b";
/// The toy turn's prompt, as the toy model writes tokens.
pub(super) const PROMPT: &str = "<1><2><3>";

/// A peer's identity.
pub(super) fn peer(seed: u8) -> EndpointId {
    EndpointId::from(iroh::SecretKey::from_bytes(&[seed; 32]).public())
}

/// The peer whose identity key signs the toy turn.
pub(super) fn toy_peer() -> EndpointId {
    peer(5)
}

pub(super) fn hf_file(name: &str) -> HfFile {
    HfFile::parse(&format!("o/m@{COMMIT}/{name}.gguf")).expect("a file name")
}

pub(super) fn no_catalog(_: &str) -> Option<CatalogFile> {
    None
}

/// The toy model as the audit loads it. Its token `n` is the text `<n>`.
pub(super) struct ToyClaimed {
    /// No layer loads, as when the native runtime cannot open the GGUF.
    pub(super) unloadable: bool,
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

/// The toy model, claimed, with its head key under `state`.
pub(super) fn toy_setup(state: &Path) -> ClaimedSetup {
    ClaimedSetup {
        weights: toy_turn::weights(),
        model: Box::new(ToyClaimed { unloadable: false }),
        layer_count: LAYERS,
        head_key: state.join("head-key.json"),
    }
}

/// The toy turn as the toy peer serves it for a request that names `key`:
/// greedy, under [`ASKED_ID`], of [`TOY_MODEL`].
pub(super) fn toy_turn(key: &vrf::SecretKey) -> Turn {
    toy_turn_with(key, |_| {})
}

/// [`toy_turn`], with what `change` does to the receipt before the peer
/// signs it.
pub(super) fn toy_turn_with(
    key: &vrf::SecretKey,
    change: impl FnOnce(&mut skippy_receipts::wire::Receipt),
) -> Turn {
    let mut turn = Turn::new();
    turn.receipt.model = TOY_MODEL.to_owned();
    turn.receipt.receipt_id = ASKED_ID.to_owned();
    change(&mut turn.receipt);
    turn.bound(key.public(), turn::GREEDY)
}

pub(super) fn client_key() -> vrf::SecretKey {
    vrf::SecretKey::from_seed([9; 32])
}

/// What this node asked the toy peer for: a plain greedy request for
/// [`TOY_MODEL`], from a peer that advertises receipts.
pub(super) fn asked() -> Asked {
    Asked {
        host: toy_peer(),
        key: client_key(),
        model: TOY_MODEL.to_owned(),
        id: ASKED_ID.to_owned(),
        request_id: Some(REQUEST_ID.to_owned()),
        prompt: Some(PROMPT.to_owned()),
        sampling: RequestedSampling::greedy(),
        text_unavailable: None,
        shaped_answer: None,
        advertised: true,
        tier: false,
        rate: 0.1,
    }
}

/// The answer the toy peer gave with `turn`: the text of its sampled
/// tokens.
pub(super) fn reply(turn: &Turn) -> Reply {
    let model = ToyClaimed { unloadable: false };
    Reply {
        content: model
            .detokenize(&turn.receipt.sampled_tokens())
            .expect("text"),
        ..Reply::default()
    }
}

/// What a node changes in the opening it committed to.
pub(super) type Change = Box<dyn Fn(&mut Opening)>;

/// What the toy peer does with a challenge.
pub(super) enum Answers {
    Honest,
    /// Nothing comes back within the wait.
    Silent,
    /// An error status, with this body.
    Status(u16, Vec<u8>),
    /// The opening comes without a signature.
    Unsigned,
    /// The signature is of other bytes.
    Missigned,
    /// A signed body that is no opening.
    Undecodable,
    /// More bytes than the challenge can produce.
    Oversized,
    /// The node opens this instead of what it committed, and signs it.
    Changed(Change),
}

/// The toy peer on the other end of the tunnel, and the clock of the test:
/// an answer takes `takes`, which moves the clock.
pub(super) struct ToyPeer {
    pub(super) turn: Turn,
    pub(super) answers: Answers,
    pub(super) takes: Duration,
    pub(super) now: Cell<Instant>,
    /// Every request: its action, the wait it was given and the largest
    /// body it was allowed.
    pub(super) asked: RefCell<Vec<(&'static str, Duration, usize)>>,
}

impl ToyPeer {
    pub(super) fn new(turn: Turn, answers: Answers) -> Self {
        Self {
            turn,
            answers,
            takes: Duration::from_millis(250),
            now: Cell::new(Instant::now()),
            asked: RefCell::default(),
        }
    }

    pub(super) fn pass(&self, time: Duration) {
        self.now.set(self.now.get() + time);
    }
}

impl Transport for ToyPeer {
    fn post(
        &self,
        action: &'static str,
        body: &[u8],
        wait: Duration,
        max_body: usize,
    ) -> Result<Answer> {
        self.asked.borrow_mut().push((action, wait, max_body));
        if matches!(self.answers, Answers::Silent) || self.takes > wait {
            self.pass(wait);
            bail!("no whole answer to /v1/receipts/{action} within {wait:?}");
        }
        self.pass(self.takes);
        if action != "open" {
            bail!("the toy model has no attention rows");
        }
        let challenge: Challenge = serde_json::from_slice(body)?;
        let mut opening = self.turn.open(&challenge);
        if let Answers::Changed(change) = &self.answers {
            change(&mut opening);
        }
        let mut bytes = serde_json::to_vec(&opening)?;
        let (status, signature) = match &self.answers {
            Answers::Status(status, refusal) => {
                bytes.clone_from(refusal);
                (*status, None)
            }
            Answers::Unsigned => (200, None),
            Answers::Missigned => (
                200,
                Some(self.turn.sign_opening(&challenge, b"another opening")),
            ),
            Answers::Undecodable => {
                bytes = b"{\"stages\": 7}".to_vec();
                (200, Some(self.turn.sign_opening(&challenge, &bytes)))
            }
            Answers::Oversized => {
                // the tunnel reads no more than it was allowed
                assert!(bytes.len() <= max_body);
                bail!("the answer is larger than the {max_body} bytes this node reads of it");
            }
            _ => (200, Some(self.turn.sign_opening(&challenge, &bytes))),
        };
        Ok(Answer {
            status,
            signature,
            retry_after: None,
            body: bytes,
        })
    }

    fn pause(&self, wait: Duration) {
        self.pass(wait);
    }
}
