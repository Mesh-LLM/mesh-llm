//! The signed turn root: one hash over everything a receipt claims, signed
//! by the serving node, and the VRF inputs a client draws from it.
//!
//! The root binds the client's VRF key, so the client is bound to its key
//! before it learns the root, the turn that came before (`parent`), so
//! a conversation's receipts form a chain, and how the node chose its
//! tokens (`sampler`), so a node that says it decoded greedily and did not
//! signs a false statement. A client checks the signature
//! and the root, then derives its challenge from the VRF of the root
//! ([`challenge_input`]) and the rows it checks from the VRF of the
//! opening's bytes ([`rows_input`]).

use anyhow::{Context, Result, bail};
use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use serde::{Deserialize, Serialize};

use crate::attention::opening::BlockChallenge;
use crate::commitment::{Hash, hex_array};
use crate::wire::{Challenge, Receipt};

const TURN_DOMAIN: &str = "mesh-llm receipts turn v1";
const OPENING_DOMAIN: &str = "mesh-llm receipts opening v2";
const ROWS_DOMAIN: &str = "mesh-llm receipts attention rows v2";

/// Response header of `POST /receipts/open` that holds the node's hex
/// signature of the opening's bytes ([`sign_opening`]).
pub const OPENING_SIGNATURE_HEADER: &str = "x-receipt-opening-signature";
const CHALLENGE_PREFIX: &[u8] = b"mesh-llm/receipts/challenge";
const ROWS_PREFIX: &[u8] = b"mesh-llm/receipts/rows";

/// How a receipt names its runtime: the Skippy ABI version and the source
/// the native library was built from ([`skippy_runtime`'s
/// `receipt_runtime_source`]). A verifier whose glue evaluation runs
/// another runtime may not reproduce the node's rounding.
pub fn runtime_identity(major: u32, minor: u32, patch: u32, source: &str) -> String {
    format!("skippy-abi {major}.{minor}.{patch}; {source}")
}

/// The sampler of a turn that took the best logit at every token.
pub const GREEDY: &str = "greedy";
/// The sampler of any other turn: temperature, penalties or logit biases.
/// The statement is `sampled` followed by a space and a JSON object of the
/// sampler's parameters and seed ([`sampled`]).
pub(crate) const SAMPLED: &str = "sampled";
/// The `draw` a sampled turn's statement names when a client can replay it:
/// the token at position `i` of the sequence is picked by a uniform that is
/// a function of the stated seed and of `i`, over the candidates the sampler
/// chain kept, in token id order (`head::draw` on the client, the
/// `receipts_draw` sampler on the node).
pub const INDEXED_DRAW: &str = "indexed-v1";
/// The field of a sampled turn's statement that names what the node built
/// the turn's reasoning-budget sampler from, so a client can replay which
/// tokens the sampler forced (`head::reasoning_budget`): the tag that starts
/// the reasoning (`start`), the tags that end it (`ends`), and the text the
/// chat template placed after the prompt (`prefill`). A turn without such
/// a sampler states none.
pub const REASONING: &str = "reasoning";

/// The sampler statement of a turn that sampled with `parameters`.
pub fn sampled(parameters: &serde_json::Value) -> String {
    format!("{SAMPLED} {parameters}")
}

/// The sampler parameters a statement names, `None` for a greedy turn or a
/// statement that names none.
pub fn sampler_parameters(sampler: &str) -> Option<serde_json::Value> {
    let parameters = sampler.strip_prefix(SAMPLED)?.strip_prefix(' ')?;
    serde_json::from_str(parameters).ok()
}

/// What binds a receipt to its client and its node.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TurnBinding {
    /// Hex VRF key the client named in its request.
    pub client_key: String,
    /// Hex turn root of the turn this one continues, as the client named it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
    /// The runtime that served the turn: Skippy ABI and llama.cpp patch
    /// queue. An auditor whose glue evaluation does not match it cannot
    /// check the glue.
    pub runtime: String,
    /// How the node chose its tokens: [`GREEDY`], or [`SAMPLED`] with the
    /// sampler's parameters and seed.
    pub sampler: String,
    /// Hex Ed25519 key of the serving node.
    pub node_key: String,
    /// Hex [`turn_root`] of the receipt.
    pub turn_root: String,
    /// Hex Ed25519 signature of the turn root under `node_key`.
    pub signature: String,
}

/// What the node binds into a receipt besides its commitments.
#[derive(Clone, Debug)]
pub struct BindingRequest {
    pub client_key: [u8; 32],
    pub parent: Option<Hash>,
    pub runtime: String,
    pub sampler: String,
}

/// What the binding adds to a receipt's fields in its root.
pub struct RootFields<'a> {
    pub client_key: &'a str,
    pub parent: Option<&'a str>,
    pub runtime: &'a str,
    pub sampler: &'a str,
    pub node_key: &'a str,
}

/// The hash over every field of `receipt` but the root and signature.
pub fn turn_root(receipt: &Receipt, fields: &RootFields<'_>) -> Hash {
    let mut h = blake3::Hasher::new_derive_key(TURN_DOMAIN);
    let text = |h: &mut blake3::Hasher, s: &str| {
        h.update(&(s.len() as u64).to_le_bytes());
        h.update(s.as_bytes());
    };
    h.update(&receipt.version.to_le_bytes());
    text(&mut h, &receipt.receipt_id);
    text(&mut h, &receipt.model);
    h.update(&receipt.position_start.to_le_bytes());
    h.update(&receipt.position_end.to_le_bytes());
    h.update(&receipt.generation_start.to_le_bytes());
    match receipt.final_token {
        Some(token) => h.update(&[1]).update(&token.to_le_bytes()),
        None => h.update(&[0]),
    };
    h.update(&(receipt.tokens.len() as u64).to_le_bytes());
    for token in &receipt.tokens {
        h.update(&token.to_le_bytes());
    }
    h.update(&(receipt.stages.len() as u64).to_le_bytes());
    for stage in &receipt.stages {
        h.update(&stage.stage_index.to_le_bytes());
        h.update(&stage.layer_start.to_le_bytes());
        h.update(&stage.layer_end.to_le_bytes());
        text(&mut h, &stage.root);
        h.update(&stage.leaf_count.to_le_bytes());
        text(&mut h, &stage.backend);
        h.update(&(stage.cache_types.len() as u64).to_le_bytes());
        for cache_type in &stage.cache_types {
            text(&mut h, cache_type);
        }
        h.update(&(stage.ubatches.len() as u64).to_le_bytes());
        for start in &stage.ubatches {
            h.update(&start.to_le_bytes());
        }
        match &stage.prefix {
            Some(prefix) => {
                h.update(&[1]);
                text(&mut h, &prefix.root);
                h.update(&prefix.positions.to_le_bytes());
                match &prefix.tokens {
                    Some(tokens) => {
                        h.update(&[1]);
                        text(&mut h, tokens);
                    }
                    None => {
                        h.update(&[0]);
                    }
                }
            }
            None => {
                h.update(&[0]);
            }
        }
        match &stage.recurrent_state {
            Some(state) => {
                h.update(&[1]);
                text(&mut h, state);
            }
            None => {
                h.update(&[0]);
            }
        }
    }
    text(&mut h, fields.client_key);
    match fields.parent {
        Some(parent) => {
            h.update(&[1]);
            text(&mut h, parent);
        }
        None => {
            h.update(&[0]);
        }
    }
    text(&mut h, fields.runtime);
    text(&mut h, fields.sampler);
    text(&mut h, fields.node_key);
    *h.finalize().as_bytes()
}

/// Binds `receipt` to the request and signs it with the node key.
pub fn bind(receipt: &mut Receipt, request: &BindingRequest, node: &SigningKey) {
    let client_key = hex::encode(request.client_key);
    let parent = request.parent.map(hex::encode);
    let node_key = hex::encode(node.verifying_key().as_bytes());
    let root = turn_root(
        receipt,
        &RootFields {
            client_key: &client_key,
            parent: parent.as_deref(),
            runtime: &request.runtime,
            sampler: &request.sampler,
            node_key: &node_key,
        },
    );
    let signature = node.sign(&root);
    receipt.binding = Some(TurnBinding {
        client_key,
        parent,
        runtime: request.runtime.clone(),
        sampler: request.sampler.clone(),
        node_key,
        turn_root: hex::encode(root),
        signature: hex::encode(signature.to_bytes()),
    });
}

/// A receipt's checked binding.
#[derive(Clone, Debug)]
pub struct VerifiedBinding {
    pub turn_root: Hash,
    pub client_key: [u8; 32],
    pub node_key: [u8; 32],
}

/// Checks that `receipt` is bound and signed, and recomputes its root.
/// `client_key` is the key the client named, when it checks its own
/// receipt.
pub fn verify(receipt: &Receipt, client_key: Option<&[u8; 32]>) -> Result<VerifiedBinding> {
    let binding = receipt
        .binding
        .as_ref()
        .context("the receipt is not bound to a client key and signed")?;
    let named =
        crate::vrf::key_from_hex(&binding.client_key).context("the receipt's client key")?;
    if let Some(expected) = client_key
        && expected != &named
    {
        bail!("the receipt names another client key");
    }
    let root = turn_root(
        receipt,
        &RootFields {
            client_key: &binding.client_key,
            parent: binding.parent.as_deref(),
            runtime: &binding.runtime,
            sampler: &binding.sampler,
            node_key: &binding.node_key,
        },
    );
    if hex::encode(root) != binding.turn_root {
        bail!("the receipt's turn root does not match its fields");
    }
    let node_key = hex_array(&binding.node_key, "the receipt's node key")?;
    verify_signed(&node_key, &root, &binding.signature)
        .context("the node's signature of the turn root does not verify")?;
    Ok(VerifiedBinding {
        turn_root: root,
        client_key: named,
        node_key,
    })
}

/// What a node signs for an opening: its turn root, the challenge it
/// answers, and the opening's bytes as sent. The opened matmul columns come
/// from a replay and are in no Merkle tree, so without the signature a
/// client could show an opening the node never sent. Without the challenge
/// under the signature, a client could show an honest opening as the
/// answer to a challenge the node was never sent.
fn opening_digest(turn_root: &Hash, challenge: &Challenge, opening_bytes: &[u8]) -> Hash {
    signed_digest(
        OPENING_DOMAIN,
        turn_root,
        &challenge.digest(),
        opening_bytes,
    )
}

fn signed_digest(domain: &str, turn_root: &Hash, request: &Hash, answer_bytes: &[u8]) -> Hash {
    let mut h = blake3::Hasher::new_derive_key(domain);
    h.update(turn_root);
    h.update(request);
    h.update(blake3::hash(answer_bytes).as_bytes());
    *h.finalize().as_bytes()
}

/// The node's hex signature of its opening of `challenge` for the turn
/// `turn_root`.
pub fn sign_opening(
    node: &SigningKey,
    turn_root: &Hash,
    challenge: &Challenge,
    opening_bytes: &[u8],
) -> String {
    hex::encode(
        node.sign(&opening_digest(turn_root, challenge, opening_bytes))
            .to_bytes(),
    )
}

/// Checks the node's signature of its opening of `challenge` for a bound
/// receipt.
pub fn verify_opening(
    binding: &VerifiedBinding,
    challenge: &Challenge,
    opening_bytes: &[u8],
    signature: &str,
) -> Result<()> {
    verify_signed(
        &binding.node_key,
        &opening_digest(&binding.turn_root, challenge, opening_bytes),
        signature,
    )
    .context("the node's signature of the opening does not verify")
}

/// The node's hex signature of the attention rows it opened for `blocks`
/// of the turn `turn_root`, which a client keeps as evidence like an
/// opening.
pub fn sign_rows(
    node: &SigningKey,
    turn_root: &Hash,
    blocks: &BlockChallenge,
    rows_bytes: &[u8],
) -> String {
    hex::encode(
        node.sign(&signed_digest(
            ROWS_DOMAIN,
            turn_root,
            &blocks.digest(),
            rows_bytes,
        ))
        .to_bytes(),
    )
}

/// Checks the node's signature of the attention rows it opened for
/// `blocks` of a bound receipt.
pub fn verify_rows(
    binding: &VerifiedBinding,
    blocks: &BlockChallenge,
    rows_bytes: &[u8],
    signature: &str,
) -> Result<()> {
    let digest = signed_digest(
        ROWS_DOMAIN,
        &binding.turn_root,
        &blocks.digest(),
        rows_bytes,
    );
    verify_signed(&binding.node_key, &digest, signature)
        .context("the node's signature of the attention rows does not verify")
}

const CANNOT_OPEN_DOMAIN: &str = "mesh-llm receipts cannot open v1";

/// The node's hex signature of its answer that it can no longer open
/// `request`, the digest of the challenge or block challenge it was sent
/// ([`Challenge::digest`], [`BlockChallenge::digest`]), for the turn
/// `turn_root`. `answer_bytes` are the bytes of the
/// [`crate::wire::CannotOpen`] as sent. A client keeps it to show that the
/// node itself said it could not open the turn.
pub fn sign_cannot_open(
    node: &SigningKey,
    turn_root: &Hash,
    request: &Hash,
    answer_bytes: &[u8],
) -> String {
    hex::encode(
        node.sign(&signed_digest(
            CANNOT_OPEN_DOMAIN,
            turn_root,
            request,
            answer_bytes,
        ))
        .to_bytes(),
    )
}

/// Checks the node's signature of a "cannot open" answer to `request` for a
/// bound receipt.
pub fn verify_cannot_open(
    binding: &VerifiedBinding,
    request: &Hash,
    answer_bytes: &[u8],
    signature: &str,
) -> Result<()> {
    let digest = signed_digest(
        CANNOT_OPEN_DOMAIN,
        &binding.turn_root,
        request,
        answer_bytes,
    );
    verify_signed(&binding.node_key, &digest, signature)
        .context("the node's signature of its cannot-open answer does not verify")
}

fn verify_signed(node_key: &[u8; 32], digest: &Hash, signature: &str) -> Result<()> {
    let verifying = VerifyingKey::from_bytes(node_key).context("the node key")?;
    let bytes = hex_array(signature, "a signature")?;
    Ok(verifying.verify_strict(digest, &Signature::from_bytes(&bytes))?)
}

/// Checks that `next` continues the conversation of `previous`, both bound
/// and verified: `next` names the root of `previous` as its parent, for the
/// same client key and node. A client keeps the chain of a conversation so
/// a node cannot present a turn as continuing one it never served.
pub fn verify_chain(
    previous: &Receipt,
    previous_binding: &VerifiedBinding,
    next: &Receipt,
    next_binding: &VerifiedBinding,
) -> Result<()> {
    let parent = next
        .binding
        .as_ref()
        .and_then(|b| b.parent.as_deref())
        .context("the receipt names no parent turn")?;
    if hex_array::<32>(parent, "the parent turn root")? != previous_binding.turn_root {
        bail!(
            "receipt {} continues another turn than receipt {}",
            next.receipt_id,
            previous.receipt_id
        );
    }
    if next_binding.client_key != previous_binding.client_key {
        bail!("the turns name different client keys");
    }
    if next_binding.node_key != previous_binding.node_key {
        bail!("the turns were signed by different nodes");
    }
    Ok(())
}

/// The VRF input the challenge is drawn from.
pub fn challenge_input(turn_root: &Hash) -> Vec<u8> {
    [CHALLENGE_PREFIX, turn_root].concat()
}

/// The VRF input the checked rows are drawn from: the opening as the
/// client received it.
pub fn rows_input(opening_bytes: &[u8]) -> Vec<u8> {
    rows_input_of_digest(blake3::hash(opening_bytes).as_bytes())
}

/// [`rows_input`] from the BLAKE3 digest of the opening's bytes.
pub fn rows_input_of_digest(opening_digest: &Hash) -> Vec<u8> {
    [ROWS_PREFIX, opening_digest].concat()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::{RECEIPT_VERSION, StageCommitment};

    fn receipt() -> Receipt {
        Receipt {
            version: RECEIPT_VERSION,
            receipt_id: "r".into(),
            model: "m".into(),
            position_start: 0,
            position_end: 3,
            tokens: vec![1, 2, 3],
            generation_start: 2,
            final_token: Some(4),
            stages: vec![StageCommitment {
                stage_index: 0,
                layer_start: 0,
                layer_end: 2,
                root: "ab".into(),
                leaf_count: 15,
                backend: "CPU".into(),
                cache_types: vec!["f16".into()],
                prefix: None,
                ubatches: Vec::new(),
                recurrent_state: None,
            }],
            binding: None,
        }
    }

    fn request() -> BindingRequest {
        BindingRequest {
            client_key: crate::vrf::SecretKey::from_seed([9; 32]).public(),
            parent: None,
            runtime: "skippy 0.1.77".into(),
            sampler: GREEDY.into(),
        }
    }

    #[test]
    fn a_bound_receipt_verifies() {
        let node = SigningKey::from_bytes(&[5; 32]);
        let mut r = receipt();
        bind(&mut r, &request(), &node);
        let verified = verify(&r, Some(&request().client_key)).unwrap();
        assert_eq!(verified.node_key, node.verifying_key().to_bytes());
    }

    #[test]
    fn a_turn_chains_to_its_parent() {
        let node = SigningKey::from_bytes(&[5; 32]);
        let mut first = receipt();
        bind(&mut first, &request(), &node);
        let first_bound = verify(&first, None).unwrap();
        let chained = |parent: Hash, signer: &SigningKey| {
            let mut next = receipt();
            next.receipt_id = "r2".into();
            let request = BindingRequest {
                parent: Some(parent),
                ..request()
            };
            bind(&mut next, &request, signer);
            let bound = verify(&next, None).unwrap();
            (next, bound)
        };
        let (next, bound) = chained(first_bound.turn_root, &node);
        verify_chain(&first, &first_bound, &next, &bound).unwrap();
        let (other, bound) = chained([1; 32], &node);
        assert!(verify_chain(&first, &first_bound, &other, &bound).is_err());
        let (moved, bound) = chained(first_bound.turn_root, &SigningKey::from_bytes(&[6; 32]));
        assert!(verify_chain(&first, &first_bound, &moved, &bound).is_err());
        assert!(verify_chain(&first, &first_bound, &first, &first_bound).is_err());
    }

    #[test]
    fn every_field_is_bound() {
        let node = SigningKey::from_bytes(&[5; 32]);
        let mut bound = receipt();
        bind(&mut bound, &request(), &node);
        type Change = Box<dyn Fn(&mut Receipt)>;
        let changes: Vec<Change> = vec![
            Box::new(|r| r.tokens[1] = 9),
            Box::new(|r| r.final_token = None),
            Box::new(|r| r.generation_start = 1),
            Box::new(|r| r.stages[0].root = "ac".into()),
            Box::new(|r| r.stages[0].backend = "CUDA0".into()),
            Box::new(|r| r.stages[0].ubatches = vec![0, 3]),
            Box::new(|r| {
                r.stages[0].prefix = Some(crate::wire::PrefixCommitment {
                    root: "ad".into(),
                    positions: 3,
                    tokens: None,
                });
            }),
            Box::new(|r| {
                r.stages[0].prefix = Some(crate::wire::PrefixCommitment {
                    root: String::new(),
                    positions: 0,
                    tokens: Some("ae".into()),
                });
            }),
            Box::new(|r| r.model = "n".into()),
            Box::new(|r| r.binding.as_mut().unwrap().runtime = "other".into()),
            Box::new(|r| r.binding.as_mut().unwrap().sampler = SAMPLED.into()),
            Box::new(|r| r.binding.as_mut().unwrap().parent = Some("00".into())),
        ];
        for change in changes {
            let mut r = bound.clone();
            change(&mut r);
            assert!(verify(&r, None).is_err());
        }
    }

    #[test]
    fn an_opening_is_signed_for_its_turn() {
        let node = SigningKey::from_bytes(&[5; 32]);
        let mut r = receipt();
        bind(&mut r, &request(), &node);
        let bound = verify(&r, None).unwrap();
        let challenge = Challenge {
            receipt_id: r.receipt_id.clone(),
            picks: vec![crate::wire::Pick {
                layer: 1,
                positions: vec![0, 2],
            }],
            proof: Some("ab".into()),
            head: vec![2],
            attention: Vec::new(),
        };
        let signature = sign_opening(&node, &bound.turn_root, &challenge, b"{}");
        verify_opening(&bound, &challenge, b"{}", &signature).unwrap();
        assert!(verify_opening(&bound, &challenge, b"{ }", &signature).is_err());
        let other_turn = VerifiedBinding {
            turn_root: [0; 32],
            ..bound.clone()
        };
        assert!(verify_opening(&other_turn, &challenge, b"{}", &signature).is_err());
        // an opening is evidence only of the challenge the node answered
        let mut other_picks = challenge.clone();
        other_picks.picks[0].positions = vec![0, 1];
        assert!(verify_opening(&bound, &other_picks, b"{}", &signature).is_err());
        let mut other_head = challenge.clone();
        other_head.head = vec![1];
        assert!(verify_opening(&bound, &other_head, b"{}", &signature).is_err());
        let mut other_proof = challenge.clone();
        other_proof.proof = None;
        assert!(verify_opening(&bound, &other_proof, b"{}", &signature).is_err());
    }

    #[test]
    fn attention_rows_are_signed_for_the_blocks_asked() {
        let node = SigningKey::from_bytes(&[5; 32]);
        let mut r = receipt();
        bind(&mut r, &request(), &node);
        let bound = verify(&r, None).unwrap();
        let blocks = BlockChallenge {
            receipt_id: r.receipt_id.clone(),
            picks: vec![crate::attention::opening::BlockPick {
                position: 2,
                layer: 1,
                blocks: vec![0, 3],
            }],
            proof: Some("cd".into()),
        };
        let signature = sign_rows(&node, &bound.turn_root, &blocks, b"{}");
        verify_rows(&bound, &blocks, b"{}", &signature).unwrap();
        assert!(verify_rows(&bound, &blocks, b"{ }", &signature).is_err());
        let mut other = blocks.clone();
        other.picks[0].blocks = vec![0, 2];
        assert!(verify_rows(&bound, &other, b"{}", &signature).is_err());
    }

    #[test]
    fn another_client_key_or_node_signature_fails() {
        let node = SigningKey::from_bytes(&[5; 32]);
        let mut r = receipt();
        bind(&mut r, &request(), &node);
        assert!(verify(&r, Some(&[1; 32])).is_err());
        // a node that re-signs with its own key must also rewrite node_key,
        // which changes the root
        let mut forged = r.clone();
        let other = SigningKey::from_bytes(&[6; 32]);
        let root = crate::vrf::key_from_hex(&forged.binding.as_ref().unwrap().turn_root).unwrap();
        forged.binding.as_mut().unwrap().signature = hex::encode(other.sign(&root).to_bytes());
        assert!(verify(&forged, None).is_err());
        assert!(verify(&receipt(), None).is_err());
    }
}
