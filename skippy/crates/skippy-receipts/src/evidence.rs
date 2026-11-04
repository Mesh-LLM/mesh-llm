//! Evidence of an audit, which anyone can recheck from public data: the
//! signed receipt, the challenge with the VRF proof that drew it, the
//! opening as the node sent and signed it, the VRF proof that drew the
//! checked rows, and the attention rows the node then sent and signed.
//!
//! [`Evidence::verify`] shows that neither side chose what was checked:
//! the node signed the receipt and the opening, and the client's VRF drew
//! the challenge from the turn root and the rows from the opening's bytes.
//! The checks themselves then run again on the claimed GGUF. The response
//! text is not part of it: the node does not sign the response, so the
//! comparison of the response with the sampled tokens convinces only the
//! client that received it.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::attention::opening::BlockChallenge;
use crate::challenge;
use crate::turn::{self, VerifiedBinding};
use crate::vrf;
use crate::wire::{Challenge, Opening, Receipt};

pub(crate) const EVIDENCE_VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Evidence {
    pub version: u32,
    pub receipt: Receipt,
    /// The challenge, with the VRF proof of the turn root in `proof`.
    pub challenge: Challenge,
    /// Base64 of the opening's bytes as the node sent them.
    pub opening: String,
    /// Hex node signature of the opening ([`turn::sign_opening`]).
    pub opening_signature: String,
    /// Hex VRF proof of the opening's bytes, which drew the checked rows.
    pub rows_proof: String,
    /// Base64 of the attention rows' bytes as the node sent them, when it
    /// sent any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rows: Option<String>,
    /// Hex node signature of the rows ([`turn::sign_rows`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rows_signature: Option<String>,
}

/// What [`Evidence::verify`] establishes, for rerunning the checks.
pub struct Provenance {
    pub binding: VerifiedBinding,
    pub opening: Opening,
    /// The VRF output the checked rows are drawn from.
    pub rows_seed: vrf::Output,
    /// The attention rows the node sent, when it sent any. Their signature
    /// is checked against the blocks a recheck draws ([`SignedRows::answering`]).
    pub rows: Option<SignedRows>,
}

impl Evidence {
    pub fn new(
        receipt: Receipt,
        challenge: Challenge,
        opening_bytes: &[u8],
        opening_signature: String,
        rows_proof: &vrf::Proof,
    ) -> Self {
        Self {
            version: EVIDENCE_VERSION,
            receipt,
            challenge,
            opening: crate::wire::encode_bytes(opening_bytes),
            opening_signature,
            rows_proof: hex::encode(rows_proof),
            rows: None,
            rows_signature: None,
        }
    }

    /// Adds the attention rows the node sent after the opening.
    #[must_use]
    pub fn with_rows(mut self, rows_bytes: &[u8], signature: Option<String>) -> Self {
        self.rows = Some(crate::wire::encode_bytes(rows_bytes));
        self.rows_signature = signature;
        self
    }

    pub fn opening_bytes(&self) -> Result<Vec<u8>> {
        crate::wire::decode_bytes(&self.opening).context("decode the opening's bytes")
    }

    /// Checks that the node signed the receipt, the opening and the rows, and that
    /// the client's VRF drew the challenge and the rows. `layer_bytes` is
    /// the claimed GGUF's weight bytes per executable layer, which the
    /// challenge was drawn with.
    pub fn verify(&self, layer_bytes: &BTreeMap<u32, u64>) -> Result<Provenance> {
        if self.version != EVIDENCE_VERSION {
            bail!(
                "evidence version {} is not {EVIDENCE_VERSION}",
                self.version
            );
        }
        let binding = turn::verify(&self.receipt, None)?;
        let proof = self
            .challenge
            .proof
            .as_deref()
            .context("the challenge carries no VRF proof")?;
        let beta = vrf::verify(
            &binding.client_key,
            &turn::challenge_input(&binding.turn_root),
            &vrf::proof_from_hex(proof)?,
        )
        .context("the challenge's VRF proof")?;
        let mut drawn = challenge::draw(&self.receipt, layer_bytes, &beta)?;
        drawn.proof = self.challenge.proof.clone();
        if drawn != self.challenge {
            bail!("the challenge is not the one the VRF of the turn root draws");
        }
        let bytes = self.opening_bytes()?;
        turn::verify_opening(&binding, &self.challenge, &bytes, &self.opening_signature)?;
        let rows_seed = vrf::verify(
            &binding.client_key,
            &turn::rows_input(&bytes),
            &vrf::proof_from_hex(&self.rows_proof)?,
        )
        .context("the rows' VRF proof")?;
        let opening = serde_json::from_slice(&bytes).context("decode the opening")?;
        let rows = match &self.rows {
            Some(rows) => Some(SignedRows {
                bytes: crate::wire::decode_bytes(rows).context("decode the rows' bytes")?,
                signature: self
                    .rows_signature
                    .clone()
                    .context("the rows carry no node signature")?,
                proof: self.rows_proof.clone(),
            }),
            None => None,
        };
        Ok(Provenance {
            binding,
            opening,
            rows_seed,
            rows,
        })
    }
}

/// The attention rows of evidence, as the node sent and signed them.
pub struct SignedRows {
    bytes: Vec<u8>,
    signature: String,
    /// Hex VRF proof the client sent with its block request.
    proof: String,
}

impl SignedRows {
    /// The rows' bytes, once the node's signature shows it sent them in
    /// answer to `blocks`: the blocks the rows' seed draws, which a
    /// recheck draws again. Rows the node sent for other blocks are not
    /// evidence.
    pub fn answering(&self, binding: &VerifiedBinding, blocks: &BlockChallenge) -> Result<&[u8]> {
        let mut asked = blocks.clone();
        asked.proof = Some(self.proof.clone());
        turn::verify_rows(binding, &asked, &self.bytes, &self.signature)
            .context("the rows' signature")?;
        Ok(&self.bytes)
    }

    pub fn signature(&self) -> &str {
        &self.signature
    }
}

#[cfg(test)]
mod tests {
    use ed25519_dalek::SigningKey;

    use super::*;
    use crate::turn::{BindingRequest, GREEDY};
    use crate::wire::{RECEIPT_VERSION, StageCommitment};

    fn evidence() -> (Evidence, BTreeMap<u32, u64>) {
        let node = SigningKey::from_bytes(&[5; 32]);
        let client = vrf::SecretKey::from_seed([9; 32]);
        let mut receipt = Receipt {
            version: RECEIPT_VERSION,
            receipt_id: "r".into(),
            model: "m".into(),
            position_start: 0,
            position_end: 8,
            tokens: (0..8).collect(),
            generation_start: 4,
            final_token: Some(9),
            stages: vec![StageCommitment {
                stage_index: 0,
                layer_start: 0,
                layer_end: 4,
                root: "00".into(),
                leaf_count: 1,
                backend: "CPU".into(),
                cache_types: Vec::new(),
                prefix: None,
                ubatches: Vec::new(),
                recurrent_state: None,
            }],
            binding: None,
        };
        turn::bind(
            &mut receipt,
            &BindingRequest {
                client_key: client.public(),
                parent: None,
                runtime: "skippy-abi 0.1.77".into(),
                sampler: GREEDY.into(),
            },
            &node,
        );
        let bound = turn::verify(&receipt, None).unwrap();
        let layer_bytes: BTreeMap<u32, u64> = (0..4).map(|l| (l, 10)).collect();
        let (beta, proof) = client.evaluate(&turn::challenge_input(&bound.turn_root));
        let mut challenge = challenge::draw(&receipt, &layer_bytes, &beta).unwrap();
        challenge.proof = Some(hex::encode(proof));
        let bytes = br#"{"receipt_id":"r","stages":[]}"#;
        let signature = turn::sign_opening(&node, &bound.turn_root, &challenge, bytes);
        let (_, rows_proof) = client.evaluate(&turn::rows_input(bytes));
        let rows = br#"{"receipt_id":"r","stages":[]}"#;
        let mut asked = blocks();
        asked.proof = Some(hex::encode(rows_proof));
        let rows_signature = turn::sign_rows(&node, &bound.turn_root, &asked, rows);
        (
            Evidence::new(receipt, challenge, bytes, signature, &rows_proof)
                .with_rows(rows, Some(rows_signature)),
            layer_bytes,
        )
    }

    /// The blocks the rows' seed is taken to draw.
    fn blocks() -> BlockChallenge {
        BlockChallenge {
            receipt_id: "r".into(),
            picks: vec![crate::attention::opening::BlockPick {
                position: 5,
                layer: 1,
                blocks: vec![0],
            }],
            proof: None,
        }
    }

    #[test]
    fn evidence_shows_who_chose_what() {
        let (evidence, layer_bytes) = evidence();
        let provenance = evidence.verify(&layer_bytes).unwrap();
        assert_eq!(provenance.opening.receipt_id, "r");
        // a round trip through JSON keeps it valid
        let text = serde_json::to_string(&evidence).unwrap();
        let back: Evidence = serde_json::from_str(&text).unwrap();
        back.verify(&layer_bytes).unwrap();
        // the rows answer the blocks the node signed them for
        let rows = provenance.rows.as_ref().expect("rows");
        rows.answering(&provenance.binding, &blocks()).unwrap();
    }

    #[test]
    fn rows_the_node_sent_for_other_blocks_are_not_evidence() {
        let (evidence, layer_bytes) = evidence();
        let provenance = evidence.verify(&layer_bytes).unwrap();
        let rows = provenance.rows.as_ref().expect("rows");
        let mut other = blocks();
        other.picks[0].blocks = vec![1];
        assert!(rows.answering(&provenance.binding, &other).is_err());
    }

    #[test]
    fn an_opening_of_another_challenge_is_not_evidence() {
        // the client sent picks of its own with the true proof, got them
        // opened, and shows the opening with the challenge the VRF draws
        let node = SigningKey::from_bytes(&[5; 32]);
        let (mut evidence, layer_bytes) = evidence();
        let bound = turn::verify(&evidence.receipt, None).unwrap();
        let mut sent = evidence.challenge.clone();
        sent.picks.truncate(1);
        let bytes = evidence.opening_bytes().unwrap();
        evidence.opening_signature = turn::sign_opening(&node, &bound.turn_root, &sent, &bytes);
        assert!(evidence.verify(&layer_bytes).is_err());
    }

    #[test]
    fn a_challenge_picked_by_hand_is_not_evidence_though_the_node_answered_it() {
        // `audit-receipt --layer` challenges one layer of the client's own
        // choosing, with the true VRF proof. The node opens it and signs
        // the opening of that very challenge, so every signature is good.
        // Shown as evidence, the challenge is still not the one the VRF
        // draws: a client could otherwise pick what an honest node's rare
        // outliers fail.
        let node = SigningKey::from_bytes(&[5; 32]);
        let (mut evidence, layer_bytes) = evidence();
        let bound = turn::verify(&evidence.receipt, None).unwrap();
        let drawn = evidence.challenge.clone();
        evidence.challenge.picks.truncate(1);
        evidence.challenge.attention =
            challenge::draw_attention(&evidence.challenge.picks, b"any seed");
        assert_ne!(evidence.challenge, drawn);
        let bytes = evidence.opening_bytes().unwrap();
        evidence.opening_signature =
            turn::sign_opening(&node, &bound.turn_root, &evidence.challenge, &bytes);
        let error = evidence
            .verify(&layer_bytes)
            .err()
            .expect("a hand-picked challenge");
        assert!(
            error.to_string().contains("not the one the VRF"),
            "{error:#}"
        );
    }

    #[test]
    fn a_changed_part_of_the_evidence_fails() {
        let (evidence, layer_bytes) = evidence();
        // a challenge the client picked by hand
        let mut picked = evidence.clone();
        picked.challenge.picks[0].layer ^= 1;
        assert!(picked.verify(&layer_bytes).is_err());
        // an opening the node did not send
        let mut forged = evidence.clone();
        forged.opening = crate::wire::encode_bytes(br#"{"receipt_id":"r","stages":[ ]}"#);
        assert!(forged.verify(&layer_bytes).is_err());
        // rows drawn from another output
        let mut rows = evidence.clone();
        rows.rows_proof = evidence.challenge.proof.clone().unwrap();
        assert!(rows.verify(&layer_bytes).is_err());
        // attention rows without a signature
        let mut unsigned = evidence.clone();
        unsigned.rows_signature = None;
        assert!(unsigned.verify(&layer_bytes).is_err());
        // attention rows the node did not sign
        let mut forged_rows = evidence.clone();
        forged_rows.rows = Some(crate::wire::encode_bytes(b"[]"));
        let provenance = forged_rows.verify(&layer_bytes).unwrap();
        let rows = provenance.rows.as_ref().expect("rows");
        assert!(rows.answering(&provenance.binding, &blocks()).is_err());
        // the opening's signature does not sign rows
        let mut swapped = evidence.clone();
        swapped.rows = Some(evidence.opening.clone());
        swapped.rows_signature = Some(evidence.opening_signature.clone());
        let provenance = swapped.verify(&layer_bytes).unwrap();
        let rows = provenance.rows.as_ref().expect("rows");
        assert!(rows.answering(&provenance.binding, &blocks()).is_err());
    }
}
