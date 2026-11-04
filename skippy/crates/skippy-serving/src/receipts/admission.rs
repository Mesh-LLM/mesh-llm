//! Which challenges a stage opens, decided before it replays anything.
//!
//! A stage opens a challenge only of a turn it keeps a record of
//! ([`super::kept_turns`]); any other receipt id is not found, on every
//! route and on the stage connection. The challenge must be one a draw
//! makes: stage 0 checks it against the receipt, every stage against the
//! turn it committed, so an opening costs a bounded replay whoever asks.
//!
//! A turn bound to a client key is opened for one challenge, and only with
//! a VRF proof of its turn root under that key, so the client cannot
//! choose what it challenges and nobody else reads the turn's activations.
//! Stage 0 knows the root from the receipt it signed. A later stage got the
//! client key with the commit and gets the root with the forwarded
//! challenge; it checks the same proof, and holds the turn to the first
//! root a proof was good for. Rows are read the same way: for blocks of an
//! admitted challenge's attention picks, and for a bound turn with the
//! client's proof of the opening they follow.
//!
//! A turn bound to no key has no proof to ask for. It is opened where a
//! stage's own receipts API or the stage before it asks, for at most
//! [`MAX_REPLAYS_PER_RECEIPT`] challenges.

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use skippy_receipts::attention::opening::BlockChallenge;
use skippy_receipts::challenge;
use skippy_receipts::commitment::Hash;
use skippy_receipts::recorder::MAX_REPLAYS_PER_RECEIPT;
use skippy_receipts::turn;
use skippy_receipts::vrf;
use skippy_receipts::wire::{Challenge, Receipt};

use super::kept_turns::{KeptTurn, LaterStages, Outcome};
use super::opening_queue::QueueKey;
use super::refusal::{NotFound, not_found};
use super::{StageReceipts, lock};

/// A receipt id a stage keeps a record under, at most this long. Stage 0
/// makes ids of 36 bytes; a later stage takes them from the stage before.
pub(super) const MAX_RECEIPT_ID_BYTES: usize = 128;
/// Bytes an admitted challenge holds at most, which a stage keeps per turn:
/// its id, its proof, and the picks, head positions and attention picks of
/// a draw.
pub(super) const MAX_ADMITTED_CHALLENGE_BYTES: usize = 1024;

/// What a stage needs of a challenge it admitted to open it on the stages
/// after it.
pub(super) struct AdmittedChallenge {
    /// The turn root the proof is over, for a bound turn.
    pub turn_root: Option<Hash>,
    pub later: LaterStages,
}

/// As [`AdmittedChallenge`], for rows.
pub(super) struct AdmittedRows {
    /// The digest of the opening the proof is over, for a bound turn.
    pub opening: Option<Hash>,
    pub later: LaterStages,
}

/// Bytes a kept challenge holds.
fn challenge_bytes(challenge: &Challenge) -> usize {
    let picks: usize = challenge
        .picks
        .iter()
        .map(|pick| size_of_val(pick) + pick.positions.len() * size_of::<u32>())
        .sum();
    size_of::<Challenge>()
        + challenge.receipt_id.len()
        + challenge.proof.as_ref().map_or(0, String::len)
        + picks
        + challenge.head.len() * size_of::<u32>()
        + size_of_val(challenge.attention.as_slice())
}

/// The record of a turn that was committed, or why there is none to open.
fn committed<'a>(kept: Option<&'a KeptTurn>, id: &str) -> Result<&'a KeptTurn> {
    let turn = kept.ok_or_else(|| not_found(id))?;
    match &turn.outcome {
        Outcome::Issued(_) | Outcome::Committed => Ok(turn),
        Outcome::Committing => Err(anyhow::Error::new(NotFound(format!(
            "receipt {id} is not committed yet"
        )))),
        Outcome::Refused(reason) => bail!("turn {id} got no receipt: {reason}"),
    }
}

impl StageReceipts {
    /// Whether the receipt `id` is bound to a client key. Fails as an
    /// opening of it would: not found for an id the stage keeps no record
    /// of.
    pub(super) fn is_bound(&self, id: &str) -> Result<bool> {
        let kept = lock(&self.kept);
        Ok(committed(kept.get(id), id)?.client_key.is_some())
    }

    /// The client key a request of receipt `id` waits under
    /// ([`super::opening_queue`]). Fails as an opening of it would, so a
    /// request for an unknown receipt takes no place in the queue.
    pub(super) fn queue_key(&self, id: &str) -> Result<QueueKey> {
        let kept = lock(&self.kept);
        Ok(committed(kept.get(id), id)?.client_key)
    }

    /// The receipt of a turn, or why it got none; `None` for an id the
    /// stage keeps no record of, or whose turn is still being committed.
    ///
    /// The receipt id is all a caller shows, so a bound receipt comes
    /// without the turn's tokens: its client got it whole with the
    /// response, and nobody else reads what the client was told.
    pub(super) fn receipt(&self, id: &str) -> Option<Result<Arc<Receipt>, String>> {
        match &lock(&self.kept).get(id)?.outcome {
            Outcome::Issued(receipt) if receipt.binding.is_some() => {
                let mut redacted = Receipt::clone(receipt);
                redacted.tokens.clear();
                redacted.final_token = None;
                Some(Ok(Arc::new(redacted)))
            }
            Outcome::Issued(receipt) => Some(Ok(receipt.clone())),
            Outcome::Refused(reason) => Some(Err(reason.clone())),
            Outcome::Committing | Outcome::Committed => None,
        }
    }

    /// Admits `challenge` for a turn this stage committed, before any
    /// replay. `forwarded_root` is the turn root the stage before sent
    /// along, which a later stage checks a bound turn's proof against.
    pub(super) fn admit_challenge(
        &self,
        challenge: &Challenge,
        forwarded_root: Option<Hash>,
    ) -> Result<AdmittedChallenge> {
        let id = &challenge.receipt_id;
        let (receipt, client_key, known_root) = {
            let kept = lock(&self.kept);
            let turn = committed(kept.get(id), id)?;
            let receipt = match &turn.outcome {
                Outcome::Issued(receipt) => Some(receipt.clone()),
                _ => None,
            };
            (receipt, turn.client_key, turn.turn_root)
        };
        // the size of the challenge first: nothing here costs more than a
        // few hashes until it is known to ask for a draw's worth
        if let Some(receipt) = &receipt {
            challenge::check_shape(challenge, receipt)?;
        }
        self.reports.recorder().check_challenge(challenge)?;
        if challenge_bytes(challenge) > MAX_ADMITTED_CHALLENGE_BYTES {
            bail!("the challenge holds more than a draw and its proof");
        }
        let root = match client_key {
            Some(key) => {
                let proof = challenge.proof.as_deref().context(
                    "the receipt is bound to a client key; the challenge needs its VRF proof",
                )?;
                let root = known_root.or(forwarded_root).context(
                    "the receipt is bound to a client key; the challenge needs the turn root \
                     its proof is over",
                )?;
                if forwarded_root.is_some_and(|forwarded| forwarded != root) {
                    bail!("the challenge names another turn root than the receipt's");
                }
                vrf::verify(
                    &key,
                    &turn::challenge_input(&root),
                    &vrf::proof_from_hex(proof)?,
                )
                .context("the challenge's VRF proof")?;
                Some(root)
            }
            None => None,
        };
        let mut kept = lock(&self.kept);
        let turn = kept.get_mut(id).ok_or_else(|| not_found(id))?;
        if turn.client_key != client_key || turn.turn_root != known_root {
            bail!("receipt {id} names another turn since the challenge arrived");
        }
        let limit = match client_key {
            Some(_) => 1,
            None => MAX_REPLAYS_PER_RECEIPT as usize,
        };
        if !turn.admit(challenge, limit) {
            bail!("receipt {id} was already challenged");
        }
        turn.turn_root = turn.turn_root.or(root);
        Ok(AdmittedChallenge {
            turn_root: root,
            later: turn.later.clone(),
        })
    }

    /// Admits `blocks` for a turn this stage opened, before any row is
    /// read. `forwarded_opening` is the digest of the opening stage 0
    /// signed, as the stage before sent it along.
    pub(super) fn admit_rows(
        &self,
        blocks: &BlockChallenge,
        forwarded_opening: Option<Hash>,
    ) -> Result<AdmittedRows> {
        let id = &blocks.receipt_id;
        let mut kept = lock(&self.kept);
        let turn = kept.get_mut(id).ok_or_else(|| not_found(id))?;
        committed(Some(&*turn), id)?;
        let Some(first) = turn.challenges.first() else {
            bail!("receipt {id} was not opened");
        };
        if !turn
            .challenges
            .iter()
            .any(|admitted| blocks.check_shape(admitted).is_ok())
        {
            blocks.check_shape(first)?;
            bail!("the rows ask for blocks of no challenge that was opened");
        }
        let opening = match turn.client_key {
            Some(key) => {
                let issued = matches!(turn.outcome, Outcome::Issued(_));
                // stage 0 signed the opening itself; a later stage is told
                // which one the proof is over
                let opened = turn
                    .opened
                    .or(forwarded_opening.filter(|_| !issued))
                    .with_context(|| format!("receipt {id} was not opened"))?;
                if forwarded_opening.is_some_and(|forwarded| forwarded != opened) {
                    bail!("the rows follow another opening than the one this stage knows");
                }
                let proof = blocks
                    .proof
                    .as_deref()
                    .context("the receipt is bound to a client key; the rows need its VRF proof")?;
                vrf::verify(
                    &key,
                    &turn::rows_input_of_digest(&opened),
                    &vrf::proof_from_hex(proof)?,
                )
                .context("the rows' VRF proof")?;
                turn.opened = Some(opened);
                Some(opened)
            }
            None => None,
        };
        Ok(AdmittedRows {
            opening,
            later: turn.later.clone(),
        })
    }

    /// Notes that this stage opened a turn, so its record is kept while the
    /// client may ask for the rows.
    pub(super) fn mark_opened(&self, id: &str) {
        if let Some(turn) = lock(&self.kept).get_mut(id) {
            turn.mark_opened(None, Instant::now());
        }
    }

    /// The root of a bound receipt this stage issued and signs for.
    fn signed_root(&self, id: &str) -> Option<Hash> {
        let kept = lock(&self.kept);
        let turn = kept.get(id)?;
        match (&turn.outcome, turn.client_key) {
            (Outcome::Issued(_), Some(_)) => turn.turn_root,
            _ => None,
        }
    }

    /// Stage 0's signature of its opening of `challenge`, for a bound
    /// receipt it issued. The rows' proof is over the opening signed last.
    pub(super) fn sign_opening(
        &self,
        challenge: &Challenge,
        bytes: &[u8],
    ) -> Result<Option<String>> {
        let id = &challenge.receipt_id;
        let Some(root) = self.signed_root(id) else {
            return Ok(None);
        };
        if let Some(turn) = lock(&self.kept).get_mut(id) {
            turn.mark_opened(Some(*blake3::hash(bytes).as_bytes()), Instant::now());
        }
        Ok(Some(turn::sign_opening(
            &self.signing_key,
            &root,
            challenge,
            bytes,
        )))
    }

    /// Stage 0's signature of the attention rows it opened for `blocks`,
    /// for a bound receipt it issued.
    pub(super) fn sign_rows(
        &self,
        blocks: &BlockChallenge,
        bytes: &[u8],
    ) -> Result<Option<String>> {
        let Some(root) = self.signed_root(&blocks.receipt_id) else {
            return Ok(None);
        };
        Ok(Some(turn::sign_rows(
            &self.signing_key,
            &root,
            blocks,
            bytes,
        )))
    }

    /// Stage 0's signature of its answer that the chain can no longer open
    /// `request`, the digest of a challenge or block challenge, for a bound
    /// receipt it issued.
    pub(super) fn sign_cannot_open(
        &self,
        id: &str,
        request: &Hash,
        bytes: &[u8],
    ) -> Option<String> {
        let root = self.signed_root(id)?;
        Some(turn::sign_cannot_open(
            &self.signing_key,
            &root,
            request,
            bytes,
        ))
    }
}

#[cfg(test)]
mod tests {
    use skippy_receipts::attention::opening::AttentionPick;
    use skippy_receipts::challenge::{
        ATTENTION_POSITIONS_PER_LAYER, HEAD_POSITIONS, LAYERS_PER_CHALLENGE, POSITIONS_PER_LAYER,
    };
    use skippy_receipts::wire::Pick;

    use super::*;

    // A stage keeps a clone of each challenge it admits; the largest one a
    // draw makes fits the bound it holds them to.
    #[test]
    fn the_largest_challenge_a_draw_makes_fits_what_a_stage_keeps() {
        let picks: Vec<Pick> = (0..LAYERS_PER_CHALLENGE as u32)
            .map(|layer| Pick {
                layer,
                positions: (0..POSITIONS_PER_LAYER as u32).collect(),
            })
            .collect();
        let attention = picks
            .iter()
            .flat_map(|pick| {
                pick.positions
                    .iter()
                    .take(ATTENTION_POSITIONS_PER_LAYER)
                    .map(|&position| AttentionPick {
                        position,
                        layer: pick.layer,
                        head_draw: u32::MAX,
                    })
            })
            .collect();
        let largest = Challenge {
            receipt_id: "r".repeat(MAX_RECEIPT_ID_BYTES),
            picks,
            proof: Some("0".repeat(2 * vrf::PROOF_LEN)),
            head: (0..HEAD_POSITIONS as u32).collect(),
            attention,
        };
        let bytes = challenge_bytes(&largest);
        assert!(bytes <= MAX_ADMITTED_CHALLENGE_BYTES, "{bytes}");
        let mut longer = largest;
        longer.proof = Some("0".repeat(MAX_ADMITTED_CHALLENGE_BYTES));
        assert!(challenge_bytes(&longer) > MAX_ADMITTED_CHALLENGE_BYTES);
    }
}
