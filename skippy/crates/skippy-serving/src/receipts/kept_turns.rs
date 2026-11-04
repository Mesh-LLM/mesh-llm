//! The turns a stage keeps a record of, by receipt id: on stage 0 the
//! receipt itself, on every stage what admitting a challenge of the turn
//! takes (the client key it is bound to, the challenges admitted so far).
//!
//! A record that a client can still challenge is never dropped to make
//! room: a bound turn is protected for its challenge window
//! ([`CHALLENGE_PIN`]) and, once opened, until its rows can no longer be
//! asked for ([`ROWS_PIN`]). Records past that go oldest first, when there
//! are more than [`UNPROTECTED_TURNS`] of them or the byte budget is
//! reached. When protected records alone fill the budget, a new turn gets
//! no receipt; no turn loses its own inside its window.
//!
//! A receipt id names one turn. A turn that reuses an id replaces the
//! record, and with it every challenge admitted for the old turn, unless
//! the id is still a protected turn of another client: then the new turn
//! gets no receipt, so a client cannot shadow a receipt it does not own.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Result, bail};
use skippy_receipts::attention::opening::ROWS_PIN;
use skippy_receipts::commitment::Hash;
use skippy_receipts::recorder::CHALLENGE_PIN;
use skippy_receipts::wire::{Challenge, Receipt, StageCommitment};

/// Bytes of records a stage keeps, at most.
pub(super) const KEPT_BYTES_BUDGET: usize = 64 << 20;
/// Records no client can challenge any more that a stage keeps, so a
/// receipt stays retrievable and a late request is told what became of its
/// turn.
pub(super) const UNPROTECTED_TURNS: usize = 256;
/// What a record costs besides its receipt and challenges.
const RECORD_BYTES: usize = 256;

/// What became of a turn.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Outcome {
    /// The turn is being committed.
    Committing,
    /// Stage 0: the turn's receipt.
    Issued(Arc<Receipt>),
    /// A later stage: the turn is committed here, and stage 0 holds its
    /// receipt.
    Committed,
    /// Why the turn got no receipt.
    Refused(String),
}

/// The positions a turn committed and the commitments of the stages after
/// this one, which bound what those stages may answer.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct LaterStages {
    pub position_start: u32,
    pub position_end: u32,
    pub commitments: Arc<[StageCommitment]>,
}

pub(super) struct KeptTurn {
    pub outcome: Outcome,
    /// The client VRF key the turn is bound to; `None` for a turn nobody
    /// can challenge with a proof.
    pub client_key: Option<[u8; 32]>,
    /// The turn's root: stage 0 knows it from the receipt it signed, a
    /// later stage from the first challenge it admitted.
    pub turn_root: Option<Hash>,
    /// The challenges admitted for the turn: one, for a bound turn.
    pub challenges: Vec<Challenge>,
    /// The digest of the opening the rows' proof is over.
    pub opened: Option<Hash>,
    pub later: LaterStages,
    kept_at: Instant,
    opened_at: Option<Instant>,
    bytes: usize,
    /// Tells this record from an older one of the same id in `order`.
    generation: u64,
}

impl KeptTurn {
    /// Whether a client can still challenge the turn, or ask for its rows.
    /// A turn being committed is protected until it is settled.
    fn protected(&self, now: Instant) -> bool {
        if matches!(self.outcome, Outcome::Committing) {
            return true;
        }
        if self.client_key.is_none() || matches!(self.outcome, Outcome::Refused(_)) {
            return false;
        }
        let until = match self.opened_at {
            Some(at) => at + ROWS_PIN,
            None => self.kept_at + CHALLENGE_PIN,
        };
        now < until
    }

    /// The stage opened the turn: its rows may be asked for from now.
    pub(super) fn mark_opened(&mut self, digest: Option<Hash>, now: Instant) {
        self.opened = digest.or(self.opened);
        self.opened_at = Some(now);
    }

    /// Admits `challenge` as one of the turn's, at most `limit` of them.
    /// `false` when the turn already has that many others.
    pub(super) fn admit(&mut self, challenge: &Challenge, limit: usize) -> bool {
        if self.challenges.contains(challenge) {
            return true;
        }
        if self.challenges.len() >= limit {
            return false;
        }
        self.challenges.push(challenge.clone());
        true
    }
}

/// Why a reserved record was not settled with its turn's outcome.
#[derive(Debug)]
pub(super) enum Unsettled {
    /// Another turn took the receipt id meanwhile; the record is that
    /// turn's.
    Replaced(String),
    /// The record does not fit beside those clients can still challenge;
    /// it now says so.
    Full(String),
}

impl std::fmt::Display for Unsettled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (Self::Replaced(why) | Self::Full(why)) = self;
        f.write_str(why)
    }
}

impl std::error::Error for Unsettled {}

/// Names the record a [`KeptTurns::reserve`] started, for its
/// [`KeptTurns::settle`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Reservation(u64);

pub(super) struct KeptTurns {
    by_id: HashMap<String, KeptTurn>,
    /// Receipt ids by when their record was made, oldest first.
    order: VecDeque<(u64, String)>,
    bytes: usize,
    budget: usize,
    generations: u64,
}

impl Default for KeptTurns {
    fn default() -> Self {
        Self::new(KEPT_BYTES_BUDGET)
    }
}

impl KeptTurns {
    pub(super) fn new(budget: usize) -> Self {
        Self {
            by_id: HashMap::new(),
            order: VecDeque::new(),
            bytes: 0,
            budget,
            generations: 0,
        }
    }

    pub(super) fn get(&self, id: &str) -> Option<&KeptTurn> {
        self.by_id.get(id)
    }

    pub(super) fn get_mut(&mut self, id: &str) -> Option<&mut KeptTurn> {
        self.by_id.get_mut(id)
    }

    /// Starts the record of a turn that is about to be committed as `id`,
    /// bound to `client_key` when a client can challenge it. Fails, keeping
    /// what was there, when `id` is another client's turn inside its
    /// window, or when the records clients can still challenge fill the
    /// budget.
    pub(super) fn reserve(
        &mut self,
        id: &str,
        client_key: Option<[u8; 32]>,
        now: Instant,
    ) -> Result<Reservation> {
        if let Some(old) = self.by_id.get(id)
            && old.protected(now)
            && old.client_key != client_key
        {
            bail!("receipt id {id} names a turn of another client that can still be challenged");
        }
        self.remove(id);
        let bytes = RECORD_BYTES + id.len();
        self.make_room(bytes, now)?;
        self.generations += 1;
        self.bytes += bytes;
        self.order.push_back((self.generations, id.to_owned()));
        self.by_id.insert(
            id.to_owned(),
            KeptTurn {
                outcome: Outcome::Committing,
                client_key,
                turn_root: None,
                challenges: Vec::new(),
                opened: None,
                later: LaterStages::default(),
                kept_at: now,
                opened_at: None,
                bytes,
                generation: self.generations,
            },
        );
        Ok(Reservation(self.generations))
    }

    /// Settles a reserved record with what became of its turn. A receipt
    /// that does not fit beside the records clients can still challenge is
    /// refused; the error says so. So is one whose id another turn took
    /// meanwhile, whose record stays as that turn left it.
    pub(super) fn settle(
        &mut self,
        id: &str,
        reservation: Reservation,
        outcome: Outcome,
        turn_root: Option<Hash>,
        later: LaterStages,
        now: Instant,
    ) -> Result<(), Unsettled> {
        let extra = match &outcome {
            Outcome::Issued(receipt) => receipt_bytes(receipt),
            Outcome::Refused(reason) => reason.len(),
            Outcome::Committing | Outcome::Committed => 0,
        } + std::mem::size_of_val(&*later.commitments);
        let fits = self.make_room(extra, now);
        let Some(kept) = self
            .by_id
            .get_mut(id)
            .filter(|kept| kept.generation == reservation.0)
        else {
            return Err(Unsettled::Replaced(format!(
                "receipt id {id} was taken by another turn while this one was committed"
            )));
        };
        kept.kept_at = now;
        if let Err(error) = fits {
            let why = format!("{error:#}");
            kept.outcome = Outcome::Refused(why.clone());
            return Err(Unsettled::Full(why));
        }
        kept.outcome = outcome;
        kept.turn_root = turn_root;
        kept.later = later;
        kept.bytes += extra;
        self.bytes += extra;
        Ok(())
    }

    fn remove(&mut self, id: &str) {
        if let Some(old) = self.by_id.remove(id) {
            self.bytes -= old.bytes;
        }
    }

    /// Drops records no client can challenge any more, oldest first, until
    /// `bytes` more fit and at most [`UNPROTECTED_TURNS`] of them are left.
    /// Fails when the protected ones leave no room.
    fn make_room(&mut self, bytes: usize, now: Instant) -> Result<()> {
        let by_id = &self.by_id;
        self.order
            .retain(|(generation, id)| by_id.get(id).is_some_and(|k| k.generation == *generation));
        let mut unprotected = self
            .by_id
            .values()
            .filter(|kept| !kept.protected(now))
            .count();
        let mut kept_order = VecDeque::with_capacity(self.order.len());
        while let Some((generation, id)) = self.order.pop_front() {
            let over = unprotected >= UNPROTECTED_TURNS || self.bytes + bytes > self.budget;
            let evict = over && self.by_id.get(&id).is_some_and(|kept| !kept.protected(now));
            if evict {
                self.remove(&id);
                unprotected -= 1;
            } else {
                kept_order.push_back((generation, id));
            }
        }
        self.order = kept_order;
        if self.bytes + bytes > self.budget {
            bail!(
                "the stage keeps {} bytes of receipts that can still be challenged, of {} at \
                 most; this turn gets none",
                self.bytes,
                self.budget
            );
        }
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.by_id.len()
    }

    #[cfg(test)]
    pub(super) fn bytes(&self) -> usize {
        self.bytes
    }
}

/// Bytes a kept receipt holds, about.
fn receipt_bytes(receipt: &Receipt) -> usize {
    let stages: usize = receipt
        .stages
        .iter()
        .map(|stage| 512 + stage.ubatches.len() * size_of::<u32>())
        .sum();
    size_of::<Receipt>()
        + receipt.model.len()
        + receipt.tokens.len() * size_of::<i32>()
        + stages
        + 1024
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use skippy_receipts::wire::{Pick, RECEIPT_VERSION};

    use super::*;

    const VICTIM: [u8; 32] = [1; 32];
    const OTHER: [u8; 32] = [2; 32];

    fn receipt(id: &str, tokens: usize) -> Arc<Receipt> {
        Arc::new(Receipt {
            version: RECEIPT_VERSION,
            receipt_id: id.into(),
            model: "m".into(),
            position_start: 0,
            position_end: tokens as u32,
            tokens: vec![0; tokens],
            generation_start: 0,
            final_token: None,
            stages: Vec::new(),
            binding: None,
        })
    }

    fn keep(turns: &mut KeptTurns, id: &str, key: Option<[u8; 32]>, now: Instant) -> Result<()> {
        let reservation = turns.reserve(id, key, now)?;
        let issued = Outcome::Issued(receipt(id, 4));
        Ok(turns.settle(id, reservation, issued, None, LaterStages::default(), now)?)
    }

    fn challenge(id: &str, layer: u32) -> Challenge {
        Challenge {
            receipt_id: id.into(),
            picks: vec![Pick {
                layer,
                positions: vec![0],
            }],
            proof: None,
            head: Vec::new(),
            attention: Vec::new(),
        }
    }

    // However many turns follow a turn inside its challenge window, its
    // receipt stays; past the window it goes as the oldest.
    #[test]
    fn a_flood_of_turns_does_not_evict_one_inside_its_window() {
        let mut turns = KeptTurns::default();
        let start = Instant::now();
        keep(&mut turns, "victim", Some(VICTIM), start).unwrap();
        for i in 0..4 * UNPROTECTED_TURNS {
            keep(&mut turns, &format!("flood-{i}"), Some(OTHER), start).unwrap();
        }
        assert!(turns.get("victim").is_some());
        assert_eq!(turns.len(), 4 * UNPROTECTED_TURNS + 1, "all inside");
        // the window passed: the oldest go, down to the count kept
        let later = start + CHALLENGE_PIN;
        keep(&mut turns, "next", Some(OTHER), later).unwrap();
        assert!(turns.get("victim").is_none());
        assert_eq!(turns.len(), UNPROTECTED_TURNS);
        assert!(turns.get("next").is_some());
    }

    // An opened turn stays until its rows can no longer be asked for.
    #[test]
    fn an_opened_turn_is_kept_for_its_rows() {
        let mut turns = KeptTurns::default();
        let start = Instant::now();
        keep(&mut turns, "opened", Some(VICTIM), start).unwrap();
        let opened_at = start + CHALLENGE_PIN - Duration::from_secs(1);
        let kept = turns.get_mut("opened").unwrap();
        kept.mark_opened(Some([9; 32]), opened_at);
        let flood = |turns: &mut KeptTurns, now| {
            for i in 0..2 * UNPROTECTED_TURNS {
                keep(turns, &format!("flood-{i}"), None, now).unwrap();
            }
        };
        flood(&mut turns, opened_at + ROWS_PIN - Duration::from_secs(1));
        assert_eq!(turns.get("opened").unwrap().opened, Some([9; 32]));
        flood(&mut turns, opened_at + ROWS_PIN);
        assert!(turns.get("opened").is_none());
    }

    // When turns that can still be challenged fill the budget, a new turn
    // gets no receipt and they keep theirs.
    #[test]
    fn a_full_budget_refuses_new_turns() {
        let start = Instant::now();
        let mut sized = KeptTurns::default();
        keep(&mut sized, "one", Some(OTHER), start).unwrap();
        let mut turns = KeptTurns::new(3 * sized.bytes() + sized.bytes() / 2);
        for id in ["a", "b", "c"] {
            keep(&mut turns, id, Some(OTHER), start).unwrap();
        }
        let error = keep(&mut turns, "d", Some(VICTIM), start).unwrap_err();
        assert!(format!("{error:#}").contains("gets none"), "{error:#}");
        assert!(["a", "b", "c"].iter().all(|id| turns.get(id).is_some()));
        assert!(turns.bytes() <= 3 * sized.bytes() + sized.bytes() / 2);
        // unprotected records make room
        keep(&mut turns, "d", Some(VICTIM), start + CHALLENGE_PIN).unwrap();
        assert!(turns.get("a").is_none() && turns.get("d").is_some());
    }

    // A reused id names the new turn, with none of the old turn's
    // challenges, unless the id is another client's inside its window.
    #[test]
    fn a_reused_id_replaces_only_its_own_clients_turn() {
        let mut turns = KeptTurns::default();
        let start = Instant::now();
        keep(&mut turns, "r", Some(VICTIM), start).unwrap();
        let kept = turns.get_mut("r").unwrap();
        assert!(kept.admit(&challenge("r", 1), 1));
        assert!(!kept.admit(&challenge("r", 2), 1), "one challenge");
        kept.mark_opened(Some([7; 32]), start);

        for shadow in [Some(OTHER), None] {
            let error = turns.reserve("r", shadow, start).unwrap_err();
            assert!(format!("{error:#}").contains("another client"), "{error:#}");
        }
        let kept = turns.get("r").unwrap();
        assert_eq!(kept.client_key, Some(VICTIM));
        assert_eq!(kept.challenges.len(), 1, "the victim's turn is untouched");

        // the same client's new turn under the id starts afresh
        keep(&mut turns, "r", Some(VICTIM), start).unwrap();
        let kept = turns.get_mut("r").unwrap();
        assert!(kept.challenges.is_empty() && kept.opened.is_none());
        assert!(kept.admit(&challenge("r", 2), 1));

        // a turn settles only the record it reserved
        let stale = turns.reserve("r", Some(VICTIM), start).unwrap();
        let current = turns.reserve("r", Some(VICTIM), start).unwrap();
        let committed = |turns: &mut KeptTurns, reservation| {
            let later = LaterStages::default();
            turns.settle("r", reservation, Outcome::Committed, None, later, start)
        };
        assert!(committed(&mut turns, stale).is_err());
        assert_eq!(turns.get("r").unwrap().outcome, Outcome::Committing);
        committed(&mut turns, current).unwrap();

        // past its window an id is anyone's
        let later = start + ROWS_PIN + CHALLENGE_PIN;
        keep(&mut turns, "r", Some(OTHER), later).unwrap();
        assert_eq!(turns.get("r").unwrap().client_key, Some(OTHER));
        assert!(turns.get("never").is_none());
    }
}
