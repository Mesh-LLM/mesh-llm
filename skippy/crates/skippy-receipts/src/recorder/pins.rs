//! Which KV cells a stage keeps out of later requests' way, so a turn a
//! client may still challenge can be opened.
//!
//! A pinned cell is skipped while a request fits elsewhere, so the cache's
//! attention window grows by the pinned cells and every later request
//! reads them as masked rows. On back-to-back Qwen3-4B requests, pinning
//! every turn for 30 seconds slowed decode by about 5%. So a stage pins a
//! turn only while a client can still challenge it, while a challenge of it
//! waits and is opened, and once opened, until the client has the rows of
//! its attention blocks.
//!
//! Pins are capped, so a client that sends bound long-prompt turns cannot
//! keep the cache pinned at no cost: a client key pins at most
//! [`PINNED_SHARE_PER_KEY`] of a stream's cells, and all turns together
//! [`PINNED_SHARE`]. A new turn always gets its pin, because a turn
//! without a receipt costs an honest node a strike in Mesh. Past a share,
//! the oldest pins give way ([`Recorder::admit_pin`]).

use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

use anyhow::{Result, bail};

use super::cells::CellMirror;
use super::{
    CommittedTurn, KvCell, PrefixCell, Recorder, attention_opening::written_cells, turn, turn_mut,
};
use crate::attention::opening::ROWS_PIN;

/// How long after its commit an unchallenged turn stays pinned: the
/// client's 10-second challenge window, and 2 seconds for the response to
/// reach it and the challenge to come back.
pub const CHALLENGE_PIN: Duration = Duration::from_secs(12);

/// How long a turn stays pinned, at most, from the moment a challenge of
/// it arrives until its opening ends ([`Recorder::hold_for_opening`]): the
/// 15 seconds a client waits for an opening, and 5 for the challenge to
/// arrive and the opening to reach it. A turn of a long prompt replays
/// for seconds, and its opening reads the turn's cells after the replay,
/// so the cells must still hold what the turn wrote then, also past
/// [`CHALLENGE_PIN`].
pub const OPENING_PIN: Duration = Duration::from_secs(20);

/// The share of a stream's KV cells that all pinned turns may hold: a
/// quarter of the cache stays free of pins for the requests being served.
pub const PINNED_SHARE: (u64, u64) = (3, 4);
/// The share of a stream's KV cells the pinned turns of one client key may
/// hold, so one key cannot take the whole of [`PINNED_SHARE`].
pub const PINNED_SHARE_PER_KEY: (u64, u64) = (1, 2);

/// Whether a committed turn's cells are pinned.
#[derive(Clone, Copy, Debug)]
pub(super) struct Pin {
    committed_at: Instant,
    /// When the latest challenge of the turn arrived, while one is being
    /// opened.
    opening_since: Option<Instant>,
    /// The challenges of the turn that hold it for their opening: each
    /// request holds the pin until its own opening ends, so one that is
    /// turned away cannot release the pin of one being replayed.
    openings: u32,
    opened_at: Option<Instant>,
    released: bool,
    /// The client key the turn is bound to, once its pin was admitted.
    client_key: Option<[u8; 32]>,
}

impl Pin {
    pub(super) fn new(committed_at: Instant) -> Self {
        Self {
            committed_at,
            opening_since: None,
            openings: 0,
            opened_at: None,
            released: false,
            client_key: None,
        }
    }

    pub(super) fn holds(&self, now: Instant) -> bool {
        self.expiry().is_some_and(|expiry| now < expiry)
    }

    /// A challenge of the turn arrived: its cells are pinned until the
    /// opening ends, for [`OPENING_PIN`] at most, also when its pin was
    /// released or gave way. A turn already opened keeps the pin of its rows.
    fn opening(&mut self, now: Instant) {
        self.openings = self.openings.saturating_add(1);
        if self.opened_at.is_none() {
            self.opening_since = Some(now);
        }
    }

    /// One challenge's opening ended; the pin holds for the others.
    fn opening_ended(&mut self) {
        self.openings = self.openings.saturating_sub(1);
        if self.openings == 0 {
            self.opening_since = None;
        }
    }

    /// Whether a client challenged the turn: it is being opened, or was.
    fn challenged(&self) -> bool {
        self.opening_since.is_some() || self.opened_at.is_some()
    }

    /// The turn was opened: with attention partials, it stays pinned until
    /// their rows are opened; without, nothing will ask for its rows.
    pub(super) fn opened(&mut self, now: Instant, partials: bool) {
        // an opening still running holds the pin until it ends
        if self.openings == 0 {
            self.opening_since = None;
        }
        if partials {
            self.opened_at = Some(now);
        } else {
            self.released = true;
        }
    }

    pub(super) fn release(&mut self) {
        self.released = true;
    }

    /// When the pin stops holding by itself, while it holds.
    fn expiry(&self) -> Option<Instant> {
        // a turn being opened holds its pin whatever happened to it before
        if let (None, Some(since)) = (self.opened_at, self.opening_since) {
            return Some(since + OPENING_PIN);
        }
        if self.released {
            return None;
        }
        Some(match self.opened_at {
            Some(at) => at + ROWS_PIN,
            None => self.committed_at + CHALLENGE_PIN,
        })
    }
}

impl Recorder {
    /// The cells that a pinned turn's opening would read and that still
    /// hold what the turn read or wrote. The stage pins them, so a later
    /// request writes other cells while it fits there.
    pub fn pinned_cells(&self, now: Instant) -> Vec<KvCell> {
        let mut state = self.lock();
        let pinned: BTreeSet<KvCell> = state
            .committed
            .iter()
            .filter(|t| t.pin.holds(now))
            .flat_map(|turn| turn_pin_cells(&state.cells, turn))
            .collect();
        let cells = pinned.iter().copied().collect();
        state.pins = pinned;
        cells
    }

    /// Sets how many cells a stream of the stage's KV cache has, which the
    /// cells pinned for challenges are a share of ([`Self::admit_pin`]).
    /// Zero, the default, caps nothing.
    pub fn set_pin_budget(&self, cells_per_stream: u32) {
        self.lock().pin_budget = cells_per_stream;
    }

    /// Admits the pin of the committed turn `receipt_id`, bound to
    /// `client_key`. Pinned cells slow every later request and are written
    /// anyway once the cache is full, so a client key pins at most
    /// [`PINNED_SHARE_PER_KEY`] of a stream, and every key together
    /// [`PINNED_SHARE`]. The new turn always gets its pin: a serving node
    /// owes its client a receipt it can open. Older pins give way for it
    /// ([`pins_to_release`]), and a turn that is past a share alone is
    /// pinned alone.
    pub fn admit_pin(&self, receipt_id: &str, client_key: [u8; 32], now: Instant) -> Result<()> {
        let mut state = self.lock();
        let budget = state.pin_budget;
        let Some(turn) = turn(&state.committed, receipt_id) else {
            bail!("receipt {receipt_id} is not committed");
        };
        if budget > 0 {
            let own: BTreeSet<KvCell> = turn_pin_cells(&state.cells, turn).collect();
            let mut held: Vec<HeldPin> = state
                .committed
                .iter()
                .filter(|t| t.receipt_id != receipt_id && t.pin.holds(now))
                .map(|other| HeldPin {
                    receipt_id: other.receipt_id.clone(),
                    opened: other.pin.challenged(),
                    committed_at: other.pin.committed_at,
                    same_key: other.pin.client_key == Some(client_key),
                    cells: turn_pin_cells(&state.cells, other).collect(),
                })
                .collect();
            // a stable sort: turns committed at the same instant keep their order
            held.sort_by_key(|pin| (pin.opened, pin.committed_at));
            for released in pins_to_release(&own, held, budget) {
                if let Some(turn) = turn_mut(&mut state.committed, &released) {
                    turn.pin.release();
                }
            }
            state.drop_unpinned_states(now);
        }
        if let Some(turn) = turn_mut(&mut state.committed, receipt_id) {
            turn.pin.client_key = Some(client_key);
        }
        Ok(())
    }

    /// The stage could not pin the cells [`Self::pinned_cells`] named: as far
    /// as a write the recorder does not see goes, no cell is pinned.
    pub fn pins_not_applied(&self) {
        self.lock().pins.clear();
    }

    /// When the next pinned turn's pin runs out, so the stage can pin anew
    /// then and not hold the turn's cells until its next commit. `None`
    /// when no turn is pinned.
    pub fn next_pin_expiry(&self, now: Instant) -> Option<Instant> {
        self.lock()
            .committed
            .iter()
            .filter(|turn| turn.pin.holds(now))
            .filter_map(|turn| turn.pin.expiry())
            .min()
    }

    /// Stops pinning a turn's cells: no client can challenge it, or the
    /// client has its rows.
    pub fn release_pin(&self, receipt_id: &str) {
        let mut state = self.lock();
        if let Some(turn) = turn_mut(&mut state.committed, receipt_id) {
            turn.pin.release();
        }
        state.drop_unpinned_states(Instant::now());
    }

    /// A challenge of the turn `receipt_id` arrived at `now`: its cells are
    /// pinned until [`Self::end_opening`], for [`OPENING_PIN`] at most,
    /// whether or not its own pin still holds. The cells it pins are
    /// those that still hold what the turn read or wrote.
    pub fn hold_for_opening(&self, receipt_id: &str, now: Instant) {
        if let Some(turn) = turn_mut(&mut self.lock().committed, receipt_id) {
            turn.pin.opening(now);
        }
    }

    /// The opening of the turn `receipt_id` ended. An opening that read
    /// attention partials pinned the turn for its rows; any other leaves the
    /// turn its challenge pin, which has most likely run out.
    pub fn end_opening(&self, receipt_id: &str) {
        let mut state = self.lock();
        if let Some(turn) = turn_mut(&mut state.committed, receipt_id) {
            turn.pin.opening_ended();
        }
        state.drop_unpinned_states(Instant::now());
    }
}

/// The cells of `turn` an opening reads that still hold what it read or
/// wrote: its prefix cells and the cells it wrote last.
fn turn_pin_cells<'a>(
    mirror: &'a CellMirror,
    turn: &'a CommittedTurn,
) -> impl Iterator<Item = KvCell> + 'a {
    let written = written_cells(turn);
    let stream = mirror.streams().of(turn.seq);
    let own = turn
        .cells
        .iter()
        .zip(turn.position_start..)
        .filter_map(move |(c, pos)| {
            Some(PrefixCell {
                pos: i32::try_from(pos).ok()?,
                stream,
                cell: *written.get(&pos)?,
                digest: c.digest,
            })
        });
    turn.prefix
        .cells
        .iter()
        .copied()
        .chain(own)
        .filter(|c| mirror.holds(c))
        .map(|c| KvCell {
            stream: c.stream,
            cell: c.cell,
        })
}

/// A pin that holds beside the turn being admitted.
struct HeldPin {
    receipt_id: String,
    /// The turn was opened and waits for its rows request.
    opened: bool,
    committed_at: Instant,
    /// Bound to the client key of the turn being admitted.
    same_key: bool,
    cells: Vec<KvCell>,
}

/// The pins that give way so the turn that pins `own` fits the shares of
/// `budget`, out of `held` in the order they give way: turns no client
/// has challenged before turns that wait for their rows request, and the
/// oldest first. Most turns are never challenged, and a client that
/// challenges does so soon after the response, so the oldest unchallenged
/// turn is the one least likely to be opened. A released turn stays
/// openable until a ubatch writes its cells.
///
/// Past its key's share, the pins of that key give way; past the share of
/// all keys, the pins of any key. A turn that is past a share with no
/// other pin left keeps its own.
fn pins_to_release(own: &BTreeSet<KvCell>, mut held: Vec<HeldPin>, budget: u32) -> Vec<String> {
    let mut released = Vec::new();
    loop {
        let mut by_key = own.clone();
        let mut all = own.clone();
        for pin in &held {
            if pin.same_key {
                by_key.extend(pin.cells.iter().copied());
            }
            all.extend(pin.cells.iter().copied());
        }
        let of_key = over_share(&by_key, budget, PINNED_SHARE_PER_KEY)
            .and_then(|_| held.iter().position(|pin| pin.same_key));
        let next = of_key.or_else(|| {
            over_share(&all, budget, PINNED_SHARE).and_then(|_| (!held.is_empty()).then_some(0))
        });
        let Some(index) = next else {
            return released;
        };
        released.push(held.remove(index).receipt_id);
    }
}

/// The most cells `cells` pins in one stream, when that is past `share` of
/// the stream's `budget` cells.
fn over_share(cells: &BTreeSet<KvCell>, budget: u32, (num, den): (u64, u64)) -> Option<usize> {
    let mut per_stream: BTreeMap<u32, usize> = BTreeMap::new();
    for cell in cells {
        *per_stream.entry(cell.stream).or_default() += 1;
    }
    let most = per_stream.into_values().max()?;
    (most as u64 * den > u64::from(budget) * num).then_some(most)
}
