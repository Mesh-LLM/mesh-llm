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
//! A client may challenge a turn for [`CHALLENGE_PIN`], and most do at
//! once. So a turn is held firmly for [`FIRM_PIN`]: nothing but the caps
//! below takes its pin, and no room is made at its cost. After that it
//! lingers: it stays pinned and openable while that costs later requests
//! little, and gives way, oldest first, when lingering turns pin more than
//! [`LINGERING_SHARE`] of a stream or the stage needs the memory they hold.
//! A client that challenges a lingering turn that gave way gets a signed
//! answer that says so.
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

/// How long after its commit an unchallenged turn stays pinned, at most:
/// the client's 10-minute challenge window, and 2 seconds for the response
/// to reach it and the challenge to come back. A client that audits many
/// turns may have a queue of them to challenge.
pub const CHALLENGE_PIN: Duration = Duration::from_secs(602);

/// How long after its commit an unchallenged turn is held whatever room it
/// takes: the 10 seconds in which a client that challenges at once does,
/// and 2 seconds for the response to reach it and the challenge to come
/// back. From then until [`CHALLENGE_PIN`] the turn lingers.
pub const FIRM_PIN: Duration = Duration::from_secs(12);

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
/// The share of a stream's KV cells that lingering turns may pin: what a
/// late challenge is worth to the requests served meanwhile, which read
/// every pinned cell as a masked row.
pub const LINGERING_SHARE: (u64, u64) = (1, 4);

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

    /// Whether the pin holds and nothing may take it for room: a client
    /// challenged the turn, or may still do so at once.
    pub(super) fn firm(&self, now: Instant) -> bool {
        self.holds(now) && (self.challenged() || now < self.committed_at + FIRM_PIN)
    }

    /// Whether the pin holds past [`FIRM_PIN`] with no challenge: it gives
    /// way when what the turn holds is needed.
    pub(super) fn lingers(&self, now: Instant) -> bool {
        self.holds(now) && !self.firm(now)
    }

    /// When the pin next changes by itself, while it holds: it starts to
    /// linger, or runs out.
    fn next_change(&self, now: Instant) -> Option<Instant> {
        let expiry = self.expiry().filter(|expiry| now < *expiry)?;
        let lingers_from = self.committed_at + FIRM_PIN;
        Some(match !self.challenged() && now < lingers_from {
            true => lingers_from.min(expiry),
            false => expiry,
        })
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

    /// Makes the pin `by` older, as if its turn was committed that long ago.
    #[cfg(test)]
    pub(super) fn age(&mut self, by: Duration) {
        self.committed_at = self
            .committed_at
            .checked_sub(by)
            .expect("a clock that started before the turn");
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
        state.shed_lingering_pins(now);
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
            state.shed_lingering_pins(now);
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

    /// When the next pinned turn's pin runs out or starts to linger, so the
    /// stage can pin anew then and not hold the turn's cells until its next
    /// commit. `None` when no turn is pinned.
    pub fn next_pin_expiry(&self, now: Instant) -> Option<Instant> {
        self.lock()
            .committed
            .iter()
            .filter_map(|turn| turn.pin.next_change(now))
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

impl super::RecorderState {
    /// Releases the pins of lingering turns, oldest first, while they pin
    /// more than [`LINGERING_SHARE`] of a stream's cells beside what the
    /// turns held firmly pin: a cell a firm turn pins costs nothing more
    /// when a lingering turn reads it too. A lingering turn none of whose
    /// cells still holds what it wrote is released as well: nothing is left
    /// to pin for it, and it would be looked at again at every pin.
    pub(super) fn shed_lingering_pins(&mut self, now: Instant) {
        let budget = u64::from(self.pin_budget);
        if budget == 0 || !self.committed.iter().any(|turn| turn.pin.lingers(now)) {
            return;
        }
        let mut pinned: BTreeSet<KvCell> = self
            .committed
            .iter()
            .filter(|turn| turn.pin.firm(now))
            .flat_map(|turn| turn_pin_cells(&self.cells, turn))
            .collect();
        let (num, den) = LINGERING_SHARE;
        let mut lingering: BTreeMap<u32, u64> = BTreeMap::new();
        let mut released = Vec::new();
        // newest first, so the turns that fit the share are the newest
        for (index, turn) in self.committed.iter().enumerate().rev() {
            if !turn.pin.lingers(now) {
                continue;
            }
            let cells: Vec<KvCell> = turn_pin_cells(&self.cells, turn).collect();
            let mut added: BTreeMap<u32, u64> = BTreeMap::new();
            for cell in cells.iter().filter(|cell| !pinned.contains(cell)) {
                *added.entry(cell.stream).or_default() += 1;
            }
            let over = added.iter().any(|(stream, more)| {
                (lingering.get(stream).copied().unwrap_or(0) + more) * den > budget * num
            });
            if cells.is_empty() || over {
                released.push(index);
                continue;
            }
            for (stream, more) in added {
                *lingering.entry(stream).or_default() += more;
            }
            pinned.extend(cells);
        }
        for index in released {
            self.committed[index].pin.release();
        }
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
