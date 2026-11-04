//! Replays the reasoning-budget sampler of a turn, to know which of its
//! tokens the sampler forced.
//!
//! A thinking model gets a reasoning budget on ordinary requests. Its
//! sampler lets every token through until the model has reasoned for the
//! budget's tokens, then forces the tag that ends the reasoning, one token
//! at a time: it leaves the forced token as the only candidate. A forced
//! token is not the choice of the logits, so the head check must know
//! where the sampler forced one. Everywhere else the token is judged as in
//! a turn without a budget.
//!
//! The sampler's state follows from what it was built from and the tokens
//! the turn sampled, which the receipt names. The node states the first in
//! its sampler statement ([`BudgetStatement`]): the budget, the tag that
//! starts the reasoning, the tags that end it, and the text the chat
//! template placed after the prompt, which the sampler reads first. The
//! client turns those texts into tokens with the claimed model's own
//! tokenizer ([`Vocabulary`]) and runs the same state machine
//! ([`replay`]).
//!
//! The machine is a port of `common/reasoning-budget.cpp` of the pinned
//! llama.cpp, and of the prefill rule of `common_sampler_init` in
//! `common/sampling.cpp`. Skippy builds the sampler in
//! `src/skippy/sampling_config.cpp` and never ends the reasoning from
//! outside it. A change of those files needs the same change here.
//!
//! What stays open: the tags are the node's statement. The client does not
//! derive them from the chat template, so a node can state an end tag of
//! its own and have its tokens pass as forced. The statement bounds the
//! tags ([`MAX_TAG_BYTES`], [`MAX_END_TAGS`]), and a report shows them.

use std::collections::BTreeMap;

use anyhow::Result;

use crate::turn;
use crate::wire::Receipt;

/// The most bytes a stated start or end tag may hold.
pub const MAX_TAG_BYTES: usize = 256;
/// The most end tags a statement may name.
pub const MAX_END_TAGS: usize = 8;
/// The most bytes the stated prefill may hold: the text a chat template
/// places after the prompt for the model to continue.
pub const MAX_PREFILL_BYTES: usize = 1024;

/// What a sampler statement says the node built the turn's reasoning-budget
/// sampler from. These are the node's words.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BudgetStatement {
    /// The tokens of reasoning the sampler allows before it forces the end.
    pub budget: i32,
    /// The tag that starts the reasoning.
    pub start: String,
    /// The tags that end it. The sampler forces the first.
    pub ends: Vec<String>,
    /// The text the chat template placed after the prompt.
    pub prefill: String,
}

impl BudgetStatement {
    /// The sampler the statement `sampler` names: `Ok(None)` when it names
    /// none, and the reason when what it names is outside the limits.
    pub fn from_sampler(sampler: &str) -> Result<Option<Self>, String> {
        match turn::sampler_parameters(sampler) {
            Some(parameters) => Self::from_parameters(&parameters),
            None => Ok(None),
        }
    }

    /// Checks that the stated reasoning sampler, if any, has the types an
    /// honest node writes: an object with a text start tag, a list of text
    /// end tags and a text prefill or none. A statement past the limits is
    /// not malformed; its tokens are left unjudged ([`Self::from_parameters`]).
    pub(crate) fn check_types(p: &serde_json::Value) -> Result<(), String> {
        let stated = &p[turn::REASONING];
        if stated.is_null() {
            return Ok(());
        }
        let texts = stated["ends"]
            .as_array()
            .is_some_and(|ends| ends.iter().all(serde_json::Value::is_string));
        let well_typed = stated.is_object()
            && stated["start"].is_string()
            && texts
            && (stated["prefill"].is_null() || stated["prefill"].is_string());
        if well_typed {
            Ok(())
        } else {
            Err("the stated reasoning sampler is not a start tag, end tags and a prefill".into())
        }
    }

    /// [`Self::from_sampler`] for the parameters of a statement.
    pub(crate) fn from_parameters(p: &serde_json::Value) -> Result<Option<Self>, String> {
        let stated = &p[turn::REASONING];
        if stated.is_null() {
            return Ok(None);
        }
        let Some(budget) = p["reasoning_budget"].as_i64().filter(|b| *b >= 0) else {
            return Err("the statement names reasoning tags and no reasoning budget".into());
        };
        let text = |value: &serde_json::Value, what: &str, most: usize| {
            let text = value
                .as_str()
                .ok_or_else(|| format!("the stated {what} is not text"))?;
            if text.len() > most {
                return Err(format!(
                    "the stated {what} holds {} bytes, more than {most}",
                    text.len()
                ));
            }
            Ok(text.to_owned())
        };
        let start = text(
            &stated["start"],
            "start tag of the reasoning",
            MAX_TAG_BYTES,
        )?;
        if start.is_empty() {
            return Err(
                "the statement names a reasoning budget sampler without a start tag".into(),
            );
        }
        let ends = stated["ends"]
            .as_array()
            .ok_or("the statement names no end tags of the reasoning")?;
        if ends.is_empty() || ends.len() > MAX_END_TAGS {
            return Err(format!(
                "the statement names {} end tags of the reasoning; it may name 1 to {MAX_END_TAGS}",
                ends.len()
            ));
        }
        let ends = ends
            .iter()
            .map(|end| text(end, "end tag of the reasoning", MAX_TAG_BYTES))
            .collect::<Result<Vec<_>, _>>()?;
        if ends.iter().any(String::is_empty) {
            return Err("the statement names an empty end tag of the reasoning".into());
        }
        let prefill = match &stated["prefill"] {
            serde_json::Value::Null => String::new(),
            value => text(value, "prefill", MAX_PREFILL_BYTES)?,
        };
        Ok(Some(Self {
            budget: i32::try_from(budget).unwrap_or(i32::MAX),
            start,
            ends,
            prefill,
        }))
    }
}

/// The claimed model's tokenizer, as the replay needs it.
pub trait Vocabulary {
    /// The tokens of `text`, with special tokens parsed and none added, as
    /// the runtime reads a tag.
    fn tokenize(&self, text: &str) -> Result<Vec<i32>>;
    /// The bytes of one token's text. A control token gives its name.
    fn piece(&self, token: i32) -> Result<Vec<u8>>;
}

/// What the reasoning budget of a turn forced, for the head check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Forcing {
    /// The statement names no reasoning-budget sampler: no token is forced.
    Free,
    /// The sampler was replayed.
    Replayed(Replayed),
    /// The sampler could not be replayed, for this reason, so the turn's
    /// tokens are not judged.
    Unjudged(String),
}

/// A replayed reasoning-budget sampler.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Replayed {
    /// What the node says the sampler was built from.
    pub statement: BudgetStatement,
    /// By position, the token the sampler forced on the token sampled
    /// from it.
    forced: BTreeMap<u32, i32>,
}

impl Replayed {
    /// The token the sampler forced on the token sampled from `position`.
    pub fn forced_at(&self, position: u32) -> Option<i32> {
        self.forced.get(&position).copied()
    }
}

impl Forcing {
    /// The token the sampler forced on the token sampled from `position`.
    pub(crate) fn forced_at(&self, position: u32) -> Option<i32> {
        match self {
            Self::Replayed(replayed) => replayed.forced_at(position),
            Self::Free | Self::Unjudged(_) => None,
        }
    }

    /// Why the turn's tokens are not judged, when the sampler could not be
    /// replayed.
    pub fn unjudged(&self) -> Option<&str> {
        match self {
            Self::Unjudged(reason) => Some(reason),
            Self::Free | Self::Replayed(_) => None,
        }
    }
}

#[cfg(test)]
impl Forcing {
    /// A replayed sampler that forced these tokens, by position.
    pub(crate) fn forced(forced: &[(u32, i32)]) -> Self {
        Self::Replayed(Replayed {
            statement: BudgetStatement {
                budget: 0,
                start: "<think>".into(),
                ends: vec!["</think>".into()],
                prefill: String::new(),
            },
            forced: forced.iter().copied().collect(),
        })
    }
}

/// Replays the reasoning-budget sampler `receipt`'s statement names over
/// the tokens the turn sampled. Without a `vocabulary` a statement that
/// names a sampler leaves the tokens unjudged. A tokenizer that fails is
/// the client's own, and leaves them unjudged too.
pub fn replay(receipt: &Receipt, vocabulary: Option<&dyn Vocabulary>) -> Forcing {
    let Some(binding) = receipt.binding.as_ref() else {
        return Forcing::Free;
    };
    let statement = match BudgetStatement::from_sampler(&binding.sampler) {
        Ok(Some(statement)) => statement,
        Ok(None) => return Forcing::Free,
        Err(reason) => return Forcing::Unjudged(reason),
    };
    let Some(vocabulary) = vocabulary else {
        return Forcing::Unjudged(
            "the check has no tokenizer to replay the reasoning budget with".into(),
        );
    };
    match forced_tokens(&statement, receipt, vocabulary) {
        Ok(forced) => Forcing::Replayed(Replayed { statement, forced }),
        Err(error) => Forcing::Unjudged(format!(
            "the claimed model's tokenizer cannot replay the reasoning budget: {error:#}"
        )),
    }
}

/// By position, the tokens the sampler of `statement` forced in the turn
/// of `receipt`.
fn forced_tokens(
    statement: &BudgetStatement,
    receipt: &Receipt,
    vocabulary: &dyn Vocabulary,
) -> Result<BTreeMap<u32, i32>> {
    let start = vocabulary.tokenize(&statement.start)?;
    let mut ends = Vec::with_capacity(statement.ends.len());
    for end in &statement.ends {
        let tokens = vocabulary.tokenize(end)?;
        if !tokens.is_empty() {
            ends.push(tokens);
        }
    }
    // the runtime builds no sampler from a tag that is no token
    if start.is_empty() || ends.is_empty() {
        return Ok(BTreeMap::new());
    }
    let forced = ends[0].clone();
    let mut machine = Machine::new(vec![start], ends, forced, statement.budget);
    let mut whole = |token: i32| Ok(ends_a_character(&vocabulary.piece(token)?));
    for token in prefill_tokens(&statement.prefill, vocabulary)? {
        machine.accept(token, &mut whole)?;
    }
    let mut out = BTreeMap::new();
    // the first sampled token comes from the last position of the prompt
    let mut position = receipt.generation_start.saturating_sub(1);
    for token in receipt.sampled_tokens() {
        if let Some(forced) = machine.forcing() {
            out.insert(position, forced);
        }
        machine.accept(token, &mut whole)?;
        position = position.saturating_add(1);
    }
    Ok(out)
}

/// The tokens of the prefill the sampler reads before the turn's own: a
/// tokenizer may put a space before the first special token, which the
/// runtime leaves out when the text starts with none.
fn prefill_tokens(prefill: &str, vocabulary: &dyn Vocabulary) -> Result<Vec<i32>> {
    let mut tokens = vocabulary.tokenize(prefill)?;
    let starts_with_space = |bytes: &[u8]| bytes.first().is_some_and(|&byte| is_c_space(byte));
    if let Some(&first) = tokens.first()
        && starts_with_space(&vocabulary.piece(first)?)
        && !starts_with_space(prefill.as_bytes())
    {
        tokens.remove(0);
    }
    Ok(tokens)
}

/// `isspace` of the C locale.
fn is_c_space(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

/// Whether text that ends with `piece` ends with a whole UTF-8 character.
/// The runtime reads a control token's piece as empty here, and the client
/// reads its name; both end a character, since a name is whole text.
fn ends_a_character(piece: &[u8]) -> bool {
    if piece.is_empty() {
        return true;
    }
    for offset in 1..=piece.len().min(4) {
        let byte = piece[piece.len() - offset];
        if byte & 0xc0 != 0x80 {
            let expected = match byte {
                0xf0.. => 4,
                0xe0.. => 3,
                0xc0.. => 2,
                _ => 1,
            };
            return offset >= expected;
        }
    }
    false
}

/// Finds the token sequences of a tag at the end of the tokens it is
/// given.
#[derive(Clone, Debug)]
struct Matcher {
    sequences: Vec<Vec<i32>>,
    recent: Vec<i32>,
    longest: usize,
}

impl Matcher {
    fn new(sequences: Vec<Vec<i32>>) -> Self {
        let mut kept: Vec<Vec<i32>> = Vec::new();
        for sequence in sequences {
            if !sequence.is_empty() && !kept.contains(&sequence) {
                kept.push(sequence);
            }
        }
        let longest = kept.iter().map(Vec::len).max().unwrap_or(0);
        Self {
            sequences: kept,
            recent: Vec::new(),
            longest,
        }
    }

    /// Whether one of the sequences ends at `token`. A match forgets the
    /// tokens before it.
    fn advance(&mut self, token: i32) -> bool {
        self.recent.push(token);
        if self.recent.len() > self.longest {
            self.recent.remove(0);
        }
        let matched = self
            .sequences
            .iter()
            .any(|sequence| self.recent.ends_with(sequence));
        if matched {
            self.recent.clear();
        }
        matched
    }

    fn reset(&mut self) {
        self.recent.clear();
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    /// Waiting for the start tag.
    Idle,
    /// Counting the tokens of the reasoning down.
    Counting,
    /// The budget ran out inside a character: waiting for its end.
    WaitingForCharacter,
    /// Forcing the end tag, token by token.
    Forcing,
    /// The reasoning ended. A new start tag counts again.
    Done,
}

/// The state machine of the reasoning-budget sampler.
#[derive(Clone, Debug)]
struct Machine {
    start: Matcher,
    end: Matcher,
    forced: Vec<i32>,
    budget: i32,
    remaining: i32,
    state: State,
    /// The next token of `forced` to force.
    force_position: usize,
    /// Whether the sampler forced a token since it began to force. Tokens
    /// it reads without sampling them, as the prefill, set none.
    forcing_started: bool,
}

impl Machine {
    fn new(start: Vec<Vec<i32>>, ends: Vec<Vec<i32>>, forced: Vec<i32>, budget: i32) -> Self {
        Self {
            start: Matcher::new(start),
            end: Matcher::new(ends),
            forced,
            budget,
            remaining: budget,
            state: State::Idle,
            force_position: 0,
            forcing_started: false,
        }
    }

    /// The token the sampler forces on the next sampled token, if it
    /// forces one. The runtime asks this before it samples each token.
    fn forcing(&mut self) -> Option<i32> {
        if self.state != State::Forcing {
            return None;
        }
        let forced = self.forced.get(self.force_position).copied()?;
        self.forcing_started = true;
        Some(forced)
    }

    fn begin_forcing(&mut self) {
        self.state = State::Forcing;
        self.force_position = 0;
        self.forcing_started = false;
    }

    /// The reasoning starts: count its tokens, or force its end at once
    /// when the budget allows none.
    fn begin_counting(&mut self) {
        self.state = State::Counting;
        self.remaining = self.budget;
        if self.remaining <= 0 {
            self.begin_forcing();
        }
    }

    /// Reads one token: a prefill token, or one the turn sampled. `whole`
    /// says whether a token's text ends a character.
    fn accept(&mut self, token: i32, whole: &mut dyn FnMut(i32) -> Result<bool>) -> Result<()> {
        match self.state {
            State::Idle => {
                if self.start.advance(token) {
                    self.begin_counting();
                }
            }
            State::Counting | State::WaitingForCharacter => {
                if self.end.advance(token) {
                    self.state = State::Done;
                    return Ok(());
                }
                if self.state == State::WaitingForCharacter {
                    if whole(token)? {
                        self.begin_forcing();
                        self.end.reset();
                    }
                    return Ok(());
                }
                self.remaining -= 1;
                if self.remaining <= 0 {
                    if whole(token)? {
                        self.begin_forcing();
                    } else {
                        self.state = State::WaitingForCharacter;
                    }
                    self.end.reset();
                }
            }
            State::Forcing => {
                let ended = self.end.advance(token);
                self.force_position += 1;
                // tokens the sampler only reads can end the reasoning
                // early; tokens it forces must also reach the tag's end
                if (!self.forcing_started && ended) || self.force_position >= self.forced.len() {
                    self.state = State::Done;
                }
            }
            State::Done => {
                if self.start.advance(token) {
                    self.end.reset();
                    self.begin_counting();
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
