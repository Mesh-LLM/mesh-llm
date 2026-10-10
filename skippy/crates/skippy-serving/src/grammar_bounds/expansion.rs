//! Bounds on how llama.cpp expands and checks a GBNF grammar.
//!
//! Initializing a grammar natively expands every repetition into copies and
//! generated rules, then walks leftmost rule references to detect left
//! recursion. Neither step has a limit, and small grammars made both blow up:
//!
//! - `x{2000}` copies `x` inline 2000 times and `x{0,2000}` adds 2000 rules,
//!   so a 100 KB grammar expanded to 3.6 GB. `x{5,3}` underflows the count of
//!   generated rules and allocates until memory runs out.
//! - The walk recurses once per leftmost rule reference, so a long chain of
//!   rules that each start with the next overflows the stack.
//! - The walk re-enters a rule for every leftmost reference to it, so rules
//!   like `r1 ::= "" | r2 r2` double the work per level and never finish.
//!
//! This module mirrors the native parser closely enough to measure those
//! costs without building the expanded grammar, and rejects a grammar that
//! exceeds them before it reaches native code. A grammar the native parser
//! rejects anyway is left for it to report.

use std::collections::HashMap;

/// Most rules a grammar may expand to, counting generated rules.
pub(super) const MAX_GRAMMAR_RULES: u64 = 1 << 17;
/// Most native grammar elements a grammar may expand to.
pub(super) const MAX_GRAMMAR_ELEMENTS: u64 = 1 << 22;
/// Deepest chain of leftmost rule references the left recursion check may
/// recurse through.
pub(super) const MAX_LEFT_REFERENCE_DEPTH: u32 = 4096;
/// Most rule visits the left recursion check may make. Native initialization
/// time grows faster than the visit count; at this limit it takes about
/// 80 ms, while ordinary grammars need a few hundred visits.
pub(super) const MAX_LEFT_REFERENCE_VISITS: u64 = 1 << 15;

/// llama.cpp's `MAX_REPETITION_THRESHOLD`.
const NATIVE_MAX_REPETITIONS: u64 = 2000;

/// Returns an error when initializing `grammar` natively would exceed one of
/// the expansion or left recursion limits.
pub(super) fn check_grammar_expansion(grammar: &str) -> Result<(), String> {
    match measure(grammar) {
        Ok(_) | Err(Stop::NativeRejects) => Ok(()),
        Err(Stop::Rejected(message)) => Err(message),
    }
}

/// Native costs of initializing a grammar the native parser accepts.
#[derive(Debug, PartialEq, Eq)]
struct GrammarCost {
    rules: u64,
    elements: u64,
    left_reference_depth: u32,
    left_reference_visits: u64,
}

#[derive(Debug, PartialEq, Eq)]
enum Stop {
    /// The native parser or initializer rejects the grammar before the
    /// expensive step, so there is nothing to bound.
    NativeRejects,
    /// The grammar is rejected here, before native code sees it.
    Rejected(String),
}

type Parsed<T> = Result<T, Stop>;

fn measure(grammar: &str) -> Parsed<GrammarCost> {
    // Native code reads the grammar as a C string.
    let source = grammar.as_bytes();
    let source = &source[..source.iter().position(|&b| b == 0).unwrap_or(source.len())];
    let mut parser = Parser::new(source);
    parser.parse()?;
    let rules = parser.finish()?;
    let (left_reference_depth, left_reference_visits) = walk_left_references(&rules)?;
    Ok(GrammarCost {
        rules: parser.rule_count,
        elements: parser.elements,
        left_reference_depth,
        left_reference_visits,
    })
}

/// A native element of the rule being built: a rule reference, or a run of
/// terminal elements counted by how many native elements it holds.
#[derive(Clone, Copy, Debug)]
enum Element {
    Ref(u32),
    Terminals(u64),
}

impl Element {
    fn native_len(self) -> u64 {
        match self {
            Self::Ref(_) => 1,
            Self::Terminals(count) => count,
        }
    }
}

/// What the left recursion check sees of one alternative: the rule
/// references before its first terminal, and whether it is empty.
#[derive(Debug, Default)]
struct Alternative {
    leftmost: Vec<u32>,
    blocked: bool,
    native_len: u64,
    /// Highest rule id the alternative references anywhere.
    max_reference: Option<u32>,
}

impl Alternative {
    fn push(&mut self, element: Element) {
        self.native_len += element.native_len();
        if let Element::Ref(rule) = element {
            self.max_reference = self.max_reference.max(Some(rule));
        }
        if self.blocked {
            return;
        }
        match element {
            Element::Ref(rule) => self.leftmost.push(rule),
            Element::Terminals(_) => self.blocked = true,
        }
    }

    fn extend(&mut self, elements: &[Element]) {
        for &element in elements {
            self.push(element);
        }
    }
}

#[derive(Debug)]
struct Rule {
    alternatives: Vec<Alternative>,
}

impl Rule {
    fn may_be_empty(&self) -> bool {
        self.alternatives
            .iter()
            .any(|alternative| alternative.native_len == 0)
    }
}

struct Parser<'a> {
    source: &'a [u8],
    /// Rule names from the source, mapped to their ids.
    symbols: HashMap<&'a [u8], u32>,
    /// Rules generated for groups and repetitions.
    generated_rules: u32,
    rules: Vec<Option<Rule>>,
    rule_count: u64,
    elements: u64,
}

impl<'a> Parser<'a> {
    fn new(source: &'a [u8]) -> Self {
        Self {
            source,
            symbols: HashMap::new(),
            generated_rules: 0,
            rules: Vec::new(),
            rule_count: 0,
            elements: 0,
        }
    }

    fn at(&self, pos: usize) -> u8 {
        self.source.get(pos).copied().unwrap_or(0)
    }

    /// Size of the native `symbol_ids` map, which is also the next free id.
    fn symbol_count(&self) -> u32 {
        u32::try_from(self.symbols.len())
            .unwrap_or(u32::MAX)
            .saturating_add(self.generated_rules)
    }

    fn symbol_id(&mut self, name: &'a [u8]) -> u32 {
        if let Some(&id) = self.symbols.get(name) {
            return id;
        }
        let id = self.symbol_count();
        self.symbols.insert(name, id);
        id
    }

    /// Mirrors `generate_symbol_id`. Generated names are `<rule>_<id>`, which
    /// no source name can spell because `_` is not a name character, so they
    /// only need counting.
    fn generated_symbol_id(&mut self) -> u32 {
        let id = self.symbol_count();
        self.generated_rules += 1;
        id
    }

    fn add_rule(&mut self, id: u32, rule: Rule) -> Parsed<()> {
        let index = id as usize;
        if self.rules.len() <= index {
            self.rules.resize_with(index + 1, || None);
        }
        if self.rules[index].replace(rule).is_none() {
            self.rule_count += 1;
            if self.rule_count > MAX_GRAMMAR_RULES {
                return Err(too_large(format!(
                    "grammar expands to more than {MAX_GRAMMAR_RULES} rules"
                )));
            }
        }
        Ok(())
    }

    fn add_elements(&mut self, count: u64) -> Parsed<()> {
        self.elements = self.elements.saturating_add(count);
        if self.elements > MAX_GRAMMAR_ELEMENTS {
            return Err(too_large(format!(
                "grammar expands to more than {MAX_GRAMMAR_ELEMENTS} elements"
            )));
        }
        Ok(())
    }

    fn parse(&mut self) -> Parsed<()> {
        let mut pos = self.parse_space(0, true);
        while self.at(pos) != 0 {
            pos = self.parse_rule(pos)?;
        }
        Ok(())
    }

    /// Mirrors the checks that run before the left recursion check: every
    /// rule slot is defined, every reference resolves, and `root` exists.
    fn finish(&mut self) -> Parsed<Vec<Rule>> {
        let rules = std::mem::take(&mut self.rules)
            .into_iter()
            .map(|rule| rule.ok_or(Stop::NativeRejects))
            .collect::<Parsed<Vec<Rule>>>()?;
        let max_reference = rules
            .iter()
            .flat_map(|rule| &rule.alternatives)
            .filter_map(|alternative| alternative.max_reference)
            .max();
        if rules.is_empty() || max_reference.is_some_and(|rule| rule as usize >= rules.len()) {
            return Err(Stop::NativeRejects);
        }
        let Some(&root) = self.symbols.get(b"root".as_slice()) else {
            return Err(Stop::NativeRejects);
        };
        // A `root` that only appears inside an item repeated zero times
        // passes the native parser without a rule, and native initialization
        // then reads past the end of its rule list.
        if root as usize >= rules.len() {
            return Err(Stop::Rejected(
                "grammar does not define its root rule".to_string(),
            ));
        }
        Ok(rules)
    }

    fn parse_rule(&mut self, pos: usize) -> Parsed<usize> {
        let name_end = self.parse_name(pos)?;
        let source = self.source;
        let name = &source[pos..name_end];
        let mut pos = self.parse_space(name_end, false);
        let rule_id = self.symbol_id(name);
        if !(self.at(pos) == b':' && self.at(pos + 1) == b':' && self.at(pos + 2) == b'=') {
            return Err(Stop::NativeRejects);
        }
        pos = self.parse_space(pos + 3, true);
        pos = self.parse_alternates(pos, rule_id, false)?;
        match self.at(pos) {
            b'\r' => pos += if self.at(pos + 1) == b'\n' { 2 } else { 1 },
            b'\n' => pos += 1,
            0 => {}
            _ => return Err(Stop::NativeRejects),
        }
        Ok(self.parse_space(pos, true))
    }

    fn parse_alternates(&mut self, pos: usize, rule_id: u32, nested: bool) -> Parsed<usize> {
        let mut alternatives = Vec::new();
        let (mut pos, first) = self.parse_sequence(pos, nested)?;
        alternatives.push(first);
        while self.at(pos) == b'|' {
            pos = self.parse_space(pos + 1, true);
            let (next, alternative) = self.parse_sequence(pos, nested)?;
            pos = next;
            alternatives.push(alternative);
        }
        // One ALT between alternatives and one END after the last.
        self.add_elements(alternatives.len() as u64)?;
        self.add_rule(rule_id, Rule { alternatives })?;
        Ok(pos)
    }

    fn parse_sequence(&mut self, pos: usize, nested: bool) -> Parsed<(usize, Alternative)> {
        let mut sequence = Sequence {
            prior_rules: 1,
            ..Sequence::default()
        };
        let mut pos = pos;
        while let Some(next) = self.parse_item(pos, nested, &mut sequence)? {
            pos = next;
        }
        sequence.alternative.extend(&sequence.item);
        Ok((pos, sequence.alternative))
    }

    /// Parses one item or repetition operator, or returns `None` where the
    /// sequence ends.
    fn parse_item(
        &mut self,
        pos: usize,
        nested: bool,
        sequence: &mut Sequence,
    ) -> Parsed<Option<usize>> {
        let next = match self.at(pos) {
            b'"' => {
                let (end, chars) = self.parse_literal(pos + 1)?;
                self.start_terminal_item(sequence, chars)?;
                self.parse_space(end, nested)
            }
            b'[' => {
                let (end, chars) = self.parse_char_class(pos + 1)?;
                self.start_terminal_item(sequence, chars)?;
                self.parse_space(end, nested)
            }
            b'<' | b'!' => {
                let start = if self.at(pos) == b'!' { pos + 1 } else { pos };
                let end = self.parse_token(start)?;
                self.start_terminal_item(sequence, 1)?;
                self.parse_space(end, nested)
            }
            byte if is_word_char(byte) => {
                let name_end = self.parse_name(pos)?;
                let source = self.source;
                let rule = self.symbol_id(&source[pos..name_end]);
                self.start_item(sequence, vec![Element::Ref(rule)])?;
                self.parse_space(name_end, nested)
            }
            b'(' => {
                let end = self.parse_group(pos, sequence)?;
                self.parse_space(end, nested)
            }
            b'.' => {
                self.start_terminal_item(sequence, 1)?;
                self.parse_space(pos + 1, nested)
            }
            b'*' | b'+' | b'?' => {
                let (min, max) = match self.at(pos) {
                    b'*' => (0, None),
                    b'+' => (1, None),
                    _ => (0, Some(1)),
                };
                let end = self.parse_space(pos + 1, nested);
                self.repeat(sequence, min, max)?;
                end
            }
            b'{' => {
                let (end, min, max) = self.parse_repetition_bounds(pos, nested)?;
                self.repeat(sequence, min, max)?;
                end
            }
            _ => return Ok(None),
        };
        Ok(Some(next))
    }

    /// Parses the body of a `"..."` literal, returning the position after the
    /// closing quote and how many characters it holds.
    fn parse_literal(&self, mut pos: usize) -> Parsed<(usize, u64)> {
        let mut chars = 0;
        while self.at(pos) != b'"' {
            pos = self.parse_char(pos)?;
            chars += 1;
        }
        Ok((pos + 1, chars))
    }

    /// Parses the body of a `[...]` class, returning the position after the
    /// closing bracket and how many native elements it holds.
    fn parse_char_class(&self, mut pos: usize) -> Parsed<(usize, u64)> {
        if self.at(pos) == b'^' {
            pos += 1;
        }
        let mut elements = 0;
        while self.at(pos) != b']' {
            pos = self.parse_char(pos)?;
            elements += 1;
            if self.at(pos) == b'-' && self.at(pos + 1) != b']' {
                pos = self.parse_char(pos + 1)?;
                elements += 1;
            }
        }
        Ok((pos + 1, elements))
    }

    /// Parses a `( ... )` group into a generated rule and returns the
    /// position after the closing parenthesis.
    fn parse_group(&mut self, pos: usize, sequence: &mut Sequence) -> Parsed<usize> {
        let pos = self.parse_space(pos + 1, true);
        let symbols_before = self.symbol_count();
        let group = self.generated_symbol_id();
        let pos = self.parse_alternates(pos, group, true)?;
        self.start_item(sequence, vec![Element::Ref(group)])?;
        // A group stands for every rule its body generated.
        sequence.prior_rules = u64::from(self.symbol_count() - symbols_before).max(1);
        if self.at(pos) != b')' {
            return Err(Stop::NativeRejects);
        }
        Ok(pos + 1)
    }

    fn start_terminal_item(&mut self, sequence: &mut Sequence, chars: u64) -> Parsed<()> {
        let item = if chars == 0 {
            Vec::new()
        } else {
            vec![Element::Terminals(chars)]
        };
        self.start_item(sequence, item)
    }

    fn start_item(&mut self, sequence: &mut Sequence, item: Vec<Element>) -> Parsed<()> {
        self.add_elements(native_len(&item))?;
        sequence.alternative.extend(&sequence.item);
        sequence.item = item;
        sequence.prior_rules = 1;
        Ok(())
    }

    /// Parses `{m}`, `{m,}` or `{m,n}`, returning the bounds the native
    /// parser passes on (`None` for no maximum).
    fn parse_repetition_bounds(
        &self,
        pos: usize,
        nested: bool,
    ) -> Parsed<(usize, u64, Option<u64>)> {
        let mut pos = self.parse_space(pos + 1, nested);
        let (int_end, min) = self.parse_int(pos)?;
        pos = self.parse_space(int_end, nested);
        let mut max = None;
        if self.at(pos) == b'}' {
            max = Some(min);
            pos = self.parse_space(pos + 1, nested);
        } else if self.at(pos) == b',' {
            pos = self.parse_space(pos + 1, nested);
            if self.at(pos).is_ascii_digit() {
                let (int_end, value) = self.parse_int(pos)?;
                max = Some(value);
                pos = self.parse_space(int_end, nested);
            }
            if self.at(pos) != b'}' {
                return Err(Stop::NativeRejects);
            }
            pos = self.parse_space(pos + 1, nested);
        } else {
            return Err(Stop::NativeRejects);
        }
        if min > NATIVE_MAX_REPETITIONS {
            return Err(Stop::NativeRejects);
        }
        if max.is_some_and(|max| max > NATIVE_MAX_REPETITIONS) {
            max = None;
        }
        Ok((pos, min, max))
    }

    /// Mirrors `handle_repetitions`: `S{m,n}` becomes `m` inline copies of `S`
    /// followed by `n - m` nested optional rules `S'(k) ::= S S'(k-1) |`, and
    /// `S{m,}` becomes `m` copies followed by `S' ::= S S' |`.
    fn repeat(&mut self, sequence: &mut Sequence, min: u64, max: Option<u64>) -> Parsed<()> {
        if sequence.item.is_empty() {
            return Err(Stop::NativeRejects);
        }
        let total_rules = match max {
            Some(max) if max > 0 => max,
            _ if min > 0 => min,
            _ => 1,
        };
        if sequence.prior_rules.saturating_mul(total_rules) > NATIVE_MAX_REPETITIONS {
            return Err(Stop::NativeRejects);
        }

        let repeated = std::mem::take(&mut sequence.item);
        let repeated_len = native_len(&repeated);
        if min > 0 {
            self.add_elements(repeated_len.saturating_mul(min - 1))?;
            for _ in 0..min {
                sequence.item.extend_from_slice(&repeated);
            }
        }
        if min == 0 {
            self.elements = self.elements.saturating_sub(repeated_len);
        }

        // The native count is unsigned, so `max < min` wraps to an enormous
        // number of rules.
        let optional_rules = max.map_or(1, |max| max.wrapping_sub(min));
        let mut last_rule = None;
        for _ in 0..optional_rules {
            let rule = self.generated_symbol_id();
            let mut alternative = Alternative::default();
            alternative.extend(&repeated);
            let next = if max.is_none() { Some(rule) } else { last_rule };
            if let Some(next) = next {
                alternative.push(Element::Ref(next));
            }
            let rule_len = alternative.native_len;
            self.add_elements(rule_len + 2)?;
            self.add_rule(
                rule,
                Rule {
                    alternatives: vec![alternative, Alternative::default()],
                },
            )?;
            last_rule = Some(rule);
        }
        if let Some(rule) = last_rule {
            self.add_elements(1)?;
            sequence.item.push(Element::Ref(rule));
        }
        sequence.prior_rules = sequence.prior_rules.saturating_mul(total_rules);
        Ok(())
    }

    fn parse_space(&self, mut pos: usize, newline_ok: bool) -> usize {
        loop {
            match self.at(pos) {
                b' ' | b'\t' => pos += 1,
                b'\r' | b'\n' if newline_ok => pos += 1,
                b'#' => {
                    while !matches!(self.at(pos), 0 | b'\r' | b'\n') {
                        pos += 1;
                    }
                }
                _ => return pos,
            }
        }
    }

    fn parse_name(&self, pos: usize) -> Parsed<usize> {
        let mut end = pos;
        while is_word_char(self.at(end)) {
            end += 1;
        }
        if end == pos {
            return Err(Stop::NativeRejects);
        }
        Ok(end)
    }

    /// Parses a decimal integer like `std::stoull`, which throws on overflow.
    fn parse_int(&self, pos: usize) -> Parsed<(usize, u64)> {
        let mut end = pos;
        let mut value = 0u64;
        while self.at(end).is_ascii_digit() {
            value = value
                .checked_mul(10)
                .and_then(|value| value.checked_add(u64::from(self.at(end) - b'0')))
                .ok_or(Stop::NativeRejects)?;
            end += 1;
        }
        if end == pos {
            return Err(Stop::NativeRejects);
        }
        Ok((end, value))
    }

    fn parse_char(&self, pos: usize) -> Parsed<usize> {
        match self.at(pos) {
            0 => Err(Stop::NativeRejects),
            b'\\' => match self.at(pos + 1) {
                b'x' => self.parse_hex(pos + 2, 2),
                b'u' => self.parse_hex(pos + 2, 4),
                b'U' => self.parse_hex(pos + 2, 8),
                b't' | b'r' | b'n' | b'\\' | b'"' | b'[' | b']' | b'-' => Ok(pos + 2),
                _ => Err(Stop::NativeRejects),
            },
            first => {
                // Mirrors the native UTF-8 decoder, which stops early at NUL.
                let len = match first >> 4 {
                    0xC | 0xD => 2,
                    0xE => 3,
                    0xF => 4,
                    _ => 1,
                };
                let mut end = pos + 1;
                while end < pos + len && self.at(end) != 0 {
                    end += 1;
                }
                Ok(end)
            }
        }
    }

    fn parse_hex(&self, pos: usize, digits: usize) -> Parsed<usize> {
        if (pos..pos + digits).all(|index| self.at(index).is_ascii_hexdigit()) {
            Ok(pos + digits)
        } else {
            Err(Stop::NativeRejects)
        }
    }

    /// Skips `<[id]>` or `<token>`. Any token counts as one terminal; the
    /// native parser additionally requires a named token to be a single
    /// vocabulary token.
    fn parse_token(&self, pos: usize) -> Parsed<usize> {
        if self.at(pos) != b'<' {
            return Err(Stop::NativeRejects);
        }
        let pos = pos + 1;
        if self.at(pos) == b'[' {
            let (end, _) = self.parse_int(pos + 1)?;
            if self.at(end) != b']' || self.at(end + 1) != b'>' {
                return Err(Stop::NativeRejects);
            }
            return Ok(end + 2);
        }
        let mut end = pos;
        while !matches!(self.at(end), 0 | b'>') {
            end += 1;
        }
        if self.at(end) != b'>' {
            return Err(Stop::NativeRejects);
        }
        Ok(end + 1)
    }
}

/// The alternative being parsed, split into what is settled and the most
/// recent item, which a following repetition operator rewrites.
#[derive(Default)]
struct Sequence {
    alternative: Alternative,
    item: Vec<Element>,
    /// The native `n_prev_rules`: how many rules the latest item stands for.
    prior_rules: u64,
}

fn native_len(elements: &[Element]) -> u64 {
    elements.iter().map(|element| element.native_len()).sum()
}

fn is_word_char(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'-'
}

fn too_large(message: String) -> Stop {
    Stop::Rejected(message)
}

#[derive(Clone, Copy)]
enum WalkState {
    Unvisited,
    InProgress,
    Done { height: u32, visits: u64 },
}

/// One rule the walk is inside, and where it is in that rule.
struct Frame {
    rule: usize,
    alternative: usize,
    reference: usize,
    height: u32,
    visits: u64,
}

/// Mirrors the native left recursion check, `llama_grammar_detect_left_recursion`,
/// which recurses into each leftmost rule reference, and past it while the
/// referenced rule has an empty alternative. Returns the deepest recursion
/// and the number of rule visits it makes.
///
/// The native check re-walks a rule every time it is referenced; this walk
/// remembers each rule's height and visit count instead, so measuring a
/// grammar costs time linear in its size.
fn walk_left_references(rules: &[Rule]) -> Parsed<(u32, u64)> {
    let may_be_empty: Vec<bool> = rules.iter().map(Rule::may_be_empty).collect();
    let mut states = vec![WalkState::Unvisited; rules.len()];
    let mut deepest = 0u32;
    let mut total_visits = 0u64;
    for start in 0..rules.len() {
        if !matches!(states[start], WalkState::Unvisited) {
            continue;
        }
        let visits = walk_from(rules, &may_be_empty, &mut states, start, &mut deepest)?;
        total_visits = add_visits(total_visits, visits)?;
    }
    Ok((deepest, total_visits))
}

fn walk_from(
    rules: &[Rule],
    may_be_empty: &[bool],
    states: &mut [WalkState],
    start: usize,
    deepest: &mut u32,
) -> Parsed<u64> {
    let mut stack = vec![enter(states, start, 1, deepest)?];
    loop {
        let depth = stack.len() as u32;
        let frame = stack.last_mut().expect("the walk stack is not empty");
        match next_reference(&rules[frame.rule], frame, may_be_empty) {
            Some(child) => match states[child] {
                // A rule that references itself leftmost: the native check
                // reports left recursion and the grammar is rejected.
                WalkState::InProgress => return Err(Stop::NativeRejects),
                WalkState::Done { height, visits } => {
                    let reached = depth.saturating_add(height);
                    check_depth(reached)?;
                    *deepest = (*deepest).max(reached);
                    frame.height = frame.height.max(height);
                    frame.visits = add_visits(frame.visits, visits)?;
                }
                WalkState::Unvisited => {
                    let child = enter(states, child, depth + 1, deepest)?;
                    stack.push(child);
                }
            },
            None => {
                let done = stack.pop().expect("the walk stack is not empty");
                let height = done.height + 1;
                states[done.rule] = WalkState::Done {
                    height,
                    visits: done.visits,
                };
                let Some(parent) = stack.last_mut() else {
                    return Ok(done.visits);
                };
                parent.height = parent.height.max(height);
                parent.visits = add_visits(parent.visits, done.visits)?;
            }
        }
    }
}

fn enter(states: &mut [WalkState], rule: usize, depth: u32, deepest: &mut u32) -> Parsed<Frame> {
    check_depth(depth)?;
    *deepest = (*deepest).max(depth);
    states[rule] = WalkState::InProgress;
    Ok(Frame {
        rule,
        alternative: 0,
        reference: 0,
        height: 0,
        visits: 1,
    })
}

/// Advances `frame` to the next rule reference the native check recurses
/// into, or returns `None` once the rule is finished.
fn next_reference(rule: &Rule, frame: &mut Frame, may_be_empty: &[bool]) -> Option<usize> {
    while let Some(alternative) = rule.alternatives.get(frame.alternative) {
        if frame.reference > 0 {
            // The native check moves past a reference only while the
            // referenced rule may be empty.
            let previous = alternative.leftmost[frame.reference - 1] as usize;
            if !may_be_empty[previous] {
                frame.alternative += 1;
                frame.reference = 0;
                continue;
            }
        }
        if let Some(&next) = alternative.leftmost.get(frame.reference) {
            frame.reference += 1;
            return Some(next as usize);
        }
        frame.alternative += 1;
        frame.reference = 0;
    }
    None
}

fn check_depth(depth: u32) -> Parsed<()> {
    if depth > MAX_LEFT_REFERENCE_DEPTH {
        return Err(too_large(format!(
            "grammar chains more than {MAX_LEFT_REFERENCE_DEPTH} leftmost rule references"
        )));
    }
    Ok(())
}

fn add_visits(visits: u64, more: u64) -> Parsed<u64> {
    let visits = visits.saturating_add(more);
    if visits > MAX_LEFT_REFERENCE_VISITS {
        return Err(too_large(format!(
            "grammar's leftmost rule references need more than {MAX_LEFT_REFERENCE_VISITS} \
             visits to check"
        )));
    }
    Ok(visits)
}

#[cfg(test)]
mod tests {
    use super::*;

    const JSON_GRAMMAR: &str = r#"root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4}) # escapes
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9] [1-9]{0,15})? ws

# Optional space: by convention, applied in grammars to avoid extra spaces
ws ::= | " " | "\n" [ \t]{0,20}
"#;

    fn chain(rules: usize) -> String {
        let mut grammar = String::from("root ::= r0\n");
        for rule in 0..rules {
            grammar.push_str(&format!("r{rule} ::= r{}\n", rule + 1));
        }
        grammar.push_str(&format!("r{rules} ::= \"a\"\n"));
        grammar
    }

    fn doubling(levels: usize) -> String {
        let mut grammar = String::from("root ::= r0 \"a\"\n");
        for level in 0..levels {
            grammar.push_str(&format!(
                "r{level} ::= \"\" | r{next} r{next}\n",
                next = level + 1
            ));
        }
        grammar.push_str(&format!("r{levels} ::= \"\" | \"b\"\n"));
        grammar
    }

    #[test]
    fn measures_the_same_costs_as_the_native_parser() {
        // Expected values come from llama.cpp's parser and left recursion
        // check run on the same grammars.
        let json = measure(JSON_GRAMMAR).unwrap();
        assert_eq!(
            json,
            GrammarCost {
                rules: 78,
                elements: 425,
                left_reference_depth: 6,
                left_reference_visits: 106,
            }
        );
        let list = measure("root ::= item+\n\n# Excludes various line break characters\nitem ::= \"- \" [^\\r\\n\\x0b\\x0c\\x85\\u2028\\u2029]+ \"\\n\"\n").unwrap();
        assert_eq!(
            list,
            GrammarCost {
                rules: 4,
                elements: 29,
                left_reference_depth: 2,
                left_reference_visits: 5,
            }
        );
    }

    #[test]
    fn accepts_ordinary_grammars() {
        check_grammar_expansion(JSON_GRAMMAR).unwrap();
        check_grammar_expansion(&chain(1000)).unwrap();
        check_grammar_expansion("root ::= [a-z]{0,2000}\n").unwrap();
    }

    #[test]
    fn rejects_long_leftmost_reference_chains() {
        // About 200,000 rules overflowed the native stack.
        let error = check_grammar_expansion(&chain(MAX_LEFT_REFERENCE_DEPTH as usize)).unwrap_err();
        assert!(error.contains("leftmost rule references"), "{error}");
        check_grammar_expansion(&chain(MAX_LEFT_REFERENCE_DEPTH as usize - 2)).unwrap();
    }

    #[test]
    fn rejects_repetitions_that_expand_past_the_element_budget() {
        // 100 KB of this shape expanded to 3.6 GB natively.
        let literal = "a".repeat(3000);
        let error =
            check_grammar_expansion(&format!("root ::= \"{literal}\"{{2000}}\n")).unwrap_err();
        assert!(error.contains("elements"), "{error}");
    }

    #[test]
    fn rejects_repetitions_that_expand_past_the_rule_budget() {
        // Each `{0,2000}` generates 2000 rules.
        let grammar = format!("root ::= {}\n", "\"x\"{0,2000} ".repeat(70));
        let error = check_grammar_expansion(&grammar).unwrap_err();
        assert!(error.contains("rules"), "{error}");
    }

    #[test]
    fn rejects_inverted_repetition_bounds() {
        // The native count of generated rules wraps around for max < min.
        let error = check_grammar_expansion("root ::= \"a\"{5,3}\n").unwrap_err();
        assert!(error.contains("rules"), "{error}");
    }

    #[test]
    fn rejects_grammars_whose_left_recursion_check_explodes() {
        // Each level doubles the native visits; 20 levels did not finish.
        let error = check_grammar_expansion(&doubling(20)).unwrap_err();
        assert!(error.contains("visits"), "{error}");
        check_grammar_expansion(&doubling(8)).unwrap();
    }

    #[test]
    fn rejects_a_root_that_has_no_rule() {
        let error = check_grammar_expansion("e1 ::= \"a\" root{0}\n").unwrap_err();
        assert!(error.contains("root"), "{error}");
    }

    #[test]
    fn references_removed_by_zero_repetitions_need_no_rule() {
        let cost = measure("root ::= c-d{0,0}\n").unwrap();
        assert_eq!(cost.rules, 1);
    }

    #[test]
    fn leaves_grammars_the_native_parser_rejects_to_it() {
        for grammar in [
            "root ::= root \"a\"\n",
            "root ::= undefined\n",
            "root ::= \"a\"{2001}\n",
            "root ::= (\"a\"\n",
            "start ::= \"a\"\n",
            "root ::= \"\\q\"\n",
            "",
        ] {
            assert_eq!(measure(grammar), Err(Stop::NativeRejects), "{grammar:?}");
            check_grammar_expansion(grammar).unwrap();
        }
    }
}
