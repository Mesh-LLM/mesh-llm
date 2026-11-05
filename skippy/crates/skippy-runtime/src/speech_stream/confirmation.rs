//! Alignment ownership for words already emitted from the current audio window.
//!
//! Canonical output never changes. These anchors track its latest observed
//! spellings and retain omitted words until their audio is actually trimmed.

use super::same_word;

#[derive(Debug, Clone)]
struct Anchor {
    text: String,
    /// The immutable emitted spelling. Hypothesis-only insertions own none.
    canonical: Option<String>,
    observed: bool,
}

impl Anchor {
    fn matches(&self, word: &str) -> bool {
        same_word(&self.text, word)
            || self
                .canonical
                .as_ref()
                .is_some_and(|text| same_word(text, word))
    }
}

#[derive(Debug, Default)]
pub(super) struct ConfirmedPrefix {
    anchors: Vec<Anchor>,
    /// Latest prefix boundary; unsupported observations do not change the
    /// last supported spelling/ownership mapping in `anchors`.
    observed_words: usize,
}

impl ConfirmedPrefix {
    pub(super) fn append(&mut self, words: &[String]) {
        self.anchors.extend(words.iter().map(|text| Anchor {
            text: text.clone(),
            canonical: Some(text.clone()),
            observed: true,
        }));
        self.observed_words += words.len();
    }

    pub(super) fn clear(&mut self) {
        self.anchors.clear();
        self.observed_words = 0;
    }

    pub(super) fn observed_len(&self) -> usize {
        self.observed_words
    }

    /// Recover a supported prefix without making uncertain audio trimmable.
    /// Repeated pending speech may shorten the last observed boundary without
    /// assigning unsupported spellings to canonical owners.
    pub(super) fn unsupported_boundary(&mut self, words: &[String], pending: &[String]) -> usize {
        if let Some(alignment) = self.recovered_alignment(words, pending) {
            self.anchors = self.aligned_anchors(words, &alignment.cells, alignment.end);
            self.observed_words = alignment.end;
            return alignment.end;
        }
        let boundary = self.observed_len().min(words.len());
        if pending.is_empty() {
            return boundary;
        }
        let (end, support) = (0..=boundary)
            .map(|end| {
                let support = words[end..]
                    .iter()
                    .zip(pending)
                    .take_while(|(a, b)| same_word(a, b))
                    .count();
                (end, support)
            })
            .max_by_key(|(end, support)| (*support, *end))
            .unwrap_or((boundary, 0));
        if support > 0 && end < boundary {
            self.observed_words = end;
        }
        end
    }

    /// A vanished confirmed prefix can leave the previous unconfirmed tail at
    /// the beginning of the new hypothesis. Prefer that supported tail only
    /// when its leading agreement exceeds the exact support in both confirmed
    /// prefix alignments. An omitted start does not erase that evidence. Equal
    /// support is ambiguous, especially for genuine repeated phrases.
    pub(super) fn align_with_pending(
        &mut self,
        words: &[String],
        pending: &[String],
        canonical: &[String],
    ) -> Option<usize> {
        if pending.is_empty() {
            return self.align(words);
        }
        let pending_support = pending
            .iter()
            .zip(words)
            .take_while(|(a, b)| same_word(a, b))
            .count();
        if pending_support == 0 {
            return self.align_pending(words, pending);
        }
        let mut canonical_prefix = Self::default();
        canonical_prefix.append(canonical);
        let confirmed_support = self
            .alignment_support(words, pending)
            .max(canonical_prefix.alignment_support(words, pending));
        if pending_support > confirmed_support {
            self.anchors.retain(|anchor| anchor.canonical.is_some());
            for anchor in &mut self.anchors {
                anchor.observed = false;
            }
            self.observed_words = 0;
            return Some(0);
        }
        self.align_pending(words, pending)
    }

    /// Consume only a supported prefix. Missing anchors can recover either
    /// their canonical or latest observed spelling, or remain absent. They
    /// cannot consume an unrelated new tail word
    /// as a fresh substitution. Observed anchors may revise their spelling.
    pub(super) fn align(&mut self, words: &[String]) -> Option<usize> {
        self.align_pending(words, &[])
    }

    fn align_pending(&mut self, words: &[String], pending: &[String]) -> Option<usize> {
        if self.anchors.is_empty() {
            self.observed_words = 0;
            return Some(0);
        }
        let alignment = self.alignment(words, pending, None)?;
        self.anchors = self.aligned_anchors(words, &alignment.cells, alignment.end);
        self.observed_words = alignment.end;
        Some(alignment.end)
    }

    fn alignment_support(&self, words: &[String], pending: &[String]) -> usize {
        self.alignment(words, pending, None)
            .map_or(0, |alignment| alignment.score.matched)
    }

    /// A cheaper unsupported path can hide a restored owner. Require one
    /// owned anchor to match, then compare the supported paths by the normal
    /// edit, pending-tail, and observed-length ordering. Ownerless insertions
    /// cannot justify consuming a new tail.
    fn recovered_alignment(&self, words: &[String], pending: &[String]) -> Option<Alignment> {
        self.anchors
            .iter()
            .enumerate()
            .filter(|(_, anchor)| {
                anchor.canonical.is_some() && words.iter().any(|word| anchor.matches(word))
            })
            .filter_map(|(required, _)| self.alignment(words, pending, Some(required)))
            .min_by_key(|alignment| {
                let tail_support = words[alignment.end..]
                    .iter()
                    .zip(pending)
                    .take_while(|(a, b)| same_word(a, b))
                    .count();
                (
                    alignment.score.preference(),
                    std::cmp::Reverse(tail_support),
                    alignment.end.abs_diff(self.observed_len()),
                    alignment.end,
                )
            })
    }

    fn tail_replays_omitted_owner(
        &self,
        words: &[String],
        cells: &[Option<Cell>],
        end: usize,
    ) -> bool {
        self.aligned_anchors(words, cells, end)
            .iter()
            .any(|anchor| {
                !anchor.observed
                    && anchor.canonical.is_some()
                    && words[end..].iter().any(|word| anchor.matches(word))
            })
    }

    /// A newly omitted owned word must not simultaneously reappear as an
    /// ownerless insertion inside the consumed prefix. Keep that owner in the
    /// alignment and recompute from the unchanged anchors. Pins only grow for
    /// this call, so there are at most one more passes than there are anchors.
    fn alignment(
        &self,
        words: &[String],
        pending: &[String],
        required: Option<usize>,
    ) -> Option<Alignment> {
        let mut pinned = vec![false; self.anchors.len()];
        // Prior unconfirmed text is independent evidence for a candidate tail.
        // It only resolves ties in edits and supported matches; empty pending
        // leaves the original length-based endpoint ordering unchanged.
        let tail_agreement = (0..=words.len())
            .map(|end| {
                pending
                    .iter()
                    .zip(&words[end..])
                    .take_while(|(a, b)| same_word(a, b))
                    .count()
            })
            .collect::<Vec<_>>();
        loop {
            let cells = self.alignment_cells(words, &pinned, required);
            let (end, score) =
                self.best_alignment(&cells, words, &tail_agreement, required.is_some())?;
            if score.matched == 0 && !words.is_empty() {
                return None;
            }
            let conflicts = self.relocated_owners(words, &cells, end, required.is_some());
            if conflicts.is_empty() {
                return Some(Alignment { cells, end, score });
            }
            for index in conflicts {
                debug_assert!(!pinned[index], "a pinned owner cannot be deleted");
                pinned[index] = true;
            }
        }
    }

    fn best_alignment(
        &self,
        cells: &[Option<Cell>],
        words: &[String],
        tail_agreement: &[usize],
        recovering: bool,
    ) -> Option<(usize, Cell)> {
        let columns = words.len() + 1;
        cells[self.anchors.len() * columns..]
            .iter()
            .enumerate()
            .filter_map(|(end, cell)| cell.map(|cell| (end, cell)))
            .filter(|(end, _)| !recovering || !self.tail_replays_omitted_owner(words, cells, *end))
            .min_by_key(|(end, cell)| {
                (
                    cell.preference(),
                    std::cmp::Reverse(tail_agreement[*end]),
                    end.abs_diff(self.observed_len()),
                    *end,
                )
            })
    }

    fn relocated_owners(
        &self,
        words: &[String],
        cells: &[Option<Cell>],
        end: usize,
        recovering: bool,
    ) -> Vec<usize> {
        let columns = words.len() + 1;
        let (mut row, mut column) = (self.anchors.len(), end);
        let mut omitted = Vec::new();
        let mut inserted = Vec::new();
        while row != 0 || column != 0 {
            match cells[row * columns + column]
                .expect("selected alignment is reachable")
                .step
            {
                Step::Consume => {
                    row -= 1;
                    column -= 1;
                }
                Step::Delete => {
                    row -= 1;
                    let anchor = &self.anchors[row];
                    if (anchor.observed || recovering) && anchor.canonical.is_some() {
                        omitted.push(row);
                    }
                }
                Step::Insert => {
                    column -= 1;
                    inserted.push(&words[column]);
                }
            }
        }
        omitted
            .into_iter()
            .filter(|&index| {
                inserted
                    .iter()
                    .any(|word| self.anchors[index].matches(word))
            })
            .collect()
    }

    /// Remove ownership through the last observed word in the cut segment.
    /// Missing words before that boundary belong to the removed audio; missing
    /// suffix words do not. Hypothesis-only insertions own no canonical word.
    pub(super) fn trim(&mut self, words: usize) -> usize {
        if words == 0 {
            return 0;
        }
        let end = self
            .anchors
            .iter()
            .enumerate()
            .filter(|(_, anchor)| anchor.observed)
            .nth(words - 1)
            .map(|(index, _)| index + 1)
            .expect("trim stays inside the aligned confirmed prefix");
        self.observed_words -= words;
        self.anchors
            .drain(..end)
            .filter(|anchor| anchor.canonical.is_some())
            .count()
    }

    fn alignment_cells(
        &self,
        words: &[String],
        pinned: &[bool],
        required: Option<usize>,
    ) -> Vec<Option<Cell>> {
        let columns = words.len() + 1;
        let mut cells = Vec::with_capacity((self.anchors.len() + 1) * columns);
        for edits in 0..columns {
            cells.push(Some(Cell {
                edits,
                matched: 0,
                step: Step::Insert,
            }));
        }
        for (row, anchor) in self.anchors.iter().enumerate() {
            let previous_row = row * columns;
            let deletion = usize::from(anchor.observed);
            let must_match = required == Some(row) || (required.is_some() && pinned[row]);
            cells.push(if pinned[row] || must_match {
                None
            } else {
                cells[previous_row].map(|cell| cell.advance(deletion, false, Step::Delete))
            });
            for (column, word) in words.iter().enumerate() {
                let matches = anchor.matches(word);
                let above = cells[previous_row + column + 1];
                let left = *cells.last().expect("row has its first cell");
                let mut best = [
                    if pinned[row] || must_match {
                        None
                    } else {
                        above.map(|cell| cell.advance(deletion, false, Step::Delete))
                    },
                    left.map(|cell| cell.advance(1, false, Step::Insert)),
                ]
                .into_iter()
                .flatten()
                .min_by_key(|cell| cell.preference());
                if (matches || (anchor.observed && !must_match))
                    && let Some(cell) = cells[previous_row + column]
                {
                    let diagonal = cell.advance(usize::from(!matches), matches, Step::Consume);
                    if best.is_none_or(|best| diagonal.preference() <= best.preference()) {
                        best = Some(diagonal);
                    }
                }
                cells.push(best);
            }
        }
        cells
    }

    fn aligned_anchors(&self, words: &[String], cells: &[Option<Cell>], end: usize) -> Vec<Anchor> {
        let columns = words.len() + 1;
        let (mut row, mut column) = (self.anchors.len(), end);
        let mut anchors = Vec::new();
        while row != 0 || column != 0 {
            match cells[row * columns + column]
                .expect("selected alignment is reachable")
                .step
            {
                Step::Consume => {
                    row -= 1;
                    column -= 1;
                    anchors.push(Anchor {
                        text: words[column].clone(),
                        observed: true,
                        ..self.anchors[row].clone()
                    });
                }
                Step::Delete => {
                    row -= 1;
                    if self.anchors[row].canonical.is_some() {
                        anchors.push(Anchor {
                            observed: false,
                            ..self.anchors[row].clone()
                        });
                    }
                }
                Step::Insert => {
                    column -= 1;
                    anchors.push(Anchor {
                        text: words[column].clone(),
                        canonical: None,
                        observed: true,
                    });
                }
            }
        }
        anchors.reverse();
        anchors
    }
}

struct Alignment {
    cells: Vec<Option<Cell>>,
    end: usize,
    score: Cell,
}

#[derive(Clone, Copy)]
enum Step {
    Consume,
    Delete,
    Insert,
}

#[derive(Clone, Copy)]
struct Cell {
    edits: usize,
    matched: usize,
    step: Step,
}

impl Cell {
    fn preference(self) -> (usize, std::cmp::Reverse<usize>) {
        (self.edits, std::cmp::Reverse(self.matched))
    }

    fn advance(self, edits: usize, matched: bool, step: Step) -> Self {
        Self {
            edits: self.edits + edits,
            matched: self.matched + usize::from(matched),
            step,
        }
    }
}

#[cfg(test)]
mod tests;
