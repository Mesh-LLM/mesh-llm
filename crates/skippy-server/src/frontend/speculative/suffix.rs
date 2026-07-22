use std::time::Instant;

use ahash::AHashMap;

use super::SUFFIX_NGRAM_MAX_WINDOW;

/// Smallest seed length the suffix index will key on.
pub(super) const SUFFIX_MIN_SEED_LEN: usize = 3;
/// Largest seed length; bounds the fixed-size seed key.
const SUFFIX_MAX_SEED_LEN: usize = 8;
/// Hard cap on seed occurrences examined per lookup. Ambiguous repetitive input
/// (many seeds with differing preceding tokens) would otherwise scan the whole
/// bucket every decode step. The bucket is walked most-recent-first, so the most
/// temporally relevant occurrences are always the ones examined.
const SUFFIX_MAX_LOOKUP_CANDIDATES: usize = 64;

/// Per-request lookup counters surfaced through response timings.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct SuffixProposalStats {
    pub(super) match_length: usize,
    pub(super) candidates_examined: usize,
    pub(super) appended_tokens: usize,
    pub(super) rebuilt: bool,
    pub(super) sync_us: u64,
    pub(super) lookup_us: u64,
}

/// A draft together with the stats gathered while producing it.
pub(super) struct SuffixProposal {
    pub(super) tokens: Vec<i32>,
    pub(super) stats: SuffixProposalStats,
}

/// Fixed-size exact key for the seed index, avoiding hash collisions.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct SeedKey {
    len: u8,
    tokens: [i32; SUFFIX_MAX_SEED_LEN],
}

impl SeedKey {
    /// Builds a key from the final `seed_len` tokens of `tokens`.
    fn from_tail(tokens: &[i32], seed_len: usize) -> Self {
        debug_assert!((SUFFIX_MIN_SEED_LEN..=SUFFIX_MAX_SEED_LEN).contains(&seed_len));
        debug_assert!(tokens.len() >= seed_len);
        let mut seed = [0; SUFFIX_MAX_SEED_LEN];
        seed[..seed_len].copy_from_slice(&tokens[tokens.len() - seed_len..]);
        Self {
            len: seed_len as u8,
            tokens: seed,
        }
    }

    /// Builds a key from the final `seed_len` tokens of the query.
    fn from_query(query_len: usize, seed_len: usize, token_at: impl Fn(usize) -> i32) -> Self {
        let mut seed = [0; SUFFIX_MAX_SEED_LEN];
        for (offset, slot) in seed[..seed_len].iter_mut().enumerate() {
            *slot = token_at(query_len - seed_len + offset);
        }
        Self {
            len: seed_len as u8,
            tokens: seed,
        }
    }
}

/// Bookkeeping returned by an index sync.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct SyncStats {
    appended_tokens: usize,
    rebuilt: bool,
    elapsed_us: u64,
}

/// Request-local prompt-lookup proposer based on the longest exact suffix.
///
/// The index contains only target-committed history. A native-MTP continuation
/// may participate in lookup, but never mutates history or the index.
pub(super) struct SuffixNgramProposer {
    min_match: usize,
    max_window: usize,
    max_proposal_tokens: usize,
    seed_len: usize,
    committed_history: Vec<i32>,
    index: AHashMap<SeedKey, Vec<u32>>,
}

impl SuffixNgramProposer {
    /// Builds a proposer, validating the match-length and window bounds.
    pub(super) fn new(
        min_match: usize,
        max_window: usize,
        max_proposal_tokens: usize,
    ) -> Result<Self, String> {
        if min_match < SUFFIX_MIN_SEED_LEN || min_match > max_window {
            return Err(format!(
                "suffix N-gram proposer requires {SUFFIX_MIN_SEED_LEN} <= min_match <= max_window"
            ));
        }
        if max_window > SUFFIX_NGRAM_MAX_WINDOW {
            return Err(format!(
                "suffix N-gram proposer max_window {max_window} exceeds {SUFFIX_NGRAM_MAX_WINDOW}"
            ));
        }
        if max_proposal_tokens == 0 {
            return Err("suffix N-gram proposer requires max_proposal_tokens > 0".to_string());
        }
        Ok(Self {
            min_match,
            max_window,
            max_proposal_tokens,
            seed_len: min_match.min(SUFFIX_MAX_SEED_LEN),
            committed_history: Vec::new(),
            index: AHashMap::new(),
        })
    }

    /// Brings the index in line with committed history: appends on the fast
    /// path, rebuilds when history diverges.
    fn sync(&mut self, committed_history: &[i32]) -> SyncStats {
        let started = Instant::now();
        let (first_new_end, appended_tokens, rebuilt) =
            if committed_history.starts_with(&self.committed_history) {
                let old_len = self.committed_history.len();
                let appended = &committed_history[old_len..];
                self.committed_history.extend_from_slice(appended);
                (old_len.max(self.seed_len - 1), appended.len(), false)
            } else {
                self.index.clear();
                self.committed_history.clear();
                self.committed_history.extend_from_slice(committed_history);
                (self.seed_len - 1, committed_history.len(), true)
            };

        for end in first_new_end..self.committed_history.len() {
            let key = SeedKey::from_tail(&self.committed_history[..=end], self.seed_len);
            self.index.entry(key).or_default().push(end as u32);
        }

        SyncStats {
            appended_tokens,
            rebuilt,
            elapsed_us: elapsed_us(started),
        }
    }

    /// Syncs the index, then returns the longest-suffix draft for the query.
    pub(super) fn propose(
        &mut self,
        committed_history: &[i32],
        continuation_prefix: &[i32],
        max_proposed_tokens: usize,
    ) -> SuffixProposal {
        let sync = self.sync(committed_history);
        let mut stats = SuffixProposalStats {
            appended_tokens: sync.appended_tokens,
            rebuilt: sync.rebuilt,
            sync_us: sync.elapsed_us,
            ..SuffixProposalStats::default()
        };
        let lookup_started = Instant::now();
        let tokens = self.lookup(continuation_prefix, max_proposed_tokens, &mut stats);
        stats.lookup_us = elapsed_us(lookup_started);
        SuffixProposal { tokens, stats }
    }

    /// Finds the longest verbatim earlier occurrence of the query suffix and
    /// returns the tokens that followed it.
    fn lookup(
        &self,
        continuation_prefix: &[i32],
        max_proposed_tokens: usize,
        stats: &mut SuffixProposalStats,
    ) -> Vec<i32> {
        if max_proposed_tokens == 0 {
            return Vec::new();
        }
        let committed_len = self.committed_history.len();
        let query_len = committed_len + continuation_prefix.len();
        if query_len < self.seed_len || query_len < self.min_match {
            return Vec::new();
        }
        let token_at = |idx: usize| -> i32 {
            if idx < committed_len {
                self.committed_history[idx]
            } else {
                continuation_prefix[idx - committed_len]
            }
        };
        let key = SeedKey::from_query(query_len, self.seed_len, token_at);
        let Some(bucket) = self.index.get(&key) else {
            return Vec::new();
        };

        let mut best_match = 0;
        let mut best_end = 0;
        for &end in bucket.iter().rev() {
            let end = end as usize;
            if end + 1 >= committed_len {
                continue;
            }
            stats.candidates_examined += 1;
            let match_len = self.match_len_backward(end, query_len, token_at);
            if match_len > best_match {
                best_match = match_len;
                best_end = end;
            }
            if best_match >= self.max_window
                || stats.candidates_examined >= SUFFIX_MAX_LOOKUP_CANDIDATES
            {
                break;
            }
        }
        stats.match_length = best_match;
        if best_match < self.min_match {
            return Vec::new();
        }
        let draft_len = max_proposed_tokens
            .min(self.max_proposal_tokens)
            .min((2 * best_match).max(4))
            .min(committed_len - (best_end + 1));
        self.committed_history[best_end + 1..best_end + 1 + draft_len].to_vec()
    }

    /// Counts matching tokens walking backward from `hist_end`, capped at `max_window`.
    fn match_len_backward(
        &self,
        hist_end: usize,
        query_len: usize,
        token_at: impl Fn(usize) -> i32,
    ) -> usize {
        let mut len = 0;
        while len < self.max_window
            && len <= hist_end
            && len < query_len
            && self.committed_history[hist_end - len] == token_at(query_len - 1 - len)
        {
            len += 1;
        }
        len
    }
}

/// Microseconds elapsed since `started`, saturating into a `u64`.
fn elapsed_us(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefers_the_longest_match_over_the_most_recent() {
        let mut proposer = SuffixNgramProposer::new(3, 16, 8).unwrap();
        let committed = [1, 2, 3, 4, 5, 6, 8, 9, 2, 3, 4, 7, 8];

        assert_eq!(
            proposer.propose(&committed, &[1, 2, 3, 4], 2).tokens,
            vec![5, 6]
        );
    }

    #[test]
    fn drafts_a_long_run_on_an_edit_workload() {
        let mut proposer = SuffixNgramProposer::new(3, 16, 7).unwrap();
        let mut committed = vec![11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22];
        committed.extend_from_slice(&[11, 12, 13, 14, 15]);

        let draft = proposer.propose(&committed, &[], 16).tokens;
        assert_eq!(draft, vec![16, 17, 18, 19, 20, 21, 22]);
        assert!(draft.len() > skippy_runtime::NGRAM_CACHE_MAX_NGRAM);
    }

    #[test]
    fn stays_silent_below_min_match() {
        let mut proposer = SuffixNgramProposer::new(5, 16, 8).unwrap();
        let committed = [1, 2, 3, 4, 9, 2, 3, 4];

        assert!(proposer.propose(&committed, &[], 4).tokens.is_empty());
    }

    #[test]
    fn treats_the_continuation_prefix_as_read_only() {
        let mut proposer = SuffixNgramProposer::new(3, 16, 4).unwrap();
        let committed = [1, 2, 3, 1, 2, 3, 1, 2];

        assert_eq!(proposer.propose(&committed, &[], 2).tokens, vec![3, 1]);
        assert!(proposer.propose(&committed, &[9], 2).tokens.is_empty());
        assert_eq!(proposer.propose(&committed, &[], 2).tokens, vec![3, 1]);
    }

    #[test]
    fn syncs_incrementally_then_rebuilds_on_divergence() {
        let mut proposer = SuffixNgramProposer::new(3, 16, 4).unwrap();
        let first = proposer.propose(&[1, 2, 3, 4], &[], 4);
        assert!(!first.stats.rebuilt);
        assert_eq!(first.stats.appended_tokens, 4);

        let appended = proposer.propose(&[1, 2, 3, 4, 1, 2, 3], &[], 4);
        assert!(!appended.stats.rebuilt);
        assert_eq!(appended.stats.appended_tokens, 3);
        assert_eq!(appended.tokens, vec![4, 1, 2, 3]);

        let rebuilt = proposer.propose(&[9, 1, 2, 3], &[], 4);
        assert!(rebuilt.stats.rebuilt);
        assert!(rebuilt.tokens.is_empty());
    }

    #[test]
    fn retains_useful_matches_beyond_eight_seed_occurrences() {
        let mut proposer = SuffixNgramProposer::new(5, 16, 8).unwrap();
        let mut committed = vec![100, 1, 2, 3, 4, 5, 9, 10, 11];
        for prefix in 200..212 {
            committed.extend_from_slice(&[prefix, 1, 2, 3, 4, 5, prefix + 100]);
        }
        committed.extend_from_slice(&[100, 1, 2, 3, 4, 5]);

        let proposal = proposer.propose(&committed, &[], 3);
        assert!(proposal.stats.candidates_examined > 8);
        assert_eq!(proposal.tokens, vec![9, 10, 11]);
    }

    #[test]
    fn lookup_candidate_scan_is_bounded_on_ambiguous_repeats() {
        let mut proposer = SuffixNgramProposer::new(5, 16, 8).unwrap();
        let mut committed = Vec::new();
        // ~100 occurrences of the seed [1,2,3,4,5], each preceded by a distinct
        // token so no backward match reaches max_window; without a budget the
        // whole bucket is scanned on every decode step.
        for prefix in 300..400 {
            committed.extend_from_slice(&[prefix, 1, 2, 3, 4, 5, prefix + 1000]);
        }
        committed.extend_from_slice(&[7, 1, 2, 3, 4, 5]);

        let proposal = proposer.propose(&committed, &[], 3);
        assert!(
            proposal.stats.candidates_examined <= SUFFIX_MAX_LOOKUP_CANDIDATES,
            "examined {}",
            proposal.stats.candidates_examined
        );
        assert!(proposal.stats.candidates_examined > 8);
    }

    #[test]
    fn proposal_budget_is_independent_from_match_length() {
        let mut proposer = SuffixNgramProposer::new(8, 16, 3).unwrap();
        let committed = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8];

        assert_eq!(proposer.propose(&committed, &[], 8).tokens, vec![9, 1, 2]);
    }
}
