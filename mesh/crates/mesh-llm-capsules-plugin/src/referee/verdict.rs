//! Two halves compared, and the ruling a referee's re-answer gives.
//!
//! The order of the checks is fixed: each half's record, a half with no
//! answer held, each answer against its signature, sampled twins, then the
//! weights, then who served them, and only then the words. Answers with the
//! same words are corroborated. Any difference is a difference: there is no
//! similarity threshold, and without a referee it is inconclusive. Only a
//! referee's re-answer can name a twin, and only the twin the referee did
//! not agree with.

use serde_json::json;

use super::half::{Half, HalfError};

pub const VERDICT_CORROBORATED: &str = "corroborated";
pub const VERDICT_INCONCLUSIVE: &str = "inconclusive";
pub const VERDICT_NOT_COMPARABLE: &str = "not_comparable";
pub const VERDICT_CONTRADICTED_PREFIX: &str = "contradicted:";

pub const STATUS_SATISFIED: &str = "SATISFIED";
pub const STATUS_CONTRADICTED: &str = "CONTRADICTED";
pub const STATUS_UNKNOWN: &str = "UNKNOWN";

/// Why two halves have no ruling. None of these counts against either twin.
pub const NO_VERDICT_NO_REQUESTER_TRANSCRIPT: &str = "no_requester_transcript";
pub const NO_VERDICT_NOT_COMPARABLE: &str = "not_comparable";
pub const NO_VERDICT_WEIGHTS_MISMATCH: &str = "weights_mismatch";
pub const NO_VERDICT_OWNER_ABSENT: &str = "owner_absent";
pub const NO_VERDICT_SAME_OWNER_TWIN: &str = "same_owner_twin";
pub const NO_VERDICT_REFEREE_NOT_INDEPENDENT: &str = "referee_not_independent";
pub const NO_VERDICT_REFEREE_UNREACHABLE: &str = "referee_unreachable";

/// The fixed members of the sealed ruling block.
pub const ADJUDICATION_SCHEMA: &str = "capsule-emit-mesh/adjudication/v1";
pub const SOURCE_TWIN_COMPARISON: &str = "twin_comparison";
pub const CAPTURE_METHOD_DETERMINISTIC_REPLAY: &str = "deterministic_replay";

pub fn contradicted(node_id: &str) -> String {
    format!("{VERDICT_CONTRADICTED_PREFIX}{node_id}")
}

/// The status a ruling carries beside it. `None` for a ruling this module
/// does not know: it is never rounded to one it does.
pub fn status_for_verdict(verdict: &str) -> Option<&'static str> {
    match verdict {
        VERDICT_CORROBORATED => Some(STATUS_SATISFIED),
        VERDICT_INCONCLUSIVE | VERDICT_NOT_COMPARABLE => Some(STATUS_UNKNOWN),
        v if v.starts_with(VERDICT_CONTRADICTED_PREFIX) => Some(STATUS_CONTRADICTED),
        _ => None,
    }
}

/// The whitespace a word ends at: Unicode white space, and the four ASCII
/// separators (0x1C-0x1F) the reference implementation also splits on.
fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// The words of a text: its whitespace-separated tokens.
pub fn words(text: &str) -> Vec<&str> {
    text.split(is_space).filter(|w| !w.is_empty()).collect()
}

/// The word at `index` of `text`, `None` past its end.
pub fn token_at(text: &str, index: usize) -> Option<&str> {
    words(text).get(index).copied()
}

/// Where two answers first differ, and the digest of the words they share
/// before it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Comparison {
    /// The first word at which they differ; `None` when the words are the same.
    pub divergence_index: Option<usize>,
    /// JSON-DIGEST of `{"prefix_tokens": [the shared words]}`; `None` when
    /// they share none.
    pub prefix_digest: Option<String>,
}

pub fn compare_transcripts(text_a: &str, text_b: &str) -> Comparison {
    let (a, b) = (words(text_a), words(text_b));
    let shared = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
    let same = shared == a.len() && shared == b.len();
    let prefix_digest = (shared > 0).then(|| {
        crate::capsule_emit::digest_json(&json!({ "prefix_tokens": &a[..shared] }))
            .expect("a list of strings always has a digest")
    });
    Comparison {
        divergence_index: (!same).then_some(shared),
        prefix_digest,
    }
}

/// The ruling a referee's re-answer gives on two twins that differ. A
/// re-answer of a request with messages counts only when it repeats the
/// agreed words exactly; its word at the difference then names the twin it
/// agrees with, and the other twin is contradicted. Anything else is
/// inconclusive, never a guess.
pub fn referee_verdict(a: &Half, b: &Half, comparison: &Comparison, referee_text: &str) -> String {
    let at = comparison.divergence_index.unwrap_or(0);
    let referee_token = if a.request_has_messages() {
        let agreed: Vec<&str> = words(a.text()).into_iter().take(at).collect();
        let theirs = words(referee_text);
        (theirs.iter().take(at).copied().collect::<Vec<_>>() == agreed)
            .then(|| token_at(referee_text, at))
            .flatten()
    } else {
        token_at(referee_text, 0)
    };
    let matches =
        |half: &Half| referee_token.is_some() && referee_token == token_at(half.text(), at);
    let (matches_a, matches_b) = (matches(a), matches(b));
    // The twin the referee did NOT agree with is the one contradicted.
    let (agreed, other) = match (matches_a, matches_b) {
        (true, false) => (a, b),
        (false, true) => (b, a),
        _ => return VERDICT_INCONCLUSIVE.to_string(),
    };
    let named = if cfg!(feature = "mutant-referee-verdict-names-wrong-twin") {
        agreed
    } else {
        other
    };
    match named.node_id.as_deref() {
        Some(node) if !node.is_empty() => contradicted(node),
        _ => VERDICT_INCONCLUSIVE.to_string(),
    }
}

/// What a referee that was asked answered.
#[derive(Debug, Clone)]
pub struct RefereeAnswer {
    pub verdict: String,
    /// The referee's own record of its answer, when it has one.
    pub capsule_id: Option<String>,
    /// Who the referee is.
    pub referee_id: String,
}

/// The referee did not answer.
#[derive(Debug, Clone, Copy)]
pub struct Unreachable;

/// Asked with both halves and where they differ.
pub type Referee<'r> =
    &'r mut dyn FnMut(&Half, &Half, &Comparison) -> Result<RefereeAnswer, Unreachable>;

/// Twins with an unknown weights digest: not comparable (the rule), or
/// compared with no shared digest claimed (the referee's door, which states
/// the reason itself, `service`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnknownWeights {
    NotComparable,
    Compare,
}

/// A ruling, or why there is none. Never both.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Outcome {
    pub verdict: Option<String>,
    pub no_verdict_reason: Option<&'static str>,
    pub divergence_index: Option<usize>,
    pub prefix_digest: Option<String>,
    pub twin_owner_distinct: Option<bool>,
    /// The weights both halves name; `None` unless both name the same.
    pub weights_digest: Option<String>,
    pub half_a_capsule_id: String,
    pub half_b_capsule_id: String,
    pub referee_called: bool,
    pub referee_capsule_id: Option<String>,
    pub referee_id: Option<String>,
}

impl Outcome {
    fn none(a: &Half, b: &Half, reason: &'static str) -> Self {
        Self {
            verdict: None,
            no_verdict_reason: Some(reason),
            divergence_index: None,
            prefix_digest: None,
            twin_owner_distinct: None,
            weights_digest: None,
            half_a_capsule_id: a.capsule_id().to_string(),
            half_b_capsule_id: b.capsule_id().to_string(),
            referee_called: false,
            referee_capsule_id: None,
            referee_id: None,
        }
    }

    /// The answer the parity corpus compares (`adjudicate` path).
    #[cfg(test)]
    pub fn to_answer(&self) -> serde_json::Value {
        json!({
            "verdict": self.verdict,
            "status": self.verdict.as_deref().and_then(status_for_verdict),
            "no_verdict_reason": self.no_verdict_reason,
            "divergence_index": self.divergence_index,
            "prefix_digest": self.prefix_digest,
            "twin_owner_distinct": self.twin_owner_distinct,
            "weights_digest": self.weights_digest,
            "referee_called": self.referee_called,
        })
    }
}

/// Compare two halves; call `referee` only when they differ. `Err` when a
/// half cannot be read at all.
pub fn adjudicate(
    a: &Half,
    b: &Half,
    referee: Option<Referee<'_>>,
    referee_node_id: Option<&str>,
    unknown_weights: UnknownWeights,
) -> Result<Outcome, HalfError> {
    a.check_record()?;
    b.check_record()?;
    if a.no_answer_held() || b.no_answer_held() {
        return Ok(Outcome::none(a, b, NO_VERDICT_NO_REQUESTER_TRANSCRIPT));
    }
    a.check_answer()?;
    b.check_answer()?;
    if a.sampled() || b.sampled() {
        return Ok(Outcome::none(a, b, NO_VERDICT_NOT_COMPARABLE));
    }
    let (weights_a, weights_b) = (a.weights_digest.as_deref(), b.weights_digest.as_deref());
    let known = |w: Option<&str>| w.is_some_and(|w| !w.is_empty());
    if unknown_weights == UnknownWeights::NotComparable && !(known(weights_a) && known(weights_b)) {
        return Ok(Outcome::none(a, b, NO_VERDICT_NOT_COMPARABLE));
    }
    if let (Some(x), Some(y)) = (weights_a, weights_b) {
        if x != y {
            return Ok(Outcome::none(a, b, NO_VERDICT_WEIGHTS_MISMATCH));
        }
    }
    let shared = match (weights_a, weights_b) {
        (Some(x), Some(_)) => Some(x.to_string()),
        _ => None,
    };
    let with_shared = |mut outcome: Outcome| {
        outcome.weights_digest = shared.clone();
        outcome
    };
    let (Some(owner_a), Some(owner_b)) = (a.node_id.as_deref(), b.node_id.as_deref()) else {
        return Ok(with_shared(Outcome::none(a, b, NO_VERDICT_OWNER_ABSENT)));
    };
    if owner_a == owner_b {
        let mut outcome = with_shared(Outcome::none(a, b, NO_VERDICT_SAME_OWNER_TWIN));
        outcome.twin_owner_distinct = Some(false);
        return Ok(outcome);
    }

    let comparison = compare_transcripts(a.text(), b.text());
    let compared = |verdict: Option<String>, reason: Option<&'static str>| Outcome {
        verdict,
        no_verdict_reason: reason,
        divergence_index: comparison.divergence_index,
        prefix_digest: comparison.prefix_digest.clone(),
        twin_owner_distinct: Some(true),
        weights_digest: shared.clone(),
        half_a_capsule_id: a.capsule_id().to_string(),
        half_b_capsule_id: b.capsule_id().to_string(),
        referee_called: false,
        referee_capsule_id: None,
        referee_id: None,
    };
    if comparison.divergence_index.is_none() {
        return Ok(compared(Some(VERDICT_CORROBORATED.to_string()), None));
    }
    let Some(referee) = referee else {
        return Ok(compared(Some(VERDICT_INCONCLUSIVE.to_string()), None));
    };
    if referee_node_id.is_some_and(|r| r == owner_a || r == owner_b) {
        return Ok(compared(None, Some(NO_VERDICT_REFEREE_NOT_INDEPENDENT)));
    }
    match referee(a, b, &comparison) {
        Err(Unreachable) => Ok(compared(None, Some(NO_VERDICT_REFEREE_UNREACHABLE))),
        Ok(answer) => Ok(Outcome {
            referee_called: true,
            referee_capsule_id: answer.capsule_id,
            referee_id: Some(answer.referee_id),
            ..compared(Some(answer.verdict), None)
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_split_on_any_whitespace() {
        assert_eq!(words(" a\tb\u{a0}c\u{1f}d \n"), ["a", "b", "c", "d"]);
        assert_eq!(token_at("a b", 1), Some("b"));
        assert_eq!(token_at("a b", 2), None);
    }

    #[test]
    fn the_first_difference_and_the_shared_words() {
        let same = compare_transcripts("a b c", "a  b c");
        assert_eq!(same.divergence_index, None);
        assert!(same.prefix_digest.is_some());
        let differ = compare_transcripts("a b c", "a b d");
        assert_eq!(differ.divergence_index, Some(2));
        let prefix = compare_transcripts("a b", "a b c");
        assert_eq!(prefix.divergence_index, Some(2), "one answer stops early");
        let none = compare_transcripts("x", "y");
        assert_eq!((none.divergence_index, none.prefix_digest), (Some(0), None));
        let empty = compare_transcripts("", " ");
        assert_eq!((empty.divergence_index, empty.prefix_digest), (None, None));
    }

    #[test]
    fn every_ruling_has_a_status_and_no_other_does() {
        assert_eq!(status_for_verdict("corroborated"), Some(STATUS_SATISFIED));
        assert_eq!(
            status_for_verdict("contradicted:node-b"),
            Some(STATUS_CONTRADICTED)
        );
        assert_eq!(status_for_verdict("not_comparable"), Some(STATUS_UNKNOWN));
        assert_eq!(status_for_verdict("something_else"), None);
    }
}
