//! When a referee is asked, and at most once per pair.
//!
//! A referee is asked only when two twins of a pair a client marked (a twin
//! bracket id) answered the same request at temperature 0, on the same model
//! and weights, and their answers differ. It is off unless the operator turns
//! it on (`adjudicate_differing_twins`). At most one call per pair, never
//! retried: a pair is the bracket id, the twins' request digest and the two
//! node ids, so re-sealed halves of one exchange are the same pair. A call
//! that was made and not answered has used the pair's one call.
//!
//! Every other outcome is "not adjudicated" with its reason, and none counts
//! against either twin: `off`, `host_does_not_mark_twins`, `twins_agree`,
//! `not_comparable` (with `because`), `no_eligible_referee`,
//! `referee_cannot_sign`, `referee_unreachable`.
//!
//! Decided (2026-09-30), pinned by [`NO_ELIGIBLE_IS_RETRIED_AUTOMATICALLY`]
//! and `no_eligible_referee_is_not_retried_on_its_own`: a pair that found no
//! eligible referee made no call, so its one call is not used, but it is not
//! retried on its own. The operator may ask again; that looks for a referee
//! again, and a call it makes is the pair's one call.

use std::collections::HashMap;
use std::path::Path;

use serde_json::{json, Value};

use super::verdict::compare_transcripts;

/// The operator's switch. Unset: off.
pub const ENV_ADJUDICATE_DIFFERING_TWINS: &str = "CAPSULES_ADJUDICATE_DIFFERING_TWINS";

/// Decided: a pair that found no eligible referee is not retried on its own;
/// only the operator asking again looks for a referee again.
pub const NO_ELIGIBLE_IS_RETRIED_AUTOMATICALLY: bool = false;

/// Beside the ledger: one line per pair outcome this node reached.
pub const PAIRS_FILENAME: &str = "referee-pairs.jsonl";

pub const STATE_ADJUDICATED: &str = "adjudicated";
pub const STATE_NOT_ADJUDICATED: &str = "not_adjudicated";

pub const REASON_OFF: &str = "off";
/// No client marked any pair on this list. The value is the legacy wire id,
/// kept so stored rows and the page keep reading it.
pub const REASON_HOST_DOES_NOT_MARK_TWINS: &str = "host_does_not_mark_twins";
pub const REASON_TWINS_AGREE: &str = "twins_agree";
pub const REASON_NOT_COMPARABLE: &str = "not_comparable";
pub const REASON_NO_ELIGIBLE_REFEREE: &str = super::select::NO_ELIGIBLE_REFEREE;
pub const REASON_REFEREE_CANNOT_SIGN: &str = "referee_cannot_sign";
pub const REASON_REFEREE_UNREACHABLE: &str = "referee_unreachable";

pub const BECAUSE_SAMPLED: &str = "sampled";
pub const BECAUSE_MODEL_HASH_DIFFERS: &str = "model_hash_differs";
/// A twin's record names no model: its model cannot be compared, and no
/// referee can be chosen for it.
pub const BECAUSE_MODEL_HASH_UNKNOWN: &str = "model_hash_unknown";
pub const BECAUSE_WEIGHTS_UNKNOWN: &str = "weights_unknown";
pub const BECAUSE_WEIGHTS_DIFFER: &str = "weights_differ";

/// The switch's value: on only when the operator says so (`1`, `on`,
/// `true`, `yes`); anything else, or unset, is off. A referee answers with
/// a third node's inference, so a client's pair marking alone never starts one.
pub fn adjudicate_differing_twins_from(raw: Option<&str>) -> bool {
    matches!(
        raw.map(|r| r.trim().to_ascii_lowercase()).as_deref(),
        Some("1" | "on" | "true" | "yes")
    )
}

pub fn adjudicate_differing_twins() -> bool {
    adjudicate_differing_twins_from(
        crate::settings::var(ENV_ADJUDICATE_DIFFERING_TWINS)
            .ok()
            .as_deref(),
    )
}

/// One twin, as the pair's row needs it.
#[derive(Debug, Clone, PartialEq)]
pub struct Twin {
    pub node_id: String,
    pub capsule_id: String,
    /// The answer text.
    pub text: String,
    /// Sampled: a temperature above 0.
    pub sampled: bool,
    pub model_hash: Option<String>,
    pub weights_digest: Option<String>,
}

/// Two twins of one request.
#[derive(Debug, Clone, PartialEq)]
pub struct Pair {
    /// The twin bracket id a client sent; `None` when no client marked a pair.
    pub twin_bracket_id: Option<String>,
    pub request_digest: String,
    pub twins: [Twin; 2],
}

/// `(bracket id, request digest, the two node ids sorted)`.
pub type PairKey = (Option<String>, String, [String; 2]);

impl Pair {
    pub fn key(&self) -> PairKey {
        let mut nodes = [self.twins[0].node_id.clone(), self.twins[1].node_id.clone()];
        nodes.sort();
        (
            self.twin_bracket_id.clone(),
            self.request_digest.clone(),
            nodes,
        )
    }
}

pub fn not_adjudicated(reason: &str, because: Option<&str>) -> Value {
    let mut row = json!({"state": STATE_NOT_ADJUDICATED, "reason": reason});
    if let Some(because) = because {
        row["because"] = json!(because);
    }
    row
}

pub fn adjudicated(verdict: &str, referee: &str, tier: u64) -> Value {
    json!({"state": STATE_ADJUDICATED, "verdict": verdict, "referee": referee, "tier": tier})
}

/// The node a row's contradiction counts against; `None` for every other row.
#[cfg(test)]
pub fn counts_against(row: &Value) -> Option<&str> {
    row.get("verdict")?
        .as_str()?
        .strip_prefix(super::verdict::VERDICT_CONTRADICTED_PREFIX)
}

/// The row a pair gets before anyone is asked, when there is one: every
/// reason that needs no referee.
pub fn before_asking(on: bool, pair: &Pair) -> Option<Value> {
    let [a, b] = &pair.twins;
    if !on {
        return Some(not_adjudicated(REASON_OFF, None));
    }
    if pair.twin_bracket_id.as_deref().is_none_or(str::is_empty) {
        return Some(not_adjudicated(REASON_HOST_DOES_NOT_MARK_TWINS, None));
    }
    if a.sampled || b.sampled {
        return Some(not_adjudicated(
            REASON_NOT_COMPARABLE,
            Some(BECAUSE_SAMPLED),
        ));
    }
    let named = |m: &Option<String>| m.as_deref().is_some_and(|m| !m.is_empty());
    if !named(&a.model_hash) || !named(&b.model_hash) {
        return Some(not_adjudicated(
            REASON_NOT_COMPARABLE,
            Some(BECAUSE_MODEL_HASH_UNKNOWN),
        ));
    }
    if a.model_hash != b.model_hash {
        return Some(not_adjudicated(
            REASON_NOT_COMPARABLE,
            Some(BECAUSE_MODEL_HASH_DIFFERS),
        ));
    }
    let known = |w: &Option<String>| w.as_deref().is_some_and(|w| !w.is_empty());
    if !known(&a.weights_digest) || !known(&b.weights_digest) {
        return Some(not_adjudicated(
            REASON_NOT_COMPARABLE,
            Some(BECAUSE_WEIGHTS_UNKNOWN),
        ));
    }
    if a.weights_digest != b.weights_digest {
        return Some(not_adjudicated(
            REASON_NOT_COMPARABLE,
            Some(BECAUSE_WEIGHTS_DIFFER),
        ));
    }
    if compare_transcripts(&a.text, &b.text)
        .divergence_index
        .is_none()
    {
        return Some(not_adjudicated(REASON_TWINS_AGREE, None));
    }
    None
}

/// Who selection chose.
#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Chosen {
    Referee { node_id: String, tier: u64 },
    NoEligible,
}

/// What the one call came back with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Called {
    /// The referee signed this ruling.
    Verdict(String),
    /// It answered, but cannot sign a verdict (it runs no plugin).
    CannotSign,
    /// It did not answer.
    Unreachable,
}

/// One attempt's result.
#[derive(Debug, Clone, PartialEq)]
pub struct Attempted {
    /// Referee calls this attempt made: 0 or 1.
    pub calls: u32,
    pub row: Value,
}

/// The pairs this node has reached an outcome for.
#[derive(Debug, Default)]
pub struct Book {
    /// Pairs whose one call is used.
    asked: HashMap<PairKey, Value>,
    /// Pairs that found no eligible referee (no call made).
    nobody: HashMap<PairKey, Value>,
}

impl Book {
    /// Decide `pair` (seen again, or asked about by the operator when
    /// `manual`): `choose` picks the referee, `call` asks it. Neither runs
    /// when the pair needs no referee or has used its one call. The live
    /// path (`live::consider`) takes the same steps with an async call.
    #[cfg(test)]
    pub fn attempt(
        &mut self,
        on: bool,
        pair: &Pair,
        manual: bool,
        choose: impl FnOnce() -> Chosen,
        call: impl FnOnce(&str, u64) -> Called,
    ) -> Attempted {
        if let Some(done) = self.gate(on, pair, manual) {
            return done;
        }
        match choose() {
            Chosen::NoEligible => self.nobody_eligible(pair),
            Chosen::Referee { node_id, tier } => {
                let called = call(&node_id, tier);
                self.called(pair, &node_id, tier, called)
            }
        }
    }

    /// The answer when no referee is to be chosen for `pair` now: it needs
    /// none, has used its one call, or found nobody before and is not being
    /// asked again. `None`: choose one.
    pub fn gate(&self, on: bool, pair: &Pair, manual: bool) -> Option<Attempted> {
        let key = pair.key();
        if let Some(row) = self.asked.get(&key) {
            if !cfg!(feature = "mutant-referee-request-asks-twice-per-pair") {
                return Some(Attempted {
                    calls: 0,
                    row: row.clone(),
                });
            }
        }
        if let Some(row) = self.nobody.get(&key) {
            if !manual && !NO_ELIGIBLE_IS_RETRIED_AUTOMATICALLY {
                return Some(Attempted {
                    calls: 0,
                    row: row.clone(),
                });
            }
        }
        before_asking(on, pair).map(|row| Attempted { calls: 0, row })
    }

    /// Selection found nobody: no call made, the pair's one call unused.
    pub fn nobody_eligible(&mut self, pair: &Pair) -> Attempted {
        let row = not_adjudicated(REASON_NO_ELIGIBLE_REFEREE, None);
        self.nobody.insert(pair.key(), row.clone());
        Attempted { calls: 0, row }
    }

    /// The one call was made to `referee`, asked from `tier`.
    pub fn called(&mut self, pair: &Pair, referee: &str, tier: u64, called: Called) -> Attempted {
        let row = match called {
            Called::Verdict(verdict) => adjudicated(&verdict, referee, tier),
            Called::CannotSign => not_adjudicated(REASON_REFEREE_CANNOT_SIGN, None),
            Called::Unreachable => not_adjudicated(REASON_REFEREE_UNREACHABLE, None),
        };
        let key = pair.key();
        self.nobody.remove(&key);
        self.asked.insert(key, row.clone());
        Attempted { calls: 1, row }
    }

    /// The book as this node's pairs file left it.
    pub fn load(ledger_dir: &Path) -> Self {
        let mut book = Self::default();
        let Ok(text) = std::fs::read_to_string(ledger_dir.join(PAIRS_FILENAME)) else {
            return book;
        };
        for line in text
            .lines()
            .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        {
            let Some(key) = key_from(&line["pair"]) else {
                continue;
            };
            let row = line["row"].clone();
            if line["called"] == json!(true) {
                book.nobody.remove(&key);
                book.asked.insert(key, row);
            } else if row["reason"] == json!(REASON_NO_ELIGIBLE_REFEREE) {
                book.nobody.insert(key, row);
            }
        }
        book
    }
}

fn key_from(value: &Value) -> Option<PairKey> {
    let nodes = value.get("node_ids")?.as_array()?;
    let [a, b] = nodes.as_slice() else {
        return None;
    };
    Some((
        value
            .get("twin_bracket_id")
            .and_then(Value::as_str)
            .map(str::to_string),
        value.get("request_digest")?.as_str()?.to_string(),
        [a.as_str()?.to_string(), b.as_str()?.to_string()],
    ))
}

/// Record one attempt's outcome for `pair` beside the ledger, and return the
/// line written.
pub fn record(
    ledger_dir: &Path,
    pair: &Pair,
    attempted: &Attempted,
    at: &str,
) -> std::io::Result<Value> {
    write_line(ledger_dir, pair, attempted, at, None)
}

fn write_line(
    ledger_dir: &Path,
    pair: &Pair,
    attempted: &Attempted,
    at: &str,
    in_flight: Option<Value>,
) -> std::io::Result<Value> {
    use std::io::Write;
    let (bracket, request_digest, nodes) = pair.key();
    let mut line = json!({
        "pair": {"twin_bracket_id": bracket, "request_digest": request_digest, "node_ids": nodes},
        "capsule_ids": [pair.twins[0].capsule_id, pair.twins[1].capsule_id],
        "row": attempted.row,
        "called": attempted.calls > 0,
        "at": at,
    });
    if let Some(in_flight) = in_flight {
        line["in_flight"] = in_flight;
    }
    std::fs::create_dir_all(ledger_dir)?;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(ledger_dir.join(PAIRS_FILENAME))?;
    writeln!(file, "{line}")?;
    file.sync_data()?;
    Ok(line)
}

/// Record, BEFORE the referee is asked, that `pair`'s one call is being made
/// to `referee` from `tier`. Until the answer is recorded the pair reads as a
/// call made and not answered, so if this node stops mid-call, the call is
/// still used after a restart: no further call is made on its own.
pub fn record_in_flight(
    ledger_dir: &Path,
    pair: &Pair,
    referee: &str,
    tier: u64,
    at: &str,
) -> std::io::Result<()> {
    let attempted = Attempted {
        calls: 1,
        row: not_adjudicated(REASON_REFEREE_UNREACHABLE, None),
    };
    write_line(
        ledger_dir,
        pair,
        &attempted,
        at,
        Some(json!({"referee": referee, "tier": tier})),
    )
    .map(|_| ())
}

/// The pairs this process is deciding right now. One decision per pair at a
/// time: the exchange handler and the operator's "ask again" take the same
/// marker, so the two can never both make the pair's one call.
fn deciding() -> &'static std::sync::Mutex<std::collections::HashSet<PairKey>> {
    static DECIDING: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<PairKey>>> =
        std::sync::OnceLock::new();
    DECIDING.get_or_init(Default::default)
}

/// Held while a pair is being decided; released when dropped.
pub struct Deciding(PairKey);

impl Deciding {
    /// The marker for `pair`, or `None` when it is already being decided.
    pub fn claim(pair: &Pair) -> Option<Self> {
        let key = pair.key();
        let mut set = deciding()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Built only when the claim succeeds: a marker built and dropped
        // here would take this same lock in its `Drop` and never return.
        set.insert(key.clone()).then(|| Self(key))
    }
}

impl Drop for Deciding {
    fn drop(&mut self) {
        deciding()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&self.0);
    }
}

/// The latest row for each twin bracket this node has an outcome for.
pub fn rows_by_bracket(ledger_dir: &Path) -> HashMap<String, Value> {
    let mut rows = HashMap::new();
    let Ok(text) = std::fs::read_to_string(ledger_dir.join(PAIRS_FILENAME)) else {
        return rows;
    };
    for line in text
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
    {
        if let Some(bracket) = line
            .pointer("/pair/twin_bracket_id")
            .and_then(Value::as_str)
        {
            rows.insert(bracket.to_string(), line["row"].clone());
        }
    }
    rows
}

/// Put each twin row's pair outcome on it (`twin.referee_row`, `null` while
/// the pair has none), and say on the list whether any client marked a pair
/// (`referee.twins_marked`) and whether the check is on. With no twin bracket
/// on any row, the page reads "Not adjudicated: no client marked these as a
/// pair": no pair is ever guessed from timing.
pub fn attach_rows(pane: &mut Value, ledger_dir: &Path) {
    let rows = rows_by_bracket(ledger_dir);
    let mut marked = false;
    if let Some(list) = pane.get_mut("rows").and_then(Value::as_array_mut) {
        for row in list {
            let Some(bracket) = row
                .get("twin_bracket_id")
                .and_then(Value::as_str)
                .map(str::to_string)
            else {
                continue;
            };
            marked = true;
            if let Some(twin) = row.get_mut("twin").filter(|t| t.is_object()) {
                twin["referee_row"] = rows.get(&bracket).cloned().unwrap_or(Value::Null);
            }
        }
    }
    pane["referee"] = json!({
        "adjudicate_differing_twins": adjudicate_differing_twins(),
        "twins_marked": marked,
        "unmarked_reason": (!marked).then_some(REASON_HOST_DOES_NOT_MARK_TWINS),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn twin(node: &str, text: &str) -> Twin {
        Twin {
            node_id: node.into(),
            capsule_id: format!("{node}-record"),
            text: text.into(),
            sampled: false,
            model_hash: Some("m".into()),
            weights_digest: Some("w".into()),
        }
    }

    fn pair(bracket: Option<&str>) -> Pair {
        Pair {
            twin_bracket_id: bracket.map(str::to_string),
            request_digest: "r".into(),
            twins: [twin("node-a", "1 2 3"), twin("node-b", "1 2 4")],
        }
    }

    fn chosen() -> Chosen {
        Chosen::Referee {
            node_id: "node-c".into(),
            tier: 1,
        }
    }

    /// Decided (c): no eligible referee, no call, no automatic retry; the
    /// operator may ask again.
    #[test]
    fn no_eligible_referee_is_not_retried_on_its_own() {
        const { assert!(!NO_ELIGIBLE_IS_RETRIED_AUTOMATICALLY) };
        let mut book = Book::default();
        let p = pair(Some("bracket-1"));
        let first = book.attempt(
            true,
            &p,
            false,
            || Chosen::NoEligible,
            |_, _| unreachable!(),
        );
        assert_eq!(
            (first.calls, first.row["reason"].clone()),
            (0, json!(REASON_NO_ELIGIBLE_REFEREE))
        );
        let again = book.attempt(true, &p, false, chosen, |_, _| {
            panic!("never retried on its own")
        });
        assert_eq!(again.calls, 0);
        let asked = book.attempt(true, &p, true, chosen, |_, _| {
            Called::Verdict("contradicted:node-b".into())
        });
        assert_eq!(
            asked.calls, 1,
            "the operator asking again makes the one call"
        );
        let after = book.attempt(true, &p, true, chosen, |_, _| {
            panic!("the one call is used")
        });
        assert_eq!(after.calls, 0);
    }

    /// A re-ask after a referee that did not answer makes no further call.
    #[test]
    fn a_reask_after_an_unreachable_referee_makes_no_further_call() {
        let mut book = Book::default();
        let p = pair(Some("bracket-1"));
        let first = book.attempt(true, &p, false, chosen, |_, _| Called::Unreachable);
        assert_eq!(first.calls, 1);
        let again = book.attempt(true, &p, true, chosen, |_, _| panic!("no second call"));
        assert_eq!(
            (again.calls, again.row["reason"].clone()),
            (0, json!(REASON_REFEREE_UNREACHABLE))
        );
    }

    #[test]
    fn no_bracket_id_calls_nothing_and_says_so() {
        let mut book = Book::default();
        let out = book.attempt(
            true,
            &pair(None),
            false,
            || panic!("nobody chosen"),
            |_, _| panic!("no call"),
        );
        assert_eq!(
            out.row,
            not_adjudicated(REASON_HOST_DOES_NOT_MARK_TWINS, None)
        );
        assert_eq!(counts_against(&out.row), None);
    }

    #[test]
    fn the_book_survives_a_restart() {
        let dir = tempfile::tempdir().unwrap();
        let p = pair(Some("bracket-1"));
        let mut book = Book::default();
        let out = book.attempt(true, &p, false, chosen, |_, _| Called::Unreachable);
        record(dir.path(), &p, &out, "2026-09-29T00:00:00Z").unwrap();
        let mut reloaded = Book::load(dir.path());
        let again = reloaded.attempt(true, &p, true, chosen, |_, _| {
            panic!("the one call was used before")
        });
        assert_eq!(again.calls, 0);
        assert_eq!(
            rows_by_bracket(dir.path())["bracket-1"]["reason"],
            json!(REASON_REFEREE_UNREACHABLE)
        );
    }

    #[test]
    fn the_switch_is_off_unless_turned_on() {
        assert!(!adjudicate_differing_twins_from(None));
        for on in ["1", "on", "true", "Yes", " on "] {
            assert!(adjudicate_differing_twins_from(Some(on)), "{on}");
        }
        for off in ["0", "off", "false", "No", "", "garbage"] {
            assert!(!adjudicate_differing_twins_from(Some(off)), "{off}");
        }
    }

    #[test]
    fn rows_without_a_twin_bracket_say_the_host_does_not_mark_twins() {
        let dir = tempfile::tempdir().unwrap();
        let mut pane = json!({"rows": [{"exchange_key": "x"}]});
        attach_rows(&mut pane, dir.path());
        assert_eq!(pane["referee"]["twins_marked"], json!(false));
        assert_eq!(
            pane["referee"]["unmarked_reason"],
            json!(REASON_HOST_DOES_NOT_MARK_TWINS)
        );

        let p = pair(Some("bracket-1"));
        let out = Book::default().attempt(
            true,
            &p,
            false,
            || Chosen::NoEligible,
            |_, _| unreachable!(),
        );
        record(dir.path(), &p, &out, "2026-09-29T00:00:00Z").unwrap();
        let mut pane = json!({"rows": [{"twin_bracket_id": "bracket-1", "twin": {"bracket_id": "bracket-1"}}]});
        attach_rows(&mut pane, dir.path());
        assert_eq!(pane["referee"]["twins_marked"], json!(true));
        assert_eq!(
            pane["rows"][0]["twin"]["referee_row"]["reason"],
            json!(REASON_NO_ELIGIBLE_REFEREE)
        );
    }

    #[test]
    fn twins_with_no_model_named_are_not_comparable() {
        let mut p = pair(Some("bracket-1"));
        p.twins[0].model_hash = None;
        p.twins[1].model_hash = None;
        let out = Book::default().attempt(
            true,
            &p,
            false,
            || panic!("nobody chosen"),
            |_, _| panic!("no call"),
        );
        assert_eq!(
            out.row,
            not_adjudicated(REASON_NOT_COMPARABLE, Some(BECAUSE_MODEL_HASH_UNKNOWN))
        );
        p.twins[1].model_hash = Some("m".into());
        assert_eq!(
            before_asking(true, &p),
            Some(not_adjudicated(
                REASON_NOT_COMPARABLE,
                Some(BECAUSE_MODEL_HASH_UNKNOWN)
            ))
        );
    }

    /// A call recorded as in flight, and never answered (the node stopped
    /// mid-call), has used the pair's one call after a restart.
    #[test]
    fn a_call_recorded_in_flight_is_the_one_call_after_a_restart() {
        let dir = tempfile::tempdir().unwrap();
        let p = pair(Some("bracket-1"));
        record_in_flight(dir.path(), &p, "node-c", 1, "2026-09-29T00:00:00Z").unwrap();
        let mut book = Book::load(dir.path());
        let again = book.attempt(true, &p, true, chosen, |_, _| panic!("no second call"));
        assert_eq!(
            (again.calls, again.row["reason"].clone()),
            (0, json!(REASON_REFEREE_UNREACHABLE))
        );
        assert!(book.gate(true, &p, false).is_some(), "nor on its own");
    }

    /// The handler and "ask again" share one marker per pair.
    #[test]
    fn only_one_decision_per_pair_at_a_time() {
        let p = pair(Some("bracket-1"));
        let first = Deciding::claim(&p).expect("free");
        assert!(
            Deciding::claim(&p).is_none(),
            "a second decision waits its turn"
        );
        assert!(
            Deciding::claim(&pair(Some("bracket-2"))).is_some(),
            "another pair is free"
        );
        drop(first);
        assert!(
            Deciding::claim(&p).is_some(),
            "released when the first is done"
        );
    }
}
