//! The served summary (`served_summary/1`): what this node served, per model,
//! over its checkpointed range. Never who asked.
//!
//! A port of the reference served summary's fold and its published value
//! (`build_served_summary`, `ServedSummary.to_value`), answered in-process by
//! `evidence_answer` under the `served_summary/1` derivation.
//!
//! The fold reads six facts per record, and no requester-identity field:
//! `effect.status`, and from the `x-mesh-poc-v1` block `latency_ms`,
//! `serving_provenance.model_canonical_ref`, `.quantization`,
//! `.weights_digest`, and the record's role. Only records this node served
//! are counted (`evidence_panes::label_role`), and padding never is.
//!
//! The facts are extracted once per ledger line, when `evidence_log` indexes
//! it ([`LeafFacts::of`]), so answering never re-reads record bodies.
//!
//! **Floor.** A model with fewer than [`FLOOR`] served exchanges publishes
//! [`FLOOR_LABEL`] for every count and no latency figures, so a rarely served
//! model cannot be used to infer one exchange.
//!
//! One departure from the Python, stated: a `latency_ms` that is not a finite
//! number (`NaN`, `inf`) is not counted. The Python would fold it into the
//! percentiles and publish `nan`.
//!
//! **Two definitions, two digests.** The fold (`by_model`) is
//! `mesh.served_summary_fold/1`, [`DEFINITION_DIGEST`]
//! (`derivations/served-summary-fold-1.json`), named inside the summary as
//! `derivation.definition_digest`. The whole artifact is `served_summary/1`,
//! [`SERVED_SUMMARY_DEFINITION_DIGEST`] (`derivations/served-summary-1.json`):
//! the fold, `adjudications_received` and `selection`, every field they read,
//! and the fixed members' exact values. That second digest names the
//! derivation on the wire ([`is_derivation`]) and in the artifact. The fold's
//! digest never names a request: it identifies a part of the artifact, not
//! the artifact.
//!
//! **The covered range.** Everything the summary counts is under the anchor
//! checkpoint: the fold, and the verdicts in `adjudications_received`, which
//! count only when the half they judge is under it too. A record appended
//! after the anchor never changes the summary for that anchor.

use evidence_request::request::Derivation;
use serde_json::{json, Value};

pub const SERVED_SUMMARY_SCHEMA: &str = "mesh-served-summary/1";
/// The bare token, a legacy alias for [`SERVED_SUMMARY_DEFINITION_DIGEST`]:
/// accepted on requests through 0.1.8, refused `derivation_unsupported` from
/// 0.1.9. New requesters send the digest.
pub const SERVED_SUMMARY_DERIVATION_TOKEN: &str = "served_summary/1";
pub const FLOOR: u64 = 5;
pub const FLOOR_LABEL: &str = "fewer than 5";

/// The digest of the fold's definition document, `mesh.served_summary_fold/1`
/// (`served_summary.MESH_SERVED_SUMMARY_DEFINITION_DIGEST`: SHA-256 over the
/// RFC 8785 bytes of the account definition). A fixed document, so a fixed
/// digest; it moves only if the definition's reads or class change.
pub const DEFINITION_DIGEST: &str =
    "6eb6f1870dbcee613573c60692b79ab979e2a6188a0f4c6191d4af65a7b773b8";

/// The fold's definition document. Only its digest is used at run time; a
/// test pins one to the other.
#[cfg(test)]
pub const FOLD_DEFINITION_DOCUMENT: &str =
    include_str!("../derivations/served-summary-fold-1.json");

/// The `served_summary/1` definition document. Only its digest is used at
/// run time; tests pin one to the other, and the fixed values to what
/// [`summary_value`] emits.
#[cfg(test)]
pub const DEFINITION_DOCUMENT: &str = include_str!("../derivations/served-summary-1.json");

/// SHA-256 over the RFC 8785 bytes of [`DEFINITION_DOCUMENT`]: the
/// derivation's name on the wire, and the artifact's `derivation`.
pub const SERVED_SUMMARY_DEFINITION_DIGEST: &str =
    "d494544ca59a4d7f997e4d59f87a83670d08f555cae65fb3752614f669757260";

/// The fixed statements every summary carries, exactly as the definition
/// document states them.
const NO_REQUESTER_IDENTIFIERS: &str =
    "this summary reads and reports no requester-identity field \
     (cross_party.initiator_ref, serving_provenance.requesting_party/counterparty_ref) \
     -- it answers WHAT was served, never WHO asked";
const NOT_A_SCORE: &str = "An account of facts + a witness handle to verify them, not a score or \
     routing recommendation.";

/// Whether `derivation` names the served summary: its definition digest, or
/// the legacy token.
pub fn is_derivation(derivation: Option<&Derivation>) -> bool {
    match derivation {
        Some(Derivation::Digest(d)) => d == SERVED_SUMMARY_DEFINITION_DIGEST,
        Some(Derivation::Token(t)) => t == SERVED_SUMMARY_DERIVATION_TOKEN,
        None => false,
    }
}

const REFUSED_STATUS_ABSENT_REASON: &str =
    "agent_action_capsule.contracts.EFFECT_STATUSES carries no distinct 'refused' value; a \
     pre-dispatch policy refusal is coarsened to effect.status='failed' before this module sees \
     it (mesh_record_emitter.py's policy_denied terminal state) -- refused is always 0 until a \
     status value distinguishes it, never inferred from other fields";
const WEIGHTS_DIGEST_ABSENT_REASON: &str =
    "no capsule field named weights_digest exists in records this sidecar emits today \
     (same absence self_accountability.rung_summary documents) -- never fabricated";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StatusBucket {
    Completed,
    Failed,
    Unknown,
}

/// The fold's facts about one record this node served.
#[derive(Clone, Debug, PartialEq)]
pub struct ServedFacts {
    model: String,
    quantization: String,
    bucket: StatusBucket,
    latency_ms: Option<f64>,
    weights_digest_present: bool,
}

/// What the served summary needs from one ledger line.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LeafFacts {
    /// `Some` for a record this node served (never padding).
    pub served: Option<ServedFacts>,
    /// For an adjudication verdict record: the two halves it judges.
    pub adjudicated_halves: Option<(Option<String>, Option<String>)>,
}

impl LeafFacts {
    pub fn of(record: &Value) -> Self {
        if crate::producer::padding::is_padding(record) {
            return Self::default();
        }
        let served = (crate::evidence_panes::label_role(record) == "served").then(|| {
            let poc = record.pointer("/model_attestation/compute_attestation/x-mesh-poc-v1");
            let provenance = poc.and_then(|p| p.get("serving_provenance"));
            let text_or_unknown = |key: &str| {
                provenance
                    .and_then(|sp| sp.get(key))
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .unwrap_or("unknown")
                    .to_string()
            };
            ServedFacts {
                model: text_or_unknown("model_canonical_ref"),
                quantization: text_or_unknown("quantization"),
                bucket: match record.pointer("/effect/status").and_then(Value::as_str) {
                    Some("confirmed") => StatusBucket::Completed,
                    Some("failed" | "reverted") => StatusBucket::Failed,
                    _ => StatusBucket::Unknown,
                },
                latency_ms: poc.and_then(|p| p.get("latency_ms")).and_then(latency),
                weights_digest_present: provenance
                    .and_then(|sp| sp.get("weights_digest"))
                    .is_some_and(|w| !w.is_null()),
            }
        });
        let adjudicated_halves = crate::producer::capsule::is_adjudication(record)
            .then(|| record.pointer("/model_attestation/compute_attestation/adjudication"))
            .flatten()
            .filter(|a| is_truthy(a))
            .map(|a| {
                let half = |k: &str| a.get(k).and_then(Value::as_str).map(str::to_string);
                (half("half_a_capsule_id"), half("half_b_capsule_id"))
            });
        Self {
            served,
            adjudicated_halves,
        }
    }
}

/// Python truthiness of a JSON value, as `if not adjudication` reads it.
fn is_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// `float(raw)`: a number, a numeric string or a bool; anything else, and a
/// value that is not finite, is no latency.
fn latency(raw: &Value) -> Option<f64> {
    let v = match raw {
        Value::Number(n) => n.as_f64()?,
        Value::String(s) => s.trim().parse::<f64>().ok()?,
        Value::Bool(b) => f64::from(u8::from(*b)),
        _ => return None,
    };
    v.is_finite().then_some(v)
}

/// Linear-interpolation percentile over a sorted, non-empty list
/// (`served_summary._percentile`, the same operations in the same order).
fn percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.len() == 1 {
        return sorted[0];
    }
    let k = (sorted.len() - 1) as f64 * pct;
    let (lo, hi) = (k.floor(), k.ceil());
    if lo == hi {
        return sorted[k as usize];
    }
    sorted[lo as usize] * (hi - k) + sorted[hi as usize] * (k - lo)
}

#[derive(Default)]
struct Bucket {
    completed: u64,
    failed: u64,
    latencies: Vec<f64>,
    weights_present: u64,
    weights_absent: u64,
    quantizations: std::collections::BTreeSet<String>,
}

impl Bucket {
    fn to_value(&self) -> Value {
        let served = self.completed + self.failed;
        let below_floor = served < FLOOR;
        let count = |n: u64| {
            if below_floor {
                json!(FLOOR_LABEL)
            } else {
                json!(n)
            }
        };
        let mut latencies = self.latencies.clone();
        latencies.sort_by(f64::total_cmp);
        let figure = |f: &dyn Fn(&[f64]) -> f64| {
            if below_floor || latencies.is_empty() {
                Value::Null
            } else {
                json!(format!("{:.3}", f(&latencies)))
            }
        };
        json!({
            "served": count(served),
            "completed": count(self.completed),
            "failed": count(self.failed),
            "refused": count(0),
            "refused_note": if served > 0 { json!(REFUSED_STATUS_ABSENT_REASON) } else { Value::Null },
            "latency_p50_ms": figure(&|l| percentile(l, 0.50)),
            "latency_p95_ms": figure(&|l| percentile(l, 0.95)),
            "latency_max_ms": figure(&|l| l[l.len() - 1]),
            "weights_digest": {
                "present": count(self.weights_present),
                "absent": count(self.weights_absent),
                "note": if self.weights_present == 0 { json!(WEIGHTS_DIGEST_ABSENT_REASON) } else { Value::Null },
            },
            "quantizations": if below_floor { json!([]) } else { json!(self.quantizations) },
            "floor_applied": below_floor,
        })
    }
}

/// The checkpoint a summary is bounded by.
pub struct Coverage<'a> {
    pub checkpoint: &'a cll::checkpoint::CheckpointRecord,
    /// Records the checkpoint covers (leaf count).
    pub covered: u64,
}

/// The published summary (`ServedSummary.to_value()`), over `leaves` (every
/// ledger line's facts, in log order) bounded by `coverage`. `leaf_index_of`
/// gives the leaf index of one of this node's own records by capsule id (for
/// the adjudications-received count).
pub fn summary_value(
    leaves: &[&LeafFacts],
    coverage: &Coverage<'_>,
    leaf_index_of: &dyn Fn(&str) -> Option<u64>,
) -> Value {
    let cp = coverage.checkpoint;
    let covered = coverage.covered.min(leaves.len() as u64);
    let mut by_model: std::collections::BTreeMap<&str, Bucket> = Default::default();
    for facts in leaves.iter().take(covered as usize) {
        let Some(s) = &facts.served else { continue };
        let bucket = by_model.entry(&s.model).or_default();
        match s.bucket {
            StatusBucket::Completed => bucket.completed += 1,
            StatusBucket::Failed => bucket.failed += 1,
            StatusBucket::Unknown => {}
        }
        if let Some(l) = s.latency_ms {
            bucket.latencies.push(l);
        }
        if s.weights_digest_present {
            bucket.weights_present += 1;
        } else {
            bucket.weights_absent += 1;
        }
        bucket.quantizations.insert(s.quantization.clone());
    }
    let by_model: serde_json::Map<String, Value> = by_model
        .iter()
        .map(|(model, bucket)| (model.to_string(), bucket.to_value()))
        .collect();
    let adjudications_received = adjudications_received(leaves, covered, leaf_index_of);
    let witnesses: std::collections::BTreeSet<&str> =
        cp.witnesses.iter().map(|w| w.ts_url.as_str()).collect();
    json!({
        "schema": SERVED_SUMMARY_SCHEMA,
        "node_id": format!("node:{}", cp.key_id.get(..16).unwrap_or(&cp.key_id)),
        "selection": {
            "from_entry": u64::from(covered > 0),
            "to_entry": covered,
            "covered_entries": covered,
            "note": "witnessed range only: entries appended after the latest witnessed \
                     checkpoint are excluded by design, same as account_capsule.py",
        },
        "derivation": {
            "kind": "served_summary_fold",
            "definition_digest": DEFINITION_DIGEST,
            "by_model": by_model,
            "note": "counts + latency distribution per model, over the selected witnessed \
                     range; an ACCOUNT of what was served, not a score. The relying party \
                     computes its own predicate.",
        },
        "coverage": {
            "checkpoint_root": cp.root,
            "mmr_size": cp.mmr_size,
            "log_id": cp.log_id,
            "timestamp": cp.timestamp,
            "witnesses": witnesses,
            "witnessed": !witnesses.is_empty(),
            "note": "cross-check handle: recompute the fold from the node's ledger, check \
                     the ledger against this root, confirm the root was witnessed",
        },
        "adjudications_received": {
            "value": adjudications_received,
            "source": "self_held",
            "note": "verdicts this node has received() into its own log about its served \
                     exchanges -- not this node's own judgment of itself. The counterparty-held \
                     references path is what carries weight in an adversarial reading; this \
                     count is never presented as a substitute for it.",
        },
        "no_requester_identifiers": NO_REQUESTER_IDENTIFIERS,
        "not_a_score": NOT_A_SCORE,
    })
}

/// Verdicts within the covered range that name one of this node's own
/// records within it. A line appended after the checkpoint, a verdict or the
/// record it names, never changes the answer for that checkpoint (R-15).
fn adjudications_received(
    leaves: &[&LeafFacts],
    covered: u64,
    leaf_index_of: &dyn Fn(&str) -> Option<u64>,
) -> usize {
    // Deliberately broken in one CI build: the whole log, as before.
    let whole_log = cfg!(feature = "mutant-served-summary-folds-whole-log");
    let is_covered_own = |id: &str| leaf_index_of(id).is_some_and(|at| whole_log || at < covered);
    leaves
        .iter()
        .take(if whole_log {
            leaves.len()
        } else {
            covered as usize
        })
        .filter_map(|f| f.adjudicated_halves.as_ref())
        .filter(|(a, b)| {
            a.as_deref().is_some_and(is_covered_own) || b.as_deref().is_some_and(is_covered_own)
        })
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_verdict_is_read_whether_it_writes_confirms_or_the_legacy_adjudicates() {
        for relation in ["confirms", "adjudicates"] {
            let record = json!({"chain": {"relation": relation},
                "model_attestation": {"compute_attestation": {"epistemic_type": "adjudication",
                    "adjudication": {"half_a_capsule_id": "a", "half_b_capsule_id": "b"}}}});
            assert_eq!(
                LeafFacts::of(&record).adjudicated_halves,
                Some((Some("a".into()), Some("b".into()))),
                "{relation}"
            );
        }
        let plain = json!({"chain": {"relation": "confirms"},
            "model_attestation": {"compute_attestation": {"adjudication": {"half_a_capsule_id": "a"}}}});
        assert_eq!(LeafFacts::of(&plain).adjudicated_halves, None);
    }

    fn served(model: &str, status: &str, latency: Value) -> Value {
        json!({
            "capsule_id": "c",
            "effect": {"status": status},
            "model_attestation": {"compute_attestation": {"x-mesh-poc-v1": {
                "role": "served", "latency_ms": latency,
                "serving_provenance": {"model_canonical_ref": model, "quantization": "q4",
                                       "requesting_party": "someone"}
            }}}
        })
    }

    fn checkpoint() -> cll::checkpoint::CheckpointRecord {
        cll::checkpoint::CheckpointRecord {
            v: 1,
            kind: "mmr_checkpoint".into(),
            log_id: "log".into(),
            mmr_size: 1,
            root: "ab".into(),
            prev_size: 0,
            prev_root: String::new(),
            key_id: "0123456789abcdef0123".into(),
            timestamp: "2026-09-29T00:00:00Z".into(),
            signature: String::new(),
            witnesses: Vec::new(),
        }
    }

    /// A verdict judging `half` against another node's half.
    fn verdict(half: &str) -> LeafFacts {
        LeafFacts::of(&json!({
            "model_attestation": {"compute_attestation": {"epistemic_type": "adjudication",
                "adjudication": {"half_a_capsule_id": half, "half_b_capsule_id": "theirs"}}}
        }))
    }

    /// The adjudications-received count over `leaves`, the first `covered`
    /// of them checkpointed, where `own` maps each own record to its leaf.
    fn received(leaves: &[LeafFacts], covered: u64, own: &[(&str, u64)]) -> Value {
        let cp = checkpoint();
        let refs: Vec<&LeafFacts> = leaves.iter().collect();
        let index_of = |id: &str| own.iter().find(|(o, _)| *o == id).map(|(_, at)| *at);
        summary_value(
            &refs,
            &Coverage {
                checkpoint: &cp,
                covered,
            },
            &index_of,
        )
    }

    #[test]
    fn a_covered_verdict_on_a_covered_own_record_is_counted() {
        let leaves = [
            LeafFacts::of(&served("m", "confirmed", json!("1"))),
            verdict("mine"),
        ];
        let v = received(&leaves, 2, &[("mine", 0)]);
        assert_eq!(v["adjudications_received"]["value"], json!(1));
    }

    #[test]
    fn a_verdict_appended_after_the_checkpoint_leaves_the_summary_unchanged() {
        let record = LeafFacts::of(&served("m", "confirmed", json!("1")));
        let at_checkpoint = received(std::slice::from_ref(&record), 1, &[("mine", 0)]);
        let grown = received(&[record, verdict("mine")], 1, &[("mine", 0)]);
        assert_eq!(grown["adjudications_received"]["value"], json!(0));
        assert_eq!(grown, at_checkpoint);
    }

    #[test]
    fn a_covered_verdict_naming_a_record_appended_later_is_not_counted() {
        let leaves = [
            LeafFacts::of(&served("m", "confirmed", json!("1"))),
            verdict("later"),
            LeafFacts::of(&served("m", "confirmed", json!("2"))),
        ];
        let v = received(&leaves, 2, &[("mine", 0), ("later", 2)]);
        assert_eq!(v["adjudications_received"]["value"], json!(0));
    }

    #[test]
    fn percentiles_interpolate_like_the_python() {
        let v = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(percentile(&v, 0.5), 2.5);
        assert!((percentile(&v, 0.95) - 3.85).abs() < 1e-12);
        assert_eq!(percentile(&[7.0], 0.95), 7.0);
    }

    #[test]
    fn a_model_below_the_floor_publishes_no_exact_counts_or_latency() {
        let cp = checkpoint();
        let facts: Vec<LeafFacts> = (0..3)
            .map(|_| LeafFacts::of(&served("m", "confirmed", json!("10.5"))))
            .collect();
        let refs: Vec<&LeafFacts> = facts.iter().collect();
        let v = summary_value(
            &refs,
            &Coverage {
                checkpoint: &cp,
                covered: 3,
            },
            &|_| None,
        );
        let m = &v["derivation"]["by_model"]["m"];
        assert_eq!(m["served"], json!(FLOOR_LABEL));
        assert_eq!(m["latency_p50_ms"], Value::Null);
        assert_eq!(m["floor_applied"], json!(true));
        assert_eq!(v["node_id"], json!("node:0123456789abcdef"));
    }

    #[test]
    fn only_the_covered_range_is_folded_and_requesters_never_appear() {
        let cp = checkpoint();
        let facts: Vec<LeafFacts> = (0..7)
            .map(|i| {
                LeafFacts::of(&served(
                    "m",
                    if i == 0 { "failed" } else { "confirmed" },
                    json!(i),
                ))
            })
            .collect();
        let refs: Vec<&LeafFacts> = facts.iter().collect();
        let v = summary_value(
            &refs,
            &Coverage {
                checkpoint: &cp,
                covered: 6,
            },
            &|_| None,
        );
        let m = &v["derivation"]["by_model"]["m"];
        assert_eq!(m["served"], json!(6));
        assert_eq!(m["failed"], json!(1));
        assert_eq!(m["latency_max_ms"], json!("5.000"));
        assert!(!v.to_string().contains("someone"));
    }

    /// The digest of a definition document: SHA-256 over its RFC 8785 bytes.
    fn digest_of_document(text: &str) -> String {
        use sha2::{Digest, Sha256};
        let document: Value = serde_json::from_str(text).unwrap();
        let bytes = evidence_request::jcs::to_string(&document).unwrap();
        hex::encode(Sha256::digest(bytes.as_bytes()))
    }

    #[test]
    fn the_fold_digest_is_the_digest_of_the_fold_definition_file() {
        assert_eq!(
            digest_of_document(FOLD_DEFINITION_DOCUMENT),
            DEFINITION_DIGEST
        );
        let document: Value = serde_json::from_str(FOLD_DEFINITION_DOCUMENT).unwrap();
        assert_eq!(document["name"], json!("mesh.served_summary_fold/1"));
        assert_eq!(document["selection_kind"], json!("range"));
    }

    #[test]
    fn the_served_summary_digest_is_the_digest_of_its_definition_file() {
        assert_eq!(
            digest_of_document(DEFINITION_DOCUMENT),
            SERVED_SUMMARY_DEFINITION_DIGEST
        );
        let document: Value = serde_json::from_str(DEFINITION_DOCUMENT).unwrap();
        assert_eq!(document["name"], json!(SERVED_SUMMARY_DERIVATION_TOKEN));
        let members = document["members"].as_array().unwrap();
        let names: Vec<&str> = members
            .iter()
            .map(|m| m["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, ["derivation", "adjudications_received", "selection"]);
        // The fold is named by its own digest, never as a derivation.
        assert_eq!(members[0]["definition"], json!(DEFINITION_DIGEST));
        // `reads` is the ordered union of every member's and fixed entry's.
        let mut union: Vec<&Value> = Vec::new();
        for entry in members.iter().chain(document["fixed"].as_array().unwrap()) {
            for read in entry["reads"].as_array().into_iter().flatten() {
                if !union.contains(&read) {
                    union.push(read);
                }
            }
        }
        let reads: Vec<&Value> = document["reads"].as_array().unwrap().iter().collect();
        assert_eq!(reads, union);
        let record_reads = reads
            .iter()
            .filter(|r| r.as_str().unwrap().starts_with("record: "))
            .count();
        assert_eq!(record_reads, 13);
    }

    /// The fixed members carry exactly the values the definition states, so
    /// their wording cannot change under an unchanged digest.
    #[test]
    fn every_constant_fixed_member_is_emitted_exactly_as_the_definition_states() {
        let document: Value = serde_json::from_str(DEFINITION_DOCUMENT).unwrap();
        let cp = checkpoint();
        let v = summary_value(
            &[],
            &Coverage {
                checkpoint: &cp,
                covered: 0,
            },
            &|_| None,
        );
        let fixed = document["fixed"].as_array().unwrap();
        let constants: Vec<&Value> = fixed.iter().filter(|f| f.get("value").is_some()).collect();
        assert_eq!(constants.len(), 3);
        for entry in constants {
            let name = entry["name"].as_str().unwrap();
            assert_eq!(v[name], entry["value"], "{name}");
        }
        // Every key the summary carries is a member or a fixed entry.
        let mut declared: Vec<&str> = document["members"]
            .as_array()
            .unwrap()
            .iter()
            .chain(fixed)
            .map(|m| m["name"].as_str().unwrap())
            .collect();
        declared.sort_unstable();
        let mut emitted: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
        emitted.sort_unstable();
        assert_eq!(emitted, declared);
    }
}
