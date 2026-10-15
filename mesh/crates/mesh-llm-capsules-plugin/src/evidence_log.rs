//! This node's ledger as the evidence-request responder sees it: the index
//! an answer is built from (`evidence_answer`) and a [`Responder`] (what a
//! request is resolved against).
//!
//! **Read-only, and its own MMR.** The plugin's checkpoint cadence owns the
//! durable node store (`mmr_nodes.dat`) and is its only writer; opening it a
//! second time would run its torn-write recovery from here. So this index
//! folds `capsules.jsonl` into an in-memory MMR of its own (the same leaves:
//! each line's `capsule_id`, padding included, in file order) and keeps it
//! current incrementally: each answer reads only the complete lines appended
//! since the last one. A file that was replaced or rewritten under the index
//! (its last indexed line no longer matches) is re-indexed from the start.
//!
//! **Only checkpoints this index re-derives.** A checkpoint from
//! `checkpoints.jsonl` joins the chain only if it is signed by this node's
//! key, chains from the stream's first checkpoint, and its root is the root
//! this index computes at its size. The first checkpoint that fails any of
//! these ends the chain: nothing after it is served, so an answer is never
//! signed under a root its proofs would not reach. A line that is not a
//! well-formed record (no 64-hex `capsule_id`) ends the index the same way,
//! as the checkpointer refuses to fold one.
//!
//! **Anchors are stated checkpoints.** An anchor is a chain checkpoint whose
//! line carries its COSE wire statement (`checkpoint_cose`), identified by
//! the SHA-256 of that statement: the `cll-checkpoint` item of the interop
//! profile (agent-action-capsule `spec/interop-profile-v0.md` §6.2.1), the
//! value a requester pins, as evidencebook and the Go responder identify
//! one. The statement must verify under this node's key and state exactly
//! the checkpoint on its line; one that does not, or a line without one
//! (the checkpointer writes it best-effort), stays in the chain but anchors
//! no answer.
//!
//! **What a record is here.** A record's digest is its `capsule_id` (the leaf
//! the checkpoints commit to) and its body is its line in `capsules.jsonl`,
//! byte for byte, producer envelope included. A requester checks one with
//! [`record_digest`]: the body's recomputed `capsule_id`, which must equal the
//! one it states. The body is not the `capsule_id` preimage, so its SHA-256 is
//! not its digest: the interop profile's Checkpointed Local Log check reports
//! such a record `NOT_CHECKED` (profile §6.2.1 step 5), and this node's
//! requester checks it instead (`evidence_artifact`).
//!
//! **Padding** (Evidence Layer -00 §12.1) is a leaf, so a range carries it,
//! but it is never a record: it is not found by digest, correlation or
//! exchange.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use cll::checkpoint::{verify_checkpoint_cose_offline, CheckpointRecord};
use cll::mmr::{
    add_leaf, leaf_count, leaf_hash, peaks, root_from_peaks, Hash, MemoryNodeStore, NodeReader,
};
use evidence_request::registry::{SubjectForm, DERIVATION_HISTORY_CARD};
use evidence_request::request::Derivation;
use evidence_request::resolve::{Anchor, Responder};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::served_summary::{LeafFacts, SERVED_SUMMARY_DERIVATION_TOKEN};

pub const CAPSULES_FILE: &str = "capsules.jsonl";
pub const CHECKPOINTS_FILE: &str = "checkpoints.jsonl";

/// The longest correlation identifier indexed. A longer value in a record is
/// not an identifier anyone asks by; not indexing it bounds the index.
const MAX_CORRELATION_ID_LEN: usize = 256;

/// The members whose text values identify an exchange a record belongs to
/// (a `correlation` subject), wherever they appear in the record. The same
/// correlators the Python reference matched by nonce and exchange id; a
/// counterparty's id is an identity, not a correlation, and is not one.
const CORRELATION_KEYS: [&str; 3] = ["nonce", "client_nonce", "exchange_id"];

/// The `client_nonce` a host-served record carries when the host forwarded
/// none. Every such record carries it, so it names no exchange and is never
/// indexed: asking by it would answer with unrelated records.
pub const HOST_SERVED_NO_NONCE: &str = "host-served-no-nonce";

/// The prefix of an exchange's digest binding (see [`exchange_binding`]).
const EXCHANGE_BINDING_PREFIX: &str = "exchange-digests:";

/// The correlation identifier that names an exchange by both of its digests:
/// `exchange-digests:<request digest>:<response digest>`, the two `effect`
/// digests both sides' records carry. A record is found by it only when both
/// digests are exactly its own. It is how a requester finds the other side's
/// record when that record carries no nonce of its own (a host-served half,
/// [`HOST_SERVED_NO_NONCE`]): the requester knows both digests from its own
/// half.
pub fn exchange_binding(request_digest: &str, response_digest: &str) -> String {
    format!("{EXCHANGE_BINDING_PREFIX}{request_digest}:{response_digest}")
}

/// A record's digest binding, when its `effect` carries both digests.
fn record_exchange_binding(record: &Value) -> Option<String> {
    let effect = record.get("effect")?;
    let request = effect
        .get("request_digest")?
        .as_str()
        .filter(|d| is_digest(d))?;
    let response = effect
        .get("response_digest")?
        .as_str()
        .filter(|d| is_digest(d))?;
    Some(exchange_binding(request, response))
}

/// One line of the ledger.
#[derive(Clone, Debug)]
pub struct Leaf {
    pub capsule_id: String,
    offset: u64,
    len: u64,
    pub padding: bool,
    /// The other side of the exchange this record is of, if it names one
    /// (`evidence_panes::full_counterparty_node_id`).
    pub counterparty: Option<String>,
    /// A local block or an owner-maintenance record: it never leaves this
    /// node, whoever asks.
    pub never_leaves: bool,
    pub facts: LeafFacts,
}

/// A position in an append-only file: how much has been read, and the last
/// complete line read (its offset and digest), to tell an append from a
/// rewrite.
#[derive(Clone, Debug, Default)]
struct Tail {
    consumed: u64,
    last_line: Option<(u64, [u8; 32])>,
}

impl Tail {
    /// Call `f` on each complete line appended since the last call, with its
    /// offset, in order, until `f` returns `false`. Returns `Ok(false)`
    /// without calling `f` when the file no longer continues what was read
    /// (replaced, truncated or rewritten): read it again from the start. A
    /// missing file is an empty one. A line still being written (no newline
    /// yet) is left for the next call.
    fn read_new(
        &self,
        path: &Path,
        mut f: impl FnMut(u64, &[u8]) -> bool,
    ) -> std::io::Result<bool> {
        let mut file = match File::open(path) {
            Ok(file) => file,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(self.consumed == 0),
            Err(e) => return Err(e),
        };
        let len = file.metadata()?.len();
        if len < self.consumed {
            return Ok(false);
        }
        if let Some((offset, digest)) = self.last_line {
            let mut line = vec![0u8; (self.consumed - offset) as usize];
            file.seek(SeekFrom::Start(offset))?;
            file.read_exact(&mut line)?;
            if <[u8; 32]>::from(Sha256::digest(&line)) != digest {
                return Ok(false);
            }
        }
        file.seek(SeekFrom::Start(self.consumed))?;
        let mut reader = std::io::BufReader::new(file.take(len - self.consumed));
        let mut offset = self.consumed;
        let mut line = Vec::new();
        loop {
            line.clear();
            let n = reader.read_until(b'\n', &mut line)?;
            if n == 0 || line.last() != Some(&b'\n') || !f(offset, &line) {
                return Ok(true);
            }
            offset += n as u64;
        }
    }

    fn advance(&mut self, offset: u64, line: &[u8]) {
        self.consumed = offset + line.len() as u64;
        self.last_line = Some((offset, Sha256::digest(line).into()));
    }
}

/// What a consistency proof is made of: the old peaks, each one's path into
/// the new MMR, and the new peaks.
pub type ConsistencyParts = (Vec<Hash>, Vec<Vec<Hash>>, Vec<Hash>);

/// One checkpoint of the chain, with its COSE wire statement when it has one
/// that states it.
#[derive(Clone, Debug)]
pub struct ChainCheckpoint {
    pub record: CheckpointRecord,
    /// The `cll-checkpoint` octets: present only for a statement that
    /// verifies under this node's key and states exactly `record`.
    pub statement: Option<Vec<u8>>,
}

impl ChainCheckpoint {
    /// The digest an anchor is pinned by: the SHA-256 of its statement.
    /// `None` for a checkpoint that anchors no answer.
    pub fn anchor_digest(&self) -> Option<String> {
        self.statement
            .as_ref()
            .map(|s| hex::encode(Sha256::digest(s)))
    }
}

/// The index of one ledger directory.
pub struct LedgerIndex {
    ledger_dir: PathBuf,
    /// This node's key (`key_id` form: raw public key, lowercase hex). Only
    /// checkpoints it signed are anchors.
    node_key_id: String,
    capsules: Tail,
    checkpoints: Tail,
    pub leaves: Vec<Leaf>,
    nodes: MemoryNodeStore,
    by_id: HashMap<String, u64>,
    correlations: HashMap<String, Vec<u64>>,
    citations: HashMap<String, Vec<u64>>,
    /// Why indexing stopped, if it did: nothing after this line is a leaf.
    capsules_stopped: Option<String>,
    /// The valid checkpoint chain, oldest first.
    pub chain: Vec<ChainCheckpoint>,
    /// Why the chain ended, if it did.
    chain_stopped: Option<String>,
}

impl LedgerIndex {
    fn new(ledger_dir: &Path, node_key_id: &str) -> Self {
        Self {
            ledger_dir: ledger_dir.to_path_buf(),
            node_key_id: node_key_id.to_string(),
            capsules: Tail::default(),
            checkpoints: Tail::default(),
            leaves: Vec::new(),
            nodes: MemoryNodeStore::new(),
            by_id: HashMap::new(),
            correlations: HashMap::new(),
            citations: HashMap::new(),
            capsules_stopped: None,
            chain: Vec::new(),
            chain_stopped: None,
        }
    }

    /// Bring the index up to date with both files.
    pub fn refresh(&mut self) -> std::io::Result<()> {
        if self.capsules_stopped.is_none() {
            let path = self.ledger_dir.join(CAPSULES_FILE);
            let start = self.capsules.clone();
            let mut tail = start.clone();
            let continues = start.read_new(&path, |offset, line| {
                if let Err(why) = self.index_line(offset, line) {
                    tracing::warn!(line_at = offset, %why, "evidence index stops at a ledger line it cannot fold");
                    self.capsules_stopped = Some(why);
                    return false;
                }
                tail.advance(offset, line);
                true
            })?;
            if !continues {
                *self = Self::new(&self.ledger_dir, &self.node_key_id);
                return self.refresh();
            }
            self.capsules = tail;
        }
        let path = self.ledger_dir.join(CHECKPOINTS_FILE);
        if !self.take_checkpoints(&path)? {
            self.checkpoints = Tail::default();
            self.chain.clear();
            self.chain_stopped = None;
            self.take_checkpoints(&path)?;
        }
        Ok(())
    }

    fn index_line(&mut self, offset: u64, line: &[u8]) -> Result<(), String> {
        let body = trim_newline(line);
        let record: Value = serde_json::from_slice(body).map_err(|e| format!("not JSON: {e}"))?;
        let capsule_id = record
            .get("capsule_id")
            .and_then(Value::as_str)
            .filter(|id| is_digest(id))
            .ok_or("no 64-hex capsule_id")?
            .to_string();
        let raw: [u8; 32] = hex::decode(&capsule_id)
            .ok()
            .and_then(|b| b.try_into().ok())
            .ok_or("capsule_id is not 32 bytes")?;
        add_leaf(&mut self.nodes, leaf_hash(&raw)).map_err(|e| e.to_string())?;
        let index = self.leaves.len() as u64;
        let padding = crate::producer::padding::is_padding(&record);
        if !padding {
            self.by_id.entry(capsule_id.clone()).or_insert(index);
            let mut ids = Vec::new();
            collect_correlations(&record, &mut ids);
            ids.extend(record_exchange_binding(&record));
            ids.sort();
            ids.dedup();
            for id in ids {
                self.correlations.entry(id).or_default().push(index);
            }
            for half in counterparty_half_citations(&record) {
                self.citations.entry(half).or_default().push(index);
            }
        }
        self.leaves.push(Leaf {
            capsule_id,
            offset,
            len: body.len() as u64,
            padding,
            counterparty: crate::evidence_panes::full_counterparty_node_id(&record),
            never_leaves: crate::evidence_panes::is_local_routing_choice(&record)
                || record
                    .pointer("/model_attestation/compute_attestation")
                    .and_then(|c| c.get(crate::producer::capsule::OWNER_MAINTENANCE_BLOCK))
                    .is_some(),
            facts: LeafFacts::of(&record),
        });
        Ok(())
    }

    /// Extend the chain from the checkpoint lines appended since the last
    /// call. `Ok(false)` when the file was rewritten.
    fn take_checkpoints(&mut self, path: &Path) -> std::io::Result<bool> {
        let start = self.checkpoints.clone();
        let mut tail = start.clone();
        let continues = start.read_new(path, |offset, line| {
            if self.chain_stopped.is_some() {
                return false;
            }
            match self.check_checkpoint(trim_newline(line)) {
                // The ledger has not reached it yet: read it again next time.
                Ok(None) => return false,
                Ok(Some(cp)) => self.chain.push(cp),
                Err(why) => {
                    tracing::warn!(line_at = offset, %why, "evidence anchors stop at a checkpoint this index cannot re-derive");
                    self.chain_stopped = Some(why);
                }
            }
            tail.advance(offset, line);
            true
        })?;
        if continues {
            self.checkpoints = tail;
        }
        Ok(continues)
    }

    /// `Some` for a checkpoint that extends the chain; `None` for one past
    /// what the index holds yet; `Err` for one that ends the chain.
    fn check_checkpoint(&self, body: &[u8]) -> Result<Option<ChainCheckpoint>, String> {
        let mut value: Value =
            serde_json::from_slice(body).map_err(|e| format!("not JSON: {e}"))?;
        let cose_hex = value
            .as_object_mut()
            .and_then(|obj| obj.remove("checkpoint_cose"));
        let cp: CheckpointRecord =
            serde_json::from_value(value).map_err(|e| format!("not a checkpoint: {e}"))?;
        if cp.key_id != self.node_key_id || !cp.verify_signature_offline() {
            return Err("not signed by this node's key".into());
        }
        match self.chain.last().map(|c| &c.record) {
            None if cp.prev_size != 0 || !cp.prev_root.is_empty() => {
                return Err("the first checkpoint does not start the stream".into())
            }
            Some(prev)
                if cp.prev_size != prev.mmr_size
                    || cp.prev_root != prev.root
                    || cp.mmr_size <= prev.mmr_size
                    || cp.log_id != prev.log_id =>
            {
                return Err("does not chain from the previous checkpoint".into())
            }
            _ => {}
        }
        leaf_count(cp.mmr_size).map_err(|e| format!("mmr_size: {e}"))?;
        if cp.mmr_size > self.nodes.size() {
            return if self.capsules_stopped.is_some() {
                Err("covers leaves past where the ledger index stopped".into())
            } else {
                Ok(None)
            };
        }
        let peak_hashes: Vec<_> = peaks(cp.mmr_size)
            .map_err(|e| e.to_string())?
            .iter()
            .map(|&p| self.nodes.node(p))
            .collect();
        if hex::encode(root_from_peaks(&peak_hashes)) != cp.root {
            return Err("its root is not this ledger's root at its size".into());
        }
        let statement = cose_hex
            .as_ref()
            .and_then(Value::as_str)
            .and_then(|h| hex::decode(h).ok())
            .filter(|s| states(s, &cp));
        if statement.is_none() {
            tracing::warn!(
                mmr_size = cp.mmr_size,
                "a checkpoint has no COSE statement of its own: it anchors no evidence answer"
            );
        }
        Ok(Some(ChainCheckpoint {
            record: cp,
            statement,
        }))
    }

    /// The inclusion path (sibling hashes, leaf upward) of `leaf_index` in
    /// the MMR of `size` nodes.
    pub fn inclusion_path(&self, leaf_index: u64, size: u64) -> Option<Vec<Hash>> {
        let proof = cll::mmr::inclusion_proof(&self.nodes, leaf_index, size).ok()?;
        hashes(&proof.witness)
    }

    /// What a consistency proof from the MMR of `from` nodes to the MMR of
    /// `to` nodes is made of: the old peaks, each one's path into the new
    /// MMR, and the new peaks.
    pub fn consistency_parts(&self, from: u64, to: u64) -> Option<ConsistencyParts> {
        let proof = cll::mmr::consistency_proof(&self.nodes, from, to).ok()?;
        Some((
            hashes(&proof.old_peaks)?,
            proof
                .witness
                .iter()
                .map(|p| hashes(p))
                .collect::<Option<_>>()?,
            hashes(&proof.new_peaks)?,
        ))
    }

    pub fn leaf_index_of(&self, capsule_id: &str) -> Option<u64> {
        self.by_id.get(capsule_id).copied()
    }

    pub fn correlated(&self, id: &str) -> &[u64] {
        self.correlations.get(id).map_or(&[], Vec::as_slice)
    }

    pub fn citing(&self, half: &str) -> &[u64] {
        self.citations.get(half).map_or(&[], Vec::as_slice)
    }

    /// A view that reads record bodies from the ledger file.
    pub fn view(&self) -> std::io::Result<LogView<'_>> {
        Ok(LogView {
            index: self,
            file: File::open(self.ledger_dir.join(CAPSULES_FILE)).ok(),
        })
    }
}

/// One process-wide index per ledger directory, kept current across answers.
static INDEXES: Mutex<Option<HashMap<PathBuf, LedgerIndex>>> = Mutex::new(None);

#[cfg(test)]
thread_local! {
    static INDEX_READS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// How many times this thread has read an index ([`with_index`]): a test
/// shows a refusal came before the ledger was read.
#[cfg(test)]
pub fn index_reads() -> u64 {
    INDEX_READS.with(std::cell::Cell::get)
}

/// Run `f` over the up-to-date index of `ledger_dir`. Answers are serialized
/// per process by this lock: the work under it is bounded by the answer
/// limits, and the index is shared state.
pub fn with_index<T>(
    ledger_dir: &Path,
    node_key_id: &str,
    f: impl FnOnce(&LedgerIndex) -> T,
) -> std::io::Result<T> {
    #[cfg(test)]
    INDEX_READS.with(|n| n.set(n.get() + 1));
    let mut guard = INDEXES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let indexes = guard.get_or_insert_with(HashMap::new);
    let index = indexes
        .entry(ledger_dir.to_path_buf())
        .or_insert_with(|| LedgerIndex::new(ledger_dir, node_key_id));
    if index.node_key_id != node_key_id {
        *index = LedgerIndex::new(ledger_dir, node_key_id);
    }
    index.refresh()?;
    Ok(f(index))
}

/// The index plus the ledger file, for reading bodies.
pub struct LogView<'a> {
    pub index: &'a LedgerIndex,
    file: Option<File>,
}

impl LogView<'_> {
    /// The record at `leaf_index`, its body read from the ledger. `None`
    /// when the line is no longer the one indexed.
    pub fn record_at(&self, leaf_index: u64) -> Option<Record> {
        let leaf = self.index.leaves.get(usize::try_from(leaf_index).ok()?)?;
        let mut body = vec![0u8; usize::try_from(leaf.len).ok()?];
        let mut file = self.file.as_ref()?;
        file.seek(SeekFrom::Start(leaf.offset)).ok()?;
        file.read_exact(&mut body).ok()?;
        let parsed: Value = serde_json::from_slice(&body).ok()?;
        (parsed.get("capsule_id").and_then(Value::as_str) == Some(leaf.capsule_id.as_str())).then(
            || Record {
                leaf_index,
                digest: leaf.capsule_id.clone(),
                body,
            },
        )
    }
}

/// One record, as an answer serves it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Record {
    /// 0-based position in the log.
    pub leaf_index: u64,
    /// The `capsule_id`, the entry the log commits.
    pub digest: String,
    /// The record's line in the ledger.
    pub body: Vec<u8>,
}

/// A record's digest as a requester computes it from the body it was sent:
/// the body's recomputed `capsule_id`, when that is the id the body states.
/// `None` for a body that is not a record of this format.
pub fn record_digest(body: &[u8]) -> Option<String> {
    let record = crate::producer::jcs::decode_record(body).ok()?;
    let stated = record.get("capsule_id").and_then(Value::as_str)?;
    let recomputed = crate::producer::jcs::compute_capsule_id(&record).ok()?;
    (recomputed == stated).then_some(recomputed)
}

/// Whether `statement` is this checkpoint's own COSE wire statement: it
/// verifies under its `kid`, which is the checkpoint's key, and it states
/// exactly the checkpoint's log, sizes, roots and time.
fn states(statement: &[u8], cp: &CheckpointRecord) -> bool {
    let verified = verify_checkpoint_cose_offline(statement);
    let Some(decoded) = verified.decoded.filter(|_| verified.ok) else {
        return false;
    };
    decoded.key_id == cp.key_id
        && decoded.to_checkpoint_record().signing_body() == cp.signing_body()
}

fn hashes(hex_list: &[String]) -> Option<Vec<Hash>> {
    hex_list
        .iter()
        .map(|h| hex::decode(h).ok()?.try_into().ok())
        .collect()
}

impl Responder for LogView<'_> {
    fn holds_record(&self, digest: &str) -> bool {
        self.index.leaf_index_of(digest).is_some()
    }
    fn holds_correlation(&self, id: &str) -> bool {
        !self.index.correlated(id).is_empty()
    }
    fn cites_exchange(&self, half_digest: &str) -> bool {
        !self.index.citing(half_digest).is_empty()
    }
    fn positional_ordering(&self) -> bool {
        true
    }
    fn anchors(&self) -> Vec<Anchor> {
        self.index
            .chain
            .iter()
            .filter_map(|cp| {
                Some(Anchor {
                    digest: cp.anchor_digest()?,
                    size: leaf_count(cp.record.mmr_size).ok()?,
                    issued_at: cp.record.timestamp.clone(),
                })
            })
            .collect()
    }
    fn derivation_forms(&self, derivation: &Derivation) -> Option<Vec<SubjectForm>> {
        match derivation {
            Derivation::Token(t) if t == DERIVATION_HISTORY_CARD => {
                Some(vec![SubjectForm::FullHistory, SubjectForm::Checkpoints])
            }
            Derivation::Token(t) if t == SERVED_SUMMARY_DERIVATION_TOKEN => {
                Some(vec![SubjectForm::FullHistory])
            }
            _ => None,
        }
    }
}

fn trim_newline(line: &[u8]) -> &[u8] {
    line.strip_suffix(b"\n").unwrap_or(line)
}

fn is_digest(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

fn collect_correlations(value: &Value, out: &mut Vec<String>) {
    match value {
        Value::Object(map) => {
            for (k, v) in map {
                if CORRELATION_KEYS.contains(&k.as_str()) {
                    if let Some(id) = v.as_str().filter(|s| {
                        !s.is_empty()
                            && s.len() <= MAX_CORRELATION_ID_LEN
                            && *s != HOST_SERVED_NO_NONCE
                    }) {
                        out.push(id.to_string());
                    }
                }
                collect_correlations(v, out);
            }
        }
        Value::Array(items) => items.iter().for_each(|v| collect_correlations(v, out)),
        _ => {}
    }
}

/// The exchange halves a record cites as its counterparty's half
/// (`references[].citation_purpose == "counterparty_half"`), the citation the
/// ledger itself keys on.
fn counterparty_half_citations(record: &Value) -> Vec<String> {
    record
        .get("references")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|r| r.get("citation_purpose").and_then(Value::as_str) == Some("counterparty_half"))
        .filter_map(|r| r.get("digest").and_then(Value::as_str))
        .filter(|d| is_digest(d))
        .map(str::to_string)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence_request_parity::{
        corpus_checkpoints, corpus_ledger, write_jsonl, NODE_KEY_SEED,
    };
    use ed25519_dalek::SigningKey;
    use serde_json::json;

    fn key_id() -> String {
        hex::encode(
            SigningKey::from_bytes(&NODE_KEY_SEED)
                .verifying_key()
                .to_bytes(),
        )
    }

    fn checkpoint_lines(ledger: &[Value], key: &SigningKey) -> Vec<Value> {
        corpus_checkpoints(ledger, key)
            .iter()
            .map(|cp| cp.line())
            .collect()
    }

    #[test]
    fn a_line_still_being_written_is_indexed_once_it_is_complete() {
        let dir = tempfile::tempdir().unwrap();
        let ledger = corpus_ledger();
        let path = dir.path().join(CAPSULES_FILE);
        let mut text: String = ledger[..3].iter().map(|l| format!("{l}\n")).collect();
        text.push_str(&ledger[3].to_string());
        std::fs::write(&path, &text).unwrap();
        let mut index = LedgerIndex::new(dir.path(), &key_id());
        index.refresh().unwrap();
        assert_eq!(index.leaves.len(), 3);
        text.push('\n');
        std::fs::write(&path, &text).unwrap();
        index.refresh().unwrap();
        assert_eq!(index.leaves.len(), 4);
    }

    #[test]
    fn a_rewritten_ledger_is_indexed_again_from_the_start() {
        let dir = tempfile::tempdir().unwrap();
        let ledger = corpus_ledger();
        let path = dir.path().join(CAPSULES_FILE);
        write_jsonl(&path, &ledger[..4]);
        let mut index = LedgerIndex::new(dir.path(), &key_id());
        index.refresh().unwrap();
        // Same length or longer, different lines: not an append.
        let mut rewritten = ledger[4..8].to_vec();
        rewritten.push(ledger[8].clone());
        write_jsonl(&path, &rewritten);
        index.refresh().unwrap();
        let ids: Vec<&str> = index.leaves.iter().map(|l| l.capsule_id.as_str()).collect();
        let want: Vec<&str> = rewritten
            .iter()
            .map(|l| l["capsule_id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, want);
    }

    #[test]
    fn only_checkpoints_this_index_re_derives_are_anchors() {
        let ledger = corpus_ledger();
        let key = SigningKey::from_bytes(&NODE_KEY_SEED);

        // Both checkpoints re-derive.
        let dir = tempfile::tempdir().unwrap();
        write_jsonl(&dir.path().join(CAPSULES_FILE), &ledger);
        write_jsonl(
            &dir.path().join(CHECKPOINTS_FILE),
            &checkpoint_lines(&ledger, &key),
        );
        let mut index = LedgerIndex::new(dir.path(), &key_id());
        index.refresh().unwrap();
        assert_eq!(index.chain.len(), 2);

        // Signed by another key: no anchors.
        let other = SigningKey::from_bytes(&[7u8; 32]);
        write_jsonl(
            &dir.path().join(CHECKPOINTS_FILE),
            &checkpoint_lines(&ledger, &other),
        );
        let mut index = LedgerIndex::new(dir.path(), &key_id());
        index.refresh().unwrap();
        assert!(index.chain.is_empty());

        // Over a different ledger (one leaf changed): the chain ends before
        // the checkpoint whose root no longer matches.
        let mut tampered = ledger.clone();
        tampered[20] = ledger[19].clone();
        write_jsonl(&dir.path().join(CAPSULES_FILE), &tampered);
        write_jsonl(
            &dir.path().join(CHECKPOINTS_FILE),
            &checkpoint_lines(&ledger, &key),
        );
        let mut index = LedgerIndex::new(dir.path(), &key_id());
        index.refresh().unwrap();
        assert_eq!(
            index.chain.len(),
            1,
            "checkpoint 0 re-derives; checkpoint 1 does not"
        );
    }

    #[test]
    fn a_checkpoint_ahead_of_the_ledger_waits_for_it() {
        let ledger = corpus_ledger();
        let key = SigningKey::from_bytes(&NODE_KEY_SEED);
        let dir = tempfile::tempdir().unwrap();
        write_jsonl(&dir.path().join(CAPSULES_FILE), &ledger[..100]);
        write_jsonl(
            &dir.path().join(CHECKPOINTS_FILE),
            &checkpoint_lines(&ledger, &key),
        );
        let mut index = LedgerIndex::new(dir.path(), &key_id());
        index.refresh().unwrap();
        assert_eq!(index.chain.len(), 1);
        write_jsonl(&dir.path().join(CAPSULES_FILE), &ledger);
        index.refresh().unwrap();
        assert_eq!(index.chain.len(), 2);
    }

    /// A host-served half carries the placeholder nonce, never the one the
    /// requester forwarded: it is found by its two digests, only both
    /// exactly, and never by the placeholder every such half shares.
    #[test]
    fn a_host_served_half_is_found_by_its_digests_not_its_placeholder_nonce() {
        let (req, resp) = ("a".repeat(64), "b".repeat(64));
        let half = |req: &str, resp: &str| {
            crate::evidence_request_parity::sealed(json!({
                "effect": {"status": "confirmed", "request_digest": req, "response_digest": resp},
                "model_attestation": {"compute_attestation": {"x-mesh-poc-v1": {
                    "role": "served",
                    "client_nonce": HOST_SERVED_NO_NONCE,
                    "client_nonce_source": "host_served_observed",
                }}},
            }))
        };
        let ours = half(&req, &resp);
        let other = half(&"c".repeat(64), &resp);
        let dir = tempfile::tempdir().unwrap();
        write_jsonl(&dir.path().join(CAPSULES_FILE), &[other, ours.clone()]);
        let mut index = LedgerIndex::new(dir.path(), &key_id());
        index.refresh().unwrap();
        assert_eq!(index.correlated(&exchange_binding(&req, &resp)), &[1]);
        assert!(index.correlated(HOST_SERVED_NO_NONCE).is_empty());
        assert!(index.correlated(&exchange_binding(&req, &req)).is_empty());
        assert!(index.correlated(&exchange_binding(&resp, &req)).is_empty());
        assert!(
            index.correlated(&req).is_empty(),
            "one digest alone is no binding"
        );
    }

    #[test]
    fn padding_is_a_leaf_but_never_a_record() {
        let ledger = corpus_ledger();
        let dir = tempfile::tempdir().unwrap();
        write_jsonl(&dir.path().join(CAPSULES_FILE), &ledger);
        let mut index = LedgerIndex::new(dir.path(), &key_id());
        index.refresh().unwrap();
        let padding_id = ledger[9]["capsule_id"].as_str().unwrap();
        assert!(index.leaves[9].padding);
        assert_eq!(index.leaf_index_of(padding_id), None);
        let view = index.view().unwrap();
        assert!(view.record_at(9).is_some());
        assert_eq!(
            record_digest(&view.record_at(9).unwrap().body).as_deref(),
            Some(padding_id)
        );
    }
}
