//! The owner's own records: status + the three cleanups the Evidence tab's
//! `Clean up records` dialog offers. Each cleanup seals an owner-maintenance record of
//! itself onto this node's chain, so cleanup is on the record too.
//!
//! **There is no single-record delete, and there will not be one.** Removing a
//! sealed record is a rewrite of the chain, not a cleanup: `Ledger::open`
//! refuses a broken link, and anyone holding a checkpoint would see it. The
//! three things an owner CAN do:
//!
//! 1. **Delete stored prompt and answer text** (`<ledger>/disclosures/*.json`,
//!    and the per-exchange copies in `disclosures/by-exchange/`, `exchange_text`).
//!    That text is a local, out-of-band attachment -- never part of a signed
//!    record -- so the records keep their digests and stay valid.
//! 2. **Rebuild the index.** Re-open the ledger from disk, re-checking every
//!    record, and swap the rebuilt in-memory index in. Nothing is lost.
//! 3. **Start a new history.** Seal a closing record now; at the next start,
//!    before the ledger or the checkpoint cadence opens, move the whole
//!    `ledger/` directory (records, statements, checkpoints, MMR store,
//!    stored text) to `archive/<n>/`, take a new log id, and seal the new
//!    history's first record citing the old one's last. Staged to startup
//!    because the checkpoint cadence task reads `capsules.jsonl` on its own
//!    schedule: moving it under a running cadence would wedge it
//!    (`NodeStoreAheadOfLedger`), so the swap happens where nothing else is
//!    running yet.
//!
//! Reached from the browser through the host's generic tool-call route
//! (`POST /api/plugins/capsules/tools/<operation>`), the same way
//! `mesh_evidence_request` is.

use crate::capsule_emit::{CapsuleState, EmittedCapsule};
use crate::producer::capsule::{OwnerMaintenance, OWNER_MAINTENANCE_BLOCK};
use mesh_llm_plugin::{PluginError, PluginResult};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

pub const STATUS_OPERATION: &str = "evidence_records_status";
pub const DELETE_STORED_TEXT_OPERATION: &str = "evidence_delete_stored_text";
pub const REBUILD_INDEX_OPERATION: &str = "evidence_rebuild_index";
pub const START_NEW_HISTORY_OPERATION: &str = "evidence_start_new_history";

/// Where this node's log id is kept once chosen (see [`resolve_log_id`]).
const LOG_ID_FILE: &str = "log_id";
/// The setting that names a node's log id instead of the default.
pub const LOG_ID_SETTING: &str = "CAPSULES_LOG_ID";
const PENDING_FILE: &str = "new-history.pending.json";
const ARCHIVE_DIR: &str = "archive";

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct NoArgs {}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct StartNewHistoryArgs {
    /// Must be `true`. A new history is not undoable from this tool.
    pub confirm: bool,
}

/// Everything the three cleanups need, shared by `Arc` into each handler.
pub struct Maintenance {
    pub data_dir: PathBuf,
    pub capsules: Arc<CapsuleState>,
    pub log_id: String,
    /// One cleanup at a time: two overlapping requests must not both stage a
    /// new history, or both count the same stored text.
    serial: Mutex<()>,
}

impl Maintenance {
    pub fn new(data_dir: PathBuf, capsules: Arc<CapsuleState>, log_id: String) -> Self {
        Self {
            data_dir,
            capsules,
            log_id,
            serial: Mutex::new(()),
        }
    }

    fn one_at_a_time(&self) -> std::sync::MutexGuard<'_, ()> {
        self.serial
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

fn internal(error: impl std::fmt::Display) -> PluginError {
    PluginError::internal(error.to_string())
}

// ---------------------------------------------------------------------------
// Log id + the staged new history
// ---------------------------------------------------------------------------

/// The log id recorded in `<data_dir>/log_id`, if any.
fn recorded_log_id(data_dir: &Path) -> Option<String> {
    fs::read_to_string(data_dir.join(LOG_ID_FILE))
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// This node's current log id: `<data_dir>/log_id` if one is recorded, else
/// `default`.
pub fn current_log_id(data_dir: &Path, default: &str) -> String {
    recorded_log_id(data_dir).unwrap_or_else(|| default.to_string())
}

/// The default log id of a node: the plugin id and the node's signing key id
/// (`<plugin_id>/<key_id>`). A witness keys continuity by log id, so two
/// nodes must never share one: with one shared id the first node's
/// checkpoints are accepted and every other node's are refused as a fork.
pub fn default_log_id(plugin_id: &str, node_key_id: &str) -> String {
    format!("{plugin_id}/{node_key_id}")
}

/// Whether `log_id` is `base` or one of its later histories (`<base>/h<n>`).
fn is_history_of(log_id: &str, base: &str) -> bool {
    log_id == base
        || log_id
            .strip_prefix(base)
            .and_then(|rest| rest.strip_prefix("/h"))
            .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
}

/// Whether this node's ledger already holds a checkpoint. Every checkpoint
/// carries the log id it was cut under, and the next one must carry the same
/// one, so a log with a checkpoint keeps its log id.
fn has_checkpoints(ledger_dir: &Path) -> bool {
    fs::read_to_string(ledger_dir.join("checkpoints.jsonl"))
        .is_ok_and(|text| text.lines().any(|line| !line.trim().is_empty()))
}

/// The log id this node runs under, chosen once and then kept in
/// `<data_dir>/log_id`. Runs BEFORE `CapsuleState::open` and the checkpoint
/// cadence, and applies a staged new history first.
///
/// - `log_id_setting` (the `CAPSULES_LOG_ID` setting) names the log
///   id instead of the default [`default_log_id`].
/// - A recorded log id is kept as it is.
/// - A ledger that already holds checkpoints but has no recorded log id was
///   checkpointed under `old_shared_log_id`, the plugin's name at the time and
///   the default before log ids were node-unique (`capsules`, even
///   after the plugin was renamed). It keeps that id: re-keying a log with checkpoints would
///   break its checkpoint chain. The id is recorded and a warning names the
///   way to a node-unique id: start a new history.
/// - A setting that differs from the log id of a log that already holds
///   checkpoints is refused, with the same advice, rather than re-keying the
///   log.
pub fn resolve_log_id(
    data_dir: &Path,
    plugin_id: &str,
    old_shared_log_id: &str,
    node_key_id: &str,
    log_id_setting: Option<&str>,
) -> anyhow::Result<String> {
    let setting = log_id_setting.map(str::trim).filter(|s| !s.is_empty());
    let base = setting
        .map(str::to_string)
        .unwrap_or_else(|| default_log_id(plugin_id, node_key_id));
    apply_pending_before_open(data_dir, &base)?;
    let checkpointed = has_checkpoints(&data_dir.join("ledger"));
    let current = match recorded_log_id(data_dir) {
        Some(recorded) => recorded,
        None if checkpointed => old_shared_log_id.to_string(),
        None => base.clone(),
    };
    let log_id = match setting {
        Some(wanted) if !is_history_of(&current, wanted) => {
            if checkpointed {
                anyhow::bail!(
                    "{LOG_ID_SETTING} is {wanted:?} but this node's log already holds checkpoints \
                     under the log id {current:?}; a log keeps its log id. Unset the setting, or \
                     start a new log (the current records are kept whole in an archive) and \
                     the new log takes the setting's log id"
                );
            }
            wanted.to_string()
        }
        _ => current,
    };
    if setting.is_none() && checkpointed && log_id == old_shared_log_id {
        tracing::warn!(
            log_id = %log_id,
            node_unique_log_id = %base,
            "this log was checkpointed under the plugin id, a log id every node had by default; \
             an outside witness accepts that log id from one node only. Start a new log to \
             take a log id of this node's own"
        );
    }
    if recorded_log_id(data_dir).as_deref() != Some(log_id.as_str()) {
        write_atomically(&data_dir.join(LOG_ID_FILE), log_id.as_bytes())?;
    }
    Ok(log_id)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct PendingNewHistory {
    requested_at: String,
    prior_log_id: String,
    /// The closing record sealed when the owner asked -- the old history's
    /// last record, which the new history's first record cites.
    closing_record_id: String,
    /// Set once the old `ledger/` has been moved; lets a restart that died
    /// between the move and the first new record finish the job.
    #[serde(default)]
    archived_as: Option<String>,
    #[serde(default)]
    prior_record_count: Option<usize>,
}

fn read_pending(data_dir: &Path) -> Option<PendingNewHistory> {
    let raw = fs::read(data_dir.join(PENDING_FILE)).ok()?;
    serde_json::from_slice(&raw).ok()
}

fn write_atomically(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, bytes)?;
    fs::File::open(&tmp)?.sync_all()?;
    fs::rename(&tmp, path)
}

fn write_pending(data_dir: &Path, pending: &PendingNewHistory) -> std::io::Result<()> {
    write_atomically(
        &data_dir.join(PENDING_FILE),
        &serde_json::to_vec_pretty(pending)?,
    )
}

fn next_archive_slot(data_dir: &Path) -> (String, PathBuf) {
    let root = data_dir.join(ARCHIVE_DIR);
    let mut n = 1u32;
    loop {
        let name = n.to_string();
        let path = root.join(&name);
        if !path.exists() {
            return (name, path);
        }
        n += 1;
    }
}

/// Whether `capsule_id` is a record of an earlier history of this node, moved
/// under `archive/` by "start new history". Such a record still exists: a
/// reader answers "archived", never "not found".
pub fn is_archived(data_dir: &Path, capsule_id: &str) -> bool {
    let Ok(slots) = fs::read_dir(data_dir.join(ARCHIVE_DIR)) else {
        return false;
    };
    slots.flatten().any(|slot| {
        fs::read_to_string(slot.path().join("capsules.jsonl")).is_ok_and(|text| {
            text.lines().any(|line| {
                serde_json::from_str::<Value>(line.trim()).is_ok_and(|record| {
                    record.get("capsule_id").and_then(Value::as_str) == Some(capsule_id)
                })
            })
        })
    })
}

/// Phase 1, BEFORE `CapsuleState::open` and the checkpoint cadence: if a new
/// history was requested, move the old `ledger/` aside and take the new log
/// id. Returns the log id to run under. Idempotent across a crash: the
/// pending file records how far it got.
pub fn apply_pending_before_open(data_dir: &Path, default_log_id: &str) -> anyhow::Result<String> {
    let Some(mut pending) = read_pending(data_dir) else {
        return Ok(current_log_id(data_dir, default_log_id));
    };
    if pending.archived_as.is_none() {
        let ledger_dir = data_dir.join("ledger");
        let (slot, archive_path) = next_archive_slot(data_dir);
        fs::create_dir_all(archive_path.parent().expect("archive slot has a parent"))?;
        // Count before moving, for the new history's first record.
        let prior_count = count_records(&ledger_dir.join("capsules.jsonl"));
        if ledger_dir.exists() {
            fs::rename(&ledger_dir, &archive_path)?;
        }
        let new_log_id = format!(
            "{}/h{}",
            default_log_id,
            slot.parse::<u32>().unwrap_or(1) + 1
        );
        write_atomically(&data_dir.join(LOG_ID_FILE), new_log_id.as_bytes())?;
        pending.archived_as = Some(format!("{ARCHIVE_DIR}/{slot}"));
        pending.prior_record_count = Some(prior_count);
        write_pending(data_dir, &pending)?;
        tracing::info!(archived_as = ?pending.archived_as, log_id = %new_log_id, "started a new history");
    }
    Ok(current_log_id(data_dir, default_log_id))
}

/// Phase 2, AFTER `CapsuleState::open` on the fresh ledger: seal the new
/// history's first record, citing the old history's last, then clear the
/// pending file. A fresh ledger that already has records means a previous
/// start sealed it and died before clearing; just clear.
pub fn finish_pending_after_open(
    data_dir: &Path,
    capsules: &CapsuleState,
    log_id: &str,
) -> anyhow::Result<()> {
    let Some(pending) = read_pending(data_dir) else {
        return Ok(());
    };
    let Some(archived_as) = pending.archived_as.as_deref() else {
        return Ok(());
    };
    if capsules.chain_head().is_none() {
        let mut facts = Map::new();
        facts.insert("log_id".into(), json!(log_id));
        facts.insert("prior_log_id".into(), json!(pending.prior_log_id));
        facts.insert(
            "prior_record_count".into(),
            json!(pending.prior_record_count),
        );
        facts.insert("requested_at".into(), json!(pending.requested_at));
        facts.insert("prior_history_kept".into(), json!(archived_as));
        capsules.emit_owner_maintenance(&OwnerMaintenance {
            kind: "history_started",
            facts,
            prior_history_head: Some(&pending.closing_record_id),
        })?;
    }
    fs::remove_file(data_dir.join(PENDING_FILE))?;
    Ok(())
}

fn count_records(capsules_path: &Path) -> usize {
    fs::read_to_string(capsules_path)
        .map(|s| s.lines().filter(|l| !l.trim().is_empty()).count())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Stored text (the disclosure store)
// ---------------------------------------------------------------------------

fn is_capsule_id(stem: &str) -> bool {
    stem.len() == 64
        && stem
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// The capsule ids that have stored text, read off `<ledger>/disclosures/`.
fn stored_text_ids(ledger_dir: &Path) -> Vec<String> {
    let Ok(entries) = fs::read_dir(ledger_dir.join("disclosures")) else {
        return Vec::new();
    };
    let mut ids: Vec<String> = entries
        .filter_map(Result::ok)
        .filter_map(|e| {
            let name = e.file_name().into_string().ok()?;
            let stem = name.strip_suffix(".json")?;
            is_capsule_id(stem).then(|| stem.to_string())
        })
        .collect();
    ids.sort();
    ids
}

/// Where the owner's opted-in prompt and answer text is kept (`exchange_text`), one file
/// per exchange, named by the exchange id rather than a record id.
const BY_EXCHANGE_DIR: &str = "by-exchange";

fn is_exchange_id(stem: &str) -> bool {
    !stem.is_empty()
        && stem.len() <= 128
        && stem
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

/// The exchange ids that have stored text, read off
/// `<ledger>/disclosures/by-exchange/`.
fn stored_exchange_text_ids(ledger_dir: &Path) -> Vec<String> {
    let Ok(entries) = fs::read_dir(ledger_dir.join("disclosures").join(BY_EXCHANGE_DIR)) else {
        return Vec::new();
    };
    let mut ids: Vec<String> = entries
        .filter_map(Result::ok)
        .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
        .filter_map(|e| {
            let name = e.file_name().into_string().ok()?;
            let stem = name.strip_suffix(".json")?;
            is_exchange_id(stem).then(|| stem.to_string())
        })
        .collect();
    ids.sort();
    ids
}

/// Remove half-written per-exchange text (`.<exchange_id>.json.tmp`, left by
/// an interrupted write). Returns how many were removed.
fn remove_leftover_exchange_texts(ledger_dir: &Path) -> std::io::Result<usize> {
    let dir = ledger_dir.join("disclosures").join(BY_EXCHANGE_DIR);
    let Ok(entries) = fs::read_dir(&dir) else {
        return Ok(0);
    };
    let mut removed = 0;
    for entry in entries.filter_map(Result::ok) {
        if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            continue;
        }
        let Ok(name) = entry.file_name().into_string() else {
            continue;
        };
        let leftover = name
            .strip_prefix('.')
            .and_then(|rest| rest.strip_suffix(".json.tmp"))
            .is_some_and(is_exchange_id);
        if leftover {
            fs::remove_file(entry.path())?;
            removed += 1;
        }
    }
    Ok(removed)
}

fn stored_text_count(ledger_dir: &Path) -> usize {
    stored_text_ids(ledger_dir).len() + stored_exchange_text_ids(ledger_dir).len()
}

/// 1-based chain positions, collapsed into inclusive `[first, last]` runs.
fn position_ranges(mut positions: Vec<usize>) -> Vec<[usize; 2]> {
    positions.sort_unstable();
    positions.dedup();
    let mut ranges: Vec<[usize; 2]> = Vec::new();
    for p in positions {
        match ranges.last_mut() {
            Some(last) if last[1] + 1 == p => last[1] = p,
            _ => ranges.push([p, p]),
        }
    }
    ranges
}

fn sealed_summary(emitted: &EmittedCapsule, record_number: usize) -> Value {
    json!({
        "capsule_id": emitted.capsule_id,
        "record_number": record_number,
        "kind": emitted.capsule["model_attestation"]["compute_attestation"][OWNER_MAINTENANCE_BLOCK]["kind"],
    })
}

impl Maintenance {
    fn ledger_dir(&self) -> &Path {
        self.capsules.ledger_dir()
    }

    pub fn status(&self) -> Value {
        self.status_signing(crate::requester_auth::asker_key())
    }

    /// [`Self::status`], with `asker` the key this node's requests are signed
    /// with.
    fn status_signing(&self, asker: &crate::requester_auth::AskerKey) -> Value {
        let ids = self.capsules.capsule_ids_in_order();
        let pending = read_pending(&self.data_dir);
        json!({
            "records_path": self.ledger_dir().display().to_string(),
            "record_count": ids.len(),
            // The ledger's last line was torn when it was opened (a write cut
            // short) and was cut back to the last whole record.
            "torn_write_cut_at": self.capsules.torn_write_cut_at(),
            // Every such cut, kept across restarts.
            "ledger_repairs": self.capsules.ledger_repairs(),
            "head": ids.last(),
            "log_id": self.log_id,
            "stored_text_count": stored_text_count(self.ledger_dir()),
            "new_history_pending": pending.as_ref().map(|p| json!({
                "requested_at": p.requested_at,
                "closing_record_id": p.closing_record_id,
            })),
            "sharing": sharing_status(),
            // Whether the requests this node sends are signed: only once its
            // host took its key (`requester_auth::AskerKey`).
            "requests_signed": asker.is_announced(),
            // Why mesh's config file is not being read as written, when it
            // is not: the switches are then its last good values, or all off.
            "config_problem": crate::settings::config_problem(),
            // Sharing switches set to a value this plugin does not know (each
            // read as off).
            "setting_problems": crate::share_policy::setting_problems(),
            // Whether this plugin keeps exchange text it is handed
            // (`exchange_text`), and for how long: the page says so.
            "exchange_text": {
                "kept": crate::exchange_text::enabled(),
                "retention_days": crate::exchange_text::configured_retention_days(),
            },
        })
    }

    /// Delete every stored prompt/answer text, then seal a record naming
    /// which records lost their text (as chain-position ranges, plus a digest
    /// of the exact id list). Nothing to delete seals nothing.
    pub fn delete_stored_text(&self) -> PluginResult<Value> {
        let _one = self.one_at_a_time();
        // Half-written copies go too, whatever else there is: they hold text.
        let leftovers = remove_leftover_exchange_texts(self.ledger_dir())
            .map_err(|e| internal(format!("could not delete leftover stored text: {e}")))?;
        if leftovers > 0 {
            tracing::info!(leftovers, "removed half-written stored text");
        }
        let ids = stored_text_ids(self.ledger_dir());
        let exchange_ids = stored_exchange_text_ids(self.ledger_dir());
        if ids.is_empty() && exchange_ids.is_empty() {
            return Ok(json!({ "deleted_count": 0, "sealed": null }));
        }
        let order = self.capsules.capsule_ids_in_order();
        let position: std::collections::HashMap<&str, usize> = order
            .iter()
            .enumerate()
            .map(|(i, id)| (id.as_str(), i + 1))
            .collect();

        let dir = self.ledger_dir().join("disclosures");
        let mut deleted: Vec<String> = Vec::new();
        for id in &ids {
            match fs::remove_file(dir.join(format!("{id}.json"))) {
                Ok(()) => deleted.push(id.clone()),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => {
                    return Err(internal(format!(
                        "could not delete stored text for {id}: {e}"
                    )))
                }
            }
        }
        let by_exchange = dir.join(BY_EXCHANGE_DIR);
        let mut deleted_exchanges = 0usize;
        for id in &exchange_ids {
            match fs::remove_file(by_exchange.join(format!("{id}.json"))) {
                Ok(()) => deleted_exchanges += 1,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => {
                    return Err(internal(format!(
                        "could not delete stored text for exchange {id}: {e}"
                    )))
                }
            }
        }
        let in_history: Vec<usize> = deleted
            .iter()
            .filter_map(|id| position.get(id.as_str()).copied())
            .collect();
        let not_in_this_history = deleted.len() - in_history.len();
        let ids_digest = hex::encode(Sha256::digest(
            serde_json::to_vec(&deleted).expect("a string list serializes"),
        ));
        let total_deleted = deleted.len() + deleted_exchanges;

        let mut facts = Map::new();
        facts.insert("deleted_count".into(), json!(total_deleted));
        facts.insert(
            "record_ranges".into(),
            json!(position_ranges(in_history.clone())),
        );
        facts.insert("not_in_this_history".into(), json!(not_in_this_history));
        facts.insert("deleted_ids_sha256".into(), json!(ids_digest));
        if deleted_exchanges > 0 {
            // Named by exchange, not by record: counted, never listed.
            facts.insert("exchange_texts_deleted".into(), json!(deleted_exchanges));
        }
        let emitted = self
            .capsules
            .emit_owner_maintenance(&OwnerMaintenance {
                kind: "stored_text_deleted",
                facts,
                prior_history_head: None,
            })
            .map_err(internal)?;
        Ok(json!({
            "deleted_count": total_deleted,
            "record_ranges": position_ranges(in_history),
            "sealed": sealed_summary(&emitted, order.len() + 1),
        }))
    }

    pub fn rebuild_index(&self) -> PluginResult<Value> {
        let _one = self.one_at_a_time();
        let rebuilt = self.capsules.rebuild_index().map_err(internal)?;
        let mut facts = Map::new();
        facts.insert("records_checked".into(), json!(rebuilt.records_checked));
        facts.insert("head".into(), json!(rebuilt.head));
        let emitted = self
            .capsules
            .emit_owner_maintenance(&OwnerMaintenance {
                kind: "index_rebuilt",
                facts,
                prior_history_head: None,
            })
            .map_err(internal)?;
        Ok(json!({
            "records_checked": rebuilt.records_checked,
            "sealed": sealed_summary(&emitted, rebuilt.records_checked + 1),
        }))
    }

    /// Seal the closing record and stage the swap for the next start. A
    /// second request while one is staged is refused, not doubled.
    pub fn start_new_history(&self, args: StartNewHistoryArgs) -> PluginResult<Value> {
        if !args.confirm {
            return Err(PluginError::invalid_params(
                "confirm must be true to start a new history",
            ));
        }
        let _one = self.one_at_a_time();
        if let Some(pending) = read_pending(&self.data_dir) {
            return Err(PluginError::invalid_params(format!(
                "a new history is already staged (asked for at {}); it starts the next time this node starts",
                pending.requested_at
            )));
        }
        let record_count = self.capsules.capsule_ids_in_order().len();
        let mut facts = Map::new();
        facts.insert("log_id".into(), json!(self.log_id));
        facts.insert("record_count".into(), json!(record_count + 1));
        facts.insert("takes_effect".into(), json!("next_start"));
        let emitted = self
            .capsules
            .emit_owner_maintenance(&OwnerMaintenance {
                kind: "history_closing",
                facts,
                prior_history_head: None,
            })
            .map_err(internal)?;
        let pending = PendingNewHistory {
            requested_at: crate::producer::timestamp::utc_now_iso8601(),
            prior_log_id: self.log_id.clone(),
            closing_record_id: emitted.capsule_id.clone(),
            archived_as: None,
            prior_record_count: None,
        };
        write_pending(&self.data_dir, &pending).map_err(internal)?;
        Ok(json!({
            "staged": true,
            "takes_effect": "next_start",
            "sealed": sealed_summary(&emitted, record_count + 1),
        }))
    }
}

// ---------------------------------------------------------------------------
// What leaves: the four sharing switches as this plugin applies them
// ---------------------------------------------------------------------------

fn switch(env: &str, default: Option<&str>) -> Value {
    match crate::settings::var(env)
        .ok()
        .filter(|v| !v.trim().is_empty())
    {
        Some(value) => {
            json!({ "value": value, "source": crate::settings::origin(env).unwrap_or("set") })
        }
        None => json!({ "value": default, "source": "default" }),
    }
}

/// A witness URL as the page may show it: scheme, host and path only. A
/// user, password, query or fragment can carry a credential, so none of them
/// leaves this process.
pub(crate) fn without_credentials(url: &str) -> String {
    let url = url.trim();
    let url = url.split(['?', '#']).next().unwrap_or(url);
    match url.split_once("://") {
        Some((scheme, rest)) => {
            let (authority, path) = rest.find('/').map_or((rest, ""), |i| rest.split_at(i));
            let host = authority
                .rsplit_once('@')
                .map_or(authority, |(_, host)| host);
            format!("{scheme}://{host}{path}")
        }
        None => url
            .rsplit_once('@')
            .map_or(url, |(_, host)| host)
            .to_string(),
    }
}

fn witness_switch() -> Value {
    let mut witness = switch("CAPSULES_CHECKPOINT_WITNESS_URLS", None);
    if let Some(value) = witness["value"].as_str() {
        let shown: Vec<String> = value
            .split(',')
            .filter(|u| !u.trim().is_empty())
            .map(without_credentials)
            .collect();
        witness["value"] = json!(shown.join(","));
    }
    witness
}

fn sharing_status() -> Value {
    json!({
        "record_at_completion": switch(crate::share_policy::ENV_RECORD_AT_COMPLETION, Some("off")),
        "history_segments": switch("CAPSULES_SHARE_HISTORY_SEGMENTS", Some("counterparties")),
        "standard_history": switch(crate::share_policy::ENV_STANDARD_HISTORY, Some("off")),
        "advertise_evidence_requests": switch(crate::share_policy::ENV_ADVERTISE_EVIDENCE_REQUESTS, Some("off")),
        "adjudications": switch("CAPSULES_SHARE_ADJUDICATIONS", Some("off")),
        "adjudicate_differing_twins": switch(crate::referee::request::ENV_ADJUDICATE_DIFFERING_TWINS, Some("off")),
        "witness": witness_switch(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::producer::ledger::Ledger;

    #[test]
    fn a_witness_url_is_shown_without_any_credential() {
        assert_eq!(
            without_credentials("https://user:pa55@witness.example/v1/log?token=abc#x"),
            "https://witness.example/v1/log"
        );
        assert_eq!(
            without_credentials("https://witness.example"),
            "https://witness.example"
        );
        assert_eq!(
            without_credentials("http://tok@127.0.0.1:9000/"),
            "http://127.0.0.1:9000/"
        );
        assert_eq!(
            without_credentials("witness.example?key=1"),
            "witness.example"
        );
    }

    #[test]
    fn a_record_moved_aside_by_a_new_history_is_archived_not_absent() {
        let dir = temp_dir("archived");
        let slot = dir.join(ARCHIVE_DIR).join("1");
        fs::create_dir_all(&slot).unwrap();
        fs::write(
            slot.join("capsules.jsonl"),
            format!("{}\n", json!({ "capsule_id": "a".repeat(64) })),
        )
        .unwrap();
        assert!(is_archived(&dir, &"a".repeat(64)));
        assert!(!is_archived(&dir, &"b".repeat(64)));
        assert!(!is_archived(&temp_dir("no-archive"), &"a".repeat(64)));
    }

    fn key_id_of_node(dir: &Path) -> String {
        crate::producer::keys::load_or_create(&dir.join("keys"))
            .unwrap()
            .key_id()
    }

    fn put_checkpoint(dir: &Path, log_id: &str) {
        fs::create_dir_all(dir.join("ledger")).unwrap();
        fs::write(
            dir.join("ledger/checkpoints.jsonl"),
            format!("{}\n", json!({ "log_id": log_id, "mmr_size": 1 })),
        )
        .unwrap();
    }

    /// Two nodes never share a log id by default: each takes its own key id,
    /// records it, and keeps it across restarts.
    #[test]
    fn the_default_log_id_is_node_unique_and_kept() {
        let a = temp_dir("log-id-a");
        let b = temp_dir("log-id-b");
        let id_a = resolve_log_id(&a, "plugin", "plugin", &key_id_of_node(&a), None).unwrap();
        let id_b = resolve_log_id(&b, "plugin", "plugin", &key_id_of_node(&b), None).unwrap();
        assert_ne!(id_a, id_b);
        assert_eq!(id_a, format!("plugin/{}", key_id_of_node(&a)));
        assert_eq!(recorded_log_id(&a).as_deref(), Some(id_a.as_str()));
        assert_eq!(
            resolve_log_id(&a, "plugin", "plugin", &key_id_of_node(&a), None).unwrap(),
            id_a
        );
        // A recorded id outlives a change of key.
        assert_eq!(
            resolve_log_id(&a, "plugin", "plugin", "other-key", None).unwrap(),
            id_a
        );
        let _ = fs::remove_dir_all(&a);
        let _ = fs::remove_dir_all(&b);
    }

    /// A log checkpointed under the old shared default keeps it: re-keying
    /// would break its checkpoint chain. The id is recorded, never changed.
    #[test]
    fn a_log_checkpointed_under_the_old_default_keeps_it() {
        let dir = temp_dir("log-id-legacy");
        put_checkpoint(&dir, "plugin");
        assert_eq!(
            resolve_log_id(&dir, "plugin", "plugin", "k1", None).unwrap(),
            "plugin"
        );
        assert_eq!(recorded_log_id(&dir).as_deref(), Some("plugin"));
        let _ = fs::remove_dir_all(&dir);
    }

    /// The plugin was renamed after logs were checkpointed under its old name.
    /// Such a log keeps the old name: the new plugin id never re-keys it.
    #[test]
    fn a_log_checkpointed_under_the_old_name_keeps_it_after_the_rename() {
        let dir = temp_dir("log-id-renamed");
        put_checkpoint(&dir, "capsule-emit-mesh");
        assert_eq!(
            resolve_log_id(&dir, "capsules", "capsule-emit-mesh", "k1", None).unwrap(),
            "capsule-emit-mesh"
        );
        assert_eq!(recorded_log_id(&dir).as_deref(), Some("capsule-emit-mesh"));
        // A fresh node takes the new name in its node-unique default.
        let fresh = temp_dir("log-id-renamed-fresh");
        assert_eq!(
            resolve_log_id(&fresh, "capsules", "capsule-emit-mesh", "k1", None).unwrap(),
            "capsules/k1"
        );
        let _ = fs::remove_dir_all(&dir);
        let _ = fs::remove_dir_all(&fresh);
    }

    #[test]
    fn the_setting_names_the_log_id_and_never_re_keys_a_checkpointed_log() {
        let fresh = temp_dir("log-id-setting");
        assert_eq!(
            resolve_log_id(&fresh, "plugin", "plugin", "k1", Some("my-node")).unwrap(),
            "my-node"
        );
        // Not checkpointed yet: a changed setting is taken.
        assert_eq!(
            resolve_log_id(&fresh, "plugin", "plugin", "k1", Some("my-node-2")).unwrap(),
            "my-node-2"
        );
        put_checkpoint(&fresh, "my-node-2");
        assert_eq!(
            resolve_log_id(&fresh, "plugin", "plugin", "k1", Some("my-node-2")).unwrap(),
            "my-node-2"
        );
        let refused =
            resolve_log_id(&fresh, "plugin", "plugin", "k1", Some("my-node-3")).unwrap_err();
        assert!(refused.to_string().contains("start a new log"));
        assert_eq!(recorded_log_id(&fresh).as_deref(), Some("my-node-2"));
        // Unset: the recorded id stays.
        assert_eq!(
            resolve_log_id(&fresh, "plugin", "plugin", "k1", None).unwrap(),
            "my-node-2"
        );

        let legacy = temp_dir("log-id-setting-legacy");
        put_checkpoint(&legacy, "plugin");
        assert!(resolve_log_id(&legacy, "plugin", "plugin", "k1", Some("my-node")).is_err());
        assert!(recorded_log_id(&legacy).is_none());
        let _ = fs::remove_dir_all(&fresh);
        let _ = fs::remove_dir_all(&legacy);
    }

    #[test]
    fn a_later_history_of_the_set_log_id_is_kept() {
        assert!(is_history_of("x/h2", "x"));
        assert!(is_history_of("x", "x"));
        assert!(!is_history_of("x/h", "x"));
        assert!(!is_history_of("x/h2a", "x"));
        assert!(!is_history_of("xy", "x"));
    }

    /// The way off the old shared default: a new history takes the node's own
    /// log id, and the old log is kept whole under its old id.
    #[test]
    fn a_new_history_moves_a_legacy_log_to_a_node_unique_id() {
        let dir = temp_dir("log-id-migrate");
        let (m, _ids) = maintenance(&dir, 1);
        drop(m);
        put_checkpoint(&dir, "node-under-test");
        let key_id = key_id_of_node(&dir);
        let log_id =
            resolve_log_id(&dir, "node-under-test", "node-under-test", &key_id, None).unwrap();
        assert_eq!(log_id, "node-under-test");
        let state = Arc::new(CapsuleState::open(&dir, "node-under-test").unwrap());
        Maintenance::new(dir.clone(), state, log_id)
            .start_new_history(StartNewHistoryArgs { confirm: true })
            .unwrap();

        // --- restart ---
        let log_id =
            resolve_log_id(&dir, "node-under-test", "node-under-test", &key_id, None).unwrap();
        assert_eq!(log_id, format!("node-under-test/{key_id}/h2"));
        let state = CapsuleState::open(&dir, "node-under-test").unwrap();
        finish_pending_after_open(&dir, &state, &log_id).unwrap();
        assert!(dir.join("archive/1/checkpoints.jsonl").exists());
        let first = reopen_clean(&dir);
        let first = first
            .lookup(first.chain_head().unwrap())
            .unwrap()
            .unwrap()
            .capsule;
        assert_eq!(block(&first)["prior_log_id"], json!("node-under-test"));
        assert_eq!(block(&first)["log_id"], json!(log_id));
        let _ = fs::remove_dir_all(&dir);
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("owner-maint-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn seal_exchange(state: &CapsuleState, n: usize) -> String {
        let req = format!(r#"{{"model":"m","messages":[{{"role":"user","content":"hi {n}"}}]}}"#);
        let exchange = crate::capsule_emit::ExchangeRecord {
            model: "m",
            client_nonce: Some("n"),
            request_bytes: req.as_bytes(),
            response_bytes:
                br#"{"id":"x","choices":[{"message":{"role":"assistant","content":"y"}}]}"#,
            latency_ms: 1.0,
            exchange_id: None,
            requesting_party: None,
            host_provenance: None,
        };
        state.emit_for_exchange(&exchange).unwrap().capsule_id
    }

    fn maintenance(dir: &Path, records: usize) -> (Maintenance, Vec<String>) {
        let state = Arc::new(CapsuleState::open(dir, "node-under-test").unwrap());
        let ids = (0..records).map(|n| seal_exchange(&state, n)).collect();
        (
            Maintenance::new(dir.to_path_buf(), state, "node-under-test".into()),
            ids,
        )
    }

    fn put_text(dir: &Path, id: &str) {
        let d = dir.join("ledger").join("disclosures");
        fs::create_dir_all(&d).unwrap();
        fs::write(d.join(format!("{id}.json")), br#"{"prompt_text":"secret"}"#).unwrap();
    }

    fn block(capsule: &Value) -> &Value {
        &capsule["model_attestation"]["compute_attestation"][OWNER_MAINTENANCE_BLOCK]
    }

    /// The chain the owner ends up with must still open clean: every link,
    /// id and signed statement re-checked by the same code that loads it.
    fn reopen_clean(dir: &Path) -> Ledger {
        crate::producer::index::open_ledger(&dir.join("ledger"))
            .expect("chain must reload clean after a cleanup")
            .0
    }

    #[test]
    fn position_ranges_collapse_runs() {
        assert_eq!(
            position_ranges(vec![5, 1, 2, 3, 7, 8, 2]),
            vec![[1, 3], [5, 5], [7, 8]]
        );
        assert!(position_ranges(vec![]).is_empty());
    }

    #[test]
    fn delete_stored_text_removes_the_text_keeps_the_records_and_seals_a_record() {
        let dir = temp_dir("delete");
        let (m, ids) = maintenance(&dir, 4);
        put_text(&dir, &ids[0]);
        put_text(&dir, &ids[1]);
        put_text(&dir, &ids[3]);
        put_text(&dir, &"f".repeat(64)); // text for a record not in this history
        fs::write(
            dir.join("ledger/disclosures/notes.txt"),
            b"not a stored text",
        )
        .unwrap();

        let out = m.delete_stored_text().unwrap();
        assert_eq!(out["deleted_count"], json!(4));
        assert_eq!(out["record_ranges"], json!([[1, 2], [4, 4]]));
        assert_eq!(stored_text_ids(&dir.join("ledger")), Vec::<String>::new());
        assert!(
            dir.join("ledger/disclosures/notes.txt").exists(),
            "only stored-text files are touched"
        );

        let ledger = reopen_clean(&dir);
        let order = ledger.capsule_ids_in_order();
        assert_eq!(
            &order[..4],
            &ids[..],
            "every record is still there, in order"
        );
        let sealed = ledger
            .lookup(order.last().unwrap())
            .unwrap()
            .unwrap()
            .capsule;
        assert_eq!(block(&sealed)["kind"], json!("stored_text_deleted"));
        assert_eq!(block(&sealed)["deleted_count"], json!(4));
        assert_eq!(block(&sealed)["not_in_this_history"], json!(1));
        assert_eq!(block(&sealed)["record_ranges"], json!([[1, 2], [4, 4]]));
        assert!(
            !sealed.to_string().contains("secret"),
            "the record never carries the text it removed"
        );
        assert_eq!(out["sealed"]["record_number"], json!(5));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn stored_text_counts_and_deletes_the_hosts_per_exchange_copies() {
        let dir = temp_dir("delete-by-exchange");
        let (m, ids) = maintenance(&dir, 2);
        put_text(&dir, &ids[0]);
        let d = dir.join("ledger/disclosures/by-exchange");
        fs::create_dir_all(&d).unwrap();
        fs::write(d.join("ex-1.json"), br#"{"response_text":"secret"}"#).unwrap();
        fs::write(d.join("ex-2.json"), br#"{"response_text":"secret"}"#).unwrap();
        fs::write(d.join("README"), b"not a stored text").unwrap();
        assert_eq!(
            m.status()["stored_text_count"],
            json!(3),
            "the per-exchange copies are disclosed"
        );

        let out = m.delete_stored_text().unwrap();
        assert_eq!(out["deleted_count"], json!(3));
        assert_eq!(m.status()["stored_text_count"], json!(0));
        assert!(!d.join("ex-1.json").exists() && !d.join("ex-2.json").exists());
        assert!(
            d.join("README").exists(),
            "only stored-text files are touched"
        );

        let ledger = reopen_clean(&dir);
        let sealed = ledger
            .lookup(ledger.capsule_ids_in_order().last().unwrap())
            .unwrap()
            .unwrap()
            .capsule;
        assert_eq!(block(&sealed)["deleted_count"], json!(3));
        assert_eq!(block(&sealed)["exchange_texts_deleted"], json!(2));
        assert_eq!(block(&sealed)["not_in_this_history"], json!(0));
        assert!(
            !sealed.to_string().contains("ex-1"),
            "exchange ids are counted, never listed"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// A half-written copy (`.<id>.json.tmp`) holds text too: delete takes it,
    /// even when there is nothing else to delete, and leaves other files be.
    #[test]
    fn delete_stored_text_also_removes_half_written_copies() {
        let dir = temp_dir("delete-leftovers");
        let (m, _ids) = maintenance(&dir, 1);
        let d = dir.join("ledger/disclosures/by-exchange");
        fs::create_dir_all(&d).unwrap();
        fs::write(d.join(".ex-1.json.tmp"), br#"{"response_text":"secret"}"#).unwrap();
        fs::write(d.join(".not an id.json.tmp"), b"not ours").unwrap();
        fs::write(d.join("README"), b"not a stored text").unwrap();

        let out = m.delete_stored_text().unwrap();
        assert_eq!(out, json!({ "deleted_count": 0, "sealed": null }));
        assert!(!d.join(".ex-1.json.tmp").exists());
        assert!(d.join(".not an id.json.tmp").exists() && d.join("README").exists());

        // Beside a real stored text too.
        fs::write(d.join(".ex-2.json.tmp"), b"{}").unwrap();
        fs::write(d.join("ex-3.json"), br#"{"response_text":"secret"}"#).unwrap();
        assert_eq!(m.delete_stored_text().unwrap()["deleted_count"], json!(1));
        assert!(!d.join(".ex-2.json.tmp").exists() && !d.join("ex-3.json").exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn status_says_whether_exchange_text_is_kept_and_for_how_long() {
        let dir = temp_dir("status-exchange-text");
        let (m, _ids) = maintenance(&dir, 1);
        let status = m.status();
        assert_eq!(
            status["exchange_text"]["kept"],
            json!(crate::exchange_text::enabled())
        );
        assert_eq!(
            status["exchange_text"]["retention_days"],
            json!(crate::exchange_text::configured_retention_days())
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// A host that refused this node's key leaves its requests unsigned,
    /// and the status says which.
    #[test]
    fn status_says_whether_this_nodes_requests_are_signed() {
        let dir = temp_dir("status-requests-signed");
        let (m, _ids) = maintenance(&dir, 1);
        let asker = crate::requester_auth::AskerKey::default();
        assert_eq!(m.status_signing(&asker)["requests_signed"], json!(false));
        asker.set_announced(ed25519_dalek::SigningKey::from_bytes(&[7; 32]));
        assert_eq!(m.status_signing(&asker)["requests_signed"], json!(true));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn delete_stored_text_with_nothing_stored_seals_nothing() {
        let dir = temp_dir("delete-none");
        let (m, ids) = maintenance(&dir, 2);
        let out = m.delete_stored_text().unwrap();
        assert_eq!(out, json!({ "deleted_count": 0, "sealed": null }));
        assert_eq!(m.capsules.chain_head().as_deref(), Some(ids[1].as_str()));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn rebuild_index_rechecks_every_record_and_seals_a_record() {
        let dir = temp_dir("rebuild");
        let (m, ids) = maintenance(&dir, 3);
        let out = m.rebuild_index().unwrap();
        assert_eq!(out["records_checked"], json!(3));
        let ledger = reopen_clean(&dir);
        let order = ledger.capsule_ids_in_order();
        assert_eq!(order.len(), 4);
        assert_eq!(&order[..3], &ids[..]);
        let sealed = ledger.lookup(&order[3]).unwrap().unwrap().capsule;
        assert_eq!(block(&sealed)["kind"], json!("index_rebuilt"));
        assert_eq!(block(&sealed)["head"], json!(ids[2]));
        let _ = fs::remove_dir_all(&dir);
    }

    /// Mutant check: a record tampered on disk makes the rebuild FAIL, and
    /// nothing is sealed onto a chain that no longer checks out.
    #[test]
    fn rebuild_index_refuses_a_tampered_chain_and_seals_nothing() {
        let dir = temp_dir("rebuild-tamper");
        let (m, ids) = maintenance(&dir, 2);
        let path = dir.join("ledger/capsules.jsonl");
        let text = fs::read_to_string(&path).unwrap();
        fs::write(&path, text.replacen("\"decide\"", "\"deny\"", 1)).unwrap();
        assert!(m.rebuild_index().is_err());
        assert_eq!(m.capsules.chain_head().as_deref(), Some(ids[1].as_str()));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn start_new_history_needs_confirm_and_refuses_a_second_request() {
        let dir = temp_dir("confirm");
        let (m, _) = maintenance(&dir, 1);
        assert!(m
            .start_new_history(StartNewHistoryArgs { confirm: false })
            .is_err());
        assert!(
            read_pending(&dir).is_none(),
            "an unconfirmed request stages nothing"
        );
        m.start_new_history(StartNewHistoryArgs { confirm: true })
            .unwrap();
        assert!(m
            .start_new_history(StartNewHistoryArgs { confirm: true })
            .is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    /// End to end across a restart: the old history is kept whole and still
    /// opens clean under `archive/1`, the new history opens clean, and its
    /// first record cites the old one's closing record.
    #[test]
    fn start_new_history_archives_the_old_log_and_starts_a_cited_new_one_at_next_start() {
        let dir = temp_dir("new-history");
        let (m, ids) = maintenance(&dir, 3);
        put_text(&dir, &ids[0]);
        let out = m
            .start_new_history(StartNewHistoryArgs { confirm: true })
            .unwrap();
        assert_eq!(out["takes_effect"], json!("next_start"));
        let closing_id = out["sealed"]["capsule_id"].as_str().unwrap().to_string();
        // Until the restart, nothing moved.
        assert_eq!(
            m.capsules.chain_head().as_deref(),
            Some(closing_id.as_str())
        );
        assert_eq!(
            m.status()["new_history_pending"]["closing_record_id"],
            json!(closing_id)
        );
        drop(m);

        // --- restart ---
        let log_id = apply_pending_before_open(&dir, "node-under-test").unwrap();
        assert_eq!(log_id, "node-under-test/h2");
        let state = CapsuleState::open(&dir, "node-under-test").unwrap();
        finish_pending_after_open(&dir, &state, &log_id).unwrap();
        assert!(read_pending(&dir).is_none());

        let (old, _) = crate::producer::index::open_ledger(&dir.join("archive/1"))
            .expect("the old history still opens clean");
        assert_eq!(old.chain_head(), Some(closing_id.as_str()));
        assert_eq!(old.len(), 4);
        assert!(
            dir.join("archive/1/disclosures").exists(),
            "stored text moves with its records"
        );

        let new = reopen_clean(&dir);
        assert_eq!(new.len(), 1);
        let first = new
            .lookup(new.chain_head().unwrap())
            .unwrap()
            .unwrap()
            .capsule;
        assert_eq!(block(&first)["kind"], json!("history_started"));
        assert_eq!(block(&first)["prior_log_id"], json!("node-under-test"));
        assert_eq!(block(&first)["prior_record_count"], json!(4));
        assert_eq!(first["references"][0]["digest"], json!(closing_id));
        assert!(first.get("chain").is_none());

        // A later start is a no-op: same log id, nothing re-sealed.
        assert_eq!(
            apply_pending_before_open(&dir, "node-under-test").unwrap(),
            "node-under-test/h2"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// A crash after the move but before the first new record: the next start
    /// finishes the job instead of moving the fresh ledger again.
    #[test]
    fn a_start_that_died_after_the_move_finishes_on_the_next_start() {
        let dir = temp_dir("crash");
        let (m, _) = maintenance(&dir, 1);
        m.start_new_history(StartNewHistoryArgs { confirm: true })
            .unwrap();
        drop(m);
        let log_id = apply_pending_before_open(&dir, "node-under-test").unwrap();
        // (died here: no open, no seal)
        assert_eq!(
            apply_pending_before_open(&dir, "node-under-test").unwrap(),
            log_id
        );
        assert!(
            !dir.join("archive/2").exists(),
            "the fresh ledger is not archived a second time"
        );
        let state = CapsuleState::open(&dir, "node-under-test").unwrap();
        finish_pending_after_open(&dir, &state, &log_id).unwrap();
        assert_eq!(reopen_clean(&dir).len(), 1);
        let _ = fs::remove_dir_all(&dir);
    }
}
