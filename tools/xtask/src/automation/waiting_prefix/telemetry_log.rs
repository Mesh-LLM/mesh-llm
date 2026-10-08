//! Bounded telemetry snapshots separating cache seeding from measured requests.
use super::{options, publish};
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{BufRead, BufReader, Read},
    path::{Path, PathBuf},
};

const MAX_LINE_BYTES: u64 = 1024 * 1024;
const MAX_LOG_BYTES: u64 = 128 * 1024 * 1024;

#[derive(Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Cursor {
    schema_version: u64,
    source_log: PathBuf,
    bytes: u64,
    sha256: String,
    generations: usize,
    capacity: usize,
    records: usize,
}

#[derive(Serialize)]
struct Event {
    attributes: Map<String, Value>,
}

#[derive(Default, Serialize)]
pub(super) struct Events {
    events: Vec<Event>,
    capacity_events: Vec<Event>,
    record_events: Vec<Event>,
}

impl Events {
    fn cursor(&self) -> Cursor {
        Cursor {
            schema_version: 1,
            generations: self.events.len(),
            capacity: self.capacity_events.len(),
            records: self.record_events.len(),
            ..Cursor::default()
        }
    }

    fn after(mut self, cursor: &Cursor, expected: usize) -> DynResult<Self> {
        if cursor.schema_version != 1
            || cursor.generations > self.events.len()
            || cursor.capacity > self.capacity_events.len()
            || cursor.records > self.record_events.len()
        {
            return Err("telemetry cursor counts exceed the current log snapshot".into());
        }
        drop(self.events.drain(..cursor.generations));
        drop(self.capacity_events.drain(..cursor.capacity));
        drop(self.record_events.drain(..cursor.records));
        if self.events.len() != expected {
            return Err(format!(
                "expected {expected} measured generation summaries, observed {}",
                self.events.len()
            )
            .into());
        }
        Ok(self)
    }
}

fn retain(events: &mut Events, bytes: &[u8]) -> DynResult<()> {
    let Ok(Value::Object(mut row)) = serde_json::from_slice::<Value>(bytes) else {
        return Ok(());
    };
    let destination = match row.get("event").and_then(Value::as_str) {
        Some("stage.openai_generation_summary") => &mut events.events,
        Some("stage.openai_kv_capacity_decision") => &mut events.capacity_events,
        Some("stage.openai_kv_record_decision") => &mut events.record_events,
        _ => return Ok(()),
    };
    let attributes = row
        .remove("attributes")
        .and_then(|value| value.as_object().cloned())
        .ok_or("recognized A/B telemetry requires an attributes object")?;
    destination.push(Event { attributes });
    Ok(())
}

struct Snapshot {
    source_log: PathBuf,
    bytes: u64,
    sha256: String,
    events: Events,
}

impl Snapshot {
    fn cursor(&self) -> Cursor {
        Cursor {
            source_log: self.source_log.clone(),
            bytes: self.bytes,
            sha256: self.sha256.clone(),
            ..self.events.cursor()
        }
    }
}

fn read(path: &Path, cursor: Option<&Cursor>) -> DynResult<Snapshot> {
    let source_log = path.canonicalize()?;
    if !source_log.is_file() {
        return Err("A/B telemetry log must be a regular file".into());
    }
    if let Some(cursor) = cursor
        && (cursor.schema_version != 1
            || cursor.source_log != source_log
            || cursor.bytes > MAX_LOG_BYTES
            || cursor.sha256.len() != 64
            || !cursor.sha256.bytes().all(|byte| byte.is_ascii_hexdigit()))
    {
        return Err("telemetry cursor source identity is invalid".into());
    }
    let mut reader = BufReader::new(File::open(&source_log)?);
    let mut total = 0_u64;
    let mut digest = Sha256::new();
    let mut prefix = Sha256::new();
    let mut events = Events::default();
    let mut seed_events = Events::default();
    loop {
        let mut bytes = Vec::new();
        let count = (&mut reader)
            .take(MAX_LINE_BYTES + 1)
            .read_until(b'\n', &mut bytes)?;
        if count == 0 {
            break;
        }
        let count = u64::try_from(count)?;
        if count > MAX_LINE_BYTES {
            return Err("A/B telemetry exceeds its bounded line budget".into());
        }
        // A live stderr write can end mid-record. Only complete lines form a cursor.
        if bytes.last() != Some(&b'\n') {
            break;
        }
        if let Some(cursor) = cursor
            && cursor.bytes > total
            && cursor.bytes
                < total
                    .checked_add(count)
                    .ok_or("telemetry log size overflow")?
        {
            return Err("telemetry cursor must end at a complete log line".into());
        }
        let preceding = total;
        total = total
            .checked_add(count)
            .ok_or("telemetry log size overflow")?;
        if count > MAX_LINE_BYTES || total > MAX_LOG_BYTES {
            return Err("A/B telemetry exceeds its bounded line or log budget".into());
        }
        if let Some(cursor) = cursor {
            let prefix_count = cursor.bytes.saturating_sub(preceding).min(count);
            let prefix_bytes = &bytes[..usize::try_from(prefix_count)?];
            prefix.update(prefix_bytes);
            retain(&mut seed_events, prefix_bytes)?;
        }
        digest.update(&bytes);
        retain(&mut events, &bytes)?;
    }
    if let Some(cursor) = cursor {
        if (
            seed_events.events.len(),
            seed_events.capacity_events.len(),
            seed_events.record_events.len(),
        ) != (cursor.generations, cursor.capacity, cursor.records)
        {
            return Err("telemetry cursor counts differ from its verified log prefix".into());
        }
        if total < cursor.bytes || hex::encode(prefix.finalize()) != cursor.sha256 {
            return Err(
                "A/B telemetry log was truncated or rewritten after its seed snapshot".into(),
            );
        }
    }
    Ok(Snapshot {
        source_log,
        bytes: total,
        sha256: hex::encode(digest.finalize()),
        events,
    })
}

pub(super) struct Collection {
    pub(super) events: Events,
    pub(super) cursor: Cursor,
}

pub(super) fn snapshot(path: &Path) -> DynResult<Cursor> {
    Ok(read(path, None)?.cursor())
}

pub(super) fn collect(
    path: &Path,
    cursor: &Cursor,
    expected: usize,
) -> DynResult<Option<Collection>> {
    let current = read(path, Some(cursor))?;
    let observed = current
        .events
        .events
        .len()
        .checked_sub(cursor.generations)
        .ok_or("current telemetry has fewer summaries than its seed cursor")?;
    if observed < expected {
        return Ok(None);
    }
    let next_cursor = current.cursor();
    Ok(Some(Collection {
        events: current.events.after(cursor, expected)?,
        cursor: next_cursor,
    }))
}

pub(super) fn run(args: &[String]) -> DynResult<()> {
    let [verb, rest @ ..] = args else {
        return Err("telemetry-log requires snapshot or collect".into());
    };
    let (allowed, required): (&[&str], &[&str]) = match verb.as_str() {
        "snapshot" => (&["--log", "--output"], &["--log", "--output"]),
        "collect" => (
            &["--log", "--cursor", "--expected-generations", "--output"],
            &["--log", "--cursor", "--expected-generations", "--output"],
        ),
        _ => return Err("telemetry-log requires snapshot or collect".into()),
    };
    let opts = options(rest, allowed, required)?;
    let cursor: Option<Cursor> = opts
        .get("--cursor")
        .map(|path| -> DynResult<Cursor> { Ok(serde_json::from_slice(&std::fs::read(path)?)?) })
        .transpose()?;
    let snapshot = read(Path::new(opts["--log"]), cursor.as_ref())?;
    let mut output = if verb == "snapshot" {
        serde_json::to_vec_pretty(&snapshot.cursor())?
    } else {
        let cursor = cursor.as_ref().ok_or("collect requires a seed cursor")?;
        serde_json::to_vec_pretty(
            &snapshot
                .events
                .after(cursor, opts["--expected-generations"].parse()?)?,
        )?
    };
    output.push(b'\n');
    publish(Path::new(opts["--output"]), &output)
}

#[cfg(test)]
#[path = "telemetry_log_tests.rs"]
mod tests;
