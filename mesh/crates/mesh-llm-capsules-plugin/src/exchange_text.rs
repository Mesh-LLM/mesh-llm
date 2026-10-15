//! The owner's opt-in to keep the TEXT of each exchange (prompt and answer)
//! beside its sealed record, so the twin comparison and "Your records" can
//! show it. Off unless `CAPSULES_KEEP_EXCHANGE_TEXT=1`; nothing is
//! written otherwise.
//!
//! The text could only come from the host, as `exchange_bodies` on the
//! `openai.exchange.v1` terminal event. mesh-llm sends no such field today,
//! so the plugin sees digests only and keeps no text, whatever this setting
//! says. Text is kept only if a host sends the bodies AND this setting is on.
//!
//! Each exchange gets one file, `<ledger>/disclosures/by-exchange/<exchange_id>.json`
//! (both directories 0700, file 0600, written whole by rename), keyed by the
//! host's `exchange_id`, which this plugin seals in `serving_provenance`.
//! Files older than `CAPSULES_KEEP_EXCHANGE_TEXT_DAYS` (default 30)
//! are removed: at plugin start, every hour after that ([`spawn_retention`]),
//! and on a write (at most once an hour per directory). The age-out runs
//! whether or not keeping is still on, so text kept before the owner turned
//! it off still goes after its retention.
//! "Clean up records" deletes them (`owner_maintenance`).
//!
//! Size: one file is at most [`MAX_FILE_BYTES`] (256 KiB). An exchange whose
//! file would be larger keeps no bodies at all (a cut body could never match
//! its sealed digest, so it would only mislead): the file says
//! `"truncated": true`, gives the size it would have had (`original_bytes`),
//! and keeps the start of the answer text that fits.
//!
//! Memory: the bodies are taken off the event as soon as it arrives
//! ([`take`]), whether or not this plugin keeps them, so no copy of the text
//! is held in memory beyond the one being written.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use serde_json::{json, Value};

use crate::lifecycle_channel::{ExchangeBodies, OpenAiExchangeEnvelope, Phase};

/// Set to `1` to keep prompt and answer text. Off by default.
pub const KEEP_EXCHANGE_TEXT_ENV: &str = "CAPSULES_KEEP_EXCHANGE_TEXT";
/// How many days a kept text stays; a positive whole number, else the default.
pub const KEEP_EXCHANGE_TEXT_DAYS_ENV: &str = "CAPSULES_KEEP_EXCHANGE_TEXT_DAYS";
const DEFAULT_KEEP_DAYS: u64 = 30;
/// The largest exchange text file this plugin writes.
pub const MAX_FILE_BYTES: usize = 256 * 1024;
/// A directory is pruned at most this often.
const PRUNE_EVERY: Duration = Duration::from_secs(60 * 60);

pub fn enabled() -> bool {
    enabled_from(crate::settings::var(KEEP_EXCHANGE_TEXT_ENV).ok().as_deref())
}

fn enabled_from(value: Option<&str>) -> bool {
    value.map(str::trim) == Some("1")
}

fn retention_days(days: Option<&str>) -> u64 {
    days.and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|days| *days > 0)
        .unwrap_or(DEFAULT_KEEP_DAYS)
}

fn retention(days: Option<&str>) -> Duration {
    Duration::from_secs(retention_days(days).saturating_mul(24 * 60 * 60))
}

/// How many days a kept text stays, as configured.
pub fn configured_retention_days() -> u64 {
    retention_days(
        crate::settings::var(KEEP_EXCHANGE_TEXT_DAYS_ENV)
            .ok()
            .as_deref(),
    )
}

/// One exchange's text, taken off its event, waiting to be written.
pub struct Pending {
    exchange_id: String,
    twin_bracket_id: Option<String>,
    bodies: ExchangeBodies,
}

impl Pending {
    /// Write it ([`keep`]).
    pub fn write(self, ledger_dir: &Path) -> std::io::Result<Option<PathBuf>> {
        keep(
            ledger_dir,
            &self.exchange_id,
            self.twin_bracket_id.as_deref(),
            &self.bodies,
        )
    }
}

/// Take the bodies off an observed event, always, so nothing downstream (the
/// in-memory event window, the split collector, the events log) ever holds
/// them. Returns them to write only when `keeping` is on and the event is a
/// terminal one with an exchange id.
pub fn take(envelope: &mut OpenAiExchangeEnvelope, keeping: bool) -> Option<Pending> {
    let bodies = envelope.exchange_bodies.take()?;
    if !keeping || envelope.phase != Phase::Terminal {
        return None;
    }
    Some(Pending {
        exchange_id: envelope.exchange_id.clone()?,
        twin_bracket_id: envelope.twin_bracket_id.clone(),
        bodies,
    })
}

/// The longest prefix of `text` that is at most `max` bytes and ends on a
/// character boundary.
fn prefix(text: &str, max: usize) -> &str {
    let mut end = text.len().min(max);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

/// Remove kept texts (and leftover temp files) last written more than
/// `max_age` before `now`. Returns how many were removed. Only a directory
/// that can't be listed fails the pass; a file that can't be read or removed
/// is logged and counted, and the pass goes on to the next one, so one stuck
/// file never keeps every later text from ageing out.
fn prune(dir: &Path, max_age: Duration, now: SystemTime) -> std::io::Result<usize> {
    let pruned = prune_with(dir, max_age, now, &|path| std::fs::remove_file(path))?;
    if pruned.failed > 0 {
        tracing::warn!(
            failed = pruned.failed,
            removed = pruned.removed,
            "some kept exchange text could not be aged out"
        );
    }
    Ok(pruned.removed)
}

/// What one prune pass did.
#[derive(Debug, Default, PartialEq, Eq)]
struct Pruned {
    removed: usize,
    /// Files that could not be read or removed, skipped.
    failed: usize,
}

/// [`prune`], with the file removal given (so a test can make one fail).
fn prune_with(
    dir: &Path,
    max_age: Duration,
    now: SystemTime,
    remove: &dyn Fn(&Path) -> std::io::Result<()>,
) -> std::io::Result<Pruned> {
    let mut pruned = Pruned::default();
    for entry in std::fs::read_dir(dir)? {
        let stale = entry.and_then(|entry| {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if !(name.ends_with(".json") || name.ends_with(".json.tmp")) {
                return Ok(None);
            }
            let metadata = entry.metadata()?;
            if !metadata.is_file() {
                return Ok(None);
            }
            let age = now
                .duration_since(metadata.modified()?)
                .unwrap_or(Duration::ZERO);
            Ok((age > max_age).then(|| entry.path()))
        });
        match stale.and_then(|path| match path {
            Some(path) => remove(&path).map(|()| true),
            None => Ok(false),
        }) {
            Ok(true) => pruned.removed += 1,
            Ok(false) => {}
            Err(error) => {
                tracing::warn!(%error, "could not age out one kept exchange text; going on");
                pruned.failed += 1;
            }
        }
    }
    Ok(pruned)
}

/// Prune `dir` unless it was pruned within the last `PRUNE_EVERY`. A failed
/// prune is logged; it never fails the write that triggered it.
fn prune_if_due(dir: &Path) {
    static LAST: std::sync::Mutex<Option<std::collections::HashMap<PathBuf, Instant>>> =
        std::sync::Mutex::new(None);
    {
        let mut last = LAST
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let last = last.get_or_insert_with(Default::default);
        let now = Instant::now();
        if last
            .get(dir)
            .is_some_and(|at| now.duration_since(*at) < PRUNE_EVERY)
        {
            return;
        }
        last.insert(dir.to_path_buf(), now);
    }
    let max_age = retention(
        crate::settings::var(KEEP_EXCHANGE_TEXT_DAYS_ENV)
            .ok()
            .as_deref(),
    );
    if let Err(error) = prune(dir, max_age, SystemTime::now()) {
        tracing::warn!(%error, "could not prune kept exchange text");
    }
}

/// The directory kept texts live in.
fn by_exchange_dir(ledger_dir: &Path) -> PathBuf {
    ledger_dir.join("disclosures").join("by-exchange")
}

/// Remove every kept text older than the configured retention, now. A
/// ledger with no kept texts removes nothing. Returns how many were removed.
pub fn age_out(ledger_dir: &Path) -> std::io::Result<usize> {
    let dir = by_exchange_dir(ledger_dir);
    if !dir.is_dir() {
        return Ok(0);
    }
    let max_age = retention(
        crate::settings::var(KEEP_EXCHANGE_TEXT_DAYS_ENV)
            .ok()
            .as_deref(),
    );
    prune(&dir, max_age, SystemTime::now())
}

/// Age kept texts out at start and every [`PRUNE_EVERY`] after, off the
/// async workers. Failures are logged, never fatal.
pub fn spawn_retention(ledger_dir: PathBuf) {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(PRUNE_EVERY);
        loop {
            tick.tick().await;
            let dir = ledger_dir.clone();
            match tokio::task::spawn_blocking(move || age_out(&dir)).await {
                Ok(Ok(0)) => {}
                Ok(Ok(removed)) => tracing::info!(removed, "aged out kept exchange text"),
                Ok(Err(error)) => tracing::warn!(%error, "could not age out kept exchange text"),
                Err(error) => tracing::warn!(%error, "kept-text age-out task did not complete"),
            }
        }
    });
}

/// An exchange id is host-minted (a UUID); anything else never names a file.
fn safe_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= 64 && id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
}

/// Write one exchange's text file from the bodies the host handed over.
/// Returns the file written, or `None` when there is nothing to keep (no
/// bodies, or an id that can't name a file). The caller checks [`enabled`].
pub fn keep(
    ledger_dir: &Path,
    exchange_id: &str,
    twin_bracket_id: Option<&str>,
    bodies: &ExchangeBodies,
) -> std::io::Result<Option<PathBuf>> {
    if !safe_id(exchange_id) || (bodies.request.is_none() && bodies.response.is_none()) {
        return Ok(None);
    }
    let response_text = bodies
        .response
        .as_ref()
        .and_then(|body| body.pointer("/choices/0/message/content"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let mut document = json!({
        "v": 1,
        "exchange_id": exchange_id,
        "request_body": bodies.request,
        "response_body": bodies.response,
        "response_text": response_text,
        "written_at": chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
    });
    if let Some(bracket) = twin_bracket_id {
        document["twin_bracket_id"] = json!(bracket);
    }
    let mut bytes = serde_json::to_vec(&document)?;
    if bytes.len() > MAX_FILE_BYTES {
        let original_bytes = bytes.len();
        document["request_body"] = Value::Null;
        document["response_body"] = Value::Null;
        document["response_text"] = Value::Null;
        document["truncated"] = json!(true);
        document["original_bytes"] = json!(original_bytes);
        let room = MAX_FILE_BYTES.saturating_sub(serde_json::to_vec(&document)?.len() + 2);
        if let Some(text) = response_text.as_deref() {
            // Room is counted in bytes the text takes once JSON-escaped; a
            // character that escapes wide makes the prefix shorter, never the
            // file bigger.
            let mut cut = prefix(text, room);
            while serde_json::to_string(cut)?.len() > room + 2 {
                cut = prefix(cut, cut.len().saturating_sub(64));
            }
            document["response_text"] = json!(cut);
        }
        bytes = serde_json::to_vec(&document)?;
    }
    let dir = by_exchange_dir(ledger_dir);
    std::fs::create_dir_all(&dir)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let disclosures = ledger_dir.join("disclosures");
        for private in [&disclosures, &dir] {
            std::fs::set_permissions(private, std::fs::Permissions::from_mode(0o700))?;
        }
    }
    let path = dir.join(format!("{exchange_id}.json"));
    let tmp = dir.join(format!(".{exchange_id}.json.tmp"));
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let written = options.open(&tmp).and_then(|mut file| {
        file.write_all(&bytes)?;
        file.sync_all()?;
        std::fs::rename(&tmp, &path)
    });
    if let Err(error) = written {
        // Never leave a half-written copy of the text behind.
        let _ = std::fs::remove_file(&tmp);
        return Err(error);
    }
    prune_if_due(&dir);
    Ok(Some(path))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bodies(request: Option<Value>, response: Option<Value>) -> ExchangeBodies {
        ExchangeBodies { request, response }
    }

    #[test]
    fn only_one_turns_it_on_and_it_is_off_by_default() {
        assert!(enabled_from(Some("1")));
        for value in [None, Some(""), Some("0"), Some("true"), Some("2")] {
            assert!(!enabled_from(value), "{value:?}");
        }
    }

    #[test]
    fn the_bodies_are_kept_whole_0600_in_a_0700_directory() {
        let dir = tempfile::tempdir().unwrap();
        let request =
            json!({"model": "m", "messages": [{"role": "user", "content": "Sky colour?"}]});
        let response =
            json!({"choices": [{"index": 0, "message": {"role": "assistant", "content": "Blue"}}]});
        let written = keep(
            dir.path(),
            "0f8fad5b-d9cb-469f-a165-70867728950e",
            Some("twin-1"),
            &bodies(Some(request.clone()), Some(response.clone())),
        )
        .unwrap()
        .expect("a file");
        let saved: Value = serde_json::from_slice(&std::fs::read(&written).unwrap()).unwrap();
        assert_eq!(
            saved["exchange_id"],
            json!("0f8fad5b-d9cb-469f-a165-70867728950e")
        );
        assert_eq!(saved["request_body"], request);
        assert_eq!(saved["response_body"], response);
        assert_eq!(saved["response_text"], json!("Blue"));
        assert_eq!(saved["twin_bracket_id"], json!("twin-1"));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode(&written), 0o600);
            assert_eq!(mode(written.parent().unwrap()), 0o700);
            assert_eq!(mode(&dir.path().join("disclosures")), 0o700);
        }
    }

    fn envelope(phase: &str, exchange_id: Option<&str>) -> OpenAiExchangeEnvelope {
        let mut value = json!({"dispatch_path": "raw_proxy", "phase": phase, "model": "m",
            "exchange_bodies": {"request": {"model": "m"}}});
        if let Some(id) = exchange_id {
            value["exchange_id"] = json!(id);
        }
        serde_json::from_value(value).unwrap()
    }

    /// The bodies always come off the event; they are handed back only to
    /// be kept, and only for a terminal event with an id.
    #[test]
    fn take_always_strips_the_bodies_and_returns_them_only_to_keep() {
        for (phase, id, keeping, kept) in [
            ("terminal", Some("ex-1"), true, true),
            ("terminal", Some("ex-1"), false, false),
            ("effective_request", Some("ex-1"), true, false),
            ("terminal", None, true, false),
        ] {
            let mut event = envelope(phase, id);
            let pending = take(&mut event, keeping);
            assert!(event.exchange_bodies.is_none(), "{phase} {id:?} {keeping}");
            assert_eq!(pending.is_some(), kept, "{phase} {id:?} {keeping}");
        }
    }

    /// A file never exceeds the cap. An oversized exchange keeps no bodies
    /// (a cut body could not match its digest), says it was truncated and
    /// how big it was, and keeps the start of the answer.
    #[test]
    fn an_oversized_exchange_is_capped_with_an_explicit_marker() {
        let dir = tempfile::tempdir().unwrap();
        let long = "é".repeat(MAX_FILE_BYTES);
        let response = json!({"choices": [{"message": {"content": long}}]});
        let written = keep(
            dir.path(),
            "0f8fad5b-d9cb-469f-a165-70867728950e",
            None,
            &bodies(Some(json!({"model": "m"})), Some(response)),
        )
        .unwrap()
        .expect("a file");
        let raw = std::fs::read(&written).unwrap();
        assert!(raw.len() <= MAX_FILE_BYTES, "{} bytes", raw.len());
        let saved: Value = serde_json::from_slice(&raw).unwrap();
        assert_eq!(saved["truncated"], json!(true));
        assert!(saved["original_bytes"].as_u64().unwrap() > MAX_FILE_BYTES as u64);
        assert!(saved["request_body"].is_null() && saved["response_body"].is_null());
        let kept = saved["response_text"].as_str().unwrap();
        assert!(!kept.is_empty() && long.starts_with(kept));
        // A file under the cap carries no marker.
        let small = keep(
            dir.path(),
            "6f9619ff-8b86-d011-b42d-00c04fc964ff",
            None,
            &bodies(Some(json!({"model": "m"})), None),
        )
        .unwrap()
        .unwrap();
        let small: Value = serde_json::from_slice(&std::fs::read(small).unwrap()).unwrap();
        assert!(small.get("truncated").is_none());
    }

    #[test]
    fn nothing_is_written_without_bodies_or_for_an_unsafe_id() {
        let dir = tempfile::tempdir().unwrap();
        assert!(keep(dir.path(), "abc", None, &bodies(None, None))
            .unwrap()
            .is_none());
        let request = Some(json!({"model": "m"}));
        for id in ["../escape", "", "a/b", &"a".repeat(65)] {
            assert!(
                keep(dir.path(), id, None, &bodies(request.clone(), None))
                    .unwrap()
                    .is_none(),
                "{id}"
            );
        }
        assert!(!dir.path().join("disclosures").exists());
    }

    fn file_aged(path: &Path, age: Duration) {
        std::fs::write(path, b"{}").unwrap();
        let file = std::fs::File::options().write(true).open(path).unwrap();
        file.set_modified(SystemTime::now() - age).unwrap();
    }

    /// Kept text does not accumulate forever. MUTANT: never remove, or
    /// remove regardless of age, and this fails.
    #[test]
    fn prune_removes_only_texts_older_than_the_retention() {
        let dir = tempfile::tempdir().unwrap();
        let day = Duration::from_secs(24 * 60 * 60);
        file_aged(&dir.path().join("old.json"), 31 * day);
        file_aged(&dir.path().join(".old.json.tmp"), 31 * day);
        file_aged(&dir.path().join("fresh.json"), day);
        file_aged(&dir.path().join("not-ours.txt"), 31 * day);

        let removed = prune(dir.path(), retention(None), SystemTime::now()).unwrap();

        assert_eq!(removed, 2);
        assert!(!dir.path().join("old.json").exists());
        assert!(!dir.path().join(".old.json.tmp").exists());
        assert!(dir.path().join("fresh.json").exists());
        assert!(dir.path().join("not-ours.txt").exists());
    }

    /// One file that can't be removed is counted and skipped; every other
    /// stale file still goes. The removal fails on whichever file comes
    /// first, so the pass must go on past an error to pass. MUTANT: stop the
    /// pass on the first error and the other stale files stay.
    #[test]
    fn one_unremovable_file_never_blocks_the_rest() {
        let dir = tempfile::tempdir().unwrap();
        let day = Duration::from_secs(24 * 60 * 60);
        let stale: Vec<PathBuf> = (0..5)
            .map(|n| {
                dir.path()
                    .join(format!("0f8fad5b-d9cb-469f-a165-00000000000{n}.json"))
            })
            .collect();
        for path in &stale {
            file_aged(path, 31 * day);
        }
        let fresh = dir.path().join("0f8fad5b-d9cb-469f-a165-000000000009.json");
        file_aged(&fresh, day);
        let stuck: std::cell::RefCell<Option<PathBuf>> = std::cell::RefCell::new(None);
        let remove = |path: &Path| {
            let mut stuck = stuck.borrow_mut();
            if stuck.is_none() {
                *stuck = Some(path.to_path_buf());
                return Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "stuck",
                ));
            }
            std::fs::remove_file(path)
        };

        let pruned = prune_with(dir.path(), retention(None), SystemTime::now(), &remove).unwrap();

        assert_eq!(
            pruned,
            Pruned {
                removed: 4,
                failed: 1
            }
        );
        let stuck = stuck
            .into_inner()
            .expect("one removal was attempted and failed");
        for path in &stale {
            assert_eq!(path.exists(), *path == stuck, "{}", path.display());
        }
        assert!(fresh.exists());
    }

    #[test]
    fn retention_is_a_positive_number_of_days_else_thirty() {
        let day = Duration::from_secs(24 * 60 * 60);
        assert_eq!(retention(None), 30 * day);
        assert_eq!(retention(Some("7")), 7 * day);
        assert_eq!(retention(Some("0")), 30 * day);
        assert_eq!(retention(Some("soon")), 30 * day);
    }

    /// Text kept before the owner turned keeping off still ages out: no
    /// write is needed. MUTANT: prune only on a write and the stale text
    /// stays.
    #[test]
    fn age_out_removes_stale_texts_without_a_write() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            age_out(dir.path()).unwrap(),
            0,
            "no kept texts, nothing to do"
        );
        let by_exchange = by_exchange_dir(dir.path());
        std::fs::create_dir_all(&by_exchange).unwrap();
        let stale = by_exchange.join("0f8fad5b-d9cb-469f-a165-000000000000.json");
        let fresh = by_exchange.join("0f8fad5b-d9cb-469f-a165-000000000001.json");
        file_aged(&stale, Duration::from_secs(400 * 24 * 60 * 60));
        file_aged(&fresh, Duration::from_secs(60));
        assert_eq!(age_out(dir.path()).unwrap(), 1);
        assert!(!stale.exists() && fresh.exists());
    }

    /// The wiring: a write prunes its directory.
    #[test]
    fn a_write_prunes_stale_texts_in_its_directory() {
        let dir = tempfile::tempdir().unwrap();
        let by_exchange = dir.path().join("disclosures").join("by-exchange");
        std::fs::create_dir_all(&by_exchange).unwrap();
        let stale = by_exchange.join("0f8fad5b-d9cb-469f-a165-000000000000.json");
        file_aged(&stale, Duration::from_secs(400 * 24 * 60 * 60));
        keep(
            dir.path(),
            "6f9619ff-8b86-d011-b42d-00c04fc964aa",
            None,
            &bodies(Some(json!({"model": "m"})), None),
        )
        .unwrap()
        .expect("a file");
        assert!(!stale.exists());
    }
}
