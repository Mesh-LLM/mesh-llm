//! The plugin's own checkpoint cadence: a tokio background task that runs
//! `crate::producer::checkpoint::CheckpointState` over this node's ledger.
//!
//! **On by default, local-only.** The cadence task runs
//! unless the operator opts OUT by setting [`ENV_ENABLE`] to `"off"` --
//! "on by default" is local checkpointing only: [`ENV_WITNESS_URLS`] stays
//! empty/unset by default, so no network call ever happens unless the
//! operator also sets a witness URL. A standalone checkpointing process
//! must NEVER run against the same `ledger_dir` as this cadence: both would
//! append lines to the same `checkpoints.jsonl` and race each other's chain.
//! That is an operator choice this module cannot detect and refuse safely (a
//! lock file would catch a same-host double run, not a process on another
//! box pointed at a shared mount), so it is documented rather than enforced.
//!
//! Age clock, only-if-new-activity, shutdown flush, and best-effort witness
//! retry are `CheckpointState`'s policy, not this module's -- this module
//! is purely the tokio scheduling shell: an interval loop plus a shutdown
//! signal.
//!
//! **Checkpoint at push.** The same
//! `CheckpointState` is shared (one lock, one writer of `checkpoints.jsonl`)
//! with the record-push sender, which asks [`CheckpointHandle::coverage_for`]
//! for a checkpoint covering the half it is about to push. Cuts are paced to
//! at most one per [`PUSH_COALESCE_WINDOW`]: a push whose record the latest
//! checkpoint already covers reuses it; otherwise it waits out the window
//! and the first push through cuts one checkpoint covering the whole burst.
//! Push cuts are local only -- the interval tick offers the latest one to the
//! witnesses, once per window (`CheckpointState::tick`).

use crate::capsule_emit::CapsuleState;
use crate::producer::anchor::AnchorClient;
use crate::producer::checkpoint::{
    CheckpointCadenceConfig, CheckpointRecord, CheckpointState, Coverage, PaddingSink,
};
use ed25519_dalek::SigningKey;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, PoisonError, RwLock};
use std::time::{Duration, Instant};

/// At most one push-time checkpoint per this window; halves sealed inside
/// it share the next cut.
pub const PUSH_COALESCE_WINDOW: Duration = Duration::from_millis(100);

/// On by default: the plugin runs its OWN checkpoint cadence unless the
/// operator sets `CAPSULES_CHECKPOINT_CADENCE=off` to opt out (e.g.
/// because a standalone process is already checkpointing this `ledger_dir`
/// -- see the module doc's race note). Unset, empty, `on`, `1`, `true` or
/// `yes` leaves it on; `off`, `0`, `false` or `no` turns it off; any other
/// value is read as `off` and listed as a setting problem.
pub(crate) const ENV_ENABLE: &str = "CAPSULES_CHECKPOINT_CADENCE";
/// Age-clock override, seconds. Defaults to `CheckpointCadenceConfig`'s own
/// 300s mesh default.
const ENV_INTERVAL_SECONDS: &str = "CAPSULES_CHECKPOINT_CADENCE_SECONDS";
/// Entry-count cadence override. Defaults to 100 (upstream `capsule_emit`'s
/// own default).
const ENV_CADENCE_ENTRIES: &str = "CAPSULES_CHECKPOINT_CADENCE_ENTRIES";
/// Comma-separated witness URLs to register checkpoints with. Anchoring is
/// OPT-IN, always (this repo's posture) -- empty/unset means
/// self-checkpointed only, no network.
const ENV_WITNESS_URLS: &str = "CAPSULES_CHECKPOINT_WITNESS_URLS";
/// Each witness's public key, as the operator gives it: a JSON object from
/// witness URL to its raw Ed25519 key in hex. A witness with no key here has
/// its key fetched from it once and pinned (`witness_status`).
const ENV_WITNESS_KEYS: &str = "CAPSULES_CHECKPOINT_WITNESS_KEYS";
/// `checkpoint_pad_bucket`: pad every checkpoint's leaf count up to a
/// multiple of this (Evidence Layer -00 §12.1). Defaults to
/// `DEFAULT_PAD_BUCKET` (32); `0` turns padding off.
const ENV_PAD_BUCKET: &str = "CAPSULES_CHECKPOINT_PAD_BUCKET";

/// Whether a witness URL is configured: checkpoints are offered to a witness.
/// The same setting [`spawn`] registers checkpoints with.
pub fn witness_configured() -> bool {
    !witness_urls().is_empty()
}

/// The witness URLs the operator named, in order: none by default.
pub fn witness_urls() -> Vec<String> {
    config_from_env().witness_urls
}

/// The witness list from the setting's raw value: none when it is unset or
/// holds no URL. A URL with a user, password, query or fragment is not used
/// (any of them can carry a credential, and the URL is kept in receipts and
/// shown on the page); it is logged instead.
fn witness_urls_from(raw: Option<&str>) -> Vec<String> {
    raw.unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .filter(|url| {
            let credential_free = crate::owner_maintenance::without_credentials(url) == *url;
            if !credential_free {
                tracing::warn!(
                    witness = %crate::owner_maintenance::without_credentials(url),
                    "a witness URL with a user, password, query or fragment is not used; name the witness without them"
                );
            }
            credential_free
        })
        .map(str::to_string)
        .collect()
}

/// The witness list the checkpoint cadence took when it started: the one it
/// registers with until mesh-llm restarts.
static ACTIVE_WITNESSES: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
/// The witness keys (and what is wrong with them) taken at start, just after
/// the list: two settings, read one after the other, both kept until restart.
#[allow(clippy::type_complexity)]
static ACTIVE_KEYS: std::sync::OnceLock<(std::collections::BTreeMap<String, String>, Vec<String>)> =
    std::sync::OnceLock::new();

/// The witness keys in force: those the cadence took at start, or the
/// setting as it is before then.
pub fn active_witness_keys() -> std::collections::BTreeMap<String, String> {
    ACTIVE_KEYS
        .get()
        .map(|(keys, _)| keys.clone())
        .unwrap_or_else(witness_keys)
}

/// The witnesses checkpoints actually go to, and whether the saved setting
/// now names different ones (they take effect at the next restart). Before
/// the cadence starts (or with it off), the setting as it is.
pub fn active_witness_urls() -> (Vec<String>, bool) {
    let saved = witness_urls();
    match ACTIVE_WITNESSES.get() {
        Some(active) => {
            let differs = active
                .iter()
                .map(|u| crate::witness_status::normalize_url(u))
                .collect::<Vec<_>>()
                != saved
                    .iter()
                    .map(|u| crate::witness_status::normalize_url(u))
                    .collect::<Vec<_>>();
            (active.clone(), differs)
        }
        None => (saved, false),
    }
}

/// The witness keys the operator gave, by witness URL (hex, lower-cased).
/// A value that is not a 32-byte hex key is dropped with a warning, so that
/// witness's receipts stay unchecked rather than checked under a guess.
pub fn witness_keys() -> std::collections::BTreeMap<String, String> {
    witness_keys_checked(crate::settings::var(ENV_WITNESS_KEYS).ok().as_deref()).0
}

/// What is wrong with the configured witness keys, for the page: with a
/// key that is not used, its witness's key would be pinned on first contact
/// (https only), so the operator should see it.
pub fn witness_keys_problems() -> Vec<String> {
    match ACTIVE_KEYS.get() {
        Some((_, problems)) => problems.clone(),
        None => witness_keys_checked(crate::settings::var(ENV_WITNESS_KEYS).ok().as_deref()).1,
    }
}

#[cfg(test)]
fn witness_keys_from(raw: Option<&str>) -> std::collections::BTreeMap<String, String> {
    witness_keys_checked(raw).0
}

fn witness_keys_checked(
    raw: Option<&str>,
) -> (std::collections::BTreeMap<String, String>, Vec<String>) {
    let Some(raw) = raw.map(str::trim).filter(|r| !r.is_empty()) else {
        return Default::default();
    };
    let Ok(map) = serde_json::from_str::<std::collections::BTreeMap<String, String>>(raw) else {
        let problem = format!(
            "{ENV_WITNESS_KEYS} is not a JSON object of witness URL to key, so no witness key is configured"
        );
        tracing::warn!("{problem}");
        return (Default::default(), vec![problem]);
    };
    let mut problems = Vec::new();
    let keys = map
        .into_iter()
        .filter_map(|(url, key)| {
            let key = key.trim().to_ascii_lowercase();
            if key.len() == 64 && key.bytes().all(|b| b.is_ascii_hexdigit()) {
                Some((url.trim().to_string(), key))
            } else {
                let problem = format!(
                    "the key configured for {} is not a 32-byte key in hex; it is not used",
                    crate::witness_status::display_url(&url)
                );
                tracing::warn!("{problem}");
                problems.push(problem);
                None
            }
        })
        .collect();
    (keys, problems)
}

pub fn is_enabled() -> bool {
    is_enabled_for(crate::settings::var(ENV_ENABLE).ok().as_deref())
}

/// Pure decision logic behind [`is_enabled`], taking the raw env value (or
/// `None` when unset) directly so the on-by-default / explicit-opt-out
/// behavior is unit-testable without mutating process-global env state
/// (`std::env::set_var` races across parallel `cargo test` threads).
///
/// Read as the sharing settings are: trimmed and in any case, empty is
/// unset. Unset, `on`, `1`, `true` or `yes` keeps checkpoints on (the same
/// words the referee switch takes); `off`, `0`, `false`, `no`, or a value
/// that is none of these, turns them off, so a typo never keeps a configured
/// witness contacted (the Evidence page lists the value it did not recognise).
fn is_enabled_for(raw: Option<&str>) -> bool {
    matches!(
        crate::share_policy::normalized(raw).as_deref(),
        None | Some("on" | "1" | "true" | "yes")
    )
}

fn config_from_env() -> CheckpointCadenceConfig {
    let mut cfg = CheckpointCadenceConfig::default();
    if let Ok(v) = crate::settings::var(ENV_INTERVAL_SECONDS) {
        if let Ok(n) = v.parse::<u64>() {
            cfg.cadence_seconds = n;
        }
    }
    if let Ok(v) = crate::settings::var(ENV_CADENCE_ENTRIES) {
        if let Ok(n) = v.parse::<u64>() {
            cfg.cadence_entries = n;
        }
    }
    if let Ok(v) = crate::settings::var(ENV_PAD_BUCKET) {
        if let Ok(n) = v.parse::<u64>() {
            cfg.pad_bucket = n;
        }
    }
    cfg.witness_urls = witness_urls_from(crate::settings::var(ENV_WITNESS_URLS).ok().as_deref());
    cfg
}

/// The latest signed checkpoint head this node has produced, held in
/// memory. The checkpoint-head source's sending half reads this to
/// feed `peer_root_ledger.rs`'s outbound side / `PeerAnnouncement.checkpoint`
/// -- this task itself does not gossip anything; it only produces and holds
/// the head.
#[derive(Clone, Default)]
pub struct LatestHead(Arc<RwLock<Option<CheckpointRecord>>>);

impl LatestHead {
    /// Not read anywhere in this binary yet -- the checkpoint-head source's
    /// sending half is the intended caller (see this struct's doc comment).
    /// Kept as the accessor that wiring needs, same "tested, ready-to-wire"
    /// pattern `peer_root_ledger.rs`'s module doc uses for its own
    /// not-yet-wired receiving half.
    #[allow(dead_code)]
    pub fn get(&self) -> Option<CheckpointRecord> {
        self.0
            .read()
            .expect("latest checkpoint head lock poisoned")
            .clone()
    }

    fn set(&self, cp: CheckpointRecord) {
        // A push cut and a tick can both publish; keep the larger head so a
        // late writer never moves the head backwards.
        let mut head = self
            .0
            .write()
            .expect("latest checkpoint head lock poisoned");
        if head
            .as_ref()
            .is_some_and(|current| current.mmr_size >= cp.mmr_size)
        {
            return;
        }
        *head = Some(cp);
    }
}

/// The shared checkpoint state: the interval task and the record-push
/// sender both go through this, so there is one writer of
/// `checkpoints.jsonl` per ledger.
#[derive(Clone)]
pub struct CheckpointHandle {
    state: Arc<Mutex<CheckpointState>>,
    signer: SigningKey,
    anchor: Arc<AnchorClient>,
    latest_head: LatestHead,
    last_push_cut: Arc<Mutex<Option<Instant>>>,
}

/// One pass of the pacing loop in [`CheckpointHandle::coverage_for`].
enum PushStep {
    Covered(Box<Coverage>),
    WaitFor(Duration),
}

/// How long a push must still wait before it may cut, given when the last
/// push cut happened. Zero when it may cut now.
fn remaining_window(last_cut: Option<Instant>, now: Instant) -> Duration {
    match last_cut {
        Some(at) => PUSH_COALESCE_WINDOW.saturating_sub(now.saturating_duration_since(at)),
        None => Duration::ZERO,
    }
}

impl CheckpointHandle {
    fn new(state: CheckpointState, signer: SigningKey) -> Self {
        Self {
            state: Arc::new(Mutex::new(state)),
            signer,
            anchor: Arc::new(anchor_for()),
            latest_head: LatestHead::default(),
            last_push_cut: Arc::new(Mutex::new(None)),
        }
    }

    pub fn latest_head(&self) -> LatestHead {
        self.latest_head.clone()
    }

    /// A signed checkpoint covering `capsule_id` (this node's own
    /// just-sealed half) plus its inclusion proof -- see the module doc's
    /// "checkpoint at push" note for the reuse/pacing rules.
    pub async fn coverage_for(&self, capsule_id: &str) -> anyhow::Result<Coverage> {
        loop {
            let this = self.clone();
            let id = capsule_id.to_string();
            let step = tokio::task::spawn_blocking(move || this.step(&id))
                .await
                .map_err(|e| anyhow::anyhow!("push checkpoint task did not complete: {e}"))??;
            match step {
                PushStep::Covered(coverage) => return Ok(*coverage),
                PushStep::WaitFor(wait) => tokio::time::sleep(wait).await,
            }
        }
    }

    /// Under the state lock: reuse the latest checkpoint if it covers the
    /// leaf; otherwise cut one now if the window allows, else say how long
    /// to wait. Holding the lock across the check and the cut is what makes
    /// "at most one cut per window" hold under concurrent pushes.
    fn step(&self, capsule_id: &str) -> anyhow::Result<PushStep> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(coverage) = state.existing_coverage(capsule_id)? {
            return Ok(PushStep::Covered(Box::new(coverage)));
        }
        let mut last_cut = self
            .last_push_cut
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let wait = remaining_window(*last_cut, Instant::now());
        if !wait.is_zero() {
            return Ok(PushStep::WaitFor(wait));
        }
        let coverage = state.checkpoint_covering(capsule_id, &self.signer, &self.anchor)?;
        *last_cut = Some(Instant::now());
        self.latest_head.set(coverage.checkpoint.clone());
        Ok(PushStep::Covered(Box::new(coverage)))
    }
}

/// The checkpoint state's padding hook over this node's one ledger writer.
struct LedgerPadder(Arc<CapsuleState>);

impl PaddingSink for LedgerPadder {
    fn pad_to_bucket(&self, bucket: u64) -> Result<u64, String> {
        self.0
            .pad_ledger_to_bucket(bucket)
            .map_err(|e| e.to_string())
    }
}

/// Spawn the background cadence task over `ledger_dir`, using `signer` for
/// every checkpoint this task ever signs (always through the
/// `CheckpointSigner` trait -- see `checkpoint.rs`'s module doc). Returns
/// the shared [`CheckpointHandle`] immediately; the task itself runs until
/// `shutdown` fires, performing a final flush before returning.
///
/// Startup catch-up (`reconnect`) runs before the interval loop starts, one
/// interval tick runs `tick()`, and a shutdown signal triggers
/// `checkpoint_on_shutdown()`.
///
/// `capsules` is the ledger's one writer: every checkpoint is padded through
/// it to `checkpoint_pad_bucket` before it is cut (see [`LedgerPadder`]).
pub fn spawn(
    ledger_dir: PathBuf,
    log_id: String,
    signer: SigningKey,
    capsules: Arc<CapsuleState>,
    mut shutdown: tokio::sync::watch::Receiver<bool>,
) -> anyhow::Result<CheckpointHandle> {
    let cfg = config_from_env();
    let witness_urls = cfg.witness_urls.clone();
    let _ = ACTIVE_WITNESSES.set(witness_urls.clone());
    let _ = ACTIVE_KEYS.set(witness_keys_checked(
        crate::settings::var(ENV_WITNESS_KEYS).ok().as_deref(),
    ));
    let log_id_for_witnesses = log_id.clone();
    let keys_dir = ledger_dir.clone();
    let interval = Duration::from_secs(cfg.cadence_seconds);
    let pad_bucket = cfg.pad_bucket;
    let (mut state, report) = CheckpointState::load(&ledger_dir, log_id, cfg).map_err(|e| {
        anyhow::anyhow!(
            "failed to load checkpoint state at {}: {e}",
            ledger_dir.display()
        )
    })?;
    if let Some(open_report) = &report.node_store {
        if open_report.truncated_bytes > 0 {
            tracing::warn!(
                truncated_bytes = open_report.truncated_bytes,
                "checkpoint node store had a torn trailing record, truncated on open"
            );
        }
    }
    state.set_padding_sink(Box::new(LedgerPadder(capsules)));
    tracing::info!(
        leaves_indexed_this_load = report.leaves_indexed_this_load,
        leaf_count = state.leaf_count(),
        pad_bucket,
        "checkpoint cadence task starting"
    );

    let handle = CheckpointHandle::new(state, signer);
    let task = handle.clone();

    tokio::spawn(async move {
        let run = |phase: &'static str,
                   f: fn(
            &mut CheckpointState,
            &SigningKey,
            &AnchorClient,
        ) -> Result<
            Option<CheckpointRecord>,
            crate::producer::checkpoint::CheckpointStateError,
        >| {
            let result = {
                let mut state = task.state.lock().unwrap_or_else(PoisonError::into_inner);
                f(&mut state, &task.signer, &task.anchor)
            };
            if let Some(cp) = report_checkpoint(result, phase) {
                task.latest_head.set(cp);
            }
        };

        run("startup reconnect", |s, k, a| s.reconnect(k, a));

        let mut ticker = tokio::time::interval(interval);
        // The first tick fires immediately; the startup reconnect above
        // already did that work, so skip it.
        ticker.tick().await;

        loop {
            tokio::select! {
                _ = ticker.tick() => {
                    run("tick", |s, k, a| s.tick(k, a));
                    // Each named witness: its key fetched when none is given,
                    // and one that lags asked again so its row can say why.
                    // Only the named witnesses, and only when one is named.
                    if !witness_urls.is_empty() {
                        let (dir, urls) = (keys_dir.clone(), witness_urls.clone());
                        let configured = active_witness_keys();
                        let log_id = log_id_for_witnesses.clone();
                        let _ = tokio::task::spawn_blocking(move || {
                            crate::witness_status::refresh_witnesses(&dir, &urls, &configured, Some(&log_id))
                        })
                        .await;
                    }
                }
                _ = shutdown.changed() => {
                    if *shutdown.borrow() {
                        break;
                    }
                }
            }
        }

        run("shutdown flush", |s, k, a| s.checkpoint_on_shutdown(k, a));
        tracing::info!("checkpoint cadence task stopped");
    });

    Ok(handle)
}

/// The client capsule-emit's checkpoint calls take. Since capsule-emit
/// fd74008 each named witness is reached through its own client, at its own
/// URL, and this one is never called; it is given no address at all, so no
/// default witness can ever be reached through it.
fn anchor_for() -> AnchorClient {
    AnchorClient::new(String::new())
}

fn report_checkpoint(
    result: Result<Option<CheckpointRecord>, crate::producer::checkpoint::CheckpointStateError>,
    phase: &str,
) -> Option<CheckpointRecord> {
    match result {
        Ok(Some(cp)) => {
            tracing::info!(
                mmr_size = cp.mmr_size,
                root = %cp.root,
                witnessed = !cp.witnesses.is_empty(),
                %phase,
                "checkpoint emitted"
            );
            Some(cp)
        }
        Ok(None) => None,
        Err(err) => {
            // Never let a checkpoint-layer failure disturb the serving
            // path or crash the plugin -- best-effort observability, as
            // for the witness leg (see checkpoint.rs's module doc).
            tracing::warn!(%err, %phase, "checkpoint cadence step failed");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Nothing set means no witness at all, so nothing is ever registered
    /// anywhere and no witness is contacted (`witness_status` contacts only a
    /// named one, and registration walks the named ones only).
    #[test]
    fn unset_means_no_witness_and_no_outbound() {
        assert!(witness_urls_from(None).is_empty());
        assert!(witness_urls_from(Some("")).is_empty());
        assert!(witness_urls_from(Some(" , ,")).is_empty());
        assert_eq!(
            witness_urls_from(Some("https://a.example, https://b.example")),
            ["https://a.example", "https://b.example"]
        );
        assert!(CheckpointCadenceConfig::default().witness_urls.is_empty());
        let dir = tempfile::tempdir().unwrap();
        crate::witness_status::refresh_witnesses(
            dir.path(),
            &witness_urls_from(None),
            &Default::default(),
            None,
        );
        assert_eq!(
            std::fs::read_dir(dir.path()).unwrap().count(),
            0,
            "nothing contacted, nothing written"
        );
    }

    #[test]
    fn a_witness_url_with_a_credential_or_a_query_is_not_used() {
        assert!(witness_urls_from(Some("https://user:pw@w.example")).is_empty());
        assert!(witness_urls_from(Some("https://w.example/?token=abc")).is_empty());
        assert!(witness_urls_from(Some("https://w.example#frag")).is_empty());
        assert_eq!(
            witness_urls_from(Some("https://w.example/, https://x.example/log")),
            ["https://w.example/", "https://x.example/log"]
        );
    }

    #[test]
    fn a_malformed_key_setting_is_a_problem_to_show() {
        let (keys, problems) = witness_keys_checked(Some("not json"));
        assert!(keys.is_empty());
        assert!(problems[0].contains("not a JSON object"));
        let (keys, problems) = witness_keys_checked(Some(r#"{"https://w.example": "short"}"#));
        assert!(keys.is_empty());
        assert!(problems[0].contains("w.example"));
        assert!(witness_keys_checked(None).1.is_empty());
    }

    #[test]
    fn configured_witness_keys_are_hex_32_byte_keys_by_url() {
        let k = "AB".repeat(32);
        let keys = witness_keys_from(Some(&format!(
            r#"{{"https://a.example": "{k}", "https://b.example": "short"}}"#
        )));
        assert_eq!(keys.len(), 1);
        assert_eq!(keys["https://a.example"], "ab".repeat(32));
        assert!(witness_keys_from(Some("not json")).is_empty());
        assert!(witness_keys_from(None).is_empty());
    }

    #[test]
    fn on_by_default_when_unset() {
        assert!(is_enabled_for(None));
    }

    #[test]
    fn explicit_off_disables() {
        assert!(!is_enabled_for(Some("off")));
    }

    #[test]
    fn on_or_empty_stays_on_and_any_spelling_of_off_or_a_typo_is_off() {
        for on in ["on", " On ", "1", "true", "TRUE", "yes", "Yes "] {
            assert!(is_enabled_for(Some(on)), "{on:?}");
        }
        assert!(is_enabled_for(Some("")), "empty is unset");
        for off in [
            "OFF", "Off", "off ", " off", "0", "false", "no", "of", "disabled",
        ] {
            assert!(!is_enabled_for(Some(off)), "{off:?}");
        }
    }

    #[test]
    fn remaining_window_is_zero_with_no_prior_cut_or_after_the_window() {
        let now = Instant::now();
        assert_eq!(remaining_window(None, now), Duration::ZERO);
        assert_eq!(
            remaining_window(Some(now), now + PUSH_COALESCE_WINDOW),
            Duration::ZERO
        );
        assert_eq!(
            remaining_window(Some(now), now + Duration::from_millis(30)),
            Duration::from_millis(70)
        );
    }

    fn append_capsule(dir: &std::path::Path, seed: &str) -> String {
        use sha2::{Digest, Sha256};
        use std::io::Write;
        let capsule_id = hex::encode(Sha256::digest(seed.as_bytes()));
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir.join("capsules.jsonl"))
            .unwrap();
        writeln!(f, "{}", serde_json::json!({ "capsule_id": capsule_id })).unwrap();
        capsule_id
    }

    fn handle_over(dir: &std::path::Path) -> CheckpointHandle {
        let (state, _) =
            CheckpointState::load(dir, "test-log", CheckpointCadenceConfig::default()).unwrap();
        CheckpointHandle::new(state, SigningKey::from_bytes(&[9u8; 32]))
    }

    struct TempDir(std::path::PathBuf);
    impl TempDir {
        fn path(&self) -> &std::path::Path {
            &self.0
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn checkpoint_lines(dir: &std::path::Path) -> usize {
        std::fs::read_to_string(dir.join("checkpoints.jsonl"))
            .map(|s| s.lines().count())
            .unwrap_or(0)
    }

    /// A burst of halves sealed inside one window shares ONE checkpoint: the
    /// first push cuts, the second (sealed after that cut) waits out the
    /// window and cuts once more, covering both itself and anything else
    /// sealed meanwhile.
    #[tokio::test]
    async fn pushes_inside_one_window_cut_at_most_once_per_window() {
        let dir = std::env::temp_dir().join(format!("cadence-push-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let dir = TempDir(dir);
        let handle = handle_over(dir.path());

        let a = append_capsule(dir.path(), "a");
        let first = handle.coverage_for(&a).await.unwrap();
        assert!(first.cut_new);

        let b = append_capsule(dir.path(), "b");
        let c = append_capsule(dir.path(), "c");
        let started = Instant::now();
        let (cov_b, cov_c) = tokio::join!(handle.coverage_for(&b), handle.coverage_for(&c));
        let (cov_b, cov_c) = (cov_b.unwrap(), cov_c.unwrap());
        assert!(
            started.elapsed() >= PUSH_COALESCE_WINDOW - Duration::from_millis(10),
            "the second cut must wait out the window"
        );
        assert_eq!(
            cov_b.checkpoint, cov_c.checkpoint,
            "b and c share one checkpoint"
        );
        assert_eq!(
            checkpoint_lines(dir.path()),
            2,
            "two windows, two cuts -- never one per push"
        );
        assert_eq!(handle.latest_head().get(), Some(cov_c.checkpoint.clone()));

        // Already covered: returns at once, no new cut.
        let again = handle.coverage_for(&a).await.unwrap();
        assert!(!again.cut_new);
        assert_eq!(checkpoint_lines(dir.path()), 2);
    }

    #[test]
    fn latest_head_never_moves_backwards() {
        let head = LatestHead::default();
        let cp = |size: u64| CheckpointRecord {
            v: 1,
            kind: "mmr_checkpoint".into(),
            log_id: "l".into(),
            mmr_size: size,
            root: String::new(),
            prev_size: 0,
            prev_root: String::new(),
            key_id: String::new(),
            timestamp: String::new(),
            signature: String::new(),
            witnesses: Vec::new(),
        };
        head.set(cp(7));
        head.set(cp(3));
        assert_eq!(head.get().unwrap().mmr_size, 7);
    }
}
