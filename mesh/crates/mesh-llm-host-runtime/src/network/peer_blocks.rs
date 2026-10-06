//! Peers this node's operator chose to stop routing to.
//!
//! Local only, like `target_health`: never gossiped, never shared. A shared
//! list would let any peer get an honest one excluded with made-up reports.
//! A block lasts seven days or until it is undone, and survives a restart
//! (`peer_blocks.json` in the identity state directory).
//!
//! The operator sets blocks through the loopback-only management API. A
//! plugin may *request* one through the host (`PeerBlockRequest`); the block
//! then names the plugin as its requester and carries the plugin's `reason`,
//! which the host stores and echoes but never reads. A plugin can change or
//! undo only a block it requested; the operator can change or undo any, and
//! re-blocking over a plugin's block takes it over. Every change is published
//! once as a [`RoutingChoice`] on [`ROUTING_CHOICE_CHANNEL`] to the local
//! plugins that declare it. The host keeps no ranking and no history.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

use iroh::EndpointId;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::inference::election::{InferenceTarget, ModelTargets};

/// The local plugin channel a [`RoutingChoice`] is published on.
pub(crate) const ROUTING_CHOICE_CHANNEL: &str = "routing.choice.v1";
const FILE_NAME: &str = "peer_blocks.json";
pub(crate) const TIMED_BLOCK_MS: u64 = 7 * 24 * 60 * 60 * 1000;
/// The largest `reason` the host keeps, as serialized JSON.
pub(crate) const MAX_REASON_BYTES: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RoutingChange {
    Block,
    Unblock,
}

/// How long a new block lasts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum BlockLength {
    SevenDays,
    UntilUndone,
}

/// Who asked for a change: `operator`, or `plugin:<id>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Requester {
    Operator,
    Plugin(String),
}

impl fmt::Display for Requester {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Operator => f.write_str("operator"),
            Self::Plugin(id) => write!(f, "plugin:{id}"),
        }
    }
}

impl Serialize for Requester {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for Requester {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        match text.strip_prefix("plugin:") {
            Some(id) if !id.is_empty() => Ok(Self::Plugin(id.to_string())),
            _ if text == "operator" => Ok(Self::Operator),
            _ => Err(serde::de::Error::custom(format!(
                "unknown requester `{text}`"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct ActiveBlock {
    pub(crate) blocked_at_ms: u64,
    /// `None` = until it is undone.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) until_ms: Option<u64>,
    pub(crate) requested_by: Requester,
    /// Opaque to the host: stored and echoed, never read.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) reason: Option<serde_json::Value>,
}

/// One block or unblock, as published to plugins.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct RoutingChoice {
    pub(crate) change: RoutingChange,
    /// The peer's endpoint id, lowercase hex.
    pub(crate) peer: String,
    pub(crate) at_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) until_ms: Option<u64>,
    pub(crate) requested_by: Requester,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) reason: Option<serde_json::Value>,
}

/// Run a block-store mutation on the blocking pool.
///
/// Every mutation's durable save fsyncs, and both callers (the loopback
/// management API and the plugin request path) are async, so the save must not
/// run on an async worker. A worker that fails to join is reported as a save
/// failure, which both callers already surface as an error.
pub(crate) async fn offload<T, F>(mutation: F) -> Result<T, BlockError>
where
    F: FnOnce() -> Result<T, BlockError> + Send + 'static,
    T: Send + 'static,
{
    tokio::task::spawn_blocking(mutation)
        .await
        .map_err(|error| {
            BlockError::Save(std::io::Error::other(format!(
                "peer block worker failed: {error}"
            )))
        })?
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum BlockError {
    #[error("this peer is not blocked")]
    NotBlocked,
    #[error("only the operator or the plugin that requested this block can change it")]
    NotRequester,
    #[error("reason must be at most {MAX_REASON_BYTES} bytes of JSON")]
    ReasonTooLarge,
    #[error("could not save peer blocks: {0}")]
    Save(#[from] std::io::Error),
}

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
struct Store {
    /// Keyed by the peer's endpoint id, lowercase hex.
    #[serde(default)]
    blocks: BTreeMap<String, ActiveBlock>,
}

impl Store {
    fn prune(&mut self, now_ms: u64) {
        self.blocks
            .retain(|_, block| block.until_ms.is_none_or(|until| now_ms < until));
    }
}

/// Shared handle to the block store. Clones share state.
#[derive(Clone, Debug)]
pub(crate) struct PeerBlocks {
    /// Read by routing on every request; swapped whole after a write is saved.
    store: Arc<RwLock<Store>>,
    /// Serialises writers across build-save-swap, so a slow disk never holds
    /// the lock routing reads.
    writer: Arc<Mutex<()>>,
    /// `None` keeps the store in memory only (tests, no state directory, or
    /// a file that could not be loaded).
    directory: Option<PathBuf>,
    /// Set when the file could not be loaded: why nothing is saved.
    not_saved: Option<Arc<str>>,
}

pub(crate) fn peer_key(peer: &EndpointId) -> String {
    hex::encode(peer.as_bytes())
}

/// Parse a 64-hex endpoint id.
pub(crate) fn parse_peer(text: &str) -> Option<EndpointId> {
    let bytes: [u8; 32] = hex::decode(text.trim()).ok()?.try_into().ok()?;
    EndpointId::from_bytes(&bytes).ok()
}

pub(crate) fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| {
            u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX)
        })
}

impl PeerBlocks {
    fn with_store(store: Store, directory: Option<PathBuf>) -> Self {
        Self {
            store: Arc::new(RwLock::new(store)),
            writer: Arc::default(),
            directory,
            not_saved: None,
        }
    }

    pub(crate) fn in_memory() -> Self {
        Self::with_store(Store::default(), None)
    }

    /// Load the store saved in `directory`.
    ///
    /// A missing file is an empty store. A bad file must never stop routing,
    /// so any other failure starts with no blocks, but never at the cost of
    /// the file: a file that cannot be read, or that cannot be decoded and
    /// then cannot be set aside as `peer_blocks.json.corrupt-<ms>`, is left
    /// where it is and nothing is saved for the rest of the process (see
    /// [`Self::not_saved`]), so a later change can never overwrite it.
    pub(crate) fn load(directory: &Path) -> Self {
        Self::load_at(directory, now_ms())
    }

    fn load_at(directory: &Path, now_ms: u64) -> Self {
        let path = directory.join(FILE_NAME);
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Self::with_store(Store::default(), Some(directory.to_path_buf()));
            }
            Err(error) => {
                return Self::not_persisting(format!(
                    "{} could not be read ({error})",
                    path.display()
                ));
            }
        };
        let error = match serde_json::from_slice(&bytes) {
            Ok(store) => return Self::with_store(store, Some(directory.to_path_buf())),
            Err(error) => error,
        };
        let aside = path.with_extension(format!("json.corrupt-{now_ms}"));
        match std::fs::rename(&path, &aside) {
            Ok(()) => {
                tracing::warn!(
                    "peer blocks: {} is unreadable ({error}); moved to {} and starting empty",
                    path.display(),
                    aside.display()
                );
                Self::with_store(Store::default(), Some(directory.to_path_buf()))
            }
            Err(rename_error) => Self::not_persisting(format!(
                "{} is unreadable ({error}) and could not be moved to {} ({rename_error})",
                path.display(),
                aside.display()
            )),
        }
    }

    /// An empty store that never saves, because the file on disk could not
    /// be loaded or set aside and must not be overwritten.
    fn not_persisting(problem: String) -> Self {
        tracing::error!(
            "peer blocks: {problem}; starting with no blocks, and changes this run will not be saved"
        );
        Self {
            not_saved: Some(Arc::from(problem)),
            ..Self::with_store(Store::default(), None)
        }
    }

    /// Why changes are not being saved this run, when the file could not be
    /// loaded. `None` when blocks persist (or the store is in memory only by
    /// design).
    pub(crate) fn not_saved(&self) -> Option<&str> {
        self.not_saved.as_deref()
    }

    /// The store for this node's identity, or an in-memory one when the
    /// identity state directory cannot be resolved.
    pub(crate) fn for_this_node() -> Self {
        match crate::mesh::identity_state_dir() {
            Ok(directory) => Self::load(&directory),
            Err(error) => {
                tracing::warn!(
                    "peer blocks: no state directory ({error}); blocks will not persist"
                );
                Self::in_memory()
            }
        }
    }

    /// Write the store durably: the new file is synced before it replaces
    /// the old one, and the directory is synced after, so a crash or power
    /// loss leaves either the old store or the new one, never a torn file.
    fn save(&self, store: &Store) -> std::io::Result<()> {
        use std::io::Write as _;

        let Some(directory) = &self.directory else {
            return Ok(());
        };
        std::fs::create_dir_all(directory)?;
        let path = directory.join(FILE_NAME);
        let temp = path.with_extension("json.tmp");
        {
            let mut file = std::fs::File::create(&temp)?;
            file.write_all(&serde_json::to_vec_pretty(store)?)?;
            file.sync_all()?;
        }
        std::fs::rename(&temp, &path)?;
        sync_directory(directory)
    }

    fn read(&self) -> std::sync::RwLockReadGuard<'_, Store> {
        self.store
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Build the next store from the current one, save it, and only then make
    /// it the one routing reads. A failed save or a refused change leaves the
    /// store in force as it was. (If only the final directory sync fails, the
    /// renamed file may already hold the change, and a restart would load it.)
    fn update<T>(
        &self,
        change: impl FnOnce(&mut Store) -> Result<T, BlockError>,
    ) -> Result<T, BlockError> {
        let _writer = self
            .writer
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut next = self.read().clone();
        let out = change(&mut next)?;
        self.save(&next)?;
        *self
            .store
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = next;
        Ok(out)
    }

    pub(crate) fn is_blocked(&self, peer: &EndpointId, now_ms: u64) -> bool {
        self.read()
            .blocks
            .get(&peer_key(peer))
            .is_some_and(|block| block.until_ms.is_none_or(|until| now_ms < until))
    }

    /// Drop blocked peers from a live host list.
    pub(crate) fn retain_unblocked(&self, hosts: &mut Vec<EndpointId>, now_ms: u64) {
        hosts.retain(|host| !self.is_blocked(host, now_ms));
    }

    /// Drop blocked remote targets from a routing snapshot. Local targets are
    /// never affected.
    pub(crate) fn without_blocked(&self, targets: &ModelTargets, now_ms: u64) -> ModelTargets {
        if self.read().blocks.is_empty() {
            return targets.clone();
        }
        let mut filtered = targets.clone();
        for hosts in filtered.targets.values_mut() {
            hosts.retain(|target| {
                !matches!(target, InferenceTarget::Remote(peer) if self.is_blocked(peer, now_ms))
            });
        }
        filtered
    }

    /// Start blocking `peer`. Blocking a peer that is already blocked
    /// replaces the block (its length, requester and reason), but only for the
    /// operator or the plugin that requested it; the operator re-blocking over
    /// a plugin's block takes it over.
    pub(crate) fn block(
        &self,
        peer: &EndpointId,
        length: BlockLength,
        requested_by: Requester,
        reason: Option<serde_json::Value>,
        now_ms: u64,
    ) -> Result<RoutingChoice, BlockError> {
        if reason
            .as_ref()
            .is_some_and(|reason| reason.to_string().len() > MAX_REASON_BYTES)
        {
            return Err(BlockError::ReasonTooLarge);
        }
        let until_ms = match length {
            BlockLength::SevenDays => Some(now_ms.saturating_add(TIMED_BLOCK_MS)),
            BlockLength::UntilUndone => None,
        };
        self.update(|store| {
            store.prune(now_ms);
            // A plugin may not re-block over someone else's block: that would
            // make it the requester, and so able to undo it.
            if requested_by != Requester::Operator
                && store
                    .blocks
                    .get(&peer_key(peer))
                    .is_some_and(|block| block.requested_by != requested_by)
            {
                return Err(BlockError::NotRequester);
            }
            store.blocks.insert(
                peer_key(peer),
                ActiveBlock {
                    blocked_at_ms: now_ms,
                    until_ms,
                    requested_by: requested_by.clone(),
                    reason: reason.clone(),
                },
            );
            Ok(RoutingChoice {
                change: RoutingChange::Block,
                peer: peer_key(peer),
                at_ms: now_ms,
                until_ms,
                requested_by,
                reason,
            })
        })
    }

    /// Stop blocking `peer`. The operator can undo any block; a plugin only
    /// one it requested.
    pub(crate) fn unblock(
        &self,
        peer: &EndpointId,
        requested_by: Requester,
        reason: Option<serde_json::Value>,
        now_ms: u64,
    ) -> Result<RoutingChoice, BlockError> {
        if reason
            .as_ref()
            .is_some_and(|reason| reason.to_string().len() > MAX_REASON_BYTES)
        {
            return Err(BlockError::ReasonTooLarge);
        }
        self.update(|store| {
            store.prune(now_ms);
            let key = peer_key(peer);
            let block = store.blocks.get(&key).ok_or(BlockError::NotBlocked)?;
            if requested_by != Requester::Operator && block.requested_by != requested_by {
                return Err(BlockError::NotRequester);
            }
            store.blocks.remove(&key);
            Ok(RoutingChoice {
                change: RoutingChange::Unblock,
                peer: key,
                at_ms: now_ms,
                until_ms: None,
                requested_by,
                reason,
            })
        })
    }

    /// Active blocks, keyed by peer.
    pub(crate) fn snapshot(&self, now_ms: u64) -> BTreeMap<String, ActiveBlock> {
        let mut store = self.read().clone();
        store.prune(now_ms);
        store.blocks
    }
}

/// Make a rename inside `directory` durable. On Unix that means syncing the
/// directory itself. The standard library cannot open a directory as a file
/// on Windows, so there the file is synced before the rename and the
/// rename's own durability is left to the filesystem.
#[cfg(unix)]
fn sync_directory(directory: &Path) -> std::io::Result<()> {
    std::fs::File::open(directory)?.sync_all()
}

#[cfg(not(unix))]
fn sync_directory(_directory: &Path) -> std::io::Result<()> {
    Ok(())
}

/// Deliver `choice` once to the local plugins that declare
/// [`ROUTING_CHOICE_CHANNEL`]. Best effort: a plugin that is slow or gone
/// never holds up or undoes the change, which is already in force.
pub(crate) async fn publish(node: &crate::mesh::Node, choice: &RoutingChoice) {
    let Some(manager) = node.plugin_manager().await else {
        return;
    };
    if !manager
        .any_plugin_declares_mesh_channel(ROUTING_CHOICE_CHANNEL)
        .await
    {
        return;
    }
    let Ok(body) = serde_json::to_vec(choice) else {
        return;
    };
    match tokio::time::timeout(
        std::time::Duration::from_secs(1),
        manager.broadcast_channel_message(
            ROUTING_CHOICE_CHANNEL,
            "application/json",
            body,
            &choice.peer,
        ),
    )
    .await
    {
        Ok(Ok(())) => {}
        Ok(Err(error)) => tracing::debug!("routing choice not delivered: {error}"),
        Err(_) => tracing::debug!("routing choice delivery timed out"),
    }
}

/// Apply a plugin's `PeerBlockRequest`, as a change requested by
/// `plugin:<plugin_id>`, and publish it. Refused unless the operator has set
/// `allow_peer_blocks = true` on that plugin's config entry.
pub(crate) async fn apply_plugin_request(
    node: &crate::mesh::Node,
    plugin_id: String,
    request: crate::plugin::proto::PeerBlockRequest,
) -> Result<crate::plugin::proto::PeerBlockResponse, crate::plugin::proto::ErrorResponse> {
    let allowed = plugin_may_request(node.config_state.lock().await.config(), &plugin_id);
    let blocks = node.peer_blocks.clone();
    let choice = tokio::task::spawn_blocking(move || {
        plugin_request_choice(&blocks, plugin_id, allowed, request, now_ms())
    })
    .await
    .map_err(|error| request_error(format!("peer block worker failed: {error}")))??;
    publish(node, &choice).await;
    let choice_json =
        serde_json::to_string(&choice).map_err(|error| request_error(error.to_string()))?;
    Ok(crate::plugin::proto::PeerBlockResponse { choice_json })
}

/// Whether the operator lets `plugin_id` request peer blocks. Read from the
/// live config on every request, so turning it off takes effect at once.
fn plugin_may_request(config: &crate::plugin::MeshConfig, plugin_id: &str) -> bool {
    config
        .plugins
        .iter()
        .any(|plugin| plugin.name == plugin_id && plugin.peer_blocks_allowed())
}

fn plugin_request_choice(
    blocks: &PeerBlocks,
    plugin_id: String,
    allowed: bool,
    request: crate::plugin::proto::PeerBlockRequest,
    now_ms: u64,
) -> Result<RoutingChoice, crate::plugin::proto::ErrorResponse> {
    use crate::plugin::proto::peer_block_request::{Change, Length};

    if !allowed {
        return Err(request_error(format!(
            "plugin `{plugin_id}` may not change peer blocks: the operator has not set \
             allow_peer_blocks = true for it"
        )));
    }
    let peer = parse_peer(&request.peer_id)
        .ok_or_else(|| request_error("peer_id must be a 64-hex endpoint id"))?;
    let reason = request
        .reason_json
        .as_deref()
        .map(serde_json::from_str::<serde_json::Value>)
        .transpose()
        .map_err(|error| request_error(format!("reason_json is not JSON: {error}")))?;
    let requester = Requester::Plugin(plugin_id);
    let result = match request.change() {
        Change::Block => {
            let length = match request.length() {
                Length::SevenDays => BlockLength::SevenDays,
                Length::UntilUndone => BlockLength::UntilUndone,
                Length::Unspecified => return Err(request_error("a block needs a length")),
            };
            blocks.block(&peer, length, requester, reason, now_ms)
        }
        Change::Unblock => blocks.unblock(&peer, requester, reason, now_ms),
        Change::Unspecified => return Err(request_error("change must be BLOCK or UNBLOCK")),
    };
    result.map_err(|error| request_error(error.to_string()))
}

fn request_error(message: impl Into<String>) -> crate::plugin::proto::ErrorResponse {
    crate::plugin::proto::ErrorResponse {
        code: rmcp::model::ErrorCode::INVALID_PARAMS.0,
        message: message.into(),
        data_json: String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn peer() -> EndpointId {
        iroh::SecretKey::generate().public()
    }

    fn plugin(id: &str) -> Requester {
        Requester::Plugin(id.into())
    }

    #[test]
    fn timed_block_holds_for_seven_days_then_lapses() {
        let blocks = PeerBlocks::in_memory();
        let bad = peer();
        blocks
            .block(
                &bad,
                BlockLength::SevenDays,
                Requester::Operator,
                None,
                1_000,
            )
            .unwrap();
        assert!(blocks.is_blocked(&bad, 1_000));
        assert!(blocks.is_blocked(&bad, 1_000 + TIMED_BLOCK_MS - 1));
        assert!(!blocks.is_blocked(&bad, 1_000 + TIMED_BLOCK_MS));
        assert!(!blocks.is_blocked(&peer(), 1_000));
        assert!(
            blocks.snapshot(1_000 + TIMED_BLOCK_MS).is_empty(),
            "a lapsed block is not listed"
        );
    }

    #[test]
    fn until_undone_holds_until_unblocked() {
        let blocks = PeerBlocks::in_memory();
        let bad = peer();
        blocks
            .block(&bad, BlockLength::UntilUndone, Requester::Operator, None, 5)
            .unwrap();
        assert!(blocks.is_blocked(&bad, u64::MAX - 1));
        let undone = blocks.unblock(&bad, Requester::Operator, None, 6).unwrap();
        assert_eq!(undone.change, RoutingChange::Unblock);
        assert!(!blocks.is_blocked(&bad, 10));
        assert!(matches!(
            blocks.unblock(&bad, Requester::Operator, None, 11),
            Err(BlockError::NotBlocked)
        ));
    }

    #[test]
    fn a_plugin_can_undo_only_its_own_block_and_the_operator_any() {
        let blocks = PeerBlocks::in_memory();
        let (by_plugin, by_operator) = (peer(), peer());
        let reason = serde_json::json!({ "rule": "opaque to the host" });
        let choice = blocks
            .block(
                &by_plugin,
                BlockLength::SevenDays,
                plugin("a"),
                Some(reason.clone()),
                0,
            )
            .unwrap();
        assert_eq!(choice.requested_by.to_string(), "plugin:a");
        assert_eq!(choice.reason.as_ref(), Some(&reason));
        blocks
            .block(
                &by_operator,
                BlockLength::UntilUndone,
                Requester::Operator,
                None,
                0,
            )
            .unwrap();

        assert!(matches!(
            blocks.unblock(&by_plugin, plugin("b"), None, 1),
            Err(BlockError::NotRequester)
        ));
        assert!(matches!(
            blocks.unblock(&by_operator, plugin("a"), None, 1),
            Err(BlockError::NotRequester)
        ));
        assert!(blocks.is_blocked(&by_plugin, 1) && blocks.is_blocked(&by_operator, 1));

        blocks.unblock(&by_plugin, plugin("a"), None, 2).unwrap();
        blocks
            .unblock(&by_operator, Requester::Operator, None, 2)
            .unwrap();
        assert!(blocks.snapshot(3).is_empty());
    }

    #[test]
    fn a_plugin_cannot_take_over_a_block_it_did_not_request() {
        let blocks = PeerBlocks::in_memory();
        let (by_operator, by_plugin) = (peer(), peer());
        blocks
            .block(
                &by_operator,
                BlockLength::UntilUndone,
                Requester::Operator,
                None,
                0,
            )
            .unwrap();
        blocks
            .block(&by_plugin, BlockLength::UntilUndone, plugin("a"), None, 0)
            .unwrap();

        for (target, intruder, owner) in [
            (by_operator, plugin("a"), Requester::Operator),
            (by_plugin, plugin("b"), plugin("a")),
        ] {
            assert!(matches!(
                blocks.block(&target, BlockLength::SevenDays, intruder.clone(), None, 1),
                Err(BlockError::NotRequester)
            ));
            assert!(matches!(
                blocks.unblock(&target, intruder, None, 2),
                Err(BlockError::NotRequester)
            ));
            let block = &blocks.snapshot(3)[&peer_key(&target)];
            assert_eq!(block.requested_by, owner);
            assert_eq!(block.until_ms, None, "still until undone, not shortened");
        }
    }

    #[test]
    fn the_requester_can_renew_and_the_operator_can_take_over() {
        let blocks = PeerBlocks::in_memory();
        let bad = peer();
        blocks
            .block(&bad, BlockLength::SevenDays, plugin("a"), None, 0)
            .unwrap();
        let renewed = blocks
            .block(&bad, BlockLength::SevenDays, plugin("a"), None, 10)
            .unwrap();
        assert_eq!(renewed.until_ms, Some(10 + TIMED_BLOCK_MS));

        blocks
            .block(
                &bad,
                BlockLength::UntilUndone,
                Requester::Operator,
                None,
                20,
            )
            .unwrap();
        assert_eq!(
            blocks.snapshot(21)[&peer_key(&bad)].requested_by,
            Requester::Operator
        );
        assert!(matches!(
            blocks.unblock(&bad, plugin("a"), None, 22),
            Err(BlockError::NotRequester)
        ));
    }

    #[test]
    fn an_oversized_reason_is_refused_and_changes_nothing() {
        let blocks = PeerBlocks::in_memory();
        let bad = peer();
        let reason = serde_json::Value::String("x".repeat(MAX_REASON_BYTES));
        assert!(matches!(
            blocks.block(&bad, BlockLength::SevenDays, plugin("a"), Some(reason), 0),
            Err(BlockError::ReasonTooLarge)
        ));
        assert!(!blocks.is_blocked(&bad, 1));
    }

    #[test]
    fn routing_snapshot_drops_blocked_remote_targets_only() {
        let blocks = PeerBlocks::in_memory();
        let (bad, good) = (peer(), peer());
        blocks
            .block(&bad, BlockLength::UntilUndone, Requester::Operator, None, 0)
            .unwrap();
        let mut targets = ModelTargets::default();
        targets.targets.insert(
            "qwen".into(),
            vec![
                InferenceTarget::Remote(bad),
                InferenceTarget::Local(9337),
                InferenceTarget::Remote(good),
            ],
        );
        let filtered = blocks.without_blocked(&targets, 1);
        assert_eq!(
            filtered.candidates("qwen"),
            vec![InferenceTarget::Local(9337), InferenceTarget::Remote(good)]
        );
        let mut hosts = vec![bad, good];
        blocks.retain_unblocked(&mut hosts, 1);
        assert_eq!(hosts, vec![good]);
    }

    #[test]
    fn survives_reload_with_requester_and_reason() {
        let directory = tempfile::tempdir().unwrap();
        let bad = peer();
        let reason = serde_json::json!(["kept", 1]);
        PeerBlocks::load(directory.path())
            .block(
                &bad,
                BlockLength::UntilUndone,
                plugin("a"),
                Some(reason.clone()),
                3,
            )
            .unwrap();

        let reloaded = PeerBlocks::load(directory.path());
        assert!(reloaded.is_blocked(&bad, 4));
        let block = &reloaded.snapshot(4)[&peer_key(&bad)];
        assert_eq!(block.requested_by, plugin("a"));
        assert_eq!(block.reason.as_ref(), Some(&reason));
    }

    #[test]
    fn a_failed_save_changes_nothing_in_force() {
        let directory = tempfile::tempdir().unwrap();
        let state = directory.path().join("state");
        std::fs::create_dir(&state).unwrap();
        let blocks = PeerBlocks::load(&state);
        assert_eq!(blocks.not_saved(), None);
        // Then a file where the directory was: every save fails.
        std::fs::remove_dir(&state).unwrap();
        std::fs::write(&state, b"").unwrap();
        let bad = peer();
        assert!(
            blocks
                .block(&bad, BlockLength::UntilUndone, Requester::Operator, None, 0)
                .is_err()
        );
        assert!(
            !blocks.is_blocked(&bad, 1),
            "an unsaved block is never enforced"
        );
        assert!(blocks.snapshot(1).is_empty(), "and never listed");
    }

    #[test]
    fn an_unreadable_file_is_set_aside_not_overwritten() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join(FILE_NAME), b"not json").unwrap();
        let blocks = PeerBlocks::load(directory.path());
        assert!(blocks.snapshot(0).is_empty());
        let aside: Vec<_> = std::fs::read_dir(directory.path())
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_name().to_string_lossy().contains(".corrupt-"))
            .collect();
        assert_eq!(aside.len(), 1, "the unreadable file is kept");
        assert_eq!(std::fs::read(aside[0].path()).unwrap(), b"not json");
    }

    #[test]
    fn a_file_that_cannot_be_read_is_left_alone_and_nothing_is_saved() {
        let directory = tempfile::tempdir().unwrap();
        // A directory where the file should be: reading it fails, and not
        // with NotFound.
        let path = directory.path().join(FILE_NAME);
        std::fs::create_dir(&path).unwrap();
        std::fs::write(path.join("keep"), b"kept").unwrap();

        let blocks = PeerBlocks::load(directory.path());
        assert!(blocks.snapshot(0).is_empty());
        assert!(
            blocks
                .not_saved()
                .is_some_and(|why| why.contains("could not be read")),
            "{:?}",
            blocks.not_saved()
        );

        // Routing carries on: a change is in force for this run, but never
        // written over what is on disk.
        let bad = peer();
        blocks
            .block(&bad, BlockLength::UntilUndone, Requester::Operator, None, 1)
            .unwrap();
        assert!(blocks.is_blocked(&bad, 2));
        assert!(path.is_dir());
        assert_eq!(std::fs::read(path.join("keep")).unwrap(), b"kept");
        assert!(!path.with_extension("json.tmp").exists());
    }

    #[test]
    fn a_missing_file_is_an_empty_store_that_saves() {
        let directory = tempfile::tempdir().unwrap();
        let blocks = PeerBlocks::load(directory.path());
        assert!(blocks.snapshot(0).is_empty());
        assert_eq!(blocks.not_saved(), None);
        blocks
            .block(
                &peer(),
                BlockLength::SevenDays,
                Requester::Operator,
                None,
                0,
            )
            .unwrap();
        assert!(directory.path().join(FILE_NAME).is_file());
        assert!(
            !directory
                .path()
                .join(FILE_NAME)
                .with_extension("json.tmp")
                .exists()
        );
    }

    #[test]
    fn a_bad_file_that_cannot_be_set_aside_is_never_overwritten() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join(FILE_NAME);
        std::fs::write(&path, b"not json").unwrap();
        // A non-empty directory already at the set-aside name: the rename fails.
        let aside = path.with_extension("json.corrupt-7");
        std::fs::create_dir(&aside).unwrap();
        std::fs::write(aside.join("occupied"), b"").unwrap();

        let blocks = PeerBlocks::load_at(directory.path(), 7);
        assert!(blocks.snapshot(0).is_empty());
        assert!(
            blocks
                .not_saved()
                .is_some_and(|why| why.contains("could not be moved")),
            "{:?}",
            blocks.not_saved()
        );

        let bad = peer();
        blocks
            .block(&bad, BlockLength::UntilUndone, Requester::Operator, None, 8)
            .unwrap();
        assert!(blocks.is_blocked(&bad, 9));
        assert_eq!(
            std::fs::read(&path).unwrap(),
            b"not json",
            "the original file is still there, unchanged"
        );
    }

    fn config_with_plugin(
        name: &str,
        allow_peer_blocks: Option<bool>,
    ) -> crate::plugin::MeshConfig {
        let mut config = crate::plugin::MeshConfig::default();
        config.plugins.push(crate::plugin::PluginConfigEntry {
            name: name.into(),
            enabled: None,
            web_ui_enabled: None,
            web_ui_primary_tab: None,
            allow_peer_blocks,
            openai_exchange_grant: None,
            command: None,
            args: Vec::new(),
            url: None,
            settings: BTreeMap::new(),
            startup: crate::plugin::PluginStartupConfig::default(),
        });
        config
    }

    #[test]
    fn only_a_plugin_the_operator_opted_in_may_request_blocks() {
        assert!(!plugin_may_request(
            &crate::plugin::MeshConfig::default(),
            "a"
        ));
        assert!(!plugin_may_request(&config_with_plugin("a", None), "a"));
        assert!(!plugin_may_request(
            &config_with_plugin("a", Some(false)),
            "a"
        ));
        assert!(plugin_may_request(
            &config_with_plugin("a", Some(true)),
            "a"
        ));
        assert!(!plugin_may_request(
            &config_with_plugin("a", Some(true)),
            "b"
        ));
    }

    #[test]
    fn a_plugin_request_without_opt_in_changes_nothing() {
        use crate::plugin::proto::{
            PeerBlockRequest,
            peer_block_request::{Change, Length},
        };
        let blocks = PeerBlocks::in_memory();
        let bad = peer();
        let mut request = PeerBlockRequest {
            peer_id: peer_key(&bad),
            ..Default::default()
        };
        request.set_change(Change::Block);
        request.set_length(Length::UntilUndone);

        let refused =
            plugin_request_choice(&blocks, "a".into(), false, request.clone(), 0).unwrap_err();
        assert!(
            refused.message.contains("allow_peer_blocks"),
            "{}",
            refused.message
        );
        assert!(!blocks.is_blocked(&bad, 1));

        plugin_request_choice(&blocks, "a".into(), true, request, 2).unwrap();
        assert!(blocks.is_blocked(&bad, 3));
        // Operator blocks never need it.
        blocks
            .block(
                &peer(),
                BlockLength::UntilUndone,
                Requester::Operator,
                None,
                4,
            )
            .unwrap();
    }

    #[test]
    fn a_plugin_request_is_applied_as_that_plugin() {
        use crate::plugin::proto::{
            PeerBlockRequest,
            peer_block_request::{Change, Length},
        };
        let blocks = PeerBlocks::in_memory();
        let bad = peer();
        let mut request = PeerBlockRequest {
            peer_id: peer_key(&bad),
            reason_json: Some(r#"{"n":3}"#.into()),
            ..Default::default()
        };
        request.set_change(Change::Block);
        request.set_length(Length::SevenDays);

        let choice = plugin_request_choice(&blocks, "a".into(), true, request.clone(), 0).unwrap();
        assert_eq!(choice.requested_by, plugin("a"));
        assert_eq!(choice.reason, Some(serde_json::json!({ "n": 3 })));
        assert_eq!(choice.until_ms, Some(TIMED_BLOCK_MS));
        assert!(blocks.is_blocked(&bad, 1));

        request.set_change(Change::Unblock);
        let refused =
            plugin_request_choice(&blocks, "b".into(), true, request.clone(), 1).unwrap_err();
        assert!(
            refused.message.contains("only the operator"),
            "{}",
            refused.message
        );
        plugin_request_choice(&blocks, "a".into(), true, request, 2).unwrap();
        assert!(!blocks.is_blocked(&bad, 3));
    }

    #[test]
    fn a_malformed_plugin_request_changes_nothing() {
        use crate::plugin::proto::{
            PeerBlockRequest,
            peer_block_request::{Change, Length},
        };
        let blocks = PeerBlocks::in_memory();
        let bad = peer();
        let mut no_length = PeerBlockRequest {
            peer_id: peer_key(&bad),
            ..Default::default()
        };
        no_length.set_change(Change::Block);
        let mut bad_reason = no_length.clone();
        bad_reason.set_length(Length::UntilUndone);
        bad_reason.reason_json = Some("not json".into());
        let mut bad_peer = bad_reason.clone();
        bad_peer.reason_json = None;
        bad_peer.peer_id = "zz".into();

        for request in [PeerBlockRequest::default(), no_length, bad_reason, bad_peer] {
            assert!(plugin_request_choice(&blocks, "a".into(), true, request, 0).is_err());
        }
        assert!(blocks.snapshot(1).is_empty());
    }

    #[test]
    fn requester_round_trips_as_text() {
        for requester in [Requester::Operator, plugin("notes")] {
            let json = serde_json::to_string(&requester).unwrap();
            assert_eq!(serde_json::from_str::<Requester>(&json).unwrap(), requester);
        }
        assert_eq!(serde_json::to_string(&plugin("x")).unwrap(), "\"plugin:x\"");
        assert!(serde_json::from_str::<Requester>("\"plugin:\"").is_err());
        assert!(serde_json::from_str::<Requester>("\"someone\"").is_err());
    }
}
