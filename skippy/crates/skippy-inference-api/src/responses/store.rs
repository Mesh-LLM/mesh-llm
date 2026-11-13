//! Bounded, caller-scoped memory of completed responses for
//! `previous_response_id` continuation.
//!
//! Each stored response keeps the structured input items it ran with and the
//! output items it produced. A child links to its parent instead of copying
//! the parent's history, so a branch shares memory with the chain it extends;
//! for admission and eviction an entry is nonetheless charged for its whole
//! transcript, which keeps the limits conservative. The store is process
//! memory only: a restart forgets every response.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use serde_json::{Map, Value};

use crate::errors::InferenceError;

/// The identity whose responses a lookup may see. It comes from the
/// transport, never from request headers or body fields.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CallerScope {
    /// Direct (non-proxied) clients on the loopback interface. This is a
    /// host-local trust domain, not an authenticated user: every local
    /// process, and anything a reverse proxy relays, shares it. Ingress must
    /// not grant it to requests that carry forwarding headers, and
    /// deployments that serve several users through a proxy should turn
    /// response storage off.
    LocalHost,
    /// An authenticated remote endpoint, by its endpoint id.
    RemoteEndpoint(String),
}

/// Request headers that mean a proxy relayed the request. Their presence
/// alone, whatever their value, denies the host-local scope.
pub const FORWARDING_HEADERS: [&str; 3] = ["forwarded", "x-forwarded-for", "x-real-ip"];

/// The scope of a request that arrived over plain HTTP from `peer`. Only a
/// direct loopback client gets one; a proxied, remote, or unknown peer has
/// none, so its continuation fails closed.
pub fn caller_scope_for_http_peer(
    peer: Option<std::net::IpAddr>,
    forwarded: bool,
) -> Option<CallerScope> {
    let peer = peer?;
    let loopback = match peer {
        std::net::IpAddr::V6(v6) => v6
            .to_ipv4_mapped()
            .map_or(v6.is_loopback(), |v4| v4.is_loopback()),
        std::net::IpAddr::V4(v4) => v4.is_loopback(),
    };
    (loopback && !forwarded).then_some(CallerScope::LocalHost)
}

/// Environment switch for response storage, read once at startup.
pub const RESPONSE_STORAGE_ENV: &str = "MESH_RESPONSE_STORAGE";

/// Whether response storage is on, from [`RESPONSE_STORAGE_ENV`]. It is on
/// unless set to `off`, `false`, `0` or `disabled`; an empty value counts as
/// unset, and any other value turns storage off so a typo fails closed.
pub fn configured_response_storage() -> bool {
    match std::env::var(RESPONSE_STORAGE_ENV) {
        Err(std::env::VarError::NotPresent) => true,
        Err(std::env::VarError::NotUnicode(_)) => false,
        Ok(value) => parse_response_storage(&value).unwrap_or_else(|| {
            tracing::warn!(
                env = RESPONSE_STORAGE_ENV,
                value = %value,
                "unrecognized response storage setting; storage is off"
            );
            false
        }),
    }
}

fn parse_response_storage(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "" | "on" | "true" | "1" | "enabled" => Some(true),
        "off" | "false" | "0" | "disabled" => Some(false),
        _ => None,
    }
}

/// Store limits. The defaults suit a single node serving local agents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResponseStoreLimits {
    /// An entry expires this long after it was stored or last continued.
    pub ttl: Duration,
    pub max_responses_per_caller: usize,
    pub max_bytes_per_caller: usize,
    pub max_callers: usize,
    pub max_bytes: usize,
}

impl Default for ResponseStoreLimits {
    fn default() -> Self {
        Self {
            ttl: Duration::from_secs(30 * 60),
            max_responses_per_caller: 128,
            max_bytes_per_caller: 64 * 1024 * 1024,
            max_callers: 256,
            max_bytes: 256 * 1024 * 1024,
        }
    }
}

/// One completed response, as continuation needs it.
#[derive(Debug, PartialEq, Eq)]
pub struct StoredResponse {
    pub id: String,
    /// The model the client asked for, inherited when a continuation omits it.
    pub model: String,
    pub parent: Option<Arc<StoredResponse>>,
    /// The instructions this response ran with. Kept as metadata only: a
    /// continuation applies its own instructions, because the specified
    /// context is `previous.input + previous.output + input`.
    pub instructions: Option<String>,
    /// Effective model-generation configuration inherited by a continuation.
    /// Per-response controls such as `store`, `stream`, metadata and
    /// instructions are deliberately excluded.
    pub configuration: Map<String, Value>,
    /// Input items this response added after its parent's history.
    pub input: Vec<Value>,
    pub output: Vec<Value>,
    /// Serialized size of the whole transcript, parents included.
    pub transcript_bytes: usize,
}

impl StoredResponse {
    pub fn new(
        id: String,
        model: String,
        parent: Option<Arc<StoredResponse>>,
        instructions: Option<String>,
        configuration: Map<String, Value>,
        input: Vec<Value>,
        output: Vec<Value>,
    ) -> Self {
        let own_bytes = input
            .iter()
            .chain(output.iter())
            .map(|item| {
                serde_json::to_vec(item)
                    .map(|bytes| bytes.len())
                    .unwrap_or(0)
            })
            .sum::<usize>()
            .saturating_add(id.len() + model.len())
            .saturating_add(instructions.as_ref().map_or(0, String::len))
            .saturating_add(serde_json::to_vec(&configuration).map_or(0, |bytes| bytes.len()));
        let transcript_bytes = parent
            .as_ref()
            .map_or(0, |parent| parent.transcript_bytes)
            .saturating_add(own_bytes);
        Self {
            id,
            model,
            parent,
            instructions,
            configuration,
            input,
            output,
            transcript_bytes,
        }
    }

    /// The conversation this response ended: every ancestor's input and
    /// output, then its own, in order.
    pub fn transcript(&self) -> Vec<Value> {
        let mut chain = Vec::new();
        let mut cursor = Some(self);
        while let Some(response) = cursor {
            chain.push(response);
            cursor = response.parent.as_deref();
        }
        chain
            .into_iter()
            .rev()
            .flat_map(|response| response.input.iter().chain(response.output.iter()))
            .cloned()
            .collect()
    }

    /// The first response of the chain.
    pub fn root_id(&self) -> &str {
        let mut cursor = self;
        while let Some(parent) = cursor.parent.as_deref() {
            cursor = parent;
        }
        &cursor.id
    }
}

/// The one error every failed lookup returns, so a caller cannot tell an
/// unknown id from another caller's, an expired, or an evicted one.
pub fn previous_response_not_found(id: &str) -> InferenceError {
    InferenceError::from_kind(
        axum::http::StatusCode::NOT_FOUND,
        crate::errors::InferenceErrorKind::NotFound,
        format!("previous response '{id}' was not found"),
    )
    .with_code("previous_response_not_found")
    .with_param("previous_response_id")
}

#[derive(Debug)]
struct Entry {
    response: Arc<StoredResponse>,
    last_used: Instant,
    tick: u64,
}

#[derive(Debug, Default)]
struct Namespace {
    entries: HashMap<String, Entry>,
    bytes: usize,
    tick: u64,
}

#[derive(Debug, Default)]
struct StoreState {
    callers: HashMap<CallerScope, Namespace>,
    bytes: usize,
    tick: u64,
}

/// The response store shared by every request on one ingress.
#[derive(Debug, Default)]
pub struct ResponseStore {
    limits: ResponseStoreLimits,
    state: Mutex<StoreState>,
}

impl ResponseStore {
    pub fn new(limits: ResponseStoreLimits) -> Self {
        Self {
            limits,
            state: Mutex::default(),
        }
    }

    pub fn limits(&self) -> ResponseStoreLimits {
        self.limits
    }

    /// Look up a response this caller stored, refreshing its expiry.
    pub fn get(
        &self,
        scope: &CallerScope,
        id: &str,
    ) -> Result<Arc<StoredResponse>, InferenceError> {
        self.get_at(scope, id, Instant::now())
    }

    fn get_at(
        &self,
        scope: &CallerScope,
        id: &str,
        now: Instant,
    ) -> Result<Arc<StoredResponse>, InferenceError> {
        let mut state = self.lock();
        state.tick += 1;
        let tick = state.tick;
        let ttl = self.limits.ttl;
        let mut removed = 0;
        let found = state.callers.get_mut(scope).and_then(|namespace| {
            removed = namespace.prune_expired(now, ttl);
            let entry = namespace.entries.get_mut(id)?;
            entry.last_used = now;
            entry.tick = tick;
            namespace.tick = tick;
            Some(entry.response.clone())
        });
        state.bytes = state.bytes.saturating_sub(removed);
        found.ok_or_else(|| previous_response_not_found(id))
    }

    /// Store a completed response for this caller. A response whose
    /// transcript exceeds the per-caller budget is not kept, and storing
    /// never evicts more than it needs to make room.
    pub fn insert(&self, scope: CallerScope, response: StoredResponse) -> bool {
        self.insert_at(scope, response, Instant::now())
    }

    fn insert_at(&self, scope: CallerScope, response: StoredResponse, now: Instant) -> bool {
        let size = response.transcript_bytes;
        if self.limits.max_responses_per_caller == 0
            || self.limits.max_callers == 0
            || self.limits.max_bytes_per_caller == 0
            || self.limits.max_bytes == 0
            || size > self.limits.max_bytes_per_caller
            || size > self.limits.max_bytes
        {
            return false;
        }
        let mut state = self.lock();
        state.prune_expired(now, self.limits.ttl);
        state.tick += 1;
        let tick = state.tick;
        if !state.callers.contains_key(&scope) {
            while state.callers.len() >= self.limits.max_callers {
                state.evict_least_recent_caller();
            }
        }
        let limits = self.limits;
        let namespace = state.callers.entry(scope.clone()).or_default();
        let mut freed = 0;
        if let Some(previous) = namespace.entries.remove(&response.id) {
            namespace.bytes -= previous.response.transcript_bytes;
            freed += previous.response.transcript_bytes;
        }
        while namespace.entries.len() >= limits.max_responses_per_caller
            || namespace.bytes + size > limits.max_bytes_per_caller
        {
            freed += namespace.evict_least_recent();
        }
        namespace.tick = tick;
        namespace.bytes += size;
        namespace.entries.insert(
            response.id.clone(),
            Entry {
                response: Arc::new(response),
                last_used: now,
                tick,
            },
        );
        state.bytes = state.bytes.saturating_sub(freed) + size;
        while state.bytes > limits.max_bytes {
            if !state.evict_least_recent_entry(&scope) {
                break;
            }
        }
        true
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, StoreState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    #[cfg(test)]
    fn bytes(&self) -> usize {
        self.lock().bytes
    }
}

impl Namespace {
    fn prune_expired(&mut self, now: Instant, ttl: Duration) -> usize {
        let before = self.bytes;
        self.entries.retain(|_, entry| {
            let keep = now.saturating_duration_since(entry.last_used) < ttl;
            if !keep {
                self.bytes -= entry.response.transcript_bytes;
            }
            keep
        });
        before - self.bytes
    }

    fn evict_least_recent(&mut self) -> usize {
        let Some(id) = self
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.tick)
            .map(|(id, _)| id.clone())
        else {
            return 0;
        };
        let entry = self.entries.remove(&id).expect("entry selected above");
        self.bytes -= entry.response.transcript_bytes;
        entry.response.transcript_bytes
    }
}

impl StoreState {
    fn prune_expired(&mut self, now: Instant, ttl: Duration) {
        let mut freed = 0;
        for namespace in self.callers.values_mut() {
            freed += namespace.prune_expired(now, ttl);
        }
        self.callers
            .retain(|_, namespace| !namespace.entries.is_empty());
        self.bytes = self.bytes.saturating_sub(freed);
    }

    fn evict_least_recent_caller(&mut self) {
        let Some(scope) = self
            .callers
            .iter()
            .min_by_key(|(_, namespace)| namespace.tick)
            .map(|(scope, _)| scope.clone())
        else {
            return;
        };
        if let Some(namespace) = self.callers.remove(&scope) {
            self.bytes = self.bytes.saturating_sub(namespace.bytes);
        }
    }

    /// Evict the globally least recently used entry, preferring other
    /// callers' entries over the newest one of `keep`. Returns whether
    /// anything was evicted.
    fn evict_least_recent_entry(&mut self, keep: &CallerScope) -> bool {
        let newest_kept = self.callers.get(keep).map(|namespace| namespace.tick);
        let candidate = self
            .callers
            .iter()
            .flat_map(|(scope, namespace)| {
                namespace
                    .entries
                    .iter()
                    .map(move |(id, entry)| (scope, id, entry.tick))
            })
            .filter(|(scope, _, tick)| *scope != keep || Some(*tick) != newest_kept)
            .min_by_key(|(_, _, tick)| *tick)
            .map(|(scope, id, _)| (scope.clone(), id.clone()));
        let Some((scope, id)) = candidate else {
            return false;
        };
        if let Some(namespace) = self.callers.get_mut(&scope)
            && let Some(entry) = namespace.entries.remove(&id)
        {
            namespace.bytes -= entry.response.transcript_bytes;
            self.bytes = self.bytes.saturating_sub(entry.response.transcript_bytes);
            if namespace.entries.is_empty() {
                self.callers.remove(&scope);
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn response(id: &str, parent: Option<Arc<StoredResponse>>, text: &str) -> StoredResponse {
        StoredResponse::new(
            id.to_string(),
            "m".to_string(),
            parent,
            None,
            Map::new(),
            vec![json!({"type": "message", "role": "user", "content": text})],
            vec![
                json!({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}),
            ],
        )
    }

    #[test]
    fn only_direct_loopback_peers_get_the_local_scope() {
        let v4: std::net::IpAddr = "127.0.0.1".parse().unwrap();
        let v6: std::net::IpAddr = "::1".parse().unwrap();
        let mapped: std::net::IpAddr = "::ffff:127.0.0.1".parse().unwrap();
        let lan: std::net::IpAddr = "192.168.1.4".parse().unwrap();
        for peer in [v4, v6, mapped] {
            assert_eq!(
                caller_scope_for_http_peer(Some(peer), false),
                Some(CallerScope::LocalHost)
            );
            assert_eq!(caller_scope_for_http_peer(Some(peer), true), None);
        }
        assert_eq!(caller_scope_for_http_peer(Some(lan), false), None);
        assert_eq!(caller_scope_for_http_peer(None, false), None);
    }

    #[test]
    fn storage_setting_parses_and_fails_closed() {
        for on in ["", "on", "TRUE", "1"] {
            assert_eq!(parse_response_storage(on), Some(true));
        }
        for off in ["off", "false", "0", "disabled"] {
            assert_eq!(parse_response_storage(off), Some(false));
        }
        assert_eq!(parse_response_storage("maybe"), None);
    }

    fn remote(name: &str) -> CallerScope {
        CallerScope::RemoteEndpoint(name.to_string())
    }

    #[test]
    fn transcript_walks_the_chain_in_order() {
        let store = ResponseStore::default();
        store.insert(CallerScope::LocalHost, response("resp_a", None, "one"));
        let parent = store.get(&CallerScope::LocalHost, "resp_a").unwrap();
        store.insert(
            CallerScope::LocalHost,
            response("resp_b", Some(parent), "two"),
        );
        let child = store.get(&CallerScope::LocalHost, "resp_b").unwrap();
        let texts = child
            .transcript()
            .iter()
            .map(|item| item.to_string())
            .collect::<Vec<_>>();
        assert_eq!(texts.len(), 4);
        assert!(texts[0].contains("one") && texts[0].contains("user"));
        assert!(texts[3].contains("two") && texts[3].contains("assistant"));
        assert_eq!(child.root_id(), "resp_a");
    }

    #[test]
    fn callers_cannot_see_each_other_and_misses_are_indistinguishable() {
        let store = ResponseStore::default();
        store.insert(remote("a"), response("resp_a", None, "x"));
        let foreign = store.get(&remote("b"), "resp_a").unwrap_err();
        let local = store.get(&CallerScope::LocalHost, "resp_a").unwrap_err();
        let unknown = store.get(&remote("a"), "resp_zz").unwrap_err();
        for error in [&foreign, &local, &unknown] {
            assert_eq!(error.status(), axum::http::StatusCode::NOT_FOUND);
            assert_eq!(
                error.body().error.code.as_deref(),
                Some("previous_response_not_found")
            );
        }
        assert!(store.get(&remote("a"), "resp_a").is_ok());
    }

    #[test]
    fn entries_expire_after_the_sliding_ttl() {
        let store = ResponseStore::new(ResponseStoreLimits {
            ttl: Duration::from_secs(10),
            ..Default::default()
        });
        let start = Instant::now();
        store.insert_at(CallerScope::LocalHost, response("resp_a", None, "x"), start);
        // A lookup inside the window slides it forward.
        assert!(
            store
                .get_at(
                    &CallerScope::LocalHost,
                    "resp_a",
                    start + Duration::from_secs(8)
                )
                .is_ok()
        );
        assert!(
            store
                .get_at(
                    &CallerScope::LocalHost,
                    "resp_a",
                    start + Duration::from_secs(16)
                )
                .is_ok()
        );
        assert!(
            store
                .get_at(
                    &CallerScope::LocalHost,
                    "resp_a",
                    start + Duration::from_secs(27)
                )
                .is_err()
        );
        assert_eq!(store.bytes(), 0);
    }

    #[test]
    fn per_caller_count_evicts_least_recently_used() {
        let store = ResponseStore::new(ResponseStoreLimits {
            max_responses_per_caller: 2,
            ..Default::default()
        });
        let scope = CallerScope::LocalHost;
        store.insert(scope.clone(), response("resp_a", None, "a"));
        store.insert(scope.clone(), response("resp_b", None, "b"));
        store.get(&scope, "resp_a").unwrap();
        store.insert(scope.clone(), response("resp_c", None, "c"));
        assert!(store.get(&scope, "resp_a").is_ok());
        assert!(store.get(&scope, "resp_b").is_err());
        assert!(store.get(&scope, "resp_c").is_ok());
    }

    #[test]
    fn byte_budgets_hold_and_oversized_transcripts_are_not_kept() {
        let one = response("resp_probe", None, "aaaa").transcript_bytes;
        let store = ResponseStore::new(ResponseStoreLimits {
            max_bytes_per_caller: one * 2,
            max_bytes: one * 3,
            ..Default::default()
        });
        for (index, caller) in ["a", "a", "a", "b", "b", "c"].iter().enumerate() {
            assert!(store.insert(
                remote(caller),
                response(&format!("resp_{index}"), None, "aaaa")
            ));
            assert!(store.bytes() <= one * 3);
        }
        // Caller a kept at most two of its three.
        assert!(store.get(&remote("a"), "resp_0").is_err());
        // The newest entry always survives its own insertion.
        assert!(store.get(&remote("c"), "resp_5").is_ok());
        let huge = response("resp_huge", None, &"x".repeat(one * 3));
        assert!(!store.insert(remote("d"), huge));
        assert!(store.get(&remote("c"), "resp_5").is_ok());
    }

    #[test]
    fn caller_count_evicts_the_least_recent_caller() {
        let store = ResponseStore::new(ResponseStoreLimits {
            max_callers: 2,
            ..Default::default()
        });
        store.insert(remote("a"), response("resp_a", None, "a"));
        store.insert(remote("b"), response("resp_b", None, "b"));
        store.get(&remote("a"), "resp_a").unwrap();
        store.insert(remote("c"), response("resp_c", None, "c"));
        assert!(store.get(&remote("a"), "resp_a").is_ok());
        assert!(store.get(&remote("b"), "resp_b").is_err());
    }

    #[test]
    fn evicted_parents_stay_reachable_through_their_children() {
        let store = ResponseStore::new(ResponseStoreLimits {
            max_responses_per_caller: 1,
            ..Default::default()
        });
        let scope = CallerScope::LocalHost;
        store.insert(scope.clone(), response("resp_a", None, "a"));
        let parent = store.get(&scope, "resp_a").unwrap();
        store.insert(scope.clone(), response("resp_b", Some(parent), "b"));
        assert!(store.get(&scope, "resp_a").is_err());
        assert_eq!(store.get(&scope, "resp_b").unwrap().transcript().len(), 4);
    }

    #[test]
    fn zero_capacity_limits_reject_without_looping() {
        for limits in [
            ResponseStoreLimits {
                max_responses_per_caller: 0,
                ..Default::default()
            },
            ResponseStoreLimits {
                max_callers: 0,
                ..Default::default()
            },
            ResponseStoreLimits {
                max_bytes_per_caller: 0,
                ..Default::default()
            },
            ResponseStoreLimits {
                max_bytes: 0,
                ..Default::default()
            },
        ] {
            assert!(
                !ResponseStore::new(limits)
                    .insert(CallerScope::LocalHost, response("resp_a", None, "x"),)
            );
        }
    }
}
