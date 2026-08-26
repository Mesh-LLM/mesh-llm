//! Request-key extraction and sticky-routing primitives.
//!
//! Host and client runtimes intentionally keep their policy-specific wrappers
//! in their existing `network::affinity` modules.  This module owns only the
//! deterministic fallback ordering that both consumers share. Cache-aware
//! target selection consumes explicit evidence from [`crate::cache_aware`]; it
//! never learns a target merely because a request succeeded.

use crate::{InferenceTarget, ModelTargets};
use iroh::EndpointId;
use serde_json::Value;
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AffinityStats {
    pub entries: usize,
    pub lookups: u64,
    pub hits: u64,
    pub misses: u64,
    pub stale: u64,
    pub routes: u64,
    pub sticky_routes: u64,
    pub session_routes: u64,
    pub learned: u64,
    pub evicted: u64,
}

#[doc(hidden)]
#[macro_export]
macro_rules! impl_affinity_stats_snapshot {
    ($snapshot:ty $({ $($field:ident: $value:expr),+ $(,)? })?) => {
        impl $snapshot {
            fn from_affinity_stats(
                stats: $crate::affinity::AffinityStats,
                cache_enabled: bool,
                sticky_enabled: bool,
            ) -> Self {
                Self {
                    prefix_enabled: cache_enabled,
                    sticky_enabled,
                    prefix_entries: stats.entries,
                    prefix_lookups: stats.lookups,
                    prefix_hits: stats.hits,
                    prefix_misses: stats.misses,
                    prefix_stale: stats.stale,
                    prefix_routes: stats.routes,
                    sticky_routes: stats.sticky_routes,
                    session_routes: stats.session_routes,
                    learned: stats.learned,
                    evicted: stats.evicted,
                    $($($field: $value),+)?
                }
            }
        }
    };
}

/// Request-derived hashes used by prefix and sticky routing.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RoutingKeys {
    /// Explicit cache/session hint hash, when one was supplied.
    pub session_hash: Option<u64>,
    /// Stable prompt/tool scaffold hash, when one was found.
    pub prefix_hash: Option<u64>,
    /// Hash used for explicit deterministic session routing.
    pub sticky_hash: Option<u64>,
}

/// Shared cache-affinity counters and sticky-routing configuration.
#[derive(Clone)]
pub struct AffinityRouter {
    inner: Arc<Mutex<AffinityStats>>,
    config: Arc<AffinityConfig>,
}

#[derive(Clone, Copy, Debug)]
struct AffinityConfig {
    prefix_enabled: bool,
    sticky_enabled: bool,
}

impl AffinityRouter {
    /// Create a shared affinity state using the process routing settings.
    pub fn new() -> Self {
        Self::with_config(
            std::env::var_os("MESH_LLM_DISABLE_PREFIX_AFFINITY").is_none(),
            std::env::var_os("MESH_LLM_DISABLE_STICKY_ROUTING").is_none(),
        )
    }

    /// Build affinity state with explicit feature flags for consumer tests.
    #[doc(hidden)]
    pub fn with_config(prefix_enabled: bool, sticky_enabled: bool) -> Self {
        Self {
            inner: Arc::new(Mutex::new(AffinityStats::default())),
            config: Arc::new(AffinityConfig {
                prefix_enabled,
                sticky_enabled,
            }),
        }
    }

    /// Return the current prefix-affinity counters and resident-entry count.
    pub fn stats_snapshot(&self) -> AffinityStats {
        self.inner.lock().unwrap().clone()
    }

    /// Whether prefix affinity is enabled for this state.
    pub fn prefix_enabled(&self) -> bool {
        self.config.prefix_enabled
    }

    /// Whether sticky/session routing is enabled for this state.
    pub fn sticky_enabled(&self) -> bool {
        self.config.sticky_enabled
    }

    /// Record a deterministic sticky route.
    pub fn record_sticky_route(&self) {
        self.inner.lock().unwrap().sticky_routes += 1;
    }

    /// Record a session-hint route.
    pub fn record_session_route(&self) {
        self.inner.lock().unwrap().session_routes += 1;
    }

    pub fn record_cache_probe(&self, hit: bool) {
        if !self.config.prefix_enabled {
            return;
        }
        let mut stats = self.inner.lock().unwrap();
        stats.lookups += 1;
        if hit {
            stats.hits += 1;
            stats.routes += 1;
        } else {
            stats.misses += 1;
        }
    }
}

impl Default for AffinityRouter {
    fn default() -> Self {
        Self::new()
    }
}

/// Selection result returned by shared model-target ordering.
pub struct TargetSelection {
    /// Target selected for this request.
    pub target: InferenceTarget,
    /// Request prefix key used to query cache evidence.
    pub prefix_hash: Option<u64>,
    /// Cache-evidence target used for this request, when one was available.
    pub cache_target: Option<InferenceTarget>,
}

/// Remote-target ordering and cache-evidence metadata.
pub struct PreparedTargets {
    /// Targets in request order.
    pub ordered: Vec<InferenceTarget>,
    /// Request prefix key used to query cache evidence.
    pub prefix_hash: Option<u64>,
    /// Cache-evidence target moved first, when one was available.
    pub cache_target: Option<InferenceTarget>,
}

/// Whether prefix-only routing has been requested by the process.
pub fn prefix_only_enabled() -> bool {
    std::env::var("MESH_LLM_PREFIX_ONLY")
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Extract a session/cache hint using the consumer's compatibility order.
pub fn extract_session_hint_from_body(body: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| body.get(*key).and_then(Value::as_str).map(str::to_string))
}

/// Compute request routing keys with consumer-selected compatibility policy.
pub fn routing_keys(
    parsed_body: Option<&Value>,
    session_hint_keys: &[&str],
    prefix_fallback_to_first_user: bool,
) -> RoutingKeys {
    let Some(body) = parsed_body else {
        return RoutingKeys::default();
    };

    let session_hash = extract_session_hint_from_body(body, session_hint_keys)
        .map(|hint| hash_bytes(hint.as_bytes()));
    let scaffold_hash = scaffold_prefix_hash_from_body(body, prefix_fallback_to_first_user);
    let first_user_hash = first_user_hash_from_body(body);
    // Namespace evidence by an explicit session hint when present. Otherwise
    // combine scaffold and first-user hashes so unrelated requests with the
    // same tools cannot claim one another's cache receipt.
    let prefix_hash = session_hash.or_else(|| {
        let mut hash = 0u64;
        let mut found = false;
        if let Some(scaffold_hash) = scaffold_hash {
            hash = hash_combine(hash, scaffold_hash);
            found = true;
        }
        if let Some(user_hash) = first_user_hash {
            hash = hash_combine(hash, user_hash);
            found = true;
        }
        found.then_some(hash)
    });
    // Cache locality must not become an implicit sticky policy. Without a
    // caller-provided session hint, stale or absent cache evidence falls back
    // to the normal load-aware target picker.
    let sticky_hash = session_hash;

    RoutingKeys {
        session_hash,
        prefix_hash,
        sticky_hash,
    }
}

/// Compute the stable scaffold hash used by prefix affinity.
pub fn scaffold_prefix_hash_from_body(
    body: &Value,
    prefix_fallback_to_first_user: bool,
) -> Option<u64> {
    let mut hash = 0u64;
    let mut found = false;

    for key in [
        "tools",
        "functions",
        "response_format",
        "tool_choice",
        "parallel_tool_calls",
    ] {
        if let Some(value) = body.get(key) {
            hash = hash_tagged_json(hash, key, value);
            found = true;
        }
    }

    if let Some(messages) = body.get("messages").and_then(Value::as_array) {
        for msg in messages {
            let role = msg.get("role").and_then(Value::as_str).unwrap_or("");
            match role {
                "system" | "developer" => {
                    if let Some(text) = message_text(msg) {
                        hash = hash_tagged_text(hash, role, &text);
                        found = true;
                    }
                }
                "user" => break,
                _ => {}
            }
        }
    }

    if prefix_fallback_to_first_user
        && !found
        && let Some(user_hash) = first_user_hash_from_body(body)
    {
        hash = hash_combine(hash, user_hash);
        found = true;
    }

    found.then_some(hash)
}

/// Compute the first user/prompt hash used as the sticky-routing fallback.
pub fn first_user_hash_from_body(body: &Value) -> Option<u64> {
    if let Some(messages) = body.get("messages").and_then(Value::as_array) {
        for msg in messages {
            if msg.get("role").and_then(Value::as_str) == Some("user") {
                return message_text(msg).map(|text| hash_tagged_text(0, "user", &text));
            }
        }
    }
    body.get("prompt")
        .and_then(Value::as_str)
        .map(|prompt| hash_tagged_text(0, "prompt", prompt))
}

/// Select a model target after the consumer has applied any local eligibility
/// policy (for example, host-side target health).
pub fn select_model_target_from_keys(
    targets: &ModelTargets,
    candidates: &[InferenceTarget],
    routing: &RoutingKeys,
    affinity: &AffinityRouter,
    cache_target: Option<InferenceTarget>,
) -> TargetSelection {
    if affinity.prefix_enabled()
        && let Some(target) = cache_target.filter(|target| candidates.contains(target))
    {
        affinity.record_cache_probe(true);
        return TargetSelection {
            target: target.clone(),
            prefix_hash: routing.prefix_hash,
            cache_target: Some(target),
        };
    }
    if routing.prefix_hash.is_some() {
        affinity.record_cache_probe(false);
    }
    if let Some(session_hash) = routing.session_hash.filter(|_| affinity.sticky_enabled()) {
        affinity.record_session_route();
        return TargetSelection {
            target: ModelTargets::pick_sticky_from(candidates, session_hash),
            prefix_hash: routing.prefix_hash,
            cache_target: None,
        };
    }

    if let Some(prefix_hash) = routing.prefix_hash {
        if prefix_only_enabled() {
            return TargetSelection {
                target: ModelTargets::pick_sticky_from(candidates, prefix_hash),
                prefix_hash: Some(prefix_hash),
                cache_target: None,
            };
        }

        if let Some(sticky_hash) = routing.sticky_hash.filter(|_| affinity.sticky_enabled()) {
            affinity.record_sticky_route();
            return TargetSelection {
                target: ModelTargets::pick_sticky_from(candidates, sticky_hash),
                prefix_hash: Some(prefix_hash),
                cache_target: None,
            };
        }

        return TargetSelection {
            target: targets.pick_from(candidates),
            prefix_hash: Some(prefix_hash),
            cache_target: None,
        };
    }

    if let Some(sticky_hash) = routing.sticky_hash.filter(|_| affinity.sticky_enabled()) {
        affinity.record_sticky_route();
        return TargetSelection {
            target: ModelTargets::pick_sticky_from(candidates, sticky_hash),
            prefix_hash: None,
            cache_target: None,
        };
    }

    TargetSelection {
        target: targets.pick_from(candidates),
        prefix_hash: None,
        cache_target: None,
    }
}

/// Prepare remote targets after the consumer has derived request keys.
pub fn prepare_remote_targets_from_keys(
    hosts: &[EndpointId],
    routing: &RoutingKeys,
    affinity: &AffinityRouter,
    cache_target: Option<InferenceTarget>,
) -> PreparedTargets {
    let mut ordered: Vec<InferenceTarget> =
        hosts.iter().copied().map(InferenceTarget::Remote).collect();
    let mut cache_target = cache_target.filter(|target| ordered.contains(target));

    if affinity.prefix_enabled()
        && let Some(target) = cache_target.as_ref()
    {
        affinity.record_cache_probe(true);
        move_target_first(&mut ordered, target);
        return PreparedTargets {
            ordered,
            prefix_hash: routing.prefix_hash,
            cache_target,
        };
    }
    if routing.prefix_hash.is_some() {
        affinity.record_cache_probe(false);
    }
    if let Some(session_hash) = routing.session_hash.filter(|_| affinity.sticky_enabled()) {
        affinity.record_session_route();
        rotate_targets_by_hash(&mut ordered, session_hash);
        return PreparedTargets {
            ordered,
            prefix_hash: routing.prefix_hash,
            cache_target: None,
        };
    }

    if let Some(prefix_hash) = routing.prefix_hash {
        if prefix_only_enabled() {
            rotate_targets_by_hash(&mut ordered, prefix_hash);
        } else if let Some(sticky_hash) = routing.sticky_hash.filter(|_| affinity.sticky_enabled())
        {
            affinity.record_sticky_route();
            rotate_targets_by_hash(&mut ordered, sticky_hash);
        }
    } else if let Some(sticky_hash) = routing.sticky_hash.filter(|_| affinity.sticky_enabled()) {
        affinity.record_sticky_route();
        rotate_targets_by_hash(&mut ordered, sticky_hash);
    }

    PreparedTargets {
        ordered,
        prefix_hash: routing.prefix_hash,
        cache_target: cache_target.take(),
    }
}

fn rotate_targets_by_hash(targets: &mut [InferenceTarget], key: u64) {
    if !targets.is_empty() {
        let idx = key as usize % targets.len();
        targets.rotate_left(idx);
    }
}

fn move_target_first(targets: &mut [InferenceTarget], target: &InferenceTarget) -> bool {
    if let Some(pos) = targets.iter().position(|candidate| candidate == target) {
        targets[..=pos].rotate_right(1);
        true
    } else {
        false
    }
}

fn message_text(msg: &Value) -> Option<String> {
    if let Some(s) = msg.get("content").and_then(Value::as_str) {
        return Some(s.to_string());
    }
    if let Some(blocks) = msg.get("content").and_then(Value::as_array) {
        let mut out = String::new();
        for block in blocks {
            if let Some(text) = block.get("text").and_then(Value::as_str) {
                out.push_str(text);
                out.push('\n');
            }
        }
        if !out.is_empty() {
            return Some(out);
        }
    }
    None
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |acc, &b| {
        (acc ^ b as u64).wrapping_mul(0x100000001b3)
    })
}

fn hash_combine(a: u64, b: u64) -> u64 {
    a.wrapping_mul(31).wrapping_add(b)
}

fn hash_tagged_text(mut acc: u64, tag: &str, text: &str) -> u64 {
    acc = hash_combine(acc, hash_bytes(tag.as_bytes()));
    hash_combine(acc, hash_bytes(text.as_bytes()))
}

fn hash_json_value(mut acc: u64, value: &Value) -> u64 {
    match value {
        Value::Null => hash_combine(acc, hash_bytes(b"null")),
        Value::Bool(boolean) => {
            acc = hash_combine(acc, hash_bytes(b"bool"));
            hash_combine(acc, hash_bytes(boolean.to_string().as_bytes()))
        }
        Value::Number(number) => {
            acc = hash_combine(acc, hash_bytes(b"number"));
            hash_combine(acc, hash_bytes(number.to_string().as_bytes()))
        }
        Value::String(text) => {
            acc = hash_combine(acc, hash_bytes(b"string"));
            hash_combine(acc, hash_bytes(text.as_bytes()))
        }
        Value::Array(items) => {
            acc = hash_combine(acc, hash_bytes(b"array"));
            acc = hash_combine(acc, items.len() as u64);
            for item in items {
                acc = hash_json_value(acc, item);
            }
            acc
        }
        Value::Object(map) => {
            acc = hash_combine(acc, hash_bytes(b"object"));
            let mut keys: Vec<_> = map.keys().collect();
            keys.sort_unstable();
            for key in keys {
                acc = hash_combine(acc, hash_bytes(key.as_bytes()));
                acc = hash_json_value(acc, &map[key]);
            }
            acc
        }
    }
}

fn hash_tagged_json(mut acc: u64, tag: &str, value: &Value) -> u64 {
    acc = hash_combine(acc, hash_bytes(tag.as_bytes()));
    hash_json_value(acc, value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use iroh::SecretKey;

    fn make_id(seed: u8) -> EndpointId {
        let mut bytes = [0u8; 32];
        bytes[0] = seed;
        SecretKey::from_bytes(&bytes).public()
    }

    fn remote(seed: u8) -> InferenceTarget {
        InferenceTarget::Remote(make_id(seed))
    }

    fn parse_body(body: &str) -> Value {
        serde_json::from_str(body).unwrap()
    }

    #[test]
    fn routing_key_policies_preserve_host_and_client_differences() {
        let body = parse_body(
            r#"{"prompt_cache_key":"cache","user":"user","messages":[{"role":"user","content":"hello"}]}"#,
        );
        let host = routing_keys(
            Some(&body),
            &["prompt_cache_key", "user", "session_id"],
            true,
        );
        let client = routing_keys(Some(&body), &["user", "session_id"], false);

        assert_ne!(host.session_hash, client.session_hash);
        assert_eq!(
            host.session_hash,
            Some(hash_bytes(b"cache")),
            "host keeps prompt_cache_key compatibility"
        );
        assert_eq!(client.session_hash, Some(hash_bytes(b"user")));

        let user_only = parse_body(r#"{"messages":[{"role":"user","content":"hello"}]}"#);
        assert!(
            routing_keys(
                Some(&user_only),
                &["prompt_cache_key", "user", "session_id"],
                true,
            )
            .prefix_hash
            .is_some()
        );
        assert!(
            routing_keys(Some(&user_only), &["user", "session_id"], false)
                .prefix_hash
                .is_some(),
            "the cache namespace includes the first user turn even when the raw scaffold is empty"
        );
    }

    #[test]
    fn shared_selection_uses_cached_prefix_target() {
        let mut targets = ModelTargets::default();
        targets
            .targets
            .insert("qwen".to_string(), vec![remote(1), remote(2)]);
        let body = parse_body(
            r#"{"tools":[{"type":"function","function":{"name":"run"}}],"messages":[{"role":"system","content":"agent"},{"role":"user","content":"task"}]}"#,
        );
        let keys = routing_keys(
            Some(&body),
            &["prompt_cache_key", "user", "session_id"],
            true,
        );
        let affinity = AffinityRouter::with_config(true, true);
        let candidates = targets.candidates("qwen");
        let cached = remote(2);
        let selection = select_model_target_from_keys(
            &targets,
            &candidates,
            &keys,
            &affinity,
            Some(cached.clone()),
        );

        assert_eq!(selection.target, cached);
        assert_eq!(selection.cache_target, Some(cached));
    }

    #[test]
    fn missing_cache_evidence_uses_normal_load_aware_fallback() {
        let first = remote(1);
        let second = remote(2);
        let mut targets = ModelTargets::default();
        targets
            .targets
            .insert("qwen".to_string(), vec![first.clone(), second.clone()]);
        let body = parse_body(
            r#"{"messages":[{"role":"system","content":"agent"},{"role":"user","content":"task"}]}"#,
        );
        let keys = routing_keys(
            Some(&body),
            &["prompt_cache_key", "user", "session_id"],
            true,
        );
        let affinity = AffinityRouter::with_config(true, true);
        let candidates = targets.candidates("qwen");

        assert!(keys.prefix_hash.is_some());
        assert_eq!(keys.sticky_hash, None);
        assert_eq!(
            select_model_target_from_keys(&targets, &candidates, &keys, &affinity, None).target,
            first
        );
        assert_eq!(
            select_model_target_from_keys(&targets, &candidates, &keys, &affinity, None).target,
            second
        );
    }

    #[test]
    fn explicit_session_hint_remains_sticky_without_cache_evidence() {
        let first = remote(1);
        let second = remote(2);
        let mut targets = ModelTargets::default();
        targets
            .targets
            .insert("qwen".to_string(), vec![first.clone(), second.clone()]);
        let body = parse_body(
            r#"{"prompt_cache_key":"session-a","messages":[{"role":"user","content":"task"}]}"#,
        );
        let keys = routing_keys(
            Some(&body),
            &["prompt_cache_key", "user", "session_id"],
            true,
        );
        let affinity = AffinityRouter::with_config(true, true);
        let candidates = targets.candidates("qwen");
        let expected = ModelTargets::pick_sticky_from(
            &candidates,
            keys.session_hash.expect("explicit session hash"),
        );

        assert_eq!(keys.sticky_hash, keys.session_hash);
        assert_eq!(
            select_model_target_from_keys(&targets, &candidates, &keys, &affinity, None).target,
            expected
        );
    }

    #[test]
    fn shared_remote_preparation_moves_cached_target_first() {
        let hosts = [make_id(1), make_id(2)];
        let body = parse_body(
            r#"{"messages":[{"role":"system","content":"agent"},{"role":"user","content":"task"}]}"#,
        );
        let keys = routing_keys(
            Some(&body),
            &["prompt_cache_key", "user", "session_id"],
            true,
        );
        let affinity = AffinityRouter::with_config(true, true);
        let cached = remote(2);

        let prepared =
            prepare_remote_targets_from_keys(&hosts, &keys, &affinity, Some(cached.clone()));

        assert_eq!(prepared.ordered.first(), Some(&cached));
        assert_eq!(prepared.cache_target, Some(cached));
    }

    #[test]
    fn shared_remote_preparation_records_missing_and_ineligible_cache_targets() {
        let hosts = [make_id(1), make_id(2)];
        let body = parse_body(
            r#"{"messages":[{"role":"system","content":"agent"},{"role":"user","content":"task"}]}"#,
        );
        let keys = routing_keys(
            Some(&body),
            &["prompt_cache_key", "user", "session_id"],
            true,
        );
        let affinity = AffinityRouter::with_config(true, true);

        prepare_remote_targets_from_keys(&hosts, &keys, &affinity, None);
        prepare_remote_targets_from_keys(&hosts, &keys, &affinity, Some(remote(3)));

        let stats = affinity.stats_snapshot();
        assert_eq!(stats.lookups, 2);
        assert_eq!(stats.misses, 2);
        assert_eq!(stats.hits, 0);
    }
}
