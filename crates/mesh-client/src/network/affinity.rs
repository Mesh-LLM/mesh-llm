//! Prefix affinity and sticky routing helpers for inference target selection.

use crate::inference::election;
use iroh::EndpointId;
use mesh_llm_routing::affinity as shared_affinity;
use serde::Serialize;
use serde_json::Value;
use std::sync::Arc;

#[derive(Clone, Debug, Default, Serialize)]
pub struct AffinityStatsSnapshot {
    pub prefix_enabled: bool,
    pub sticky_enabled: bool,
    pub prefix_entries: usize,
    pub prefix_lookups: u64,
    pub prefix_hits: u64,
    pub prefix_misses: u64,
    pub prefix_stale: u64,
    pub prefix_routes: u64,
    pub sticky_routes: u64,
    pub session_routes: u64,
    pub learned: u64,
    pub evicted: u64,
}

mesh_llm_routing::impl_prefix_affinity_stats_snapshot!(AffinityStatsSnapshot);

#[derive(Clone)]
pub struct AffinityRouter {
    inner: Arc<shared_affinity::AffinityRouter>,
}

impl AffinityRouter {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(shared_affinity::AffinityRouter::new()),
        }
    }

    #[cfg(test)]
    fn with_config(prefix_enabled: bool, sticky_enabled: bool) -> Self {
        Self {
            inner: Arc::new(shared_affinity::AffinityRouter::with_config(
                prefix_enabled,
                sticky_enabled,
            )),
        }
    }

    pub fn stats_snapshot(&self) -> AffinityStatsSnapshot {
        AffinityStatsSnapshot::from_prefix_affinity_stats(
            self.inner.stats_snapshot(),
            self.inner.prefix_enabled(),
            self.inner.sticky_enabled(),
        )
    }

    pub fn sticky_enabled(&self) -> bool {
        self.inner.sticky_enabled()
    }

    pub fn record_sticky_route(&self) {
        self.inner.record_sticky_route();
    }

    pub fn record_session_route(&self) {
        self.inner.record_session_route();
    }

    pub fn lookup_target(
        &self,
        model: &str,
        prefix_hash: u64,
        candidates: &[election::InferenceTarget],
    ) -> Option<election::InferenceTarget> {
        self.inner.lookup_target(model, prefix_hash, candidates)
    }

    pub fn learn_target(&self, model: &str, prefix_hash: u64, target: &election::InferenceTarget) {
        self.inner.learn_target(model, prefix_hash, target);
    }

    pub fn forget_target(&self, model: &str, prefix_hash: u64, target: &election::InferenceTarget) {
        self.inner.forget_target(model, prefix_hash, target);
    }
}

impl Default for AffinityRouter {
    fn default() -> Self {
        Self::new()
    }
}

type RoutingKeys = shared_affinity::RoutingKeys;
pub use shared_affinity::{PreparedTargets, TargetSelection};

#[cfg(test)]
pub(crate) fn extract_session_hint_from_body(body: &Value) -> Option<String> {
    shared_affinity::extract_session_hint_from_body(body, &["user", "session_id"])
}

fn routing_keys(parsed_body: Option<&Value>) -> RoutingKeys {
    shared_affinity::routing_keys(parsed_body, &["user", "session_id"], false)
}

#[cfg(test)]
fn scaffold_prefix_hash_from_body(body: &Value) -> Option<u64> {
    shared_affinity::scaffold_prefix_hash_from_body(body, false)
}

/// Select an inference target for a model request from a caller-supplied candidate
/// list instead of pulling it from `targets`. This avoids cloning the entire
/// `ModelTargets` when the caller has already reordered the candidates (e.g. by
/// context capacity).
pub fn select_model_target_from_candidates(
    targets: &election::ModelTargets,
    candidates: &[election::InferenceTarget],
    model: &str,
    parsed_body: Option<&Value>,
    affinity: &AffinityRouter,
) -> TargetSelection {
    let routing = routing_keys(parsed_body);
    shared_affinity::select_model_target_from_keys(
        targets,
        candidates,
        model,
        &routing,
        &affinity.inner,
    )
}

pub fn prepare_remote_targets_for_request(
    model: &str,
    hosts: &[EndpointId],
    parsed_body: Option<&Value>,
    affinity: &AffinityRouter,
) -> PreparedTargets {
    let routing = routing_keys(parsed_body);
    shared_affinity::prepare_remote_targets_from_keys(model, hosts, &routing, &affinity.inner)
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

    fn remote(seed: u8) -> election::InferenceTarget {
        election::InferenceTarget::Remote(make_id(seed))
    }

    fn parse_body(body: &str) -> Value {
        serde_json::from_str(body).unwrap()
    }

    #[test]
    fn legacy_result_paths_reexport_shared_structs() {
        let target = remote(1);
        let selection: TargetSelection = TargetSelection {
            target: target.clone(),
            learn_prefix_hash: None,
            cached_target: None,
        };
        let prepared: PreparedTargets = PreparedTargets {
            ordered: vec![target],
            learn_prefix_hash: selection.learn_prefix_hash,
            cached_target: selection.cached_target,
        };
        assert_eq!(prepared.ordered.len(), 1);
    }

    #[test]
    fn test_extract_session_hint_from_body_user_preferred() {
        let body =
            parse_body(r#"{"prompt_cache_key":"ignored","user":"bob","session_id":"sess-1"}"#);
        assert_eq!(
            extract_session_hint_from_body(&body),
            Some("bob".to_string())
        );
    }

    #[test]
    fn user_only_chat_has_no_scaffold_prefix_hash() {
        let body = parse_body(r#"{"messages":[{"role":"user","content":"hello"}]}"#);

        assert_eq!(scaffold_prefix_hash_from_body(&body), None);
    }

    #[test]
    fn prefix_cache_tracks_hits_misses_and_stale_candidates() {
        let affinity = AffinityRouter::with_config(true, true);
        let cached = remote(1);
        let available = [cached.clone()];

        assert_eq!(affinity.lookup_target("qwen", 7, &available), None);
        affinity.learn_target("qwen", 7, &cached);
        assert_eq!(affinity.lookup_target("qwen", 7, &available), Some(cached));
        assert_eq!(affinity.lookup_target("qwen", 7, &[remote(2)]), None);

        let stats = affinity.stats_snapshot();
        assert_eq!(stats.prefix_entries, 0);
        assert_eq!(stats.prefix_lookups, 3);
        assert_eq!(stats.prefix_hits, 1);
        assert_eq!(stats.prefix_misses, 2);
        assert_eq!(stats.prefix_stale, 1);
        assert_eq!(stats.prefix_routes, 1);
        assert_eq!(stats.learned, 1);
    }

    #[test]
    fn prefix_capacity_evicts_the_least_recently_used_entry() {
        let affinity = AffinityRouter::with_config(true, true);
        let target = remote(1);
        for prefix_hash in 0..=mesh_llm_routing::prefix_affinity::PREFIX_AFFINITY_MAX_ENTRIES as u64
        {
            affinity.learn_target("qwen", prefix_hash, &target);
        }

        let stats = affinity.stats_snapshot();

        assert_eq!(
            stats.prefix_entries,
            mesh_llm_routing::prefix_affinity::PREFIX_AFFINITY_MAX_ENTRIES
        );
        assert_eq!(stats.evicted, 1);
        assert_eq!(affinity.lookup_target("qwen", 0, &[target]), None);
    }

    #[test]
    fn forgetting_a_different_target_preserves_the_entry() {
        let affinity = AffinityRouter::with_config(true, true);
        let cached = remote(1);
        affinity.learn_target("qwen", 13, &cached);

        affinity.forget_target("qwen", 13, &remote(2));

        assert_eq!(
            affinity.lookup_target("qwen", 13, std::slice::from_ref(&cached)),
            Some(cached)
        );
        assert_eq!(affinity.stats_snapshot().prefix_stale, 0);
    }

    #[test]
    fn disabled_prefix_affinity_is_side_effect_free() {
        let affinity = AffinityRouter::with_config(false, true);
        let target = remote(1);

        affinity.learn_target("qwen", 17, &target);
        assert_eq!(
            affinity.lookup_target("qwen", 17, std::slice::from_ref(&target)),
            None
        );
        affinity.forget_target("qwen", 17, &target);

        let stats = affinity.stats_snapshot();
        assert!(!stats.prefix_enabled);
        assert_eq!(stats.prefix_entries, 0);
        assert_eq!(stats.prefix_lookups, 0);
        assert_eq!(stats.prefix_stale, 0);
        assert_eq!(stats.learned, 0);
    }

    #[test]
    fn test_routing_keys_prefix_shared_across_first_user_changes() {
        let req_a = parse_body(
            r#"{"tools":[{"type":"function","function":{"name":"run"}}],"messages":[{"role":"system","content":"You are an agent."},{"role":"user","content":"fix bug A"}]}"#,
        );
        let req_b = parse_body(
            r#"{"tools":[{"type":"function","function":{"name":"run"}}],"messages":[{"role":"system","content":"You are an agent."},{"role":"user","content":"fix bug B"}]}"#,
        );

        let keys_a = routing_keys(Some(&req_a));
        let keys_b = routing_keys(Some(&req_b));

        assert_eq!(keys_a.prefix_hash, keys_b.prefix_hash);
        assert_ne!(keys_a.sticky_hash, keys_b.sticky_hash);
    }

    #[test]
    fn test_routing_keys_prefix_ignores_object_key_order() {
        let req_a = parse_body(
            r#"{"tools":[{"type":"function","function":{"name":"run","description":"Run a command","parameters":{"type":"object","properties":{"path":{"type":"string"},"mode":{"type":"string"}},"required":["path","mode"]}}}],"messages":[{"role":"system","content":"You are an agent."},{"role":"user","content":"fix bug A"}]}"#,
        );
        let req_b = parse_body(
            r#"{"tools":[{"function":{"parameters":{"required":["path","mode"],"properties":{"mode":{"type":"string"},"path":{"type":"string"}},"type":"object"},"description":"Run a command","name":"run"},"type":"function"}],"messages":[{"role":"system","content":"You are an agent."},{"role":"user","content":"fix bug B"}]}"#,
        );

        let keys_a = routing_keys(Some(&req_a));
        let keys_b = routing_keys(Some(&req_b));

        assert_eq!(keys_a.prefix_hash, keys_b.prefix_hash);
        assert_ne!(keys_a.sticky_hash, keys_b.sticky_hash);
    }

    #[test]
    fn test_select_model_target_uses_cached_prefix_target() {
        let id_a = make_id(1);
        let id_b = make_id(2);
        let mut targets = election::ModelTargets::default();
        targets.targets.insert(
            "qwen".to_string(),
            vec![
                election::InferenceTarget::Remote(id_a),
                election::InferenceTarget::Remote(id_b),
            ],
        );

        let affinity = AffinityRouter::with_config(true, true);
        let req_a = parse_body(
            r#"{"tools":[{"type":"function","function":{"name":"run"}}],"messages":[{"role":"system","content":"You are an agent."},{"role":"user","content":"task A"}]}"#,
        );
        let req_b = parse_body(
            r#"{"tools":[{"type":"function","function":{"name":"run"}}],"messages":[{"role":"system","content":"You are an agent."},{"role":"user","content":"task B"}]}"#,
        );

        let candidates = targets.candidates("qwen");
        let first = select_model_target_from_candidates(
            &targets,
            &candidates,
            "qwen",
            Some(&req_a),
            &affinity,
        );
        let prefix_hash = first.learn_prefix_hash.unwrap();
        affinity.learn_target("qwen", prefix_hash, &first.target);

        let second = select_model_target_from_candidates(
            &targets,
            &candidates,
            "qwen",
            Some(&req_b),
            &affinity,
        );
        assert_eq!(Some(second.target.clone()), second.cached_target);
        assert_eq!(first.target, second.target);
    }

    #[test]
    fn test_prepare_remote_targets_prefers_cached_host() {
        let id_a = make_id(1);
        let id_b = make_id(2);
        let hosts = vec![id_a, id_b];
        let affinity = AffinityRouter::with_config(true, true);
        let req = parse_body(
            r#"{"messages":[{"role":"system","content":"You are an agent."},{"role":"user","content":"task A"}]}"#,
        );

        let prefix_hash = routing_keys(Some(&req)).prefix_hash.unwrap();
        affinity.learn_target(
            "qwen",
            prefix_hash,
            &election::InferenceTarget::Remote(id_b),
        );

        let prepared = prepare_remote_targets_for_request("qwen", &hosts, Some(&req), &affinity);
        assert_eq!(
            prepared.ordered.first(),
            Some(&election::InferenceTarget::Remote(id_b))
        );
        assert_eq!(
            prepared.cached_target,
            Some(election::InferenceTarget::Remote(id_b))
        );
    }
}
