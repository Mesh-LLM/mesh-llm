//! Prefix affinity and sticky routing helpers for inference target selection.

use crate::inference::election;
use crate::network::target_health::{TargetHealth, TargetHealthOutcome, TargetReputationStats};
use iroh::EndpointId;
use mesh_llm_routing::affinity as shared_affinity;
#[cfg(test)]
use mesh_llm_routing::prefix_affinity::PREFIX_AFFINITY_MAX_ENTRIES;
use mesh_llm_routing::prefix_affinity::PREFIX_AFFINITY_TTL;
use serde::Serialize;
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// How long a remembered auto-routed model stays valid for a given session
/// key. Matches the prefix affinity TTL so sticky chats and sticky routing
/// expire in lockstep.
const AUTO_MODEL_TTL: Duration = PREFIX_AFFINITY_TTL;
/// Upper bound on the auto-model cache. Each entry is small (session hash +
/// model name + timestamp) so this is generous.
const AUTO_MODEL_MAX_ENTRIES: usize = 1024;

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
    pub target_reputation: TargetReputationStats,
}

mesh_llm_routing::impl_prefix_affinity_stats_snapshot!(AffinityStatsSnapshot {
    target_reputation: TargetReputationStats::default(),
});

#[derive(Clone, Debug)]
struct AutoModelEntry {
    model: String,
    last_used: Instant,
}

#[derive(Default)]
struct AffinityState {
    auto_models: HashMap<u64, AutoModelEntry>,
    auto_lru: VecDeque<u64>,
}

#[derive(Clone)]
pub struct AffinityRouter {
    inner: Arc<Mutex<AffinityState>>,
    prefix: Arc<shared_affinity::AffinityRouter>,
    target_health: TargetHealth,
}

impl AffinityRouter {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(AffinityState::default())),
            prefix: Arc::new(shared_affinity::AffinityRouter::new()),
            target_health: TargetHealth::default(),
        }
    }

    #[cfg(test)]
    fn with_config(prefix_enabled: bool, sticky_enabled: bool) -> Self {
        Self {
            inner: Arc::new(Mutex::new(AffinityState::default())),
            prefix: Arc::new(shared_affinity::AffinityRouter::with_config(
                prefix_enabled,
                sticky_enabled,
            )),
            target_health: TargetHealth::default(),
        }
    }

    pub fn stats_snapshot(&self) -> AffinityStatsSnapshot {
        let mut stats = AffinityStatsSnapshot::from_prefix_affinity_stats(
            self.prefix.stats_snapshot(),
            self.prefix.prefix_enabled(),
            self.prefix.sticky_enabled(),
        );
        stats.target_reputation = self.target_health.reputation_stats();
        stats
    }

    pub(crate) fn route_eligible_candidates(
        &self,
        model: &str,
        candidates: &[election::InferenceTarget],
    ) -> Vec<election::InferenceTarget> {
        self.target_health.eligible_candidates(model, candidates)
    }

    pub(crate) fn route_strict_eligible_candidates(
        &self,
        model: &str,
        candidates: &[election::InferenceTarget],
    ) -> Vec<election::InferenceTarget> {
        self.target_health
            .strict_eligible_candidates(model, candidates)
    }

    pub(crate) fn record_target_outcome(
        &self,
        model: Option<&str>,
        target: &election::InferenceTarget,
        outcome: TargetHealthOutcome,
    ) {
        self.target_health.record_outcome(model, target, outcome);
    }

    #[cfg(test)]
    pub fn lookup_target(
        &self,
        model: &str,
        prefix_hash: u64,
        candidates: &[election::InferenceTarget],
    ) -> Option<election::InferenceTarget> {
        self.prefix.lookup_target(model, prefix_hash, candidates)
    }

    pub fn learn_target(&self, model: &str, prefix_hash: u64, target: &election::InferenceTarget) {
        self.prefix.learn_target(model, prefix_hash, target);
    }

    pub fn forget_target(&self, model: &str, prefix_hash: u64, target: &election::InferenceTarget) {
        self.prefix.forget_target(model, prefix_hash, target);
    }

    /// Look up a previously-classified model name for an auto-routed session.
    ///
    /// Auto routing classifies each request and picks a model. Without
    /// memory, a follow-up turn whose classification shifts (e.g. "hi" then
    /// "write code") would get routed to a different model on a different
    /// peer with a cold KV cache. Remembering the first pick keeps the
    /// whole chat on one model, so prefix affinity actually has a chance to
    /// keep it on one peer too.
    pub fn lookup_auto_model(&self, session_key: u64) -> Option<String> {
        if !self.prefix.sticky_enabled() {
            return None;
        }
        let mut state = self.inner.lock().unwrap();
        state.prune_auto_expired();
        let entry = state.auto_models.get(&session_key).cloned()?;
        state.touch_auto_key(session_key);
        if let Some(existing) = state.auto_models.get_mut(&session_key) {
            existing.last_used = Instant::now();
        }
        Some(entry.model)
    }

    pub fn remember_auto_model(&self, session_key: u64, model: &str) {
        if !self.prefix.sticky_enabled() {
            return;
        }
        let mut state = self.inner.lock().unwrap();
        state.prune_auto_expired();
        state.auto_models.insert(
            session_key,
            AutoModelEntry {
                model: model.to_string(),
                last_used: Instant::now(),
            },
        );
        state.touch_auto_key(session_key);
        while state.auto_models.len() > AUTO_MODEL_MAX_ENTRIES {
            if let Some(oldest) = state.auto_lru.pop_front() {
                state.auto_models.remove(&oldest);
            } else {
                break;
            }
        }
    }

    pub fn forget_auto_model(&self, session_key: u64) {
        let mut state = self.inner.lock().unwrap();
        state.remove_auto_key(session_key);
    }
}

/// Compute the session-level key used to cache an auto-routed model choice.
///
/// Prefers an explicit cache/session hint from the request body (e.g.
/// OpenAI-style `prompt_cache_key` or `user` fields), then falls back to the same
/// prefix/first-user-message hash sticky routing already uses. That way
/// turn 2+ of a chat reliably maps to the same key.
pub fn auto_model_session_key(parsed_body: Option<&Value>) -> Option<u64> {
    routing_keys(parsed_body).sticky_hash
}

impl Default for AffinityRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl AffinityState {
    fn prune_auto_expired(&mut self) {
        let now = Instant::now();
        while let Some(key) = self.auto_lru.front().copied() {
            match self.auto_models.get(&key) {
                Some(entry) => {
                    if now.duration_since(entry.last_used) > AUTO_MODEL_TTL {
                        self.auto_lru.pop_front();
                        self.auto_models.remove(&key);
                    } else {
                        break;
                    }
                }
                None => {
                    self.auto_lru.pop_front();
                }
            }
        }
    }

    fn touch_auto_key(&mut self, key: u64) {
        if let Some(pos) = self.auto_lru.iter().position(|existing| *existing == key) {
            self.auto_lru.remove(pos);
        }
        self.auto_lru.push_back(key);
    }

    fn remove_auto_key(&mut self, key: u64) {
        self.auto_models.remove(&key);
        if let Some(pos) = self.auto_lru.iter().position(|existing| *existing == key) {
            self.auto_lru.remove(pos);
        }
    }
}

type RoutingKeys = shared_affinity::RoutingKeys;
pub use shared_affinity::{PreparedTargets, TargetSelection};

#[cfg(test)]
pub(crate) fn extract_session_hint_from_body(body: &Value) -> Option<String> {
    shared_affinity::extract_session_hint_from_body(
        body,
        &["prompt_cache_key", "user", "session_id"],
    )
}

#[cfg(test)]
fn scaffold_prefix_hash_from_body(body: &Value) -> Option<u64> {
    shared_affinity::scaffold_prefix_hash_from_body(body, true)
}

fn routing_keys(parsed_body: Option<&Value>) -> RoutingKeys {
    shared_affinity::routing_keys(
        parsed_body,
        &["prompt_cache_key", "user", "session_id"],
        true,
    )
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
    let eligible_candidates = affinity.route_eligible_candidates(model, candidates);
    let routing = routing_keys(parsed_body);
    shared_affinity::select_model_target_from_keys(
        targets,
        &eligible_candidates,
        model,
        &routing,
        &affinity.prefix,
    )
}

pub fn prepare_remote_targets_for_request(
    model: &str,
    hosts: &[EndpointId],
    parsed_body: Option<&Value>,
    affinity: &AffinityRouter,
) -> PreparedTargets {
    let routing = routing_keys(parsed_body);
    let mut prepared =
        shared_affinity::prepare_remote_targets_from_keys(model, hosts, &routing, &affinity.prefix);

    let eligible = affinity.route_eligible_candidates(model, &prepared.ordered);
    if eligible.len() != prepared.ordered.len() {
        if let (Some(prefix_hash), Some(target)) =
            (prepared.learn_prefix_hash, prepared.cached_target.as_ref())
            && !eligible.contains(target)
        {
            affinity.forget_target(model, prefix_hash, target);
            prepared.cached_target = None;
        }
        prepared.ordered = eligible;
    }

    prepared
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::target_health::TargetHealthOutcome;
    use iroh::SecretKey;

    const TEST_MODEL: &str = "qwen";

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

    struct SimulatedMeshRouter {
        affinity: AffinityRouter,
        hosts: Vec<EndpointId>,
    }

    impl SimulatedMeshRouter {
        fn new(host_seeds: &[u8]) -> Self {
            Self {
                affinity: AffinityRouter::default(),
                hosts: host_seeds.iter().map(|seed| make_id(*seed)).collect(),
            }
        }

        fn route_order(&self) -> Vec<election::InferenceTarget> {
            prepare_remote_targets_for_request(TEST_MODEL, &self.hosts, None, &self.affinity)
                .ordered
        }

        fn record_peer_outcome(&self, host_index: usize, outcome: TargetHealthOutcome) {
            let target = election::InferenceTarget::Remote(self.hosts[host_index]);
            self.affinity
                .record_target_outcome(Some(TEST_MODEL), &target, outcome);
        }
    }

    #[test]
    fn test_extract_session_hint_from_body_prompt_cache_key_preferred() {
        let body =
            parse_body(r#"{"prompt_cache_key":"cache-1","user":"bob","session_id":"sess-1"}"#);
        assert_eq!(
            extract_session_hint_from_body(&body),
            Some("cache-1".to_string())
        );
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
        for prefix_hash in 0..=PREFIX_AFFINITY_MAX_ENTRIES as u64 {
            affinity.learn_target("qwen", prefix_hash, &target);
        }

        let stats = affinity.stats_snapshot();

        assert_eq!(stats.prefix_entries, PREFIX_AFFINITY_MAX_ENTRIES);
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
        affinity.record_target_outcome(
            Some("qwen"),
            &election::InferenceTarget::Remote(id_b),
            TargetHealthOutcome::Unavailable,
        );
        let prepared = prepare_remote_targets_for_request("qwen", &hosts, Some(&req), &affinity);
        assert_eq!(
            prepared.ordered,
            vec![election::InferenceTarget::Remote(id_a)]
        );
        assert_eq!(prepared.cached_target, None);
    }

    #[test]
    fn test_prepare_remote_targets_filters_cooling_session_hint_target() {
        let id_a = make_id(1);
        let id_b = make_id(2);
        let hosts = vec![id_a, id_b];
        let affinity = AffinityRouter::with_config(true, true);
        let req = parse_body(
            r#"{"prompt_cache_key":"cache-1","messages":[{"role":"user","content":"task A"}]}"#,
        );

        let prepared = prepare_remote_targets_for_request("qwen", &hosts, Some(&req), &affinity);
        let cooling_target = prepared.ordered.first().cloned().unwrap();

        affinity.record_target_outcome(
            Some("qwen"),
            &cooling_target,
            TargetHealthOutcome::Unavailable,
        );

        let prepared = prepare_remote_targets_for_request("qwen", &hosts, Some(&req), &affinity);
        assert!(!prepared.ordered.contains(&cooling_target));
        assert_eq!(prepared.ordered.len(), 1);
        assert_eq!(prepared.learn_prefix_hash, None);
        assert_eq!(prepared.cached_target, None);
    }

    #[test]
    fn strict_eligible_candidates_drop_single_cooling_auto_target() {
        let id = make_id(1);
        let target = election::InferenceTarget::Remote(id);
        let affinity = AffinityRouter::with_config(true, true);

        affinity.record_target_outcome(Some("qwen"), &target, TargetHealthOutcome::Unavailable);

        assert_eq!(
            affinity.route_eligible_candidates("qwen", std::slice::from_ref(&target)),
            vec![target.clone()]
        );
        assert!(
            affinity
                .route_strict_eligible_candidates("qwen", std::slice::from_ref(&target))
                .is_empty()
        );
    }

    #[test]
    fn stats_snapshot_exposes_local_target_reputation() {
        let id_a = make_id(1);
        let id_b = make_id(2);
        let first = election::InferenceTarget::Remote(id_a);
        let second = election::InferenceTarget::Remote(id_b);
        let affinity = AffinityRouter::with_config(true, true);

        affinity.record_target_outcome(Some("qwen"), &first, TargetHealthOutcome::Unavailable);

        assert_eq!(
            affinity.route_eligible_candidates("qwen", &[first.clone(), second.clone()]),
            vec![second]
        );
        let stats = affinity.stats_snapshot();
        assert_eq!(stats.target_reputation.penalized_targets, 1);
        assert_eq!(stats.target_reputation.routes_penalized, 0);
    }

    #[test]
    fn simulated_multi_node_reputation_changes_remote_request_flow() {
        let mesh = SimulatedMeshRouter::new(&[1, 2, 3]);

        assert_eq!(mesh.route_order(), vec![remote(1), remote(2), remote(3)]);

        mesh.record_peer_outcome(0, TargetHealthOutcome::Unavailable);

        assert_eq!(mesh.route_order(), vec![remote(2), remote(3)]);
        assert_eq!(
            mesh.affinity
                .stats_snapshot()
                .target_reputation
                .penalized_targets,
            1
        );

        mesh.record_peer_outcome(0, TargetHealthOutcome::Success);

        assert_eq!(mesh.route_order(), vec![remote(1), remote(2), remote(3)]);
    }

    #[test]
    fn scaffold_prefix_hash_falls_back_to_first_user_message() {
        // No system/developer prompt, no tools — the old behavior returned
        // None here and the prefix cache never learned anything. Now it
        // hashes the first user message so chats without system prompts
        // can still stick to the same peer on turn 2+.
        let req = parse_body(
            r#"{"messages":[{"role":"user","content":"hello"},{"role":"assistant","content":"hi"}]}"#,
        );
        let hash = scaffold_prefix_hash_from_body(&req);
        assert!(
            hash.is_some(),
            "expected a prefix hash for a chat with only a user message"
        );
    }

    #[test]
    fn scaffold_prefix_hash_stable_across_chat_turns() {
        // Same first user message, growing conversation — the prefix hash
        // must be identical so both turns map to the same affinity key.
        let turn_1 = parse_body(r#"{"messages":[{"role":"user","content":"tell me a joke"}]}"#);
        let turn_2 = parse_body(
            r#"{"messages":[{"role":"user","content":"tell me a joke"},{"role":"assistant","content":"why did ..."},{"role":"user","content":"another one"}]}"#,
        );
        assert_eq!(
            scaffold_prefix_hash_from_body(&turn_1),
            scaffold_prefix_hash_from_body(&turn_2),
        );
    }

    #[test]
    fn scaffold_prefix_hash_differs_between_sessions() {
        let a = parse_body(r#"{"messages":[{"role":"user","content":"topic a"}]}"#);
        let b = parse_body(r#"{"messages":[{"role":"user","content":"topic b"}]}"#);
        assert_ne!(
            scaffold_prefix_hash_from_body(&a),
            scaffold_prefix_hash_from_body(&b),
        );
    }

    #[test]
    fn auto_model_cache_round_trip() {
        let affinity = AffinityRouter::new();
        let key = 0xabcdef123456u64;
        assert_eq!(affinity.lookup_auto_model(key), None);
        affinity.remember_auto_model(key, "Qwen3.5-9B-Q4_K_M");
        assert_eq!(
            affinity.lookup_auto_model(key),
            Some("Qwen3.5-9B-Q4_K_M".to_string())
        );
    }

    #[test]
    fn auto_model_cache_forget() {
        let affinity = AffinityRouter::new();
        let key = 42u64;
        affinity.remember_auto_model(key, "some-model");
        affinity.forget_auto_model(key);
        assert_eq!(affinity.lookup_auto_model(key), None);
    }

    #[test]
    fn auto_model_cache_evicts_oldest_over_capacity() {
        let affinity = AffinityRouter::new();
        for i in 0..(AUTO_MODEL_MAX_ENTRIES as u64 + 10) {
            affinity.remember_auto_model(i, "model-x");
        }
        // The very first inserts should have been evicted.
        assert_eq!(affinity.lookup_auto_model(0), None);
        assert_eq!(affinity.lookup_auto_model(1), None);
        // Recent inserts survive.
        let recent = AUTO_MODEL_MAX_ENTRIES as u64 + 5;
        assert_eq!(
            affinity.lookup_auto_model(recent),
            Some("model-x".to_string())
        );
    }

    #[test]
    fn auto_model_session_key_matches_sticky_hash() {
        let body = parse_body(
            r#"{"messages":[{"role":"system","content":"be helpful"},{"role":"user","content":"hi"}]}"#,
        );
        let key = auto_model_session_key(Some(&body)).expect("expected a session key");
        let sticky = routing_keys(Some(&body)).sticky_hash.unwrap();
        assert_eq!(key, sticky);
    }

    #[test]
    fn auto_model_cache_disabled_when_sticky_disabled() {
        let affinity = AffinityRouter::with_config(true, false);
        affinity.remember_auto_model(1, "model");
        assert_eq!(affinity.lookup_auto_model(1), None);
    }

    #[test]
    fn auto_model_cache_survives_forget_target_calls() {
        // forget_target operates on prefix affinity, not the auto-model
        // memo. A transient per-host prefix miss shouldn't flush the
        // session's model choice.
        let affinity = AffinityRouter::new();
        affinity.remember_auto_model(7, "chat-model");
        let target = election::InferenceTarget::Remote(make_id(5));
        affinity.forget_target("chat-model", 0xdead_beef, &target);
        assert_eq!(
            affinity.lookup_auto_model(7),
            Some("chat-model".to_string())
        );
    }
}
