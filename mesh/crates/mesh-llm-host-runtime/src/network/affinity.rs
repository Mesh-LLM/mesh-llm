//! Prefix affinity and sticky routing helpers for inference target selection.

use crate::inference::election;
use crate::network::reservations::{RoutingReservation, RoutingReservations};
use crate::network::target_health::{TargetHealth, TargetHealthOutcome, TargetReputationStats};
use iroh::EndpointId;
use mesh_llm_routing::affinity as shared_affinity;
use mesh_llm_routing::request_cost;

/// Fewest prompt tokens a reply needs before it teaches the bytes-per-token
/// ratio, and how many models the ratio is kept for.
const MIN_PROMPT_SIZE_SAMPLE_TOKENS: u64 = 64;
const PROMPT_SIZE_MAX_MODELS: usize = 1_024;

/// Smallest credited prefix that lets a cache hit override normal routing.
const MIN_SAVED_TOKENS: u32 = mesh_llm_routing::cache_aware::DEFAULT_MIN_SAVED_TOKENS;
use serde::Serialize;
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// How long a remembered auto-routed model stays valid for a given session
/// key. Matches the prefix affinity TTL so sticky chats and sticky routing
/// expire in lockstep.
const AUTO_MODEL_TTL: Duration = Duration::from_secs(20 * 60);
/// Upper bound on the auto-model cache. Each entry is small (session hash +
/// model name + timestamp) so this is generous.
const AUTO_MODEL_MAX_ENTRIES: usize = 1024;
const CACHE_LEASE_TTL: Duration = Duration::from_secs(2);
const CACHE_LEASE_MAX_ENTRIES: usize = 1024;

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
    /// Legacy status compatibility. Long-lived learned prefix mappings were
    /// removed; this counter is permanently zero.
    pub learned: u64,
    /// Legacy status compatibility paired with `learned`; permanently zero.
    pub evicted: u64,
    pub target_reputation: TargetReputationStats,
    /// Requests currently holding an in-flight route reservation.
    pub reservation_active: usize,
}

mesh_llm_routing::impl_affinity_stats_snapshot!(AffinityStatsSnapshot {
    target_reputation: TargetReputationStats::default(),
    reservation_active: 0,
});

#[derive(Clone, Debug)]
struct AutoModelEntry {
    model: String,
    last_used: Instant,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct CacheLeaseKey {
    model: String,
    prefix_hash: u64,
}

#[derive(Clone, Debug)]
struct CacheLeaseEntry {
    target: election::InferenceTarget,
    expires_at: Instant,
}

#[derive(Default)]
struct AffinityState {
    auto_models: HashMap<u64, AutoModelEntry>,
    auto_lru: VecDeque<u64>,
    cache_leases: HashMap<CacheLeaseKey, CacheLeaseEntry>,
    cache_lease_lru: VecDeque<CacheLeaseKey>,
    cache_lease_epoch: u64,
    /// Prompt bytes per token learned per model from completed replies, in
    /// thousandths.
    prompt_bytes_per_token_milli: HashMap<String, u32>,
}

#[derive(Clone)]
pub struct AffinityRouter {
    inner: Arc<Mutex<AffinityState>>,
    prefix: Arc<shared_affinity::AffinityRouter>,
    target_health: TargetHealth,
    reservations: RoutingReservations,
}

impl AffinityRouter {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(AffinityState::default())),
            prefix: Arc::new(shared_affinity::AffinityRouter::new()),
            target_health: TargetHealth::default(),
            reservations: RoutingReservations::default(),
        }
    }

    /// The production router: registers its route-health table on `node`
    /// so peer vouches (plugin UNBLOCKs) can lift cooldowns on it.
    pub(crate) fn for_node(node: &crate::mesh::Node) -> Self {
        let router = Self::new();
        let _ = node.route_target_health.set(router.target_health.clone());
        router
    }

    #[cfg(test)]
    pub(crate) fn with_config(prefix_enabled: bool, sticky_enabled: bool) -> Self {
        Self {
            inner: Arc::new(Mutex::new(AffinityState::default())),
            prefix: Arc::new(shared_affinity::AffinityRouter::with_config(
                prefix_enabled,
                sticky_enabled,
            )),
            target_health: TargetHealth::default(),
            reservations: RoutingReservations::default(),
        }
    }

    /// Learn how many prompt bytes `model` spends per token from a reply's
    /// `usage.prompt_tokens`, so the next request is sized from its body at a
    /// measured ratio rather than a fixed guess. Small prompts are skipped:
    /// the chat template's own tokens would dominate them.
    pub(crate) fn observe_prompt_size(&self, model: &str, prompt_bytes: usize, prompt_tokens: u64) {
        if prompt_tokens < MIN_PROMPT_SIZE_SAMPLE_TOKENS || prompt_bytes == 0 {
            return;
        }
        let ratio = u64::try_from(prompt_bytes)
            .unwrap_or(u64::MAX)
            .saturating_mul(1_000)
            / prompt_tokens;
        let ratio = u32::try_from(ratio).unwrap_or(u32::MAX).clamp(
            request_cost::MIN_BYTES_PER_TOKEN_MILLI,
            request_cost::MAX_BYTES_PER_TOKEN_MILLI,
        );
        let mut state = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let known = state.prompt_bytes_per_token_milli.contains_key(model);
        if !known && state.prompt_bytes_per_token_milli.len() >= PROMPT_SIZE_MAX_MODELS {
            return;
        }
        state
            .prompt_bytes_per_token_milli
            .entry(model.to_string())
            .and_modify(|previous| *previous = (previous.saturating_mul(3) + ratio) / 4)
            .or_insert(ratio);
    }

    pub(crate) fn prompt_bytes_per_token_milli(&self, model: &str) -> Option<u32> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .prompt_bytes_per_token_milli
            .get(model)
            .copied()
    }

    pub fn stats_snapshot(&self) -> AffinityStatsSnapshot {
        let mut stats = AffinityStatsSnapshot::from_affinity_stats(
            self.prefix.stats_snapshot(),
            self.prefix.prefix_enabled(),
            self.prefix.sticky_enabled(),
        );
        stats.target_reputation = self.target_health.reputation_stats();
        stats.reservation_active = self.reservations.active_total();
        stats
    }

    pub(crate) fn prefix_enabled(&self) -> bool {
        self.prefix.prefix_enabled()
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

    pub(crate) fn reserve_route(
        &self,
        model: &str,
        candidates: &[election::InferenceTarget],
        spread_limit: usize,
        preferred: &election::InferenceTarget,
        affinity_applied: bool,
    ) -> Option<(election::InferenceTarget, RoutingReservation)> {
        self.reservations
            .reserve(model, candidates, spread_limit, preferred, affinity_applied)
    }

    /// Two-choice host pick and reservation over `candidates` for a request
    /// without affinity; see [`RoutingReservations::reserve_two_choice`].
    /// Hosts with a reputation penalty, and those `deprioritized` marks, are
    /// spillover: drawn only when no other candidate remains.
    pub(crate) fn reserve_two_choice_target(
        &self,
        model: &str,
        candidates: &[election::InferenceTarget],
        deprioritized: impl Fn(&election::InferenceTarget) -> bool,
        weight_of: impl Fn(&election::InferenceTarget) -> crate::network::router::PlacementWeight,
    ) -> Option<(election::InferenceTarget, RoutingReservation)> {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        let penalized = self.target_health.penalized_targets(model, candidates);
        self.reservations.reserve_two_choice(
            model,
            candidates,
            |target| penalized.contains(target) || deprioritized(target),
            weight_of,
            seed,
        )
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

    pub(crate) fn lookup_cache_lease(
        &self,
        model: &str,
        prefix_hash: u64,
        candidates: &[election::InferenceTarget],
    ) -> Option<election::InferenceTarget> {
        let mut state = self.inner.lock().unwrap();
        state.prune_cache_leases();
        let key = CacheLeaseKey {
            model: model.to_string(),
            prefix_hash,
        };
        let target = state.cache_leases.get(&key)?.target.clone();
        if !candidates.contains(&target) {
            return None;
        }
        if let Some(position) = state.cache_lease_lru.iter().position(|item| item == &key) {
            state.cache_lease_lru.remove(position);
        }
        state.cache_lease_lru.push_back(key);
        Some(target)
    }

    #[cfg(test)]
    pub(crate) fn remember_cache_lease(
        &self,
        model: &str,
        prefix_hash: u64,
        target: &election::InferenceTarget,
    ) {
        let mut state = self.inner.lock().unwrap();
        state.remember_cache_lease(model, prefix_hash, target);
    }

    pub(crate) fn cache_lease_epoch(&self) -> u64 {
        self.inner.lock().unwrap().cache_lease_epoch
    }

    pub(crate) fn remember_cache_lease_if_epoch(
        &self,
        model: &str,
        prefix_hash: u64,
        target: &election::InferenceTarget,
        expected_epoch: u64,
    ) -> bool {
        let mut state = self.inner.lock().unwrap();
        if state.cache_lease_epoch != expected_epoch {
            return false;
        }
        state.remember_cache_lease(model, prefix_hash, target);
        true
    }

    pub(crate) fn forget_cache_lease(&self, model: &str, prefix_hash: u64) -> bool {
        let mut state = self.inner.lock().unwrap();
        state.cache_lease_epoch = state.cache_lease_epoch.wrapping_add(1);
        let key = CacheLeaseKey {
            model: model.to_string(),
            prefix_hash,
        };
        let removed = state.cache_leases.remove(&key).is_some();
        if removed
            && let Some(position) = state.cache_lease_lru.iter().position(|item| item == &key)
        {
            state.cache_lease_lru.remove(position);
        }
        removed
    }

    pub(crate) fn forget_cache_leases_for_target(
        &self,
        target: &election::InferenceTarget,
    ) -> usize {
        let mut state = self.inner.lock().unwrap();
        state.cache_lease_epoch = state.cache_lease_epoch.wrapping_add(1);
        let before = state.cache_leases.len();
        state
            .cache_leases
            .retain(|_, entry| &entry.target != target);
        let live_keys: std::collections::HashSet<_> = state.cache_leases.keys().cloned().collect();
        state.cache_lease_lru.retain(|key| live_keys.contains(key));
        before.saturating_sub(state.cache_leases.len())
    }

    pub(crate) fn record_cache_probe(&self, hit: bool) {
        self.prefix.record_cache_probe(hit);
    }
}

/// Compute the session-level key used to cache an auto-routed model choice.
///
/// Prefers the OpenAI-style `prompt_cache_key`, then falls back to an explicit
/// `user` or `session_id` hint. The cache key remains valid for model-choice
/// memory without also becoming a sticky target-routing key.
pub fn auto_model_session_key(parsed_body: Option<&Value>) -> Option<u64> {
    let routing = routing_keys(parsed_body);
    if parsed_body.is_some_and(|body| {
        body.get("prompt_cache_key")
            .and_then(Value::as_str)
            .is_some()
    }) {
        routing.prefix_hash
    } else {
        routing.sticky_hash
    }
}

impl Default for AffinityRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl AffinityState {
    fn remember_cache_lease(
        &mut self,
        model: &str,
        prefix_hash: u64,
        target: &election::InferenceTarget,
    ) {
        self.prune_cache_leases();
        let key = CacheLeaseKey {
            model: model.to_string(),
            prefix_hash,
        };
        self.cache_leases.insert(
            key.clone(),
            CacheLeaseEntry {
                target: target.clone(),
                expires_at: Instant::now() + CACHE_LEASE_TTL,
            },
        );
        if let Some(position) = self.cache_lease_lru.iter().position(|item| item == &key) {
            self.cache_lease_lru.remove(position);
        }
        self.cache_lease_lru.push_back(key);
        while self.cache_leases.len() > CACHE_LEASE_MAX_ENTRIES {
            if let Some(oldest) = self.cache_lease_lru.pop_front() {
                self.cache_leases.remove(&oldest);
            } else {
                break;
            }
        }
    }

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

    fn prune_cache_leases(&mut self) {
        let now = Instant::now();
        self.cache_leases.retain(|_, entry| entry.expires_at > now);
        self.cache_lease_lru
            .retain(|key| self.cache_leases.contains_key(key));
    }
}

type RoutingKeys = shared_affinity::RoutingKeys;
pub use shared_affinity::{PreparedTargets, TargetSelection};

pub(in crate::network) struct LocalCacheCost {
    pub(in crate::network) queue_delay_micros: u64,
    pub(in crate::network) restore_micros: u64,
    pub(in crate::network) prefill_micros_per_token: Option<u64>,
}

#[cfg(test)]
pub(crate) fn extract_session_hint_from_body(body: &Value) -> Option<String> {
    shared_affinity::extract_session_hint_from_body(body, &["user", "session_id"])
}

#[cfg(test)]
fn scaffold_prefix_hash_from_body(body: &Value) -> Option<u64> {
    shared_affinity::scaffold_prefix_hash_from_body(body, true)
}

fn routing_keys(parsed_body: Option<&Value>) -> RoutingKeys {
    shared_affinity::routing_keys(
        parsed_body,
        &["prompt_cache_key"],
        &["user", "session_id"],
        true,
    )
}

pub(crate) fn cache_prefix_hash(parsed_body: Option<&Value>) -> Option<u64> {
    routing_keys(parsed_body).prefix_hash
}

/// Size of this request and of the prefix it provably shares, with the same
/// scaffold rule as `routing_keys`, so the credited prefix is what was hashed,
/// at the model's learned bytes per token when one is known.
pub(crate) fn request_cost_input(
    parsed_body: Option<&Value>,
    affinity: &AffinityRouter,
    model: &str,
) -> request_cost::RequestCostInput {
    request_cost::request_cost_input_from_body_calibrated(
        parsed_body,
        true,
        affinity.prompt_bytes_per_token_milli(model),
    )
}

/// Prompt bytes of a request body, as `request_cost_input` measures them.
pub(crate) fn prompt_bytes(parsed_body: Option<&Value>) -> Option<usize> {
    parsed_body.map(shared_affinity::prompt_bytes_from_body)
}

impl crate::mesh::Node {
    /// Record only provider-confirmed local L1 reuse. A successful request with
    /// zero cached tokens is intentionally not evidence of residency.
    #[cfg(test)]
    pub(crate) fn record_local_cache_hit(
        &self,
        model: &str,
        prefix_hash: u64,
        cached_tokens: u32,
        suffix_prefill_tokens: u32,
        queue_delay_micros: u64,
    ) {
        self.cache_affinity_inventory
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .record_l1_hit(
                model,
                prefix_hash,
                cached_tokens,
                suffix_prefill_tokens,
                queue_delay_micros,
            );
    }

    pub(in crate::network) fn record_local_cache_hit_with_cost(
        &self,
        model: &str,
        prefix_hash: u64,
        cached_tokens: u32,
        suffix_prefill_tokens: u32,
        cost: LocalCacheCost,
    ) {
        let mut inventory = self
            .cache_affinity_inventory
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(prefill_micros_per_token) = cost.prefill_micros_per_token {
            inventory.observe_prefill_cost(model, prefill_micros_per_token);
        }
        inventory.record_l1_hit_with_cost(
            model,
            prefix_hash,
            cached_tokens,
            suffix_prefill_tokens,
            cost.restore_micros,
            cost.queue_delay_micros,
        );
    }

    pub(crate) fn observe_local_prefill_cost(
        &self,
        model: &str,
        prefill_micros_per_token: Option<u64>,
    ) {
        let Some(prefill_micros_per_token) = prefill_micros_per_token else {
            return;
        };
        self.cache_affinity_inventory
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .observe_prefill_cost(model, prefill_micros_per_token);
    }

    /// Remove provider-refuted local cache evidence immediately instead of
    /// waiting for its bounded gossip TTL to expire.
    pub(crate) fn invalidate_local_cache_evidence(&self, model: &str, prefix_hash: u64) -> bool {
        self.cache_affinity_inventory
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .invalidate(model, prefix_hash)
    }

    /// Choose a target for this request from cache evidence, shared by the
    /// model route and the remote-host ordering so both apply the same
    /// estimate. The estimate comes first: a short lease, the memory of a
    /// recent decision, only stands in when nothing can be costed yet (a hit
    /// not advertised so far), so it never overrides a costed comparison.
    pub(crate) async fn cache_target_with_lease(
        &self,
        affinity: &AffinityRouter,
        model: &str,
        prefix_hash: u64,
        candidates: &[election::InferenceTarget],
        request: request_cost::RequestCostInput,
    ) -> Option<election::InferenceTarget> {
        if !request_cost::can_earn_cache_credit(request, MIN_SAVED_TOKENS) {
            return None;
        }
        let lease_epoch = affinity.cache_lease_epoch();
        match self
            .cache_route_decision(model, prefix_hash, candidates, request)
            .await
        {
            request_cost::CacheRouteDecision::CacheHitWins { target, .. }
            | request_cost::CacheRouteDecision::ColdWins { target, .. } => {
                affinity.remember_cache_lease_if_epoch(model, prefix_hash, &target, lease_epoch);
                Some(target)
            }
            request_cost::CacheRouteDecision::Marginal => None,
            request_cost::CacheRouteDecision::Incomparable(_)
            | request_cost::CacheRouteDecision::NoEvidence => {
                affinity.lookup_cache_lease(model, prefix_hash, candidates)
            }
        }
    }

    /// Probe bounded local and peer advertisements for the exact request
    /// prefix and compare the best hit with serving the request cold on every
    /// costed candidate (`request_cost::decide_cache_route`): the target is the
    /// hit when it wins clearly, the cheapest cold candidate when that is
    /// estimated cheaper. A marginal difference, or missing, expired,
    /// unmeasured or incomparable evidence, preserves normal candidate order.
    /// Routing goes through `cache_target_with_lease`; this lease-free form
    /// lets the tests check the decision on its own.
    #[cfg(test)]
    pub(crate) async fn select_cache_target(
        &self,
        model: &str,
        prefix_hash: u64,
        candidates: &[election::InferenceTarget],
        request: request_cost::RequestCostInput,
    ) -> Option<election::InferenceTarget> {
        if !request_cost::can_earn_cache_credit(request, MIN_SAVED_TOKENS) {
            return None;
        }
        match self
            .cache_route_decision(model, prefix_hash, candidates, request)
            .await
        {
            request_cost::CacheRouteDecision::CacheHitWins { target, .. }
            | request_cost::CacheRouteDecision::ColdWins { target, .. } => Some(target),
            _ => None,
        }
    }

    async fn cache_route_decision(
        &self,
        model: &str,
        prefix_hash: u64,
        candidates: &[election::InferenceTarget],
        request: request_cost::RequestCostInput,
    ) -> request_cost::CacheRouteDecision {
        let (local_evidence, local_rate) = {
            let mut inventory = self
                .cache_affinity_inventory
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            (
                inventory.probe_local(model, prefix_hash),
                inventory.model_prefill_micros_per_token(model),
            )
        };
        let mut spread_candidates = candidates.to_vec();
        if !spread_candidates.is_empty() {
            let offset = prefix_hash as usize % spread_candidates.len();
            spread_candidates.rotate_left(offset);
        }
        let now_unix_ms = crate::mesh::current_time_unix_ms();
        let state = self.state.lock().await;
        let inputs: Vec<_> = spread_candidates
            .iter()
            .filter_map(|target| {
                let (entry, rate) = match target {
                    election::InferenceTarget::Local(_) => (local_evidence.clone(), local_rate),
                    election::InferenceTarget::Remote(peer_id) => {
                        let peer = state.peers.get(peer_id);
                        let advertisement = peer.and_then(|peer| peer.cache_affinity.as_ref());
                        // The peer's dedicated per-model rate first; a node
                        // that predates it still carries one on its cache
                        // entries while they last.
                        let rate = peer
                            .and_then(|peer| {
                                peer.advertised_model_throughput
                                    .iter()
                                    .find(|hint| hint.model_name == model)
                            })
                            .and_then(|hint| hint.prefill)
                            .and_then(|prefill| prefill.fresh_micros_per_token(now_unix_ms))
                            .or_else(|| {
                                advertisement.and_then(|advertisement| {
                                    advertisement.model_prefill_micros_per_token(model, now_unix_ms)
                                })
                            });
                        (
                            advertisement.and_then(|advertisement| {
                                advertisement.probe(model, prefix_hash, now_unix_ms)
                            }),
                            rate,
                        )
                    }
                    election::InferenceTarget::None => return None,
                };
                Some(request_cost::candidate_cost_input(
                    target.clone(),
                    entry.as_ref(),
                    rate,
                ))
            })
            .collect();
        drop(state);
        let decision = request_cost::decide_cache_route(request, &inputs, MIN_SAVED_TOKENS);
        tracing::debug!(model, ?decision, "cache route decision");
        decision
    }
}

/// Select from an already health-filtered snapshot. Callers that also reserve
/// or retry targets must use this same snapshot for those decisions.
pub(crate) fn select_model_target_from_eligible_candidates(
    targets: &election::ModelTargets,
    candidates: &[election::InferenceTarget],
    parsed_body: Option<&Value>,
    affinity: &AffinityRouter,
    cache_target: Option<election::InferenceTarget>,
) -> TargetSelection {
    let routing = routing_keys(parsed_body);
    shared_affinity::select_model_target_from_keys(
        targets,
        candidates,
        &routing,
        &affinity.prefix,
        cache_target,
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
        shared_affinity::prepare_remote_targets_from_keys(hosts, &routing, &affinity.prefix, None);

    let eligible = affinity.route_eligible_candidates(model, &prepared.ordered);
    if eligible.len() != prepared.ordered.len() {
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
            prefix_hash: None,
            cache_target: None,
            affinity_applied: false,
        };
        let prepared: PreparedTargets = PreparedTargets {
            ordered: vec![target],
            prefix_hash: selection.prefix_hash,
            cache_target: selection.cache_target,
            affinity_applied: selection.affinity_applied,
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
    fn test_extract_session_hint_from_body_user_preferred() {
        let body =
            parse_body(r#"{"prompt_cache_key":"cache-1","user":"bob","session_id":"sess-1"}"#);
        assert_eq!(
            extract_session_hint_from_body(&body),
            Some("bob".to_string())
        );
    }

    #[test]
    fn short_cache_lease_never_returns_an_ineligible_target() {
        let affinity = AffinityRouter::with_config(true, true);
        let leased = remote(1);
        affinity.remember_cache_lease(TEST_MODEL, 7, &leased);

        assert_eq!(
            affinity.lookup_cache_lease(TEST_MODEL, 7, std::slice::from_ref(&leased)),
            Some(leased)
        );
        assert_eq!(
            affinity.lookup_cache_lease(TEST_MODEL, 7, &[remote(2)]),
            None
        );
    }

    #[test]
    fn cache_lease_lookup_refreshes_recency() {
        let affinity = AffinityRouter::with_config(true, true);
        let target = remote(1);
        affinity.remember_cache_lease(TEST_MODEL, 7, &target);
        affinity.remember_cache_lease(TEST_MODEL, 8, &target);

        assert_eq!(
            affinity.lookup_cache_lease(TEST_MODEL, 7, std::slice::from_ref(&target)),
            Some(target)
        );
        assert_eq!(
            affinity
                .inner
                .lock()
                .unwrap()
                .cache_lease_lru
                .back()
                .map(|key| key.prefix_hash),
            Some(7)
        );
    }

    #[test]
    fn cache_lease_invalidation_removes_only_the_failed_target() {
        let affinity = AffinityRouter::with_config(true, true);
        let failed = remote(1);
        let healthy = remote(2);
        affinity.remember_cache_lease(TEST_MODEL, 7, &failed);
        affinity.remember_cache_lease(TEST_MODEL, 8, &healthy);

        assert_eq!(affinity.forget_cache_leases_for_target(&failed), 1);
        assert_eq!(
            affinity.lookup_cache_lease(TEST_MODEL, 7, std::slice::from_ref(&failed)),
            None
        );
        assert_eq!(
            affinity.lookup_cache_lease(TEST_MODEL, 8, std::slice::from_ref(&healthy)),
            Some(healthy)
        );
    }

    #[test]
    fn exact_cache_lease_invalidation_preserves_other_prefixes() {
        let affinity = AffinityRouter::with_config(true, true);
        let target = remote(1);
        affinity.remember_cache_lease(TEST_MODEL, 7, &target);
        affinity.remember_cache_lease(TEST_MODEL, 8, &target);

        assert!(affinity.forget_cache_lease(TEST_MODEL, 7));
        assert!(!affinity.forget_cache_lease(TEST_MODEL, 7));
        assert_eq!(
            affinity.lookup_cache_lease(TEST_MODEL, 7, std::slice::from_ref(&target)),
            None
        );
        assert_eq!(
            affinity.lookup_cache_lease(TEST_MODEL, 8, std::slice::from_ref(&target)),
            Some(target)
        );
    }

    #[test]
    fn cache_lease_invalidation_fences_a_stale_selection() {
        let affinity = AffinityRouter::with_config(true, true);
        let target = remote(1);
        let selection_epoch = affinity.cache_lease_epoch();

        assert!(!affinity.forget_cache_lease(TEST_MODEL, 7));
        assert!(!affinity.remember_cache_lease_if_epoch(TEST_MODEL, 7, &target, selection_epoch));
        assert_eq!(
            affinity.lookup_cache_lease(TEST_MODEL, 7, std::slice::from_ref(&target)),
            None
        );
    }

    #[test]
    fn test_routing_keys_prefix_is_shared_across_first_users() {
        let req_a = parse_body(
            r#"{"tools":[{"type":"function","function":{"name":"run"}}],"messages":[{"role":"system","content":"You are an agent."},{"role":"user","content":"fix bug A"}]}"#,
        );
        let req_b = parse_body(
            r#"{"tools":[{"type":"function","function":{"name":"run"}}],"messages":[{"role":"system","content":"You are an agent."},{"role":"user","content":"fix bug B"}]}"#,
        );

        let keys_a = routing_keys(Some(&req_a));
        let keys_b = routing_keys(Some(&req_b));

        assert_eq!(keys_a.prefix_hash, keys_b.prefix_hash);
        assert_eq!(keys_a.sticky_hash, None);
        assert_eq!(keys_b.sticky_hash, None);
    }

    #[test]
    fn test_routing_keys_prefix_ignores_object_key_order() {
        let req_a = parse_body(
            r#"{"tools":[{"type":"function","function":{"name":"run","description":"Run a command","parameters":{"type":"object","properties":{"path":{"type":"string"},"mode":{"type":"string"}},"required":["path","mode"]}}}],"messages":[{"role":"system","content":"You are an agent."},{"role":"user","content":"fix bug A"}]}"#,
        );
        let req_b = parse_body(
            r#"{"tools":[{"function":{"parameters":{"required":["path","mode"],"properties":{"mode":{"type":"string"},"path":{"type":"string"}},"type":"object"},"description":"Run a command","name":"run"},"type":"function"}],"messages":[{"role":"system","content":"You are an agent."},{"role":"user","content":"fix bug A"}]}"#,
        );

        let keys_a = routing_keys(Some(&req_a));
        let keys_b = routing_keys(Some(&req_b));

        assert_eq!(keys_a.prefix_hash, keys_b.prefix_hash);
        assert_eq!(keys_a.sticky_hash, keys_b.sticky_hash);
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
        let candidates = targets.candidates("qwen");
        let cached = election::InferenceTarget::Remote(id_b);
        let selection = select_model_target_from_eligible_candidates(
            &targets,
            &candidates,
            Some(&req_a),
            &affinity,
            Some(cached.clone()),
        );
        assert_eq!(selection.target, cached);
        assert_eq!(selection.cache_target, Some(cached));
    }

    #[test]
    fn test_prepare_remote_targets_has_no_unverified_cache_target() {
        let id_a = make_id(1);
        let id_b = make_id(2);
        let hosts = vec![id_a, id_b];
        let affinity = AffinityRouter::with_config(true, true);
        let req = parse_body(
            r#"{"messages":[{"role":"system","content":"You are an agent."},{"role":"user","content":"task A"}]}"#,
        );

        let prepared = prepare_remote_targets_for_request("qwen", &hosts, Some(&req), &affinity);
        assert_eq!(prepared.cache_target, None);
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
        assert_eq!(prepared.cache_target, None);
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
        assert_eq!(prepared.prefix_hash, cache_prefix_hash(Some(&req)));
        assert_eq!(prepared.cache_target, None);
    }

    #[test]
    fn the_prompt_ratio_is_learned_from_large_enough_replies() {
        let affinity = AffinityRouter::new();
        assert_eq!(affinity.prompt_bytes_per_token_milli("qwen"), None);
        // The chat template dominates a tiny prompt, so it teaches nothing.
        affinity.observe_prompt_size("qwen", 400, MIN_PROMPT_SIZE_SAMPLE_TOKENS - 1);
        assert_eq!(affinity.prompt_bytes_per_token_milli("qwen"), None);
        affinity.observe_prompt_size("qwen", 6_300, 1_000);
        assert_eq!(affinity.prompt_bytes_per_token_milli("qwen"), Some(6_300));
        affinity.observe_prompt_size("qwen", 4_000, 1_000);
        assert_eq!(affinity.prompt_bytes_per_token_milli("qwen"), Some(5_725));
        // Absurd ratios are clamped before they enter the average.
        affinity.observe_prompt_size("other", 10, 1_000);
        assert_eq!(
            affinity.prompt_bytes_per_token_milli("other"),
            Some(request_cost::MIN_BYTES_PER_TOKEN_MILLI)
        );
    }

    #[tokio::test]
    async fn provider_confirmed_local_hit_is_selectable() {
        let node =
            crate::mesh::Node::new_for_tests(crate::mesh::NodeRole::Host { http_port: 9337 })
                .await
                .expect("test node");
        let candidates = vec![remote(1), election::InferenceTarget::Local(9337)];

        node.record_local_cache_hit("qwen", 0xfeed_beef, 512, 24, 0);
        node.observe_local_prefill_cost("qwen", Some(1_000));

        // 512 of 1,024 tokens credited: 0.512 s warm plus the 256-token
        // margin beats 1.024 s cold; the remote has no rate to compare.
        assert_eq!(
            node.select_cache_target(
                "qwen",
                0xfeed_beef,
                &candidates,
                request_cost::RequestCostInput::heuristic(1_024, 512),
            )
            .await,
            Some(election::InferenceTarget::Local(9337))
        );
    }

    #[tokio::test]
    async fn an_unmeasured_local_hit_does_not_override_routing() {
        let node =
            crate::mesh::Node::new_for_tests(crate::mesh::NodeRole::Host { http_port: 9337 })
                .await
                .expect("test node");
        let candidates = vec![remote(1), election::InferenceTarget::Local(9337)];

        // A hit with no prefill rate cannot be costed, so it is no evidence of
        // an advantage, and a request sharing too little earns no credit.
        node.record_local_cache_hit("qwen", 0xfeed_beef, 512, 24, 0);
        let request = request_cost::RequestCostInput::heuristic(1_024, 512);
        assert_eq!(
            node.select_cache_target("qwen", 0xfeed_beef, &candidates, request)
                .await,
            None
        );
        node.observe_local_prefill_cost("qwen", Some(1_000));
        let short_shared = request_cost::RequestCostInput::heuristic(1_024, MIN_SAVED_TOKENS - 1);
        assert_eq!(
            node.select_cache_target("qwen", 0xfeed_beef, &candidates, short_shared)
                .await,
            None
        );
    }

    #[tokio::test]
    async fn provider_confirmed_local_miss_invalidates_evidence() {
        let node =
            crate::mesh::Node::new_for_tests(crate::mesh::NodeRole::Host { http_port: 9337 })
                .await
                .expect("test node");
        let candidates = vec![remote(1), election::InferenceTarget::Local(9337)];

        node.record_local_cache_hit("qwen", 0xfeed_beef, 512, 24, 0);
        node.observe_local_prefill_cost("qwen", Some(1_000));
        assert!(node.invalidate_local_cache_evidence("qwen", 0xfeed_beef));

        assert_eq!(
            node.select_cache_target(
                "qwen",
                0xfeed_beef,
                &candidates,
                request_cost::RequestCostInput::heuristic(1_024, 512),
            )
            .await,
            None
        );
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
    fn explicit_auto_model_session_key_matches_sticky_hash() {
        let body = parse_body(
            r#"{"user":"sess-1","messages":[{"role":"system","content":"be helpful"},{"role":"user","content":"hi"}]}"#,
        );
        let key = auto_model_session_key(Some(&body)).expect("expected a session key");
        let sticky = routing_keys(Some(&body)).sticky_hash.unwrap();
        assert_eq!(key, sticky);
    }

    #[test]
    fn prompt_cache_key_remains_an_auto_model_session_key() {
        let body = parse_body(
            r#"{"prompt_cache_key":"cache-1","messages":[{"role":"user","content":"hi"}]}"#,
        );
        let routing = routing_keys(Some(&body));

        assert_eq!(auto_model_session_key(Some(&body)), routing.prefix_hash);
        assert!(routing.prefix_hash.is_some());
        assert_eq!(routing.sticky_hash, None);
    }

    #[test]
    fn auto_model_cache_disabled_when_sticky_disabled() {
        let affinity = AffinityRouter::with_config(true, false);
        affinity.remember_auto_model(1, "model");
        assert_eq!(affinity.lookup_auto_model(1), None);
    }

    #[test]
    fn auto_model_cache_is_independent_of_cache_evidence() {
        let affinity = AffinityRouter::new();
        affinity.remember_auto_model(7, "chat-model");
        assert_eq!(
            affinity.lookup_auto_model(7),
            Some("chat-model".to_string())
        );
    }
}
