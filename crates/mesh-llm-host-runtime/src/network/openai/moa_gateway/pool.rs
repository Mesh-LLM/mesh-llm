//! MoA worker-pool assembly.
//!
//! Owns the discovery-and-assembly side of the MoA gateway: turning the
//! node's mesh-wide model view into a concrete `(backends, models)` worker
//! pool. `build_moa_config` (in [`super::workers`]) is the orchestrator that
//! calls [`assemble_worker_pool`] and [`compute_actor_candidates`] here.

use super::context_selection;
use super::workers::{LocalModelBackend, RemoteModelBackend};
use crate::inference::election;
use crate::mesh;
use mesh_mixture_of_agents as moa;

/// Try each alias in `aliases` until one resolves to a backend, then stop.
///
/// Aliases are pre-sorted by `group_aliases_by_canonical_base` so the most
/// preferred (locally-served first, then shortest) is tried first. Falls
/// back to longer aliases when the preferred one's peer is unreachable.
#[allow(clippy::too_many_arguments)]
async fn resolve_one_worker_from_aliases(
    node: &mesh::Node,
    targets: Option<&election::ModelTargets>,
    http: &reqwest::Client,
    aliases: &[String],
    required_tokens: Option<u32>,
    backends: &mut Vec<std::sync::Arc<dyn moa::ModelBackend>>,
    models: &mut Vec<moa::ModelEntry>,
    local_count: &mut usize,
) {
    let resolution = WorkerBackendResolution {
        node,
        targets,
        http,
        required_tokens,
    };
    for name in aliases {
        if add_worker_backend(&resolution, name, backends, models, local_count).await {
            return;
        }
    }
}

/// Group all advertised model names by their canonical base so each
/// canonical model contributes exactly one worker, but the resolver gets
/// to pick the alias that actually has a reachable backend.
///
/// The earlier shape committed to a single alias per base *before* trying
/// to resolve a backend. Two failure modes:
///
///   1. The chosen alias is advertised only by a peer that drops between
///      gossip refresh and orchestration — `hosts_for_model` returns
///      empty, the worker is dropped, and longer-form aliases for the
///      same canonical model from still-reachable peers are rejected as
///      duplicates.
///   2. The local node advertises a longer convention
///      (e.g. `unsloth/Qwen3-8B-GGUF:Q4_K_M`) while a peer advertises a
///      shorter variant (e.g. `Qwen3-8B-Q4_K_M`). The shortest-name rule
///      picks the peer alias, `add_worker_backend` looks for a local port
///      under that specific string, finds nothing, and forces a
///      QUIC-tunnel backend even though the model is right here.
///
/// Both failure modes are fixed by grouping first and resolving second.
/// Within each group the aliases are ordered so the most likely
/// optimization wins first try: locally-served name (skippy-port fast
/// path) before remote names, then shortest first as a tiebreaker.
fn group_aliases_by_canonical_base(
    names: Vec<String>,
    targets: Option<&election::ModelTargets>,
) -> Vec<Vec<String>> {
    let mut by_base: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    for name in names {
        by_base
            .entry(canonical_base_name(&name))
            .or_default()
            .push(name);
    }
    // Deterministic group order so the worker list is stable across
    // builds even though HashMap iteration is not. Sort group entries
    // (locally-served first, then shortest), then sort groups by their
    // first ("best") alias.
    let mut groups: Vec<Vec<String>> = by_base
        .into_values()
        .map(|mut aliases| {
            aliases.sort_by(|a, b| {
                let la = is_locally_served(a, targets);
                let lb = is_locally_served(b, targets);
                lb.cmp(&la) // local (true) before remote (false)
                    .then_with(|| a.len().cmp(&b.len()))
                    .then_with(|| a.cmp(b))
            });
            aliases
        })
        .collect();
    groups.sort_by(|a, b| a[0].cmp(&b[0]));
    groups
}

/// Does the local routing table have a backend port for this exact name?
fn is_locally_served(name: &str, targets: Option<&election::ModelTargets>) -> bool {
    targets
        .and_then(|t| {
            t.targets.get(name).map(|tv| {
                tv.iter()
                    .any(|t| matches!(t, election::InferenceTarget::Local(_)))
            })
        })
        .unwrap_or(false)
}

/// Resolve `name` to a backend (local skippy port if available, else first
/// remote host) and append it to `backends`/`models`. Returns true if a
/// backend was added.
struct WorkerBackendResolution<'a> {
    node: &'a mesh::Node,
    targets: Option<&'a election::ModelTargets>,
    http: &'a reqwest::Client,
    required_tokens: Option<u32>,
}

async fn add_worker_backend(
    resolution: &WorkerBackendResolution<'_>,
    name: &str,
    backends: &mut Vec<std::sync::Arc<dyn moa::ModelBackend>>,
    models: &mut Vec<moa::ModelEntry>,
    local_count: &mut usize,
) -> bool {
    // Prefer local skippy port when this node serves the model.
    let local_port = resolution.targets.and_then(|t| {
        t.targets.get(name).and_then(|tv| {
            tv.iter().find_map(|t| match t {
                election::InferenceTarget::Local(p) => Some(*p),
                _ => None,
            })
        })
    });
    if let Some(port) = local_port {
        let context_length = resolution.node.local_model_context_length(name).await;
        if context_selection::context_can_satisfy(resolution.required_tokens, context_length) {
            let backend_idx = backends.len();
            backends.push(std::sync::Arc::new(LocalModelBackend {
                port,
                http: resolution.http.clone(),
            }));
            models.push(moa::ModelEntry {
                name: name.to_string(),
                backend_index: backend_idx,
            });
            *local_count += 1;
            return true;
        } else {
            tracing::info!(
                "MoA: skipping local worker {name}; context {:?} cannot fit {:?} required tokens",
                context_length,
                resolution.required_tokens
            );
        }
    }

    // Otherwise find a remote host. hosts_for_model returns peers in
    // hash-preferred order; prefer hosts with enough advertised context.
    let remote_hosts = resolution.node.hosts_for_model(name).await;
    if let Some(peer_id) = context_selection::select_remote_host(
        resolution.node,
        name,
        resolution.required_tokens,
        remote_hosts,
    )
    .await
    {
        let backend_idx = backends.len();
        backends.push(std::sync::Arc::new(RemoteModelBackend {
            node: resolution.node.clone(),
            peer_id,
        }));
        models.push(moa::ModelEntry {
            name: name.to_string(),
            backend_index: backend_idx,
        });
        return true;
    }
    false
}

/// Discover and assemble the MoA worker pool: resolve one worker per distinct
/// model, apply admission control, then self-fill same-model instances.
///
/// Returns parallel `(backends, models)` vecs linked by `backend_index`.
pub(super) async fn assemble_worker_pool(
    node: &mesh::Node,
    targets: Option<&election::ModelTargets>,
    required_tokens: Option<u32>,
    http: &reqwest::Client,
) -> (
    Vec<std::sync::Arc<dyn moa::ModelBackend>>,
    Vec<moa::ModelEntry>,
) {
    let mut backends: Vec<std::sync::Arc<dyn moa::ModelBackend>> = Vec::new();
    let mut models: Vec<moa::ModelEntry> = Vec::new();
    let mut local_count = 0usize;

    // Full mesh-wide model list (local + every peer's advertised routable
    // models).
    let all_models: Vec<String> = node
        .models_being_served()
        .await
        .into_iter()
        .filter(|n| n != moa::VIRTUAL_MODEL_NAME)
        .collect();

    // Group aliases by canonical base and resolve one worker per base, trying
    // aliases in order so a longer-named reachable alias still resolves when
    // the shortest one is offline (PR #566).
    for aliases in group_aliases_by_canonical_base(all_models, targets) {
        resolve_one_worker_from_aliases(
            node,
            targets,
            http,
            &aliases,
            required_tokens,
            &mut backends,
            &mut models,
            &mut local_count,
        )
        .await;
    }

    // Admission control: a weak worker must not drag down a pool that already
    // has a stronger one. Aggregation is sensitive to proposal quality
    // (Self-MoA, arXiv:2502.00674), so an 8B draft added to a 24-32B pool is
    // expected noise-to-harm. When tiers are mixed, keep only big-tier workers;
    // an all-small or all-big pool is untouched. A lone big model then serves
    // solo (fails the caller's <2 check), the safe outcome.
    apply_admission_control(&mut backends, &mut models);

    // Same-model fill: if only one model resolved but it is served by more than
    // one node, add the extra nodes as workers. Self-MoA shows repeated
    // sampling of one model ensembles as well as different models, so a mesh
    // where every node serves the SAME model should still get MoA — the "add a
    // modest node and it helps" case. Distinct remote endpoints only, never the
    // same node twice, so the added capacity is real.
    if models.len() == 1 {
        self_fill_from_extra_instances(node, http, &mut backends, &mut models).await;
    }

    // Committee cap: fan-out cost is ~2N+1 model calls per turn (N drafts + N
    // refines + 1 synthesis), and measured quality is flat past ~4 workers
    // while latency and spend keep climbing. On a big shared mesh (say 20
    // nodes) an uncapped pool would fan out to all of them — 41 calls for no
    // quality gain. Keep the best MAX_COMMITTEE_WORKERS by capability ranking;
    // the rest are standbys (they still serve direct traffic, just not this
    // committee).
    cap_committee(node, &mut backends, &mut models).await;

    (backends, models)
}

/// Largest committee we will fan out to. Measured quality is flat past ~4
/// diverse workers (evals/moa-openrouter/RESULTS.md), so more than this is pure
/// latency/cost. A big shared mesh has standbys beyond this, not bigger turns.
const MAX_COMMITTEE_WORKERS: usize = 4;

/// Trim the pool to the best [`MAX_COMMITTEE_WORKERS`] by capability ranking.
async fn cap_committee(
    node: &mesh::Node,
    backends: &mut Vec<std::sync::Arc<dyn moa::ModelBackend>>,
    models: &mut Vec<moa::ModelEntry>,
) {
    if models.len() <= MAX_COMMITTEE_WORKERS {
        return;
    }
    // Rank strongest-first (gossiped tool_use, then size, then stable index),
    // reusing the actor-selection ranking, and keep the top N.
    let ranked = compute_actor_candidates(node, models).await;
    let keep: std::collections::HashSet<usize> =
        ranked.into_iter().take(MAX_COMMITTEE_WORKERS).collect();

    let mut kept_backends: Vec<std::sync::Arc<dyn moa::ModelBackend>> = Vec::new();
    let mut kept_models: Vec<moa::ModelEntry> = Vec::new();
    for (i, m) in models.iter().enumerate() {
        if !keep.contains(&i) {
            tracing::info!("MoA: capping committee, dropping worker {}", m.name);
            continue;
        }
        let new_idx = kept_backends.len();
        kept_backends.push(backends[m.backend_index].clone());
        kept_models.push(moa::ModelEntry {
            name: m.name.clone(),
            backend_index: new_idx,
        });
    }
    *backends = kept_backends;
    *models = kept_models;
}

/// Cap on same-model instances added by self-fill. Two is enough to switch a
/// single-model mesh from solo to a working committee; beyond that the extra
/// draft's marginal value falls and it is just latency/cost.
const SELF_FILL_TARGET_WORKERS: usize = 2;

/// When only one model resolved, add extra reachable *nodes* serving that same
/// model as additional workers, up to [`SELF_FILL_TARGET_WORKERS`].
///
/// Only genuinely distinct remote endpoints are added — never the local backend
/// again and never the same peer twice — so each added worker is real capacity
/// from a node that joined the mesh. This is what makes a same-model mesh get
/// MoA at all; without it `build_moa_config` returns None for such a mesh.
async fn self_fill_from_extra_instances(
    node: &mesh::Node,
    _http: &reqwest::Client,
    backends: &mut Vec<std::sync::Arc<dyn moa::ModelBackend>>,
    models: &mut Vec<moa::ModelEntry>,
) {
    let Some(existing) = models.first().cloned() else {
        return;
    };
    let name = existing.name.clone();

    for peer_id in node.hosts_for_model(&name).await {
        if models.len() >= SELF_FILL_TARGET_WORKERS {
            break;
        }
        let backend_idx = backends.len();
        backends.push(std::sync::Arc::new(RemoteModelBackend {
            node: node.clone(),
            peer_id,
        }));
        models.push(moa::ModelEntry {
            name: name.clone(),
            backend_index: backend_idx,
        });
        tracing::info!("MoA: self-fill added instance of {name} from peer {peer_id}");
    }
}

/// Drop small-tier workers when any big-tier worker is present.
///
/// A weak draft can contaminate synthesis, and aggregation quality tracks
/// proposal quality (Self-MoA, arXiv:2502.00674), so a modest node must not be
/// admitted into a committee that already has a stronger member. When the pool
/// is mixed we keep only the big-tier workers; an all-small or all-big pool is
/// left untouched. `backends` and `models` are parallel vecs linked by
/// `backend_index`, so we rebuild both and reindex.
fn apply_admission_control(
    backends: &mut Vec<std::sync::Arc<dyn moa::ModelBackend>>,
    models: &mut Vec<moa::ModelEntry>,
) {
    let big_count = models
        .iter()
        .filter(|m| !moa::model_name_is_small_tier(&m.name))
        .count();
    let has_small = models
        .iter()
        .any(|m| moa::model_name_is_small_tier(&m.name));
    // Only exclude small-tier workers when doing so still leaves a committee
    // (>=2 big-tier). Measured:
    //   * 32B x2 + 8B  -> dropping the 8B leaves 32B x2, and the 8B added
    //     nothing (arm C: no upside, losses 2->5) — so drop it.
    //   * 32B + 8B     -> dropping the 8B collapses to a solo 32B, but the
    //     mixed committee beats solo decisively (47W/27T/5L, p=1e-9) — so
    //     KEEP the 8B. Admission must not throw away MoA to protect a pool
    //     that no longer exists.
    // See `evals/moa-openrouter/RESULTS.md`.
    if !(has_small && big_count >= 2) {
        return;
    }

    let mut kept_backends: Vec<std::sync::Arc<dyn moa::ModelBackend>> = Vec::new();
    let mut kept_models: Vec<moa::ModelEntry> = Vec::new();
    for m in models.iter() {
        if moa::model_name_is_small_tier(&m.name) {
            tracing::info!(
                "MoA: excluding small-tier worker {} (big-tier present)",
                m.name
            );
            continue;
        }
        let new_idx = kept_backends.len();
        kept_backends.push(backends[m.backend_index].clone());
        kept_models.push(moa::ModelEntry {
            name: m.name.clone(),
            backend_index: new_idx,
        });
    }
    *backends = kept_backends;
    *models = kept_models;
}

/// Rank the pool best-tool-caller-first (indices into `models`) for the actor.
///
/// Ordering: gossiped `tool_use` (`Supported` > `Likely` > `None`), then size
/// tier, then stable index. Capabilities match pool entries by canonical base
/// name (so `unsloth/Qwen3-8B-GGUF:Q4_K_M` supplies `Qwen3-8B-Q4_K_M`). Always
/// returns a full ranking; the engine reads an empty vec as "no host guidance".
pub(super) async fn compute_actor_candidates(
    node: &mesh::Node,
    models: &[moa::ModelEntry],
) -> Vec<usize> {
    // canonical base name -> best tool_use level seen across the mesh.
    let mut tool_use_by_base: std::collections::HashMap<String, crate::models::CapabilityLevel> =
        std::collections::HashMap::new();
    for descriptor in node.all_served_model_descriptors().await {
        let base = canonical_base_name(&descriptor.identity.model_name);
        let level = descriptor.capabilities.tool_use;
        tool_use_by_base
            .entry(base)
            .and_modify(|existing| {
                if level > *existing {
                    *existing = level;
                }
            })
            .or_insert(level);
    }

    let mut ranked: Vec<usize> = (0..models.len()).collect();
    ranked.sort_by(|&a, &b| {
        let ma = &models[a];
        let mb = &models[b];
        let tool_a = tool_use_by_base
            .get(&canonical_base_name(&ma.name))
            .copied()
            .unwrap_or(crate::models::CapabilityLevel::None);
        let tool_b = tool_use_by_base
            .get(&canonical_base_name(&mb.name))
            .copied()
            .unwrap_or(crate::models::CapabilityLevel::None);
        // 1) higher tool_use first
        tool_b
            .cmp(&tool_a)
            // 2) big-tier before small-tier
            .then_with(|| {
                let small_a = moa::model_name_is_small_tier(&ma.name);
                let small_b = moa::model_name_is_small_tier(&mb.name);
                small_a.cmp(&small_b) // false (big) sorts before true (small)
            })
            // 3) stable index order
            .then_with(|| a.cmp(&b))
    });
    ranked
}

/// Canonical name used for cross-peer dedup. Different peers advertise the
/// same model under different conventions (`unsloth/Qwen3-8B-GGUF:Q4_K_M`
/// vs `Qwen3-8B-Q4_K_M`); normalize before comparing.
///
/// Strategy: strip the publisher prefix, the `-gguf` suffix, any `@branch`
/// suffix, then keep only `[a-z0-9]` characters so `:` vs `-` separators
/// don't matter.
pub(super) fn canonical_base_name(name: &str) -> String {
    let lower = name.to_lowercase();
    // Drop an `@branch` segment if present, keeping anything after the
    // next `:` so quant tags survive (e.g. `repo@main:q4_k_m` → `repo:q4_k_m`).
    let no_branch = match lower.find('@') {
        Some(at) => {
            let after = &lower[at + 1..];
            let rest = after.find(':').map(|c| &after[c..]).unwrap_or("");
            format!("{}{}", &lower[..at], rest)
        }
        None => lower,
    };
    let stripped = no_branch
        .replace("-gguf", "")
        .replace("unsloth/", "")
        .replace("meshllm/", "");
    stripped
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal backend stub for admission-control tests.
    struct FakeBackend;
    #[async_trait::async_trait]
    impl moa::ModelBackend for FakeBackend {
        async fn chat_completion(
            &self,
            _model: &str,
            _messages: &[serde_json::Value],
            _tools: Option<&serde_json::Value>,
            _max_tokens: u32,
            _timeout: std::time::Duration,
            _sampling: moa::SamplingParams,
        ) -> Result<serde_json::Value, String> {
            Ok(serde_json::json!({"choices":[{"message":{"content":"x"}}]}))
        }
    }

    fn pool(
        names: &[&str],
    ) -> (
        Vec<std::sync::Arc<dyn moa::ModelBackend>>,
        Vec<moa::ModelEntry>,
    ) {
        let mut b: Vec<std::sync::Arc<dyn moa::ModelBackend>> = Vec::new();
        let mut m = Vec::new();
        for name in names {
            m.push(moa::ModelEntry {
                name: (*name).to_string(),
                backend_index: b.len(),
            });
            b.push(std::sync::Arc::new(FakeBackend));
        }
        (b, m)
    }

    #[test]
    fn admission_drops_small_when_two_big_remain() {
        // Dropping the small workers still leaves a committee (2x 32B), and the
        // small drafts add nothing there — so exclude them.
        let (mut b, mut m) = pool(&["Qwen3-32B", "Qwen3-32B", "Qwen3-8B", "Ministral-8B"]);
        apply_admission_control(&mut b, &mut m);
        assert_eq!(m.len(), 2);
        assert!(m.iter().all(|e| e.name == "Qwen3-32B"));
        assert_eq!(b.len(), 2);
        // backends stay aligned and reindexed
        assert_eq!(m[0].backend_index, 0);
        assert_eq!(m[1].backend_index, 1);
    }

    #[test]
    fn admission_keeps_mix_when_dropping_would_collapse_to_solo() {
        // One strong + one weak: dropping the 8B leaves a solo 32B, but the
        // mixed committee beats solo (47W/27T/5L) — so keep the mix.
        let (mut b, mut m) = pool(&["Qwen3-32B", "Qwen3-8B"]);
        apply_admission_control(&mut b, &mut m);
        assert_eq!(m.len(), 2, "must not collapse a lone-strong pool to solo");
    }

    #[test]
    fn admission_keeps_all_small_pool() {
        let (mut b, mut m) = pool(&["Qwen3-8B", "Llama-3.1-8B", "Ministral-8B"]);
        apply_admission_control(&mut b, &mut m);
        assert_eq!(m.len(), 3);
    }

    #[test]
    fn admission_keeps_all_big_pool() {
        let (mut b, mut m) = pool(&["Qwen3-32B", "Mistral-Small-24B"]);
        apply_admission_control(&mut b, &mut m);
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn admission_keeps_homogeneous_big_pool() {
        let (mut b, mut m) = pool(&["Qwen3-32B", "Qwen3-32B"]);
        apply_admission_control(&mut b, &mut m);
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn canonical_base_dedupes_unsloth_and_gguf_variants() {
        assert_eq!(
            canonical_base_name("unsloth/Qwen3-8B-GGUF:Q4_K_M"),
            canonical_base_name("Qwen3-8B-Q4_K_M")
        );
        assert_eq!(
            canonical_base_name("unsloth/Qwen3-8B-GGUF@main:Q4_K_M"),
            canonical_base_name("Qwen3-8B-Q4_K_M")
        );
    }

    #[test]
    fn canonical_base_keeps_distinct_models_distinct() {
        assert_ne!(
            canonical_base_name("unsloth/Qwen3-8B-GGUF:Q4_K_M"),
            canonical_base_name("unsloth/Qwen3-32B-GGUF:Q4_K_M")
        );
        assert_ne!(
            canonical_base_name("unsloth/Qwen3-32B-GGUF:Q4_K_M"),
            canonical_base_name("unsloth/MiniMax-M2.5-GGUF:Q4_K_M")
        );
    }
    fn make_targets(local_names: &[&str]) -> election::ModelTargets {
        let mut t = election::ModelTargets::default();
        for (i, name) in local_names.iter().enumerate() {
            t.targets.insert(
                (*name).to_string(),
                vec![election::InferenceTarget::Local(50000 + i as u16)],
            );
        }
        t
    }

    #[test]
    fn group_aliases_keeps_all_aliases_per_canonical_base() {
        // Regression for PR #566 review (item #10): the dedup-then-resolve
        // shape committed to a single alias per base before checking
        // backend reachability. Now every alias is retained so the
        // resolver can fall back if the preferred alias is unreachable.
        let groups = group_aliases_by_canonical_base(
            vec![
                "Qwen3-8B-Q4_K_M".to_string(),
                "unsloth/Qwen3-8B-GGUF:Q4_K_M".to_string(),
            ],
            None,
        );
        assert_eq!(groups.len(), 1, "both names share a canonical base");
        assert_eq!(groups[0].len(), 2, "both aliases retained");
    }

    #[test]
    fn group_aliases_prefers_locally_served_alias_even_when_longer() {
        // Without a targets table, length-order wins and the shorter peer
        // alias would be tried first — forcing an unnecessary QUIC hop
        // when the model is right here under a different alias.
        // With targets, the local-served alias must come first.
        let local = "unsloth/Qwen3-8B-GGUF:Q4_K_M";
        let peer = "Qwen3-8B-Q4_K_M";
        let targets = make_targets(&[local]);
        let groups = group_aliases_by_canonical_base(
            vec![peer.to_string(), local.to_string()],
            Some(&targets),
        );
        assert_eq!(groups.len(), 1);
        assert_eq!(
            groups[0].first().map(String::as_str),
            Some(local),
            "locally-served alias must win even though it's longer"
        );
    }

    #[test]
    fn group_aliases_falls_back_to_shortest_when_no_local() {
        // No targets table at all (pure --client --auto node) — shortest
        // alias should win, but the longer alias is still in the group so
        // it can be tried if the shortest one is unreachable.
        let groups = group_aliases_by_canonical_base(
            vec![
                "unsloth/Qwen3-8B-GGUF:Q4_K_M".to_string(),
                "Qwen3-8B-Q4_K_M".to_string(),
            ],
            None,
        );
        assert_eq!(groups.len(), 1);
        assert_eq!(
            groups[0].first().map(String::as_str),
            Some("Qwen3-8B-Q4_K_M")
        );
        assert_eq!(groups[0].len(), 2, "longer alias kept as fallback");
    }

    #[test]
    fn group_aliases_distinct_models_stay_in_separate_groups() {
        let groups = group_aliases_by_canonical_base(
            vec![
                "unsloth/Qwen3-8B-GGUF:Q4_K_M".to_string(),
                "unsloth/Qwen3-32B-GGUF:Q4_K_M".to_string(),
                "unsloth/MiniMax-M2.5-GGUF:Q4_K_M".to_string(),
            ],
            None,
        );
        assert_eq!(groups.len(), 3);
    }
}
