use super::context_selection;
use crate::inference::election;
use crate::mesh;
use mesh_mixture_of_agents as moa;

/// MoA's opinionated default: workers do not think unless the caller
/// explicitly asks for it. Workers are short-budget internal slots, not
/// user-facing reasoning steps. The fast worker's 256-token budget is
/// far too small to fit `<think>…</think>` + answer, and the reducer
/// doesn't want reasoning prose as candidate input.
///
/// The caller can still explicitly enable thinking (e.g. for
/// experimentation) via any of the recognised knobs — see
/// [`extract_enable_thinking_override`]. When no preference is
/// expressed, MoA picks for them: off (always `Some(false)`).
pub(super) fn effective_enable_thinking_for_moa(body: &serde_json::Value) -> Option<bool> {
    extract_enable_thinking_override(body).or(Some(false))
}

/// Pull the caller's "disable / enable thinking" preference out of an
/// inbound chat-completion or responses JSON body. Mirrors the same
/// shapes that `openai_frontend::common::normalize_reasoning_template_options`
/// recognises so MoA users get the same surface as direct callers.
///
/// Recognised inputs (any one is enough):
/// * `reasoning_effort: "none"` (off) or any non-`"none"` value (on)
/// * `reasoning: { enabled: false }` (off) / `{ enabled: true }` (on)
/// * `reasoning: { effort: "none" }` / `{ max_tokens: 0 }` (off)
/// * Any of `THINKING_BOOLEAN_ALIASES` as a top-level field with bool
/// * `thinking_budget: 0` (off)
/// * `chat_template_kwargs.enable_thinking` (or any alias) as bool
///
/// Returns `None` when the caller hasn't expressed a preference. The
/// MoA-specific policy layer in [`effective_enable_thinking_for_moa`]
/// turns that `None` into `Some(false)` so MoA workers default off.
fn extract_enable_thinking_override(body: &serde_json::Value) -> Option<bool> {
    let obj = body.as_object()?;
    let mut result: Option<bool> = None;

    // reasoning: { enabled, effort, max_tokens }
    if let Some(r) = obj.get("reasoning").and_then(|v| v.as_object()) {
        if r.get("enabled") == Some(&serde_json::Value::Bool(false))
            || r.get("effort").and_then(|v| v.as_str()) == Some("none")
            || r.get("max_tokens").and_then(|v| v.as_u64()) == Some(0)
        {
            result = Some(false);
        } else if r.get("enabled") == Some(&serde_json::Value::Bool(true))
            || r.get("effort").is_some()
            || r.get("max_tokens").is_some()
        {
            result = Some(true);
        }
    }

    // reasoning_effort: "none" / "low" / etc.
    if let Some(effort) = obj.get("reasoning_effort").and_then(|v| v.as_str()) {
        result = Some(effort != "none");
    }

    // Top-level boolean aliases (enable_thinking, enable_reasoning, etc.).
    for alias in openai_frontend::common::THINKING_BOOLEAN_ALIASES {
        if let Some(b) = obj.get(*alias).and_then(|v| v.as_bool()) {
            result = Some(b);
        }
    }

    if obj.get("thinking_budget").and_then(|v| v.as_u64()) == Some(0) {
        result = Some(false);
    }

    // chat_template_kwargs.{enable_thinking, ...}
    if let Some(kwargs) = obj.get("chat_template_kwargs").and_then(|v| v.as_object()) {
        for alias in openai_frontend::common::THINKING_BOOLEAN_ALIASES {
            if let Some(b) = kwargs.get(*alias).and_then(|v| v.as_bool()) {
                result = Some(b);
            }
        }
    }

    result
}

/// Build a MoA gateway config from this node's mesh-wide view.
///
/// Every distinct model in the mesh becomes a worker:
/// - Models served by this node → `LocalModelBackend` (direct skippy port)
/// - Models served by a peer → `RemoteModelBackend` (QUIC tunnel)
///
/// Models are deduplicated by canonical base name so e.g.
/// `unsloth/Qwen3-8B-GGUF:Q4_K_M` and `Qwen3-8B-Q4_K_M` (different naming
/// conventions for the same model from different peers) only show up once.
///
/// Returns `None` if fewer than 2 distinct models exist — MoA needs at
/// least two workers to be meaningfully different from a single call.
///
/// `targets` is the runtime's local routing table, used to discover the
/// skippy port for locally-served models. In passive (`--client`) mode
/// this is `None` — every backend goes over QUIC. In `serve` mode it's
/// `Some`, so locally-served models bypass the tunnel.
pub async fn build_moa_config(
    node: &mesh::Node,
    targets: Option<&election::ModelTargets>,
    required_tokens: Option<u32>,
) -> Option<moa::GatewayConfig> {
    let http = reqwest::Client::new();
    let mut backends: Vec<std::sync::Arc<dyn moa::ModelBackend>> = Vec::new();
    let mut models: Vec<moa::ModelEntry> = Vec::new();
    let mut local_count = 0usize;

    // Full mesh-wide model list (local + every peer's advertised
    // routable models).
    let all_models: Vec<String> = node
        .models_being_served()
        .await
        .into_iter()
        .filter(|n| n != moa::VIRTUAL_MODEL_NAME)
        .collect();

    // Group aliases by canonical base. The old shape sorted by name
    // length, took the *first* alias per base, and dropped the rest —
    // which silently dropped the model from the worker pool whenever the
    // shortest-named peer was unreachable (regression flagged by PR #566
    // review). Now we keep every alias per base and try them in order so
    // a longer-named reachable alias can still resolve when the shortest
    // one is offline.
    let groups = group_aliases_by_canonical_base(all_models, targets);
    for aliases in groups {
        resolve_one_worker_from_aliases(
            node,
            targets,
            &http,
            &aliases,
            required_tokens,
            &mut backends,
            &mut models,
            &mut local_count,
        )
        .await;
    }

    if models.len() < 2 {
        tracing::warn!(
            "MoA: only {} model(s) reachable, need ≥2 (models={:?})",
            models.len(),
            models.iter().map(|m| &m.name).collect::<Vec<_>>()
        );
        return None;
    }

    tracing::info!(
        required_tokens = ?required_tokens,
        "MoA config: {} workers ({} local, {} remote): {:?}",
        models.len(),
        local_count,
        models.len() - local_count,
        models.iter().map(|m| m.name.as_str()).collect::<Vec<_>>(),
    );

    Some(moa::GatewayConfig {
        backends,
        models,
        // Bumped from 15s → 60s. 15s was tight for big-context interactive
        // turns: a large model with a 10–20k-token prompt and tool schema
        // (typical for agent harnesses like OpenCode/Goose) can need 20–30s
        // just to produce a first tool-call. Workers were getting killed
        // mid-inference and MoA reported `kind=early-exit` with the small
        // worker, never the strong one. 60s gives the strong worker room
        // to land without making the no-progress wait painful.
        worker_timeout: std::time::Duration::from_secs(60),
        // Per-attempt cap; hedged_reducer_call hedges across candidates so the
        // end-to-end wait is roughly reducer_timeout + a couple of hedge delays.
        reducer_timeout: std::time::Duration::from_secs(60),
        // Start a second reducer candidate after 5s if the first hasn't replied
        // (or sooner on outright failure). Cheap on the happy path, big win on
        // the cold-KV / stale-peer tail.
        hedge_delay: std::time::Duration::from_secs(5),
        // Chat-only grace: after this long since dispatch, if at least
        // one qualifying Answer is in hand we ship the highest-confidence
        // one. Tool turns bypass this entirely (consensus continues to
        // arbitrate tool proposals).
        //
        // 3 seconds is empirically good across the public mesh today.
        // Long enough that slow-but-good workers (studio MiniMax
        // landing at ~1s, mini Qwen3.5 at ~700ms) finish before the
        // timer; short enough that chat doesn't sit on a multi-second
        // ceiling on every turn. Lab data: median mesh_chat dropped
        // from ~6s (old default) to ~2s with this value, no quality
        // regression measured on factual / arithmetic / short-creative
        // prompts.
        //
        // The previous 6s was conservative because the original grace
        // logic only armed on a sole answer — it had to wait for a
        // second non-matching answer to arrive before becoming useless.
        // With the relaxed eligibility added in this change, the timer
        // is the dominant chat path, so a tighter default is the right
        // default.
        first_answer_grace: std::time::Duration::from_secs(3),
        // Tier-gate patience: how long small-tier-only answers/consensus
        // are held when a big-tier strong worker (e.g. MiniMax) is still
        // running. 20s covers the strong worker's typical first-token
        // latency on agent-sized prompts over the public mesh without
        // approaching worker_timeout (60s). Hard-bounded: at expiry all
        // decision rules revert to ungated behavior. Same-tier pools are
        // unaffected, so "many small models lift each other" keeps its
        // current latency profile.
        strong_patience: std::time::Duration::from_secs(20),
        // Defaults to leaving each model's thinking behavior alone.
        // `try_handle_moa` overrides this from the inbound request body
        // when the caller has expressed a preference
        // (`reasoning_effort: "none"`, `enable_thinking: false`, etc.).
        enable_thinking: None,
    })
}

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

/// Canonical name used for cross-peer dedup. Different peers advertise the
/// same model under different conventions (`unsloth/Qwen3-8B-GGUF:Q4_K_M`
/// vs `Qwen3-8B-Q4_K_M`); normalize before comparing.
///
/// Strategy: strip the publisher prefix, the `-gguf` suffix, any `@branch`
/// suffix, then keep only `[a-z0-9]` characters so `:` vs `-` separators
/// don't matter.
fn canonical_base_name(name: &str) -> String {
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

/// Backend that calls a local model directly on its skippy HTTP port.
struct LocalModelBackend {
    port: u16,
    http: reqwest::Client,
}

#[async_trait::async_trait]
impl moa::ModelBackend for LocalModelBackend {
    async fn chat_completion(
        &self,
        model: &str,
        messages: &[serde_json::Value],
        tools: Option<&serde_json::Value>,
        max_tokens: u32,
        timeout: std::time::Duration,
        sampling: moa::SamplingParams,
    ) -> Result<serde_json::Value, String> {
        let url = format!("http://127.0.0.1:{}/v1/chat/completions", self.port);
        let mut body = serde_json::json!({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "stream": false,
            "mesh_hooks": false,
        });
        if let Some(tools) = tools {
            body.as_object_mut()
                .unwrap()
                .insert("tools".to_string(), tools.clone());
        }
        moa::apply_enable_thinking(&mut body, sampling.enable_thinking);
        let resp = self
            .http
            .post(&url)
            .json(&body)
            .timeout(timeout)
            .send()
            .await
            .map_err(|e| format!("local:{} failed: {e}", self.port))?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(format!(
                "HTTP {status}: {}",
                moa::truncate_chars(&text, 200)
            ));
        }
        resp.json::<serde_json::Value>()
            .await
            .map_err(|e| format!("parse: {e}"))
    }
}

/// Backend that calls a remote model over the QUIC tunnel.
struct RemoteModelBackend {
    node: mesh::Node,
    peer_id: iroh::EndpointId,
}

#[async_trait::async_trait]
impl moa::ModelBackend for RemoteModelBackend {
    async fn chat_completion(
        &self,
        model: &str,
        messages: &[serde_json::Value],
        tools: Option<&serde_json::Value>,
        max_tokens: u32,
        timeout: std::time::Duration,
        sampling: moa::SamplingParams,
    ) -> Result<serde_json::Value, String> {
        let mut body = serde_json::json!({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "stream": false,
            "mesh_hooks": false,
        });
        if let Some(tools) = tools {
            body.as_object_mut()
                .unwrap()
                .insert("tools".to_string(), tools.clone());
        }
        moa::apply_enable_thinking(&mut body, sampling.enable_thinking);
        let body_bytes = serde_json::to_vec(&body).map_err(|e| format!("serialize: {e}"))?;
        let http_request = format!(
            "POST /v1/chat/completions HTTP/1.1\r\n\
             Host: localhost\r\n\
             Content-Type: application/json\r\n\
             Content-Length: {}\r\n\
             \r\n",
            body_bytes.len()
        );
        let mut raw = http_request.into_bytes();
        raw.extend_from_slice(&body_bytes);

        tokio::time::timeout(timeout, async {
            let (mut send, mut recv) = self
                .node
                .open_http_tunnel(self.peer_id)
                .await
                .map_err(|e| format!("tunnel: {e}"))?;
            send.write_all(&raw)
                .await
                .map_err(|e| format!("send: {e}"))?;
            send.finish().map_err(|e| format!("finish: {e}"))?;
            let response = recv
                .read_to_end(4 * 1024 * 1024)
                .await
                .map_err(|e| format!("recv: {e}"))?;
            parse_quic_http_response(&response)
        })
        .await
        .map_err(|_| format!("remote timeout after {}s", timeout.as_secs()))?
    }
}

fn parse_quic_http_response(response: &[u8]) -> Result<serde_json::Value, String> {
    let s = String::from_utf8_lossy(response);
    let header_end = s
        .find("\r\n\r\n")
        .ok_or_else(|| "malformed HTTP response".to_string())?;
    let status_line = s[..header_end].lines().next().unwrap_or("");
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    if status != 200 {
        return Err(format!("HTTP {status}: {}", moa::truncate_chars(&s, 200)));
    }
    let body = &s[header_end + 4..];
    serde_json::from_str(body).map_err(|e| format!("parse: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

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
    // ── extract_enable_thinking_override ────────────────────────────────
    //
    // Mirrors the shapes that `openai_frontend::common::normalize_reasoning_template_options`
    // accepts, so MoA users get the same surface as direct callers. If we
    // forget a shape, the model never gets told to stop thinking and the
    // fast worker burns its budget inside `<think>`.

    #[test]
    fn extract_no_knobs_returns_none() {
        let body = serde_json::json!({"model": "mesh", "messages": []});
        assert_eq!(extract_enable_thinking_override(&body), None);
    }

    #[test]
    fn extract_reasoning_effort_none_disables() {
        let body = serde_json::json!({"reasoning_effort": "none"});
        assert_eq!(extract_enable_thinking_override(&body), Some(false));
    }

    #[test]
    fn extract_reasoning_effort_low_enables() {
        let body = serde_json::json!({"reasoning_effort": "low"});
        assert_eq!(extract_enable_thinking_override(&body), Some(true));
    }

    #[test]
    fn extract_reasoning_enabled_false_disables() {
        let body = serde_json::json!({"reasoning": {"enabled": false}});
        assert_eq!(extract_enable_thinking_override(&body), Some(false));
    }

    #[test]
    fn extract_reasoning_max_tokens_zero_disables() {
        let body = serde_json::json!({"reasoning": {"max_tokens": 0}});
        assert_eq!(extract_enable_thinking_override(&body), Some(false));
    }

    #[test]
    fn extract_top_level_enable_thinking_false() {
        let body = serde_json::json!({"enable_thinking": false});
        assert_eq!(extract_enable_thinking_override(&body), Some(false));
    }

    #[test]
    fn extract_top_level_enable_thinking_alias() {
        // `use_thinking` is one of THINKING_BOOLEAN_ALIASES.
        let body = serde_json::json!({"use_thinking": false});
        assert_eq!(extract_enable_thinking_override(&body), Some(false));
    }

    #[test]
    fn extract_thinking_budget_zero_disables() {
        let body = serde_json::json!({"thinking_budget": 0});
        assert_eq!(extract_enable_thinking_override(&body), Some(false));
    }

    #[test]
    fn extract_chat_template_kwargs_passes_through() {
        let body = serde_json::json!({
            "chat_template_kwargs": {"enable_thinking": false}
        });
        assert_eq!(extract_enable_thinking_override(&body), Some(false));
    }

    #[test]
    fn extract_latest_wins_when_multiple_set() {
        // chat_template_kwargs is read last and so wins. Whatever ordering
        // we choose, picking ONE consistently is the contract.
        let body = serde_json::json!({
            "reasoning_effort": "low",                                  // enable
            "chat_template_kwargs": {"enable_thinking": false},         // disable
        });
        assert_eq!(extract_enable_thinking_override(&body), Some(false));
    }

    // ── MoA opinionated default ────────────────────────────────────────────────────
    //
    // For `model: "mesh"`, MoA does NOT let reasoning models think on
    // worker slots. The fast worker has a 256-token budget that doesn't
    // fit `<think>...</think>` + answer, and the reducer doesn't want
    // reasoning prose as candidate input. Callers can explicitly turn
    // reasoning back on, but the default is off.

    #[test]
    fn effective_default_is_no_thinking_when_caller_silent() {
        // No knobs in the body → MoA's opinion applies.
        let body = serde_json::json!({"model": "mesh", "messages": []});
        assert_eq!(effective_enable_thinking_for_moa(&body), Some(false));
    }

    #[test]
    fn effective_respects_explicit_disable_from_caller() {
        let body = serde_json::json!({
            "reasoning_effort": "none",
            "model": "mesh",
        });
        assert_eq!(effective_enable_thinking_for_moa(&body), Some(false));
    }

    #[test]
    fn effective_lets_caller_explicitly_enable_thinking() {
        // Escape hatch: a caller who really wants reasoning on MoA can
        // ask for it via any of the recognised knobs.
        let body = serde_json::json!({
            "reasoning_effort": "low",
            "model": "mesh",
        });
        assert_eq!(effective_enable_thinking_for_moa(&body), Some(true));
    }

    #[test]
    fn effective_default_for_tool_calling_request_still_no_thinking() {
        // Agentic / tool turns get the same opinionated default.
        // The grace-bypass / consensus path in MoA already runs
        // differently for tool turns, but thinking is independent of
        // that and should still be off unless the caller insists.
        let body = serde_json::json!({
            "model": "mesh",
            "messages": [],
            "tools": [{"type": "function", "function": {"name": "x"}}],
        });
        assert_eq!(effective_enable_thinking_for_moa(&body), Some(false));
    }
}
