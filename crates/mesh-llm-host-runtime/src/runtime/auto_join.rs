use super::operational_logging::{DiscoveryOperationalEvent, record_discovery_operational_event};
use super::{
    RunAutoJoinOutcome, StartupModelPlan, join_sources, lan_rediscovery, nostr_rediscovery,
    record_first_joined_mesh_ts, run_auto_start_new_mesh, should_start_lan_rediscovery,
    start_new_mesh,
};
use super::{RuntimeOptions, RuntimeSurface};
use crate::mesh::{self};
use crate::models;
use crate::network::{discovery as mesh_discovery, nostr};
use anyhow::{Context, Result};
use mesh_llm_events::{ConsoleSessionMode, OutputEvent, emit_event};
use std::path::{Path, PathBuf};

pub(super) async fn maybe_discover_join_candidates(
    options: &mut RuntimeOptions,
    has_startup_models: bool,
    auto_join_candidates: &mut Vec<(String, Option<String>)>,
    host_ram_offload: bool,
) -> Result<()> {
    // Ask the resolver rather than `options.join`: a token that lives only in
    // a file is still a configured token, and discovery must not run over it.
    let effective_join_tokens = options.effective_join_tokens();
    let discover_active = options.auto || options.discover.is_some();
    if !discover_active || !effective_join_tokens.is_empty() {
        return Ok(());
    }

    if let Some(name) = options.discover.as_ref().filter(|name| !name.is_empty())
        && options.mesh_name.is_none()
    {
        options.mesh_name = Some(name.clone());
    }

    // Plan the model a new mesh starts with on what this node can hold without
    // spilling into RAM, the same budget as the local fit and the capacity
    // the node advertises, unless the owner opted into host-RAM offload.
    let my_vram_gb = mesh::detect_local_fit_bytes(options.max_vram, host_ram_offload) as f64 / 1e9;
    let target_name = options.mesh_name.clone();

    match options.mesh_discovery_mode {
        mesh_discovery::MeshDiscoveryMode::Nostr => {
            discover_nostr_join_candidates(
                options,
                has_startup_models,
                auto_join_candidates,
                my_vram_gb,
                target_name.clone(),
            )
            .await?;
        }
        mesh_discovery::MeshDiscoveryMode::Mdns => {
            let _ = emit_event(OutputEvent::DiscoveryStarting {
                source: mesh_discovery::discovery_source_label(
                    options.mesh_discovery_mode,
                    "auto-discovery",
                ),
            });
            let filter = nostr::MeshFilter {
                name: target_name.clone(),
                region: options.region.clone(),
                ..Default::default()
            };
            let candidates = mesh_discovery::discover_lan_join_candidates(
                &filter,
                effective_join_tokens.first().map(String::as_str),
                std::time::Duration::from_secs(5),
            )
            .await
            .inspect_err(|_| {
                record_discovery_operational_event(DiscoveryOperationalEvent::DiscoveryFailed);
            })?;

            if candidates.is_empty() {
                record_discovery_operational_event(DiscoveryOperationalEvent::DiscoveryFailed);
                let _ = emit_event(OutputEvent::DiscoveryFailed {
                    message: "No joinable LAN meshes found — mDNS requires a supplied invite token"
                        .to_string(),
                    detail: Some("Pass --join <token> or start a new LAN mesh.".to_string()),
                });
                let models = default_models_for_vram_blocking(my_vram_gb).await?;
                if options.client {
                    let _ = emit_event(OutputEvent::Info {
                        message:
                            "No joinable LAN mesh yet — starting client API; pass --join with a LAN invite token to connect"
                                .to_string(),
                        context: None,
                    });
                } else {
                    start_new_mesh(options, &models, my_vram_gb, has_startup_models);
                }
            } else {
                for (token, mesh) in candidates {
                    let _ = emit_event(OutputEvent::MeshFound {
                        mesh: mesh
                            .listing
                            .name
                            .as_deref()
                            .unwrap_or("unnamed")
                            .to_string(),
                        peers: mesh.listing.node_count,
                        region: mesh.listing.region.clone(),
                    });
                    auto_join_candidates.push((token, mesh.listing.name));
                }
            }
        }
    }

    Ok(())
}

pub(super) async fn discover_nostr_join_candidates(
    options: &mut RuntimeOptions,
    has_startup_models: bool,
    auto_join_candidates: &mut Vec<(String, Option<String>)>,
    my_vram_gb: f64,
    target_name: Option<String>,
) -> Result<()> {
    options.nostr_discovery = true;
    let _ = emit_event(OutputEvent::DiscoveryStarting {
        source: mesh_discovery::discovery_source_label(
            options.mesh_discovery_mode,
            "auto-discovery",
        ),
    });

    let relays = nostr_relays(&options.nostr_relay);
    let meshes = discover_nostr_meshes(&relays).await?;
    log_nostr_auto_candidates(&meshes, target_name.as_ref());
    handle_auto_decision(
        options,
        smart_auto_blocking(meshes.clone(), my_vram_gb, target_name).await?,
        auto_join_candidates,
        my_vram_gb,
        has_startup_models,
    )
    .await
}

pub(super) async fn discover_nostr_meshes(relays: &[String]) -> Result<Vec<nostr::DiscoveredMesh>> {
    let filter = nostr::MeshFilter::default();
    match nostr::discover(relays, &filter, None).await {
        Ok(meshes) => Ok(meshes),
        Err(err) => {
            record_discovery_operational_event(DiscoveryOperationalEvent::DiscoveryFailed);
            let _ = emit_event(OutputEvent::DiscoveryFailed {
                message: "Nostr auto-discovery failed".to_string(),
                detail: Some(err.to_string()),
            });
            Err(err)
        }
    }
}

pub(super) fn log_nostr_auto_candidates(
    meshes: &[nostr::DiscoveredMesh],
    target_name: Option<&String>,
) {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let last_mesh_id = mesh::load_last_mesh_id();
    let listed: Vec<&nostr::DiscoveredMesh> = if target_name.is_some() {
        meshes.iter().collect()
    } else {
        meshes
            .iter()
            .filter(|m| nostr::is_auto_eligible(m))
            .collect()
    };
    for mesh in &listed {
        let score = nostr::score_mesh(mesh, now, last_mesh_id.as_deref());
        let _ = emit_event(OutputEvent::MeshFound {
            mesh: mesh
                .listing
                .name
                .as_deref()
                .unwrap_or("unnamed")
                .to_string(),
            peers: mesh.listing.node_count,
            region: mesh.listing.region.clone(),
        });
        tracing::debug!(
            "Nostr auto-discovery candidate: {} score={} nodes={} vram_gb={:.0} clients={}",
            mesh.listing.name.as_deref().unwrap_or("unnamed"),
            score,
            mesh.listing.node_count,
            mesh.listing.total_vram_bytes as f64 / 1e9,
            mesh.listing.client_count
        );
    }
}

pub(super) fn initial_console_session_mode(
    explicit_surface: Option<RuntimeSurface>,
) -> ConsoleSessionMode {
    initial_console_session_mode_for_surface(
        explicit_surface,
        mesh_llm_events::current_console_session_mode(),
    )
}

pub fn console_session_mode_for_runtime_surface(
    explicit_surface: Option<RuntimeSurface>,
) -> ConsoleSessionMode {
    initial_console_session_mode(explicit_surface)
}

pub(super) fn initial_console_session_mode_for_surface(
    explicit_surface: Option<RuntimeSurface>,
    current_mode: ConsoleSessionMode,
) -> ConsoleSessionMode {
    match explicit_surface {
        Some(RuntimeSurface::Serve | RuntimeSurface::Client) => current_mode,
        _ => ConsoleSessionMode::None,
    }
}

/// Pick which model this node should serve.
///
/// Priority:
/// 1. Models the mesh needs that we already have on disk
/// 2. Models in the mesh catalog that nobody is serving yet (on disk preferred)
///
/// Parse a catalog size string like "18.3GB" or "491MB" into bytes.
pub(super) async fn smart_auto_blocking(
    meshes: Vec<nostr::DiscoveredMesh>,
    my_vram_gb: f64,
    target_name: Option<String>,
) -> Result<nostr::AutoDecision> {
    tokio::task::spawn_blocking(move || {
        nostr::smart_auto(&meshes, my_vram_gb, target_name.as_deref())
    })
    .await
    .context("join smart auto task")
}

pub(super) async fn handle_auto_decision(
    options: &mut RuntimeOptions,
    decision: nostr::AutoDecision,
    auto_join_candidates: &mut Vec<(String, Option<String>)>,
    my_vram_gb: f64,
    has_startup_models: bool,
) -> Result<()> {
    match decision {
        nostr::AutoDecision::Join { candidates } => {
            if options.client {
                // Clients skip health probe — joining itself is the test.
                // Queue all candidates so we can fall back if the top one is unreachable.
                let (_, mesh) = &candidates[0];
                if options.mesh_name.is_none()
                    && let Some(ref name) = mesh.listing.name
                {
                    options.mesh_name = Some(name.clone());
                }
                for (token, _) in &candidates {
                    options.join.push(token.clone());
                }
            } else {
                // GPU nodes try each candidate directly. The real join path can use relays,
                // so a separate local probe would reject reachable meshes behind firewalls.
                let mut joined = false;
                for (token, mesh) in &candidates {
                    let _ = emit_event(OutputEvent::MeshFound {
                        mesh: mesh
                            .listing
                            .name
                            .as_deref()
                            .unwrap_or("unnamed")
                            .to_string(),
                        peers: mesh.listing.node_count,
                        region: mesh.listing.region.clone(),
                    });
                    auto_join_candidates.push((token.clone(), mesh.listing.name.clone()));
                    joined = true;
                }
                if !joined {
                    record_discovery_operational_event(DiscoveryOperationalEvent::DiscoveryFailed);
                    let _ = emit_event(OutputEvent::DiscoveryFailed {
                        message: "No meshes found — starting new".to_string(),
                        detail: None,
                    });
                    let models = default_models_for_vram_blocking(my_vram_gb).await?;
                    start_new_mesh(options, &models, my_vram_gb, has_startup_models);
                }
            }
        }
        nostr::AutoDecision::StartNew { models } => {
            if options.client {
                // Client mode should still expose its local proxy and management API while
                // it waits for a mesh to appear.
                let _ = emit_event(OutputEvent::Info {
                    message: "No meshes found yet — starting client API while discovery continues"
                        .to_string(),
                    context: None,
                });
            } else {
                start_new_mesh(options, &models, my_vram_gb, has_startup_models);
            }
        }
    }
    Ok(())
}

pub(super) async fn default_models_for_vram_blocking(my_vram_gb: f64) -> Result<Vec<String>> {
    tokio::task::spawn_blocking(move || nostr::default_models_for_vram(my_vram_gb))
        .await
        .context("join default model selection task")
}

pub use super::discovery::nostr_relays;

pub(super) async fn attempt_run_auto_join(
    node: &mesh::Node,
    join_attempts: &[(String, Option<String>)],
    prefer_fast_probe: bool,
) -> RunAutoJoinOutcome {
    record_discovery_operational_event(DiscoveryOperationalEvent::JoinStarted);
    let mut outcome = RunAutoJoinOutcome {
        joined: false,
        last_join_error: None,
        successful_join: None,
    };

    if prefer_fast_probe {
        match attempt_fast_auto_join(node, join_attempts).await {
            Some(Ok(successful_join)) => {
                return build_successful_run_auto_join(node, successful_join).await;
            }
            Some(Err(err)) => outcome.last_join_error = Some(format!("{err:#}")),
            None => {}
        }
    }

    for (token, mesh_name) in join_attempts {
        match node.join_with_retry(token).await {
            Ok(()) => {
                if node.mesh_id().await.is_some() {
                    record_first_joined_mesh_ts(node).await;
                }
                let _ = emit_event(OutputEvent::Info {
                    message: "Connected to bootstrap peer; awaiting mesh admission".to_string(),
                    context: None,
                });
                let _ = emit_event(OutputEvent::DiscoveryJoined {
                    mesh: successful_join_mesh_label(mesh_name.as_deref()),
                });
                record_discovery_operational_event(DiscoveryOperationalEvent::JoinSucceeded);
                outcome.joined = true;
                outcome.successful_join = Some((token.clone(), mesh_name.clone()));
                break;
            }
            Err(err) => {
                tracing::warn!("Failed to join via token: {err}");
                outcome.last_join_error = Some(format!("{err:#}"));
            }
        }
    }

    if !outcome.joined {
        record_discovery_operational_event(DiscoveryOperationalEvent::JoinFailed);
    }

    outcome
}

pub(super) async fn attempt_fast_auto_join(
    node: &mesh::Node,
    join_attempts: &[(String, Option<String>)],
) -> Option<Result<(String, Option<String>)>> {
    match node.join_first_responsive_candidate(join_attempts).await {
        Ok(Some(successful_join)) => Some(Ok(successful_join)),
        Ok(None) => None,
        Err(err) => {
            tracing::warn!("Fast auto-join probe failed: {err:#}");
            Some(Err(err))
        }
    }
}

pub(super) async fn build_successful_run_auto_join(
    node: &mesh::Node,
    successful_join: (String, Option<String>),
) -> RunAutoJoinOutcome {
    if node.mesh_id().await.is_some() {
        record_first_joined_mesh_ts(node).await;
    }
    let _ = emit_event(OutputEvent::Info {
        message: "Connected to bootstrap peer; awaiting mesh admission".to_string(),
        context: None,
    });
    let _ = emit_event(OutputEvent::DiscoveryJoined {
        mesh: successful_join_mesh_label(successful_join.1.as_deref()),
    });
    record_discovery_operational_event(DiscoveryOperationalEvent::JoinSucceeded);
    RunAutoJoinOutcome {
        joined: true,
        last_join_error: None,
        successful_join: Some(successful_join),
    }
}

pub(super) fn successful_join_mesh_label(mesh_name: Option<&str>) -> String {
    mesh_name.unwrap_or("unnamed").to_string()
}

pub(super) fn update_cli_with_successful_run_auto_join(
    options: &mut RuntimeOptions,
    successful_join: Option<(String, Option<String>)>,
) {
    if !options.join.is_empty() {
        return;
    }

    options.join.clear();
    if let Some((token, mesh_name)) = successful_join {
        // A token that came from a file-backed source must never be recorded
        // as a literal: the rejoin loop re-reads that file on every tick, and
        // a frozen copy would survive the next rotation and be retried
        // forever. Literals and discovery-supplied tokens are still recorded.
        if !join_sources::file_backed_join_tokens(options).contains(&token) {
            options.join.push(token);
        }
        if options.mesh_name.is_none()
            && let Some(name) = mesh_name
        {
            options.mesh_name = Some(name);
        }
    }
}

pub(super) async fn run_auto_join_existing_mesh(
    options: &mut RuntimeOptions,
    node: &mesh::Node,
    auto_join_candidates: &[(String, Option<String>)],
) {
    // Resolve rather than read `options.join`: a token that lives only in a
    // file is still a configured token, and this is what keeps such a token
    // out of `options.join` as a frozen literal.
    let effective_join_tokens = options.effective_join_tokens();
    let join_attempts: Vec<(String, Option<String>)> = if !effective_join_tokens.is_empty() {
        effective_join_tokens
            .into_iter()
            .map(|token| (token, None))
            .collect()
    } else {
        auto_join_candidates.to_vec()
    };
    let prefer_fast_probe = should_prefer_fast_auto_join(options, auto_join_candidates);
    let outcome = attempt_run_auto_join(node, &join_attempts, prefer_fast_probe).await;
    update_cli_with_successful_run_auto_join(options, outcome.successful_join);

    if !outcome.joined {
        let reason = outcome.last_join_error.as_deref().unwrap_or("unknown");
        let _ = emit_event(OutputEvent::Warning {
            message: format!("Failed to join any peer — running standalone ({reason})"),
            context: None,
        });
    }

    spawn_run_auto_post_join_tasks(options, node).await;
}

pub(super) fn should_prefer_fast_auto_join(
    options: &RuntimeOptions,
    auto_join_candidates: &[(String, Option<String>)],
) -> bool {
    options.client
        || (options.effective_join_tokens().is_empty() && !auto_join_candidates.is_empty())
}

/// Resolve one rejoin tick's invite tokens, plus one message describing every
/// unusable source on that tick.
///
/// File-backed tokens are read here rather than captured once at startup, so a
/// rotated invite token is picked up by the next tick with no service restart
/// and no unit edit. `literals` is argv/`MESH_LLM_JOIN` only — a file-derived
/// token is resolved from its file on every tick and never frozen, which is
/// what stops a rotated-out token from being retried forever.
pub(super) fn resolve_rejoin_tokens(
    literals: &[String],
    join_files: &[PathBuf],
    config_override: Option<&Path>,
) -> (Vec<String>, Option<String>) {
    let resolved = join_sources::resolve_invite_tokens(literals, join_files, config_override);
    // Report every broken source, not just the first: with two, an operator
    // should not have to fix one to discover the other.
    let failure = (!resolved.errors.is_empty()).then(|| resolved.errors.join("; "));
    (resolved.tokens, failure)
}

pub(super) async fn spawn_run_auto_post_join_tasks(options: &RuntimeOptions, node: &mesh::Node) {
    let save_node = node.clone();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
        if let Some(id) = save_node.mesh_id().await {
            record_first_joined_mesh_ts(&save_node).await;
            if let Err(error) = mesh::save_last_mesh_id(&id) {
                tracing::warn!(error = %error, "failed to save last mesh ID");
            }
            tracing::info!("Mesh ID: {id}");
        }
    });

    let mesh_id = node
        .mesh_id()
        .await
        .unwrap_or_else(|| "pending".to_string());
    let _ = emit_event(OutputEvent::InviteToken {
        token: node.invite_token().await,
        mesh_id,
        mesh_name: options.mesh_name.clone(),
    });

    let rejoin_node = node.clone();
    // Literals only, by construction: `options.join` never carries a
    // file-derived token, so re-solving the sources each tick cannot
    // resurrect a token a rotation retired.
    let rejoin_tokens: Vec<String> = options.join.clone();
    let rejoin_join_files: Vec<PathBuf> = options.join_files.clone();
    let rejoin_config: Option<PathBuf> = options.config.clone();
    tokio::spawn(async move {
        // A node is expected to stay joined, so report each change into a
        // failing rejoin once. An expired invite token or an unreadable token
        // file used to loop here at debug level forever, which is how a
        // private-mesh service retried a dead credential for days with nothing
        // an operator would see.
        let mut last_failure: Option<String> = None;
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(60)).await;
            // Re-resolve file-backed tokens on every tick so a rotated invite
            // token is picked up without a service restart or a unit edit.
            let (tokens, file_failure) =
                resolve_rejoin_tokens(&rejoin_tokens, &rejoin_join_files, rejoin_config.as_deref());
            let mut join_failure: Option<String> = None;
            for t in &tokens {
                if let Err(e) = rejoin_node.join(t).await {
                    join_failure = Some(format!("Rejoin failed: {e}"));
                }
            }
            // Prefer the token-source failure: a stale in-memory token failing
            // to join is the consequence, and the unreadable file is what an
            // operator has to fix.
            let failure = file_failure.or(join_failure);
            if failure != last_failure {
                match failure.as_deref() {
                    Some(message) => {
                        // Scope the wording to the token source: another
                        // source may have joined fine, in which case the node
                        // is not the thing that is failing.
                        let message =
                            format!("invite token source: {message} — invite join keeps retrying");
                        tracing::warn!("{message}");
                        let _ = emit_event(OutputEvent::Warning {
                            message,
                            context: None,
                        });
                    }
                    None if last_failure.is_some() => {
                        tracing::info!("mesh rejoin recovered");
                    }
                    None => {}
                }
                last_failure = failure;
            }
            rejoin_node.redial_join_targets().await;
            rejoin_node.refresh_adopted_mesh_membership().await;
        }
    });

    if options.mesh_discovery_mode == mesh_discovery::MeshDiscoveryMode::Nostr
        && (options.auto || options.discover.is_some())
    {
        let rediscover_node = node.clone();
        let rediscover_relays = nostr_relays(&options.nostr_relay);
        let rediscover_relay_urls = options.relay.clone();
        let rediscover_mesh_name = options.mesh_name.clone();
        tokio::spawn(Box::pin(nostr_rediscovery(
            rediscover_node,
            rediscover_relays,
            rediscover_relay_urls,
            rediscover_mesh_name,
        )));
    } else if should_start_lan_rediscovery(
        options.mesh_discovery_mode,
        &options.effective_join_tokens(),
    ) {
        let rediscover_node = node.clone();
        let rediscover_join_tokens = options.effective_join_tokens();
        let rediscover_mesh_name = options.mesh_name.clone();
        let rediscover_region = options.region.clone();
        tokio::spawn(Box::pin(lan_rediscovery(
            rediscover_node,
            rediscover_join_tokens,
            rediscover_mesh_name,
            rediscover_region,
        )));
    }
}

pub(super) async fn run_auto_join_mesh_phase(
    options: &mut RuntimeOptions,
    node: &mesh::Node,
    auto_join_candidates: &[(String, Option<String>)],
) -> Result<()> {
    if !options.effective_join_tokens().is_empty() || !auto_join_candidates.is_empty() {
        record_discovery_operational_event(DiscoveryOperationalEvent::DecisionJoin);
        run_auto_join_existing_mesh(options, node, auto_join_candidates).await;
    } else {
        record_discovery_operational_event(DiscoveryOperationalEvent::DecisionStartNew);
        run_auto_start_new_mesh(options, node).await?;
    }
    Ok(())
}

pub(super) fn run_auto_model_identity(
    primary_startup_model: Option<&StartupModelPlan>,
    model: &Path,
) -> (String, String) {
    let model_name = primary_startup_model
        .map(|startup_model| startup_model.declared_ref.clone())
        .unwrap_or_else(|| models::model_ref_for_path(model));
    let model_source = primary_startup_model
        .map(|startup_model| startup_model.model_source.clone())
        .unwrap_or_else(|| model_name.clone());
    (model_name, model_source)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn successful_join_mesh_label_preserves_named_and_unnamed_meshes() {
        assert_eq!(successful_join_mesh_label(Some("mesh-llm")), "mesh-llm");
        assert_eq!(successful_join_mesh_label(None), "unnamed");
    }

    #[test]
    fn rejoin_ticks_re_read_file_backed_tokens() {
        let temp = tempfile::tempdir().expect("tempdir should exist");
        let token_path = temp.path().join("invite.token");
        let join_files = vec![token_path.clone()];

        std::fs::write(&token_path, "first-token\n").expect("token file should write");
        let (tokens, failure) = resolve_rejoin_tokens(&[], &join_files, None);
        assert_eq!(tokens, ["first-token"]);
        assert!(failure.is_none(), "{failure:?}");

        // A rotated token must be observed by the next tick: this is the
        // property that lets a private-mesh service rotate an expiring invite
        // token without a restart or a unit-file edit.
        std::fs::write(&token_path, "rotated-token\n").expect("token file should rewrite");
        let (tokens, failure) = resolve_rejoin_tokens(&[], &join_files, None);
        assert_eq!(tokens, ["rotated-token"]);
        assert!(failure.is_none(), "{failure:?}");

        // Argv tokens stay in play even when the file-backed source breaks.
        std::fs::remove_file(&token_path).expect("token file should remove");
        let (tokens, failure) =
            resolve_rejoin_tokens(&["argv-token".to_string()], &join_files, None);
        assert_eq!(tokens, ["argv-token"]);
        assert!(
            failure
                .as_deref()
                .is_some_and(|message| message.contains("cannot read join token file")),
            "{failure:?}"
        );
    }

    /// Regression for the rotation gap in the original design: a startup
    /// fold of the file token into `options.join` meant the rejoin tick
    /// resolved `[stale-startup-token, fresh-file-token]` after a rotation and
    /// drove a rejected wire join on the dead token every 60s, forever, on a
    /// node that was already joined. The pre-existing test above used an empty
    /// literal set, which is exactly the shape that hides it.
    #[test]
    fn rejoin_ticks_drop_a_rotated_out_file_token_even_with_literals() {
        let temp = tempfile::tempdir().expect("tempdir should exist");
        let token_path = temp.path().join("invite.token");
        let options = RuntimeOptions {
            join: vec!["argv-token".to_string()],
            join_files: vec![token_path.clone()],
            ..RuntimeOptions::default()
        };

        std::fs::write(&token_path, "first-token\n").expect("token file should write");
        join_sources::validate_join_token_sources(&options)
            .expect("startup validation should pass");

        // The rejoin task captures the literal set exactly the way the runtime
        // does, after the startup step has run.
        let rejoin_literals = options.join.clone();
        let rejoin_join_files = options.join_files.clone();
        assert_eq!(
            rejoin_literals,
            ["argv-token"],
            "startup must not fold the file-backed token into the literal set"
        );

        let (tokens, failure) = resolve_rejoin_tokens(&rejoin_literals, &rejoin_join_files, None);
        assert_eq!(tokens, ["argv-token", "first-token"]);
        assert!(failure.is_none(), "{failure:?}");

        std::fs::write(&token_path, "rotated-token\n").expect("token file should rewrite");
        let (tokens, failure) = resolve_rejoin_tokens(&rejoin_literals, &rejoin_join_files, None);
        assert_eq!(tokens, ["argv-token", "rotated-token"]);
        assert!(
            !tokens.iter().any(|token| token == "first-token"),
            "the pre-rotation token must not survive in the tick's set: {tokens:?}"
        );
        assert!(failure.is_none(), "{failure:?}");
    }

    /// Two unusable sources must both be visible on one tick: an operator
    /// should not have to fix the first to discover the second.
    #[test]
    fn rejoin_reports_every_unusable_token_source() {
        let temp = tempfile::tempdir().expect("tempdir should exist");
        let first = temp.path().join("first.token");
        let second = temp.path().join("second.token");

        let (tokens, failure) = resolve_rejoin_tokens(&[], &[first.clone(), second.clone()], None);

        assert!(tokens.is_empty(), "{tokens:?}");
        let failure = failure.expect("both sources are unusable");
        assert!(failure.contains(&first.display().to_string()), "{failure}");
        assert!(failure.contains(&second.display().to_string()), "{failure}");
    }
}
