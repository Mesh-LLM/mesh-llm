//! Automatic contribution for an unconfigured serving daemon.
//!
//! This owns selection only. The existing intent consumer owns resolution,
//! capacity admission, loading, manual overrides, and failures.
use super::{IntentSource, ModelIntent, RuntimeOptions};
use crate::{mesh, models, plugin};
use anyhow::{Context, Result};
use mesh_llm_events::{OutputEvent, emit_event};
use std::{
    collections::BTreeSet,
    time::{Duration, Instant},
};

pub(super) fn enabled(options: &RuntimeOptions, config: &plugin::MeshConfig) -> bool {
    !options.client
        && config.runtime.mode == mesh_llm_config::RuntimeMode::Serve
        && !super::cli_has_explicit_models(options)
        && config.models.is_empty()
        && !options.local_model_only
        && !options.split
}

#[derive(Debug, PartialEq, Eq)]
enum Decision {
    AwaitAdmission,
    Covered,
    Warming,
    Select,
}

fn decide(explicit_join: bool, admitted_peers: usize, has_model: bool) -> Decision {
    if explicit_join && admitted_peers == 0 {
        Decision::AwaitAdmission
    } else if has_model {
        Decision::Covered
    } else {
        Decision::Select
    }
}

fn report(message: impl Into<String>) {
    let _ = emit_event(OutputEvent::Info {
        message: message.into(),
        context: None,
    });
}

pub(super) fn spawn(
    options: &RuntimeOptions,
    config: &plugin::MeshConfig,
    node: &mesh::Node,
    local_models: &[String],
) -> tokio::task::JoinSet<()> {
    let mut tasks = tokio::task::JoinSet::new();
    if enabled(options, config) {
        tasks.spawn(run(
            options.clone(),
            config.clone(),
            node.clone(),
            local_models.to_vec(),
        ));
    }
    tasks
}

pub(super) async fn run(
    options: RuntimeOptions,
    config: plugin::MeshConfig,
    node: mesh::Node,
    local_models: Vec<String>,
) {
    if let Err(error) = contribute(&options, &config, &node, local_models).await {
        let _ = emit_event(OutputEvent::Warning {
            message: "Automatic serving could not start a model".into(),
            context: Some(format!("{error:#}")),
        });
    }
}

async fn wait_for_selection(node: &mesh::Node, explicit_join: bool) -> bool {
    let mut last_decision = None;
    let mut warming_since = None;
    loop {
        // Human intent, including an unload while selection is pending, owns
        // this session. Never race it by introducing a different model.
        if has_manual_intent(node)
            || !node.serving_models().await.is_empty()
            || !node.hosted_models().await.is_empty()
        {
            return false;
        }
        let peers = node.peers().await;
        let ready = peers.iter().any(peer_has_usable_coverage);
        let warming = peers
            .iter()
            .any(|peer| peer_accepts_remote_inference(peer) && !peer.serving_models.is_empty());
        let elapsed = if warming && !ready {
            warming_since.get_or_insert_with(Instant::now).elapsed()
        } else {
            warming_since = None;
            Duration::ZERO
        };
        let decision = coverage_decision(explicit_join, peers.len(), ready, warming, elapsed);
        if last_decision.as_ref() != Some(&decision) {
            report(match decision {
                Decision::AwaitAdmission => {
                    "Automatic serving is waiting for admission to the requested mesh; public fallback is disabled"
                }
                Decision::Covered => {
                    "Selected mesh has a usable model; using peers without loading a duplicate"
                }
                Decision::Warming => {
                    "Waiting up to 60 seconds for the selected mesh's assigned models to become usable"
                }
                Decision::Select => "Selecting a model for this machine's available capacity",
            });
        }
        if decision == Decision::Select {
            return true;
        }
        last_decision = Some(decision);
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
}

fn peer_accepts_remote_inference(peer: &mesh::PeerInfo) -> bool {
    !matches!(
        peer.inference_admission_state,
        Some(
            crate::proto::node::InferenceAdmissionState::RemotePaused
                | crate::proto::node::InferenceAdmissionState::AllPaused
        )
    )
}

fn peer_has_usable_coverage(peer: &mesh::PeerInfo) -> bool {
    peer_accepts_remote_inference(peer) && !peer.http_routable_models().is_empty()
}

fn has_manual_intent(node: &mesh::Node) -> bool {
    node.runtime_intents
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .iter()
        .any(|intent| intent.source != IntentSource::MeshDemand)
}

fn coverage_decision(
    explicit_join: bool,
    peers: usize,
    ready: bool,
    warming: bool,
    elapsed: Duration,
) -> Decision {
    match decide(explicit_join, peers, ready) {
        Decision::Select if warming && elapsed < Duration::from_secs(60) => Decision::Warming,
        decision => decision,
    }
}

async fn contribute(
    options: &RuntimeOptions,
    config: &plugin::MeshConfig,
    node: &mesh::Node,
    local_models: Vec<String>,
) -> Result<()> {
    if !wait_for_selection(node, !options.join.is_empty()).await {
        return Ok(());
    }
    let capacity = node.local_runtime_capacity_bytes();
    let local_only = config
        .defaults
        .as_ref()
        .and_then(|defaults| defaults.skippy.as_ref())
        .and_then(|skippy| skippy.source_policy.as_deref())
        == Some("local-required");
    let mut attempted = BTreeSet::new();
    let mut last_error = None;
    for _ in 0..3 {
        let local = local_models.clone();
        let excluded = attempted.clone();
        let candidate = tokio::task::spawn_blocking(move || {
            select_model(capacity, &local, local_only, &excluded)
        })
        .await
        .context("automatic model selector task failed")?;
        let candidate = match candidate {
            Ok(candidate) => candidate,
            Err(error) => {
                report(format!(
                    "Automatic catalog selection failed: {error:#}; retrying with a bounded delay"
                ));
                last_error = Some(error);
                tokio::time::sleep(Duration::from_secs(5)).await;
                if !wait_for_selection(node, !options.join.is_empty()).await {
                    return Ok(());
                }
                continue;
            }
        };
        let Some(candidate) = candidate else { break };
        attempted.insert(candidate.clone());
        if !wait_for_selection(node, !options.join.is_empty()).await {
            return Ok(());
        }
        // Absolute paths are private loader inputs, never gossip identifiers.
        let key = if std::path::Path::new(&candidate).is_absolute() {
            models::model_ref_for_path(std::path::Path::new(&candidate))
        } else {
            candidate.clone()
        };
        let request = node.request_automatic_model(key.clone());
        node.regossip().await;
        if !settle_contribution(node, !options.join.is_empty(), &key).await {
            drop(request);
            node.regossip().await;
            return Ok(());
        }
        let result = submit_load(node, candidate).await;
        drop(request);
        node.regossip().await;
        match result {
            Ok(()) => return Ok(()),
            Err(error) => {
                report(format!(
                    "Automatic model attempt failed: {error:#}; trying another suitable model"
                ));
                last_error = Some(error);
            }
        }
    }
    anyhow::bail!(
        "Automatic serving exhausted suitable candidates (capacity {capacity} bytes, {} attempts): {}",
        attempted.len(),
        last_error.map_or_else(
            || "no complete supported chat model fits".into(),
            |e| format!("{e:#}")
        )
    )
}

/// Best-effort duplicate avoidance, not a lease: a silent requester can defer
/// us only once for this attempt. Gossip delay/partitions can still duplicate.
async fn settle_contribution(node: &mesh::Node, explicit_join: bool, key: &str) -> bool {
    let started = Instant::now();
    let mesh_id = node.mesh_id().await;
    loop {
        if has_manual_intent(node)
            || !node.serving_models().await.is_empty()
            || node.mesh_id().await != mesh_id
        {
            return false;
        }
        let peers = node.peers().await;
        if (explicit_join && peers.is_empty()) || peers.iter().any(peer_has_usable_coverage) {
            // Continue watching coverage/admission without bypassing the final checks.
            if !wait_for_selection(node, explicit_join).await {
                return false;
            }
            continue;
        }
        let defer_to_peer = peers
            .iter()
            .any(|peer| defers_to_peer(peer, &mesh_id, peer.id < node.id(), key));
        // A request becoming an assignment keeps the same deadline; neither
        // repeated gossip nor a stuck worker starts another warming interval.
        if arbitration_ready(started.elapsed(), defer_to_peer) {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

fn defers_to_peer(
    peer: &mesh::PeerInfo,
    mesh_id: &Option<String>,
    precedes_us: bool,
    key: &str,
) -> bool {
    peer.is_admitted()
        && peer_accepts_remote_inference(peer)
        && &peer.mesh_id == mesh_id
        && !matches!(peer.role, mesh::NodeRole::Client)
        && peer.last_seen.elapsed() < Duration::from_secs(mesh::PEER_STALE_SECS)
        && (!peer.serving_models.is_empty()
            || (precedes_us && peer.requested_models.iter().any(|model| model == key)))
}

fn arbitration_ready(elapsed: Duration, defer_to_peer: bool) -> bool {
    elapsed >= Duration::from_secs(3) && (!defer_to_peer || elapsed >= Duration::from_secs(60))
}

async fn submit_load(node: &mesh::Node, candidate: String) -> Result<()> {
    report(format!("Automatically loading {candidate}"));
    let tx = node
        .model_intent_tx
        .lock()
        .await
        .clone()
        .context("runtime intent queue is unavailable")?;
    let (completion, response) = tokio::sync::oneshot::channel();
    tx.send(ModelIntent::Load {
        intent_id: None,
        spec: candidate,
        config_model_id: None,
        profile: String::new(),
        source: IntentSource::MeshDemand,
        completion: Some(completion),
    })
    .await
    .context("runtime intent queue closed")?;
    response
        .await
        .context("automatic model load response was cancelled")??;
    report("Automatic model is ready to serve");
    Ok(())
}

#[derive(Debug)]
struct Candidate {
    model_ref: String,
    bytes: u64,
    local: bool,
}

fn choose(candidates: Vec<Candidate>, capacity: u64) -> Option<String> {
    candidates
        .into_iter()
        .filter(|candidate| {
            candidate.bytes > 0 && super::model_fits_runtime_capacity(candidate.bytes, capacity)
        })
        .max_by(|a, b| {
            a.local
                .cmp(&b.local)
                .then(a.bytes.cmp(&b.bytes))
                .then_with(|| b.model_ref.cmp(&a.model_ref))
        })
        .map(|candidate| candidate.model_ref)
}

fn select_model(
    capacity: u64,
    local_models: &[String],
    local_only: bool,
    excluded: &BTreeSet<String>,
) -> Result<Option<String>> {
    if capacity == 0 {
        return Ok(None);
    }
    // Never interpret missing size information as a zero-byte model that fits.
    // Only complete local GGUFs are eligible; partial layer packages require an
    // explicit split plan and cannot bootstrap a standalone chat model.
    let mut candidates = Vec::new();
    for model_ref in local_models {
        let path = models::find_model_path(model_ref);
        if !path.is_file()
            || !suitable_cached_chat(model_ref, &path)
            || models::gguf::scan_gguf_bundle_total_parameters(&path).is_none()
        {
            continue;
        }
        if let (Ok(bytes), Some(spec)) = (
            super::runtime_model_planning_bytes(&path),
            cached_load_spec(model_ref, &path, local_only),
        ) {
            candidates.push(Candidate {
                model_ref: spec,
                bytes,
                local: true,
            });
        }
    }
    candidates.retain(|candidate| !excluded.contains(&candidate.model_ref));
    if let Some(local) = choose(candidates, capacity) {
        return Ok(Some(local));
    }
    if local_only {
        return Ok(None);
    }
    models::remote_catalog::ensure_catalog()?;
    let catalog = models::remote_catalog::loaded_models()?;
    Ok(choose_catalog_model(&catalog, capacity, excluded))
}

fn choose_catalog_model(
    catalog: &[models::remote_catalog::RemoteCatalogModel],
    capacity: u64,
    excluded: &BTreeSet<String>,
) -> Option<String> {
    // Curated chat preferences match the smart-auto hardware tiers. Never
    // interpret a missing catalog entry/size as a fit or load demand seeds en masse.
    for (minimum_gb, name) in CHAT_PREFERENCES {
        if capacity < minimum_gb * 1_000_000_000 {
            continue;
        }
        let Some(model) = catalog.iter().find(|model| model.name == name) else {
            continue;
        };
        let model_ref = models::remote_catalog_model_ref(model);
        if excluded.contains(&model_ref) {
            continue;
        }
        let Some(bytes) = model.size.as_deref().and_then(super::parse_size_str) else {
            continue;
        };
        if bytes > 0 && super::model_fits_runtime_capacity(bytes, capacity) {
            return Some(model_ref);
        }
    }
    None
}

fn cached_load_spec(model_ref: &str, path: &std::path::Path, local_only: bool) -> Option<String> {
    if !local_only {
        return Some(model_ref.into());
    }
    let absolute = std::path::absolute(path).ok()?;
    // Preserve strict-local's existing non-symlink policy. In particular, do not
    // canonicalize an HF snapshot link before validation to bypass the policy.
    super::startup_models::validate_local_required_source(&absolute, model_ref).ok()?;
    Some(absolute.to_string_lossy().into_owned())
}

const CHAT_PREFERENCES: [(u64, &str); 8] = [
    (179, "MiniMax-M2.5-Q4_K_M"),
    (63, "Qwen3-Coder-Next-Q4_K_M"),
    (50, "GLM-4.7-Flash-Q4_K_M"),
    (24, "Qwen3.5-27B-Q4_K_M"),
    (8, "Gemma-4-E4B-it-Q4_K_M"),
    (0, "Qwen3-4B-Q4_K_M"),
    (0, "Qwen3-1.7B-Q4_K_M"),
    (0, "Qwen3-0.6B-Q4_K_M"),
];

fn suitable_cached_chat(model_ref: &str, path: &std::path::Path) -> bool {
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    let name = format!("{model_ref} {filename}").to_ascii_lowercase();
    if [
        "-layers",
        "embedding",
        "rerank",
        "mmproj",
        "draft",
        "-base",
        "-mtp",
    ]
    .iter()
    .any(|excluded| name.contains(excluded))
    {
        return false;
    }
    // Capability flags describe modalities, not generative-chat suitability.
    // Be conservative for automatic startup; arbitrary models remain explicitly loadable.
    let known_chat = ["qwen3", "minimax-m2", "glm-4", "-instruct", "-it-", "-chat"]
        .iter()
        .any(|signal| name.contains(signal));
    known_chat
        && models::gguf::scan_gguf_compact_meta(path).is_some_and(|meta| {
            meta.layer_count > 0
                && !meta.architecture.is_empty()
                && !meta.tokenizer_model_name.is_empty()
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_chat_gguf(path: &std::path::Path) {
        fn string(bytes: &mut Vec<u8>, value: &str) {
            bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
            bytes.extend_from_slice(value.as_bytes());
        }
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes()); // one tensor
        bytes.extend_from_slice(&3u64.to_le_bytes()); // three metadata entries
        for (key, value) in [
            ("general.architecture", "qwen3"),
            ("tokenizer.ggml.model", "gpt2"),
        ] {
            string(&mut bytes, key);
            bytes.extend_from_slice(&8u32.to_le_bytes());
            string(&mut bytes, value);
        }
        string(&mut bytes, "qwen3.block_count");
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&2u32.to_le_bytes());
        string(&mut bytes, "weights");
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&16u64.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.resize(1024, 0);
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn cache_selection_rejects_incomplete_shards_and_obvious_non_chat_models() {
        let dir = tempfile::tempdir().unwrap();
        let first = dir.path().join("Qwen3-4B-Q4_K_M-00001-of-00002.gguf");
        let second = dir.path().join("Qwen3-4B-Q4_K_M-00002-of-00002.gguf");
        write_chat_gguf(&first);
        let model_ref = models::model_ref_for_path(&first);
        let local = vec![model_ref.clone()];
        assert!(
            select_model(u64::MAX, &local, true, &BTreeSet::new())
                .unwrap()
                .is_none()
        );
        write_chat_gguf(&second);
        assert!(
            select_model(u64::MAX, &local, true, &BTreeSet::new())
                .unwrap()
                .is_some()
        );
        assert!(suitable_cached_chat(&model_ref, &first));
        let misleading_parent = dir.path().join("draft-models");
        std::fs::create_dir(&misleading_parent).unwrap();
        let relocated = misleading_parent.join("Qwen3-4B-Q4_K_M.gguf");
        write_chat_gguf(&relocated);
        assert!(suitable_cached_chat("Qwen3-4B-Q4_K_M", &relocated));
        for name in [
            "Qwen3-Embedding",
            "Qwen3-draft",
            "Qwen3-reranker",
            "Qwen3-base",
            "Qwen3-mtp",
        ] {
            assert!(!suitable_cached_chat(name, &first));
        }
        let arbitrary = dir.path().join("unknown.gguf");
        write_chat_gguf(&arbitrary);
        assert!(!suitable_cached_chat("unknown", &arbitrary));
    }

    #[cfg(unix)]
    #[test]
    fn strict_local_candidate_uses_real_absolute_file_and_rejects_snapshot_symlink() {
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().join("model-00001-of-00002.gguf");
        std::fs::write(&real, b"fixture").unwrap();
        let snapshot = dir.path().join("snapshot.gguf");
        std::os::unix::fs::symlink(&real, &snapshot).unwrap();
        let spec = cached_load_spec("canonical-ref", &real, true).unwrap();
        assert!(std::path::Path::new(&spec).is_absolute());
        assert!(
            super::super::startup_models::validate_local_required_source(
                std::path::Path::new(&spec),
                "canonical-ref"
            )
            .is_ok()
        );
        assert!(cached_load_spec("canonical-ref", &snapshot, true).is_none());
        assert_eq!(
            cached_load_spec("canonical-ref", &snapshot, false),
            Some("canonical-ref".into())
        );
    }

    fn catalog_model(name: &str, size: Option<&str>) -> models::remote_catalog::RemoteCatalogModel {
        models::remote_catalog::RemoteCatalogModel {
            name: name.into(),
            file: format!("{name}.gguf"),
            repo: format!("org/{name}"),
            revision: None,
            source_file: format!("{name}.gguf"),
            size: size.map(str::to_owned),
            description: None,
            draft: None,
            extra_files: vec![],
            mmproj: None,
        }
    }

    #[test]
    fn curated_fallback_requires_catalog_positive_size_fit_and_not_previously_failed() {
        let unknown = catalog_model("Gemma-4-E4B-it-Q4_K_M", None);
        let small = catalog_model("Qwen3-4B-Q4_K_M", Some("2.5GB"));
        let tiny = catalog_model("Qwen3-0.6B-Q4_K_M", Some("0.5GB"));
        let models = vec![unknown, small.clone(), tiny.clone()];
        let selected = models::remote_catalog_model_ref(&small);
        assert_ne!(selected, models::remote_catalog_model_ref(&tiny));
        assert_eq!(
            choose_catalog_model(&models, 16_000_000_000, &BTreeSet::new()),
            Some(selected.clone())
        );
        let excluded = BTreeSet::from([selected]);
        assert_eq!(
            choose_catalog_model(&models, 16_000_000_000, &excluded),
            Some(models::remote_catalog_model_ref(&tiny))
        );
        assert!(choose_catalog_model(&models, 100, &excluded).is_none());
        assert!(choose_catalog_model(&[], u64::MAX, &excluded).is_none());
        assert!(
            choose_catalog_model(
                &[catalog_model("Qwen3-4B-Q4_K_M", Some("0GB"))],
                u64::MAX,
                &BTreeSet::new()
            )
            .is_none()
        );
    }

    #[test]
    fn repeated_requests_do_not_extend_deferral_and_winner_loss_releases_it() {
        assert!(!arbitration_ready(Duration::ZERO, false));
        assert!(arbitration_ready(Duration::from_secs(3), false));
        for seconds in 3..60 {
            assert!(!arbitration_ready(Duration::from_secs(seconds), true));
        }
        assert!(arbitration_ready(Duration::from_secs(60), true));
        assert!(arbitration_ready(Duration::from_secs(10), false));
    }

    async fn advertise_request(source: &mesh::Node, destination: &mesh::Node) {
        let data = source.snapshot_local_announcement_data().await;
        let ann = source.build_local_announcement(data);
        let mut peer = mesh::PeerInfo::from_announcement(
            source.id(),
            ann.addr.clone(),
            &ann,
            Default::default(),
        );
        peer.admitted = true;
        destination.insert_test_peer(peer).await;
    }

    #[tokio::test]
    async fn slow_download_then_warming_shares_one_fixed_deadline() {
        let a = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
            .await
            .unwrap();
        let b = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
            .await
            .unwrap();
        let _request = a.request_automatic_model("chat".into());
        advertise_request(&a, &b).await;
        let mut peer = b.peers().await.remove(0);
        let mesh_id = b.mesh_id().await;
        peer.last_seen = Instant::now() - Duration::from_secs(45);
        assert!(defers_to_peer(&peer, &mesh_id, true, "chat"));
        assert!(!arbitration_ready(
            Duration::from_secs(45),
            defers_to_peer(&peer, &mesh_id, true, "chat")
        ));
        peer.requested_models.clear();
        peer.serving_models.push("chat".into());
        assert!(peer.http_routable_models().is_empty());
        assert!(!arbitration_ready(
            Duration::from_secs(59),
            defers_to_peer(&peer, &mesh_id, true, "chat")
        ));
        assert!(arbitration_ready(
            Duration::from_secs(60),
            defers_to_peer(&peer, &mesh_id, true, "chat")
        ));
        peer.serving_models.clear();
        assert!(arbitration_ready(
            Duration::from_secs(50),
            defers_to_peer(&peer, &mesh_id, true, "chat")
        ));
        peer.serving_models.push("chat".into());
        peer.role = mesh::NodeRole::Host { http_port: 12345 };
        peer.hosted_models_known = true;
        peer.hosted_models.push("chat".into());
        assert!(!peer.http_routable_models().is_empty());
        peer.admitted = false;
        assert!(!defers_to_peer(&peer, &mesh_id, true, "chat"));
    }

    #[tokio::test]
    async fn paused_sole_host_does_not_suppress_selection_or_arbitration() {
        use crate::proto::node::InferenceAdmissionState;

        let host = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
            .await
            .unwrap();
        let joiner = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
            .await
            .unwrap();
        advertise_request(&host, &joiner).await;
        let mut peer = joiner.peers().await.remove(0);
        peer.role = mesh::NodeRole::Host { http_port: 12345 };
        peer.hosted_models_known = true;
        peer.hosted_models.push("chat".into());
        peer.serving_models.push("chat".into());
        peer.requested_models.push("chat".into());
        let mesh_id = joiner.mesh_id().await;
        // Missing state preserves mixed-version behavior.
        assert!(peer_has_usable_coverage(&peer));
        for state in [
            InferenceAdmissionState::RemotePaused,
            InferenceAdmissionState::AllPaused,
        ] {
            peer.inference_admission_state = Some(state);
            assert!(!peer.http_routable_models().is_empty());
            assert!(!peer_has_usable_coverage(&peer));
            assert!(!defers_to_peer(&peer, &mesh_id, true, "chat"));
            joiner.insert_test_peer(peer.clone()).await;
            assert!(
                tokio::time::timeout(Duration::from_secs(1), wait_for_selection(&joiner, true))
                    .await
                    .expect("paused assignments must not start a warming wait")
            );
        }
        peer.inference_admission_state = None;
        assert!(peer_has_usable_coverage(&peer));
        assert!(defers_to_peer(&peer, &mesh_id, true, "chat"));
    }

    #[tokio::test]
    async fn simultaneous_requesters_choose_same_winner_in_either_gossip_order() {
        let a = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
            .await
            .unwrap();
        let b = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
            .await
            .unwrap();
        let _a = a.request_automatic_model("chat".into());
        let _b = b.request_automatic_model("chat".into());
        for reverse in [false, true] {
            if reverse {
                advertise_request(&b, &a).await;
                advertise_request(&a, &b).await;
            } else {
                advertise_request(&a, &b).await;
                advertise_request(&b, &a).await;
            }
            let (winner, loser) = if a.id() < b.id() { (&a, &b) } else { (&b, &a) };
            let (won, deferred) = tokio::join!(
                settle_contribution(winner, true, "chat"),
                tokio::time::timeout(
                    Duration::from_millis(3200),
                    settle_contribution(loser, true, "chat")
                )
            );
            assert!(won);
            assert!(deferred.is_err());
        }
    }

    #[tokio::test]
    async fn manual_override_during_settling_cancels_automatic_request() {
        let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
            .await
            .unwrap();
        let request = node.request_automatic_model("chat".into());
        let manual = async {
            tokio::time::sleep(Duration::from_millis(50)).await;
            let mut state = super::super::ModelTargetReconciliationState::with_shared_history(
                node.runtime_intents.clone(),
            );
            state.add_desired("manual", "", IntentSource::ApiLoad);
        };
        let (selected, ()) = tokio::join!(settle_contribution(&node, false, "chat"), manual);
        assert!(!selected);
        drop(request);
        assert!(node.requested_models().await.is_empty());
    }

    #[test]
    fn admission_precedes_model_selection() {
        assert_eq!(decide(true, 0, false), Decision::AwaitAdmission);
        assert_eq!(decide(true, 0, true), Decision::AwaitAdmission);
        assert_eq!(decide(true, 1, false), Decision::Select);
        assert_eq!(decide(true, 1, true), Decision::Covered);
        assert_eq!(decide(false, 0, false), Decision::Select);
    }

    #[test]
    fn assigned_worker_is_not_permanent_coverage() {
        assert_eq!(
            coverage_decision(true, 1, false, true, Duration::ZERO),
            Decision::Warming
        );
        assert_eq!(
            coverage_decision(true, 1, false, true, Duration::from_secs(60)),
            Decision::Select
        );
        assert_eq!(
            coverage_decision(true, 0, false, true, Duration::ZERO),
            Decision::AwaitAdmission
        );
        assert_eq!(
            coverage_decision(true, 1, true, false, Duration::from_secs(120)),
            Decision::Covered
        );
    }

    #[tokio::test]
    async fn manual_intent_prevents_automatic_selection() {
        let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
            .await
            .unwrap();
        let mut state = super::super::ModelTargetReconciliationState::with_shared_history(
            node.runtime_intents.clone(),
        );
        state.add_desired("manual", "", IntentSource::ApiLoad);
        assert!(!wait_for_selection(&node, false).await);
    }

    #[tokio::test]
    async fn rejected_join_never_submits_a_load() {
        let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
            .await
            .unwrap();
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        *node.model_intent_tx.lock().await = Some(tx);
        let options = RuntimeOptions {
            join: vec!["not-admitted".into()],
            ..RuntimeOptions::default()
        };
        let config = plugin::MeshConfig::default();
        assert!(
            tokio::time::timeout(
                Duration::from_millis(30),
                contribute(&options, &config, &node, Vec::new())
            )
            .await
            .is_err()
        );
        assert!(rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn selection_submits_low_priority_intent_and_surfaces_loader_failure() {
        let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
            .await
            .unwrap();
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        *node.model_intent_tx.lock().await = Some(tx);
        let consumer = async {
            let ModelIntent::Load {
                spec,
                source,
                completion,
                config_model_id,
                ..
            } = rx.recv().await.unwrap()
            else {
                panic!("expected load")
            };
            assert_eq!(spec, "/cached/model.gguf");
            assert_eq!(source, IntentSource::MeshDemand);
            assert!(config_model_id.is_none());
            completion
                .unwrap()
                .send(Err(anyhow::anyhow!("native failed")))
                .unwrap();
        };
        let (result, ()) = tokio::join!(submit_load(&node, "/cached/model.gguf".into()), consumer);
        assert!(result.unwrap_err().to_string().contains("native failed"));
    }

    #[test]
    fn automatic_serving_respects_modes_and_explicit_models() {
        let options = RuntimeOptions::default();
        let mut config = plugin::MeshConfig::default();
        assert!(enabled(&options, &config));
        config.runtime.mode = mesh_llm_config::RuntimeMode::OnDemand;
        assert!(!enabled(&options, &config));
        config.runtime.mode = mesh_llm_config::RuntimeMode::Client;
        assert!(!enabled(&options, &config));
        config.runtime.mode = mesh_llm_config::RuntimeMode::Serve;
        assert!(!enabled(
            &RuntimeOptions {
                client: true,
                ..options.clone()
            },
            &config
        ));
        assert!(!enabled(
            &RuntimeOptions {
                model: vec!["chosen".into()],
                ..options.clone()
            },
            &config
        ));
        assert!(!enabled(
            &RuntimeOptions {
                gguf: vec!["chosen.gguf".into()],
                ..options.clone()
            },
            &config
        ));
        assert!(!enabled(
            &RuntimeOptions {
                split: true,
                ..options
            },
            &config
        ));
    }

    fn candidate(name: &str, bytes: u64, local: bool) -> Candidate {
        Candidate {
            model_ref: name.into(),
            bytes,
            local,
        }
    }

    #[test]
    fn selection_prefers_cached_fit_and_rejects_unknown_or_oversized_models() {
        assert_eq!(
            choose(
                vec![
                    candidate("unknown", 0, true),
                    candidate("large", 1000, true)
                ],
                1000
            ),
            None
        );
        assert_eq!(
            choose(
                vec![
                    candidate("download", 800, false),
                    candidate("cached", 500, true)
                ],
                1000
            ),
            Some("cached".into())
        );
        assert_eq!(
            choose(
                vec![
                    candidate("small", 100, false),
                    candidate("fits", 800, false)
                ],
                1000
            ),
            Some("fits".into())
        );
        assert_eq!(choose(vec![candidate("a", 100, false)], 0), None);
    }
}
