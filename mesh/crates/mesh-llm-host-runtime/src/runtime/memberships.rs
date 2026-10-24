//! Adding and removing a node's mesh memberships while it runs: the further
//! meshes it serves its models into, each with its own identity, allowlist,
//! peers and local API port. `mesh::node::membership` says what a membership
//! node shares with the node it belongs to and what it keeps to itself.

use super::RuntimeOptions;
use super::run_auto::relay_policy_for_runtime_options;
use crate::inference::election;
use crate::mesh;
use crate::network::affinity::AffinityRouter;
use crate::network::openai::accept::IngressListener;
use anyhow::{Context, Result};
use std::path::PathBuf;
use std::time::Duration;

/// How often a membership re-dials its join tokens and checks its peers.
const MEMBERSHIP_MAINTENANCE_INTERVAL: Duration = Duration::from_secs(60);

/// A further mesh to serve this node's models into.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeshMembershipConfig {
    /// Names the membership on this node, and the mesh: every node of one
    /// mesh uses the same name. Letters, digits, `.`, `_` and `-`.
    pub name: String,
    /// Invite tokens of the mesh's nodes, joined now and re-dialled while the
    /// membership lasts. With none, this node waits to be joined by its own
    /// invite token.
    pub join_tokens: Vec<String>,
    /// The owner IDs whose nodes the membership admits. Nobody else is
    /// admitted, so an empty list admits no one.
    pub trusted_owners: Vec<String>,
    /// The loopback port of the membership's OpenAI API, which routes among
    /// this node and the mesh's peers only.
    pub api_port: u16,
    /// Where the membership's node key is kept. By default, next to the
    /// node's own key, under `memberships/<name>.key`.
    pub node_key_path: Option<PathBuf>,
    /// Whether to advertise the GPU name and hostname to this mesh.
    pub enumerate_host: bool,
}

/// A running membership.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeshMembershipStatus {
    pub name: String,
    pub api_base_url: String,
    /// How the mesh's other nodes join this one.
    pub invite_token: String,
    pub endpoint_id: String,
    /// Peers connected in this mesh.
    pub peers: usize,
}

/// What a membership needs from the running node.
pub(super) struct MembershipContext<'a> {
    pub(super) options: &'a RuntimeOptions,
    pub(super) node: &'a mesh::Node,
    pub(super) targets: tokio::sync::watch::Receiver<election::ModelTargets>,
}

/// Starts a membership, replacing one of the same name.
pub(super) async fn add_membership(
    ctx: MembershipContext<'_>,
    config: MeshMembershipConfig,
) -> Result<MeshMembershipStatus> {
    validate(ctx.options, &config)?;
    if let Some(previous) = ctx.node.memberships.remove(&config.name).await {
        previous.close().await;
    }
    let other_ports = ctx.node.memberships.list().await;
    anyhow::ensure!(
        other_ports
            .iter()
            .all(|(_, _, port)| *port != config.api_port),
        "membership API port {} is already another membership's",
        config.api_port
    );
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", config.api_port))
        .await
        .with_context(|| format!("bind membership API port {}", config.api_port))?;
    let secret_key = membership_key(&config)?;
    let relay_auths: std::collections::HashMap<String, String> =
        ctx.options.relay_auth.iter().cloned().collect();
    let (node, channels) = ctx
        .node
        .start_membership_node(mesh::MembershipNodeConfig {
            secret_key,
            relay: mesh::RelayConfig {
                urls: &ctx.options.relay,
                auths: &relay_auths,
                policy: relay_policy_for_runtime_options(ctx.options),
            },
            trusted_owners: &config.trusted_owners,
            enumerate_host: config.enumerate_host,
        })
        .await?;
    // Named meshes derive their ID from the name, so every member agrees.
    if let Err(error) = node
        .initialize_mesh_identity_as_originator(Some(&config.name), None)
        .await
    {
        node.close_endpoint().await;
        return Err(error);
    }

    let affinity = AffinityRouter::new();
    let tasks = vec![
        tokio::spawn(crate::network::tunnel::serve_membership_http_ingress(
            node.clone(),
            channels.http,
            ctx.targets.clone(),
            affinity.clone(),
        ))
        .abort_handle(),
        tokio::spawn(super::proxy::api_proxy(
            node.clone(),
            config.api_port,
            ctx.targets,
            Some(IngressListener::from(listener)),
            false,
            affinity,
        ))
        .abort_handle(),
        tokio::spawn(maintain_membership(
            node.clone(),
            config.name.clone(),
            config.join_tokens.clone(),
        ))
        .abort_handle(),
    ];
    let status = membership_status(&config.name, &node, config.api_port).await;
    tracing::info!(
        membership = %config.name,
        api_port = config.api_port,
        endpoint_id = %status.endpoint_id,
        "mesh membership started"
    );
    let entry = mesh::MembershipEntry {
        node,
        api_port: config.api_port,
        tasks,
    };
    if let Some(raced) = ctx.node.memberships.insert(config.name, entry).await {
        raced.close().await;
    }
    Ok(status)
}

/// Stops a membership. Returns whether there was one by that name.
pub(super) async fn remove_membership(node: &mesh::Node, name: &str) -> bool {
    let Some(entry) = node.memberships.remove(name).await else {
        return false;
    };
    entry.close().await;
    tracing::info!(membership = %name, "mesh membership stopped");
    true
}

pub(super) async fn membership_statuses(node: &mesh::Node) -> Vec<MeshMembershipStatus> {
    let mut statuses = Vec::new();
    for (name, node, api_port) in node.memberships.list().await {
        statuses.push(membership_status(&name, &node, api_port).await);
    }
    statuses
}

async fn membership_status(name: &str, node: &mesh::Node, api_port: u16) -> MeshMembershipStatus {
    MeshMembershipStatus {
        name: name.to_owned(),
        api_base_url: format!("http://127.0.0.1:{api_port}/v1"),
        invite_token: node.invite_token().await,
        endpoint_id: node.id().to_string(),
        peers: node.connected_peer_ids().await.len(),
    }
}

fn validate(options: &RuntimeOptions, config: &MeshMembershipConfig) -> Result<()> {
    anyhow::ensure!(
        valid_name(&config.name),
        "membership name {:?} must be 1 to 64 letters, digits, '.', '_' or '-', not starting with '.'",
        config.name
    );
    anyhow::ensure!(
        config.api_port != 0
            && config.api_port != options.port
            && config.api_port != options.console,
        "membership API port {} must be fixed and differ from the node's API and console ports",
        config.api_port
    );
    Ok(())
}

fn valid_name(name: &str) -> bool {
    (1..=64).contains(&name.len())
        && !name.starts_with('.')
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

/// The membership's node key, made on first use. One key per membership keeps
/// its endpoint, and so its invite token, stable across restarts.
fn membership_key(config: &MeshMembershipConfig) -> Result<iroh::SecretKey> {
    let path = match &config.node_key_path {
        Some(path) => path.clone(),
        None => default_membership_key_path(&config.name)?,
    };
    if path.exists() {
        return mesh_llm_membership::load_node_key_from_path(&path)
            .with_context(|| format!("load membership key {}", path.display()));
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let key = iroh::SecretKey::generate();
    mesh_llm_membership::save_node_key_to_path(&path, &key)
        .with_context(|| format!("save membership key {}", path.display()))?;
    Ok(key)
}

fn default_membership_key_path(name: &str) -> Result<PathBuf> {
    let node_key = mesh_llm_membership::default_node_key_path()?;
    let dir = node_key
        .parent()
        .context("the node key path has no parent directory")?;
    Ok(dir.join("memberships").join(format!("{name}.key")))
}

/// Joins the membership's tokens, then keeps it joined and its peers checked
/// until the membership is removed.
async fn maintain_membership(node: mesh::Node, name: String, join_tokens: Vec<String>) {
    let mut last_failure: Option<String> = None;
    let mut heartbeat_failures = std::collections::HashMap::new();
    loop {
        let failure = join_all(&node, &join_tokens).await;
        if failure != last_failure {
            report_join_change(&name, failure.as_deref(), last_failure.is_some());
            last_failure = failure;
        }
        tokio::time::sleep(MEMBERSHIP_MAINTENANCE_INTERVAL).await;
        node.run_heartbeat_cycle(&mut heartbeat_failures).await;
        node.redial_join_targets().await;
    }
}

/// Joins every token, returning the last failure.
async fn join_all(node: &mesh::Node, join_tokens: &[String]) -> Option<String> {
    let mut failure = None;
    for token in join_tokens {
        if let Err(error) = node.join(token).await {
            failure = Some(format!("{error:#}"));
        }
    }
    failure
}

/// Logs a membership's joins starting to fail, or recovering, once each.
fn report_join_change(name: &str, failure: Option<&str>, was_failing: bool) {
    match failure {
        Some(error) => {
            tracing::warn!(membership = %name, "mesh membership join failed, retrying: {error}");
        }
        None if was_failing => {
            tracing::info!(membership = %name, "mesh membership join recovered");
        }
        None => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn membership_names_are_path_safe() {
        for good in ["personal", "gi-org_01", "a.b", "A9"] {
            assert!(valid_name(good), "{good}");
        }
        let long = "x".repeat(65);
        for bad in ["", ".hidden", "a/b", "..", "with space", long.as_str()] {
            assert!(!valid_name(bad), "{bad:?}");
        }
    }

    #[test]
    fn membership_ports_must_be_their_own() {
        let options = RuntimeOptions::default();
        let config = |api_port| MeshMembershipConfig {
            name: "team".into(),
            join_tokens: Vec::new(),
            trusted_owners: Vec::new(),
            api_port,
            node_key_path: None,
            enumerate_host: false,
        };
        assert!(validate(&options, &config(9338)).is_ok());
        for taken in [0, options.port, options.console] {
            assert!(validate(&options, &config(taken)).is_err(), "{taken}");
        }
    }
}
