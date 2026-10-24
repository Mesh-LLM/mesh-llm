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
    /// Unsigned invite tokens of the mesh's nodes, joined now and re-dialled
    /// while the membership lasts. This node's own token is skipped, so a
    /// list of every member's token can be passed as is. With none, this node
    /// waits to be joined by its own invite token.
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

/// Starts a membership, replacing one of the same name. A replacement that
/// fails its port or key checks leaves the old one running; past those, the
/// old one stops first, so if the new one cannot start, neither runs.
pub(super) async fn add_membership(
    ctx: MembershipContext<'_>,
    config: MeshMembershipConfig,
) -> Result<MeshMembershipStatus> {
    validate(ctx.options, &config)?;
    // Checks that need no port or endpoint run before the membership being
    // replaced stops, so failing them leaves it running.
    let memberships = ctx.node.memberships.list().await;
    anyhow::ensure!(
        memberships
            .iter()
            .all(|(name, _, port)| *name == config.name || *port != config.api_port),
        "membership API port {} is already another membership's",
        config.api_port
    );
    let secret_key = membership_key(&config)?;
    if let Some(previous) = ctx.node.memberships.remove(&config.name).await {
        previous.close().await;
    }
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", config.api_port))
        .await
        .with_context(|| format!("bind membership API port {}", config.api_port))?;
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
        )),
        tokio::spawn(super::proxy::api_proxy(
            node.clone(),
            config.api_port,
            ctx.targets,
            Some(IngressListener::from(listener)),
            false,
            affinity,
        )),
        tokio::spawn(maintain_membership(
            node.clone(),
            config.name.clone(),
            config.join_tokens.clone(),
        )),
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
        if let Err(error) = node.join_membership(token).await {
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

    fn free_port() -> u16 {
        std::net::TcpListener::bind("127.0.0.1:0")
            .and_then(|listener| listener.local_addr())
            .map(|addr| addr.port())
            .expect("free loopback port")
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_membership_is_replaced_on_its_own_port_and_removed() -> Result<()> {
        let mut node = mesh::Node::new_for_tests(mesh::NodeRole::Worker).await?;
        node.owner_keypair = Some(crate::crypto::OwnerKeypair::generate());
        node.quic_bind.ip = Some(std::net::Ipv4Addr::LOCALHOST.into());
        let options = RuntimeOptions {
            disable_iroh_relays: true,
            ..RuntimeOptions::default()
        };
        let keys = tempfile::tempdir()?;
        let port = free_port();
        let (_targets_tx, targets) = tokio::sync::watch::channel(election::ModelTargets::default());
        let ctx = || MembershipContext {
            options: &options,
            node: &node,
            targets: targets.clone(),
        };
        let config = MeshMembershipConfig {
            name: "team".into(),
            join_tokens: Vec::new(),
            trusted_owners: Vec::new(),
            api_port: port,
            node_key_path: Some(keys.path().join("team.key")),
            enumerate_host: false,
        };

        let first = add_membership(ctx(), config.clone()).await?;
        // The replacement takes the same port the first one held, and keeps
        // its key; its own token among the join tokens is skipped.
        let replaced = add_membership(
            ctx(),
            MeshMembershipConfig {
                join_tokens: vec![first.invite_token.clone()],
                ..config.clone()
            },
        )
        .await?;
        assert_eq!(replaced.endpoint_id, first.endpoint_id);
        assert_eq!(membership_statuses(&node).await, vec![replaced]);
        tokio::net::TcpStream::connect(("127.0.0.1", port)).await?;

        // A replacement that takes another membership's port is refused,
        // and the membership it would have replaced keeps running.
        let other_port = free_port();
        add_membership(
            ctx(),
            MeshMembershipConfig {
                name: "other".into(),
                api_port: other_port,
                node_key_path: Some(keys.path().join("other.key")),
                ..config.clone()
            },
        )
        .await?;
        let clash = MeshMembershipConfig {
            api_port: other_port,
            ..config.clone()
        };
        assert!(add_membership(ctx(), clash).await.is_err());
        tokio::net::TcpStream::connect(("127.0.0.1", port)).await?;
        assert!(remove_membership(&node, "other").await);

        assert!(remove_membership(&node, "team").await);
        assert!(!remove_membership(&node, "team").await);
        assert!(membership_statuses(&node).await.is_empty());
        assert!(
            tokio::net::TcpStream::connect(("127.0.0.1", port))
                .await
                .is_err()
        );
        node.close_endpoint().await;
        Ok(())
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
