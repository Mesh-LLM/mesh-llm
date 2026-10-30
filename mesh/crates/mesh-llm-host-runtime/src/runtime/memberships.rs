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
use iroh::{EndpointId, SecretKey};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// How often a membership re-dials its join tokens.
const MEMBERSHIP_JOIN_INTERVAL: Duration = Duration::from_secs(60);
/// How often a membership checks its peers, and renews its certificate.
const MEMBERSHIP_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(60);

/// A further mesh to serve this node's models into.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeshMembershipConfig {
    /// Names the membership on this node, and the mesh: every node of one
    /// mesh uses the same name. Lowercase letters, digits, `.`, `_` and `-`.
    pub name: String,
    /// Unsigned invite tokens of the mesh's other nodes, dialled now and
    /// again every minute while the membership lasts. This node's own token
    /// is skipped, so a list of every member's token can be passed as is.
    /// With none, this node waits to be joined by its own invite token.
    pub join_tokens: Vec<String>,
    /// The owner IDs whose nodes the membership admits. Nobody else is
    /// admitted, so an empty list admits no one.
    pub trusted_owners: Vec<String>,
    /// The loopback port of the membership's OpenAI API. Its requests run on
    /// this node's models when it serves them, and otherwise on the mesh's
    /// peers; never on the peers of the node's other meshes.
    pub api_port: u16,
    /// Where the membership's node key is kept. By default, beside the
    /// node's own key, under `memberships/<node endpoint ID>/<name>.key`.
    pub node_key_path: Option<PathBuf>,
    /// Whether to advertise the GPU name, hostname and benchmarks to this
    /// mesh.
    pub enumerate_host: bool,
}

/// A running membership.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct MeshMembershipStatus {
    pub name: String,
    pub api_base_url: String,
    /// How the mesh's other nodes join this one. It carries the node's
    /// addresses: the membership keeps its UDP port across restarts when it
    /// can, and when it cannot, the token changes and has to be shared again.
    pub invite_token: String,
    pub endpoint_id: String,
    /// Admitted peers this membership knows of in its mesh.
    pub peers: usize,
    /// Why the latest round of joins failed, if it did.
    pub join_error: Option<String>,
}

/// What a membership needs from the running node.
pub(super) struct MembershipContext {
    pub(super) options: Arc<RuntimeOptions>,
    pub(super) node: mesh::Node,
    pub(super) targets: tokio::sync::watch::Receiver<election::ModelTargets>,
}

/// What a membership is checked for before the one it replaces stops.
struct Prepared {
    secret_key: SecretKey,
    key_path: PathBuf,
    /// The tokens to dial: the membership's own is left out.
    join_tokens: Vec<String>,
    /// Bound up front when the port is not the replaced membership's own.
    listener: Option<tokio::net::TcpListener>,
}

/// Starts a membership, replacing one of the same name. A replacement that
/// fails its checks (name, port, tokens, key) leaves the old one running. One
/// that fails later stops the old one first and then starts it again; the
/// error says whether that worked.
pub(super) async fn add_membership(
    ctx: &MembershipContext,
    config: MeshMembershipConfig,
) -> Result<MeshMembershipStatus> {
    let _changes = ctx.node.memberships.lock_changes().await;
    let prepared = prepare(ctx, &config).await?;
    let Some(previous) = ctx.node.memberships.remove(&config.name).await else {
        return start(ctx, config, prepared, None).await;
    };
    let quic_port = previous.quic_port();
    let previous_config = previous.config.clone();
    previous.close().await;
    let error = match start(ctx, config, prepared, quic_port).await {
        Ok(status) => return Ok(status),
        Err(error) => error,
    };
    let restored = match prepare(ctx, &previous_config).await {
        Ok(prepared) => start(ctx, previous_config, prepared, quic_port).await,
        Err(restore_error) => Err(restore_error),
    };
    match restored {
        Ok(_) => Err(error.context("the membership it was replacing was restarted")),
        Err(restore_error) => Err(error.context(format!(
            "the membership it was replacing is stopped and could not be restarted: {restore_error:#}"
        ))),
    }
}

/// Stops a membership. Returns whether there was one by that name.
pub(super) async fn remove_membership(node: &mesh::Node, name: &str) -> bool {
    let _changes = node.memberships.lock_changes().await;
    let Some(entry) = node.memberships.remove(name).await else {
        return false;
    };
    entry.close().await;
    tracing::info!(membership = %name, "mesh membership stopped");
    true
}

pub(super) async fn membership_statuses(node: &mesh::Node) -> Vec<MeshMembershipStatus> {
    let memberships = node
        .memberships
        .with_each(|name, entry| {
            let join_error = entry.join_error.lock().ok().and_then(|error| error.clone());
            (
                name.to_owned(),
                entry.node.clone(),
                entry.api_port,
                join_error,
            )
        })
        .await;
    let mut statuses = Vec::new();
    for (name, node, api_port, join_error) in memberships {
        statuses.push(membership_status(name, &node, api_port, join_error).await);
    }
    statuses
}

async fn membership_status(
    name: String,
    node: &mesh::Node,
    api_port: u16,
    join_error: Option<String>,
) -> MeshMembershipStatus {
    MeshMembershipStatus {
        name,
        api_base_url: format!("http://127.0.0.1:{api_port}/v1"),
        invite_token: node.invite_token().await,
        endpoint_id: node.id().to_string(),
        peers: node.peers().await.len(),
        join_error,
    }
}

/// Every check that needs no endpoint, run while the membership being
/// replaced still serves.
async fn prepare(ctx: &MembershipContext, config: &MeshMembershipConfig) -> Result<Prepared> {
    validate(&ctx.options, config)?;
    let others = ctx
        .node
        .memberships
        .with_each(|name, entry| (name.to_owned(), entry.api_port, entry.key_path.clone()))
        .await;
    let replaced_port = others
        .iter()
        .find(|(name, ..)| *name == config.name)
        .map(|(_, port, _)| *port);
    let key_path = match &config.node_key_path {
        Some(path) => path.clone(),
        None => default_membership_key_path(ctx.node.id(), &config.name)?,
    };
    for (name, port, path) in others.iter().filter(|(name, ..)| *name != config.name) {
        anyhow::ensure!(
            *port != config.api_port,
            "membership API port {} is already membership {name:?}'s",
            config.api_port
        );
        anyhow::ensure!(
            *path != key_path,
            "membership key {} is already membership {name:?}'s",
            key_path.display()
        );
    }
    anyhow::ensure!(
        mesh_llm_membership::default_node_key_path().ok().as_ref() != Some(&key_path),
        "a membership cannot use the node's own key {}",
        key_path.display()
    );
    let secret_key = load_or_create_key(key_path.clone()).await?;
    let join_tokens = checked_join_tokens(&ctx.node, secret_key.public(), &config.join_tokens)?;
    let listener = match replaced_port {
        Some(port) if port == config.api_port => None,
        _ => Some(bind_api_port(config.api_port).await?),
    };
    Ok(Prepared {
        secret_key,
        key_path,
        join_tokens,
        listener,
    })
}

/// The join tokens to dial: each must be an unsigned invite token naming
/// another node of this mesh, not one of this node's other meshes.
fn checked_join_tokens(
    node: &mesh::Node,
    own: EndpointId,
    tokens: &[String],
) -> Result<Vec<String>> {
    let mut dialled = Vec::new();
    for (index, token) in tokens.iter().enumerate() {
        let target = mesh::membership_token_endpoint(token)
            .with_context(|| format!("join token {index}"))?;
        if target == own {
            continue;
        }
        anyhow::ensure!(
            !node.local_endpoints.contains(target),
            "join token {index} names another mesh this node serves"
        );
        dialled.push(token.clone());
    }
    Ok(dialled)
}

async fn bind_api_port(port: u16) -> Result<tokio::net::TcpListener> {
    tokio::net::TcpListener::bind(("127.0.0.1", port))
        .await
        .with_context(|| format!("bind membership API port {port}"))
}

async fn start(
    ctx: &MembershipContext,
    config: MeshMembershipConfig,
    prepared: Prepared,
    quic_port: Option<u16>,
) -> Result<MeshMembershipStatus> {
    let quic_port = quic_port.or_else(|| read_quic_port(&prepared.key_path));
    let relay_auths: std::collections::HashMap<String, String> =
        ctx.options.relay_auth.iter().cloned().collect();
    let (node, channels) = ctx
        .node
        .start_membership_node(mesh::MembershipNodeConfig {
            mesh_name: &config.name,
            secret_key: prepared.secret_key,
            relay: mesh::RelayConfig {
                urls: &ctx.options.relay,
                auths: &relay_auths,
                policy: relay_policy_for_runtime_options(&ctx.options),
            },
            quic_port,
            trusted_owners: &config.trusted_owners,
            enumerate_host: config.enumerate_host,
        })
        .await?;
    // Bound once the endpoint is up, so a local client never waits on the
    // relay; bound earlier only when a replacement moves to a new port.
    let listener = match prepared.listener {
        Some(listener) => listener,
        None => match bind_api_port(config.api_port).await {
            Ok(listener) => listener,
            Err(error) => {
                node.close_endpoint().await;
                ctx.node.local_endpoints.remove(node.id());
                return Err(error);
            }
        },
    };
    let join_error = Arc::new(Mutex::new(None));
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
            ctx.targets.clone(),
            Some(IngressListener::from(listener)),
            false,
            affinity,
        )),
        tokio::spawn(keep_joined(
            node.clone(),
            config.name.clone(),
            prepared.join_tokens,
            join_error.clone(),
        )),
        tokio::spawn(heartbeat(node.clone())),
    ];
    let entry = mesh::MembershipEntry {
        node: node.clone(),
        api_port: config.api_port,
        key_path: prepared.key_path,
        config: config.clone(),
        join_error,
        tasks,
    };
    if let Some(port) = entry.quic_port() {
        write_quic_port(&entry.key_path, port);
    }
    if let Err(entry) = ctx
        .node
        .memberships
        .insert(config.name.clone(), entry)
        .await
    {
        entry.close().await;
        anyhow::bail!("the node is shutting down");
    }
    let status = membership_status(config.name.clone(), &node, config.api_port, None).await;
    tracing::info!(
        membership = %config.name,
        api_port = config.api_port,
        endpoint_id = %status.endpoint_id,
        "mesh membership started"
    );
    Ok(status)
}

fn validate(options: &RuntimeOptions, config: &MeshMembershipConfig) -> Result<()> {
    anyhow::ensure!(
        valid_name(&config.name),
        "membership name {:?} must be 1 to 64 lowercase letters, digits, '.', '_' or '-', not starting with '.'",
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

/// Lowercase only, so two names never share a key file on a case-insensitive
/// filesystem.
fn valid_name(name: &str) -> bool {
    mesh_llm_config::memberships::valid_membership_name(name)
}

/// The membership's node key, made on first use. One key per membership keeps
/// its endpoint ID stable across restarts.
async fn load_or_create_key(path: PathBuf) -> Result<SecretKey> {
    tokio::task::spawn_blocking(move || {
        if path.exists() {
            return mesh_llm_membership::load_node_key_from_path(&path)
                .with_context(|| format!("load membership key {}", path.display()));
        }
        if let Some(parent) = path.parent() {
            create_private_dir(parent)?;
        }
        let key = SecretKey::generate();
        mesh_llm_membership::save_node_key_to_path(&path, &key)
            .with_context(|| format!("save membership key {}", path.display()))?;
        Ok(key)
    })
    .await
    .context("membership key task")?
}

fn create_private_dir(dir: &Path) -> Result<()> {
    let mut builder = std::fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
    builder
        .create(dir)
        .with_context(|| format!("create {}", dir.display()))
}

/// Beside the node's own key, and under the node's endpoint ID, so two nodes
/// sharing a home directory never load the same membership key.
fn default_membership_key_path(node: EndpointId, name: &str) -> Result<PathBuf> {
    let node_key = mesh_llm_membership::default_node_key_path()?;
    let dir = node_key
        .parent()
        .context("the node key path has no parent directory")?;
    Ok(dir
        .join("memberships")
        .join(hex::encode(node.as_bytes()))
        .join(format!("{name}.key")))
}

fn quic_port_path(key_path: &Path) -> PathBuf {
    key_path.with_extension("port")
}

/// The UDP port the membership last bound, kept beside its key so its invite
/// token survives a restart.
fn read_quic_port(key_path: &Path) -> Option<u16> {
    std::fs::read_to_string(quic_port_path(key_path))
        .ok()?
        .trim()
        .parse()
        .ok()
}

fn write_quic_port(key_path: &Path, port: u16) {
    if let Err(error) = std::fs::write(quic_port_path(key_path), port.to_string()) {
        tracing::debug!("could not record the membership's UDP port: {error}");
    }
}

/// Dials the membership's join tokens, all at once, now and every minute
/// until the membership is removed, recording the latest failures.
async fn keep_joined(
    node: mesh::Node,
    name: String,
    join_tokens: Vec<String>,
    join_error: Arc<Mutex<Option<String>>>,
) {
    let mut ticks = tokio::time::interval(MEMBERSHIP_JOIN_INTERVAL);
    let mut was_failing = false;
    loop {
        ticks.tick().await;
        let failure = join_all(&node, &join_tokens).await;
        if failure.is_some() != was_failing {
            report_join_change(&name, failure.as_deref());
            was_failing = failure.is_some();
        }
        if let Ok(mut error) = join_error.lock() {
            *error = failure;
        }
    }
}

/// Joins every token concurrently, returning what failed, if anything did.
async fn join_all(node: &mesh::Node, join_tokens: &[String]) -> Option<String> {
    let results =
        futures_util::future::join_all(join_tokens.iter().map(|token| node.join_membership(token)))
            .await;
    let failures: Vec<String> = results
        .into_iter()
        .filter_map(|result| result.err().map(|error| format!("{error:#}")))
        .collect();
    (!failures.is_empty()).then(|| {
        format!(
            "{} of {} join tokens failed: {}",
            failures.len(),
            join_tokens.len(),
            failures.join("; ")
        )
    })
}

/// Logs a membership's joins starting to fail, or recovering, once each.
fn report_join_change(name: &str, failure: Option<&str>) {
    match failure {
        Some(error) => {
            tracing::warn!(membership = %name, "mesh membership join failed, retrying: {error}");
        }
        None => tracing::info!(membership = %name, "mesh membership join recovered"),
    }
}

/// Checks the membership's peers and renews its certificate, on its own
/// timer so slow joins never hold it back.
async fn heartbeat(node: mesh::Node) {
    let mut ticks = tokio::time::interval(MEMBERSHIP_HEARTBEAT_INTERVAL);
    ticks.tick().await;
    let mut failures = std::collections::HashMap::new();
    loop {
        ticks.tick().await;
        node.run_heartbeat_cycle(&mut failures).await;
        node.redial_join_targets().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[test]
    fn membership_names_are_path_safe() {
        for good in ["personal", "gi-org_01", "a.b", "a9"] {
            assert!(valid_name(good), "{good}");
        }
        let long = "x".repeat(65);
        for bad in [
            "",
            ".hidden",
            "a/b",
            "..",
            "with space",
            "Team",
            long.as_str(),
        ] {
            assert!(!valid_name(bad), "{bad:?}");
        }
    }

    #[test]
    fn membership_ports_must_be_their_own() {
        let options = RuntimeOptions::default();
        let config = |api_port| MeshMembershipConfig {
            api_port,
            ..team_config(PathBuf::from("team.key"), 9338)
        };
        assert!(validate(&options, &config(9338)).is_ok());
        for taken in [0, options.port, options.console] {
            assert!(validate(&options, &config(taken)).is_err(), "{taken}");
        }
    }

    fn team_config(key_path: PathBuf, api_port: u16) -> MeshMembershipConfig {
        MeshMembershipConfig {
            name: "team".into(),
            join_tokens: Vec::new(),
            trusted_owners: Vec::new(),
            api_port,
            node_key_path: Some(key_path),
            enumerate_host: false,
        }
    }

    /// A loopback port held open, so nothing else takes it, until the test
    /// lets it go.
    fn held_port() -> std::net::TcpListener {
        std::net::TcpListener::bind("127.0.0.1:0").expect("loopback port")
    }

    fn port_of(listener: &std::net::TcpListener) -> u16 {
        listener.local_addr().expect("loopback address").port()
    }

    async fn membership_context(
        targets: election::ModelTargets,
        with_owner: bool,
    ) -> Result<(
        MembershipContext,
        tokio::sync::watch::Sender<election::ModelTargets>,
    )> {
        let mut node = mesh::Node::new_for_tests(mesh::NodeRole::Worker).await?;
        if with_owner {
            node.owner_keypair = Some(crate::crypto::OwnerKeypair::generate());
        }
        node.quic_bind.ip = Some(std::net::Ipv4Addr::LOCALHOST.into());
        let (targets_tx, targets) = tokio::sync::watch::channel(targets);
        let ctx = MembershipContext {
            options: Arc::new(RuntimeOptions {
                disable_iroh_relays: true,
                ..RuntimeOptions::default()
            }),
            node,
            targets,
        };
        Ok((ctx, targets_tx))
    }

    async fn answers(port: u16) -> bool {
        tokio::net::TcpStream::connect(("127.0.0.1", port))
            .await
            .is_ok()
    }

    async fn assert_current_invite(
        ctx: &MembershipContext,
        config: &MeshMembershipConfig,
        status: &MeshMembershipStatus,
    ) -> Result<std::net::SocketAddr> {
        let bound = ctx
            .node
            .memberships
            .with_each(|_, entry| (entry.node.id(), entry.node.endpoint.bound_sockets()))
            .await;
        assert_eq!(bound.len(), 1);
        let (id, sockets) = &bound[0];
        assert_eq!(sockets.len(), 1, "the test binds only IPv4 loopback");
        let addr = sockets[0];
        let invite = mesh::Node::decode_invite_token(&status.invite_token)?;
        assert_eq!(invite.id, *id);
        assert_eq!(status.endpoint_id, id.to_string());
        assert_eq!(invite.addrs.len(), 1);
        assert!(invite.addrs.contains(&iroh::TransportAddr::Ip(addr)));
        assert_eq!(
            read_quic_port(config.node_key_path.as_ref().expect("test key path")),
            Some(addr.port())
        );
        Ok(addr)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_membership_is_replaced_on_its_own_port_and_removed() -> Result<()> {
        let (ctx, _targets) = membership_context(election::ModelTargets::default(), true).await?;
        let keys = tempfile::tempdir()?;
        let port = port_of(&held_port());
        let config = team_config(keys.path().join("team.key"), port);

        let first = add_membership(&ctx, config.clone()).await?;
        let first_addr = assert_current_invite(&ctx, &config, &first).await?;
        // The replacement keeps its key and API port; its own token is
        // skipped. UDP reuse is best-effort: iroh retains a closed socket
        // until the last endpoint clone is dropped, including serving tasks.
        let replaced = add_membership(
            &ctx,
            MeshMembershipConfig {
                join_tokens: vec![first.invite_token.clone()],
                ..config.clone()
            },
        )
        .await?;
        assert_eq!(replaced.endpoint_id, first.endpoint_id);
        let replaced_addr = assert_current_invite(&ctx, &config, &replaced).await?;
        if replaced_addr == first_addr {
            assert_eq!(replaced.invite_token, first.invite_token);
        }
        assert_eq!(membership_statuses(&ctx.node).await, vec![replaced]);
        assert!(answers(port).await);

        assert!(remove_membership(&ctx.node, "team").await);
        assert!(!remove_membership(&ctx.node, "team").await);
        assert!(membership_statuses(&ctx.node).await.is_empty());
        assert!(!answers(port).await);
        ctx.node.close_endpoint().await;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_replacement_reports_and_persists_its_fallback_udp_port() -> Result<()> {
        let (ctx, _targets) = membership_context(election::ModelTargets::default(), true).await?;
        let keys = tempfile::tempdir()?;
        let port = port_of(&held_port());
        let config = team_config(keys.path().join("team.key"), port);
        let first = add_membership(&ctx, config.clone()).await?;
        let first_addr = assert_current_invite(&ctx, &config, &first).await?;
        // Keep the old UDP socket alive even after close, deterministically
        // exercising the documented fallback instead of relying on timing.
        let held_endpoint = ctx
            .node
            .memberships
            .with_each(|_, entry| entry.node.endpoint.clone())
            .await
            .pop()
            .expect("running membership");
        let replaced = add_membership(
            &ctx,
            MeshMembershipConfig {
                join_tokens: vec![first.invite_token.clone()],
                ..config.clone()
            },
        )
        .await?;
        assert!(held_endpoint.is_closed());
        assert_eq!(replaced.endpoint_id, first.endpoint_id);
        assert_ne!(
            assert_current_invite(&ctx, &config, &replaced).await?,
            first_addr
        );
        assert_ne!(replaced.invite_token, first.invite_token);
        assert!(answers(port).await);
        assert!(remove_membership(&ctx.node, "team").await);
        drop(held_endpoint);
        ctx.node.close_endpoint().await;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_refused_replacement_leaves_the_membership_running() -> Result<()> {
        let (ctx, _targets) = membership_context(election::ModelTargets::default(), true).await?;
        let keys = tempfile::tempdir()?;
        let port = port_of(&held_port());
        let config = team_config(keys.path().join("team.key"), port);
        add_membership(&ctx, config.clone()).await?;
        let other_port = port_of(&held_port());
        add_membership(
            &ctx,
            MeshMembershipConfig {
                name: "other".into(),
                ..team_config(keys.path().join("other.key"), other_port)
            },
        )
        .await?;

        let busy = held_port();
        let refusals = [
            // Another membership's port.
            MeshMembershipConfig {
                api_port: other_port,
                ..config.clone()
            },
            // Another membership's key.
            MeshMembershipConfig {
                node_key_path: Some(keys.path().join("other.key")),
                ..config.clone()
            },
            // A port something else holds.
            MeshMembershipConfig {
                api_port: port_of(&busy),
                ..config.clone()
            },
            // A token that is not an invite token.
            MeshMembershipConfig {
                join_tokens: vec!["not a token".into()],
                ..config.clone()
            },
            // The node's own invite token.
            MeshMembershipConfig {
                join_tokens: vec![ctx.node.invite_token().await],
                ..config.clone()
            },
        ];
        for refused in refusals {
            assert!(add_membership(&ctx, refused).await.is_err());
            assert!(answers(port).await, "the membership keeps serving");
            assert_eq!(membership_statuses(&ctx.node).await.len(), 2);
        }
        ctx.node.close_memberships().await;
        ctx.node.close_endpoint().await;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_replacement_that_fails_to_start_says_the_old_one_stopped() -> Result<()> {
        let (ctx, _targets) = membership_context(election::ModelTargets::default(), true).await?;
        let keys = tempfile::tempdir()?;
        let port = port_of(&held_port());
        add_membership(&ctx, team_config(keys.path().join("team.key"), port)).await?;
        // The same node and memberships, without the owner key a membership
        // node needs: the checks pass, the start does not.
        let (mut keyless, _keyless_targets) =
            membership_context(election::ModelTargets::default(), false).await?;
        keyless.node.memberships = ctx.node.memberships.clone();
        keyless.node.local_endpoints = ctx.node.local_endpoints.clone();
        let error = add_membership(&keyless, team_config(keys.path().join("team.key"), port))
            .await
            .expect_err("a membership node needs an owner key");
        assert!(format!("{error:#}").contains("stopped"), "{error:#}");
        assert!(membership_statuses(&ctx.node).await.is_empty());
        ctx.node.close_endpoint().await;
        keyless.node.close_endpoint().await;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_membership_api_request_runs_on_the_nodes_model() -> Result<()> {
        let backend = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let backend_port = backend.local_addr()?.port();
        let captured = tokio::spawn(async move {
            let (mut stream, _) = backend.accept().await?;
            let mut request = vec![0; 8192];
            let read = stream.read(&mut request).await?;
            request.truncate(read);
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                .await?;
            Ok::<_, anyhow::Error>(request)
        });
        let mut targets = election::ModelTargets::default();
        targets.targets.insert(
            "team-model".into(),
            vec![election::InferenceTarget::Local(backend_port)],
        );
        let (ctx, _targets) = membership_context(targets, true).await?;
        ctx.node.set_hosted_models(vec!["team-model".into()]).await;
        let keys = tempfile::tempdir()?;
        let port = port_of(&held_port());
        let team = add_membership(&ctx, team_config(keys.path().join("team.key"), port)).await?;
        assert_eq!(team.api_base_url, format!("http://127.0.0.1:{port}/v1"));

        let body = r#"{"model":"team-model","messages":[{"role":"user","content":"hello"}],"max_tokens":1}"#;
        let mut client = tokio::net::TcpStream::connect(("127.0.0.1", port)).await?;
        client
            .write_all(
                format!(
                    "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                    body.len()
                )
                .as_bytes(),
            )
            .await?;
        let request = tokio::time::timeout(Duration::from_secs(5), captured).await???;
        assert!(String::from_utf8_lossy(&request).contains("team-model"));
        ctx.node.close_memberships().await;
        ctx.node.close_endpoint().await;
        Ok(())
    }
}
