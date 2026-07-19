use super::*;

#[test]
fn mdns_discovery_uses_lan_only_relay_policy() {
    assert_eq!(
        relay_policy_for_mesh_discovery_mode(mesh_discovery::MeshDiscoveryMode::Mdns),
        mesh::RelayPolicy::Disabled
    );
    assert_eq!(
        relay_policy_for_mesh_discovery_mode(mesh_discovery::MeshDiscoveryMode::Nostr),
        mesh::RelayPolicy::DefaultPublic
    );
}

#[test]
fn explicit_disable_iroh_relays_overrides_nostr_relay_policy() {
    let options = RuntimeOptions {
        mesh_discovery_mode: mesh_discovery::MeshDiscoveryMode::Nostr,
        disable_iroh_relays: true,
        ..RuntimeOptions::default()
    };

    assert_eq!(
        relay_policy_for_runtime_options(&options),
        mesh::RelayPolicy::ExplicitlyDisabled
    );
    assert!(!relay_policy_for_runtime_options(&options).uses_relay());
}

#[test]
fn mdns_discovery_does_not_start_relay_health_monitor() {
    assert!(!should_start_relay_health_monitor(
        mesh_discovery::MeshDiscoveryMode::Mdns
    ));
}

#[test]
fn nostr_discovery_starts_relay_health_monitor() {
    assert!(should_start_relay_health_monitor(
        mesh_discovery::MeshDiscoveryMode::Nostr
    ));
}

#[test]
fn mdns_discovery_starts_lan_rediscovery_only_with_join_token() {
    assert!(should_start_lan_rediscovery(
        mesh_discovery::MeshDiscoveryMode::Mdns,
        &["join-token".to_string()]
    ));
    assert!(!should_start_lan_rediscovery(
        mesh_discovery::MeshDiscoveryMode::Mdns,
        &[]
    ));
    assert!(!should_start_lan_rediscovery(
        mesh_discovery::MeshDiscoveryMode::Nostr,
        &["join-token".to_string()]
    ));
}

fn make_cli(args: &[&str]) -> RuntimeOptions {
    runtime_options_for_test(args)
}

fn make_runtime_cli(args: &[&str]) -> RuntimeOptions {
    runtime_options_for_test(args)
}

#[test]
fn swarm_capture_client_registers_runtime_owner() {
    let options = make_runtime_cli(&[
        "mesh-llm",
        "client",
        "--auto",
        "--swarm-capture",
        "/tmp/mesh-capture",
    ]);

    assert!(options.client);
    assert!(swarm_capture_observer_requested(&options));
}

#[test]
fn plain_client_still_skips_runtime_owner_registration() {
    let options = make_runtime_cli(&["mesh-llm", "client", "--auto"]);

    assert!(options.client);
    assert!(!swarm_capture_observer_requested(&options));
}

#[test]
#[serial_test::serial]
fn swarm_capture_env_client_registers_runtime_owner() {
    let key = crate::capture::SWARM_CAPTURE_ENV;
    let _env_guard = EnvVarGuard::set(key, "/tmp/mesh-capture");
    let options = make_runtime_cli(&["mesh-llm", "client", "--auto"]);

    assert!(swarm_capture_observer_requested(&options));
}

struct EnvVarGuard {
    key: &'static str,
    previous: Option<std::ffi::OsString>,
}

impl EnvVarGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let guard = Self {
            key,
            previous: std::env::var_os(key),
        };
        // SAFETY: these serial tests mutate the process environment before
        // building runtime options and restore it via Drop before the next test.
        unsafe { std::env::set_var(key, value) };
        guard
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        restore_env(self.key, self.previous.take());
    }
}

#[test]
fn mesh_name_does_not_force_publish() {
    let options = make_cli(&[
        "mesh-llm",
        "--model",
        "dummy-model",
        "--mesh-name",
        "my-mesh",
    ]);
    assert!(!options.publish, "mesh_name alone must not set publish");
    assert_eq!(options.mesh_name.as_deref(), Some("my-mesh"));
}

#[test]
fn explicit_publish_remains_enabled() {
    let options = make_cli(&["mesh-llm", "--model", "dummy-model", "--publish"]);
    assert!(
        options.publish,
        "explicit --publish must set publish=true even without mesh_name"
    );
}

#[test]
fn publish_with_mesh_name_is_public_and_named() {
    let options = make_cli(&[
        "mesh-llm",
        "--model",
        "dummy-model",
        "--publish",
        "--mesh-name",
        "named-public",
    ]);
    assert!(
        options.publish,
        "publish + mesh_name must keep publish=true"
    );
    assert_eq!(
        options.mesh_name.as_deref(),
        Some("named-public"),
        "mesh_name must be preserved alongside publish"
    );
}

#[test]
fn auto_without_publish_stays_private() {
    let options = make_cli(&["mesh-llm", "--model", "dummy-model", "--auto"]);
    assert!(!options.publish, "--auto alone must not imply publish");
    assert!(options.auto, "--auto flag should still be true");
}

/// Task 2: Named private mesh keeps private identity (no implicit publish).
#[test]
fn named_private_mesh_keeps_private_identity() {
    // A named mesh without --publish must have publish=false.
    // The is_public gate in runtime startup uses `options.auto || options.publish`,
    // so a named-only mesh should NOT trigger public identity handling.
    let options = make_cli(&[
        "mesh-llm",
        "--model",
        "dummy-model",
        "--mesh-name",
        "private-named",
    ]);
    assert!(!options.publish);
    assert!(!options.auto);
    let is_public = options.auto || options.publish;
    assert!(
        !is_public,
        "named-only mesh must be treated as private for identity purposes"
    );
}

/// Task 3: start_new_mesh helper does not auto-enable publish.
#[test]
fn start_new_mesh_does_not_auto_enable_publish() {
    use crate::runtime::discovery::start_new_mesh;
    let mut options = make_cli(&["mesh-llm", "--model", "dummy-model"]);
    assert!(!options.publish, "precondition: publish starts false");
    start_new_mesh(&mut options, &["dummy-model".to_string()], 16.0, false);
    assert!(
        !options.publish,
        "start_new_mesh must NOT set publish=true when it was not requested"
    );
}

/// Task 3: Explicit --publish survives start_new_mesh unchanged.
#[test]
fn start_new_mesh_preserves_explicit_publish() {
    use crate::runtime::discovery::start_new_mesh;
    let mut options = make_cli(&["mesh-llm", "--model", "dummy-model", "--publish"]);
    assert!(options.publish, "precondition: publish is true");
    start_new_mesh(&mut options, &["dummy-model".to_string()], 16.0, false);
    assert!(
        options.publish,
        "explicit --publish must survive start_new_mesh call"
    );
}

#[test]
fn bootstrap_proxy_gate_fires_when_cli_join_is_set() {
    // Classic invite-token path (`--join <token>`).
    let options = runtime_options_for_test(&["mesh-llm", "--join", "tok-abc"]);
    assert!(should_start_bootstrap_proxy(&options, &[]));
}

#[test]
fn bootstrap_proxy_gate_fires_for_serve_auto_via_auto_join_candidates() {
    // serve --auto leaves options.join empty and stages discovery results in
    // auto_join_candidates instead. The proxy must still spawn so :9337
    // proxies through the mesh while the local GPU loads.
    let options = runtime_options_for_test(&["mesh-llm", "--auto"]);
    assert!(
        options.join.is_empty(),
        "precondition: serve --auto has empty options.join"
    );
    let candidates = vec![(
        "tok-from-discovery".to_string(),
        Some("mesh-llm".to_string()),
    )];
    assert!(should_start_bootstrap_proxy(&options, &candidates));
}

#[test]
fn bootstrap_proxy_gate_does_not_fire_for_client_auto_with_no_candidates() {
    // --client --auto with zero discovery results: nothing to tunnel to.
    // This matches the pre-1bd62389 behavior — the gate stays closed
    // until discovery turns up a peer, at which point handle_auto_decision
    // populates options.join and the gate fires on the next pass through
    // run_auto. We don't pre-bind the proxy speculatively for --client.
    let options = runtime_options_for_test(&["mesh-llm", "--client", "--auto"]);
    assert!(!should_start_bootstrap_proxy(&options, &[]));
}

#[test]
fn bootstrap_proxy_gate_fires_for_client_auto_with_join_populated() {
    // --client --auto with a successful discovery hit: handle_auto_decision
    // pushed the token into options.join, so the gate fires (unchanged from
    // pre-regression behavior).
    let options = runtime_options_for_test(&["mesh-llm", "--client", "--auto", "--join", "tok-x"]);
    assert!(should_start_bootstrap_proxy(&options, &[]));
}

#[test]
fn bootstrap_proxy_gate_does_not_fire_for_standalone_serve() {
    // Plain `mesh-llm` with no join, no auto candidates, no --client:
    // this node intends to start a new mesh standalone. Nothing to tunnel
    // through, so the bootstrap proxy should stay quiet.
    let options = runtime_options_for_test(&["mesh-llm"]);
    assert!(!should_start_bootstrap_proxy(&options, &[]));
}

#[test]
fn serve_auto_prefers_fast_join_probe_for_discovered_candidates() {
    let options = runtime_options_for_test(&["mesh-llm", "--auto"]);
    let candidates = vec![("tok-from-discovery".to_string(), None)];
    assert!(
        should_prefer_fast_auto_join(&options, &candidates),
        "serve --auto should avoid serial retry when discovery found candidates"
    );
}

#[test]
fn explicit_serve_join_keeps_serial_join_path() {
    let options = runtime_options_for_test(&["mesh-llm", "serve", "--join", "tok-explicit"]);
    assert!(
        !should_prefer_fast_auto_join(&options, &[]),
        "explicit serve --join keeps the established serial join path"
    );
}

#[test]
fn explicit_serve_join_ignores_discovered_fast_join_candidates() {
    let options = runtime_options_for_test(&["mesh-llm", "serve", "--join", "tok-explicit"]);
    let candidates = vec![("tok-from-discovery".to_string(), None)];
    assert!(
        !should_prefer_fast_auto_join(&options, &candidates),
        "explicit serve --join should not be switched to discovery fast-probe"
    );
}

#[test]
fn client_auto_keeps_fast_join_probe() {
    let options = runtime_options_for_test(&["mesh-llm", "--client", "--auto"]);
    assert!(
        should_prefer_fast_auto_join(&options, &[]),
        "client auto-join keeps the existing fast probe behavior"
    );
}

#[tokio::test]
async fn bootstrap_proxy_binds_listener_for_serve_auto() {
    // End-to-end check: `serve --auto` with a non-empty auto_join_candidates
    // vec must actually bind a TCP listener on the chosen port. Before the
    // fix this returned None and no listener was bound.
    use crate::network::affinity;

    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .expect("test node");
    let options = runtime_options_for_test(&["mesh-llm", "--auto"]);
    let candidates = vec![("tok".to_string(), None)];
    let router = affinity::AffinityRouter::default();

    // Pick an ephemeral port by binding+releasing first.
    let scratch = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = scratch.local_addr().unwrap().port();
    drop(scratch);

    let stop_tx = start_run_auto_bootstrap_proxy(&options, &node, port, &router, &candidates);
    assert!(
        stop_tx.is_some(),
        "serve --auto with auto_join_candidates must spawn bootstrap proxy"
    );

    // Give the spawned task a moment to bind, then confirm the port is
    // actually accepting connections (i.e. bootstrap_proxy ran far enough
    // to listen, not just that we got a stop_tx back).
    let mut connected = false;
    for _ in 0..20 {
        if tokio::net::TcpStream::connect(("127.0.0.1", port))
            .await
            .is_ok()
        {
            connected = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    assert!(connected, "bootstrap proxy should be listening on :{port}");

    // Hand the listener back so the proxy task can exit cleanly.
    let (give_tx, give_rx) = tokio::sync::oneshot::channel();
    let _ = stop_tx.unwrap().send(give_tx).await;
    let _ = tokio::time::timeout(std::time::Duration::from_secs(1), give_rx).await;
}

#[tokio::test]
async fn bootstrap_proxy_not_spawned_for_standalone_serve() {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .expect("test node");
    let options = runtime_options_for_test(&["mesh-llm"]);
    let router = affinity::AffinityRouter::default();

    let stop_tx = start_run_auto_bootstrap_proxy(&options, &node, 0, &router, &[]);
    assert!(
        stop_tx.is_none(),
        "standalone serve without join candidates must not spawn bootstrap proxy"
    );
}
