//! Exercise real certificate files without changing the parent test process's HOME.

use super::*;
use crate::crypto::{
    OwnerKeypair, TrustPolicy, TrustStore, load_node_ownership, verify_node_ownership,
};
use crate::mesh::{
    MembershipNodeConfig, NodeRole, NodeStartupSpec, OwnerRuntimeConfig, QuicBindSelection,
    RelayConfig, RelayPolicy, ScopedNodeIdentity,
};

const DAY_MS: u64 = 24 * 60 * 60 * 1000;

/// Use the existing isolated child-test runner, never a process-wide env mutation.
fn isolated_home(test: &str) -> bool {
    let directory = tempfile::tempdir().unwrap();
    if !mesh_llm_config::env_override_in_child_for_test(
        test,
        "HOME",
        directory.path().to_str().unwrap(),
    ) {
        return false;
    }
    let home = std::env::var_os("HOME").unwrap();
    assert_eq!(
        default_node_ownership_path().unwrap(),
        Path::new(&home).join(".mesh-llm/node-ownership.json"),
        "never write to the real user's default certificate"
    );
    true
}

fn owner_config(name: &str, seed: u8) -> OwnerRuntimeConfig {
    OwnerRuntimeConfig {
        keypair: Some(OwnerKeypair::from_bytes(&[seed; 32], &[seed + 1; 32]).unwrap()),
        control_bind: None,
        control_advertise_addr: None,
        node_label: Some(name.into()),
        trust_store: TrustStore::default(),
        trust_policy: TrustPolicy::Off,
        activity: Default::default(),
        drain_timeout_secs: 30,
        drain_timeout_max_secs: 30,
        public_mesh: false,
    }
}

/// Use the production scoped constructor, including its initial certificate write.
async fn scoped_node(directory: &Path, name: &str, seed: u8) -> Node {
    let (node, _) = Node::start_with_spec(NodeStartupSpec {
        role: NodeRole::Host { http_port: 1 },
        relay: RelayConfig {
            urls: &[],
            auths: &Default::default(),
            policy: RelayPolicy::Disabled,
        },
        quic_bind: QuicBindSelection {
            ip: Some(std::net::Ipv4Addr::LOCALHOST.into()),
            port: None,
        },
        max_vram_gb: Some(0.0),
        enumerate_host: false,
        owner_config: Some(owner_config(name, seed)),
        config_path: Some(&directory.join("config.toml")),
        local_mesh_requirements: Default::default(),
        identity: Some(ScopedNodeIdentity {
            key: iroh::SecretKey::generate(),
            state_dir: directory.to_owned(),
        }),
    })
    .await
    .unwrap();
    node
}

/// The ordinary identity uses its existing startup signer and global path.
async fn ordinary_node() -> Node {
    let mut node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
    let config = owner_config("ordinary", 0x51);
    let runtime = super::super::startup::init_owner_runtime(
        Some(&config),
        node.id(),
        Some("ordinary-host".into()),
        None,
    )
    .unwrap();
    node.owner_keypair = config.keypair;
    node.owner_attestation_path = super::super::startup::saved_owner_attestation_path(None);
    *node.owner_attestation.lock().await = runtime.owner_attestation;
    node
}

async fn renew_and_check(node: &Node, path: &Path) -> Vec<u8> {
    let initial = load_node_ownership(path).unwrap();
    let now = initial.claim.expires_at_unix_ms - DAY_MS;
    node.renew_owner_attestation_on_heartbeat(now).await;
    let renewed = load_node_ownership(path).unwrap();
    assert_ne!(renewed.claim.cert_id, initial.claim.cert_id);
    assert_eq!(
        renewed.claim.node_endpoint_id,
        hex::encode(node.id().as_bytes())
    );
    assert_eq!(
        renewed.claim.owner_id,
        node.owner_keypair.as_ref().unwrap().owner_id()
    );
    assert_eq!(renewed.claim.node_label, initial.claim.node_label);
    assert_eq!(renewed.claim.hostname_hint, initial.claim.hostname_hint);
    assert_eq!(node.owner_attestation.lock().await.as_ref(), Some(&renewed));
    assert_eq!(
        node.owner_summary().await.cert_id.as_deref(),
        Some(renewed.claim.cert_id.as_str())
    );
    assert!(
        verify_node_ownership(
            Some(&renewed),
            node.id().as_bytes(),
            &TrustStore::default(),
            TrustPolicy::Off,
            now,
        )
        .verified
    );
    std::fs::read(path).unwrap()
}

#[tokio::test]
async fn scoped_renewal_preserves_other_memberships_and_the_ordinary_certificate() {
    if !isolated_home(concat!(
        module_path!(),
        "::scoped_renewal_preserves_other_memberships_and_the_ordinary_certificate"
    )) {
        return;
    }
    let directory = tempfile::tempdir().unwrap();
    let ordinary = ordinary_node().await;
    let default_path = default_node_ownership_path().unwrap();
    let default_before = std::fs::read(&default_path).unwrap();
    let alpha_dir = directory.path().join("alpha");
    let beta_dir = directory.path().join("beta");
    let alpha = scoped_node(&alpha_dir, "alpha", 0x61).await;
    let beta = scoped_node(&beta_dir, "beta", 0x71).await;
    let alpha_path = alpha_dir.join("node-ownership.json");
    let beta_path = beta_dir.join("node-ownership.json");
    assert_ne!(alpha.id(), beta.id());
    let beta_before = std::fs::read(&beta_path).unwrap();
    let alpha_before = std::fs::read(&alpha_path).unwrap();

    // An ordinary heartbeat outside the renewal window must not rewrite the file.
    alpha.run_heartbeat_cycle(&mut Default::default()).await;
    assert_eq!(std::fs::read(&alpha_path).unwrap(), alpha_before);

    let alpha_after = renew_and_check(&alpha, &alpha_path).await;
    assert_eq!(std::fs::read(&beta_path).unwrap(), beta_before);
    assert_eq!(std::fs::read(&default_path).unwrap(), default_before);
    let beta_after = renew_and_check(&beta, &beta_path).await;
    assert_eq!(std::fs::read(&alpha_path).unwrap(), alpha_after);
    assert_eq!(std::fs::read(&default_path).unwrap(), default_before);

    renew_and_check(&ordinary, &default_path).await;
    assert_eq!(std::fs::read(&alpha_path).unwrap(), alpha_after);
    assert_eq!(std::fs::read(&beta_path).unwrap(), beta_after);
    alpha.close_endpoint().await;
    beta.close_endpoint().await;
    ordinary.close_endpoint().await;
}

#[tokio::test]
async fn sdk_membership_renewal_preserves_scoped_and_default_certificates() {
    if !isolated_home(concat!(
        module_path!(),
        "::sdk_membership_renewal_preserves_scoped_and_default_certificates"
    )) {
        return;
    }
    let directory = tempfile::tempdir().unwrap();
    let ordinary = ordinary_node().await;
    let default_path = default_node_ownership_path().unwrap();
    let default_before = std::fs::read(&default_path).unwrap();
    let scoped = scoped_node(directory.path(), "scoped", 0x61).await;
    let scoped_path = directory.path().join("node-ownership.json");
    let scoped_before = std::fs::read(&scoped_path).unwrap();
    let owners = [scoped.owner_keypair.as_ref().unwrap().owner_id()];
    let (membership, _) = scoped
        .start_membership_node(MembershipNodeConfig {
            mesh_name: "sdk-membership",
            secret_key: iroh::SecretKey::generate(),
            relay: RelayConfig {
                urls: &[],
                auths: &Default::default(),
                policy: RelayPolicy::ExplicitlyDisabled,
            },
            quic_port: None,
            trusted_owners: &owners,
            enumerate_host: false,
        })
        .await
        .unwrap();
    assert!(membership.owner_attestation_path.is_none());
    let initial = membership.owner_attestation.lock().await.clone().unwrap();
    let now = initial.claim.expires_at_unix_ms - DAY_MS;
    membership.renew_owner_attestation_on_heartbeat(now).await;
    assert!(membership.unsaved_owner_attestation.lock().await.is_none());
    let renewed = membership.owner_attestation.lock().await.clone().unwrap();
    assert_ne!(renewed.claim.cert_id, initial.claim.cert_id);
    assert!(
        verify_node_ownership(
            Some(&renewed),
            membership.id().as_bytes(),
            &TrustStore::default(),
            TrustPolicy::Off,
            now,
        )
        .verified
    );
    assert_eq!(std::fs::read(&scoped_path).unwrap(), scoped_before);
    assert_eq!(std::fs::read(&default_path).unwrap(), default_before);
    renew_and_check(&scoped, &scoped_path).await;
    assert_eq!(
        membership.owner_attestation.lock().await.as_ref(),
        Some(&renewed)
    );
    assert_eq!(std::fs::read(&default_path).unwrap(), default_before);
    membership.close_endpoint().await;
    scoped.close_endpoint().await;
    ordinary.close_endpoint().await;
}

#[tokio::test]
async fn failed_scoped_persistence_keeps_renewed_identity_without_global_fallback() {
    if !isolated_home(concat!(
        module_path!(),
        "::failed_scoped_persistence_keeps_renewed_identity_without_global_fallback"
    )) {
        return;
    }
    let directory = tempfile::tempdir().unwrap();
    let ordinary = ordinary_node().await;
    let default_path = default_node_ownership_path().unwrap();
    let default_before = std::fs::read(&default_path).unwrap();
    let scoped_dir = directory.path().join("alpha");
    let scoped = scoped_node(&scoped_dir, "alpha", 0x61).await;
    let beta_dir = directory.path().join("beta");
    let beta = scoped_node(&beta_dir, "beta", 0x71).await;
    let beta_path = beta_dir.join("node-ownership.json");
    let beta_before = std::fs::read(&beta_path).unwrap();
    let path = scoped_dir.join("node-ownership.json");
    let initial = load_node_ownership(&path).unwrap();
    // A directory at the certificate destination deterministically rejects atomic rename.
    std::fs::remove_file(&path).unwrap();
    std::fs::create_dir(&path).unwrap();
    let now = initial.claim.expires_at_unix_ms - DAY_MS;
    scoped.renew_owner_attestation_on_heartbeat(now).await;
    let renewed = scoped.owner_attestation.lock().await.clone().unwrap();
    assert_ne!(renewed.claim.cert_id, initial.claim.cert_id);
    assert_eq!(
        scoped.owner_summary().await.cert_id.as_deref(),
        Some(renewed.claim.cert_id.as_str())
    );
    assert!(
        verify_node_ownership(
            Some(&renewed),
            scoped.id().as_bytes(),
            &TrustStore::default(),
            TrustPolicy::Off,
            now,
        )
        .verified
    );
    assert!(path.is_dir());
    assert_eq!(
        scoped.unsaved_owner_attestation.lock().await.as_ref(),
        Some(&renewed)
    );
    assert_eq!(std::fs::read(&beta_path).unwrap(), beta_before);
    assert_eq!(std::fs::read(&default_path).unwrap(), default_before);

    // A later ordinary heartbeat retries only alpha's file, without re-signing.
    std::fs::remove_dir(&path).unwrap();
    scoped.run_heartbeat_cycle(&mut Default::default()).await;
    assert_eq!(load_node_ownership(&path).unwrap(), renewed);
    assert!(scoped.unsaved_owner_attestation.lock().await.is_none());
    assert_eq!(std::fs::read(&beta_path).unwrap(), beta_before);
    assert_eq!(std::fs::read(&default_path).unwrap(), default_before);

    // Successful persistence clears the pending write rather than rewriting forever.
    std::fs::remove_file(&path).unwrap();
    scoped.run_heartbeat_cycle(&mut Default::default()).await;
    assert!(!path.exists());
    scoped.close_endpoint().await;
    beta.close_endpoint().await;
    ordinary.close_endpoint().await;
}
