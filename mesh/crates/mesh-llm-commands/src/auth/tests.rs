use super::*;
use mesh_llm_identity::save_trust_store;
use serial_test::serial;

struct FailedAuthAudit;

impl mesh_llm_events::audit::AuditSink for FailedAuthAudit {
    fn emit_audit(&self, event: &mesh_llm_events::audit::AuditEvent) -> std::io::Result<()> {
        if matches!(
            event.action.as_str(),
            "auth_init"
                | "auth_sign_node"
                | "auth_rotate_node"
                | "auth_revoke_owner"
                | "auth_revoke_node_cert"
                | "auth_revoke_node_id"
                | "auth_rotate_owner"
        ) {
            return Err(std::io::Error::other("audit disk is full"));
        }
        Ok(())
    }

    fn flush(
        &self,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = std::io::Result<()>> + Send + '_>> {
        Box::pin(async { Ok(()) })
    }
}

struct RestoreAuditSink(Option<std::sync::Arc<dyn mesh_llm_events::audit::AuditSink>>);

impl Drop for RestoreAuditSink {
    fn drop(&mut self) {
        if let Some(sink) = self.0.take() {
            mesh_llm_events::audit::set_audit_sink(sink);
        } else {
            mesh_llm_events::audit::clear_audit_sink();
        }
    }
}

fn install_failed_auth_audit() -> RestoreAuditSink {
    let restore = RestoreAuditSink(mesh_llm_events::audit::audit_sink());
    mesh_llm_events::audit::set_audit_sink(std::sync::Arc::new(FailedAuthAudit));
    restore
}

struct FailFirstCertificateAudit {
    fail_certificate: std::sync::atomic::AtomicBool,
    attempts: std::sync::Mutex<Vec<String>>,
}

impl mesh_llm_events::audit::AuditSink for FailFirstCertificateAudit {
    fn emit_audit(&self, event: &mesh_llm_events::audit::AuditEvent) -> std::io::Result<()> {
        if !matches!(
            event.action.as_str(),
            "auth_revoke_node_cert" | "auth_revoke_node_id"
        ) {
            return Ok(());
        }
        self.attempts.lock().unwrap().push(event.action.clone());
        if event.action == "auth_revoke_node_cert"
            && self
                .fail_certificate
                .swap(false, std::sync::atomic::Ordering::SeqCst)
        {
            return Err(std::io::Error::other("certificate audit write failed"));
        }
        Ok(())
    }

    fn flush(
        &self,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = std::io::Result<()>> + Send + '_>> {
        Box::pin(async { Ok(()) })
    }
}

#[test]
#[serial]
fn mixed_revocation_attempts_both_audits_after_first_write_fails() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("trust-store.json");
    let node_id = hex::encode(SecretKey::generate().public().as_bytes());
    let sink = std::sync::Arc::new(FailFirstCertificateAudit {
        fail_certificate: std::sync::atomic::AtomicBool::new(true),
        attempts: std::sync::Mutex::new(Vec::new()),
    });
    let _restore = RestoreAuditSink(mesh_llm_events::audit::audit_sink());
    mesh_llm_events::audit::set_audit_sink(sink.clone());

    let error = run_revoke_node(
        Some("cert-a".into()),
        Some(node_id.clone()),
        None,
        Some(path.clone()),
    )
    .expect_err("the failed audit write must still reach the caller");
    assert!(error.to_string().contains("saved"), "{error:#}");
    assert!(format!("{error:#}").contains("certificate audit write failed"));
    let saved = load_effective_trust_store(&path).unwrap();
    assert_eq!(saved.revoked_node_certs[0].cert_id, "cert-a");
    assert_eq!(saved.revoked_node_ids[0].node_endpoint_id, node_id);
    assert_eq!(
        *sink.attempts.lock().unwrap(),
        ["auth_revoke_node_cert", "auth_revoke_node_id"]
    );
}

#[tokio::test]
#[serial]
async fn revocation_audit_write_failure_reports_saved_store() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("trust-store.json");
    let _restore = install_failed_auth_audit();

    let owner_error = run_revoke_owner("owner-a".into(), None, Some(path.clone()))
        .expect_err("a saved revocation must report its failed audit write");
    assert!(owner_error.to_string().contains("saved"), "{owner_error:#}");
    assert_eq!(
        load_effective_trust_store(&path)
            .unwrap()
            .revoked_owners
            .len(),
        1
    );

    let cert_error = run_revoke_node(Some("cert-a".into()), None, None, Some(path.clone()))
        .expect_err("a saved certificate revocation must report its failed audit write");
    assert!(cert_error.to_string().contains("saved"), "{cert_error:#}");
    assert_eq!(
        load_effective_trust_store(&path)
            .unwrap()
            .revoked_node_certs
            .len(),
        1
    );

    let node_id = hex::encode(SecretKey::generate().public().as_bytes());
    let node_error = run_revoke_node(None, Some(node_id), None, Some(path.clone()))
        .expect_err("a saved node revocation must report its failed audit write");
    assert!(node_error.to_string().contains("saved"), "{node_error:#}");
    assert_eq!(
        load_effective_trust_store(&path)
            .unwrap()
            .revoked_node_ids
            .len(),
        1
    );
    mesh_llm_events::audit::flush_audit().await.unwrap();
}

#[test]
#[serial]
fn keystore_audit_write_failure_reports_saved_changes() {
    let directory = tempfile::tempdir().unwrap();
    let owner_path = directory.path().join("owner.json");
    let node_path = directory.path().join("node.key");
    let certificate_path = directory.path().join("node.json");
    let trust_path = directory.path().join("trust.json");
    let _restore = install_failed_auth_audit();

    let error = run_init(Some(owner_path.clone()), false, true, false).unwrap_err();
    assert!(error.to_string().contains("saved"), "{error:#}");
    let original_owner = load_owner_keypair_from_path(&owner_path)
        .unwrap()
        .owner_id()
        .to_string();
    save_node_key_to_path(&node_path, &SecretKey::generate()).unwrap();

    let error = run_sign_node(
        Some(owner_path.clone()),
        Some(node_path.clone()),
        Some(certificate_path.clone()),
        None,
        None,
        1,
    )
    .unwrap_err();
    assert!(error.to_string().contains("saved"), "{error:#}");
    let original_certificate = load_node_ownership(&certificate_path).unwrap();

    let error = run_rotate_node(
        Some(owner_path.clone()),
        Some(node_path.clone()),
        Some(certificate_path.clone()),
        None,
        None,
        1,
        false,
        None,
        Some(trust_path),
    )
    .unwrap_err();
    assert!(error.to_string().contains("saved"), "{error:#}");
    let rotated_certificate = load_node_ownership(&certificate_path).unwrap();
    assert_ne!(
        rotated_certificate.claim.cert_id,
        original_certificate.claim.cert_id
    );

    let error = run_rotate_owner(Some(owner_path.clone()), true, true).unwrap_err();
    assert!(error.to_string().contains("saved"), "{error:#}");
    assert_ne!(
        load_owner_keypair_from_path(&owner_path)
            .unwrap()
            .owner_id()
            .to_string(),
        original_owner
    );
}

fn temporary_auth_test_dir(label: &str) -> std::path::PathBuf {
    let path =
        std::env::temp_dir().join(format!("mesh-llm-auth-{label}-{}", rand::random::<u64>()));
    std::fs::create_dir_all(&path).unwrap();
    path
}

#[test]
fn defaults_to_keychain_for_new_keystore_when_available() {
    assert!(should_default_to_keychain(false, false, true));
}

#[test]
fn does_not_default_to_keychain_for_existing_keystore() {
    assert!(!should_default_to_keychain(true, false, true));
}

#[test]
fn does_not_default_to_keychain_when_unavailable() {
    assert!(!should_default_to_keychain(false, false, false));
}

#[test]
fn does_not_default_to_keychain_with_no_passphrase() {
    assert!(!should_default_to_keychain(false, true, true));
}

#[test]
fn reports_stale_keychain_entry_as_encrypted_keystore() {
    let message = encrypted_keystore_keychain_status(OwnerKeychainLoadError::Crypto(
        mesh_llm_identity::CryptoError::DecryptionFailed,
    ));

    assert!(message.contains("keychain entry could not unlock this keystore"));
    assert!(message.contains("remove the stale keychain entry for this path"));
}

#[test]
#[serial]
fn force_keychain_save_failure_restores_previous_secret() {
    if !mesh_llm_identity::keychain_available() {
        eprintln!("keychain backend unavailable, skipping");
        return;
    }

    let tmp_dir =
        std::env::temp_dir().join(format!("mesh-llm-force-rollback-{}", rand::random::<u64>()));
    std::fs::create_dir_all(&tmp_dir).unwrap();
    let blocking_file = tmp_dir.join("blocker");
    std::fs::write(&blocking_file, b"not a directory").unwrap();
    let bad_path = blocking_file.join("owner-keystore.json");

    let account = mesh_llm_identity::owner_keychain_account_for_path(&bad_path);
    let previous_secret = "previous-unlock-secret-do-not-lose";
    mesh_llm_identity::keychain_set(KEYCHAIN_SERVICE, &account, previous_secret).unwrap();

    let result = run_init(Some(bad_path.clone()), true, false, true);
    assert!(
        result.is_err(),
        "run_init must fail when save cannot succeed"
    );

    let restored = mesh_llm_identity::keychain_get(KEYCHAIN_SERVICE, &account).unwrap();
    assert_eq!(
        restored.as_deref(),
        Some(previous_secret),
        "previous keychain secret must be restored after failed force-init"
    );

    mesh_llm_identity::keychain_delete(KEYCHAIN_SERVICE, &account).ok();
    std::fs::remove_dir_all(&tmp_dir).ok();
}

#[test]
#[serial]
fn fresh_keychain_save_failure_leaves_no_orphan() {
    if !mesh_llm_identity::keychain_available() {
        eprintln!("keychain backend unavailable, skipping");
        return;
    }

    let tmp_dir =
        std::env::temp_dir().join(format!("mesh-llm-fresh-rollback-{}", rand::random::<u64>()));
    std::fs::create_dir_all(&tmp_dir).unwrap();
    let blocking_file = tmp_dir.join("blocker");
    std::fs::write(&blocking_file, b"not a directory").unwrap();
    let bad_path = blocking_file.join("owner-keystore.json");

    let account = mesh_llm_identity::owner_keychain_account_for_path(&bad_path);
    mesh_llm_identity::keychain_delete(KEYCHAIN_SERVICE, &account).ok();

    let result = run_init(Some(bad_path.clone()), false, false, true);
    assert!(
        result.is_err(),
        "run_init must fail when save cannot succeed"
    );

    let residual = mesh_llm_identity::keychain_get(KEYCHAIN_SERVICE, &account).unwrap();
    assert_eq!(
        residual, None,
        "a fresh init failure must leave no keychain entry behind"
    );

    std::fs::remove_dir_all(&tmp_dir).ok();
}

#[test]
#[serial]
fn init_defaults_to_keychain_then_load_round_trip() {
    if !mesh_llm_identity::keychain_available() {
        eprintln!("keychain backend unavailable, skipping");
        return;
    }

    let dir = std::env::temp_dir().join(format!("mesh-llm-keychain-rt-{}", rand::random::<u64>()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("owner-keystore.json");

    run_init(Some(path.clone()), false, false, false)
        .expect("auth init should default to keychain when available");

    assert!(path.exists(), "keystore file should exist");
    let info = keystore_metadata(&path).unwrap();
    assert!(
        info.encrypted,
        "keystore should be encrypted when using keychain"
    );

    let account = mesh_llm_identity::owner_keychain_account_for_path(&path);
    let stored = mesh_llm_identity::keychain_get(KEYCHAIN_SERVICE, &account).unwrap();
    assert!(
        stored.is_some(),
        "keychain must have a passphrase entry for this keystore path"
    );

    let kp = load_owner_keypair_from_keychain(&path).expect("load via keychain must succeed");
    assert_eq!(kp.owner_id(), info.owner_id);

    mesh_llm_identity::keychain_delete(KEYCHAIN_SERVICE, &account).ok();
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn revoke_node_validates_all_inputs_before_persisting() {
    let dir = temporary_auth_test_dir("revoke-validation");
    let trust_store_path = dir.join("trust-store.json");
    save_trust_store(&trust_store_path, &TrustStore::default()).unwrap();
    let before = std::fs::read(&trust_store_path).unwrap();

    let result = run_revoke_node(
        Some("valid-cert".to_owned()),
        Some("not-a-node-id".to_owned()),
        Some("compromised".to_owned()),
        Some(trust_store_path.clone()),
    );

    assert!(result.is_err());
    assert_eq!(std::fs::read(&trust_store_path).unwrap(), before);
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn revoke_node_save_failure_leaves_blocking_file_unchanged() {
    let dir = temporary_auth_test_dir("revoke-save-failure");
    let blocking_file = dir.join("blocker");
    std::fs::write(&blocking_file, b"not a directory").unwrap();

    let result = run_revoke_node(
        Some("valid-cert".to_owned()),
        None,
        Some("compromised".to_owned()),
        Some(blocking_file.join("trust-store.json")),
    );

    assert!(result.is_err());
    assert_eq!(std::fs::read(&blocking_file).unwrap(), b"not a directory");
    std::fs::remove_dir_all(dir).ok();
}
