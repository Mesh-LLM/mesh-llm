use super::*;
use serial_test::serial;

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
