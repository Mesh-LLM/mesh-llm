mod keychain;
mod keystore;
mod ownership;
pub(crate) mod release_attestation;

pub use self::keychain::{
    DEFAULT_OWNER_ACCOUNT, KEYCHAIN_SERVICE, OwnerKeychainLoadError,
    delete_secret as keychain_delete, get_secret as keychain_get,
    is_available as keychain_available, load_owner_keypair_from_keychain,
    owner_account_for_path as owner_keychain_account_for_path, save_keystore_with_keychain,
    set_secret as keychain_set,
};
pub(crate) use self::keystore::write_keystore_bytes_atomically;
pub use self::keystore::{
    KeystoreInfo, default_keystore_path, keystore_exists, keystore_metadata, load_keystore,
    save_keystore,
};
pub use self::ownership::{
    ControlPlaneAuthError, DEFAULT_NODE_CERT_LIFETIME_SECS, DEFAULT_NODE_CERT_RENEW_WINDOW_SECS,
    NodeOwnershipClaim, OwnershipStatus, OwnershipSummary, SignedNodeOwnership, TrustPolicy,
    TrustStore, certificate_needs_renewal, default_node_ownership_path, default_trust_store_path,
    load_node_ownership, load_trust_store, save_node_ownership, save_trust_store,
    sign_node_ownership, verify_control_plane_peer_ownership, verify_control_plane_target_node,
    verify_node_ownership,
};
pub use self::release_attestation::{
    EmbeddedReleaseAttestation, LoadedEmbeddedReleaseAttestation, ReleaseAttestationClaims,
    ReleaseAttestationError, ReleaseAttestationStatus, ReleaseAttestationSummary,
    ReleaseBuildAttestation, ReleaseSignerTrustStore, TrustedReleaseSigner,
    default_release_signer_trust_store_path, load_embedded_release_attestation_for_binary,
    load_release_signer_trust_store, parse_release_signer_public_key, release_signer_key_id,
    save_release_signer_trust_store, verify_release_attestation,
};
pub use mesh_llm_identity::{
    CryptoError, OpenedMessage, OwnerKeypair, SignedEncryptedEnvelope, open_message,
    owner_id_from_verifying_key, seal_message,
};
