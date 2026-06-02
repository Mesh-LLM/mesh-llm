#![forbid(unsafe_code)]

pub mod envelope;
pub mod error;
#[cfg(feature = "host-io")]
pub mod keychain;
pub mod keys;
#[cfg(feature = "host-io")]
pub mod keystore;
#[cfg(feature = "host-io")]
pub mod node_key;
#[cfg(feature = "host-io")]
pub mod ownership;
pub mod provider;

pub use envelope::{OpenedMessage, SignedEncryptedEnvelope, open_message, seal_message};
pub use error::CryptoError;
#[cfg(feature = "host-io")]
pub use keychain::{
    DEFAULT_OWNER_ACCOUNT, KEYCHAIN_SERVICE, OwnerKeychainLoadError,
    delete_secret as keychain_delete, get_secret as keychain_get,
    is_available as keychain_available, load_owner_keypair_from_keychain,
    owner_account_for_path as owner_keychain_account_for_path, save_keystore_with_keychain,
    set_secret as keychain_set,
};
pub use keys::{OwnerKeypair, owner_id_from_verifying_key};
#[cfg(feature = "host-io")]
pub use keystore::{
    KeystoreInfo, default_keystore_path, keystore_exists, keystore_metadata, load_keystore,
    save_keystore,
};
#[cfg(feature = "host-io")]
pub use node_key::{
    NODE_KEY_BYTES, default_node_key_path, load_node_key_bytes_from_path,
    save_node_key_bytes_to_path,
};
#[cfg(feature = "host-io")]
pub use ownership::{
    DEFAULT_NODE_CERT_LIFETIME_SECS, DEFAULT_NODE_CERT_RENEW_WINDOW_SECS, NodeOwnershipClaim,
    OwnershipStatus, OwnershipSummary, SignedNodeOwnership, TrustPolicy, TrustStore,
    certificate_needs_renewal, default_node_ownership_path, default_trust_store_path,
    load_node_ownership, load_trust_store, save_node_ownership, save_trust_store,
    sign_node_ownership, verify_node_ownership,
};
pub use provider::{InMemoryKeyProvider, KeyProvider, KeyProviderError};
