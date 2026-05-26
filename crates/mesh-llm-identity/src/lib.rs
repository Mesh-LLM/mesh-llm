#![forbid(unsafe_code)]

pub mod envelope;
pub mod error;
pub mod keys;
pub mod provider;

pub use envelope::{OpenedMessage, SignedEncryptedEnvelope, open_message, seal_message};
pub use error::CryptoError;
pub use keys::{OwnerKeypair, owner_id_from_verifying_key};
pub use provider::{InMemoryKeyProvider, KeyProvider, KeyProviderError};
