use mesh_llm_sdk::OwnerKeypair;

use crate::errors::FfiError;

/// Generate a fresh owner keypair, returning its hex-encoded form.
///
/// Callers should persist this value on first run and pass it back to
/// `create_node` on subsequent launches so the embedded node keeps a stable
/// identity. Generating a new keypair on every launch will make the app look
/// like a different owner to the mesh each time.
#[uniffi::export]
pub fn generate_owner_keypair_hex() -> String {
    OwnerKeypair::generate().to_hex()
}

pub(super) fn parse_owner_keypair(owner_keypair_bytes_hex: &str) -> Result<OwnerKeypair, FfiError> {
    // An empty keypair is rejected rather than silently generating a fresh identity:
    // a caller that forgets to pass their persisted owner keypair would otherwise
    // get a brand-new identity every launch with no error. Callers that genuinely
    // want a new keypair should create one explicitly before calling create_node.
    let trimmed = owner_keypair_bytes_hex.trim();
    if trimmed.is_empty() {
        return Err(FfiError::InvalidOwnerKeypair(
            "owner keypair must not be empty".to_string(),
        ));
    }
    OwnerKeypair::from_hex(trimmed)
        .map_err(|error| FfiError::InvalidOwnerKeypair(error.to_string()))
}
