//! What a wallet plugin author implements.
//!
//! A [`WalletBackend`] is a factory over a concrete wallet: it inspects and
//! opens persisted state under a host-supplied directory and hands back a
//! [`WalletProvider`] plus its [`WalletIdentity`]. Everything else — the
//! operation router, error mapping, open-state tracking — is provided by
//! [`crate::plugin_server`].

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;

use crate::contract::WalletIdentity;
use crate::provider::WalletProvider;

/// An opened wallet and the identity the host will pin.
pub struct OpenedWallet {
    pub identity: WalletIdentity,
    pub provider: Arc<dyn WalletProvider>,
    /// True if `open` created a new wallet rather than loading one.
    pub created: bool,
}

#[async_trait]
pub trait WalletBackend: Send + Sync + 'static {
    /// Short provider label, e.g. `"lexe"`. Reported in [`WalletIdentity`].
    fn provider_name(&self) -> &'static str;

    /// Inspect local state without network calls or side effects.
    fn is_provisioned(&self, directory: &Path) -> bool;

    /// Open or provision the wallet under `directory`. May contact the
    /// provider network. Must be safe to call again after a crash between
    /// seed persistence and provisioning.
    async fn open(&self, directory: &Path) -> Result<OpenedWallet>;
}
