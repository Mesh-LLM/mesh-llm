//! The seam between a host and a payments engine. The host names only these
//! traits; an engine crate implements [`PaymentsEngineProvider`] and the
//! embedder installs it, so the host never links the settlement engine.

use std::any::Any;
use std::collections::BTreeMap;
use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;

use crate::pricing::Pricing;
use anyhow::Result;
use mesh_llm_wallet::provisioning::WalletFactory;

pub type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;

/// An opened engine. The host reads only what gossip and ingress need
/// synchronously; everything else goes through `payments.v1` operations.
pub trait PaymentsEngine: Send + Sync + 'static {
    /// Prices this node advertises, per model.
    fn pricing(&self) -> Result<BTreeMap<String, Pricing>>;
    /// Whether a wallet is open or provisioned for this node.
    fn has_wallet(&self) -> bool;
    /// For the provider to recover its concrete engine type.
    fn into_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync>;
}

/// Opens (lazily, at most once) this node's engine.
pub type EngineSource = Arc<dyn Fn() -> BoxFuture<Result<Arc<dyn PaymentsEngine>>> + Send + Sync>;

/// Advertised seller prices, read without opening an engine the node does not
/// have. `None` means this node has no payments state to advertise. The host
/// supplies this alongside `EngineSource` so a builtin provider can answer
/// `payments.v1` pricing without creating a ledger to read an empty table.
pub type AdvertisedPrices =
    Arc<dyn Fn() -> BoxFuture<Result<Option<BTreeMap<String, Pricing>>>> + Send + Sync>;

/// Supplied by the embedder that links a payments engine.
pub trait PaymentsEngineProvider: Send + Sync + 'static {
    /// Open the engine over `directory`, using `wallet` for wallet access.
    fn open(
        &self,
        directory: &Path,
        wallet: Arc<dyn WalletFactory>,
    ) -> Result<Arc<dyn PaymentsEngine>>;
    /// Serve `payments.v1` over an in-process plugin stream. `source` opens
    /// the engine on first use, so a node that never pays never creates one.
    fn serve(
        &self,
        plugin_name: &str,
        version: &str,
        source: EngineSource,
        prices: AdvertisedPrices,
        stream: mesh_llm_plugin::LocalStream,
    ) -> BoxFuture<Result<()>>;
}
