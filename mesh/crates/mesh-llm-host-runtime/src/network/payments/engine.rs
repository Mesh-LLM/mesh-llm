//! Process registry for the payments engine. The traits live in the types
//! crate; the shipped binary installs the real engine with
//! [`install_payments_engine`], so this crate never links it.

use std::sync::{Arc, OnceLock};

pub use mesh_llm_payments_types::engine::{EngineSource, PaymentsEngine, PaymentsEngineProvider};

static PROVIDER: OnceLock<Arc<dyn PaymentsEngineProvider>> = OnceLock::new();

/// Install the payments engine for this process. Returns `false` if one was
/// already installed (the first wins). Without one, the node runs free-only:
/// no `payments.v1` provider, so paid peers are excluded.
pub fn install_payments_engine(provider: Arc<dyn PaymentsEngineProvider>) -> bool {
    PROVIDER.set(provider).is_ok()
}

pub(crate) fn provider() -> Option<&'static Arc<dyn PaymentsEngineProvider>> {
    PROVIDER.get()
}
