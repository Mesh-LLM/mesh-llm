use std::sync::Mutex;

#[cfg(feature = "embedded-runtime")]
use std::sync::Arc;

use mesh_llm_sdk::MeshClient;
use mesh_llm_sdk::node::MeshNode;

#[cfg(feature = "embedded-runtime")]
use mesh_llm_sdk::embedded_runtime::EmbeddedServingController;
#[cfg(feature = "embedded-runtime")]
use mesh_llm_sdk::provider_host::ProviderHost;

#[derive(uniffi::Object)]
pub struct MeshClientHandle {
    pub(crate) client: tokio::sync::Mutex<MeshClient>,
}

#[derive(uniffi::Object)]
pub struct MeshNodeHandle {
    pub(crate) node: MeshNode,
    #[cfg(feature = "embedded-runtime")]
    pub(crate) local_serving: Option<Arc<EmbeddedServingController>>,
}

#[derive(uniffi::Object)]
pub struct ConsoleHandle {
    pub(crate) inner: Mutex<Option<mesh_llm_sdk::console::ConsoleServerHandle>>,
    pub(crate) url: String,
}

#[derive(uniffi::Object)]
pub struct ProviderHostHandle {
    #[cfg(feature = "embedded-runtime")]
    pub(crate) host: tokio::sync::Mutex<Option<ProviderHost>>,
    pub(crate) api_base_url: String,
}
