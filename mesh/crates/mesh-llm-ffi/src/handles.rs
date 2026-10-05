use std::sync::Mutex;

#[cfg(feature = "embedded-runtime")]
use std::collections::HashMap;
#[cfg(feature = "embedded-runtime")]
use std::sync::Arc;
#[cfg(feature = "embedded-runtime")]
use std::sync::atomic::AtomicBool;
#[cfg(feature = "embedded-runtime")]
use tokio::sync::Notify;

use mesh_llm_sdk::MeshClient;
use mesh_llm_sdk::node::MeshNode;

#[cfg(feature = "embedded-runtime")]
use mesh_llm_sdk::embedded_runtime::EmbeddedServingController;

#[derive(uniffi::Object)]
pub struct MeshClientHandle {
    pub(crate) client: tokio::sync::RwLock<MeshClient>,
}

#[derive(uniffi::Object)]
pub struct MeshNodeHandle {
    pub(crate) node: MeshNode,
    #[cfg(feature = "embedded-runtime")]
    pub(crate) local_serving: Option<Arc<EmbeddedServingController>>,
    #[cfg(feature = "embedded-runtime")]
    pub(crate) local_openai_streams: LocalOpenAiStreamMap,
}

#[cfg(feature = "embedded-runtime")]
pub(crate) type LocalOpenAiStreamMap = Arc<Mutex<HashMap<String, LocalOpenAiStreamCancellation>>>;

#[cfg(feature = "embedded-runtime")]
pub(crate) struct LocalOpenAiStreamCancellation {
    pub(crate) cancelled: Arc<AtomicBool>,
    pub(crate) cancel_notify: Arc<Notify>,
}

#[derive(uniffi::Object)]
pub struct ConsoleHandle {
    pub(crate) inner: Mutex<Option<mesh_llm_sdk::console::ConsoleServerHandle>>,
    pub(crate) url: String,
}
