//! Native discovery of mesh-advertised models, independent of OpenAI inference.

use napi::bindgen_prelude::*;
use napi_derive::napi;

use super::{Node, to_napi_error};

#[napi]
impl Node {
    #[napi(js_name = "meshModelsJson")]
    pub async fn mesh_models_json(&self, query_json: String) -> Result<String> {
        let query = serde_json::from_str(&query_json).map_err(to_napi_error)?;
        let node = self.node.lock().await;
        let node = node
            .as_ref()
            .ok_or_else(|| Error::from_reason("node is not running"))?;
        let page = node.models().list(query).await.map_err(to_napi_error)?;
        serde_json::to_string(&page).map_err(to_napi_error)
    }

    #[napi(js_name = "meshModelJson")]
    pub async fn mesh_model_json(&self, id: String) -> Result<String> {
        let node = self.node.lock().await;
        let node = node
            .as_ref()
            .ok_or_else(|| Error::from_reason("node is not running"))?;
        let model = node.models().get(&id).await.map_err(to_napi_error)?;
        serde_json::to_string(&model).map_err(to_napi_error)
    }
}
