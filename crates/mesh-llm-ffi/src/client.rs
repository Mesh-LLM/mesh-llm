use std::sync::Arc;

use mesh_llm_sdk::{
    ClientBuilder, InviteToken, RequestId, create_auto_client as sdk_create_auto_client,
};

use crate::errors::{FfiError, map_mesh_api_error};
use crate::events::EventListenerBridge;
use crate::identity::parse_owner_keypair;
use crate::request_types::{
    ChatRequestNative, ClientStatus, ModelNative, PublicMeshQuery, ResponsesRequestNative,
};
use crate::runtime_blocking::block_on;
use crate::{EventListener, MeshClientHandle};

#[uniffi::export]
pub fn create_auto_client(
    owner_keypair_bytes_hex: String,
    query: PublicMeshQuery,
) -> Result<Arc<MeshClientHandle>, FfiError> {
    let kp = parse_owner_keypair(&owner_keypair_bytes_hex)?;
    block_on(sdk_create_auto_client(kp, query.into()))
        .map(|result| {
            Arc::new(MeshClientHandle {
                client: tokio::sync::Mutex::new(result.client),
            })
        })
        .map_err(map_mesh_api_error)
}

#[uniffi::export]
pub fn create_client(
    owner_keypair_bytes_hex: String,
    invite_token: String,
) -> Result<Arc<MeshClientHandle>, FfiError> {
    let token = invite_token
        .parse::<InviteToken>()
        .map_err(FfiError::InvalidInviteToken)?;
    let kp = parse_owner_keypair(&owner_keypair_bytes_hex)?;
    let client = ClientBuilder::new(kp, token)
        .build()
        .map_err(|error| FfiError::BuildFailed(error.to_string()))?;
    Ok(Arc::new(MeshClientHandle {
        client: tokio::sync::Mutex::new(client),
    }))
}

#[uniffi::export]
impl MeshClientHandle {
    pub fn start(&self) -> Result<(), FfiError> {
        block_on(async {
            let mut client = self.client.lock().await;
            client.join().await
        })
        .map_err(|error| FfiError::JoinFailed(error.to_string()))
    }

    pub fn stop(&self) {
        block_on(async {
            self.client.lock().await.disconnect().await;
        });
    }

    pub fn reconnect(&self) -> Result<(), FfiError> {
        block_on(async {
            let mut client = self.client.lock().await;
            client.reconnect().await
        })
        .map_err(|error| FfiError::ReconnectFailed(error.to_string()))
    }

    pub fn status(&self) -> ClientStatus {
        let status = block_on(async { self.client.lock().await.status().await });
        ClientStatus {
            connected: status.connected,
            peer_count: status.peer_count as u64,
        }
    }

    pub fn inference_list_models(&self) -> Result<Vec<ModelNative>, FfiError> {
        block_on(async { self.client.lock().await.list_models().await })
            .map(|models| {
                models
                    .into_iter()
                    .map(|m| ModelNative {
                        id: m.id,
                        name: m.name,
                    })
                    .collect()
            })
            .map_err(|error| FfiError::DiscoveryFailed(error.to_string()))
    }

    pub fn chat(
        &self,
        request: ChatRequestNative,
        listener: Box<dyn EventListener>,
    ) -> Result<String, FfiError> {
        let bridge = Arc::new(EventListenerBridge { inner: listener });
        let request_id = block_on(async { self.client.lock().await.chat(request.into(), bridge) });
        Ok(request_id.0)
    }

    pub fn responses(
        &self,
        request: ResponsesRequestNative,
        listener: Box<dyn EventListener>,
    ) -> Result<String, FfiError> {
        let bridge = Arc::new(EventListenerBridge { inner: listener });
        let request_id =
            block_on(async { self.client.lock().await.responses(request.into(), bridge) });
        Ok(request_id.0)
    }

    pub fn cancel(&self, request_id: String) {
        block_on(async {
            self.client.lock().await.cancel(RequestId(request_id));
        });
    }
}
