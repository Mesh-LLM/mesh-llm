#![forbid(unsafe_code)]

mod client;
mod discover;
pub mod events;
mod identity;
mod token;

pub use client::{
    ChatMessage, ChatRequest, ClientBuilder, ClientConfig, MeshApiError, MeshClient, Model,
    RequestId, ResponsesRequest, Status, MAX_RECONNECT_ATTEMPTS,
};
pub use discover::{
    create_auto_client, discover_public_meshes, select_public_mesh, AutoConnectResult, PublicMesh,
    PublicMeshQuery,
};
pub use identity::OwnerKeypair;
pub use token::InviteToken;
