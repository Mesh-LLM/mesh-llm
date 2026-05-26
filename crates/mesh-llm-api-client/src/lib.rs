#![forbid(unsafe_code)]

mod client;
mod discover;
pub mod events;
mod identity;
mod token;

pub use client::{
    ChatMessage, ChatRequest, ClientBuilder, ClientConfig, MAX_RECONNECT_ATTEMPTS, MeshApiError,
    MeshClient, Model, RequestId, ResponsesRequest, Status,
};
pub use discover::{
    AutoConnectResult, PublicMesh, PublicMeshQuery, create_auto_client, discover_public_meshes,
    select_public_mesh,
};
pub use identity::OwnerKeypair;
pub use token::InviteToken;
