use super::*;
use crate::api::status::decode_runtime_model_path;
use crate::crypto::{OwnerKeypair, default_keystore_path, save_keystore};
use crate::plugin;
use crate::plugins::blobstore;
use base64::Engine;
use mesh_client::proto::node::{
    ConfigApplyMode, NodeConfigSnapshot, OwnerControlApplyConfigRequest,
    OwnerControlApplyConfigResponse, OwnerControlConfigSnapshot, OwnerControlEnvelope,
    OwnerControlError, OwnerControlErrorCode, OwnerControlGetConfigResponse, OwnerControlResponse,
};
use mesh_llm_plugin::MeshVisibility;
use mesh_llm_protocol::{ALPN_CONTROL_V1, decode_owner_control_envelope, write_len_prefixed};
use prost::Message;
use rmcp::model::ErrorCode;
use serde_json::json;
use serial_test::serial;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, oneshot};

mod apply_config_diagnostics;
mod apply_config_validation_authority;
mod runtime_config;
mod runtime_config_validation_authority;
mod runtime_control_state;
mod runtime_control_state_builder;
mod runtime_control_state_options;

include!("support.rs");
include!("gpu_status.rs");
include!("runtime_status.rs");
include!("node_state.rs");
include!("control_plane.rs");
include!("management_http.rs");
include!("runtime_data.rs");
include!("catalog_routes.rs");
include!("model_targets.rs");
include!("wakeable_inventory.rs");
include!("status_metrics.rs");
include!("openai_smoke.rs");
include!("ui_routes.rs");
