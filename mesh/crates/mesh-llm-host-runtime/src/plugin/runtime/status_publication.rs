//! A plugin connection's `/api/status` publications, on their way to the node.
//!
//! Each connection hands its publications to one lane, which takes them in
//! order. A publication waits for the connection to finish initializing, since
//! a plugin may publish from `on_initialized` before the host has read its
//! manifest, and it is handed to the node under the runtime lock while that
//! connection is still the plugin's current one. Every notice that a
//! connection is gone is sent after its runtime was taken under the same lock,
//! so the node never gets a snapshot from a connection after that notice.

use super::PluginRuntime;
use crate::plugin::{PROTOCOL_VERSION, PluginMeshEvent, proto};
use rmcp::model::ErrorCode;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, mpsc, oneshot};

/// What a connection's manifest says about `/api/status`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum StatusDeclaration {
    /// The connection has not finished initializing.
    #[default]
    Pending,
    Undeclared,
    Declared {
        schema_version: u32,
    },
}

impl StatusDeclaration {
    pub(crate) fn from_manifest(manifest: Option<&proto::PluginManifest>) -> Self {
        match manifest.and_then(|manifest| manifest.plugin_status.as_ref()) {
            Some(status) => Self::Declared {
                schema_version: status.schema_version,
            },
            None => Self::Undeclared,
        }
    }
}

/// Refuse a declaration whose schema version is 0: versions start at 1.
pub(crate) fn validate_declaration(
    plugin_name: &str,
    manifest: &proto::PluginManifest,
) -> anyhow::Result<()> {
    if manifest
        .plugin_status
        .as_ref()
        .is_some_and(|status| status.schema_version == 0)
    {
        anyhow::bail!(
            "Plugin '{plugin_name}' declares a status contribution with schema_version 0; it starts at 1"
        );
    }
    Ok(())
}

/// Publications a lane holds before it answers new ones as busy. The host
/// accepts one a second, so a well-behaved plugin never has more than one.
const LANE_CAPACITY: usize = 4;
/// How long a publication waits for its connection to finish initializing,
/// and then for the node to answer.
const PUBLICATION_TIMEOUT: Duration = Duration::from_secs(crate::plugin::REQUEST_TIMEOUT_SECS);
const INITIALIZATION_POLL: Duration = Duration::from_millis(10);

type Publication = (u64, proto::PublishPluginStatusRequest);

pub(crate) struct StatusLane {
    plugin_name: String,
    tx: mpsc::Sender<Publication>,
    outbound_tx: mpsc::Sender<proto::Envelope>,
}

impl StatusLane {
    /// Start the lane for connection `generation` of `plugin_name`. It ends
    /// when the lane is dropped and its queue is drained.
    pub(crate) fn spawn(
        plugin_name: String,
        generation: u64,
        runtime: Arc<Mutex<Option<PluginRuntime>>>,
        mesh_tx: mpsc::Sender<PluginMeshEvent>,
        outbound_tx: mpsc::Sender<proto::Envelope>,
    ) -> Self {
        let (tx, mut rx) = mpsc::channel::<Publication>(LANE_CAPACITY);
        let lane = Self {
            plugin_name: plugin_name.clone(),
            tx,
            outbound_tx: outbound_tx.clone(),
        };
        tokio::spawn(async move {
            while let Some((request_id, request)) = rx.recv().await {
                let payload = match publish(&plugin_name, generation, &runtime, &mesh_tx, request)
                    .await
                {
                    Ok(response) => proto::envelope::Payload::PublishPluginStatusResponse(response),
                    Err(error) => proto::envelope::Payload::ErrorResponse(error),
                };
                let _ = outbound_tx
                    .send(envelope(&plugin_name, request_id, payload))
                    .await;
            }
        });
        lane
    }

    /// Queue a publication without waiting; a full lane answers it at once.
    pub(crate) fn push(&self, request_id: u64, request: proto::PublishPluginStatusRequest) {
        if let Err(mpsc::error::TrySendError::Full(_)) = self.tx.try_send((request_id, request)) {
            let outbound_tx = self.outbound_tx.clone();
            let reply = envelope(
                &self.plugin_name,
                request_id,
                proto::envelope::Payload::ErrorResponse(error(
                    ErrorCode::INVALID_REQUEST,
                    format!(
                        "plugin `{}` has {LANE_CAPACITY} status publications waiting; retry later",
                        self.plugin_name
                    ),
                )),
            );
            tokio::spawn(async move {
                let _ = outbound_tx.send(reply).await;
            });
        }
    }
}

async fn publish(
    plugin_name: &str,
    generation: u64,
    runtime: &Mutex<Option<PluginRuntime>>,
    mesh_tx: &mpsc::Sender<PluginMeshEvent>,
    request: proto::PublishPluginStatusRequest,
) -> Result<proto::PublishPluginStatusResponse, proto::ErrorResponse> {
    let deadline = tokio::time::Instant::now() + PUBLICATION_TIMEOUT;
    let response_rx =
        hand_over(plugin_name, generation, runtime, mesh_tx, request, deadline).await?;
    match tokio::time::timeout_at(deadline, response_rx).await {
        Ok(Ok(response)) => response,
        Ok(Err(_)) => Err(error(
            ErrorCode::INTERNAL_ERROR,
            "status publication handler dropped the response",
        )),
        Err(_) => Err(error(
            ErrorCode::INTERNAL_ERROR,
            "status publication timed out",
        )),
    }
}

type PublicationResponse =
    oneshot::Receiver<Result<proto::PublishPluginStatusResponse, proto::ErrorResponse>>;

async fn hand_over(
    plugin_name: &str,
    generation: u64,
    runtime: &Mutex<Option<PluginRuntime>>,
    mesh_tx: &mpsc::Sender<PluginMeshEvent>,
    request: proto::PublishPluginStatusRequest,
    deadline: tokio::time::Instant,
) -> Result<PublicationResponse, proto::ErrorResponse> {
    loop {
        {
            let runtime = runtime.lock().await;
            let Some(current) = runtime
                .as_ref()
                .filter(|runtime| runtime.generation == generation)
            else {
                return Err(error(
                    ErrorCode::INVALID_REQUEST,
                    format!(
                        "plugin `{plugin_name}` published its status from a connection that is no longer current"
                    ),
                ));
            };
            match current.status_declaration {
                StatusDeclaration::Pending => {}
                StatusDeclaration::Undeclared => {
                    return Err(error(
                        ErrorCode::INVALID_REQUEST,
                        format!(
                            "plugin `{plugin_name}` does not declare `plugin_status` in its manifest"
                        ),
                    ));
                }
                StatusDeclaration::Declared { schema_version } => {
                    let (response_tx, response_rx) = oneshot::channel();
                    // Still under the runtime lock: see the module docs.
                    return match mesh_tx.try_send(PluginMeshEvent::PluginStatus {
                        plugin_id: plugin_name.to_string(),
                        generation,
                        schema_version,
                        request,
                        response_tx,
                    }) {
                        Ok(()) => Ok(response_rx),
                        Err(mpsc::error::TrySendError::Full(_)) => Err(error(
                            ErrorCode::INTERNAL_ERROR,
                            "the node is busy; retry the status publication later",
                        )),
                        Err(mpsc::error::TrySendError::Closed(_)) => Err(error(
                            ErrorCode::INTERNAL_ERROR,
                            "status publication handler is unavailable",
                        )),
                    };
                }
            }
        }
        if tokio::time::Instant::now() >= deadline {
            return Err(error(
                ErrorCode::INVALID_REQUEST,
                format!(
                    "plugin `{plugin_name}` published its status before it finished initializing"
                ),
            ));
        }
        tokio::time::sleep(INITIALIZATION_POLL).await;
    }
}

fn envelope(
    plugin_name: &str,
    request_id: u64,
    payload: proto::envelope::Payload,
) -> proto::Envelope {
    proto::Envelope {
        protocol_version: PROTOCOL_VERSION,
        plugin_id: plugin_name.to_string(),
        request_id,
        payload: Some(payload),
    }
}

fn error(code: ErrorCode, message: impl Into<String>) -> proto::ErrorResponse {
    proto::ErrorResponse {
        code: code.0,
        message: message.into(),
        data_json: String::new(),
    }
}

#[cfg(test)]
#[path = "status_publication_tests.rs"]
mod tests;
