//! A plugin connection's `SubscribeInferenceObservationRequest`.

use super::hub::{ObservationHub, SubscriberId};
use super::{delivery, proto};
use crate::plugin::runtime::PluginRuntime;
use crate::plugin::runtime::status_publication::StatusDeclaration;
use crate::plugin::{PROTOCOL_VERSION, PluginMeshEvent};
use rmcp::model::ErrorCode;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, mpsc, oneshot};

const SUBSCRIBE_TIMEOUT: Duration = Duration::from_secs(crate::plugin::REQUEST_TIMEOUT_SECS);
const INITIALIZATION_POLL: Duration = Duration::from_millis(10);

/// The subscription a plugin connection holds, replaced by each new request
/// and ended with the connection.
#[derive(Clone, Default)]
pub(crate) struct ConnectionSubscription(Arc<std::sync::Mutex<Option<SubscriberId>>>);

impl ConnectionSubscription {
    fn replace(&self, hub: &ObservationHub, id: SubscriberId) {
        let previous = self
            .0
            .lock()
            .expect("subscription slot poisoned")
            .replace(id);
        if let Some(previous) = previous {
            hub.unsubscribe(previous);
        }
    }

    /// The connection is gone.
    pub(crate) fn end(&self, hub: &ObservationHub) {
        if let Some(id) = self.0.lock().expect("subscription slot poisoned").take() {
            hub.unsubscribe(id);
        }
    }
}

pub(crate) struct SubscribeRequest {
    pub(crate) plugin_name: String,
    pub(crate) generation: u64,
    pub(crate) request_id: u64,
    pub(crate) runtime: Arc<Mutex<Option<PluginRuntime>>>,
    pub(crate) mesh_tx: mpsc::Sender<PluginMeshEvent>,
    pub(crate) outbound_tx: mpsc::Sender<proto::Envelope>,
    pub(crate) subscription: ConnectionSubscription,
}

/// Answer the request on its own task: a snapshot and then the feed, or the
/// reason it is refused.
pub(crate) fn spawn(hub: Arc<ObservationHub>, request: SubscribeRequest) {
    tokio::spawn(async move {
        let reply = |payload| proto::Envelope {
            protocol_version: PROTOCOL_VERSION,
            plugin_id: request.plugin_name.clone(),
            request_id: request.request_id,
            payload: Some(payload),
        };
        if let Err(error) = admit(&request).await {
            let _ = request
                .outbound_tx
                .send(reply(proto::envelope::Payload::ErrorResponse(error)))
                .await;
            return;
        }
        let (id, notify, snapshot) = hub.subscribe();
        request.subscription.replace(&hub, id);
        let answer = proto::envelope::Payload::SubscribeInferenceObservationResponse(
            proto::SubscribeInferenceObservationResponse {
                snapshot: Some(snapshot),
            },
        );
        // The answer goes out before the first batch, on the same queue.
        if request.outbound_tx.send(reply(answer)).await.is_err() {
            hub.unsubscribe(id);
            return;
        }
        delivery::spawn(
            hub,
            id,
            notify,
            request.plugin_name.clone(),
            request.outbound_tx.clone(),
        );
    });
}

/// Whether this connection may subscribe: still the plugin's current one,
/// initialized, declaring `inference_observation`, and granted by the operator.
async fn admit(request: &SubscribeRequest) -> Result<(), proto::ErrorResponse> {
    let name = &request.plugin_name;
    let deadline = tokio::time::Instant::now() + SUBSCRIBE_TIMEOUT;
    loop {
        {
            let runtime = request.runtime.lock().await;
            let Some(current) = runtime
                .as_ref()
                .filter(|runtime| runtime.generation == request.generation)
            else {
                return Err(error(format!(
                    "plugin `{name}` subscribed from a connection that is no longer current"
                )));
            };
            if current.status_declaration != StatusDeclaration::Pending {
                if !current.declares_inference_observation {
                    return Err(error(format!(
                        "plugin `{name}` does not declare `inference_observation` in its manifest"
                    )));
                }
                break;
            }
        }
        if tokio::time::Instant::now() >= deadline {
            return Err(error(format!(
                "plugin `{name}` subscribed before it finished initializing"
            )));
        }
        tokio::time::sleep(INITIALIZATION_POLL).await;
    }
    if granted(request, deadline).await {
        Ok(())
    } else {
        Err(error(format!(
            "plugin `{name}` may not observe local inference: the operator has not set \
             allow_inference_observation = true for it"
        )))
    }
}

/// Ask the node, which reads the operator's live config.
async fn granted(request: &SubscribeRequest, deadline: tokio::time::Instant) -> bool {
    let (response_tx, response_rx) = oneshot::channel();
    let asked = tokio::time::timeout_at(
        deadline,
        request
            .mesh_tx
            .send(PluginMeshEvent::InferenceObservationGrant {
                plugin_id: request.plugin_name.clone(),
                response_tx,
            }),
    )
    .await;
    if !matches!(asked, Ok(Ok(()))) {
        return false;
    }
    matches!(
        tokio::time::timeout_at(deadline, response_rx).await,
        Ok(Ok(true))
    )
}

fn error(message: String) -> proto::ErrorResponse {
    proto::ErrorResponse {
        code: ErrorCode::INVALID_REQUEST.0,
        message,
        data_json: String::new(),
    }
}
