//! Bound calls never start, restart, select or retry a provider.
use super::{ExternalPlugin, PROTOCOL_VERSION, proto};
use mesh_llm_plugin::operation_binding::BoundOperationResponse as Response;
use std::sync::atomic::Ordering;
use tokio::sync::{mpsc, oneshot};
use tokio::time::Instant;

impl ExternalPlugin {
    pub(crate) async fn bound_generation(&self) -> Option<u64> {
        let runtime = self.runtime.lock().await;
        let runtime = runtime.as_ref()?;
        if !runtime.authenticated_peer || self.manifest.lock().await.is_none() {
            return None;
        }
        Some(runtime.generation)
    }

    pub(crate) async fn admit_bound_operation(
        &self,
        generation: u64,
        operation: &str,
        input_json: &str,
        deadline: Instant,
    ) -> Result<BoundAdmission, Response> {
        let handles = {
            let runtime = self.runtime.lock().await;
            runtime
                .as_ref()
                .filter(|runtime| runtime.generation == generation && runtime.authenticated_peer)
                .map(|runtime| (runtime.outbound_tx.clone(), runtime.pending.clone()))
        };
        let Some((sender, pending)) = handles else {
            return Err(Response::NotDispatched {
                reason: "provider generation unavailable".into(),
            });
        };
        let permit = match tokio::time::timeout_at(deadline, sender.reserve_owned()).await {
            Ok(Ok(permit)) => permit,
            _ => {
                return Err(Response::NotDispatched {
                    reason: "provider queue unavailable".into(),
                });
            }
        };
        let request_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        let Ok(mut locked) = tokio::time::timeout_at(deadline, pending.lock()).await else {
            return Err(Response::NotDispatched {
                reason: "provider admission deadline".into(),
            });
        };
        locked.insert(request_id, tx);
        drop(locked);
        let cleanup = super::request_dispatch::PendingRequest {
            pending,
            request_id,
        };
        let envelope = proto::Envelope {
            protocol_version: PROTOCOL_VERSION,
            plugin_id: self.spec.name.clone(),
            request_id,
            payload: Some(proto::envelope::Payload::InvokeServiceRequest(
                proto::InvokeServiceRequest {
                    kind: proto::ServiceKind::Operation as i32,
                    service_name: operation.into(),
                    input_json: input_json.into(),
                },
            )),
        };
        Ok(BoundAdmission {
            permit,
            envelope,
            rx,
            cleanup,
        })
    }
}

pub(crate) struct BoundAdmission {
    permit: mpsc::OwnedPermit<proto::Envelope>,
    envelope: proto::Envelope,
    rx: oneshot::Receiver<anyhow::Result<proto::Envelope>>,
    cleanup: super::request_dispatch::PendingRequest,
}

pub(crate) struct BoundDispatched {
    rx: oneshot::Receiver<anyhow::Result<proto::Envelope>>,
    _cleanup: super::request_dispatch::PendingRequest,
}

impl BoundAdmission {
    pub(crate) fn send(self) -> BoundDispatched {
        self.permit.send(self.envelope);
        BoundDispatched {
            rx: self.rx,
            _cleanup: self.cleanup,
        }
    }
}

impl BoundDispatched {
    pub(crate) async fn response(self, deadline: Instant) -> Response {
        match tokio::time::timeout_at(deadline, self.rx).await {
            Ok(Ok(Ok(envelope))) => match envelope.payload {
                Some(proto::envelope::Payload::InvokeServiceResponse(response)) => {
                    Response::Responded {
                        handle: String::new(),
                        output_json: response.output_json,
                        is_error: response.is_error,
                    }
                }
                Some(proto::envelope::Payload::ErrorResponse(error)) => Response::Responded {
                    handle: String::new(),
                    output_json: error.data_json,
                    is_error: true,
                },
                _ => Response::Uncertain {
                    reason: "unexpected provider reply".into(),
                },
            },
            _ => Response::Uncertain {
                reason: "provider reply unavailable after enqueue".into(),
            },
        }
    }
}
