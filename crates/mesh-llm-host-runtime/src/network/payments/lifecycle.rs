//! Local payment observations over the existing plugin channel transport.
use crate::mesh::Node;
use mesh_llm_payments::service::PaymentService;
use std::sync::Arc;

pub(crate) const CHANNEL: &str = "payment.lifecycle.v1";
tokio::task_local! { static EXCHANGE_ID: Option<String>; }

pub(crate) fn exchange_id() -> Option<String> {
    EXCHANGE_ID.try_with(Clone::clone).ok().flatten()
}

pub(crate) async fn scope<T>(
    id: Option<String>,
    future: impl std::future::Future<Output = T>,
) -> T {
    EXCHANGE_ID.scope(id, future).await
}

pub(crate) fn subscribe(node: Node, service: &Arc<PaymentService>) {
    let mut events = service.ledger.subscribe_events();
    tokio::spawn(async move {
        loop {
            let received = tokio::select! {
                result = events.recv() => result,
                _ = tokio::time::sleep(std::time::Duration::from_secs(1)) => {
                    if node.endpoint.is_closed() { break; }
                    continue;
                }
            };
            let event = match received {
                Ok(event) => event,
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                    tracing::warn!("payment lifecycle observer lagged; events were dropped");
                    continue;
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            };
            if node.endpoint.is_closed() {
                break;
            }
            let Some(manager) = node.plugin_manager().await else {
                continue;
            };
            let Ok(body) = serde_json::to_vec(&event) else {
                continue;
            };
            // Bounded best-effort observation, never a dependency of settlement.
            let _ = tokio::time::timeout(
                std::time::Duration::from_secs(1),
                manager.broadcast_channel_message(
                    CHANNEL,
                    "application/json",
                    body,
                    &event.exchange_id,
                ),
            )
            .await;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn concurrent_scopes_keep_the_evidence_id_and_do_not_leak() {
        assert_eq!(exchange_id(), None);
        let check = |id: &'static str| {
            scope(Some(id.into()), async move {
                tokio::task::yield_now().await;
                assert_eq!(exchange_id().as_deref(), Some(id));
                assert_eq!(tokio::spawn(async { exchange_id() }).await.unwrap(), None);
            })
        };
        tokio::join!(check("one"), check("two"));
        assert_eq!(exchange_id(), None);
    }
}
