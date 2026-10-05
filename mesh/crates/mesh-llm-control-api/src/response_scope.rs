//! Request-scoped response observation supplied by the owning management host.
//! Framing records only committed response status; admission, terminalization
//! and cancellation remain the lifecycle owner's responsibility.
use mesh_llm_events::logging::identifiers::RequestId;
use std::sync::Arc;

pub trait ResponseObserver: Send + Sync {
    fn request_id(&self) -> RequestId;
    fn record_status(&self, status: u16);
}

tokio::task_local! {
    static RESPONSE_OBSERVER: Arc<dyn ResponseObserver>;
}

pub async fn scope<F: std::future::Future>(
    observer: Arc<dyn ResponseObserver>,
    future: F,
) -> F::Output {
    RESPONSE_OBSERVER.scope(observer, future).await
}

pub fn record_response_status(status: u16) {
    let _ = RESPONSE_OBSERVER.try_with(|observer| observer.record_status(status));
}

pub fn response_request_id_header() -> Option<String> {
    RESPONSE_OBSERVER
        .try_with(|observer| observer.request_id().as_uuid().to_string())
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU16, Ordering};

    struct Observer {
        request_id: RequestId,
        status: AtomicU16,
    }
    impl Observer {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                request_id: RequestId::new(),
                status: AtomicU16::new(0),
            })
        }
    }
    impl ResponseObserver for Observer {
        fn request_id(&self) -> RequestId {
            self.request_id
        }
        fn record_status(&self, status: u16) {
            self.status.store(status, Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn nested_response_scopes_restore_the_outer_observer() {
        let outer = Observer::new();
        let inner = Observer::new();
        assert!(response_request_id_header().is_none());
        scope(outer.clone(), async {
            assert_eq!(
                response_request_id_header(),
                Some(outer.request_id.as_uuid().to_string())
            );
            scope(inner.clone(), async {
                record_response_status(409);
                assert_eq!(
                    response_request_id_header(),
                    Some(inner.request_id.as_uuid().to_string())
                );
            })
            .await;
            record_response_status(201);
            assert_eq!(
                response_request_id_header(),
                Some(outer.request_id.as_uuid().to_string())
            );
        })
        .await;
        assert!(response_request_id_header().is_none());
        assert_eq!(outer.status.load(Ordering::SeqCst), 201);
        assert_eq!(inner.status.load(Ordering::SeqCst), 409);
    }

    #[tokio::test]
    async fn response_scope_rewrites_duplicate_upstream_ids_without_recording_unsent_status() {
        let observer = Observer::new();
        scope(observer.clone(), async {
            let (head, status) = crate::http::managed_response_head(
                b"HTTP/1.1 503 Unavailable\r\nX-Request-Id: upstream\r\nx-request-id: duplicate\r\nContent-Length: 0\r\n\r\n".to_vec()
            ).unwrap();
            let head = String::from_utf8(head).unwrap();
            assert_eq!(status, 503);
            assert_eq!(head.matches("x-request-id:").count(), 1);
            assert!(head.contains(&observer.request_id.as_uuid().to_string()));
            assert!(!head.contains("upstream"));
            assert!(!head.contains("duplicate"));
            assert_eq!(observer.status.load(Ordering::SeqCst), 0);
        }).await;
    }
}
