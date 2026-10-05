//! Metadata-only lifecycle ownership for one management API request.

use std::{
    net::SocketAddr,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use mesh_llm_events::logging::{
    events::LifecycleEvent, identifiers::RequestId, replay::ReplayChannel,
};

use super::{
    CallerPathType, LifecycleGuard, LoggingService, RequestSummaryMetadata, TerminalOutcome,
};

pub(crate) struct ManagementRequestLifecycle {
    request_id: RequestId,
    guard: LifecycleGuard,
    service: Arc<LoggingService>,
    terminalized: AtomicBool,
}

impl ManagementRequestLifecycle {
    pub(crate) fn register(
        service: Arc<LoggingService>,
        request_id: RequestId,
        method_route: &'static str,
    ) -> Self {
        Self::register_with_caller_addr(service, request_id, method_route, None)
    }

    pub(crate) fn register_with_source_addr(
        service: Arc<LoggingService>,
        request_id: RequestId,
        method_route: &'static str,
        source_addr: SocketAddr,
    ) -> Self {
        Self::register_with_caller_addr(service, request_id, method_route, Some(source_addr))
    }

    fn register_with_caller_addr(
        service: Arc<LoggingService>,
        request_id: RequestId,
        method_route: &'static str,
        source_addr: Option<SocketAddr>,
    ) -> Self {
        let caller_addr = source_addr.map(|addr| addr.to_string());
        let metadata = RequestSummaryMetadata::from_parts(
            Some(method_route),
            None,
            Some("management_api"),
            Some(method_route),
        )
        .with_source(Some("direct_http"))
        .with_method(Some(management_method_label(method_route)))
        .with_caller_identity(
            None,
            caller_addr.as_deref(),
            caller_addr.as_ref().map(|_| CallerPathType::LocalHttp),
        );
        let (guard, _) = service.register_request_with_metadata(request_id, metadata.clone());
        if let Ok(payload) = serde_json::to_string(&LifecycleEvent::RouteSelected {
            model: None,
            provider: metadata.provider().map(str::to_owned),
            engine: metadata.engine().map(str::to_owned),
        }) {
            let _ = service.enqueue_event(request_id, ReplayChannel::Operations, payload);
        }
        Self {
            request_id,
            guard,
            service,
            terminalized: AtomicBool::new(false),
        }
    }

    pub(crate) fn finish_status(&self, status: u16) {
        let outcome = if status < 400 {
            TerminalOutcome::CompletedWithStatus(status)
        } else if status < 500 {
            TerminalOutcome::RejectedWithStatus {
                reason: Some("management_http_rejected".into()),
                status_code: status,
            }
        } else {
            TerminalOutcome::FailedWithStatus {
                error: "management_http_failed".into(),
                status_code: status,
            }
        };
        self.terminal(outcome);
    }

    pub(crate) fn fail_dispatch(&self) {
        self.terminal(TerminalOutcome::Failed("management_dispatch_failed".into()));
    }

    pub(crate) fn request_id(&self) -> RequestId {
        self.request_id
    }

    fn terminal(&self, outcome: TerminalOutcome) {
        if self.terminalized.swap(true, Ordering::AcqRel) {
            return;
        }
        let _ = self
            .service
            .transition_terminal(self.request_id, &self.guard, outcome);
    }
}

fn management_method_label(method_route: &str) -> &'static str {
    if method_route.starts_with("management_get_") {
        "GET"
    } else if method_route == "management_post" {
        "POST"
    } else if method_route == "management_put" {
        "PUT"
    } else if method_route == "management_patch" {
        "PATCH"
    } else if method_route == "management_delete" {
        "DELETE"
    } else {
        "OTHER"
    }
}

impl Drop for ManagementRequestLifecycle {
    fn drop(&mut self) {
        // A connection task can be aborted while an SSE response owns the
        // future. The guard alone cannot observe that scope loss because it
        // is still retained by this lifecycle object, so make the terminal
        // transition explicit and deterministic.
        self.terminal(TerminalOutcome::Dropped(Some(
            "management_lifecycle_dropped".into(),
        )));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn management_registration_captures_only_static_summary_metadata() {
        let service = Arc::new(LoggingService::new_disabled(Default::default()));
        let request_id = RequestId::new();
        let lifecycle = ManagementRequestLifecycle::register(
            Arc::clone(&service),
            request_id,
            "management_get_status",
        );

        lifecycle.finish_status(200);

        let summary = service
            .registry_ref()
            .get_recent(&request_id.as_uuid().to_string())
            .expect("terminal request summary");
        assert_eq!(summary.metadata.route(), Some("management_get_status"));
        assert_eq!(summary.metadata.source(), Some("direct_http"));
        assert_eq!(summary.metadata.method(), Some("GET"));
        assert!(summary.metadata.model().is_none());
        assert_eq!(summary.metadata.provider(), Some("management_api"));
        assert_eq!(summary.metadata.engine(), Some("management_get_status"));
        assert!(summary.metadata.caller_endpoint_id().is_none());
        assert!(summary.metadata.caller_addr().is_none());
        assert!(summary.metadata.caller_path_type().is_none());
    }

    #[test]
    fn management_registration_captures_truthful_local_http_caller() {
        let service = Arc::new(LoggingService::new_disabled(Default::default()));
        let request_id = RequestId::new();
        let source_addr = "[::1]:40123".parse().expect("IPv6 source address");
        let lifecycle = ManagementRequestLifecycle::register_with_source_addr(
            Arc::clone(&service),
            request_id,
            "management_post",
            source_addr,
        );

        lifecycle.finish_status(200);

        let summary = service
            .registry_ref()
            .get_recent(&request_id.as_uuid().to_string())
            .expect("terminal request summary");
        assert_eq!(summary.metadata.caller_addr(), Some("[::1]:40123"));
        assert_eq!(summary.metadata.caller_path_type(), Some("local_http"));
        assert!(summary.metadata.caller_endpoint_id().is_none());
    }

    #[test]
    fn patch_management_registration_preserves_the_http_method() {
        let service = Arc::new(LoggingService::new_disabled(Default::default()));
        let request_id = RequestId::new();
        let lifecycle = ManagementRequestLifecycle::register(
            Arc::clone(&service),
            request_id,
            "management_patch",
        );

        lifecycle.finish_status(200);

        let summary = service
            .registry_ref()
            .get_recent(&request_id.as_uuid().to_string())
            .expect("terminal PATCH request summary");
        assert_eq!(summary.metadata.route(), Some("management_patch"));
        assert_eq!(summary.metadata.method(), Some("PATCH"));
    }

    #[test]
    fn dropped_management_lifecycle_terminalizes_the_registered_request() {
        let service = Arc::new(LoggingService::new_disabled(Default::default()));
        let request_id = RequestId::new();
        let lifecycle = ManagementRequestLifecycle::register(
            Arc::clone(&service),
            request_id,
            "management_get_events",
        );

        drop(lifecycle);

        let summary = service
            .registry_ref()
            .get_recent(&request_id.as_uuid().to_string())
            .expect("dropped request is terminalized");
        assert_eq!(summary.state, "dropped");
    }
}
