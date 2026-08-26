use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use axum::{
    body::Body,
    http::{Request, StatusCode},
    response::Response,
};
use futures_util::stream;
use http_body_util::BodyExt;
use openai_frontend::{
    CancellationToken, ChatCompletionResponse, ChatCompletionStream, CompletionResponse,
    CompletionStream, ModelObject, OpenAiBackend, OpenAiBackendOperation, OpenAiFailure,
    OpenAiFrontendConfig, OpenAiFrontendRoute, OpenAiLifecycleEvent, OpenAiLifecycleObserver,
    OpenAiRejection, OpenAiRequestContext, OpenAiResult, OpenAiTerminalResult, OpenAiUsage, Usage,
    router_for_with_config,
};
use serde_json::json;
use tower::ServiceExt;

const HEALTH_ID: &str = "35be0fbb-d9c2-470f-98a2-2132f4d84ffc";
const READY_ID: &str = "9d73db08-073a-4e0a-a35c-2c3f17af2540";
const MODELS_ID: &str = "0245b681-963d-4e1e-b702-8897028af6ba";
const CHAT_ID: &str = "93e50221-4d99-49f8-aeaf-36bc64c70a96";
const BACKEND_FAILURE_ID: &str = "2512a930-83f4-4a97-8a5a-befa57e878ae";
const COMPLETION_ID: &str = "1cb1059c-a1eb-45b1-ab77-857b9f0db40c";
const RESPONSES_ID: &str = "7d3e12a4-8641-4703-9368-b31a442c2297";
const INVALID_ID: &str = "eaaef3b4-1f02-4d73-a0be-16cb917324a0";
const OVERSIZED_ID: &str = "d575fac3-5e76-49a4-95d0-b4f4bcb108ea";
const METHOD_ID: &str = "6e326f78-5e07-4143-8910-cdd884b4d59f";
const NOT_FOUND_ID: &str = "a28971f7-2f42-4e61-9233-a8a02872c418";
const STREAM_ID: &str = "80421e2b-4821-426b-8b53-0e10488789f6";
const STREAM_ERROR_ID: &str = "3dd41d54-70ca-4d1a-a1fc-758d78ee6a3e";
const STREAM_DROP_ID: &str = "34bbce17-f008-4d8d-9d2c-2f6fd3220c7d";
const STREAM_CANCEL_ID: &str = "d81c76f6-09c7-4b02-a20f-d4e50643a6c9";

#[derive(Default)]
struct RecordingObserver {
    events: Mutex<Vec<OpenAiLifecycleEvent>>,
}

impl RecordingObserver {
    fn events(&self) -> Vec<OpenAiLifecycleEvent> {
        self.events
            .lock()
            .expect("test observer lock poisoned")
            .clone()
    }
}

impl OpenAiLifecycleObserver for RecordingObserver {
    fn observe(&self, event: &OpenAiLifecycleEvent) {
        self.events
            .lock()
            .expect("test observer lock poisoned")
            .push(event.clone());
    }
}

#[derive(Default)]
struct TestBackend {
    stream_cancellation: Arc<Mutex<Option<CancellationToken>>>,
    stream_request_ids: Arc<Mutex<Vec<String>>>,
    stream_context_flags: Arc<Mutex<Vec<(bool, bool)>>>,
}

#[async_trait]
impl OpenAiBackend for TestBackend {
    async fn models(&self) -> OpenAiResult<Vec<ModelObject>> {
        Ok(vec![ModelObject::new("test-model")])
    }

    async fn chat_completion(
        &self,
        request: openai_frontend::ChatCompletionRequest,
    ) -> OpenAiResult<ChatCompletionResponse> {
        if request.model == "backend-error" {
            return Err(openai_frontend::OpenAiError::backend("backend failed"));
        }
        Ok(ChatCompletionResponse::new(
            request.model,
            "ok",
            Usage::new(2, 1).with_cached_tokens(1),
        ))
    }

    async fn chat_completion_stream(
        &self,
        request: openai_frontend::ChatCompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionStream> {
        self.stream_request_ids
            .lock()
            .expect("test request-id lock poisoned")
            .push(
                context
                    .request_id()
                    .expect("frontend stream context should carry a request ID")
                    .as_ref()
                    .to_string(),
            );
        self.stream_context_flags
            .lock()
            .expect("test stream-context lock poisoned")
            .push((
                context.observes_stream_usage(),
                context.has_trusted_agent_session(),
            ));
        if request.model == "stream-error" {
            return Ok(Box::pin(stream::iter(vec![Err(
                openai_frontend::OpenAiError::backend("stream failed"),
            )])));
        }
        if request.model == "pending" {
            *self
                .stream_cancellation
                .lock()
                .expect("test cancellation lock poisoned") = Some(context.cancellation_token());
            return Ok(Box::pin(stream::pending()));
        }

        Ok(Box::pin(stream::iter(vec![
            Ok(openai_frontend::ChatCompletionChunk::usage(
                request.model.clone(),
                Usage::new(8, 2).with_cached_tokens(5),
            )),
            Ok(openai_frontend::ChatCompletionChunk::done(request.model)),
        ])))
    }

    async fn completion(
        &self,
        request: openai_frontend::CompletionRequest,
    ) -> OpenAiResult<CompletionResponse> {
        Ok(CompletionResponse::new(
            request.model,
            "ok",
            Usage::new(2, 1),
        ))
    }

    async fn completion_stream(
        &self,
        request: openai_frontend::CompletionRequest,
        _context: OpenAiRequestContext,
    ) -> OpenAiResult<CompletionStream> {
        Ok(Box::pin(stream::iter(vec![Ok(
            openai_frontend::CompletionChunk::done(request.model),
        )])))
    }
}

#[tokio::test]
async fn observer_tracks_non_streaming_ingress_rejections_and_backend_dispatch() {
    let observer = Arc::new(RecordingObserver::default());
    let app = observed_app(Arc::new(TestBackend::default()), observer.clone());

    assert_response_id(
        app.clone()
            .oneshot(request("GET", "/health", HEALTH_ID, Body::empty()))
            .await
            .expect("health response"),
        HEALTH_ID,
    );
    assert_response_id(
        app.clone()
            .oneshot(request("GET", "/readyz", READY_ID, Body::empty()))
            .await
            .expect("ready response"),
        READY_ID,
    );
    assert_response_id(
        app.clone()
            .oneshot(request("GET", "/v1/models", MODELS_ID, Body::empty()))
            .await
            .expect("models response"),
        MODELS_ID,
    );
    assert_response_id(
        post_json(
            &app,
            "/v1/chat/completions",
            CHAT_ID,
            json!({"model":"test-model","messages":[{"role":"user","content":"hello"}]}),
        )
        .await,
        CHAT_ID,
    );
    assert_response_id(
        post_json(
            &app,
            "/v1/chat/completions",
            BACKEND_FAILURE_ID,
            json!({"model":"backend-error","messages":[{"role":"user","content":"hello"}]}),
        )
        .await,
        BACKEND_FAILURE_ID,
    );
    assert_response_id(
        post_json(
            &app,
            "/v1/completions",
            COMPLETION_ID,
            json!({"model":"test-model","prompt":"hello"}),
        )
        .await,
        COMPLETION_ID,
    );
    assert_response_id(
        post_json(
            &app,
            "/v1/responses",
            RESPONSES_ID,
            json!({"model":"test-model","input":"hello"}),
        )
        .await,
        RESPONSES_ID,
    );
    assert_response_id(
        app.clone()
            .oneshot(request(
                "POST",
                "/v1/chat/completions",
                INVALID_ID,
                Body::from("{"),
            ))
            .await
            .expect("invalid JSON response"),
        INVALID_ID,
    );
    assert_response_id(
        app.clone()
            .oneshot(request(
                "GET",
                "/v1/chat/completions",
                METHOD_ID,
                Body::empty(),
            ))
            .await
            .expect("method rejection response"),
        METHOD_ID,
    );
    assert_response_id(
        app.clone()
            .oneshot(request("GET", "/missing", NOT_FOUND_ID, Body::empty()))
            .await
            .expect("not-found response"),
        NOT_FOUND_ID,
    );

    let limited = router_for_with_config(
        Arc::new(TestBackend::default()),
        OpenAiFrontendConfig::default()
            .with_max_request_body_bytes(32)
            .with_lifecycle_observer(observer.clone()),
    );
    assert_response_id(
        limited
            .oneshot(request(
                "POST",
                "/v1/chat/completions",
                OVERSIZED_ID,
                Body::from(
                    json!({"model":"test-model","messages":[{"role":"user","content":"body larger than thirty two bytes"}]}).to_string(),
                ),
            ))
            .await
            .expect("oversized response"),
        OVERSIZED_ID,
    );

    let events = observer.events();
    for request_id in [
        HEALTH_ID,
        READY_ID,
        MODELS_ID,
        CHAT_ID,
        BACKEND_FAILURE_ID,
        COMPLETION_ID,
        RESPONSES_ID,
        INVALID_ID,
        OVERSIZED_ID,
        METHOD_ID,
        NOT_FOUND_ID,
    ] {
        assert_admitted_and_has_one_terminal(&events, request_id);
    }
    assert_backend_dispatches_have_one_terminal(&events);
    let chat_usage = events_for(&events, CHAT_ID)
        .into_iter()
        .filter_map(|event| match event {
            OpenAiLifecycleEvent::ResponseCompleted { usage, .. } => Some(*usage),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        chat_usage,
        vec![OpenAiUsage {
            prompt_tokens: 2,
            cached_tokens: Some(1),
            completion_tokens: 1,
            total_tokens: 3,
        }]
    );
    assert_admitted_route(&events, HEALTH_ID, OpenAiFrontendRoute::Health);
    assert_admitted_route(&events, READY_ID, OpenAiFrontendRoute::Readyz);
    assert_admitted_route(&events, MODELS_ID, OpenAiFrontendRoute::Models);

    assert_terminal_matches(&events, HEALTH_ID, |event| {
        matches!(
            event,
            OpenAiLifecycleEvent::NonStreamTerminal {
                result: OpenAiTerminalResult::Completed { status_code: 200 },
                ..
            }
        )
    });
    assert_terminal_matches(&events, BACKEND_FAILURE_ID, |event| {
        matches!(
            event,
            OpenAiLifecycleEvent::NonStreamTerminal {
                result: OpenAiTerminalResult::Failed {
                    status_code: 502,
                    failure: OpenAiFailure::Backend,
                },
                ..
            }
        )
    });
    assert_terminal_matches(&events, INVALID_ID, |event| {
        matches!(
            event,
            OpenAiLifecycleEvent::Rejected {
                status_code: 400,
                rejection: OpenAiRejection::InvalidRequest,
                ..
            }
        )
    });
    assert_terminal_matches(&events, OVERSIZED_ID, |event| {
        matches!(
            event,
            OpenAiLifecycleEvent::Rejected {
                status_code: 413,
                rejection: OpenAiRejection::PayloadTooLarge,
                ..
            }
        )
    });
    assert_terminal_matches(&events, METHOD_ID, |event| {
        matches!(
            event,
            OpenAiLifecycleEvent::Rejected {
                status_code: 405,
                rejection: OpenAiRejection::MethodNotAllowed,
                ..
            }
        )
    });
    assert_terminal_matches(&events, NOT_FOUND_ID, |event| {
        matches!(
            event,
            OpenAiLifecycleEvent::Rejected {
                status_code: 404,
                rejection: OpenAiRejection::NotFound,
                ..
            }
        )
    });

    assert_eq!(
        backend_operations(&events),
        vec![
            (READY_ID.to_string(), OpenAiBackendOperation::Models),
            (MODELS_ID.to_string(), OpenAiBackendOperation::Models),
            (CHAT_ID.to_string(), OpenAiBackendOperation::ChatCompletion),
            (
                BACKEND_FAILURE_ID.to_string(),
                OpenAiBackendOperation::ChatCompletion,
            ),
            (
                COMPLETION_ID.to_string(),
                OpenAiBackendOperation::Completion,
            ),
            (RESPONSES_ID.to_string(), OpenAiBackendOperation::Responses),
        ]
    );
}

#[tokio::test]
async fn observer_tracks_stream_completion_error_drop_and_cancel_once() {
    let observer = Arc::new(RecordingObserver::default());
    let backend = Arc::new(TestBackend::default());
    let app = observed_app(backend.clone(), observer.clone());

    let response = post_json(
        &app,
        "/v1/chat/completions",
        STREAM_ID,
        json!({"model":"test-model","messages":[{"role":"user","content":"hello"}],"stream":true}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let _ = response.into_body().collect().await.expect("stream body");

    let response = post_json(
        &app,
        "/v1/chat/completions",
        STREAM_ERROR_ID,
        json!({"model":"stream-error","messages":[{"role":"user","content":"hello"}],"stream":true}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let _ = response
        .into_body()
        .collect()
        .await
        .expect("error stream body");

    let response = post_json(
        &app,
        "/v1/chat/completions",
        STREAM_DROP_ID,
        json!({"model":"pending","messages":[{"role":"user","content":"hello"}],"stream":true}),
    )
    .await;
    drop(response);
    let dropped_token = backend
        .stream_cancellation
        .lock()
        .expect("test cancellation lock poisoned")
        .clone()
        .expect("stream backend should receive cancellation context");
    assert!(dropped_token.is_cancelled());

    let response = post_json(
        &app,
        "/v1/chat/completions",
        STREAM_CANCEL_ID,
        json!({"model":"pending","messages":[{"role":"user","content":"hello"}],"stream":true}),
    )
    .await;
    backend
        .stream_cancellation
        .lock()
        .expect("test cancellation lock poisoned")
        .as_ref()
        .expect("stream backend should receive cancellation context")
        .cancel();
    drop(response);

    let events = observer.events();
    for request_id in [STREAM_ID, STREAM_ERROR_ID, STREAM_DROP_ID, STREAM_CANCEL_ID] {
        assert_admitted_and_has_one_terminal(&events, request_id);
    }
    assert_backend_dispatches_have_one_terminal(&events);
    for request_id in [STREAM_ID, STREAM_ERROR_ID] {
        assert_eq!(
            events_for(&events, request_id)
                .into_iter()
                .filter(|event| matches!(event, OpenAiLifecycleEvent::StreamFirstItem { .. }))
                .count(),
            1,
            "a non-empty backend stream emits one first-item event"
        );
    }
    assert_terminal_matches(&events, STREAM_ID, |event| {
        matches!(
            event,
            OpenAiLifecycleEvent::StreamTerminal {
                result: OpenAiTerminalResult::CompletedWithUsage {
                    status_code: 200,
                    usage,
                },
                ..
            } if usage.prompt_tokens == Some(8)
                && usage.completion_tokens == Some(2)
                && usage.total_tokens == Some(10)
        )
    });
    assert_terminal_matches(&events, STREAM_ERROR_ID, |event| {
        matches!(
            event,
            OpenAiLifecycleEvent::StreamTerminal {
                result: OpenAiTerminalResult::Failed {
                    status_code: 502,
                    failure: OpenAiFailure::Backend
                },
                ..
            }
        )
    });
    assert_terminal_matches(&events, STREAM_DROP_ID, |event| {
        matches!(event, OpenAiLifecycleEvent::StreamDropped { .. })
    });
    assert_terminal_matches(&events, STREAM_CANCEL_ID, |event| {
        matches!(event, OpenAiLifecycleEvent::StreamCancelled { .. })
    });
    assert_eq!(
        backend_operations(&events),
        vec![
            (
                STREAM_ID.to_string(),
                OpenAiBackendOperation::ChatCompletionStream,
            ),
            (
                STREAM_ERROR_ID.to_string(),
                OpenAiBackendOperation::ChatCompletionStream,
            ),
            (
                STREAM_DROP_ID.to_string(),
                OpenAiBackendOperation::ChatCompletionStream,
            ),
            (
                STREAM_CANCEL_ID.to_string(),
                OpenAiBackendOperation::ChatCompletionStream,
            ),
        ]
    );
    assert_eq!(
        *backend
            .stream_request_ids
            .lock()
            .expect("test request-id lock poisoned"),
        vec![
            STREAM_ID.to_string(),
            STREAM_ERROR_ID.to_string(),
            STREAM_DROP_ID.to_string(),
            STREAM_CANCEL_ID.to_string(),
        ]
    );
    assert_eq!(
        *backend
            .stream_context_flags
            .lock()
            .expect("test stream-context lock poisoned"),
        vec![(true, false), (true, false), (true, false), (true, false)]
    );
    let completed_usage =
        events_for(&events, STREAM_ID)
            .into_iter()
            .find_map(|event| match event {
                OpenAiLifecycleEvent::ResponseCompleted { usage, .. } => Some(*usage),
                _ => None,
            });
    assert_eq!(
        completed_usage,
        Some(OpenAiUsage {
            prompt_tokens: 8,
            cached_tokens: Some(5),
            completion_tokens: 2,
            total_tokens: 10,
        })
    );
    for request_id in [STREAM_ERROR_ID, STREAM_DROP_ID, STREAM_CANCEL_ID] {
        assert!(
            events_for(&events, request_id)
                .into_iter()
                .all(|event| { !matches!(event, OpenAiLifecycleEvent::ResponseCompleted { .. }) })
        );
    }
}

fn observed_app(backend: Arc<TestBackend>, observer: Arc<RecordingObserver>) -> axum::Router {
    router_for_with_config(
        backend,
        OpenAiFrontendConfig::default().with_lifecycle_observer(observer),
    )
}

fn request(method: &str, uri: &str, request_id: &str, body: Body) -> Request<Body> {
    Request::builder()
        .method(method)
        .uri(uri)
        .header("x-request-id", request_id)
        .header("content-type", "application/json")
        .body(body)
        .expect("test request should build")
}

async fn post_json(
    app: &axum::Router,
    uri: &str,
    request_id: &str,
    value: serde_json::Value,
) -> Response {
    app.clone()
        .oneshot(request(
            "POST",
            uri,
            request_id,
            Body::from(value.to_string()),
        ))
        .await
        .expect("post response")
}

fn assert_response_id(response: Response, request_id: &str) {
    assert_eq!(response.headers()["x-request-id"], request_id);
}

fn assert_admitted_and_has_one_terminal(events: &[OpenAiLifecycleEvent], request_id: &str) {
    let request_events = events_for(events, request_id);
    assert!(matches!(
        request_events.first(),
        Some(OpenAiLifecycleEvent::Admitted { .. })
    ));
    assert_eq!(
        request_events
            .iter()
            .filter(|event| is_terminal(event))
            .count(),
        1,
        "request {request_id} should have exactly one terminal lifecycle event"
    );
}

fn assert_admitted_route(
    events: &[OpenAiLifecycleEvent],
    request_id: &str,
    route: OpenAiFrontendRoute,
) {
    assert!(matches!(
        events_for(events, request_id).first(),
        Some(OpenAiLifecycleEvent::Admitted { context }) if context.route == route
    ));
}

fn assert_terminal_matches(
    events: &[OpenAiLifecycleEvent],
    request_id: &str,
    matches_terminal: impl Fn(&OpenAiLifecycleEvent) -> bool,
) {
    let terminal = events_for(events, request_id)
        .into_iter()
        .find(|event| is_terminal(event))
        .expect("request should have a terminal event");
    assert!(matches_terminal(terminal));
}

fn backend_operations(events: &[OpenAiLifecycleEvent]) -> Vec<(String, OpenAiBackendOperation)> {
    events
        .iter()
        .filter_map(|event| match event {
            OpenAiLifecycleEvent::BackendDispatched { context, operation } => {
                Some((context.request_id.as_ref().to_string(), *operation))
            }
            _ => None,
        })
        .collect()
}

fn assert_backend_dispatches_have_one_terminal(events: &[OpenAiLifecycleEvent]) {
    for event in events {
        let OpenAiLifecycleEvent::BackendDispatched { context, operation } = event else {
            continue;
        };
        let matching = events
            .iter()
            .filter(|candidate| {
                matches!(
                    candidate,
                    OpenAiLifecycleEvent::BackendTerminal {
                        context: terminal_context,
                        operation: terminal_operation,
                        ..
                    } if terminal_context == context && terminal_operation == operation
                )
            })
            .count();
        assert_eq!(
            matching, 1,
            "every backend dispatch must have exactly one correlated terminal"
        );
    }
}

fn events_for<'a>(
    events: &'a [OpenAiLifecycleEvent],
    request_id: &str,
) -> Vec<&'a OpenAiLifecycleEvent> {
    events
        .iter()
        .filter(|event| event_request_id(event) == request_id)
        .collect()
}

fn event_request_id(event: &OpenAiLifecycleEvent) -> String {
    event_context(event).request_id.as_ref().to_string()
}

fn event_context(event: &OpenAiLifecycleEvent) -> &openai_frontend::OpenAiLifecycleContext {
    match event {
        OpenAiLifecycleEvent::Admitted { context }
        | OpenAiLifecycleEvent::StreamDropped { context }
        | OpenAiLifecycleEvent::StreamCancelled { context }
        | OpenAiLifecycleEvent::RequestCancelled { context }
        | OpenAiLifecycleEvent::Rejected { context, .. }
        | OpenAiLifecycleEvent::BackendDispatched { context, .. }
        | OpenAiLifecycleEvent::BackendTerminal { context, .. }
        | OpenAiLifecycleEvent::StreamFirstItem { context, .. }
        | OpenAiLifecycleEvent::ResponseCompleted { context, .. }
        | OpenAiLifecycleEvent::NonStreamTerminal { context, .. }
        | OpenAiLifecycleEvent::StreamTerminal { context, .. } => context,
    }
}

fn is_terminal(event: &OpenAiLifecycleEvent) -> bool {
    matches!(
        event,
        OpenAiLifecycleEvent::Rejected { .. }
            | OpenAiLifecycleEvent::NonStreamTerminal { .. }
            | OpenAiLifecycleEvent::StreamTerminal { .. }
            | OpenAiLifecycleEvent::StreamDropped { .. }
            | OpenAiLifecycleEvent::StreamCancelled { .. }
            | OpenAiLifecycleEvent::RequestCancelled { .. }
    )
}
