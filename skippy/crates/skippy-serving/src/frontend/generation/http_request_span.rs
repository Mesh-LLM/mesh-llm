//! One span per OpenAI HTTP request. Everything the handler does, including
//! blocking work it hands off, nests beneath it. A streamed response keeps the
//! span open until its body ends or the client goes away, so the span covers
//! the whole generation instead of stopping at the response headers.

use std::{
    pin::Pin,
    task::{Context, Poll},
};

use axum::{
    body::{Body, BodyDataStream, Bytes},
    extract::State,
    http::{Request, header},
    middleware::Next,
    response::Response,
};
use futures_util::Stream;
use serde_json::json;

use crate::telemetry::{
    Telemetry,
    active_span::{ActiveSpan, SpanLevel},
    trace_context,
};

pub(in crate::frontend) async fn openai_http_telemetry(
    State(telemetry): State<Telemetry>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let mut span = ActiveSpan::start(
        Some(&telemetry),
        "stage.openai_http_request",
        SpanLevel::Summary,
    )
    .with_metrics();
    telemetry.record_resources_on(span.frame(), "request_start");
    let method = request.method().to_string();
    let path = request.uri().path().to_string();
    let response = trace_context::in_task(span.frame().clone(), next.run(request)).await;
    telemetry.record_resources_on(span.frame(), "request_response");
    let streamed = is_event_stream(&response);
    span.extend_attributes([
        ("llama_stage.http_method".to_string(), json!(method)),
        ("llama_stage.http_path".to_string(), json!(path)),
        (
            "llama_stage.http_status".to_string(),
            json!(response.status().as_u16()),
        ),
        // Time to response headers; the span itself runs to the body's end.
        (
            "llama_stage.elapsed_ms".to_string(),
            json!(span.elapsed().as_secs_f64() * 1_000.0),
        ),
        ("llama_stage.http_streamed".to_string(), json!(streamed)),
    ]);
    if !streamed {
        span.end();
        return response;
    }
    response.map(|body| {
        Body::from_stream(SpannedBody {
            inner: body.into_data_stream(),
            _span: span,
        })
    })
}

fn is_event_stream(response: &Response) -> bool {
    response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("text/event-stream"))
}

/// A response body that ends the request span when it is dropped: after the
/// last chunk is sent, or when the client disconnects.
struct SpannedBody {
    inner: BodyDataStream,
    _span: ActiveSpan,
}

impl Stream for SpannedBody {
    type Item = Result<Bytes, axum::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(cx)
    }
}

#[cfg(test)]
mod tests {
    use axum::{Router, middleware, routing::get};
    use serde_json::json;
    use tower::ServiceExt;

    use super::*;
    use crate::telemetry::{TelemetryLevel, capture_stage_config};

    fn router(telemetry: Telemetry) -> Router {
        Router::new()
            .route(
                "/stream",
                get(|| async {
                    Response::builder()
                        .header(header::CONTENT_TYPE, "text/event-stream")
                        .body(Body::from_stream(futures_util::stream::iter([Ok::<
                            _,
                            std::io::Error,
                        >(
                            "data: x\n\n",
                        )])))
                        .expect("stream response")
                }),
            )
            .route("/json", get(|| async { "{}" }))
            .layer(middleware::from_fn_with_state(
                telemetry,
                openai_http_telemetry,
            ))
    }

    fn get_request(path: &str) -> Request<Body> {
        Request::get(path).body(Body::empty()).expect("request")
    }

    #[tokio::test]
    async fn a_streamed_response_keeps_its_span_open_until_the_body_ends() {
        let (telemetry, rx) = Telemetry::captured(capture_stage_config(), TelemetryLevel::Summary);
        let response = router(telemetry)
            .oneshot(get_request("/stream"))
            .await
            .expect("response");
        assert!(
            rx.try_recv().is_err(),
            "headers alone must not end the span"
        );

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        assert_eq!(body, "data: x\n\n");
        let span = rx.try_recv().expect("span ends with the body");
        assert_eq!(span.event, "stage.openai_http_request");
        assert_eq!(span.attributes["llama_stage.http_streamed"], json!(true));
    }

    #[tokio::test]
    async fn a_buffered_response_ends_its_span_at_the_response() {
        let (telemetry, rx) = Telemetry::captured(capture_stage_config(), TelemetryLevel::Summary);
        let _response = router(telemetry)
            .oneshot(get_request("/json"))
            .await
            .expect("response");
        let span = rx.try_recv().expect("span ends with the response");
        assert_eq!(span.attributes["llama_stage.http_streamed"], json!(false));
        assert_eq!(span.attributes["llama_stage.http_status"], json!(200));
    }
}
