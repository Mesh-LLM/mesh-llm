//! Issue #2212: the KV-cache mutation routes must reject a request whose body
//! is not explicitly JSON.
//!
//! The trusted-local gate accepts any loopback origin, so a page served by any
//! other local port (`http://localhost:5173`, a dev server) is already a
//! trusted caller. What stops that page from evicting the operator's caches is
//! the media type: a browser sends a cross-origin request without a CORS
//! preflight only for `text/plain`, `application/x-www-form-urlencoded`, and
//! `multipart/form-data`. Requiring `application/json` therefore forces the
//! preflight that a management listener never answers.

use super::*;

fn kv_cache_request(
    method: &str,
    path: &str,
    content_type: &str,
    origin: &str,
    body: &str,
) -> String {
    format!(
        "{method} {path} HTTP/1.1\r\nHost: localhost:3131\r\nOrigin: {origin}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
}

async fn send_kv_cache_request(request: String) -> String {
    let state = build_test_mesh_api().await;
    let (address, server) = spawn_management_test_server(state).await;
    let response = send_management_request(address, request).await;
    server
        .await
        .expect("management task joins")
        .expect("request");
    response
}

/// A hostile page on another loopback port sends the simple `text/plain` POST a
/// browser will deliver without a preflight. The prune must not execute.
#[tokio::test]
async fn kv_cache_prune_rejects_a_cross_origin_loopback_simple_request() {
    let response = send_kv_cache_request(kv_cache_request(
        "POST",
        "/api/runtime/kv-cache/prune",
        "text/plain",
        "http://localhost:5173",
        r#"{"target_bytes":0}"#,
    ))
    .await;

    assert!(
        response.starts_with("HTTP/1.1 415"),
        "prune accepted a text/plain body instead of requiring JSON: {response}"
    );
}

/// The clear route removes every inactive disk prompt-cache entry with no
/// parameters at all, so an empty `text/plain` body is a complete attack.
#[tokio::test]
async fn kv_cache_clear_rejects_a_cross_origin_loopback_simple_request() {
    let response = send_kv_cache_request(kv_cache_request(
        "DELETE",
        "/api/runtime/kv-cache",
        "text/plain",
        "http://localhost:5173",
        "",
    ))
    .await;

    assert!(
        response.starts_with("HTTP/1.1 415"),
        "clear accepted a text/plain body instead of requiring JSON: {response}"
    );
}

/// The owner-control fan-out reaches every owned endpoint, so it is the
/// highest-blast-radius of the three and follows the same rule.
#[tokio::test]
async fn kv_cache_control_fanout_rejects_a_non_json_content_type() {
    let response = send_kv_cache_request(kv_cache_request(
        "POST",
        "/api/runtime/control/kv-cache",
        "application/x-www-form-urlencoded",
        "http://localhost:5173",
        "operation=clear",
    ))
    .await;

    assert!(
        response.starts_with("HTTP/1.1 415"),
        "control fan-out accepted a form body instead of requiring JSON: {response}"
    );
}

/// A request that carries no `Content-Type` at all cannot be shown to have
/// avoided a preflight, so it is rejected too.
#[tokio::test]
async fn kv_cache_prune_rejects_a_missing_content_type() {
    let response = send_kv_cache_request(
        "POST /api/runtime/kv-cache/prune HTTP/1.1\r\nHost: localhost:3131\r\nContent-Length: 0\r\n\r\n"
            .to_string(),
    )
    .await;

    assert!(
        response.starts_with("HTTP/1.1 415"),
        "prune accepted a body-less request with no media type: {response}"
    );
}

/// A media type that merely starts with the JSON token is not JSON.
#[tokio::test]
async fn kv_cache_prune_rejects_a_lookalike_media_type() {
    let response = send_kv_cache_request(kv_cache_request(
        "POST",
        "/api/runtime/kv-cache/prune",
        "application/jsonp",
        "http://localhost:3131",
        r#"{"target_bytes":0}"#,
    ))
    .await;

    assert!(
        response.starts_with("HTTP/1.1 415"),
        "prune accepted application/jsonp as JSON: {response}"
    );
}

/// The operator CLI sends `application/json` from loopback with no browser
/// origin, and must keep reaching the handler. `409` is the handler's own
/// "disk prompt cache is not active" answer in a test host with no cache.
#[tokio::test]
async fn kv_cache_prune_still_accepts_a_trusted_local_json_request() {
    let response = send_kv_cache_request(kv_cache_request(
        "POST",
        "/api/runtime/kv-cache/prune",
        "application/json",
        "http://localhost:3131",
        r#"{"target_bytes":0}"#,
    ))
    .await;

    assert!(
        !response.starts_with("HTTP/1.1 415") && !response.starts_with("HTTP/1.1 403"),
        "legitimate JSON prune was refused: {response}"
    );
}

/// A `charset` parameter is still exactly the JSON media type.
#[tokio::test]
async fn kv_cache_prune_accepts_json_with_a_charset_parameter() {
    let response = send_kv_cache_request(kv_cache_request(
        "POST",
        "/api/runtime/kv-cache/prune",
        "application/json; charset=utf-8",
        "http://localhost:3131",
        r#"{"target_bytes":0}"#,
    ))
    .await;

    assert!(
        !response.starts_with("HTTP/1.1 415") && !response.starts_with("HTTP/1.1 403"),
        "JSON with a charset parameter was refused: {response}"
    );
}

/// The existing browser-origin protection stays in force: a non-loopback
/// origin is rejected before the media type is even considered.
#[tokio::test]
async fn kv_cache_prune_rejects_an_attacker_origin() {
    let response = send_kv_cache_request(kv_cache_request(
        "POST",
        "/api/runtime/kv-cache/prune",
        "application/json",
        "https://attacker.example",
        r#"{"target_bytes":0}"#,
    ))
    .await;

    assert!(
        response.starts_with("HTTP/1.1 403"),
        "prune accepted a non-loopback origin: {response}"
    );
}
