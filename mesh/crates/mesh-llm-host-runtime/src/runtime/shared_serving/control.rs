//! Local operator control, not exposed on any membership's QUIC endpoint.

use std::io::Write;
use std::sync::Arc;

use anyhow::Result;
use axum::{
    Json, Router,
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    routing::{get, post},
};
use sha2::{Digest, Sha256};

use super::host::SharedServingStatus;
use super::{SharedServingConfig, SharedServingHost};

#[derive(Clone)]
struct ControlState {
    host: Arc<SharedServingHost>,
    token_hash: [u8; 32],
}

fn authorized(headers: &HeaderMap, expected: &[u8; 32]) -> bool {
    let Some(token) = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
    else {
        return false;
    };
    let actual = Sha256::digest(token.as_bytes());
    actual
        .iter()
        .zip(expected)
        .fold(0u8, |mismatch, (a, b)| mismatch | (a ^ b))
        == 0
}

async fn status(
    State(state): State<ControlState>,
    headers: HeaderMap,
) -> Result<Json<SharedServingStatus>, StatusCode> {
    if !authorized(&headers, &state.token_hash) {
        return Err(StatusCode::UNAUTHORIZED);
    }
    Ok(Json(state.host.status().await))
}

async fn action(
    State(state): State<ControlState>,
    Path((name, action)): Path<(String, String)>,
    headers: HeaderMap,
) -> Result<StatusCode, (StatusCode, &'static str)> {
    if !authorized(&headers, &state.token_hash) {
        return Err((StatusCode::UNAUTHORIZED, "operator authentication required"));
    }
    let result = match action.as_str() {
        "pause" => state.host.pause(&name).await,
        "resume" => state.host.resume(&name).await,
        "leave" => state.host.leave(&name).await,
        "reconnect" => state.host.reconnect(&name).await,
        _ => return Err((StatusCode::NOT_FOUND, "unknown membership action")),
    };
    result.map(|()| StatusCode::NO_CONTENT).map_err(|error| {
        tracing::warn!(membership = name, %error, "shared membership operation failed");
        (
            StatusCode::CONFLICT,
            "membership operation failed; inspect status and local diagnostics",
        )
    })
}

pub(super) fn router(host: Arc<SharedServingHost>, token: &str) -> Router {
    Router::new()
        .route("/api/shared-serving", get(status))
        .route(
            "/api/shared-serving/memberships/{name}/{action}",
            post(action),
        )
        .with_state(ControlState {
            host,
            token_hash: Sha256::digest(token.as_bytes()).into(),
        })
}

/// Foreground CLI runtime. It never advertises a management URL to a mesh.
pub async fn run_shared_serving(config: SharedServingConfig, management_port: u16) -> Result<()> {
    let token_path = config.state_dir.join("management.token");
    let host = SharedServingHost::load(config).await?;
    let token = format!(
        "{}{}",
        uuid::Uuid::new_v4().simple(),
        uuid::Uuid::new_v4().simple()
    );
    let mut token_file = super::configuration::private_file(&token_path)?;
    token_file.set_len(0)?;
    token_file.write_all(token.as_bytes())?;
    token_file.sync_all()?;
    let listener =
        tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, management_port)).await?;
    let address = listener.local_addr()?;
    let app = router(host.clone(), &token);
    let control_shutdown = skippy_inference_api::CancellationToken::new();
    let listener = super::local_http::AttachmentListener::new(listener, control_shutdown.clone());
    let mut server = tokio::spawn(async move { axum::serve(listener, app).await });
    let _ = mesh_llm_events::emit_event(mesh_llm_events::OutputEvent::Info {
        message: format!(
            "Shared serving control: http://{address}/api/shared-serving; bearer token file: {}",
            token_path.display()
        ),
        context: None,
    });
    let result = tokio::select! {
        signal = shutdown_signal() => signal,
        () = async { host.connect_all().await; std::future::pending::<()>().await } => unreachable!(),
        stopped = &mut server => match stopped {
            Ok(Ok(())) => Err(anyhow::anyhow!("shared serving management listener stopped")),
            Ok(Err(error)) => Err(error.into()),
            Err(error) => Err(error.into()),
        },
    };
    control_shutdown.cancel();
    server.abort();
    host.shutdown().await?;
    result
}

async fn shutdown_signal() -> Result<()> {
    #[cfg(unix)]
    {
        let mut terminate =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
        tokio::select! {
            signal = tokio::signal::ctrl_c() => signal?,
            _ = terminate.recv() => {},
        }
    }
    #[cfg(not(unix))]
    tokio::signal::ctrl_c().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn control_requires_the_operator_token() {
        let hash = Sha256::digest(b"operator-token").into();
        let mut headers = HeaderMap::new();
        assert!(!authorized(&headers, &hash));
        headers.insert(
            axum::http::header::AUTHORIZATION,
            "Bearer wrong".parse().unwrap(),
        );
        assert!(!authorized(&headers, &hash));
        headers.insert(
            axum::http::header::AUTHORIZATION,
            "Bearer operator-token".parse().unwrap(),
        );
        assert!(authorized(&headers, &hash));
    }
}
