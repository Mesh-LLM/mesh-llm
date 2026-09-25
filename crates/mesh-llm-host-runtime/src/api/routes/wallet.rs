use mesh_llm_payments_types::contract::ops;
use mesh_llm_payments_types::control::ControlCommand;
use tokio::net::TcpStream;

use super::super::{
    MeshApi,
    http::{respond_error, respond_json},
};

/// The API boundary classifies every wallet route as trusted-local before
/// dispatch. Do not mount this handler on the peer inference transport.
pub(super) async fn handle(
    stream: &mut TcpStream,
    state: &MeshApi,
    body: &str,
) -> anyhow::Result<()> {
    let mut body: serde_json::Value = match serde_json::from_str(body) {
        Ok(value) => value,
        Err(_) => return respond_error(stream, 400, "invalid wallet command").await,
    };
    if let Some(expected) = body.as_object_mut().and_then(|v| v.remove("expected_pid"))
        && expected.as_u64() != Some(u64::from(std::process::id()))
    {
        return respond_error(
            stream,
            409,
            "wallet process changed; refresh before retrying",
        )
        .await;
    }
    let node = state.inner.lock().await.node.clone();
    if let Some(expected) = body
        .as_object_mut()
        .and_then(|v| v.remove("expected_directory"))
    {
        let actual = node.config_state.lock().await.payment_directory();
        let matches = expected.as_str().is_some_and(|expected| {
            let expected = std::path::Path::new(expected);
            match (
                expected.parent().and_then(|p| p.canonicalize().ok()),
                actual.parent().and_then(|p| p.canonicalize().ok()),
            ) {
                (Some(a), Some(b)) => a == b && expected.file_name() == actual.file_name(),
                _ => false,
            }
        });
        if !matches {
            return respond_error(
                stream,
                409,
                "wallet data directory does not match this runtime",
            )
            .await;
        }
    }
    let command: ControlCommand = match serde_json::from_value(body) {
        Ok(command) => command,
        Err(_) => return respond_error(stream, 400, "invalid wallet command").await,
    };
    // The engine spawns durable settlement itself, so it continues even if
    // this HTTP connection disappears.
    let result: anyhow::Result<serde_json::Value> =
        crate::network::payments::client::call_node(&node, ops::CONTROL, &command).await;
    match result {
        Ok(value) => respond_json(stream, 200, &value).await,
        Err(error) => respond_error(stream, 400, &error.to_string()).await,
    }
}
