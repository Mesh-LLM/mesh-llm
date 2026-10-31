//! Retirement response for the former unauthenticated token endpoint.

use tokio::net::TcpStream;

pub(super) async fn reject_token_disclosure(stream: &mut TcpStream) -> anyhow::Result<()> {
    super::respond_json(
        stream,
        403,
        &serde_json::json!({"error": "owner-control credentials must be obtained from the local owner-only token file"}),
    )
    .await
}
