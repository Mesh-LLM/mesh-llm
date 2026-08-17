use super::super::{
    MeshApi,
    http::{respond_error, respond_json},
};
use crate::network::{discovery, nostr};
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;

pub(super) async fn handle(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    let (mode, relays) = {
        let inner = state.inner.lock().await;
        (inner.mesh_discovery_mode, inner.nostr_relays.clone())
    };
    let filter = nostr::MeshFilter::default();
    let json = match mode {
        discovery::MeshDiscoveryMode::Nostr => {
            match nostr::discover(&relays, &filter, None).await {
                Ok(meshes) => serde_json::to_string(&meshes),
                Err(e) => {
                    respond_error(stream, 500, &format!("Discovery failed: {e}")).await?;
                    return Ok(());
                }
            }
        }
        discovery::MeshDiscoveryMode::Mdns => {
            match discovery::discover_lan(&filter, None, std::time::Duration::from_secs(3)).await {
                Ok(meshes) => serde_json::to_string(&meshes),
                Err(e) => {
                    respond_error(stream, 500, &format!("Discovery failed: {e}")).await?;
                    return Ok(());
                }
            }
        }
    };

    match json {
        Ok(json) => {
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                json.len(),
                json
            );
            stream.write_all(resp.as_bytes()).await?;
        }
        Err(_) => respond_error(stream, 500, "Failed to serialize").await?,
    }
    Ok(())
}

pub(super) async fn handle_lan_details(
    stream: &mut TcpStream,
    state: &MeshApi,
    body: &str,
) -> anyhow::Result<()> {
    let request = match serde_json::from_str::<discovery::LanDetailsProofRequest>(body) {
        Ok(request) => request,
        Err(err) => {
            respond_error(stream, 400, &format!("Invalid JSON body: {err}")).await?;
            return Ok(());
        }
    };
    let (mode, node, mesh_name, mesh_region, mesh_max_clients) = {
        let inner = state.inner.lock().await;
        (
            inner.mesh_discovery_mode,
            inner.node.clone(),
            inner.mesh_name.clone(),
            inner.mesh_region.clone(),
            inner.mesh_max_clients,
        )
    };
    if mode != discovery::MeshDiscoveryMode::Mdns {
        respond_error(
            stream,
            404,
            "LAN discovery details are only available in mDNS discovery mode",
        )
        .await?;
        return Ok(());
    }

    let invite_token = node.invite_token().await;
    if !discovery::verify_lan_details_token_proof(
        &invite_token,
        &request.token_fingerprint,
        &request.challenge,
        &request.proof,
        current_unix_secs(),
    ) {
        respond_error(stream, 403, "Invalid LAN discovery proof").await?;
        return Ok(());
    }

    let listing =
        discovery::build_local_mesh_listing(&node, mesh_name, mesh_region, mesh_max_clients).await;
    let response = discovery::LanDetailsResponse::from_local_listing(
        listing,
        request.token_fingerprint,
        request.challenge,
        Some(crate::VERSION),
    );
    respond_json(stream, 200, &response).await?;
    Ok(())
}

fn current_unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
