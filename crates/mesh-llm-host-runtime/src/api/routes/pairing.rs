use super::super::{
    MeshApi,
    http::{respond_error, respond_json},
};
use crate::mesh::PairingDecision;
use serde::Deserialize;
use tokio::net::TcpStream;

#[derive(Deserialize)]
struct ConnectRequest {
    offer: String,
}

pub(super) async fn handle(
    stream: &mut TcpStream,
    state: &MeshApi,
    method: &str,
    path: &str,
    body: &str,
) -> anyhow::Result<()> {
    let node = state.node().await;
    match (method, path) {
        ("GET", "/api/pairing") | ("GET", "/api/pairing/sessions") => {
            respond_json(
                stream,
                200,
                &serde_json::json!({ "sessions": node.pairing_sessions().await }),
            )
            .await
        }
        ("POST", "/api/pairing/offers") => match node.create_pairing_offer().await {
            Ok(offer) => respond_json(stream, 201, &offer).await,
            Err(error) => respond_error(stream, 503, &error.to_string()).await,
        },
        ("POST", "/api/pairing/connect") => connect(stream, &node, body).await,
        ("POST", path) => decide(stream, &node, path).await,
        _ => respond_error(stream, 405, "Method not allowed").await,
    }
}

async fn connect(
    stream: &mut TcpStream,
    node: &crate::mesh::Node,
    body: &str,
) -> anyhow::Result<()> {
    let request: ConnectRequest = match serde_json::from_str(body) {
        Ok(request) => request,
        Err(_) => return respond_error(stream, 400, "Invalid JSON body").await,
    };
    match node.start_pairing(&request.offer).await {
        Ok(session) => respond_json(stream, 202, &session).await,
        Err(error) => respond_error(stream, 400, &error.to_string()).await,
    }
}

async fn decide(
    stream: &mut TcpStream,
    node: &crate::mesh::Node,
    path: &str,
) -> anyhow::Result<()> {
    let Some((session_id, decision)) = parse_decision_path(path) else {
        return respond_error(stream, 404, "Pairing route not found").await;
    };
    match node.decide_pairing(session_id, decision).await {
        Ok(session) => respond_json(stream, 200, &session).await,
        Err(error) if error.to_string().contains("not found") => {
            respond_error(stream, 404, &error.to_string()).await
        }
        Err(error) => respond_error(stream, 409, &error.to_string()).await,
    }
}

fn parse_decision_path(path: &str) -> Option<(&str, PairingDecision)> {
    let suffix = path.strip_prefix("/api/pairing/sessions/")?;
    let (session_id, action) = suffix.split_once('/')?;
    if session_id.is_empty() || action.contains('/') {
        return None;
    }
    let decision = match action {
        "approve" => PairingDecision::Approve,
        "reject" => PairingDecision::Reject,
        "cancel" => PairingDecision::Cancel,
        _ => return None,
    };
    Some((session_id, decision))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decision_paths_are_strict() {
        assert_eq!(
            parse_decision_path("/api/pairing/sessions/abc/approve"),
            Some(("abc", PairingDecision::Approve))
        );
        assert!(parse_decision_path("/api/pairing/sessions/abc/approve/more").is_none());
        assert!(parse_decision_path("/api/pairing/sessions//approve").is_none());
        assert!(parse_decision_path("/api/pairing/sessions/abc/trust").is_none());
    }
}
