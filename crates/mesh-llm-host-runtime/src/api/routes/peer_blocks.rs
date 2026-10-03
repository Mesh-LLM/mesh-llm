//! `/api/peer-blocks`: the operator's local "stop routing to this peer".
//!
//! `GET` lists the active blocks. `POST` blocks a peer;
//! `POST /api/peer-blocks/unblock` undoes a block, whoever requested it. Each
//! change takes effect in the router first and is then published once on
//! `routing.choice.v1` to the local plugins that declare it.
//!
//! Both carry `not_saved` (why) when `peer_blocks.json` could not be loaded
//! at start: changes still apply for this run but are not saved.
//!
//! Loopback-only (`api::access::requires_trusted_local_access`).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use tokio::net::TcpStream;

use super::super::{
    MeshApi,
    http::{respond_error, respond_json},
};
use crate::network::peer_blocks::{
    self, ActiveBlock, BlockError, BlockLength, Requester, RoutingChoice, now_ms, parse_peer,
};

const ROUTE: &str = "/api/peer-blocks";
const UNBLOCK_ROUTE: &str = "/api/peer-blocks/unblock";

pub(super) fn is_route(path: &str) -> bool {
    path == ROUTE || path == UNBLOCK_ROUTE
}

#[derive(Debug, Deserialize)]
struct BlockRequest {
    peer: String,
    length: BlockLength,
    #[serde(default)]
    reason: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct UnblockRequest {
    peer: String,
    #[serde(default)]
    reason: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct ListResponse<'a> {
    blocks: BTreeMap<String, ActiveBlock>,
    /// Why changes are not being saved this run, when `peer_blocks.json`
    /// could not be loaded.
    #[serde(skip_serializing_if = "Option::is_none")]
    not_saved: Option<&'a str>,
}

#[derive(Debug, Serialize)]
struct ChangeResponse<'a> {
    choice: RoutingChoice,
    #[serde(skip_serializing_if = "Option::is_none")]
    not_saved: Option<&'a str>,
}

pub(super) async fn handle(
    stream: &mut TcpStream,
    state: &MeshApi,
    method: &str,
    path: &str,
    body: &str,
) -> anyhow::Result<()> {
    let node = state.inner.lock().await.node.clone();
    let result = match (method, path) {
        ("GET", ROUTE) => {
            return respond_json(
                stream,
                200,
                &ListResponse {
                    blocks: node.peer_blocks.snapshot(now_ms()),
                    not_saved: node.peer_blocks.not_saved(),
                },
            )
            .await;
        }
        ("POST", ROUTE) => {
            let request: BlockRequest = match serde_json::from_str(body) {
                Ok(request) => request,
                Err(error) => return respond_error(stream, 400, &error.to_string()).await,
            };
            let Some(peer) = parse_peer(&request.peer) else {
                return respond_error(stream, 400, "peer must be a 64-hex endpoint id").await;
            };
            let blocks = node.peer_blocks.clone();
            peer_blocks::offload(move || {
                blocks.block(
                    &peer,
                    request.length,
                    Requester::Operator,
                    request.reason,
                    now_ms(),
                )
            })
            .await
        }
        ("POST", UNBLOCK_ROUTE) => {
            let request: UnblockRequest = match serde_json::from_str(body) {
                Ok(request) => request,
                Err(error) => return respond_error(stream, 400, &error.to_string()).await,
            };
            let Some(peer) = parse_peer(&request.peer) else {
                return respond_error(stream, 400, "peer must be a 64-hex endpoint id").await;
            };
            let blocks = node.peer_blocks.clone();
            peer_blocks::offload(move || {
                blocks.unblock(&peer, Requester::Operator, request.reason, now_ms())
            })
            .await
        }
        _ => return respond_error(stream, 405, "method not allowed").await,
    };
    match result {
        Ok(choice) => {
            peer_blocks::publish(&node, &choice).await;
            respond_json(
                stream,
                200,
                &ChangeResponse {
                    choice,
                    not_saved: node.peer_blocks.not_saved(),
                },
            )
            .await
        }
        Err(error) => respond_error(stream, error_status(&error), &error.to_string()).await,
    }
}

fn error_status(error: &BlockError) -> u16 {
    match error {
        BlockError::NotBlocked => 409,
        BlockError::NotRequester => 403,
        BlockError::ReasonTooLarge => 400,
        BlockError::Save(_) => 500,
    }
}
