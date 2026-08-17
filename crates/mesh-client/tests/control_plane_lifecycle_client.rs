use base64::Engine;
use iroh::{Endpoint, EndpointAddr, SecretKey};
use mesh_client::proto::node::{
    OwnerControlDrainModelResponse, OwnerControlEnsureModelResponse, OwnerControlEnvelope,
    OwnerControlLoadModelResponse, OwnerControlRequest, OwnerControlResponse,
    OwnerControlUnloadModelResponse,
};
use mesh_client::protocol::{
    ALPN_CONTROL_V1, NODE_PROTOCOL_GENERATION, decode_owner_control_envelope, read_len_prefixed,
    write_len_prefixed,
};
use mesh_client::{
    ClientBuilder, ControlPlaneBootstrapOptions, ControlPlaneConnection, InviteToken,
    OwnerControlClient, OwnerKeypair,
};
use prost::Message;
use std::str::FromStr;
use std::sync::Arc;
use tokio::sync::{Mutex, oneshot};

fn owner_keypair() -> OwnerKeypair {
    OwnerKeypair::from_bytes(&[0x31; 32], &[0x32; 32]).expect("test owner keypair must be valid")
}

fn endpoint_token(addr: &EndpointAddr) -> String {
    base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(serde_json::to_vec(addr).expect("endpoint addr should serialize"))
}

async fn read_envelope(recv: &mut iroh::endpoint::RecvStream) -> OwnerControlEnvelope {
    let bytes = read_len_prefixed(recv)
        .await
        .expect("frame should be len-prefixed");
    decode_owner_control_envelope(&bytes).expect("owner-control envelope should decode")
}

async fn write_envelope(send: &mut iroh::endpoint::SendStream, envelope: OwnerControlEnvelope) {
    write_len_prefixed(send, &envelope.encode_to_vec())
        .await
        .expect("owner-control envelope should write");
}

fn success_response(request: &OwnerControlRequest) -> OwnerControlResponse {
    let mut response = OwnerControlResponse {
        request_id: request.request_id,
        ..Default::default()
    };
    if let Some(command) = request.load_model.as_ref() {
        response.load_model = Some(OwnerControlLoadModelResponse {
            intent_id: "load-intent".to_string(),
            accepted_state: "present".to_string(),
            target: command.model.clone(),
        });
    } else if let Some(command) = request.unload_model.as_ref() {
        response.unload_model = Some(OwnerControlUnloadModelResponse {
            intent_id: "unload-intent".to_string(),
            accepted_state: "absent".to_string(),
            target: command.model.clone(),
        });
    } else if let Some(command) = request.ensure_model.as_ref() {
        response.ensure_model = Some(OwnerControlEnsureModelResponse {
            intent_id: "ensure-intent".to_string(),
            accepted_state: "present".to_string(),
            target: command.model.clone(),
        });
    } else if let Some(command) = request.drain_model.as_ref() {
        response.drain_model = Some(OwnerControlDrainModelResponse {
            intent_id: "drain-intent".to_string(),
            accepted_state: "draining".to_string(),
            target: command.model.clone(),
        });
    } else {
        panic!("expected one lifecycle command");
    }
    response
}

async fn spawn_lifecycle_server() -> (
    Endpoint,
    String,
    Arc<Mutex<Vec<OwnerControlRequest>>>,
    oneshot::Sender<()>,
) {
    let endpoint = Endpoint::builder(iroh::endpoint::presets::Minimal)
        .secret_key(SecretKey::generate())
        .alpns(vec![ALPN_CONTROL_V1.to_vec()])
        .relay_mode(iroh::endpoint::RelayMode::Disabled)
        .bind_addr(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))
        .expect("test bind address should be valid")
        .bind()
        .await
        .expect("test endpoint should bind");
    let token = endpoint_token(&endpoint.addr());
    let requests = Arc::new(Mutex::new(Vec::new()));
    let server_endpoint = endpoint.clone();
    let received = Arc::clone(&requests);
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    tokio::spawn(async move {
        let incoming = server_endpoint
            .accept()
            .await
            .expect("server should accept connection");
        let connection = incoming.await.expect("server connection should complete");
        for _ in 0..4 {
            let (mut send, mut recv) = connection.accept_bi().await.expect("stream should open");
            let _handshake = read_envelope(&mut recv).await;
            let request = read_envelope(&mut recv)
                .await
                .request
                .expect("second envelope should contain a request");
            let response = success_response(&request);
            received.lock().await.push(request);
            write_envelope(
                &mut send,
                OwnerControlEnvelope {
                    r#gen: NODE_PROTOCOL_GENERATION,
                    response: Some(response),
                    ..Default::default()
                },
            )
            .await;
            send.finish().expect("response stream should finish");
        }
        let _ = shutdown_rx.await;
    });
    (endpoint, token, requests, shutdown_tx)
}

async fn connect(token: String) -> OwnerControlClient {
    let client = ClientBuilder::new(
        owner_keypair(),
        InviteToken::from_str("mesh-test:lifecycle").expect("test invite should parse"),
    )
    .build()
    .expect("mesh client should build");
    match client
        .connect_control_plane(ControlPlaneBootstrapOptions::new().with_control_endpoint(token))
        .await
        .expect("control session should connect")
    {
        ControlPlaneConnection::OwnerControl(client) => *client,
    }
}

#[tokio::test]
async fn lifecycle_commands_preserve_payloads_request_ids_and_responses() {
    let (server, token, requests, shutdown) = spawn_lifecycle_server().await;
    let control = connect(token).await;

    let load = control
        .load_model("repo/load:Q4_K_M".to_string(), Some("fast".to_string()))
        .await
        .expect("load should succeed");
    let unload = control
        .unload_model(String::new(), Some("instance-unload".to_string()))
        .await
        .expect("unload should succeed");
    let ensure = control
        .ensure_model("repo/ensure:Q6_K".to_string(), Some("balanced".to_string()))
        .await
        .expect("ensure should succeed");
    let drain = control
        .drain_model(String::new(), Some("instance-drain".to_string()))
        .await
        .expect("drain should succeed");

    assert_eq!(load.intent_id, "load-intent");
    assert_eq!(unload.intent_id, "unload-intent");
    assert_eq!(ensure.intent_id, "ensure-intent");
    assert_eq!(drain.intent_id, "drain-intent");

    let requests = requests.lock().await;
    assert_eq!(requests.len(), 4);
    assert_eq!(
        requests
            .iter()
            .map(|request| request.request_id)
            .collect::<Vec<_>>(),
        vec![1, 2, 3, 4]
    );

    let load = requests[0].load_model.as_ref().expect("load payload");
    assert_eq!(
        load.model.as_ref().unwrap().canonical_model_ref,
        "repo/load:Q4_K_M"
    );
    assert_eq!(load.profile.as_deref(), Some("fast"));
    let unload = requests[1].unload_model.as_ref().expect("unload payload");
    assert_eq!(unload.model.as_ref().unwrap().canonical_model_ref, "");
    assert_eq!(
        unload.model.as_ref().unwrap().instance_id.as_deref(),
        Some("instance-unload")
    );
    let ensure = requests[2].ensure_model.as_ref().expect("ensure payload");
    assert_eq!(
        ensure.model.as_ref().unwrap().canonical_model_ref,
        "repo/ensure:Q6_K"
    );
    assert_eq!(ensure.profile.as_deref(), Some("balanced"));
    let drain = requests[3].drain_model.as_ref().expect("drain payload");
    assert_eq!(drain.model.as_ref().unwrap().canonical_model_ref, "");
    assert_eq!(
        drain.model.as_ref().unwrap().instance_id.as_deref(),
        Some("instance-drain")
    );
    assert_eq!(drain.drain_timeout_secs, None);

    control.close().await;
    let _ = shutdown.send(());
    server.close().await;
}
