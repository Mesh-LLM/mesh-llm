use super::application_protocols::start_node;
use crate::mesh::Node;
use mesh_llm_transport::protocol::{
    IncomingProtocolStream, MeshProtocol, PeerId, ProtocolContext, ProtocolDescriptor,
    ProtocolStream, REQUIRED_STAGE_PROVIDER_FEATURE, StageTransportPolicy, StageTransportProvider,
    async_trait,
};
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[derive(Clone, Copy, Debug)]
enum Behavior {
    Available,
    Unavailable,
    Decline,
    Failure,
}

#[derive(Debug)]
struct Provider(Mutex<Behavior>);

#[async_trait]
impl StageTransportProvider for Provider {
    fn stage_available(&self, _: PeerId) -> bool {
        !matches!(*self.0.lock().unwrap(), Behavior::Unavailable)
    }
    async fn open_stage(
        &self,
        context: ProtocolContext,
        peer: PeerId,
    ) -> anyhow::Result<Option<ProtocolStream>> {
        let behavior = *self.0.lock().unwrap();
        match behavior {
            Behavior::Available => Ok(Some(context.open(peer).await?)),
            Behavior::Decline | Behavior::Unavailable => Ok(None),
            Behavior::Failure => anyhow::bail!("fixture setup failed"),
        }
    }
}

#[async_trait]
impl MeshProtocol for Provider {
    fn descriptor(&self) -> ProtocolDescriptor {
        ProtocolDescriptor::new("required-stage-test", 1)
    }
    fn stage_transport(&self) -> Option<&dyn StageTransportProvider> {
        Some(self)
    }
    async fn accept(
        &self,
        context: ProtocolContext,
        incoming: IncomingProtocolStream,
    ) -> anyhow::Result<()> {
        context.accept_stage(incoming.peer, incoming.stream).await
    }
}

#[derive(Debug)]
struct ControlEcho;
#[async_trait]
impl MeshProtocol for ControlEcho {
    fn descriptor(&self) -> ProtocolDescriptor {
        ProtocolDescriptor::new("control-echo", 1)
    }
    async fn accept(
        &self,
        _: ProtocolContext,
        mut incoming: IncomingProtocolStream,
    ) -> anyhow::Result<()> {
        let value = incoming.stream.read_u32().await?;
        incoming.stream.write_u32(value + 1).await?;
        incoming.stream.shutdown().await?;
        Ok(())
    }
}

async fn connect(sender: &Node, receiver: &Node) {
    sender.start_accepting();
    receiver.start_accepting();
    sender
        .connect_to_peer(receiver.endpoint_addr_for_advertisement())
        .await
        .unwrap();
    sender.refresh_application_peers().await;
    receiver.refresh_application_peers().await;
}

#[tokio::test]
async fn required_stage_provider_rejects_fallback_peers_and_unavailable_or_declined_setup() {
    for policy in [
        StageTransportPolicy::PreferProvider,
        StageTransportPolicy::RequireProvider,
    ] {
        let sender = start_node().await;
        let receiver = start_node().await;
        let provider = Arc::new(Provider(Mutex::new(Behavior::Available)));
        sender
            .install_application_protocols_with_policy(
                vec![provider.clone()],
                StageTransportPolicy::RequireProvider,
            )
            .await
            .unwrap();
        receiver
            .install_application_protocols_with_policy(
                vec![Arc::new(Provider(Mutex::new(Behavior::Available)))],
                policy,
            )
            .await
            .unwrap();
        connect(&sender, &receiver).await;
        let capable = policy == StageTransportPolicy::RequireProvider;
        assert_eq!(
            sender
                .stage_required_peer_allowed(receiver.id(), true)
                .await,
            capable
        );
        if !capable {
            assert!(
                sender
                    .open_application_stage_stream(receiver.id())
                    .await
                    .is_err()
            );
            assert!(
                sender
                    .verify_required_stage_provider(receiver.id())
                    .await
                    .is_err()
            );
        } else {
            assert!(
                sender.application_protocols.descriptors()[0]
                    .features
                    .iter()
                    .any(|feature| feature == REQUIRED_STAGE_PROVIDER_FEATURE)
            );
            sender
                .verify_required_stage_provider(receiver.id())
                .await
                .unwrap();
            for behavior in [Behavior::Unavailable, Behavior::Decline, Behavior::Failure] {
                *provider.0.lock().unwrap() = behavior;
                assert_eq!(
                    sender
                        .stage_required_peer_allowed(receiver.id(), true)
                        .await,
                    !matches!(behavior, Behavior::Unavailable)
                );
                assert!(
                    sender
                        .verify_required_stage_provider(receiver.id())
                        .await
                        .is_err()
                );
            }
        }
        assert!(
            sender
                .open_stage_transport_stream(receiver.id(), "topology", "run", "stage")
                .await
                .is_err()
        );
        sender.close_endpoint().await;
        receiver.close_endpoint().await;
    }
}

#[tokio::test]
async fn required_stage_provider_blocks_direct_activation_and_other_protocol_contexts_but_allows_control()
 {
    let sender = start_node().await;
    let receiver = start_node().await;
    for node in [&sender, &receiver] {
        node.install_application_protocols_with_policy(
            vec![
                Arc::new(Provider(Mutex::new(Behavior::Available))),
                Arc::new(ControlEcho),
            ],
            StageTransportPolicy::RequireProvider,
        )
        .await
        .unwrap();
    }
    receiver
        .set_stage_control_handle(crate::inference::skippy::spawn_stage_control_loop(
            Default::default(),
        ))
        .await;
    connect(&sender, &receiver).await;
    let context = sender.application_protocol_context(ControlEcho.descriptor());
    let (stream, _other) = tokio::io::duplex(64);
    assert!(
        context
            .accept_stage(receiver.id(), ProtocolStream::new(stream))
            .await
            .unwrap_err()
            .to_string()
            .contains("registered provider")
    );
    let mut control = context.open(receiver.id()).await.unwrap();
    control.write_u32(41).await.unwrap();
    assert_eq!(control.read_u32().await.unwrap(), 42);
    let status = sender
        .send_stage_control(
            receiver.id(),
            crate::inference::skippy::StageControlRequest::Status(
                crate::inference::skippy::StageStatusFilter {
                    topology_id: None,
                    run_id: None,
                    stage_id: None,
                },
            ),
        )
        .await
        .unwrap();
    assert!(
        matches!(status, crate::inference::skippy::StageControlResponse::Status(statuses) if statuses.is_empty())
    );

    let connection = sender
        .stage_connection_to_peer(receiver.id())
        .await
        .unwrap();
    let (mut send, mut recv) = connection.open_bi().await.unwrap();
    let _ = send
        .write_all(&[skippy_protocol::STAGE_STREAM_TRANSPORT])
        .await;
    let mut byte = [0];
    let received = tokio::time::timeout(std::time::Duration::from_secs(2), recv.read(&mut byte))
        .await
        .unwrap();
    assert!(matches!(received, Ok(None) | Err(_)));
    sender.close_endpoint().await;
    receiver.close_endpoint().await;
}

#[tokio::test]
async fn direct_stage_connections_require_identity_and_never_dial_raw_endpoints() {
    let node = start_node().await;
    let mut load = super::stage_load_request();
    load.downstream = None;
    load.upstream = Some(crate::inference::skippy::StagePeerDescriptor {
        stage_id: "stage-0".into(),
        stage_index: 0,
        endpoint: "192.0.2.1:9002".into(),
        node_id: None,
    });
    assert!(node.direct_stage_transport(&load).await.is_err());
    load.upstream.as_mut().unwrap().node_id = Some(node.id());
    let transport = node.direct_stage_transport(&load).await.unwrap();
    tokio::task::spawn_blocking(move || {
        assert!(
            transport
                .connect(
                    "192.0.2.1:9002",
                    std::time::Instant::now() + std::time::Duration::from_secs(1),
                    None
                )
                .is_err()
        );
    })
    .await
    .unwrap();
    node.close_endpoint().await;
}

#[tokio::test]
async fn handed_off_stage_streams_are_cancelled_on_policy_or_admission_loss() {
    for require_provider in [true, false] {
        let sender = start_node().await;
        let receiver = start_node().await;
        for node in [&sender, &receiver] {
            node.install_application_protocols_with_policy(
                if require_provider {
                    vec![Arc::new(Provider(Mutex::new(Behavior::Available)))]
                } else {
                    vec![]
                },
                if require_provider {
                    StageTransportPolicy::RequireProvider
                } else {
                    StageTransportPolicy::PreferProvider
                },
            )
            .await
            .unwrap();
        }
        connect(&sender, &receiver).await;
        let mut load = super::stage_load_request();
        load.topology_id = "topology".into();
        load.run_id = "run".into();
        load.stage_id = "destination".into();
        load.upstream = None;
        load.downstream = None;
        load.topology_stages = [("source", sender.id()), ("destination", receiver.id())]
            .into_iter()
            .enumerate()
            .map(
                |(index, (stage, node))| crate::inference::skippy::StageTopologyStageDescriptor {
                    stage_id: stage.into(),
                    stage_index: index as u32,
                    node_id: node,
                    layer_start: index as u32,
                    layer_end: index as u32 + 1,
                    bind_addr: String::new(),
                },
            )
            .collect();
        receiver.record_stage_load_topology(&load).await;
        let transport = receiver.direct_stage_transport(&load).await.unwrap();
        transport.ready();
        let (mut held_open, stream) = tokio::io::duplex(4096);
        let task = {
            let receiver = receiver.clone();
            let peer = sender.id();
            tokio::spawn(async move {
                receiver
                    .accept_application_stage_stream(peer, ProtocolStream::new(stream))
                    .await
            })
        };
        use prost::Message;
        let header = skippy_protocol::proto::stage::StageTransportOpen {
            r#gen: skippy_protocol::STAGE_PROTOCOL_GENERATION,
            requester_id: sender.id().as_bytes().to_vec(),
            topology_id: "topology".into(),
            run_id: "run".into(),
            stage_id: "destination".into(),
        }
        .encode_to_vec();
        held_open.write_u32_le(header.len() as u32).await.unwrap();
        held_open.write_all(&header).await.unwrap();
        task.await.unwrap().unwrap();
        let mut accepted = transport.try_accept().unwrap();
        let reader = tokio::task::spawn_blocking(move || {
            use std::io::Read;
            accepted.read(&mut [0]).unwrap_err().kind()
        });
        if require_provider {
            for protocol in &mut receiver
                .state
                .lock()
                .await
                .peers
                .get_mut(&sender.id())
                .unwrap()
                .application_protocols
            {
                protocol
                    .features
                    .retain(|feature| feature != REQUIRED_STAGE_PROVIDER_FEATURE);
            }
            receiver.refresh_application_peers().await;
        } else {
            receiver.state.lock().await.peers.remove(&sender.id());
            receiver.peer_change_tx.send_replace(0);
        }
        let error = tokio::time::timeout(std::time::Duration::from_secs(1), reader)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(error, std::io::ErrorKind::ConnectionAborted);
        sender.close_endpoint().await;
        receiver.close_endpoint().await;
    }
}

#[tokio::test]
async fn direct_stage_ingress_is_bounded_and_teardown_interrupts_execution() {
    let node = start_node().await;
    let mut load = super::stage_load_request();
    load.upstream = None;
    load.downstream = None;
    let transport = node.direct_stage_transport(&load).await.unwrap();
    let open = skippy_protocol::proto::stage::StageTransportOpen {
        r#gen: skippy_protocol::STAGE_PROTOCOL_GENERATION,
        requester_id: node.id().as_bytes().to_vec(),
        topology_id: load.topology_id.clone(),
        run_id: load.run_id.clone(),
        stage_id: load.stage_id.clone(),
    };
    let (_other, stream) = tokio::io::duplex(64);
    assert!(
        node.deliver_direct_stage(node.id(), &open, ProtocolStream::new(stream))
            .await
            .is_err()
    );
    transport.ready();
    let mut peers = Vec::new();
    for _ in 0..32 {
        let (other, stream) = tokio::io::duplex(64);
        peers.push(other);
        node.deliver_direct_stage(node.id(), &open, ProtocolStream::new(stream))
            .await
            .unwrap();
    }
    let (_other, overflow) = tokio::io::duplex(64);
    assert!(
        node.deliver_direct_stage(node.id(), &open, ProtocolStream::new(overflow))
            .await
            .is_err()
    );
    let mut stream = transport.try_accept().unwrap();
    let reader = tokio::task::spawn_blocking(move || {
        use std::io::Read;
        stream.read(&mut [0]).unwrap_err().kind()
    });
    node.stop_direct_stage_connections(&load.topology_id, &load.run_id, &load.stage_id)
        .await;
    assert_eq!(
        tokio::time::timeout(std::time::Duration::from_secs(1), reader)
            .await
            .unwrap()
            .unwrap(),
        std::io::ErrorKind::ConnectionAborted
    );
    assert!(transport.try_accept().is_err());
    let (_other, stale) = tokio::io::duplex(64);
    assert!(
        node.deliver_direct_stage(node.id(), &open, ProtocolStream::new(stale))
            .await
            .is_err()
    );
    node.close_endpoint().await;
}
