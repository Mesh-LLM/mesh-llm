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
                    .ensure_stage_transport_bridge(receiver.id(), "topology", "run", "stage")
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
                        .ensure_stage_transport_bridge(receiver.id(), "topology", "run", "stage")
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
async fn required_stage_provider_rejects_identity_free_or_nonlocal_stage_endpoints() {
    let node = start_node().await;
    node.install_application_protocols_with_policy(
        vec![Arc::new(Provider(Mutex::new(Behavior::Available)))],
        StageTransportPolicy::RequireProvider,
    )
    .await
    .unwrap();
    let mut peer = crate::inference::skippy::StagePeerDescriptor {
        stage_id: "stage".into(),
        stage_index: 1,
        endpoint: "127.0.0.1:9002".into(),
        node_id: None,
    };
    assert!(
        node.resolve_stage_peer_endpoint("topology", "run", &mut peer)
            .await
            .is_err()
    );
    peer.node_id = Some(node.id());
    peer.endpoint = "192.0.2.1:9002".into();
    assert!(
        node.resolve_stage_peer_endpoint("topology", "run", &mut peer)
            .await
            .is_err()
    );
    peer.endpoint = "127.0.0.1:9002".into();
    node.resolve_stage_peer_endpoint("topology", "run", &mut peer)
        .await
        .unwrap();
    node.close_endpoint().await;
}

#[tokio::test]
async fn required_stage_provider_capability_loss_terminates_an_accepted_channel() {
    let sender = start_node().await;
    let receiver = start_node().await;
    for node in [&sender, &receiver] {
        node.install_application_protocols_with_policy(
            vec![Arc::new(Provider(Mutex::new(Behavior::Available)))],
            StageTransportPolicy::RequireProvider,
        )
        .await
        .unwrap();
    }
    connect(&sender, &receiver).await;
    let (_held_open, stream) = tokio::io::duplex(64);
    let task = {
        let receiver = receiver.clone();
        let peer = sender.id();
        tokio::spawn(async move {
            receiver
                .accept_application_stage_stream(peer, ProtocolStream::new(stream))
                .await
        })
    };
    tokio::task::yield_now().await;
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
    let error = tokio::time::timeout(std::time::Duration::from_secs(1), task)
        .await
        .unwrap()
        .unwrap()
        .unwrap_err();
    assert!(error.to_string().contains("required stage provider policy"));
    sender.close_endpoint().await;
    receiver.close_endpoint().await;
}
