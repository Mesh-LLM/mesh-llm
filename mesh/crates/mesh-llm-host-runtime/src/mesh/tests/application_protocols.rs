use super::*;
use mesh_llm_transport::protocol::{
    IncomingProtocolStream, MeshProtocol, ProtocolContext, ProtocolDescriptor, async_trait,
    validate_protocols,
};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[derive(Debug)]
struct Echo {
    name: &'static str,
    major: u32,
    contexts: tokio::sync::mpsc::UnboundedSender<ProtocolContext>,
    received: tokio::sync::mpsc::UnboundedSender<iroh::EndpointId>,
}

#[async_trait]
impl MeshProtocol for Echo {
    fn descriptor(&self) -> ProtocolDescriptor {
        ProtocolDescriptor::new(self.name, self.major)
    }

    async fn run(&self, context: ProtocolContext) -> anyhow::Result<()> {
        self.contexts.send(context.clone())?;
        context.cancelled().await;
        Ok(())
    }

    async fn accept(
        &self,
        _context: ProtocolContext,
        mut incoming: IncomingProtocolStream,
    ) -> anyhow::Result<()> {
        self.received.send(incoming.peer)?;
        let value = incoming.stream.read_u32().await?;
        incoming.stream.write_u32(value + 1).await?;
        incoming.stream.shutdown().await?;
        Ok(())
    }
}

fn echo(name: &'static str, major: u32) -> Arc<dyn MeshProtocol> {
    Arc::new(Echo {
        name,
        major,
        contexts: tokio::sync::mpsc::unbounded_channel().0,
        received: tokio::sync::mpsc::unbounded_channel().0,
    })
}

#[test]
fn application_protocol_registration_rejects_invalid_or_reserved_descriptors() {
    for handlers in [
        vec![echo("echo", 1), echo("echo", 1)],
        vec![echo("skippy-stage", 1)],
        vec![echo("Echo", 1)],
        vec![echo("echo", 0)],
        vec![echo("echo", 1); 17],
    ] {
        assert!(validate_protocols(&handlers).is_err());
    }
    assert!(validate_protocols(&[echo("echo", 1), echo("echo", 2)]).is_ok());
}

pub(super) async fn start_node() -> Node {
    let (node, _channels) = Node::start(
        super::super::NodeRole::Client,
        RelayConfig {
            urls: &Vec::new(),
            auths: &std::collections::HashMap::new(),
            policy: RelayPolicy::Disabled,
        },
        QuicBindSelection {
            ip: Some(std::net::Ipv4Addr::LOCALHOST.into()),
            port: Some(0),
        },
        Some(0.0),
        false,
        None,
        None,
        crate::MeshRequirements::unrestricted(),
    )
    .await
    .expect("start application protocol node");
    node
}

#[tokio::test]
async fn application_protocols_exchange_on_admitted_connections_and_stop_with_node() {
    let sender = start_node().await;
    let receiver = start_node().await;
    let (contexts, mut context_rx) = tokio::sync::mpsc::unbounded_channel();
    let (received, mut received_rx) = tokio::sync::mpsc::unbounded_channel();
    let handler = Arc::new(Echo {
        name: "echo",
        major: 1,
        contexts,
        received,
    });
    sender
        .install_application_protocols(vec![handler.clone()])
        .await
        .unwrap();
    receiver
        .install_application_protocols(vec![handler])
        .await
        .unwrap();
    let first = context_rx.recv().await.unwrap();
    let second = context_rx.recv().await.unwrap();
    let context = if first.local_peer_id() == sender.id() {
        first
    } else {
        second
    };
    assert!(context.open(receiver.id()).await.is_err());
    sender.start_accepting();
    receiver.start_accepting();
    sender
        .connect_to_peer(receiver.endpoint_addr_for_advertisement())
        .await
        .unwrap();
    sender.refresh_application_peers().await;
    let peers = context.watch_peers();
    assert!(
        peers
            .borrow()
            .iter()
            .any(|peer| peer.id == receiver.id() && peer.supports(context.descriptor()))
    );
    let mut stream = context.open(receiver.id()).await.unwrap();
    stream.write_u32(41).await.unwrap();
    assert_eq!(
        tokio::time::timeout(std::time::Duration::from_secs(5), stream.read_u32())
            .await
            .unwrap()
            .unwrap(),
        42
    );
    assert_eq!(received_rx.recv().await.unwrap(), sender.id());

    sender
        .state
        .lock()
        .await
        .peers
        .get_mut(&receiver.id())
        .unwrap()
        .application_protocols
        .clear();
    assert!(context.open(receiver.id()).await.is_err());
    sender.close_endpoint().await;
    receiver.close_endpoint().await;
    assert!(peers.borrow().is_empty());
    assert!(context.open(receiver.id()).await.is_err());
    tokio::time::timeout(std::time::Duration::from_secs(1), context.cancelled())
        .await
        .unwrap();
}

#[derive(Debug)]
struct StageBridge(
    &'static str,
    std::sync::Mutex<Option<mesh_llm_transport::protocol::StageLinkObservation>>,
);
impl StageBridge {
    fn new(name: &'static str) -> Self {
        Self(name, std::sync::Mutex::new(None))
    }
}
#[async_trait]
impl mesh_llm_transport::protocol::StageTransportProvider for StageBridge {
    fn stage_link(
        &self,
        _: EndpointId,
    ) -> Option<mesh_llm_transport::protocol::StageLinkObservation> {
        *self.1.lock().unwrap()
    }
    async fn open_stage(
        &self,
        context: ProtocolContext,
        peer: EndpointId,
    ) -> anyhow::Result<Option<mesh_llm_transport::protocol::ProtocolStream>> {
        Ok(Some(context.open(peer).await?))
    }
}
#[async_trait]
impl MeshProtocol for StageBridge {
    fn descriptor(&self) -> ProtocolDescriptor {
        ProtocolDescriptor::new(self.0, 1)
    }
    fn stage_transport(&self) -> Option<&dyn mesh_llm_transport::protocol::StageTransportProvider> {
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

#[tokio::test]
async fn application_stage_provider_routes_only_authorized_runs_and_preserves_half_close()
-> anyhow::Result<()> {
    for policy in [
        mesh_llm_transport::protocol::StageTransportPolicy::PreferProvider,
        mesh_llm_transport::protocol::StageTransportPolicy::RequireProvider,
    ] {
        authorized_stage_exchange(policy).await?;
    }
    Ok(())
}

async fn authorized_stage_exchange(
    policy: mesh_llm_transport::protocol::StageTransportPolicy,
) -> anyhow::Result<()> {
    use crate::mesh::stage_transport::StageTopologyInstance;
    assert!(
        validate_protocols(&[
            Arc::new(StageBridge::new("one")),
            Arc::new(StageBridge::new("two"))
        ])
        .is_err()
    );
    let sender = start_node().await;
    let receiver = start_node().await;
    assert!(
        sender
            .open_application_stage_stream(receiver.id())
            .await?
            .is_none()
    );
    sender
        .install_application_protocols_with_policy(
            vec![Arc::new(StageBridge::new("stage-bridge"))],
            policy,
        )
        .await?;
    receiver
        .install_application_protocols_with_policy(
            vec![Arc::new(StageBridge::new("stage-bridge"))],
            policy,
        )
        .await?;
    sender.start_accepting();
    receiver.start_accepting();
    sender
        .connect_to_peer(receiver.endpoint_addr_for_advertisement())
        .await?;
    sender.refresh_application_peers().await;
    receiver.refresh_application_peers().await;
    let assignment = |name: &str, index, id| StageAssignment {
        stage_id: name.into(),
        stage_index: index,
        node_id: id,
        layer_start: index,
        layer_end: index + 1,
        endpoint: StageEndpoint {
            bind_addr: String::new(),
        },
    };
    receiver
        .record_stage_topology(StageTopologyInstance {
            topology_id: "topology".into(),
            run_id: "run".into(),
            model_id: "model".into(),
            package_ref: String::new(),
            manifest_sha256: String::new(),
            admissions: Default::default(),
            stages: vec![
                assignment("source", 0, sender.id()),
                assignment("destination", 1, receiver.id()),
            ],
        })
        .await;
    reject_invalid_stage_headers(&sender, &receiver).await?;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    receiver
        .register_stage_transport_alias(
            "topology",
            "run",
            "destination",
            listener.local_addr()?.to_string(),
        )
        .await;
    let address = sender
        .ensure_stage_transport_bridge(receiver.id(), "topology", "run", "destination")
        .await?;
    let exchange = async {
        let mut client = tokio::net::TcpStream::connect(address).await?;
        client.write_all(b"activation bytes").await?;
        client.shutdown().await?;
        let mut reply = Vec::new();
        client.read_to_end(&mut reply).await?;
        assert_eq!(reply, b"stage result");
        Ok::<_, anyhow::Error>(())
    };
    let stage = async {
        let (mut socket, _) = listener.accept().await?;
        let mut data = Vec::new();
        socket.read_to_end(&mut data).await?;
        assert_eq!(data, b"activation bytes");
        socket.write_all(b"stage result").await?;
        socket.shutdown().await?;
        Ok::<_, anyhow::Error>(())
    };
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        tokio::try_join!(exchange, stage)
    })
    .await??;
    sender.close_endpoint().await;
    receiver.close_endpoint().await;
    Ok(())
}

async fn reject_invalid_stage_headers(sender: &Node, receiver: &Node) -> anyhow::Result<()> {
    use mesh_llm_transport::protocol::ProtocolStream;
    use prost::Message;
    for (requester, run, stage) in [
        (receiver.id(), "run", "destination"),
        (sender.id(), "stale-run", "destination"),
        (sender.id(), "run", "unknown-stage"),
    ] {
        let open = skippy_protocol::proto::stage::StageTransportOpen {
            r#gen: skippy_protocol::STAGE_PROTOCOL_GENERATION,
            requester_id: requester.as_bytes().to_vec(),
            topology_id: "topology".into(),
            run_id: run.into(),
            stage_id: stage.into(),
        }
        .encode_to_vec();
        let (mut writer, stream) = tokio::io::duplex(4096);
        writer.write_u32_le(u32::try_from(open.len())?).await?;
        writer.write_all(&open).await?;
        assert!(
            receiver
                .accept_application_stage_stream(sender.id(), ProtocolStream::new(stream))
                .await
                .is_err()
        );
    }
    Ok(())
}

#[tokio::test]
async fn application_stage_setup_is_cancelled_on_admission_loss_and_shutdown() -> anyhow::Result<()>
{
    use mesh_llm_transport::protocol::ProtocolStream;
    for shutdown in [false, true] {
        let sender = start_node().await;
        let receiver = start_node().await;
        receiver
            .install_application_protocols(vec![echo("stage-setup", 1)])
            .await?;
        sender.start_accepting();
        receiver.start_accepting();
        sender
            .connect_to_peer(receiver.endpoint_addr_for_advertisement())
            .await?;
        receiver.refresh_application_peers().await;
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
        if shutdown {
            receiver.close_endpoint().await;
        } else {
            receiver.state.lock().await.peers.remove(&sender.id());
            receiver.refresh_application_peers().await;
        }
        let error = tokio::time::timeout(std::time::Duration::from_secs(1), task)
            .await??
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains(if shutdown { "stopped" } else { "admitted" })
        );
        sender.close_endpoint().await;
        receiver.close_endpoint().await;
    }
    Ok(())
}

#[tokio::test]
async fn stage_link_planning_requires_fresh_samples_and_current_admission() {
    use mesh_llm_transport::protocol::StageLinkObservation;
    let sender = start_node().await;
    let receiver = start_node().await;
    let provider = Arc::new(StageBridge::new("measured-stage"));
    sender
        .install_application_protocols(vec![provider.clone()])
        .await
        .unwrap();
    receiver
        .install_application_protocols(vec![Arc::new(StageBridge::new("measured-stage"))])
        .await
        .unwrap();
    let valid = StageLinkObservation {
        acknowledgement_latency: std::time::Duration::from_micros(150),
        payload_bytes_per_second: 10_000_000,
        samples: 3,
        age: std::time::Duration::ZERO,
    };
    *provider.1.lock().unwrap() = Some(valid);
    assert!(sender.application_stage_link(receiver.id()).await.is_none());
    sender.start_accepting();
    receiver.start_accepting();
    sender
        .connect_to_peer(receiver.endpoint_addr_for_advertisement())
        .await
        .unwrap();
    sender.refresh_application_peers().await;
    assert_eq!(
        sender.application_stage_link(receiver.id()).await,
        Some(valid)
    );
    for invalid in [
        StageLinkObservation {
            samples: 2,
            ..valid
        },
        StageLinkObservation {
            age: std::time::Duration::from_secs(30),
            ..valid
        },
        StageLinkObservation {
            acknowledgement_latency: std::time::Duration::ZERO,
            ..valid
        },
    ] {
        *provider.1.lock().unwrap() = Some(invalid);
        assert!(sender.application_stage_link(receiver.id()).await.is_none());
    }
    *provider.1.lock().unwrap() = Some(valid);
    sender
        .state
        .lock()
        .await
        .peers
        .get_mut(&receiver.id())
        .unwrap()
        .application_protocols
        .clear();
    sender.refresh_application_peers().await;
    assert!(sender.application_stage_link(receiver.id()).await.is_none());
    sender.close_endpoint().await;
    receiver.close_endpoint().await;
    assert!(sender.application_stage_link(receiver.id()).await.is_none());
}
