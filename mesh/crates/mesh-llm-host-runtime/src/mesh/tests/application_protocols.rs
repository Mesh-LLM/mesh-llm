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

async fn start_node() -> Node {
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
