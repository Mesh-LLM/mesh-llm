use anyhow::{Result, ensure};
use iroh::EndpointId;
use mesh_llm_transport::protocol::ProtocolStream;
use prost::Message;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use crate::{
    mesh::Node,
    protocol::{MAX_CONTROL_FRAME_BYTES, read_frame},
};

pub(crate) async fn accept(
    node: Node,
    remote: EndpointId,
    mut stream: ProtocolStream,
) -> Result<()> {
    ensure!(
        node.stage_stream_admitted(remote).await,
        "stage peer is not admitted"
    );
    let bytes = tokio::time::timeout(
        Duration::from_secs(10),
        read_frame(&mut stream, MAX_CONTROL_FRAME_BYTES),
    )
    .await??;
    let open = skippy_protocol::proto::stage::StageTransportOpen::decode(bytes.as_slice())?;
    skippy_protocol::validate_stage_transport_open(&open).map_err(anyhow::Error::msg)?;
    ensure!(
        open.requester_id.as_slice() == remote.as_bytes(),
        "stage requester does not match authenticated peer"
    );
    ensure!(
        node.stage_transport_allowed(remote, &open).await,
        "stage peer is not part of topology run"
    );
    let address = super::resolve_stage_transport_bind_addr(&node, &open).await?;
    let mut tcp = TcpStream::connect(address).await?;
    tcp.set_nodelay(true)?;
    tokio::io::copy_bidirectional(&mut tcp, &mut stream).await?;
    Ok(())
}

pub(crate) async fn outbound(
    node: &Node,
    peer: EndpointId,
    target: (&str, &str, &str),
    mut tcp: TcpStream,
    mut stream: ProtocolStream,
    delay: Option<Duration>,
) -> Result<()> {
    let (topology, run, stage) = target;
    let open = skippy_protocol::proto::stage::StageTransportOpen {
        r#gen: skippy_protocol::STAGE_PROTOCOL_GENERATION,
        requester_id: node.endpoint.id().as_bytes().to_vec(),
        topology_id: topology.into(),
        run_id: run.into(),
        stage_id: stage.into(),
    };
    skippy_protocol::validate_stage_transport_open(&open).map_err(anyhow::Error::msg)?;
    let mut peers = node.application_stage_peers();
    let mut cancellation = node.application_stage_cancellation();
    let transfer = async {
        let bytes = open.encode_to_vec();
        crate::protocol::ensure_control_frame_size(&bytes)?;
        stream.write_u32_le(u32::try_from(bytes.len())?).await?;
        stream.write_all(&bytes).await?;
        let (mut read, mut write) = tokio::io::split(&mut stream);
        let (mut tcp_read, mut tcp_write) = tcp.split();
        let sending = send_bytes(&mut tcp_read, &mut write, delay);
        let receiving = async {
            let mut first = [0];
            let n =
                tokio::time::timeout(Duration::from_secs(5 * 60), read.read(&mut first)).await??;
            tcp_write.write_all(&first[..n]).await?;
            tokio::io::copy(&mut read, &mut tcp_write).await?;
            tcp_write.shutdown().await?;
            Ok::<_, anyhow::Error>(())
        };
        tokio::try_join!(
            async { sending.await.map_err(anyhow::Error::from) },
            receiving
        )?;
        Ok::<_, anyhow::Error>(())
    };
    tokio::pin!(transfer);
    loop {
        ensure!(!*cancellation.borrow(), "stage transport stopped");
        ensure!(
            node.stage_required_peer_allowed(peer, false).await,
            "peer no longer enforces the required stage provider policy"
        );
        ensure!(
            peers.borrow().iter().any(|p| p.id == peer),
            "stage peer is no longer admitted"
        );
        tokio::select! {
            result = &mut transfer => return result,
            changed = peers.changed() => { changed?; },
            _ = cancellation.changed() => {},
        }
    }
}

async fn send_bytes(
    read: &mut (impl tokio::io::AsyncRead + Unpin),
    write: &mut (impl tokio::io::AsyncWrite + Unpin),
    delay: Option<Duration>,
) -> std::io::Result<()> {
    let Some(delay) = delay else {
        tokio::io::copy(read, write).await?;
        return write.shutdown().await;
    };
    // Delay release independently of reads so the RTT emulator does not become
    // an artificial per-chunk bandwidth limit on application transports.
    let (tx, mut rx) = tokio::sync::mpsc::channel::<(tokio::time::Instant, Vec<u8>)>(64);
    let reading = async {
        let mut buffer = vec![0; 64 * 1024];
        loop {
            let count = read.read(&mut buffer).await?;
            if count == 0 {
                break;
            }
            tx.send((
                tokio::time::Instant::now() + delay,
                buffer[..count].to_vec(),
            ))
            .await
            .map_err(|_| std::io::Error::other("stage delay writer stopped"))?;
        }
        drop(tx);
        Ok::<_, std::io::Error>(())
    };
    let writing = async {
        while let Some((release, bytes)) = rx.recv().await {
            tokio::time::sleep_until(release).await;
            write.write_all(&bytes).await?;
        }
        write.shutdown().await
    };
    tokio::try_join!(reading, writing)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[tokio::test(start_paused = true)]
    async fn emulated_delay_holds_chunks_without_serializing_their_waits() {
        let bytes = vec![7; 256 * 1024];
        let mut input = bytes.as_slice();
        let mut output = Vec::new();
        let delay = std::time::Duration::from_millis(50);
        let start = tokio::time::Instant::now();
        super::send_bytes(&mut input, &mut output, Some(delay))
            .await
            .unwrap();
        assert_eq!(output, bytes);
        assert_eq!(start.elapsed(), delay);
    }
}
