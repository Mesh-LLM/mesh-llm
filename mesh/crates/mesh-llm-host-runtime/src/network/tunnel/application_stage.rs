use anyhow::{Result, ensure};
use iroh::EndpointId;
use mesh_llm_transport::protocol::ProtocolStream;
use prost::Message;
use std::time::Duration;

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
    node.deliver_direct_stage(remote, &open, stream).await
}
