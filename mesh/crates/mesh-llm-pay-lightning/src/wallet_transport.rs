//! Multiplex the child control connection; long incoming waits cannot block sends.
use anyhow::{Context, Result, ensure};
use mesh_llm_plugin::{LocalStream, PROTOCOL_VERSION, proto};
use prost::Message;
use std::{
    collections::BTreeMap,
    sync::atomic::{AtomicU64, Ordering},
};
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWriteExt},
    sync::{mpsc, oneshot},
};

/// The transport rejected this request before attempting any socket write.
#[derive(Debug)]
pub(crate) struct NotDispatched(pub(crate) &'static str);
impl std::fmt::Display for NotDispatched {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for NotDispatched {}

type Reply = oneshot::Sender<Result<proto::envelope::Payload>>;
struct Request {
    id: u64,
    payload: proto::envelope::Payload,
    reply: Reply,
}
pub struct Transport {
    tx: mpsc::Sender<Request>,
    next: AtomicU64,
}

impl Transport {
    pub fn start(stream: LocalStream, plugin: String) -> Self {
        let (tx, rx) = mpsc::channel(32);
        tokio::spawn(run(stream, rx, plugin));
        Self {
            tx,
            next: AtomicU64::new(1),
        }
    }
    pub async fn exchange(
        &self,
        payload: proto::envelope::Payload,
    ) -> Result<proto::envelope::Payload> {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        let (reply, response) = oneshot::channel();
        self.tx
            .send(Request { id, payload, reply })
            .await
            .map_err(|_| NotDispatched("wallet transport stopped"))?;
        response.await.context("wallet transport lost response")?
    }
}

async fn run(stream: LocalStream, mut rx: mpsc::Receiver<Request>, plugin: String) {
    let (mut read, mut write) = stream.into_split();
    let (incoming, mut responses) = mpsc::channel(32);
    let reader = tokio::spawn(async move {
        loop {
            let response = read_frame(&mut *read).await;
            let failed = response.is_err();
            if incoming.send(response).await.is_err() || failed {
                break;
            }
        }
    });
    let mut pending = BTreeMap::<u64, Reply>::new();
    loop {
        tokio::select! {
            request = rx.recv() => {
                let Some(request) = request else { break; };
                // The reader owns partial framing; cancelling this select never drops bytes.
                if pending.len() >= 64 {
                    let _ = request.reply.send(Err(NotDispatched("wallet in-flight limit").into()));
                    continue;
                }
                let envelope = proto::Envelope { protocol_version: PROTOCOL_VERSION, plugin_id: plugin.clone(),
                    request_id: request.id, payload: Some(request.payload) };
                let body = envelope.encode_to_vec();
                pending.insert(request.id, request.reply);
                if write.write_all(&(body.len() as u32).to_le_bytes()).await.is_err()
                    || write.write_all(&body).await.is_err() { break; }
            }
            response = responses.recv() => {
                let Some(Ok(response)) = response else { break; };
                if response.protocol_version != PROTOCOL_VERSION || response.plugin_id != plugin { break; }
                let Some(reply) = pending.remove(&response.request_id) else { break; };
                let _ = reply.send(response.payload.context("empty wallet response"));
            }
        }
    }
    reader.abort();
    // Every unresolved call fails uncertain at the adapter; never reconnect/replay.
    for (_, reply) in pending {
        let _ = reply.send(Err(anyhow::anyhow!("wallet connection closed")));
    }
}
async fn read_frame<R: AsyncRead + Unpin + ?Sized>(read: &mut R) -> Result<proto::Envelope> {
    let len = read.read_u32_le().await?;
    ensure!(len <= 16 * 1024 * 1024, "oversized wallet frame");
    let mut bytes = vec![0; len as usize];
    read.read_exact(&mut bytes).await?;
    Ok(proto::Envelope::decode(bytes.as_slice())?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_llm_plugin::{read_envelope, write_envelope};
    #[tokio::test]
    async fn stopped_before_enqueue_is_definitely_not_dispatched() {
        let (tx, rx) = mpsc::channel(1);
        drop(rx);
        let transport = Transport {
            tx,
            next: AtomicU64::new(1),
        };
        let error = transport
            .exchange(proto::envelope::Payload::HealthRequest(Default::default()))
            .await
            .unwrap_err();
        assert!(error.is::<NotDispatched>());
    }

    #[tokio::test]
    async fn admitted_request_disconnect_remains_uncertain() {
        let (client, server) = tokio::io::duplex(8192);
        let transport = Transport::start(LocalStream::Memory(client), "mock-wallet".into());
        let server = tokio::spawn(async move {
            let mut stream = LocalStream::Memory(server);
            read_envelope(&mut stream).await.unwrap();
        });
        let error = transport
            .exchange(proto::envelope::Payload::HealthRequest(Default::default()))
            .await
            .unwrap_err();
        assert!(!error.is::<NotDispatched>());
        server.await.unwrap();
    }

    #[tokio::test]
    async fn out_of_order_responses_are_correlated() {
        let (client, server) = tokio::io::duplex(8192);
        let transport = Transport::start(LocalStream::Memory(client), "mock-wallet".into());
        let server = tokio::spawn(async move {
            let mut stream = LocalStream::Memory(server);
            let first = read_envelope(&mut stream).await.unwrap();
            let second = read_envelope(&mut stream).await.unwrap();
            for envelope in [second, first] {
                let response = proto::Envelope {
                    payload: Some(proto::envelope::Payload::HealthResponse(
                        proto::HealthResponse {
                            status: 1,
                            detail: envelope.request_id.to_string(),
                        },
                    )),
                    ..envelope
                };
                write_envelope(&mut stream, &response).await.unwrap();
            }
        });
        let (first, second) = tokio::join!(
            transport.exchange(proto::envelope::Payload::HealthRequest(Default::default())),
            transport.exchange(proto::envelope::Payload::HealthRequest(Default::default()))
        );
        for (reply, expected) in [(first, "1"), (second, "2")] {
            let proto::envelope::Payload::HealthResponse(reply) = reply.unwrap() else {
                panic!("wrong reply")
            };
            assert_eq!(reply.detail, expected);
        }
        server.await.unwrap();
    }
}
