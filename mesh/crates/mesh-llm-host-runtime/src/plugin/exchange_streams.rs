//! Bounded ordered copies on negotiated local side streams, never the control socket.
use super::transport::LocalStream;
use super::{PluginManager, proto};
use anyhow::{Context, Result, bail};
use futures_util::stream::{FuturesUnordered, StreamExt};
use mesh_llm_config::OpenAiExchangeGrant;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::time::Duration;
use tokio::{sync::mpsc, task::JoinHandle, time::Instant};
const COPY_CHUNK: usize = 16 * 1024;

pub(super) async fn open_body_stream(
    manager: &PluginManager,
    name: &str,
    exchange_id: &str,
    kind: &str,
    expected_bytes: Option<u64>,
    deadline: Instant,
) -> Result<LocalStream> {
    let mut revision = manager.exchange_grant_revision(name);
    tokio::select! {
        biased;
        _ = revision.changed() => bail!("observer body grant was revoked"),
        stream = open_body_stream_granted(manager, name, exchange_id,kind,expected_bytes,deadline) => stream,
    }
}

async fn open_body_stream_granted(
    manager: &PluginManager,
    name: &str,
    exchange_id: &str,
    kind: &str,
    expected_bytes: Option<u64>,
    deadline: Instant,
) -> Result<LocalStream> {
    let stream_id = uuid::Uuid::new_v4().to_string();
    let response = tokio::time::timeout_at(
        deadline,
        manager.open_stream(
            name,
            proto::OpenStreamRequest {
                stream_id: stream_id.clone(),
                purpose: proto::StreamPurpose::HttpResponseBody as i32,
                mode: proto::StreamMode::RawBytes as i32,
                bidirectional: true,
                content_type: Some("application/octet-stream".into()),
                correlation_id: Some(exchange_id.into()),
                metadata_json: Some(
                    json!({"kind":kind,"exchange_id":exchange_id,"receipt_protocol":"sha256-v1"})
                        .to_string(),
                ),
                expected_bytes,
                idle_timeout_ms: Some(30_000),
            },
        ),
    )
    .await??;
    if !response.accepted || response.stream_id != stream_id {
        bail!("observer did not accept the correlated body stream");
    }
    let token = response
        .token
        .context("observer side stream has no authentication token")?;
    if uuid::Uuid::parse_str(&token).is_err() || token.len() != 36 {
        bail!("observer side stream token must be a UUID");
    }
    let endpoint = response
        .endpoint
        .context("observer side stream has no endpoint")?;
    let mut stream = tokio::time::timeout_at(
        deadline,
        super::transport::connect_side_stream(&endpoint, response.transport_kind),
    )
    .await??;
    tokio::time::timeout_at(deadline, stream.write_all(token.as_bytes())).await??;
    Ok(stream)
}

pub(super) async fn send_request_body(
    manager: &PluginManager,
    name: &str,
    exchange_id: &str,
    kind: &str,
    bytes: &[u8],
    deadline: Instant,
) -> Result<()> {
    let mut revision = manager.exchange_grant_revision(name);
    tokio::select! {
        biased;
        _ = revision.changed() => bail!("observer body grant was revoked"),
        result = send_request_body_granted(manager,name,exchange_id,kind,bytes,deadline) => result,
    }
}

async fn send_request_body_granted(
    manager: &PluginManager,
    name: &str,
    exchange_id: &str,
    kind: &str,
    bytes: &[u8],
    deadline: Instant,
) -> Result<()> {
    let mut stream = open_body_stream(
        manager,
        name,
        exchange_id,
        kind,
        Some(bytes.len() as u64),
        deadline,
    )
    .await?;
    tokio::time::timeout_at(deadline, stream.write_all(bytes)).await??;
    tokio::time::timeout_at(deadline, stream.shutdown()).await??;
    read_receipt(
        &mut stream,
        &hex::encode(Sha256::digest(bytes)),
        bytes.len() as u64,
        deadline,
    )
    .await?;
    Ok(())
}

struct StreamCopy {
    tx: Option<mpsc::Sender<Vec<u8>>>,
    failed: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    max_body_bytes: u64,
    max_queue_bytes: u64,
    queued_bytes: Arc<AtomicU64>,
    name: String,
    revision: tokio::sync::watch::Receiver<u64>,
    required: bool,
}

/// A response subscription retains capacity until its worker exits or is aborted.
struct ResponsePermit {
    health: super::exchange_lifecycle::HealthStates,
    name: String,
    success: bool,
}
impl ResponsePermit {
    fn acquire(manager: &PluginManager, name: &str, maximum: u32) -> Option<Self> {
        let mut health = manager.inner.exchange_health.lock().unwrap();
        let state = health.entry(name.into()).or_default();
        if state
            .response_open_until
            .is_some_and(|until| until > Instant::now())
        {
            return None;
        }
        if state.response_in_flight >= maximum {
            state.response_failures = state.response_failures.saturating_add(1);
            return None;
        }
        state.response_in_flight += 1;
        Some(Self {
            health: manager.inner.exchange_health.clone(),
            name: name.into(),
            success: false,
        })
    }
}
impl Drop for ResponsePermit {
    fn drop(&mut self) {
        if let Some(state) = self.health.lock().unwrap().get_mut(&self.name) {
            state.response_in_flight = state.response_in_flight.saturating_sub(1);
            if self.success {
                state.response_failures = 0;
                state.response_open_until = None;
            } else {
                state.response_failures = state.response_failures.saturating_add(1);
                if state.response_failures >= 3 {
                    state.response_open_until = Some(Instant::now() + Duration::from_secs(30));
                }
            }
        }
    }
}

#[derive(Default)]
pub(super) struct ResponseCopies {
    copies: Vec<StreamCopy>,
}
impl ResponseCopies {
    #[cfg(test)]
    pub(super) fn for_test_stream(
        stream: LocalStream,
        grant: &OpenAiExchangeGrant,
        revision: tokio::sync::watch::Receiver<u64>,
    ) -> Self {
        Self {
            copies: vec![Self::copy(Ok(stream), grant, "observer".into(), revision)],
        }
    }
    pub(super) async fn open(
        manager: &PluginManager,
        event: &serde_json::Value,
        outer_deadline: Instant,
    ) -> Self {
        let mut subscriptions = Vec::new();
        let mut copies = Vec::new();
        for (name, plugin) in &manager.inner.plugins {
            let Some(manifest) = plugin.manifest_snapshot().await else {
                continue;
            };
            let revision = manager.exchange_grant_revision(name);
            let configured_grant = manager.effective_exchange_grant(name);
            let declaration = manifest.openai_exchange_hook.as_deref();
            if !declaration.is_some_and(|hook| hook.response_body)
                || !super::exchange_permissions::subscribes(
                    declaration,
                    configured_grant.as_ref(),
                    event["endpoint"].as_str().unwrap_or_default(),
                    "exchange_finished",
                )
            {
                continue;
            }
            manager.refresh_exchange_permissions_status(name, declaration);
            let negotiation = mesh_llm_plugin::openai_exchange::negotiate_openai_exchange(
                manifest.openai_exchange_hook.as_deref(),
                configured_grant.as_ref(),
            );
            let grant = match negotiation {
                Ok(Some(grant)) => grant,
                Ok(None) => continue,
                Err(_) => {
                    let mut copy = Self::copy_with_permit(
                        Err(anyhow::anyhow!("observer permissions unavailable")),
                        configured_grant.as_ref().unwrap(),
                        name.clone(),
                        revision,
                        None,
                    );
                    copy.required = true;
                    copies.push(copy);
                    continue;
                }
            };
            if !grant.response_body
                || !grant
                    .phases
                    .iter()
                    .any(|phase| phase == "exchange_finished")
                || !grant
                    .endpoints
                    .iter()
                    .any(|v| event["endpoint"] == v.as_str())
            {
                continue;
            }
            subscriptions.push((name, grant, revision));
        }
        let deadline = (Instant::now()
            + Duration::from_millis(
                subscriptions
                    .iter()
                    .map(|(_, grant, _)| grant.deadline_ms)
                    .min()
                    .unwrap_or(1),
            ))
        .min(outer_deadline);
        let mut tasks = FuturesUnordered::new();
        for (name, grant, revision) in subscriptions {
            tasks.push(async move {
                let permit = ResponsePermit::acquire(manager, name, grant.max_in_flight);
                let stream = if permit.is_some() {
                    open_body_stream(
                        manager,
                        name,
                        event["exchange_id"].as_str().unwrap_or_default(),
                        "openai_exchange_response",
                        None,
                        deadline,
                    )
                    .await
                } else {
                    Err(anyhow::anyhow!("response observer capacity exhausted"))
                };
                Self::copy_with_permit(stream, &grant, name.clone(), revision, permit)
            });
        }
        while let Some(copy) = tasks.next().await {
            copies.push(copy);
        }
        Self { copies }
    }
    #[cfg(test)]
    fn copy(
        stream: Result<LocalStream>,
        grant: &OpenAiExchangeGrant,
        name: String,
        revision: tokio::sync::watch::Receiver<u64>,
    ) -> StreamCopy {
        Self::copy_with_permit(stream, grant, name, revision, None)
    }
    fn copy_with_permit(
        stream: Result<LocalStream>,
        grant: &OpenAiExchangeGrant,
        name: String,
        revision: tokio::sync::watch::Receiver<u64>,
        permit: Option<ResponsePermit>,
    ) -> StreamCopy {
        let failed = Arc::new(AtomicBool::new(stream.is_err()));
        let required =
            grant.failure_policy == mesh_llm_config::OpenAiExchangeFailurePolicy::Required;
        let queued_bytes = Arc::new(AtomicU64::new(0));
        let Ok(mut stream) = stream else {
            return StreamCopy {
                tx: None,
                failed,
                worker: None,
                max_body_bytes: grant.max_body_bytes,
                max_queue_bytes: grant.max_queue_bytes,
                queued_bytes,
                name,
                revision,
                required,
            };
        };
        // Frames may be much smaller than COPY_CHUNK. Byte accounting below
        // enforces the exact budget independently of this bounded item count.
        let capacity = usize::try_from(grant.max_queue_bytes)
            .unwrap_or(usize::MAX)
            .clamp(1, 4096);
        let (tx, mut rx) = mpsc::channel::<Vec<u8>>(capacity);
        let worker_failed = failed.clone();
        let worker_queued = queued_bytes.clone();
        let mut worker_revision = revision.clone();
        let worker = tokio::spawn(async move {
            let mut permit = permit;
            let work = async {
                let mut hash = Sha256::new();
                let mut count = 0u64;
                while let Some(bytes) = rx.recv().await {
                    let succeeded = matches!(
                        tokio::time::timeout(Duration::from_secs(30), stream.write_all(&bytes))
                            .await,
                        Ok(Ok(()))
                    );
                    worker_queued.fetch_sub(bytes.len() as u64, Ordering::AcqRel);
                    if !succeeded {
                        worker_failed.store(true, Ordering::Release);
                        break;
                    }
                    hash.update(&bytes);
                    count += bytes.len() as u64;
                }
                let receipt_deadline = Instant::now() + Duration::from_secs(1);
                if !matches!(
                    tokio::time::timeout_at(receipt_deadline, stream.shutdown()).await,
                    Ok(Ok(()))
                ) || read_receipt(
                    &mut stream,
                    &hex::encode(hash.finalize()),
                    count,
                    receipt_deadline,
                )
                .await
                .is_err()
                {
                    worker_failed.store(true, Ordering::Release);
                }
            };
            tokio::select! {
                biased;
                _ = worker_revision.changed() => { worker_failed.store(true, Ordering::Release); },
                _ = work => {},
            }
            if let Some(permit) = permit.as_mut() {
                permit.success = !worker_failed.load(Ordering::Acquire);
            }
        });
        StreamCopy {
            tx: Some(tx),
            failed,
            worker: Some(worker),
            max_body_bytes: grant.max_body_bytes,
            max_queue_bytes: grant.max_queue_bytes,
            queued_bytes,
            name,
            revision,
            required,
        }
    }
    pub(super) fn admission_failure(&self) -> (bool, bool) {
        let unavailable = self
            .copies
            .iter()
            .any(|copy| copy.failed.load(Ordering::Acquire));
        let required = self
            .copies
            .iter()
            .any(|copy| copy.required && copy.failed.load(Ordering::Acquire));
        (unavailable, required)
    }
    pub(super) fn enqueue(&self, offset: u64, bytes: &[u8]) -> bool {
        let mut complete = true;
        for copy in &self.copies {
            if !matches!(copy.revision.has_changed(), Ok(false)) {
                copy.failed.store(true, Ordering::Release);
            }
            if offset.saturating_add(bytes.len() as u64) > copy.max_body_bytes {
                copy.failed.store(true, Ordering::Release);
            }
            if !copy.failed.load(Ordering::Acquire) {
                for chunk in bytes.chunks(COPY_CHUNK) {
                    let size = chunk.len() as u64;
                    if copy
                        .queued_bytes
                        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |queued| {
                            queued
                                .checked_add(size)
                                .filter(|total| *total <= copy.max_queue_bytes)
                        })
                        .is_err()
                    {
                        copy.failed.store(true, Ordering::Release);
                        break;
                    }
                    if copy
                        .tx
                        .as_ref()
                        .is_none_or(|tx| tx.try_send(chunk.to_vec()).is_err())
                    {
                        copy.queued_bytes.fetch_sub(size, Ordering::AcqRel);
                        copy.failed.store(true, Ordering::Release);
                        break;
                    }
                }
            }
            complete &= !copy.failed.load(Ordering::Acquire);
        }
        complete
    }
    pub(super) async fn close_with_receipts(&mut self) -> std::collections::BTreeMap<String, bool> {
        let deadline = Instant::now() + Duration::from_secs(2);
        // Close every producer first, so all workers drain concurrently.
        for copy in &mut self.copies {
            copy.tx.take();
        }
        let mut complete = std::collections::BTreeMap::new();
        for copy in &mut self.copies {
            if let Some(mut worker) = copy.worker.take() {
                match tokio::time::timeout_at(deadline, &mut worker).await {
                    Ok(Ok(())) => {}
                    _ => {
                        copy.failed.store(true, Ordering::Release);
                        worker.abort();
                    }
                }
            }
            complete.insert(copy.name.clone(), !copy.failed.load(Ordering::Acquire));
        }
        complete
    }
}

async fn read_receipt(
    stream: &mut LocalStream,
    digest: &str,
    count: u64,
    deadline: Instant,
) -> Result<()> {
    let mut receipt = Vec::new();
    let mut byte = [0u8; 1];
    while receipt.len() < 512 {
        if tokio::time::timeout_at(deadline, stream.read(&mut byte)).await?? != 1 {
            bail!("observer disconnected before receipt");
        }
        if byte[0] == b'\n' {
            let value: serde_json::Value = serde_json::from_slice(&receipt)?;
            if value["sha256"] == digest && value["byte_count"] == count {
                return Ok(());
            }
            bail!("observer byte receipt does not match host stream");
        }
        receipt.push(byte[0]);
    }
    bail!("observer byte receipt exceeds limit")
}

impl Drop for ResponseCopies {
    fn drop(&mut self) {
        for copy in &mut self.copies {
            copy.tx.take();
            if let Some(worker) = copy.worker.take() {
                worker.abort();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[test]
    fn persistent_capacity_is_released_and_delivery_health_recovers() {
        let health: super::super::exchange_lifecycle::HealthStates = Arc::default();
        for _ in 0..3 {
            health
                .lock()
                .unwrap()
                .entry("observer".into())
                .or_default()
                .response_in_flight += 1;
            drop(ResponsePermit {
                health: health.clone(),
                name: "observer".into(),
                success: false,
            });
        }
        let state = health.lock().unwrap();
        assert_eq!(state["observer"].response_in_flight, 0);
        assert_eq!(state["observer"].response_failures, 3);
        assert!(state["observer"].response_open_until.unwrap() > Instant::now());
        drop(state);
        health
            .lock()
            .unwrap()
            .get_mut("observer")
            .unwrap()
            .response_in_flight = 1;
        drop(ResponsePermit {
            health: health.clone(),
            name: "observer".into(),
            success: true,
        });
        let state = health.lock().unwrap();
        assert_eq!(state["observer"].response_in_flight, 0);
        assert_eq!(state["observer"].response_failures, 0);
        assert!(state["observer"].response_open_until.is_none());
    }

    async fn acknowledge(mut peer: tokio::io::DuplexStream, wrong: bool) -> Vec<u8> {
        let mut received = Vec::new();
        peer.read_to_end(&mut received).await.unwrap();
        let digest = if wrong {
            "incorrect".to_owned()
        } else {
            hex::encode(Sha256::digest(&received))
        };
        let receipt = json!({"sha256":digest,"byte_count":received.len()});
        peer.write_all(format!("{receipt}\n").as_bytes())
            .await
            .unwrap();
        peer.shutdown().await.unwrap();
        received
    }

    #[tokio::test]
    async fn failed_recipient_does_not_stop_ordered_healthy_recipient() {
        let grant = OpenAiExchangeGrant {
            max_body_bytes: 1024,
            max_queue_bytes: 1024,
            ..Default::default()
        };
        let (_revision_tx, revision) = tokio::sync::watch::channel(0);
        let (stream, peer) = tokio::io::duplex(1024);
        let receipt = tokio::spawn(acknowledge(peer, false));
        let mut copies = ResponseCopies {
            copies: vec![
                ResponseCopies::copy(
                    Err(anyhow::anyhow!("offline")),
                    &grant,
                    "failed".into(),
                    revision.clone(),
                ),
                ResponseCopies::copy(
                    Ok(LocalStream::Memory(stream)),
                    &grant,
                    "healthy".into(),
                    revision,
                ),
            ],
        };
        assert!(!copies.enqueue(0, b"data: first\n\n"));
        assert!(!copies.enqueue(13, b"data: [DONE]\n\n"));
        let delivery = copies.close_with_receipts().await;
        assert!(!delivery["failed"]);
        assert!(delivery["healthy"]);
        assert_eq!(receipt.await.unwrap(), b"data: first\n\ndata: [DONE]\n\n");
    }

    #[tokio::test]
    async fn receipt_mismatch_is_incomplete_even_after_successful_write() {
        let grant = OpenAiExchangeGrant {
            max_body_bytes: 1024,
            max_queue_bytes: 1024,
            ..Default::default()
        };
        let (_revision_tx, revision) = tokio::sync::watch::channel(0);
        let (stream, peer) = tokio::io::duplex(1024);
        let receipt = tokio::spawn(acknowledge(peer, true));
        let mut copies = ResponseCopies {
            copies: vec![ResponseCopies::copy(
                Ok(LocalStream::Memory(stream)),
                &grant,
                "observer".into(),
                revision,
            )],
        };
        assert!(copies.enqueue(0, b"abc"));
        assert!(!copies.close_with_receipts().await["observer"]);
        assert_eq!(receipt.await.unwrap(), b"abc");
    }

    #[tokio::test]
    async fn queue_budget_and_revocation_apply_before_copy_allocation() {
        let grant = OpenAiExchangeGrant {
            max_body_bytes: 1024,
            max_queue_bytes: 2,
            ..Default::default()
        };
        let (revision_tx, revision) = tokio::sync::watch::channel(0);
        let (stream, _peer) = tokio::io::duplex(1024);
        let mut copies = ResponseCopies {
            copies: vec![ResponseCopies::copy(
                Ok(LocalStream::Memory(stream)),
                &grant,
                "observer".into(),
                revision,
            )],
        };
        assert!(!copies.enqueue(0, b"abc"));
        assert_eq!(copies.copies[0].queued_bytes.load(Ordering::Acquire), 0);
        revision_tx.send(1).unwrap();
        assert!(!copies.enqueue(0, b"a"));
        assert!(!copies.close_with_receipts().await["observer"]);
    }
}
