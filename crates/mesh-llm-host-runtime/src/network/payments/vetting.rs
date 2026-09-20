//! Selected-provider sanity checks. This is not a proof of model identity.
use crate::{
    inference::election::{InferenceTarget, ModelTargets},
    mesh::Node,
};
use anyhow::{Context, Result, ensure};
use mesh_llm_payments::vetting::{CHALLENGE_VERSION, VettingRecord};
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
// tokio's clock, so paused-clock tests drive these windows deterministically.
use tokio::time::Instant;

pub(crate) const UPGRADE: &[u8] =
    b"POST /mesh/vetting/v1 HTTP/1.1\r\nHost: mesh\r\nContent-Length: 0\r\n\r\n";
const DEADLINE: Duration = Duration::from_secs(10);
// One provider reservation window, shared by both sides so a client that waits out
// its own cooldown is actually eligible at the provider rather than refused again.
const PEER_PROBE_RESERVATION: Duration = Duration::from_secs(60);
const MAX_FRAME: usize = 4096;
// Process-wide limits also bound callers rotating their endpoint identities.
static PROVIDER_SLOT: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(1);
static LAST_PROBE: std::sync::Mutex<Option<Instant>> = std::sync::Mutex::new(None);
static PEER_PROBES: LazyLock<
    std::sync::Mutex<std::collections::HashMap<iroh::EndpointId, Instant>>,
> = LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));
static FAILED_PROBES: LazyLock<
    std::sync::Mutex<std::collections::HashMap<iroh::EndpointId, Instant>>,
> = LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

fn reserve_peer_probe(peer: iroh::EndpointId) -> Result<()> {
    let mut peers = PEER_PROBES
        .lock()
        .map_err(|_| anyhow::anyhow!("probe limiter unavailable"))?;
    peers.retain(|_, time| time.elapsed() < PEER_PROBE_RESERVATION);
    ensure!(
        !peers.contains_key(&peer) && peers.len() < 1024,
        "peer probe rate limited"
    );
    peers.insert(peer, Instant::now());
    Ok(())
}

// Read-only: rejecting an attempt locally must never postpone eligibility, or steady
// traffic during the cooldown would keep pushing the retry out indefinitely.
fn check_retry_cooldown(peer: iroh::EndpointId) -> Result<()> {
    let mut failures = FAILED_PROBES
        .lock()
        .map_err(|_| anyhow::anyhow!("probe cooldown unavailable"))?;
    failures.retain(|_, time| time.elapsed() < PEER_PROBE_RESERVATION);
    ensure!(
        !failures.contains_key(&peer),
        "provider probe retry cooldown"
    );
    Ok(())
}

// Only an attempted remote probe may start a cooldown. This covers the whole attempt
// from opening the tunnel onward, so it includes failures where the provider was never
// reached; what it excludes is a purely local refusal (cooldown gate or queue wait).
fn record_probe_failure(peer: iroh::EndpointId) {
    if let Ok(mut failures) = FAILED_PROBES.lock() {
        failures.retain(|_, time| time.elapsed() < PEER_PROBE_RESERVATION);
        if failures.len() < 1024 {
            failures.insert(peer, Instant::now());
        }
    }
}

// Serialize cache misses and recheck under the lock: concurrent requests for the
// same provider share a successful observation rather than launching duplicates.
static CLIENT_PROBES: LazyLock<tokio::sync::Mutex<()>> =
    LazyLock::new(|| tokio::sync::Mutex::new(()));

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Challenge {
    version: u32,
    model: String,
    nonce: String,
    left: u8,
    right: u8,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Answer {
    version: u32,
    nonce: String,
    text: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Capability {
    version: u32,
    supported: bool,
}

fn validate_answer(challenge: &Challenge, answer: &Answer) -> Result<()> {
    ensure!(
        answer.version == CHALLENGE_VERSION && answer.nonce == challenge.nonce,
        "probe identity mismatch"
    );
    ensure!(
        answer.text.trim() == (u16::from(challenge.left) + u16::from(challenge.right)).to_string(),
        "provider failed inference sanity check"
    );
    Ok(())
}

pub(crate) fn is_upgrade(prefix: &[u8]) -> bool {
    prefix.starts_with(b"POST /mesh/vetting/v1 HTTP/1.1\r\n")
}

pub(crate) async fn verify(node: &Node, peer: iroh::EndpointId, model: &str) -> Result<()> {
    let service = node.payment_service().await?;
    let policy = service.ledger.vetting_policy()?;
    if !policy.required {
        return Ok(());
    }
    let id = peer.to_string();
    if service
        .ledger
        .provider_vetted(&id, mesh_llm_payments::now_ms(), policy.ttl_ms)?
    {
        return Ok(());
    }
    // Queueing behind another caller and the local cooldown gate are bounded outside the
    // probe deadline: neither is evidence about the provider, so neither may be recorded
    // as a probe failure below.
    let _single_flight = tokio::time::timeout(DEADLINE, CLIENT_PROBES.lock())
        .await
        .context("provider sanity check queue wait exceeded")?;
    if service
        .ledger
        .provider_vetted(&id, mesh_llm_payments::now_ms(), policy.ttl_ms)?
    {
        return Ok(());
    }
    check_retry_cooldown(peer)?;
    let result = tokio::time::timeout(DEADLINE, async {
        let random = uuid::Uuid::new_v4();
        let challenge = Challenge {
            version: CHALLENGE_VERSION,
            model: model.into(),
            nonce: random.to_string(),
            left: random.as_bytes()[0] % 20,
            right: random.as_bytes()[1] % 20,
        };
        let (mut send, mut recv) = node.open_http_tunnel(peer).await?;
        send.write_all(UPGRADE).await?;
        let capability: Capability = read_json(&mut recv).await?;
        ensure!(
            capability.version == CHALLENGE_VERSION && capability.supported,
            "provider does not offer sanity probes"
        );
        write_json(&mut send, &challenge).await?;
        let answer: Answer = read_json(&mut recv).await?;
        validate_answer(&challenge, &answer)?;
        service.ledger.record_provider_vetted(VettingRecord {
            provider_id: id,
            model: model.into(),
            checked_at_ms: mesh_llm_payments::now_ms(),
            challenge_version: CHALLENGE_VERSION,
        })
    })
    .await
    .context("provider sanity check deadline exceeded")
    .and_then(|result| result);
    if result.is_err() {
        record_probe_failure(peer);
    }
    result
}

pub(crate) async fn serve(
    remote: iroh::EndpointId,
    node: &Node,
    mut reader: impl AsyncRead + Unpin,
    mut writer: impl AsyncWrite + Unpin,
    targets: ModelTargets,
) -> Result<()> {
    let enabled = node
        .payment_service()
        .await?
        .ledger
        .vetting_policy()?
        .serve_probes;
    tokio::time::timeout(
        DEADLINE,
        write_json(
            &mut writer,
            &Capability {
                version: CHALLENGE_VERSION,
                supported: enabled,
            },
        ),
    )
    .await??;
    ensure!(enabled, "provider sanity probes disabled");
    let _slot = PROVIDER_SLOT.try_acquire().context("probe busy")?;
    ensure!(
        node.state
            .lock()
            .await
            .peers
            .get(&remote)
            .is_some_and(|peer| peer.is_admitted()),
        "probe requires admitted peer"
    );
    {
        let mut last = LAST_PROBE
            .lock()
            .map_err(|_| anyhow::anyhow!("probe limiter unavailable"))?;
        ensure!(
            last.is_none_or(|time| time.elapsed() >= Duration::from_secs(1)),
            "probe rate limited"
        );
        *last = Some(Instant::now());
    }
    reserve_peer_probe(remote)?;
    tokio::time::timeout(DEADLINE, async {
        let challenge: Challenge = read_json(&mut reader).await?;
        ensure!(challenge.version == CHALLENGE_VERSION && challenge.left < 20 && challenge.right < 20, "unsupported challenge");
        ensure!(!challenge.model.is_empty() && challenge.model.len() <= 1024, "invalid probe model");
        uuid::Uuid::parse_str(&challenge.nonce)?;
        let port = targets.candidates(&challenge.model).iter().find_map(|target| match target { InferenceTarget::Local(port) => Some(*port), _ => None }).context("probe requires a ready local model")?;
        let _active = node.begin_runtime_instance_request(port).await?;
        let mut response = reqwest::Client::new().post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
            .json(&serde_json::json!({"model":challenge.model,"messages":[{"role":"user","content":format!("What is {} + {}? Reply with only the integer, no explanation.",challenge.left,challenge.right)}],"max_tokens":32,"temperature":0,"stream":false}))
            .send().await?.error_for_status()?;
        let mut bytes = Vec::new();
        while let Some(chunk) = response.chunk().await? {
            ensure!(bytes.len() + chunk.len() <= 16 * 1024, "probe backend response too large");
            bytes.extend_from_slice(&chunk);
        }
        let body: serde_json::Value = serde_json::from_slice(&bytes)?;
        let text = body["choices"][0]["message"]["content"].as_str().context("probe answer missing")?;
        ensure!(text.len() <= 256, "probe answer too large");
        write_json(&mut writer, &Answer { version: CHALLENGE_VERSION, nonce: challenge.nonce, text: text.into() }).await
    }).await.context("provider probe deadline exceeded")?
}

async fn read_json<T: serde::de::DeserializeOwned>(
    reader: &mut (impl AsyncRead + Unpin),
) -> Result<T> {
    let length = reader.read_u32().await? as usize;
    ensure!(length > 0 && length <= MAX_FRAME, "probe frame too large");
    let mut bytes = vec![0; length];
    reader.read_exact(&mut bytes).await?;
    Ok(serde_json::from_slice(&bytes)?)
}
async fn write_json(writer: &mut (impl AsyncWrite + Unpin), value: &impl Serialize) -> Result<()> {
    let bytes = serde_json::to_vec(value)?;
    ensure!(bytes.len() <= MAX_FRAME, "probe frame too large");
    writer.write_u32(bytes.len() as u32).await?;
    writer.write_all(&bytes).await?;
    writer.flush().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn fresh_cached_provider_needs_no_connection_or_wallet_calls() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let client = Node::new_for_tests(crate::mesh::NodeRole::Client).await?;
        let provider = Node::new_for_tests(crate::mesh::NodeRole::Client).await?;
        let service = std::sync::Arc::new(mesh_llm_payments::service::PaymentService::open(
            directory.path(),
        )?);
        service
            .ledger
            .set_vetting_policy(&mesh_llm_payments::vetting::VettingPolicy {
                required: true,
                serve_probes: false,
                ttl_ms: 1000,
            })?;
        client
            .payments
            .set(service.clone())
            .map_err(|_| anyhow::anyhow!("already initialized"))?;
        service.ledger.record_provider_vetted(VettingRecord {
            provider_id: provider.id().to_string(),
            model: "test".into(),
            checked_at_ms: mesh_llm_payments::now_ms(),
            challenge_version: CHALLENGE_VERSION,
        })?;
        // No server accepts a stream: a fresh cached call must finish.
        verify(&client, provider.id(), "test").await?;
        assert!(
            FAILED_PROBES.lock().unwrap().get(&provider.id()).is_none(),
            "a successful probe must leave no cooldown"
        );
        assert!(!service.has_wallet());
        assert!(service.ledger.requests()?.is_empty());
        client.endpoint.close().await;
        provider.endpoint.close().await;
        Ok(())
    }

    #[tokio::test]
    async fn real_tunnel_probe_runs_local_backend_then_reuses_cache() -> Result<()> {
        let client_dir = tempfile::tempdir()?;
        let provider_dir = tempfile::tempdir()?;
        let (client, client_channels) = test_node(client_dir.path()).await?;
        let (provider, channels) = test_node(provider_dir.path()).await?;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let port = listener.local_addr()?.port();
        let backend = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await?;
            let mut raw = Vec::new();
            let (offset, length) = loop {
                let mut chunk = [0; 4096];
                let n = stream.read(&mut chunk).await?;
                ensure!(n > 0, "probe HTTP closed");
                raw.extend_from_slice(&chunk[..n]);
                let mut headers = [httparse::EMPTY_HEADER; 32];
                let mut request = httparse::Request::new(&mut headers);
                if let httparse::Status::Complete(offset) = request.parse(&raw)? {
                    let length = request
                        .headers
                        .iter()
                        .find(|h| h.name.eq_ignore_ascii_case("content-length"))
                        .context("length missing")?;
                    let length: usize = std::str::from_utf8(length.value)?.parse()?;
                    if raw.len() >= offset + length {
                        break (offset, length);
                    }
                }
            };
            let body: serde_json::Value = serde_json::from_slice(&raw[offset..offset + length])?;
            assert_eq!(body["max_tokens"], 32);
            let prompt = body["messages"][0]["content"].as_str().unwrap();
            let numbers = prompt
                .split_whitespace()
                .filter_map(|word| word.trim_end_matches('?').parse::<u16>().ok())
                .collect::<Vec<_>>();
            assert_eq!(numbers.len(), 2);
            let response = serde_json::json!({"choices":[{"message":{"content":(numbers[0]+numbers[1]).to_string()}}]}).to_string();
            stream
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        response.len(),
                        response
                    )
                    .as_bytes(),
                )
                .await?;
            anyhow::Ok(())
        });
        let mut targets = ModelTargets::default();
        targets
            .targets
            .insert("test".into(), vec![InferenceTarget::Local(port)]);
        let (_tx, rx) = tokio::sync::watch::channel(targets);
        let manager = crate::network::tunnel::Manager::start(
            provider.clone(),
            channels.rpc,
            channels.http,
            channels.stage,
        )
        .await?;
        manager.set_http_ingress(rx, crate::network::affinity::AffinityRouter::new());
        provider
            .payment_service()
            .await?
            .ledger
            .set_vetting_policy(&mesh_llm_payments::vetting::VettingPolicy {
                required: false,
                serve_probes: true,
                ttl_ms: 60_000,
            })?;
        client.start_accepting();
        provider.start_accepting();
        client
            .connect_to_peer(provider.endpoint_addr_for_advertisement())
            .await?;
        let service = client.payment_service().await?;
        service
            .ledger
            .set_vetting_policy(&mesh_llm_payments::vetting::VettingPolicy {
                required: true,
                serve_probes: false,
                ttl_ms: 60_000,
            })?;
        verify(&client, provider.id(), "test").await?;
        backend.await??;
        verify(&client, provider.id(), "test").await?;
        assert!(!service.has_wallet());
        assert!(service.ledger.requests()?.is_empty());
        client.endpoint.close().await;
        provider.endpoint.close().await;
        drop(client_channels);
        Ok(())
    }

    async fn test_node(directory: &std::path::Path) -> Result<(Node, crate::mesh::TunnelChannels)> {
        Node::start(
            crate::mesh::NodeRole::Client,
            crate::mesh::RelayConfig {
                urls: &[],
                auths: &std::collections::HashMap::new(),
                policy: crate::mesh::RelayPolicy::Disabled,
            },
            crate::mesh::QuicBindSelection {
                ip: Some("127.0.0.1".parse()?),
                port: Some(0),
            },
            Some(0.0),
            false,
            None,
            Some(&directory.join("config.toml")),
            crate::MeshRequirements::unrestricted(),
        )
        .await
    }

    // The limiter maps are process-wide, so every test that drives them is serialized;
    // a paused clock in one test would otherwise expire another test's entries.
    fn forget_peer(peer: iroh::EndpointId) {
        FAILED_PROBES.lock().unwrap().remove(&peer);
        PEER_PROBES.lock().unwrap().remove(&peer);
    }

    // A client with required vetting and no reachable provider.
    async fn vetting_client(
        directory: &std::path::Path,
    ) -> Result<(
        Node,
        std::sync::Arc<mesh_llm_payments::service::PaymentService>,
    )> {
        let client = Node::new_for_tests(crate::mesh::NodeRole::Client).await?;
        let service =
            std::sync::Arc::new(mesh_llm_payments::service::PaymentService::open(directory)?);
        service
            .ledger
            .set_vetting_policy(&mesh_llm_payments::vetting::VettingPolicy {
                required: true,
                serve_probes: false,
                ttl_ms: 60_000,
            })?;
        client
            .payments
            .set(service.clone())
            .map_err(|_| anyhow::anyhow!("already initialized"))?;
        Ok((client, service))
    }

    // Steady traffic kept the old cooldown sliding: each locally refused call went
    // through the failure handler and rewrote the timestamp, so a busy client never
    // became eligible again. Every call below is refused before any I/O, so the paused
    // clock only moves where the test advances it.
    #[tokio::test(start_paused = true)]
    #[serial_test::serial(vetting_limiter)]
    async fn continuous_traffic_during_cooldown_does_not_postpone_retry() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let (client, _service) = vetting_client(directory.path()).await?;
        let peer = iroh::SecretKey::generate().public();
        record_probe_failure(peer);
        for _ in 0..(PEER_PROBE_RESERVATION.as_secs() - 1) {
            tokio::time::advance(Duration::from_secs(1)).await;
            let error = verify(&client, peer, "test")
                .await
                .expect_err("cooldown must hold for the full window");
            assert!(
                error.to_string().contains("retry cooldown"),
                "unexpected refusal: {error}"
            );
        }
        tokio::time::advance(Duration::from_secs(2)).await;
        check_retry_cooldown(peer)
            .expect("eligible one window after the failure, despite continuous traffic");
        forget_peer(peer);
        client.endpoint.close().await;
        Ok(())
    }

    // A client that waits out its own cooldown must be eligible at the provider too.
    #[tokio::test(start_paused = true)]
    #[serial_test::serial(vetting_limiter)]
    async fn retry_eligibility_matches_the_provider_reservation() -> Result<()> {
        let peer = iroh::SecretKey::generate().public();
        reserve_peer_probe(peer)?;
        record_probe_failure(peer);
        tokio::time::advance(PEER_PROBE_RESERVATION - Duration::from_secs(1)).await;
        assert!(check_retry_cooldown(peer).is_err());
        assert!(reserve_peer_probe(peer).is_err(), "provider still reserved");
        tokio::time::advance(Duration::from_secs(2)).await;
        check_retry_cooldown(peer)?;
        reserve_peer_probe(peer)?;
        forget_peer(peer);
        Ok(())
    }

    #[tokio::test(start_paused = true)]
    #[serial_test::serial(vetting_limiter)]
    async fn a_new_probe_failure_starts_a_new_cooldown() -> Result<()> {
        let peer = iroh::SecretKey::generate().public();
        record_probe_failure(peer);
        tokio::time::advance(PEER_PROBE_RESERVATION + Duration::from_secs(1)).await;
        check_retry_cooldown(peer)?;
        record_probe_failure(peer);
        assert!(check_retry_cooldown(peer).is_err());
        tokio::time::advance(PEER_PROBE_RESERVATION - Duration::from_secs(1)).await;
        assert!(check_retry_cooldown(peer).is_err());
        tokio::time::advance(Duration::from_secs(2)).await;
        check_retry_cooldown(peer)?;
        forget_peer(peer);
        Ok(())
    }

    // Waiting behind another caller is not evidence about the provider.
    #[tokio::test(start_paused = true)]
    #[serial_test::serial(vetting_limiter)]
    async fn queue_wait_timeout_does_not_arm_a_cooldown() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let (client, _service) = vetting_client(directory.path()).await?;
        let provider = iroh::SecretKey::generate().public();
        let held = CLIENT_PROBES.lock().await;
        // No peer connection is ever attempted: the caller only ever waits for the lock.
        assert!(verify(&client, provider, "test").await.is_err());
        drop(held);
        check_retry_cooldown(provider).expect("a queue wait must not arm the cooldown");
        forget_peer(provider);
        client.endpoint.close().await;
        Ok(())
    }

    #[test]
    #[serial_test::serial(vetting_limiter)]
    fn peer_limits_and_failure_cooldown_are_bounded_and_isolated() -> Result<()> {
        let first = iroh::SecretKey::generate().public();
        let other = iroh::SecretKey::generate().public();
        reserve_peer_probe(first)?;
        assert!(reserve_peer_probe(first).is_err());
        reserve_peer_probe(other)?;
        check_retry_cooldown(first)?;
        FAILED_PROBES.lock().unwrap().insert(first, Instant::now());
        assert!(check_retry_cooldown(first).is_err());
        check_retry_cooldown(other)?;
        FAILED_PROBES.lock().unwrap().remove(&first);
        PEER_PROBES.lock().unwrap().remove(&first);
        PEER_PROBES.lock().unwrap().remove(&other);
        Ok(())
    }

    #[test]
    fn wrong_answer_replay_and_version_are_rejected() {
        let challenge = Challenge {
            version: CHALLENGE_VERSION,
            model: "m".into(),
            nonce: uuid::Uuid::new_v4().to_string(),
            left: 3,
            right: 8,
        };
        let mut answer = Answer {
            version: CHALLENGE_VERSION,
            nonce: challenge.nonce.clone(),
            text: "11".into(),
        };
        assert!(validate_answer(&challenge, &answer).is_ok());
        answer.text = "12".into();
        assert!(validate_answer(&challenge, &answer).is_err());
        answer.text = "11".into();
        answer.nonce = uuid::Uuid::new_v4().to_string();
        assert!(validate_answer(&challenge, &answer).is_err());
        answer.nonce = challenge.nonce.clone();
        answer.version += 1;
        assert!(validate_answer(&challenge, &answer).is_err());
    }

    #[tokio::test]
    async fn provider_opt_out_negotiates_without_backend_or_wallet() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let node = Node::new_for_tests(crate::mesh::NodeRole::Client).await?;
        let service = std::sync::Arc::new(mesh_llm_payments::service::PaymentService::open(
            directory.path(),
        )?);
        node.payments
            .set(service.clone())
            .map_err(|_| anyhow::anyhow!("already set"))?;
        let (mut caller, server) = tokio::io::duplex(4096);
        let (read, write) = tokio::io::split(server);
        let result = serve(
            iroh::SecretKey::generate().public(),
            &node,
            read,
            write,
            ModelTargets::default(),
        )
        .await;
        assert!(result.is_err());
        let capability: Capability = read_json(&mut caller).await?;
        assert!(!capability.supported);
        assert!(!service.has_wallet());
        node.endpoint.close().await;
        Ok(())
    }

    #[tokio::test]
    async fn probe_frame_limit_is_checked_before_allocation() -> Result<()> {
        let (mut send, mut recv) = tokio::io::duplex(16);
        send.write_u32((MAX_FRAME + 1) as u32).await?;
        assert!(read_json::<Challenge>(&mut recv).await.is_err());
        Ok(())
    }
}
