//! Prompt-free TEE challenge on the authenticated Mesh connection.
//!
//! The wire exchange is hardware-neutral. Intel TDX on a dstack guest is the
//! first producer and verifier; an unsupported peer is never treated as TEE.

use super::Node;
use crate::protocol::STREAM_TEE_ATTESTATION;
use anyhow::{Context, Result, bail, ensure};
use dcap_qvl::collateral::{CollateralClient, INTEL_PCS_URL};
use iroh::EndpointId;
use rand::RngExt;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256, Sha384};
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const TDX_FORMAT_V1: &str = "mesh-tee-endpoint-v1";
const PLATFORM_FORMAT_V2: &str = "mesh-tee-endpoint-v2";
const REPORT_DATA_DOMAIN: &[u8] = b"mesh-tee-endpoint-v1\0";
const MAX_EVIDENCE_BYTES: usize = 2 * 1024 * 1024;
const MAX_QUOTE_BYTES: usize = 64 * 1024;
const CHALLENGE_TIMEOUT: Duration = Duration::from_secs(20);

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TdxPolicyFile {
    mr_td: String,
    rt_mr0: String,
    rt_mr1: String,
    rt_mr2: String,
    rt_mr3: String,
    app_compose_sha256: String,
}

/// Exact, client-owned workload measurements. Never sourced from the peer.
pub(crate) struct TdxPolicy {
    mr_td: [u8; 48],
    rt_mr0: [u8; 48],
    rt_mr1: [u8; 48],
    rt_mr2: [u8; 48],
    rt_mr3: [u8; 48],
    app_compose_sha256: [u8; 32],
}

/// The client chooses the verifier. A peer cannot select a weaker platform
/// policy by changing the evidence format it returns.
pub(crate) enum ClientTeePolicy {
    IntelTdxDstack(TdxPolicy),
}

impl ClientTeePolicy {
    pub(crate) fn configured() -> Result<Option<Self>> {
        let Some(path) = std::env::var_os("MESH_TEE_TDX_POLICY") else {
            return Ok(None);
        };
        Ok(Some(Self::IntelTdxDstack(TdxPolicy::load(Path::new(
            &path,
        ))?)))
    }
}

/// Only a successful, platform-specific verifier may construct this result.
pub(crate) struct VerifiedTeePeer {
    pub(crate) endpoint_id: EndpointId,
    pub(crate) platform: &'static str,
}

impl TdxPolicy {
    pub(crate) fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read(path).context("read TDX policy")?;
        ensure!(content.len() <= 4096, "TDX policy is too large");
        let fields: TdxPolicyFile = serde_json::from_slice(&content).context("parse TDX policy")?;
        Ok(Self {
            mr_td: fixed_hex(&fields.mr_td, "mr_td")?,
            rt_mr0: fixed_hex(&fields.rt_mr0, "rt_mr0")?,
            rt_mr1: fixed_hex(&fields.rt_mr1, "rt_mr1")?,
            rt_mr2: fixed_hex(&fields.rt_mr2, "rt_mr2")?,
            rt_mr3: fixed_hex(&fields.rt_mr3, "rt_mr3")?,
            app_compose_sha256: fixed_hex(&fields.app_compose_sha256, "app_compose_sha256")?,
        })
    }
}

fn fixed_hex<const N: usize>(input: &str, field: &str) -> Result<[u8; N]> {
    ensure!(input.len() == N * 2, "{field} must be {N} bytes of hex");
    let bytes = hex::decode(input).with_context(|| format!("invalid {field} hex"))?;
    Ok(bytes.try_into().expect("validated exact byte length"))
}

fn report_data(nonce: &[u8; 32], peer: EndpointId) -> [u8; 64] {
    let mut digest = Sha256::new();
    digest.update(REPORT_DATA_DOMAIN);
    digest.update(nonce);
    digest.update(peer.as_bytes());
    let mut result = [0u8; 64];
    result[..32].copy_from_slice(&digest.finalize());
    result
}

#[derive(Serialize, Deserialize)]
struct Challenge {
    version: u8,
    nonce: String,
}

#[derive(Serialize, Deserialize)]
struct TdxEvidenceV1 {
    format: String,
    endpoint_id: String,
    nonce: String,
    quote: String,
    event_log: Value,
    compose_hash: String,
    app_compose: String,
}

/// New platform adapters use a typed, bounded outer envelope. The payload is
/// never trusted until the verifier selected by the client policy appraises it.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PlatformEvidenceV2 {
    format: String,
    platform: String,
    endpoint_id: String,
    nonce: String,
    payload: Value,
}

enum PeerEvidence {
    IntelTdxV1(TdxEvidenceV1),
    PlatformV2(PlatformEvidenceV2),
}

fn parse_peer_evidence(body: &[u8]) -> Result<PeerEvidence> {
    let value: Value = serde_json::from_slice(body).context("parse TEE evidence")?;
    let format = value
        .get("format")
        .and_then(Value::as_str)
        .context("TEE evidence has no format")?;
    match format {
        TDX_FORMAT_V1 => Ok(PeerEvidence::IntelTdxV1(serde_json::from_value(value)?)),
        PLATFORM_FORMAT_V2 => {
            let evidence: PlatformEvidenceV2 = serde_json::from_value(value)?;
            ensure!(
                !evidence.platform.is_empty()
                    && evidence.platform.len() <= 64
                    && evidence.platform.bytes().all(|byte| {
                        byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-'
                    }),
                "invalid TEE platform identifier"
            );
            ensure!(evidence.payload.is_object(), "invalid TEE platform payload");
            Ok(PeerEvidence::PlatformV2(evidence))
        }
        _ => bail!("unknown TEE evidence format"),
    }
}

fn verify_selected_identity(
    nonce: &str,
    endpoint_id: &str,
    expected_nonce: &[u8; 32],
    peer: EndpointId,
) -> Result<()> {
    ensure!(
        fixed_hex::<32>(nonce, "nonce")? == *expected_nonce,
        "wrong TEE nonce"
    );
    ensure!(
        fixed_hex::<32>(endpoint_id, "endpoint_id")? == *peer.as_bytes(),
        "attested peer differs from selected peer"
    );
    Ok(())
}

async fn read_bounded(recv: &mut iroh::endpoint::RecvStream, max: usize) -> Result<Vec<u8>> {
    let mut length = [0u8; 4];
    recv.read_exact(&mut length).await?;
    let length = u32::from_le_bytes(length) as usize;
    ensure!(length <= max, "TEE frame exceeds {max} bytes");
    let mut body = vec![0u8; length];
    recv.read_exact(&mut body).await?;
    Ok(body)
}

async fn write_bounded(send: &mut iroh::endpoint::SendStream, body: &[u8]) -> Result<()> {
    ensure!(
        body.len() <= MAX_EVIDENCE_BYTES,
        "TEE evidence is too large"
    );
    send.write_all(&(body.len() as u32).to_le_bytes()).await?;
    send.write_all(body).await?;
    Ok(())
}

impl Node {
    pub(crate) fn spawn_tee_attestation_stream(
        &self,
        send: iroh::endpoint::SendStream,
        recv: iroh::endpoint::RecvStream,
    ) {
        let node = self.clone();
        tokio::spawn(async move {
            if let Err(error) = node.handle_tee_attestation_stream(send, recv).await {
                tracing::debug!(%error, "TEE attestation stream failed");
            }
        });
    }

    async fn handle_tee_attestation_stream(
        &self,
        mut send: iroh::endpoint::SendStream,
        mut recv: iroh::endpoint::RecvStream,
    ) -> Result<()> {
        // A persistent key imported from the host would defeat the key-to-TEE
        // binding. The measured deployment must opt into in-guest key creation.
        ensure!(
            std::env::var("MESH_LLM_EPHEMERAL_KEY").is_ok(),
            "TEE serving requires an ephemeral in-guest identity"
        );
        let body = tokio::time::timeout(Duration::from_secs(5), read_bounded(&mut recv, 256))
            .await
            .context("TEE challenge timed out")??;
        let challenge: Challenge = serde_json::from_slice(&body)?;
        ensure!(challenge.version == 1, "unsupported TEE challenge version");
        let nonce = fixed_hex::<32>(&challenge.nonce, "nonce")?;
        let evidence = dstack_evidence(self.id(), &nonce).await?;
        write_bounded(&mut send, &serde_json::to_vec(&evidence)?).await?;
        send.finish()?;
        Ok(())
    }

    /// Verify one peer before sending prompt bytes to it. A fresh challenge is
    /// issued on the same authenticated iroh identity later used for routing.
    pub(crate) async fn verify_tee_peer(
        &self,
        peer: EndpointId,
        policy: &ClientTeePolicy,
    ) -> Result<VerifiedTeePeer> {
        tokio::time::timeout(CHALLENGE_TIMEOUT, async {
            let mut nonce = [0u8; 32];
            rand::rng().fill(&mut nonce);
            let connection = self.connection_to_peer(peer).await?;
            ensure!(connection.remote_id() == peer, "iroh peer identity changed");
            let (mut send, mut recv) = connection.open_bi().await?;
            send.write_all(&[STREAM_TEE_ATTESTATION]).await?;
            let challenge = serde_json::to_vec(&Challenge {
                version: 1,
                nonce: hex::encode(nonce),
            })?;
            write_bounded(&mut send, &challenge).await?;
            send.finish()?;
            let body = read_bounded(&mut recv, MAX_EVIDENCE_BYTES).await?;
            appraise_peer_evidence(parse_peer_evidence(&body)?, &nonce, peer, policy).await
        })
        .await
        .context("TEE challenge or verification timed out")?
    }
}

async fn appraise_peer_evidence(
    evidence: PeerEvidence,
    nonce: &[u8; 32],
    peer: EndpointId,
    policy: &ClientTeePolicy,
) -> Result<VerifiedTeePeer> {
    match (policy, evidence) {
        (ClientTeePolicy::IntelTdxDstack(policy), PeerEvidence::IntelTdxV1(evidence)) => {
            verify_tdx_evidence(&evidence, nonce, peer, policy).await?;
            Ok(VerifiedTeePeer {
                endpoint_id: peer,
                platform: "intel-tdx-dstack",
            })
        }
        (ClientTeePolicy::IntelTdxDstack(_), PeerEvidence::PlatformV2(evidence)) => {
            ensure!(
                evidence.format == PLATFORM_FORMAT_V2,
                "unknown TEE evidence format"
            );
            verify_selected_identity(&evidence.nonce, &evidence.endpoint_id, nonce, peer)?;
            bail!("client TDX policy rejects platform {}", evidence.platform)
        }
    }
}

#[cfg(unix)]
async fn dstack_evidence(peer: EndpointId, nonce: &[u8; 32]) -> Result<TdxEvidenceV1> {
    let socket = std::env::var_os("MESH_TEE_DSTACK_SOCKET")
        .context("MESH_TEE_DSTACK_SOCKET is not configured")?;
    let client = reqwest::Client::builder()
        .unix_socket(Path::new(&socket))
        .timeout(Duration::from_secs(15))
        .build()?;
    let report_input = report_data(nonce, peer);
    let quote = dstack_request(
        &client,
        "/GetQuote",
        Some(json!({ "report_data": hex::encode(&report_input[..32]) })),
    )
    .await?;
    let info = dstack_request(&client, "/Info", None).await?;
    let tcb_info = match info.get("tcb_info") {
        Some(Value::String(raw)) => serde_json::from_str::<Value>(raw)?,
        Some(value) => value.clone(),
        None => bail!("dstack Info omitted tcb_info"),
    };
    let field = |value: &Value, name: &str| -> Result<String> {
        value
            .get(name)
            .and_then(Value::as_str)
            .map(str::to_owned)
            .with_context(|| format!("dstack response omitted {name}"))
    };
    Ok(TdxEvidenceV1 {
        format: TDX_FORMAT_V1.to_owned(),
        endpoint_id: hex::encode(peer.as_bytes()),
        nonce: hex::encode(nonce),
        quote: field(&quote, "quote")?,
        event_log: quote
            .get("event_log")
            .cloned()
            .context("dstack quote omitted event_log")?,
        compose_hash: field(&info, "compose_hash")?,
        app_compose: field(&tcb_info, "app_compose")?,
    })
}

#[cfg(not(unix))]
async fn dstack_evidence(_peer: EndpointId, _nonce: &[u8; 32]) -> Result<TdxEvidenceV1> {
    bail!("dstack quote generation requires Unix")
}

#[cfg(unix)]
async fn dstack_request(
    client: &reqwest::Client,
    path: &str,
    body: Option<Value>,
) -> Result<Value> {
    let request = if let Some(body) = body {
        client.post(format!("http://dstack{path}")).json(&body)
    } else {
        client.get(format!("http://dstack{path}"))
    };
    let mut response = request.send().await?.error_for_status()?;
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await? {
        ensure!(
            bytes.len().saturating_add(chunk.len()) <= MAX_EVIDENCE_BYTES,
            "dstack response is too large"
        );
        bytes.extend_from_slice(&chunk);
    }
    Ok(serde_json::from_slice(&bytes)?)
}

async fn verify_tdx_evidence(
    evidence: &TdxEvidenceV1,
    nonce: &[u8; 32],
    peer: EndpointId,
    policy: &TdxPolicy,
) -> Result<()> {
    ensure!(
        evidence.format == TDX_FORMAT_V1,
        "unknown TEE evidence format"
    );
    verify_selected_identity(&evidence.nonce, &evidence.endpoint_id, nonce, peer)?;
    ensure!(
        evidence.quote.len() <= MAX_QUOTE_BYTES * 2,
        "quote is too large"
    );
    let quote = hex::decode(&evidence.quote).context("quote is not hex")?;
    let collateral = CollateralClient::with_default_http(INTEL_PCS_URL)?
        .fetch(&quote)
        .await
        .context("fetch Intel-signed collateral")?;
    let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let verdict = dcap_qvl::verify::verify(&quote, &collateral, now)
        .context("verify Intel TDX quote and collateral")?;
    ensure!(
        verdict.status == "UpToDate" && verdict.advisory_ids.is_empty(),
        "TDX TCB is not current and advisory-free"
    );
    let report = verdict.report.as_td10().context("quote is not Intel TDX")?;
    ensure!(
        report.td_attributes[0] & 1 == 0,
        "TDX debug mode is enabled"
    );
    ensure!(
        report.report_data == report_data(nonce, peer),
        "quote does not bind iroh peer"
    );
    ensure!(report.mr_td == policy.mr_td, "unapproved TDX mr_td");
    ensure!(report.rt_mr0 == policy.rt_mr0, "unapproved TDX rt_mr0");
    ensure!(report.rt_mr1 == policy.rt_mr1, "unapproved TDX rt_mr1");
    ensure!(report.rt_mr2 == policy.rt_mr2, "unapproved TDX rt_mr2");
    ensure!(report.rt_mr3 == policy.rt_mr3, "unapproved TDX rt_mr3");
    let compose_hash = Sha256::digest(evidence.app_compose.as_bytes());
    ensure!(
        compose_hash.as_slice() == policy.app_compose_sha256,
        "unapproved application Compose"
    );
    ensure!(
        fixed_hex::<32>(&evidence.compose_hash, "compose_hash")? == policy.app_compose_sha256,
        "dstack Compose hash differs from policy"
    );
    ensure!(
        replay_rtmr3(&evidence.event_log, &policy.app_compose_sha256)? == report.rt_mr3,
        "runtime event log differs from quoted RTMR3"
    );
    Ok(())
}

fn replay_rtmr3(log: &Value, expected_compose: &[u8; 32]) -> Result<[u8; 48]> {
    let parsed;
    let events = if let Some(raw) = log.as_str() {
        parsed = serde_json::from_str::<Value>(raw)?;
        parsed
            .as_array()
            .context("runtime event log is not an array")?
    } else {
        log.as_array()
            .context("runtime event log is not an array")?
    };
    ensure!(events.len() <= 256, "runtime event log is too large");
    let mut register = [0u8; 48];
    let mut compose_events = 0;
    for event in events {
        let imr = event
            .get("imr")
            .and_then(Value::as_u64)
            .context("runtime event has invalid register")?;
        if imr != 3 {
            continue;
        }
        let version_ok = event
            .get("version")
            .is_none_or(|value| value.as_u64() == Some(1));
        ensure!(
            event.get("event_type").and_then(Value::as_u64) == Some(0x0800_0001) && version_ok,
            "unsupported RTMR3 event"
        );
        let name = event
            .get("event")
            .and_then(Value::as_str)
            .context("event name missing")?;
        let payload = event
            .get("event_payload")
            .and_then(Value::as_str)
            .context("event payload missing")?
            .trim_start_matches("0x");
        let payload = hex::decode(payload).context("event payload is not hex")?;
        if name == "compose-hash" {
            compose_events += 1;
            ensure!(
                payload == expected_compose,
                "measured Compose hash differs from policy"
            );
        }
        let mut hasher = Sha384::new();
        hasher.update(0x0800_0001u32.to_le_bytes());
        hasher.update(b":");
        hasher.update(name.as_bytes());
        hasher.update(b":");
        hasher.update(&payload);
        let digest = hasher.finalize();
        if let Some(advertised) = event.get("digest") {
            let advertised = advertised
                .as_str()
                .context("runtime event digest is not a string")?;
            if !advertised.is_empty() {
                ensure!(
                    hex::decode(advertised.trim_start_matches("0x"))? == digest.as_slice(),
                    "runtime event digest mismatch"
                );
            }
        }
        register = Sha384::digest([register.as_slice(), digest.as_slice()].concat()).into();
    }
    ensure!(compose_events == 1, "expected one Compose hash event");
    Ok(register)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_data_binds_both_nonce_and_endpoint() {
        let first = iroh::SecretKey::generate().public();
        let second = iroh::SecretKey::generate().public();
        let nonce = [7u8; 32];
        let baseline = report_data(&nonce, first);
        assert_ne!(baseline, report_data(&[8u8; 32], first));
        assert_ne!(baseline, report_data(&nonce, second));
        assert_eq!(&baseline[32..], &[0u8; 32]);
    }

    #[test]
    fn rtmr3_replay_rejects_changed_or_duplicate_compose() {
        let approved = [9u8; 32];
        let event = json!({
            "imr": 3,
            "event_type": 0x0800_0001,
            "version": 1,
            "event": "compose-hash",
            "event_payload": hex::encode(approved),
        });
        let digest = replay_rtmr3(&json!([event.clone()]), &approved).unwrap();
        assert_ne!(digest, [0u8; 48]);
        assert!(replay_rtmr3(&json!([event.clone()]), &[8u8; 32]).is_err());
        assert!(replay_rtmr3(&json!([event.clone(), event.clone()]), &approved).is_err());
        assert!(replay_rtmr3(&json!([{"imr": "3"}]), &approved).is_err());
        let mut invalid_version = event.clone();
        invalid_version["version"] = json!("1");
        assert!(replay_rtmr3(&json!([invalid_version]), &approved).is_err());
        let mut invalid_digest = event;
        invalid_digest["digest"] = json!(7);
        assert!(replay_rtmr3(&json!([invalid_digest]), &approved).is_err());
    }

    #[tokio::test]
    async fn rejects_replayed_or_swapped_evidence_before_collateral_lookup() {
        let peer = iroh::SecretKey::generate().public();
        let policy = TdxPolicy {
            mr_td: [0; 48],
            rt_mr0: [0; 48],
            rt_mr1: [0; 48],
            rt_mr2: [0; 48],
            rt_mr3: [0; 48],
            app_compose_sha256: [0; 32],
        };
        let mut evidence = TdxEvidenceV1 {
            format: TDX_FORMAT_V1.to_owned(),
            endpoint_id: hex::encode(peer.as_bytes()),
            nonce: hex::encode([1u8; 32]),
            quote: String::new(),
            event_log: json!([]),
            compose_hash: hex::encode([0u8; 32]),
            app_compose: String::new(),
        };
        assert!(
            verify_tdx_evidence(&evidence, &[2u8; 32], peer, &policy)
                .await
                .is_err()
        );
        evidence.nonce = hex::encode([2u8; 32]);
        evidence.endpoint_id = hex::encode([3u8; 32]);
        assert!(
            verify_tdx_evidence(&evidence, &[2u8; 32], peer, &policy)
                .await
                .is_err()
        );
        evidence.endpoint_id = hex::encode(peer.as_bytes());
        evidence.format = "unknown-platform".to_owned();
        assert!(
            verify_tdx_evidence(&evidence, &[2u8; 32], peer, &policy)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn client_tdx_policy_rejects_other_platform_before_hardware_lookup() {
        let peer = iroh::SecretKey::generate().public();
        let nonce = [5u8; 32];
        let policy = ClientTeePolicy::IntelTdxDstack(TdxPolicy {
            mr_td: [0; 48],
            rt_mr0: [0; 48],
            rt_mr1: [0; 48],
            rt_mr2: [0; 48],
            rt_mr3: [0; 48],
            app_compose_sha256: [0; 32],
        });
        let body = serde_json::to_vec(&PlatformEvidenceV2 {
            format: PLATFORM_FORMAT_V2.to_owned(),
            platform: "snp-direct".to_owned(),
            endpoint_id: hex::encode(peer.as_bytes()),
            nonce: hex::encode(nonce),
            payload: json!({ "report": "unverified" }),
        })
        .unwrap();
        let evidence = parse_peer_evidence(&body).unwrap();
        let error = appraise_peer_evidence(evidence, &nonce, peer, &policy)
            .await
            .err()
            .unwrap();
        assert!(error.to_string().contains("rejects platform snp-direct"));
    }

    #[test]
    fn evidence_parser_preserves_legacy_tdx_response() {
        let body = json!({
            "format": TDX_FORMAT_V1,
            "endpoint_id": hex::encode([1u8; 32]),
            "nonce": hex::encode([2u8; 32]),
            "quote": "",
            "event_log": [],
            "compose_hash": hex::encode([3u8; 32]),
            "app_compose": "",
        });
        let evidence = parse_peer_evidence(&serde_json::to_vec(&body).unwrap()).unwrap();
        assert!(matches!(evidence, PeerEvidence::IntelTdxV1(_)));
    }

    #[test]
    fn evidence_parser_rejects_unknown_and_malformed_v2_formats() {
        let unknown = json!({ "format": "mesh-tee-endpoint-v3" });
        assert!(parse_peer_evidence(&serde_json::to_vec(&unknown).unwrap()).is_err());

        let malformed = json!({
            "format": PLATFORM_FORMAT_V2,
            "platform": "snp-direct",
            "endpoint_id": hex::encode([0u8; 32]),
            "nonce": hex::encode([0u8; 32]),
            "payload": "not an object",
        });
        assert!(parse_peer_evidence(&serde_json::to_vec(&malformed).unwrap()).is_err());

        let invalid_platform = json!({
            "format": PLATFORM_FORMAT_V2,
            "platform": "snp-direct\nspoofed-log-line",
            "endpoint_id": hex::encode([0u8; 32]),
            "nonce": hex::encode([0u8; 32]),
            "payload": {},
        });
        assert!(parse_peer_evidence(&serde_json::to_vec(&invalid_platform).unwrap()).is_err());

        let extra_field = json!({
            "format": PLATFORM_FORMAT_V2,
            "platform": "snp-direct",
            "endpoint_id": hex::encode([0u8; 32]),
            "nonce": hex::encode([0u8; 32]),
            "payload": {},
            "unrecognized": true,
        });
        assert!(parse_peer_evidence(&serde_json::to_vec(&extra_field).unwrap()).is_err());
    }
}
