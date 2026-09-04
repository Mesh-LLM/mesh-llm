use super::*;
use serde::de::DeserializeOwned;
use sha2::Digest;
use tokio::sync::{oneshot, watch};

const PAIRING_OFFER_LIFETIME_SECS: u64 = 10 * 60;
const PAIRING_SESSION_LIFETIME_SECS: u64 = 2 * 60;
const PAIRING_RETENTION_SECS: u64 = 10 * 60;
const MAX_PAIRING_SESSIONS: usize = 64;
const MAX_PAIRING_FRAME_BYTES: usize = 64 * 1024;
const MAX_PAIRING_OFFER_ENCODED_BYTES: usize = 16 * 1024;
const MAX_PAIRING_NAME_CHARS: usize = 80;
const PAIRING_CODE_DOMAIN: &[u8] = b"mesh-llm-pairing-code-v1\0";

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PairingDirection {
    Incoming,
    Outgoing,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PairingSessionStatus {
    Connecting,
    AwaitingApproval,
    WaitingForPeer,
    Joining,
    Approved,
    Rejected,
    Cancelled,
    Expired,
    Failed,
}

impl PairingSessionStatus {
    fn terminal(self) -> bool {
        matches!(
            self,
            Self::Approved | Self::Rejected | Self::Cancelled | Self::Expired | Self::Failed
        )
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PairingDecision {
    Approve,
    Reject,
    Cancel,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PairingOffer {
    pub version: u8,
    pub offer_id: String,
    pub endpoint_addr: EndpointAddr,
    pub device_name: String,
    pub expires_at: u64,
}

impl PairingOffer {
    pub fn encode(&self) -> Result<String> {
        let body = serde_json::to_vec(self)?;
        Ok(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(body))
    }

    pub fn url(&self) -> Result<String> {
        Ok(format!("mesh-llm://pair/{}", self.encode()?))
    }

    pub fn decode(value: &str) -> Result<Self> {
        let value = value.trim();
        let encoded = value
            .strip_prefix("mesh-llm://pair/")
            .or_else(|| value.strip_prefix("mesh-llm://pair"))
            .unwrap_or(value)
            .trim_start_matches('/');
        anyhow::ensure!(!encoded.is_empty(), "pairing offer is empty");
        anyhow::ensure!(
            encoded.len() <= MAX_PAIRING_OFFER_ENCODED_BYTES,
            "pairing offer is too large"
        );
        let body = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(encoded)
            .context("pairing offer is not valid base64url")?;
        anyhow::ensure!(
            body.len() <= MAX_PAIRING_FRAME_BYTES,
            "pairing offer is too large"
        );
        let offer: Self = serde_json::from_slice(&body).context("pairing offer is not valid")?;
        anyhow::ensure!(offer.version == 1, "unsupported pairing offer version");
        anyhow::ensure!(
            uuid::Uuid::parse_str(&offer.offer_id).is_ok(),
            "pairing offer identifier is invalid"
        );
        anyhow::ensure!(offer.expires_at > now_secs(), "pairing offer has expired");
        Ok(offer)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct PairingOfferResponse {
    pub offer: String,
    pub url: String,
    pub expires_at: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct PairingSession {
    pub id: String,
    pub direction: PairingDirection,
    pub peer_name: String,
    pub peer_id: String,
    pub comparison_code: Option<String>,
    pub status: PairingSessionStatus,
    pub created_at: u64,
    pub expires_at: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

struct PairingSessionRecord {
    session: PairingSession,
    decision: Option<oneshot::Sender<PairingDecision>>,
    cancel: Option<watch::Sender<bool>>,
}

struct PairingOfferRecord {
    expires_at: u64,
    used: bool,
}

#[derive(Default)]
struct PairingState {
    offers: HashMap<String, PairingOfferRecord>,
    sessions: HashMap<String, PairingSessionRecord>,
}

#[derive(Clone, Default)]
pub(crate) struct PairingService {
    inner: Arc<Mutex<PairingState>>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum PairingWire {
    Hello {
        session_id: String,
        offer_id: String,
        device_name: String,
    },
    Ready {
        device_name: String,
        comparison_code: String,
        expires_at: u64,
    },
    Decision {
        decision: PairingDecision,
    },
    Cancel,
    Complete {
        approved: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        invite_token: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    JoinResult {
        joined: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
}

impl Node {
    pub async fn display_name(&self) -> String {
        self.display_name
            .lock()
            .await
            .clone()
            .filter(|name| !name.trim().is_empty())
            .unwrap_or_else(|| self.id().fmt_short().to_string())
    }

    pub async fn create_pairing_offer(&self) -> Result<PairingOfferResponse> {
        let now = now_secs();
        let offer = PairingOffer {
            version: 1,
            offer_id: uuid::Uuid::new_v4().to_string(),
            endpoint_addr: self.advertised_endpoint_addr(),
            device_name: bounded_device_name(&self.display_name().await),
            expires_at: now + PAIRING_OFFER_LIFETIME_SECS,
        };
        {
            let mut state = self.pairing.inner.lock().await;
            prune_pairing_state(&mut state, now);
            state.offers.insert(
                offer.offer_id.clone(),
                PairingOfferRecord {
                    expires_at: offer.expires_at,
                    used: false,
                },
            );
        }
        Ok(PairingOfferResponse {
            offer: offer.encode()?,
            url: offer.url()?,
            expires_at: offer.expires_at,
        })
    }

    pub async fn pairing_sessions(&self) -> Vec<PairingSession> {
        let now = now_secs();
        let mut state = self.pairing.inner.lock().await;
        prune_pairing_state(&mut state, now);
        let mut sessions = state
            .sessions
            .values()
            .map(|record| record.session.clone())
            .collect::<Vec<_>>();
        sessions.sort_by_key(|session| std::cmp::Reverse(session.created_at));
        sessions
    }

    pub async fn start_pairing(&self, encoded_offer: &str) -> Result<PairingSession> {
        let offer = PairingOffer::decode(encoded_offer)?;
        anyhow::ensure!(
            offer.endpoint_addr.id != self.id(),
            "cannot pair with this device"
        );
        let now = now_secs();
        let id = uuid::Uuid::new_v4().to_string();
        let expires_at = now + PAIRING_SESSION_LIFETIME_SECS;
        let (decision, decision_rx) = oneshot::channel();
        let (cancel, cancel_rx) = watch::channel(false);
        let session = PairingSession {
            id: id.clone(),
            direction: PairingDirection::Outgoing,
            peer_name: bounded_device_name(&offer.device_name),
            peer_id: hex::encode(offer.endpoint_addr.id.as_bytes()),
            comparison_code: None,
            status: PairingSessionStatus::Connecting,
            created_at: now,
            expires_at,
            error: None,
        };
        {
            let mut state = self.pairing.inner.lock().await;
            prune_pairing_state(&mut state, now);
            anyhow::ensure!(
                state.sessions.len() < MAX_PAIRING_SESSIONS,
                "too many active pairing sessions"
            );
            state.sessions.insert(
                id.clone(),
                PairingSessionRecord {
                    session: session.clone(),
                    decision: Some(decision),
                    cancel: Some(cancel),
                },
            );
        }
        let node = self.clone();
        tokio::spawn(async move {
            if let Err(error) = node
                .run_outgoing_pairing(offer, id.clone(), expires_at, decision_rx, cancel_rx)
                .await
            {
                node.fail_pairing_session(&id, error.to_string()).await;
            }
        });
        Ok(session)
    }

    pub async fn decide_pairing(
        &self,
        id: &str,
        decision: PairingDecision,
    ) -> Result<PairingSession> {
        let mut state = self.pairing.inner.lock().await;
        prune_pairing_state(&mut state, now_secs());
        let record = state
            .sessions
            .get_mut(id)
            .ok_or_else(|| anyhow::anyhow!("pairing session not found"))?;
        anyhow::ensure!(
            !record.session.status.terminal(),
            "pairing session is already complete"
        );
        if decision == PairingDecision::Cancel && record.decision.is_none() {
            let cancel = record
                .cancel
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("pairing session is no longer active"))?;
            cancel
                .send(true)
                .map_err(|_| anyhow::anyhow!("pairing session is no longer active"))?;
            record.session.status = PairingSessionStatus::Cancelled;
            return Ok(record.session.clone());
        }
        let sender = record
            .decision
            .take()
            .ok_or_else(|| anyhow::anyhow!("pairing decision was already submitted"))?;
        record.session.status = match decision {
            PairingDecision::Approve => PairingSessionStatus::WaitingForPeer,
            PairingDecision::Reject => PairingSessionStatus::Rejected,
            PairingDecision::Cancel => PairingSessionStatus::Cancelled,
        };
        sender
            .send(decision)
            .map_err(|_| anyhow::anyhow!("pairing session is no longer active"))?;
        Ok(record.session.clone())
    }

    // This is a linear, bounded protocol ceremony; keeping its transitions together makes the
    // bilateral approval and token-release boundary auditable.
    #[allow(clippy::cognitive_complexity, clippy::too_many_lines)]
    pub(crate) async fn handle_pairing_incoming(
        &self,
        connection: Connection,
        remote: EndpointId,
    ) -> Result<()> {
        let (mut send, mut recv) =
            tokio::time::timeout(std::time::Duration::from_secs(10), connection.accept_bi())
                .await
                .context("timed out waiting for pairing stream")??;
        let hello = read_pairing_wire(&mut recv).await?;
        let PairingWire::Hello {
            session_id,
            offer_id,
            device_name,
        } = hello
        else {
            anyhow::bail!("expected pairing hello")
        };
        anyhow::ensure!(
            uuid::Uuid::parse_str(&session_id).is_ok(),
            "invalid pairing session identifier"
        );
        let now = now_secs();
        let expires_at = now + PAIRING_SESSION_LIFETIME_SECS;
        let comparison_code = pairing_code(remote, self.id(), &session_id);
        let (local_decision_tx, mut local_decision_rx) = oneshot::channel();
        let (cancel_tx, mut cancel_rx) = watch::channel(false);
        {
            let mut state = self.pairing.inner.lock().await;
            prune_pairing_state(&mut state, now);
            anyhow::ensure!(
                state.sessions.len() < MAX_PAIRING_SESSIONS,
                "too many active pairing sessions"
            );
            anyhow::ensure!(
                !state.sessions.contains_key(&session_id),
                "pairing session was already used"
            );
            let offer = state
                .offers
                .get_mut(&offer_id)
                .ok_or_else(|| anyhow::anyhow!("pairing offer is unknown or expired"))?;
            anyhow::ensure!(!offer.used, "pairing offer was already used");
            anyhow::ensure!(offer.expires_at > now, "pairing offer has expired");
            offer.used = true;
            state.sessions.insert(
                session_id.clone(),
                PairingSessionRecord {
                    session: PairingSession {
                        id: session_id.clone(),
                        direction: PairingDirection::Incoming,
                        peer_name: bounded_device_name(&device_name),
                        peer_id: hex::encode(remote.as_bytes()),
                        comparison_code: Some(comparison_code.clone()),
                        status: PairingSessionStatus::AwaitingApproval,
                        created_at: now,
                        expires_at,
                        error: None,
                    },
                    decision: Some(local_decision_tx),
                    cancel: Some(cancel_tx),
                },
            );
        }
        write_pairing_wire(
            &mut send,
            &PairingWire::Ready {
                device_name: self.display_name().await,
                comparison_code,
                expires_at,
            },
        )
        .await?;

        let deadline = tokio::time::Instant::now()
            + std::time::Duration::from_secs(PAIRING_SESSION_LIFETIME_SECS);
        let mut peer_decision = None;
        let mut local_decision = None;
        loop {
            if matches!(
                peer_decision,
                Some(PairingDecision::Reject | PairingDecision::Cancel)
            ) || matches!(
                local_decision,
                Some(PairingDecision::Reject | PairingDecision::Cancel)
            ) || (peer_decision == Some(PairingDecision::Approve)
                && local_decision == Some(PairingDecision::Approve))
            {
                break;
            }
            tokio::select! {
                result = read_pairing_wire(&mut recv) => {
                    match result? {
                        PairingWire::Decision { decision } if peer_decision.is_none() => {
                            peer_decision = Some(decision);
                        }
                        PairingWire::Cancel => peer_decision = Some(PairingDecision::Cancel),
                        _ => anyhow::bail!("expected peer pairing decision"),
                    }
                }
                result = &mut local_decision_rx, if local_decision.is_none() => {
                    local_decision = Some(result.context("local pairing decision was cancelled")?);
                }
                result = cancel_rx.changed(), if local_decision == Some(PairingDecision::Approve) => {
                    result.context("local pairing cancellation was dropped")?;
                    if *cancel_rx.borrow() {
                        local_decision = Some(PairingDecision::Cancel);
                    }
                }
                _ = tokio::time::sleep_until(deadline) => {
                    self.set_pairing_status(&session_id, PairingSessionStatus::Expired, None).await;
                    write_pairing_wire(&mut send, &PairingWire::Complete {
                        approved: false,
                        invite_token: None,
                        reason: Some("expired".to_string()),
                    }).await?;
                    finish_pairing_send(&mut send).await?;
                    return Ok(());
                }
            }
        }

        let approved = peer_decision == Some(PairingDecision::Approve)
            && local_decision == Some(PairingDecision::Approve);
        if approved {
            let invite_token = self.invite_token().await;
            anyhow::ensure!(
                !invite_token.is_empty(),
                "this mesh cannot create an invite token"
            );
            write_pairing_wire(
                &mut send,
                &PairingWire::Complete {
                    approved: true,
                    invite_token: Some(invite_token),
                    reason: None,
                },
            )
            .await?;
            let result: JoinWait<std::convert::Infallible, PairingWire> =
                tokio::time::timeout_at(deadline, async {
                tokio::select! {
                    wire = read_pairing_wire(&mut recv) => Ok::<_, anyhow::Error>(JoinWait::PeerMessage(wire?)),
                    changed = cancel_rx.changed() => {
                        changed.context("local pairing cancellation was dropped")?;
                        Ok(JoinWait::LocalCancelled)
                    }
                }
                })
                .await
                .map_err(|_| anyhow::anyhow!("pairing session expired"))??;
            let (joined, reason) = match result {
                JoinWait::PeerMessage(PairingWire::JoinResult { joined, reason }) => {
                    (joined, reason)
                }
                JoinWait::PeerMessage(PairingWire::Cancel) => {
                    acknowledge_pairing_cancel(&mut send).await?;
                    self.set_pairing_status(&session_id, PairingSessionStatus::Cancelled, None)
                        .await;
                    return Ok(());
                }
                JoinWait::LocalCancelled => {
                    send_pairing_cancel(&mut send, &mut recv).await?;
                    self.set_pairing_status(&session_id, PairingSessionStatus::Cancelled, None)
                        .await;
                    return Ok(());
                }
                JoinWait::PeerMessage(_) => anyhow::bail!("expected pairing join result"),
                JoinWait::Completed(never) => match never {},
            };
            if joined {
                self.set_pairing_status(&session_id, PairingSessionStatus::Approved, None)
                    .await;
            } else if reason.as_deref() == Some("cancelled") {
                self.set_pairing_status(&session_id, PairingSessionStatus::Cancelled, None)
                    .await;
            } else {
                self.set_pairing_status(
                    &session_id,
                    PairingSessionStatus::Failed,
                    Some(
                        reason.unwrap_or_else(|| "Mesh admission rejected the device".to_string()),
                    ),
                )
                .await;
            }
        } else {
            let status = if matches!(
                peer_decision.or(local_decision),
                Some(PairingDecision::Cancel)
            ) {
                PairingSessionStatus::Cancelled
            } else {
                PairingSessionStatus::Rejected
            };
            write_pairing_wire(
                &mut send,
                &PairingWire::Complete {
                    approved: false,
                    invite_token: None,
                    reason: Some(
                        match status {
                            PairingSessionStatus::Cancelled => "cancelled",
                            _ => "rejected",
                        }
                        .to_string(),
                    ),
                },
            )
            .await?;
            finish_pairing_send(&mut send).await?;
            self.set_pairing_status(&session_id, status, None).await;
        }
        Ok(())
    }

    // Keep the initiator's protocol transitions adjacent to the corresponding wire operations.
    #[allow(clippy::cognitive_complexity)]
    async fn run_outgoing_pairing(
        &self,
        offer: PairingOffer,
        session_id: String,
        expires_at: u64,
        mut decision_rx: oneshot::Receiver<PairingDecision>,
        mut cancel_rx: watch::Receiver<bool>,
    ) -> Result<()> {
        let connection = self
            .endpoint
            .connect(offer.endpoint_addr.clone(), ALPN_PAIRING_V1)
            .await
            .context("could not reach the other device")?;
        anyhow::ensure!(
            connection.remote_id() == offer.endpoint_addr.id,
            "pairing endpoint identity mismatch"
        );
        let (mut send, mut recv) = connection.open_bi().await?;
        write_pairing_wire(
            &mut send,
            &PairingWire::Hello {
                session_id: session_id.clone(),
                offer_id: offer.offer_id,
                device_name: self.display_name().await,
            },
        )
        .await?;
        let PairingWire::Ready {
            device_name,
            comparison_code,
            expires_at: remote_expires_at,
        } = read_pairing_wire(&mut recv).await?
        else {
            anyhow::bail!("expected pairing readiness response")
        };
        let expected_code = pairing_code(self.id(), connection.remote_id(), &session_id);
        anyhow::ensure!(
            comparison_code == expected_code,
            "pairing comparison code mismatch"
        );
        {
            let mut state = self.pairing.inner.lock().await;
            let record = state
                .sessions
                .get_mut(&session_id)
                .ok_or_else(|| anyhow::anyhow!("pairing session disappeared"))?;
            record.session.peer_name = bounded_device_name(&device_name);
            record.session.comparison_code = Some(comparison_code);
            record.session.expires_at = expires_at.min(remote_expires_at);
            record.session.status = PairingSessionStatus::AwaitingApproval;
        }

        let remaining = expires_at.saturating_sub(now_secs());
        let first = tokio::time::timeout(std::time::Duration::from_secs(remaining), async {
            tokio::select! {
                decision = &mut decision_rx => Ok::<_, anyhow::Error>(Either::Left(
                    decision.context("local pairing decision was cancelled")?
                )),
                message = read_pairing_wire(&mut recv) => Ok(Either::Right(message?)),
            }
        })
        .await
        .map_err(|_| anyhow::anyhow!("pairing session expired"))??;
        let decision = match first {
            Either::Left(decision) => decision,
            Either::Right(PairingWire::Complete {
                approved: false,
                reason,
                ..
            }) => {
                let status = completion_rejection_status(reason.as_deref());
                self.set_pairing_status(&session_id, status, None).await;
                return Ok(());
            }
            Either::Right(_) => {
                anyhow::bail!("unexpected pairing completion before local decision")
            }
        };
        write_pairing_wire(&mut send, &PairingWire::Decision { decision }).await?;
        if decision != PairingDecision::Approve {
            send.finish()?;
            let PairingWire::Complete {
                approved: false, ..
            } = tokio::time::timeout(
                std::time::Duration::from_secs(3),
                read_pairing_wire(&mut recv),
            )
            .await
            .map_err(|_| anyhow::anyhow!("timed out confirming pairing rejection"))??
            else {
                anyhow::bail!("expected pairing rejection confirmation")
            };
            return Ok(());
        }
        self.set_pairing_status(&session_id, PairingSessionStatus::WaitingForPeer, None)
            .await;
        let completion = tokio::time::timeout(
            std::time::Duration::from_secs(expires_at.saturating_sub(now_secs())),
            async {
                tokio::select! {
                    wire = read_pairing_wire(&mut recv) => Ok::<_, anyhow::Error>(Either::Left(wire?)),
                    changed = cancel_rx.changed() => {
                        changed.context("local pairing cancellation was dropped")?;
                        Ok(Either::Right(()))
                    }
                }
            },
        )
        .await
        .map_err(|_| anyhow::anyhow!("pairing session expired"))??;
        let (approved, invite_token, reason) = match completion {
            Either::Left(PairingWire::Complete {
                approved,
                invite_token,
                reason,
            }) => (approved, invite_token, reason),
            Either::Right(()) => {
                self.set_pairing_status(&session_id, PairingSessionStatus::Cancelled, None)
                    .await;
                send_pairing_cancel(&mut send, &mut recv).await?;
                return Ok(());
            }
            Either::Left(_) => anyhow::bail!("expected pairing completion response"),
        };
        if !approved {
            let status = completion_rejection_status(reason.as_deref());
            self.set_pairing_status(&session_id, status, None).await;
            return Ok(());
        }
        let invite_token =
            invite_token.ok_or_else(|| anyhow::anyhow!("approved pairing omitted invite token"))?;
        self.set_pairing_status(&session_id, PairingSessionStatus::Joining, None)
            .await;
        let join_result = tokio::select! {
            result = self.join_with_retry(&invite_token) => JoinWait::Completed(result),
            changed = cancel_rx.changed() => {
                changed.context("local pairing cancellation was dropped")?;
                JoinWait::LocalCancelled
            }
            wire = read_pairing_wire(&mut recv) => {
                match wire? {
                    PairingWire::Cancel => JoinWait::PeerMessage(()),
                    _ => anyhow::bail!("unexpected pairing message while joining"),
                }
            }
        };
        match join_result {
            JoinWait::Completed(Ok(())) => {
                write_pairing_wire(
                    &mut send,
                    &PairingWire::JoinResult {
                        joined: true,
                        reason: None,
                    },
                )
                .await?;
                finish_pairing_send(&mut send).await?;
                self.set_pairing_status(&session_id, PairingSessionStatus::Approved, None)
                    .await;
            }
            JoinWait::Completed(Err(error)) => {
                let reason = pairing_join_error(&error);
                write_pairing_wire(
                    &mut send,
                    &PairingWire::JoinResult {
                        joined: false,
                        reason: Some(reason.clone()),
                    },
                )
                .await?;
                finish_pairing_send(&mut send).await?;
                self.set_pairing_status(&session_id, PairingSessionStatus::Failed, Some(reason))
                    .await;
            }
            JoinWait::LocalCancelled => {
                send_pairing_cancel(&mut send, &mut recv).await?;
                self.set_pairing_status(&session_id, PairingSessionStatus::Cancelled, None)
                    .await;
            }
            JoinWait::PeerMessage(()) => {
                acknowledge_pairing_cancel(&mut send).await?;
                self.set_pairing_status(&session_id, PairingSessionStatus::Cancelled, None)
                    .await;
            }
        }
        Ok(())
    }

    async fn set_pairing_status(
        &self,
        id: &str,
        status: PairingSessionStatus,
        error: Option<String>,
    ) {
        if let Some(record) = self.pairing.inner.lock().await.sessions.get_mut(id) {
            record.session.status = status;
            record.session.error = error;
            if status.terminal() {
                record.decision = None;
                record.cancel = None;
            }
        }
    }

    async fn fail_pairing_session(&self, id: &str, error: String) {
        let status = if error.contains("expired") {
            PairingSessionStatus::Expired
        } else {
            PairingSessionStatus::Failed
        };
        self.set_pairing_status(id, status, Some(error)).await;
    }
}

fn bounded_device_name(value: &str) -> String {
    let clean = value
        .chars()
        .filter(|character| !character.is_control())
        .take(MAX_PAIRING_NAME_CHARS)
        .collect::<String>();
    if clean.trim().is_empty() {
        "Mesh device".to_string()
    } else {
        clean
    }
}

enum Either<L, R> {
    Left(L),
    Right(R),
}

enum JoinWait<T, M = T> {
    Completed(T),
    LocalCancelled,
    PeerMessage(M),
}

fn completion_rejection_status(reason: Option<&str>) -> PairingSessionStatus {
    match reason {
        Some("cancelled") => PairingSessionStatus::Cancelled,
        Some("expired") => PairingSessionStatus::Expired,
        _ => PairingSessionStatus::Rejected,
    }
}

fn pairing_join_error(error: &anyhow::Error) -> String {
    let message = format!("{error:#}");
    if message.contains("join rejected") || message.contains("admission") {
        "Mesh admission policy rejected this device".to_string()
    } else {
        "The device was approved, but the mesh connection failed".to_string()
    }
}

fn pairing_code(initiator: EndpointId, responder: EndpointId, session_id: &str) -> String {
    let mut hasher = sha2::Sha256::new();
    hasher.update(PAIRING_CODE_DOMAIN);
    hasher.update(initiator.as_bytes());
    hasher.update(responder.as_bytes());
    hasher.update(session_id.as_bytes());
    let digest = hasher.finalize();
    let value = u32::from_be_bytes(digest[..4].try_into().expect("four digest bytes")) % 1_000_000;
    format!("{value:06}")
}

fn prune_pairing_state(state: &mut PairingState, now: u64) {
    state.offers.retain(|_, offer| offer.expires_at > now);
    state.sessions.retain(|_, record| {
        if !record.session.status.terminal() && record.session.expires_at <= now {
            record.session.status = PairingSessionStatus::Expired;
            record.decision = None;
            record.cancel = None;
        }
        !record.session.status.terminal()
            || record
                .session
                .expires_at
                .saturating_add(PAIRING_RETENTION_SECS)
                > now
    });
}

async fn write_pairing_wire(
    send: &mut iroh::endpoint::SendStream,
    message: &PairingWire,
) -> Result<()> {
    let body = serde_json::to_vec(message)?;
    anyhow::ensure!(
        body.len() <= MAX_PAIRING_FRAME_BYTES,
        "pairing frame is too large"
    );
    write_len_prefixed(send, &body).await
}

async fn finish_pairing_send(send: &mut iroh::endpoint::SendStream) -> Result<()> {
    send.finish()?;
    match tokio::time::timeout(std::time::Duration::from_secs(3), send.stopped()).await {
        Ok(Ok(None)) => Ok(()),
        Ok(Ok(Some(code))) => anyhow::bail!("peer stopped the pairing stream with code {code}"),
        Ok(Err(error)) => Err(error.into()),
        Err(_) => anyhow::bail!("timed out confirming pairing message delivery"),
    }
}

async fn send_pairing_cancel(
    send: &mut iroh::endpoint::SendStream,
    recv: &mut iroh::endpoint::RecvStream,
) -> Result<()> {
    write_pairing_wire(send, &PairingWire::Cancel).await?;
    send.finish()?;
    let PairingWire::Complete {
        approved: false,
        reason: Some(reason),
        ..
    } = tokio::time::timeout(std::time::Duration::from_secs(3), read_pairing_wire(recv))
        .await
        .map_err(|_| anyhow::anyhow!("timed out confirming pairing cancellation"))??
    else {
        anyhow::bail!("expected pairing cancellation confirmation")
    };
    anyhow::ensure!(
        reason == "cancelled",
        "pairing cancellation was not confirmed"
    );
    Ok(())
}

async fn acknowledge_pairing_cancel(send: &mut iroh::endpoint::SendStream) -> Result<()> {
    write_pairing_wire(
        send,
        &PairingWire::Complete {
            approved: false,
            invite_token: None,
            reason: Some("cancelled".to_string()),
        },
    )
    .await?;
    finish_pairing_send(send).await
}

async fn read_pairing_wire<T: DeserializeOwned>(
    recv: &mut iroh::endpoint::RecvStream,
) -> Result<T> {
    let mut len = [0u8; 4];
    recv.read_exact(&mut len).await?;
    let len = u32::from_le_bytes(len) as usize;
    anyhow::ensure!(len <= MAX_PAIRING_FRAME_BYTES, "pairing frame is too large");
    let mut body = vec![0; len];
    recv.read_exact(&mut body).await?;
    serde_json::from_slice(&body).context("invalid pairing frame")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairing_offer_round_trip_does_not_contain_invite_material() {
        let secret = SecretKey::generate();
        let offer = PairingOffer {
            version: 1,
            offer_id: uuid::Uuid::new_v4().to_string(),
            endpoint_addr: EndpointAddr::new(secret.public()),
            device_name: "Studio".to_string(),
            expires_at: now_secs() + 60,
        };
        let encoded = offer.encode().unwrap();
        assert_eq!(PairingOffer::decode(&encoded).unwrap(), offer);
        assert_eq!(PairingOffer::decode(&offer.url().unwrap()).unwrap(), offer);
        assert!(!encoded.contains("invite_token"));
    }

    #[test]
    fn comparison_code_is_directional_and_six_digits() {
        let left = SecretKey::generate().public();
        let right = SecretKey::generate().public();
        let code = pairing_code(left, right, "session");
        assert_eq!(code.len(), 6);
        assert!(code.chars().all(|character| character.is_ascii_digit()));
        assert_ne!(code, pairing_code(right, left, "session"));
    }
}
