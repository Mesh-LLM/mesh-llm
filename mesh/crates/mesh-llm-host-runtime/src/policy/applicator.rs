use crate::policy::{PolicyStore, PolicyUpdateResult};
use crate::proto::node::{MeshModelIntentAction, MeshPolicyPayload, SignedMeshPolicy};
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::Arc;

pub(crate) struct PolicyApplicator {
    store: Arc<PolicyStore>,
    /// Atomically updatable so the actual OS-assigned port can be written
    /// after the console listener binds (fixes `--console 0` / port 0).
    management_port: Arc<AtomicU16>,
    owner_control_token: String,
    data_dir: Option<std::path::PathBuf>,
    /// The owner IDs authorised to inject policy into this node.
    /// Passed to `PolicyStore::update` on every inbound gossip policy.
    authorized_owner_ids: Vec<String>,
}

impl PolicyApplicator {
    pub fn new(
        store: Arc<PolicyStore>,
        management_port: u16,
        owner_control_token: String,
        data_dir: Option<std::path::PathBuf>,
        authorized_owner_ids: Vec<String>,
    ) -> Arc<Self> {
        Arc::new(Self {
            store,
            management_port: Arc::new(AtomicU16::new(management_port)),
            owner_control_token,
            data_dir,
            authorized_owner_ids,
        })
    }

    /// Update the management port after the console listener has bound.
    /// Called from the serving surface once the actual ephemeral port is known.
    pub fn set_management_port(&self, port: u16) {
        self.management_port.store(port, Ordering::Relaxed);
    }

    pub async fn apply_if_newer(self: &Arc<Self>, incoming: SignedMeshPolicy) {
        // Basic decode check and expiry before passing to store
        if incoming.policy_payload_bytes.is_empty() {
            return;
        }
        // Check expiry
        if let Ok(payload) =
            prost::Message::decode(incoming.policy_payload_bytes.as_slice())
                as Result<MeshPolicyPayload, _>
        {
            if payload.expires_at_unix_ms > 0 {
                let now_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64;
                if now_ms > payload.expires_at_unix_ms {
                    return;
                }
            }
        }
        // Pass None for node_mesh_id on the gossip path: mesh_id scoping is
        // validated at the management-API ingestion point instead.
        match self.store.update(incoming, &self.authorized_owner_ids, None).await {
            PolicyUpdateResult::Accepted { .. } => {
                if let Some(state_dir) = self.data_dir.clone() {
                    let store = Arc::clone(&self.store);
                    tokio::spawn(async move {
                        if let Err(e) = store.persist(&state_dir).await {
                            tracing::warn!("PolicyApplicator: gossip policy persist failed: {e:#}");
                        }
                    });
                }
                if let Some(stored) = self.store.current().await {
                    let applicator = Arc::clone(self);
                    let payload = stored.payload.clone();
                    tokio::spawn(async move {
                        applicator.apply_model_intents(&payload).await;
                    });
                }
            }
            _ => {}
        }
    }

    /// Apply model intents from a payload directly, bypassing the store's
    /// generation check. Used after an owner-signed policy is accepted via the
    /// management API so intents run on this host immediately without waiting
    /// for the next gossip round (a second gossip call would return `Stale`).
    pub(crate) async fn apply_policy_payload(self: &Arc<Self>, payload: &MeshPolicyPayload) {
        self.apply_model_intents(payload).await;
    }

    async fn intent_already_satisfied(
        &self,
        action: MeshModelIntentAction,
        model_ref: &str,
        serving_models: &[String],
    ) -> bool {
        match action {
            MeshModelIntentAction::Ensure => serving_models.iter().any(|m| m == model_ref),
            MeshModelIntentAction::Drain => !serving_models.iter().any(|m| m == model_ref),
            MeshModelIntentAction::Unspecified => true,
        }
    }

    async fn fetch_local_control_endpoint(&self, client: &reqwest::Client) -> Option<String> {
        let port = self.management_port.load(Ordering::Relaxed);
        let url = format!("http://127.0.0.1:{port}/api/runtime/control-bootstrap");
        let Ok(resp) = client.get(&url).send().await else {
            return None;
        };
        let Ok(json) = resp.json::<serde_json::Value>().await else {
            return None;
        };
        json.get("endpoint")?.as_str().map(str::to_string)
    }

    async fn apply_model_intents(&self, payload: &MeshPolicyPayload) {
        let client = match mesh_llm_system::local_http::local_http_client_builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
        {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!("PolicyApplicator: failed to build HTTP client: {e}");
                return;
            }
        };

        // Resolve the local control endpoint token (a real QUIC endpoint address,
        // not the literal string "local").
        let local_endpoint = match self.fetch_local_control_endpoint(&client).await {
            Some(ep) => ep,
            None => {
                tracing::warn!(
                    "PolicyApplicator: control-bootstrap endpoint not ready yet; \
                     skipping model intent application"
                );
                return;
            }
        };

        // Fetch current serving models for idempotency check
        let serving_models = self.fetch_serving_models(&client).await;

        for intent in &payload.model_intents {
            let action = match MeshModelIntentAction::try_from(intent.action) {
                Ok(a) => a,
                Err(_) => continue,
            };
            // Skip if already satisfied
            if self
                .intent_already_satisfied(action, &intent.canonical_model_ref, &serving_models)
                .await
            {
                continue;
            }
            let path = match action {
                MeshModelIntentAction::Ensure => "/api/runtime/control/ensure-model",
                MeshModelIntentAction::Drain => "/api/runtime/control/drain-model",
                MeshModelIntentAction::Unspecified => continue,
            };
            let body = serde_json::json!({
                "endpoint": local_endpoint,
                "model": intent.canonical_model_ref,
            });
            let port = self.management_port.load(Ordering::Relaxed);
            let url = format!("http://127.0.0.1:{port}{path}");
            let _ = client
                .post(&url)
                .header(
                    reqwest::header::AUTHORIZATION,
                    format!("Bearer {}", self.owner_control_token),
                )
                .json(&body)
                .send()
                .await;
        }
    }

    async fn fetch_serving_models(&self, client: &reqwest::Client) -> Vec<String> {
        let port = self.management_port.load(Ordering::Relaxed);
        let url = format!("http://127.0.0.1:{port}/api/runtime");
        let Ok(response) = client.get(&url).send().await else {
            return Vec::new();
        };
        let Ok(json) = response.json::<serde_json::Value>().await else {
            return Vec::new();
        };
        json.get("serving_models")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    }
}
