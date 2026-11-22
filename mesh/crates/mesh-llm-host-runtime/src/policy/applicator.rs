use crate::policy::{PolicyStore, PolicyUpdateResult};
use crate::proto::node::{MeshModelIntentAction, MeshPolicyPayload, SignedMeshPolicy};
use std::sync::Arc;

pub(crate) struct PolicyApplicator {
    store: Arc<PolicyStore>,
    management_port: u16,
    owner_control_token: String,
}

impl PolicyApplicator {
    pub fn new(
        store: Arc<PolicyStore>,
        management_port: u16,
        owner_control_token: String,
    ) -> Arc<Self> {
        Arc::new(Self {
            store,
            management_port,
            owner_control_token,
        })
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
        match self.store.update(incoming).await {
            PolicyUpdateResult::Accepted { .. } => {
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

    async fn apply_model_intents(&self, payload: &MeshPolicyPayload) {
        let client = match reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
        {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!("PolicyApplicator: failed to build HTTP client: {e}");
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
                "endpoint": "local",
                "model": intent.canonical_model_ref,
            });
            let url = format!("http://127.0.0.1:{}{}", self.management_port, path);
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
        let url = format!("http://127.0.0.1:{}/api/runtime", self.management_port);
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
