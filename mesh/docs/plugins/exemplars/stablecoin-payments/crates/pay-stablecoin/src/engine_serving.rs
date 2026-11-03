//! Tie authenticated serving ingress to the host's post-resolution admission hook.
use super::*;

pub(super) struct RemoteIngress {
    peer: String,
    body: Option<Value>,
    endpoint: Option<String>,
}

/// Observation IDs are host-minted and shared by this node's lifecycle phases.
pub(super) fn correlation(event: &Value) -> Result<&str> {
    let id = event["observation_id"]
        .as_str()
        .or_else(|| event["exchange_id"].as_str())
        .context("host exchange correlation is missing")?;
    ensure!(
        !id.is_empty() && id.len() <= 256,
        "invalid host exchange correlation"
    );
    Ok(id)
}

impl Engine {
    pub(super) fn remember_ingress(&self, event: &Value) -> Result<()> {
        if event["phase"] != "request_received" || self.config.prices.is_empty() {
            return Ok(());
        }
        let Some(peer) = event["requested_by_node_id"].as_str() else {
            return Ok(());
        };
        validate_peer(peer)?;
        let key = correlation(event)?;
        let mut ingress = self.ingress.lock().unwrap();
        ensure!(
            ingress.contains_key(key) || ingress.len() < 256,
            "paid serving request tracking capacity exhausted"
        );
        if let Some(existing) = ingress.get(key) {
            ensure!(
                existing.peer == peer,
                "authenticated serving peer changed within one observation"
            );
            return Ok(());
        }
        let body = event.get("body").filter(|body| !body.is_null()).cloned();
        if let Some(body) = &body {
            ensure!(
                serde_json::to_vec(body)?.len() <= 64 * 1024,
                "serving request body exceeds tracking limit"
            );
        }
        ingress.insert(
            key.into(),
            RemoteIngress {
                peer: peer.into(),
                body,
                endpoint: event["endpoint"].as_str().map(str::to_owned),
            },
        );
        Ok(())
    }
    pub(super) fn restore_ingress(&self, event: &mut Value) {
        if event["phase"] != "backend_selected" {
            return;
        }
        let Ok(key) = correlation(event) else {
            return;
        };
        let ingress = self.ingress.lock().unwrap();
        let Some(original) = ingress.get(key) else {
            return;
        };
        event["requested_by_node_id"] = Value::String(original.peer.clone());
        if event.get("body").is_none_or(Value::is_null)
            && let Some(body) = &original.body
        {
            event["body"] = body.clone();
        }
        if event.get("endpoint").is_none_or(Value::is_null)
            && let Some(endpoint) = &original.endpoint
        {
            event["endpoint"] = Value::String(endpoint.clone());
        }
    }
    pub(super) fn is_remote_paid_selection(&self, event: &Value) -> bool {
        event["requested_by_node_id"].as_str().is_some()
            && event["model"]
                .as_str()
                .is_some_and(|model| self.config.prices.contains_key(model))
    }
    pub(super) fn forget_ingress(&self, event: &Value) {
        if let Ok(key) = correlation(event) {
            self.ingress.lock().unwrap().remove(key);
        }
    }
}
