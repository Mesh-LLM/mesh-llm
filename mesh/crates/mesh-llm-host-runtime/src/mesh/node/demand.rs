//! The mesh-wide model demand map, and the models this node requests.
use super::*;

impl Node {
    /// Record a request for a model — updates the demand map.
    /// Called from API proxy on every request (including misses for unserved models).
    /// Uses std::sync::Mutex (not tokio) so it can be called from sync context too.
    pub fn record_request(&self, model: &str) {
        // "auto" is a routing directive, not a real model — don't pollute demand
        if model == "auto" || model.is_empty() {
            return;
        }
        let model_ref = canonical_demand_model_ref(model);
        let mut demand = self.model_demand.lock().unwrap();
        let entry = demand.entry(model_ref).or_default();
        entry.last_active = now_secs();
        entry.request_count += 1;
    }

    /// Get the current demand map (for gossip and assignment decisions).
    pub fn get_demand(&self) -> HashMap<String, ModelDemand> {
        self.model_demand.lock().unwrap().clone()
    }

    /// Merge incoming demand from gossip into our local map.
    pub fn merge_remote_demand(&self, remote: &HashMap<String, ModelDemand>) {
        let mut demand = self.model_demand.lock().unwrap();
        merge_demand(&mut demand, remote);
    }

    /// Remove demand entries that have expired (past TTL and not pinned).
    /// Call periodically to prevent unbounded map growth.
    pub async fn gc_demand(&self) {
        let now = now_secs();
        let pinned = self.pinned_requested_models().await;

        let mut demand = self.model_demand.lock().unwrap();
        demand.retain(|model, d| {
            pinned.contains(model) || now.saturating_sub(d.last_active) < DEMAND_TTL_SECS
        });
    }

    /// Get active demand entries (within TTL or pinned by a live node).
    /// This replaces mesh_wanted_models().
    pub async fn active_demand(&self) -> HashMap<String, ModelDemand> {
        let now = now_secs();
        let demand = self.model_demand.lock().unwrap().clone();
        let pinned = self.pinned_requested_models().await;

        demand
            .into_iter()
            .filter(|(model, d)| {
                pinned.contains(model) || now.saturating_sub(d.last_active) < DEMAND_TTL_SECS
            })
            .collect()
    }

    async fn pinned_requested_models(&self) -> std::collections::HashSet<String> {
        let my_requested = self.requested_models.lock().await;
        let peers = self.state.lock().await;
        let mut pinned: std::collections::HashSet<String> = my_requested.iter().cloned().collect();
        for peer in peers.peers.values() {
            pinned.extend(peer.requested_models.iter().cloned());
        }
        pinned
    }

    pub async fn set_requested_models(&self, models: Vec<String>) {
        let models = models
            .into_iter()
            .map(|model| canonical_demand_model_ref(&model))
            .collect::<Vec<_>>();
        // Seed demand entries for --model declarations
        {
            let mut demand = self.model_demand.lock().unwrap();
            let now = now_secs();
            for m in &models {
                let entry = demand.entry(m.clone()).or_default();
                entry.last_active = entry.last_active.max(now);
            }
        }
        *self.requested_models.lock().await = models;
    }

    pub async fn requested_models(&self) -> Vec<String> {
        self.requested_models.lock().await.clone()
    }

    pub async fn set_explicit_model_interests(&self, mut model_refs: Vec<String>) {
        model_refs.retain(|model_ref| !model_ref.trim().is_empty());
        model_refs.sort();
        model_refs.dedup();
        *self.explicit_model_interests.lock().await = model_refs;
    }

    pub async fn explicit_model_interests(&self) -> Vec<String> {
        self.explicit_model_interests.lock().await.clone()
    }
}
