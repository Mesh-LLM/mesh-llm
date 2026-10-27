//! Plugin-owned model gates. Membership retains raw advertisements; only local
//! routing projections remove models the local plugin set cannot consume.
use super::{Node, PeerInfo, ServedModelDescriptor};
use crate::plugin::proto;
use std::collections::HashSet;

pub(crate) type ModelRequirements = std::sync::Arc<
    tokio::sync::Mutex<
        std::collections::HashMap<
            String,
            std::collections::HashMap<String, Vec<mesh_llm_types::mesh::ChannelRequirement>>,
        >,
    >,
>;

impl Node {
    pub(crate) async fn apply_model_requirement(
        &self,
        plugin: &str,
        request: proto::ModelRequirementRequest,
    ) -> Result<proto::ModelRequirementResponse, proto::ErrorResponse> {
        let invalid = |message: &str| proto::ErrorResponse {
            code: -32602,
            message: message.to_string(),
            data_json: String::new(),
        };
        if request.model.trim().is_empty()
            || request.model.len() > 1024
            || request.requires.len() > 8
            || request
                .requires
                .iter()
                .map(|r| r.terms.len())
                .sum::<usize>()
                > 4096
        {
            return Err(invalid("invalid model requirement"));
        }
        let manager = self
            .plugin_manager
            .lock()
            .await
            .clone()
            .ok_or_else(|| invalid("plugin manager unavailable"))?;
        // Validate the entire replacement before mutating anything.
        for channel in &request.requires {
            if channel.terms.len() > 1024
                || !manager
                    .plugin_declares_mesh_channel(plugin, &channel.channel)
                    .await
            {
                return Err(invalid("plugin cannot require an undeclared channel"));
            }
        }
        let mut requirements = self.model_requirements.lock().await;
        let owned = requirements.entry(plugin.to_string()).or_default();
        if request.requires.is_empty() {
            owned.remove(&request.model);
        } else {
            let mut channels = request.requires;
            channels.sort_by(|a, b| a.channel.cmp(&b.channel).then(a.terms.cmp(&b.terms)));
            channels.dedup();
            owned.insert(
                request.model,
                channels
                    .into_iter()
                    .map(|r| mesh_llm_types::mesh::ChannelRequirement {
                        channel: r.channel,
                        terms: r.terms,
                    })
                    .collect(),
            );
        }
        Ok(proto::ModelRequirementResponse {})
    }

    pub(crate) async fn descriptors_with_requirements(
        &self,
        models: &[String],
    ) -> Vec<ServedModelDescriptor> {
        let mut descriptors = self.served_model_descriptors.lock().await.clone();
        // Plugin-serving models may not yet have a native descriptor.
        for model in models {
            if !descriptors.iter().any(|d| d.identity.model_name == *model) {
                descriptors.extend(super::infer_remote_served_descriptors(
                    model,
                    std::slice::from_ref(model),
                    None,
                ));
            }
        }
        let requirements = self.model_requirements.lock().await;
        for descriptor in &mut descriptors {
            for owned in requirements.values() {
                for (model, channels) in owned {
                    if model != &descriptor.identity.model_name
                        && super::public_model_id_from_identity(&descriptor.identity).as_ref()
                            != Some(model)
                    {
                        continue;
                    }
                    descriptor.requires.extend(channels.iter().cloned());
                }
            }
            descriptor
                .requires
                .sort_by(|a, b| a.channel.cmp(&b.channel).then(a.terms.cmp(&b.terms)));
            descriptor.requires.dedup();
        }
        self.report_unmatched_requirements(&descriptors, &requirements)
            .await;
        descriptors
    }

    async fn report_unmatched_requirements(
        &self,
        descriptors: &[ServedModelDescriptor],
        requirements: &std::collections::HashMap<
            String,
            std::collections::HashMap<String, Vec<mesh_llm_types::mesh::ChannelRequirement>>,
        >,
    ) {
        let mut unmatched = HashSet::new();
        let mut warned = self.unmatched_model_requirements.lock().await;
        for (plugin, owned) in requirements.iter() {
            for model in owned.keys() {
                if !descriptors.iter().any(|d| {
                    &d.identity.model_name == model
                        || super::public_model_id_from_identity(&d.identity).as_ref() == Some(model)
                }) {
                    let key = (plugin.clone(), model.clone());
                    unmatched.insert(key.clone());
                    if warned.insert(key) {
                        tracing::warn!(
                            plugin,
                            model,
                            "Model requirement matched no hosted descriptor"
                        );
                    } else {
                        tracing::debug!(
                            plugin,
                            model,
                            "Model requirement still matches no hosted descriptor"
                        );
                    }
                }
            }
        }
        warned.retain(|key| unmatched.contains(key));
    }

    pub(crate) async fn supported_model_channels(&self, peers: &[PeerInfo]) -> HashSet<String> {
        let manager = self.plugin_manager.lock().await.clone();
        let Some(manager) = manager else {
            return HashSet::new();
        };
        let channels: HashSet<_> = peers
            .iter()
            .flat_map(|p| &p.served_model_descriptors)
            .flat_map(|d| d.requires.iter().map(|r| &r.channel))
            .cloned()
            .collect();
        let mut supported = HashSet::new();
        for channel in channels {
            if manager.any_plugin_declares_mesh_channel(&channel).await {
                supported.insert(channel);
            }
        }
        supported
    }

    pub(crate) async fn filter_model_requirements(&self, peers: &mut [PeerInfo]) {
        let supported = self.supported_model_channels(peers).await;
        for peer in peers {
            filter_peer_models(peer, &supported);
        }
    }

    pub(crate) async fn model_channels_unavailable(&self, peer: &PeerInfo, model: &str) -> bool {
        let supported = self
            .supported_model_channels(std::slice::from_ref(peer))
            .await;
        peer.served_model_descriptors.iter().any(|d| {
            (d.identity.model_name == model
                || super::public_model_id_for_routable_model(peer, &d.identity.model_name) == model)
                && d.requires.iter().any(|ch| !supported.contains(&ch.channel))
        })
    }
}

fn filter_peer_models(peer: &mut PeerInfo, supported: &HashSet<String>) {
    let excluded: HashSet<_> = peer
        .served_model_descriptors
        .iter()
        .filter(|d| d.requires.iter().any(|ch| !supported.contains(&ch.channel)))
        .map(|d| d.identity.model_name.clone())
        .collect();
    // This is a disposable routing snapshot, never written back into membership.
    peer.served_model_descriptors
        .retain(|d| !excluded.contains(&d.identity.model_name));
    peer.hosted_models.retain(|m| !excluded.contains(m));
    peer.serving_models.retain(|m| !excluded.contains(m));
}
