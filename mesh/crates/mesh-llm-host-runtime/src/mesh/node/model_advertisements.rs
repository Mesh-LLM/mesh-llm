//! Local model advertisement publication and native status notifications.
use super::{
    ModelRuntimeDescriptor, ModelSourceKind, Node, ServedModelDescriptor, ServedModelIdentity,
};

impl Node {
    fn publish_model_status_change(&self) {
        self.runtime_data_collector()
            .mark_dirty(crate::runtime_data::RuntimeDataDirty::STATUS);
    }

    pub async fn set_served_model_descriptors(&self, mut descriptors: Vec<ServedModelDescriptor>) {
        let model_names: std::collections::HashSet<_> = descriptors
            .iter()
            .map(|descriptor| descriptor.identity.model_name.clone())
            .collect();
        let mut generations = self.served_model_generations.lock().await;
        let mut current = self.served_model_descriptors.lock().await;
        for descriptor in &mut descriptors {
            if descriptor.identity.weights_digest.is_none() {
                descriptor.identity.weights_digest = current
                    .iter()
                    .find(|existing| existing.identity.model_name == descriptor.identity.model_name)
                    .and_then(|existing| existing.identity.weights_digest.clone());
            }
        }
        for removed in current
            .iter()
            .filter(|descriptor| !model_names.contains(&descriptor.identity.model_name))
        {
            let generation = generations
                .entry(removed.identity.model_name.clone())
                .or_default();
            *generation = generation.wrapping_add(1);
        }
        *current = descriptors;
        drop(current);
        drop(generations);
        self.model_runtime_descriptors
            .lock()
            .await
            .retain(|runtime| model_names.contains(&runtime.model_name));
        self.publish_model_status_change();
    }

    pub(crate) async fn update_served_model_descriptor(
        &self,
        model_name: &str,
        update: impl FnOnce(Option<ServedModelDescriptor>) -> ServedModelDescriptor,
    ) {
        let mut descriptors = self.served_model_descriptors.lock().await;
        if let Some(existing) = descriptors
            .iter_mut()
            .find(|descriptor| descriptor.identity.model_name == model_name)
        {
            *existing = update(Some(existing.clone()));
        } else {
            descriptors.push(update(None));
        }
        drop(descriptors);
        self.publish_model_status_change();
    }

    pub(crate) async fn begin_served_model_generation(&self, model_name: &str) -> u64 {
        let mut generations = self.served_model_generations.lock().await;
        let generation = generations.entry(model_name.to_string()).or_default();
        *generation = generation.wrapping_add(1);
        let generation = *generation;

        let mut descriptors = self.served_model_descriptors.lock().await;
        if let Some(descriptor) = descriptors
            .iter_mut()
            .find(|descriptor| descriptor.identity.model_name == model_name)
        {
            descriptor.identity.weights_digest = None;
        } else {
            descriptors.push(ServedModelDescriptor {
                identity: ServedModelIdentity {
                    model_name: model_name.to_string(),
                    source_kind: ModelSourceKind::LocalGguf,
                    local_file_name: Some(format!("{model_name}.gguf")),
                    ..Default::default()
                },
                requires: Vec::new(),
                capabilities_known: false,
                capabilities: crate::models::ModelCapabilities::default(),
                topology: None,
                metadata: crate::models::served_model_metadata_for_model(model_name),
            });
        }
        drop(descriptors);
        drop(generations);
        self.publish_model_status_change();
        generation
    }

    pub(crate) async fn set_served_model_weights_digest_for_generation(
        &self,
        model_name: &str,
        generation: u64,
        weights_digest: Option<String>,
    ) -> bool {
        let generations = self.served_model_generations.lock().await;
        if generations.get(model_name).copied() != Some(generation) {
            return false;
        }
        let mut descriptors = self.served_model_descriptors.lock().await;
        let Some(descriptor) = descriptors
            .iter_mut()
            .find(|descriptor| descriptor.identity.model_name == model_name)
        else {
            return false;
        };
        descriptor.identity.weights_digest = weights_digest;
        drop(descriptors);
        drop(generations);
        self.publish_model_status_change();
        true
    }

    pub async fn upsert_served_model_descriptor(&self, descriptor: ServedModelDescriptor) {
        let mut descriptors = self.served_model_descriptors.lock().await;
        if let Some(existing) = descriptors
            .iter_mut()
            .find(|existing| existing.identity.model_name == descriptor.identity.model_name)
        {
            *existing = descriptor;
        } else {
            descriptors.push(descriptor);
        }
        drop(descriptors);
        self.publish_model_status_change();
    }

    pub async fn remove_served_model_descriptor(&self, model_name: &str) {
        let mut generations = self.served_model_generations.lock().await;
        let generation = generations.entry(model_name.to_string()).or_default();
        *generation = generation.wrapping_add(1);
        self.served_model_descriptors
            .lock()
            .await
            .retain(|descriptor| descriptor.identity.model_name != model_name);
        drop(generations);
        self.model_runtime_descriptors
            .lock()
            .await
            .retain(|runtime| runtime.model_name != model_name);
        self.publish_model_status_change();
    }

    pub async fn set_model_runtime_context_length(
        &self,
        model_name: &str,
        context_length: Option<u32>,
    ) {
        let identity_hash = self
            .served_model_descriptors
            .lock()
            .await
            .iter()
            .find(|descriptor| descriptor.identity.model_name == model_name)
            .and_then(|descriptor| descriptor.identity.identity_hash.clone());
        let mut runtimes = self.model_runtime_descriptors.lock().await;
        if let Some(context_length) = context_length {
            if let Some(runtime) = runtimes
                .iter_mut()
                .find(|runtime| runtime.model_name == model_name)
            {
                runtime.identity_hash = identity_hash.or_else(|| runtime.identity_hash.clone());
                runtime.context_length = Some(context_length);
                runtime.ready = true;
            } else {
                runtimes.push(ModelRuntimeDescriptor {
                    model_name: model_name.to_string(),
                    identity_hash,
                    context_length: Some(context_length),
                    ready: true,
                });
            }
        } else {
            runtimes.retain(|runtime| runtime.model_name != model_name);
        }
        drop(runtimes);
        self.publish_model_status_change();
    }

    pub async fn set_hosted_models(&self, models: Vec<String>) {
        *self.hosted_models.lock().await = models;
        self.publish_model_status_change();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::NodeRole;

    #[tokio::test]
    async fn model_metadata_changes_notify_subscribers_without_a_runtime_tick() {
        let node = Node::new_for_tests(NodeRole::Client).await.unwrap();
        let mut updates = node.runtime_data_collector().subscribe();
        let mut descriptor = ServedModelDescriptor::default();
        descriptor.identity.model_name = "chat".into();
        node.set_served_model_descriptors(vec![descriptor.clone()])
            .await;
        assert!(updates.has_changed().unwrap());
        updates.borrow_and_update();
        node.update_served_model_descriptor("chat", |existing| {
            let mut descriptor = existing.unwrap();
            descriptor.identity.revision = Some("revision".into());
            descriptor
        })
        .await;
        assert!(updates.has_changed().unwrap());
        assert_eq!(
            node.served_model_descriptors().await[0]
                .identity
                .revision
                .as_deref(),
            Some("revision")
        );
        updates.borrow_and_update();
        node.remove_served_model_descriptor("chat").await;
        assert!(updates.has_changed().unwrap());
        assert!(node.served_model_descriptors().await.is_empty());
        node.close_endpoint().await;
    }
}
