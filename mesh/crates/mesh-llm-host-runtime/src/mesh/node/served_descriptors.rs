//! The descriptors of the models this node serves, and the load
//! generations that scope a descriptor's weights digest to one runtime.
use super::*;

impl Node {
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
                receipts: false,
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
        generation
    }

    /// Record the local runtime `port` the generation `begin_served_model_generation`
    /// returned will serve on. Ignored when a newer generation has begun.
    pub(crate) async fn bind_served_model_generation_to_port(
        &self,
        model_name: &str,
        generation: u64,
        port: u16,
    ) {
        let generations = self.served_model_generations.lock().await;
        if generations.get(model_name).copied() == Some(generation) {
            self.served_model_weights_ports
                .lock()
                .await
                .insert(model_name.to_string(), (generation, port));
        }
    }

    /// Load a model on `port` the way the runtime does: a new generation
    /// bound to that port, carrying `digest`.
    #[cfg(test)]
    pub(crate) async fn load_model_for_tests(&self, model_name: &str, port: u16, digest: &str) {
        let generation = self.begin_served_model_generation(model_name).await;
        self.bind_served_model_generation_to_port(model_name, generation, port)
            .await;
        assert!(
            self.set_served_model_weights_digest_for_generation(
                model_name,
                generation,
                Some(digest.to_owned()),
            )
            .await
        );
    }

    /// Drop `identity.weights_digest` unless it was computed for the runtime
    /// instance on `port`. The node-wide descriptor keeps only the newest
    /// generation's digest, so a request that reached an older instance of the
    /// same model name (its file replaced between loads) must not report the
    /// newer file's bytes as the ones it ran.
    pub(crate) async fn scope_weights_digest_to_port(
        &self,
        identity: &mut ServedModelIdentity,
        port: u16,
    ) {
        if identity.weights_digest.is_none() {
            return;
        }
        let name = &identity.model_name;
        let current = self
            .served_model_generations
            .lock()
            .await
            .get(name)
            .copied();
        let bound = self
            .served_model_weights_ports
            .lock()
            .await
            .get(name)
            .copied();
        if bound.is_none() || bound != current.map(|generation| (generation, port)) {
            identity.weights_digest = None;
        }
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
    }

    pub async fn remove_served_model_descriptor(&self, model_name: &str) {
        self.set_issues_receipts(model_name, false);
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
    }

    /// Locally served descriptors only -- unlike [`Self::all_served_model_descriptors`],
    /// never includes a peer's gossiped copy, so a caller matching on
    /// `model_name` cannot be handed a peer's descriptor for a same-named
    /// model. `network/openai/ingress.rs`'s serving-provenance lookup relies
    /// on that: gossip strips `weights_digest` before it crosses the wire
    /// (see `protocol/convert.rs`), so a peer descriptor would silently read
    /// back `None` even when this host's own load-time digest is known.
    /// Endpoint-local workload admission also snapshots only this node's
    /// models, never a peer's copy.
    pub async fn served_model_descriptors(&self) -> Vec<ServedModelDescriptor> {
        self.served_model_descriptors.lock().await.clone()
    }

    pub async fn all_served_model_descriptors(&self) -> Vec<ServedModelDescriptor> {
        // Local descriptors carry the channel requirements plugins attached,
        // exactly as peers see them in announcements.
        let hosted_models = self.hosted_models.lock().await.clone();
        let mut descriptors = self.descriptors_with_requirements(&hosted_models).await;
        let peer_descriptors = {
            let state = self.state.lock().await;
            state
                .peers
                .values()
                .flat_map(|peer| peer.served_model_descriptors.clone())
                .collect::<Vec<_>>()
        };
        descriptors.extend(peer_descriptors);
        descriptors
    }

    pub(crate) async fn refresh_served_model_descriptors(&self) {
        let serving_models = self.serving_models.lock().await.clone();
        let existing_by_name: HashMap<_, _> = self
            .served_model_descriptors
            .lock()
            .await
            .iter()
            .map(|descriptor| (descriptor.identity.model_name.clone(), descriptor.clone()))
            .collect();
        let mut descriptors = if let Some(primary_model_name) = serving_models.first() {
            let model_source = self.model_source.lock().await.clone();
            let primary_model_path = crate::models::find_model_path(primary_model_name);
            infer_served_model_descriptors(
                primary_model_name,
                &serving_models,
                model_source.as_deref(),
                Some(primary_model_path.as_path()),
            )
        } else {
            Vec::new()
        };
        for descriptor in &mut descriptors {
            if descriptor.metadata.is_none() {
                descriptor.metadata =
                    crate::models::served_model_metadata_for_model(&descriptor.identity.model_name);
            }
            if let Some(existing) = existing_by_name.get(&descriptor.identity.model_name) {
                if existing.identity.source_kind == ModelSourceKind::LocalGguf
                    || (descriptor.identity.model_name.starts_with("local-gguf/")
                        && existing.identity.source_kind != ModelSourceKind::Unknown)
                {
                    descriptor.identity = existing.identity.clone();
                }
                descriptor.capabilities = existing.capabilities;
                descriptor.capabilities_known = existing.capabilities_known;
                if existing.topology.is_some() {
                    descriptor.topology = existing.topology.clone();
                }
                if existing.metadata.is_some() {
                    descriptor.metadata = existing.metadata.clone();
                }
            }
        }
        self.set_served_model_descriptors(descriptors).await;
    }
}
