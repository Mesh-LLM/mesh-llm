//! The models this node serves with inference receipts
//! (`mesh/docs/design/RECEIPTS.md`): what the runtime records after a load
//! and the node then advertises with the model's descriptor.
use super::*;

impl Node {
    /// The served models whose runtime issues receipts. A thread that
    /// panicked while it held the lock left a whole set behind, since every
    /// change is one insert or remove, so a poisoned lock is read like any
    /// other: routing and gossip must not panic for it.
    fn receipt_models(&self) -> std::sync::MutexGuard<'_, std::collections::HashSet<String>> {
        self.receipt_models
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Records whether the runtime now serving `model_name` issues inference
    /// receipts, which this node advertises with the model and which makes
    /// it a candidate under the `receipts` trust tier
    /// (`network/request_trust.rs`). The runtime says so itself after a
    /// load: all of it runs on this node, and its recorder started
    /// (`runtime/local/receipts.rs`, `runtime/local_split/receipts.rs`).
    pub(crate) fn set_issues_receipts(&self, model_name: &str, issues: bool) {
        let mut models = self.receipt_models();
        if issues {
            models.insert(model_name.to_owned());
        } else {
            models.remove(model_name);
        }
    }

    /// Whether this node's runtime for `model_name` issues inference receipts.
    pub(crate) fn issues_receipts(&self, model_name: &str) -> bool {
        self.receipt_models().contains(model_name)
    }

    /// The served models this node advertises with receipts.
    fn models_issuing_receipts(&self) -> std::collections::HashSet<String> {
        self.receipt_models().clone()
    }

    /// Marks, on the descriptors of the models this node serves, the ones
    /// it advertises with receipts: those whose runtime issues them, under a
    /// name a client can audit them by ([`client_can_tie`]).
    pub(in crate::mesh) fn mark_receipts(&self, descriptors: &mut [ServedModelDescriptor]) {
        let receipt_models = self.models_issuing_receipts();
        for descriptor in descriptors {
            descriptor.receipts = receipt_models.contains(&descriptor.identity.model_name)
                && client_can_tie(descriptor);
        }
    }
}

/// Whether a client can tie the name it routes `descriptor` by to a
/// Hugging Face file, as an auditor does before it audits a turn asked
/// under it (`network::receipt_audit::ties_model_name`). A GGUF of the
/// Hugging Face cache served by its path is routed as a local file's id,
/// `local-gguf/sha256-…`: its receipts name the Hugging Face file, but the
/// audit holds a turn to the file the name it asked for stands for, so
/// every audit of it would end as no audit, and its receipts must not
/// satisfy the `receipts` tier.
fn client_can_tie(descriptor: &ServedModelDescriptor) -> bool {
    let identity = &descriptor.identity;
    let routed = crate::mesh::public_model_id_from_identity(identity).unwrap_or_else(|| {
        crate::mesh::model_identity::canonical_demand_model_ref(&identity.model_name)
    });
    crate::network::receipt_audit::ties_model_name(&routed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_poisoned_receipts_lock_stops_neither_routing_nor_gossip() {
        let node = Node::new_for_tests(NodeRole::Worker)
            .await
            .expect("test node should start");
        node.set_issues_receipts("m", true);
        let models = node.receipt_models.clone();
        let poisoner = std::thread::spawn(move || {
            let _held = models.lock().expect("not poisoned yet");
            panic!("poison the receipts lock");
        });
        assert!(poisoner.join().is_err());
        assert!(node.receipt_models.is_poisoned());

        assert!(node.issues_receipts("m"));
        assert!(node.models_issuing_receipts().contains("m"));
        node.local_descriptors_with_plugins().await;
        node.set_issues_receipts("m", false);
        assert!(!node.issues_receipts("m"));
    }

    /// A runtime names a direct GGUF by a local file's id. Loaded by a model
    /// reference, its descriptor names the Hugging Face file and clients
    /// route it by the reference; loaded by its path, they route it by that
    /// id, which no auditor can tie to a file.
    #[tokio::test]
    async fn receipts_are_advertised_only_under_a_name_a_client_can_tie() {
        let node = Node::new_for_tests(NodeRole::Worker)
            .await
            .expect("test node should start");
        let local = "local-gguf/sha256-0123456789abcdef";
        node.set_issues_receipts(local, true);
        let by_path = ServedModelDescriptor {
            identity: ServedModelIdentity {
                model_name: local.to_owned(),
                source_kind: ModelSourceKind::LocalGguf,
                ..Default::default()
            },
            ..Default::default()
        };
        let by_ref = ServedModelDescriptor {
            identity: ServedModelIdentity {
                model_name: local.to_owned(),
                source_kind: ModelSourceKind::HuggingFace,
                repository: Some("unsloth/Qwen3-4B-GGUF".to_owned()),
                revision: Some("0123456789abcdef0123456789abcdef01234567".to_owned()),
                artifact: Some("Qwen3-4B-Q8_0.gguf".to_owned()),
                ..Default::default()
            },
            ..Default::default()
        };
        let mut descriptors = vec![by_path, by_ref];
        node.mark_receipts(&mut descriptors);
        assert!(!descriptors[0].receipts, "routed by a local file's id");
        assert!(descriptors[1].receipts, "routed by a model reference");
    }
}
