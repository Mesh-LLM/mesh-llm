use crate::{mesh, models};

/// Runtime facts and the generation-bound file digest can arrive in either order.
#[tokio::test]
async fn workload_publication_and_weights_digest_preserve_each_other() {
    for workload_class in [
        mesh::ModelWorkloadClass::CausalGeneration,
        mesh::ModelWorkloadClass::Embedding,
        mesh::ModelWorkloadClass::SpeechSynthesis,
    ] {
        for digest_first in [true, false] {
            let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
                .await
                .expect("isolated test node");
            let model_name = "runtime-workload";
            let generation = node.begin_served_model_generation(model_name).await;
            let capabilities = models::ModelCapabilities::default();
            let publish_workload = || {
                super::set_runtime_verified_served_model_capabilities(
                    &node,
                    model_name,
                    model_name,
                    capabilities,
                    workload_class,
                )
            };
            if !digest_first {
                publish_workload().await;
            }
            assert!(
                super::set_local_model_weights_digest(
                    &node,
                    model_name,
                    generation,
                    Some("sha256:verified-weights".into()),
                )
                .await
            );
            if digest_first {
                publish_workload().await;
            }

            let descriptors = node.served_model_descriptors().await;
            let descriptor = descriptors
                .iter()
                .find(|descriptor| descriptor.identity.model_name == model_name)
                .expect("runtime descriptor");
            assert_eq!(
                descriptor.identity.weights_digest.as_deref(),
                Some("sha256:verified-weights")
            );
            assert!(descriptor.identity.is_primary);
            assert!(descriptor.capabilities_known);
            assert_eq!(descriptor.capabilities, capabilities);
            assert_eq!(
                descriptor.metadata.as_ref().unwrap().workload_class,
                Some(workload_class)
            );
        }
    }
}
