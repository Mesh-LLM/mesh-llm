//! Local announcement snapshots and legacy model descriptor backfill.
//!
//! The host composes model/plugin state here. Peer merge policy and rebroadcast
//! selection are owned by `mesh_llm_membership::announcements`.

use super::{
    ModelDemand, ModelRuntimeDescriptor, Node, NodeRole, PeerAnnouncement, ServedModelDescriptor,
    ServedModelIdentity, ServedModelMetadata, SignedNodeOwnership, infer_remote_served_descriptors,
};
use crate::mesh::requirements::current_time_unix_ms;
use crate::models::append_external_inference_models;
use std::collections::HashMap;

pub(crate) struct LocalAnnouncementData {
    role: NodeRole,
    first_joined_mesh_ts: Option<u64>,
    models: Vec<String>,
    model_source: Option<String>,
    serving_models: Vec<String>,
    hosted_models: Vec<String>,
    available_models: Vec<String>,
    requested_models: Vec<String>,
    explicit_model_interests: Vec<String>,
    model_demand: HashMap<String, ModelDemand>,
    mesh_id: Option<String>,
    mesh_policy_hash: Option<String>,
    signed_genesis_policy: Option<crate::SignedMeshGenesisPolicy>,
    release_attestation: Option<crate::ReleaseBuildAttestation>,
    direct_admission_proof: Option<crate::DirectNodeAdmissionProof>,
    available_model_metadata: Vec<crate::proto::node::CompactModelMetadata>,
    available_model_sizes: HashMap<String, u64>,
    served_model_descriptors: Vec<ServedModelDescriptor>,
    served_model_runtime: Vec<ModelRuntimeDescriptor>,
    owner_attestation: Option<SignedNodeOwnership>,
    artifact_transfer_supported: bool,
    advertised_model_throughput: Vec<crate::network::metrics::ModelThroughputHint>,
    #[cfg(feature = "payments")]
    lightning_offers: std::collections::BTreeMap<String, mesh_llm_payments_types::pricing::Pricing>,
    cache_affinity: Option<mesh_llm_routing::cache_inventory::CacheAffinityAdvertisement>,
    gpu_mem_bandwidth_gbps: Option<String>,
    gpu_compute_tflops_fp32: Option<String>,
    gpu_compute_tflops_fp16: Option<String>,
    inference_admission_state: Option<crate::proto::node::InferenceAdmissionState>,
}

pub fn backfill_legacy_descriptors(ann: &mut PeerAnnouncement) {
    if ann.served_model_descriptors.is_empty() {
        let primary_model_name = ann
            .serving_models
            .first()
            .map(String::as_str)
            .unwrap_or_default()
            .to_string();
        ann.served_model_descriptors = infer_remote_served_descriptors(
            &primary_model_name,
            &ann.serving_models,
            ann.model_source.as_deref(),
        );
    }
}

impl Node {
    pub(crate) async fn collect_rebroadcast_announcements(
        &self,
        stale_cutoff: std::time::Instant,
    ) -> mesh_llm_membership::announcements::RebroadcastAnnouncements {
        self.state
            .lock()
            .await
            .collect_rebroadcast_announcements(stale_cutoff)
    }

    #[expect(
        clippy::cognitive_complexity,
        reason = "local gossip snapshots intentionally gather many independent advertised fields in one atomic view"
    )]
    pub(crate) async fn snapshot_local_announcement_data(&self) -> LocalAnnouncementData {
        let owner_summary = self.owner_summary.lock().await.clone();
        let plugin_models = self.plugin_inference_models().await;
        let mut models = self.models.lock().await.clone();
        append_external_inference_models(&mut models, &plugin_models);
        let mut serving_models = self.serving_models.lock().await.clone();
        append_external_inference_models(&mut serving_models, &plugin_models);
        let mut hosted_models = self.hosted_models.lock().await.clone();
        append_external_inference_models(&mut hosted_models, &plugin_models);
        let activity_advertisement = self
            .activity_policy_guard
            .advertisement_decision(self.public_mesh);
        if activity_advertisement.withdraw_model_availability {
            serving_models.clear();
            hosted_models.clear();
        }
        let advertised_model_throughput = self
            .routing_metrics
            .advertisable_model_throughput(&hosted_models);
        let now_unix_ms = current_time_unix_ms();
        let cache_affinity = mesh_llm_membership::local_advertisement(
            &self.cache_affinity_inventory,
            self.endpoint.id().as_bytes(),
            now_unix_ms,
        );
        let release_attestation = self.release_attestation.lock().await.clone();
        let (mesh_id, mesh_policy_hash, signed_genesis_policy) =
            if let Some(state) = self.requirement_mesh_state.lock().await.clone() {
                (
                    Some(state.mesh_id),
                    Some(state.policy_hash),
                    state.signed_policy,
                )
            } else {
                (
                    self.mesh_id.lock().await.clone(),
                    self.mesh_policy_hash.lock().await.clone(),
                    self.signed_genesis_policy.lock().await.clone(),
                )
            };
        let direct_admission_proof = match (mesh_id.as_deref(), mesh_policy_hash.as_deref()) {
            (Some(mesh_id), Some(policy_hash)) => self.build_self_direct_admission_proof(
                mesh_id,
                policy_hash,
                release_attestation.as_ref(),
            ),
            _ => None,
        };
        LocalAnnouncementData {
            role: self.role.lock().await.clone(),
            first_joined_mesh_ts: *self.first_joined_mesh_ts.lock().await,
            models,
            model_source: self.model_source.lock().await.clone(),
            serving_models,
            hosted_models,
            available_models: self.available_models.lock().await.clone(),
            requested_models: self.requested_models.lock().await.clone(),
            explicit_model_interests: self.explicit_model_interests.lock().await.clone(),
            model_demand: self.get_demand(),
            mesh_id,
            mesh_policy_hash,
            signed_genesis_policy,
            release_attestation,
            direct_admission_proof,
            available_model_metadata: Vec::new(),
            available_model_sizes: HashMap::new(),
            served_model_descriptors: self.local_descriptors_with_plugins().await,
            served_model_runtime: self.model_runtime_descriptors.lock().await.clone(),
            owner_attestation: self.owner_attestation.lock().await.clone(),
            // A membership carries no stage transport, so no artifacts either.
            artifact_transfer_supported: !self.is_membership
                && crate::models::artifact_transfer::artifact_transfer_advertised(&owner_summary),
            advertised_model_throughput,
            #[cfg(feature = "payments")]
            lightning_offers: self.advertised_payment_offers().await.unwrap_or_default(),
            cache_affinity: Some(cache_affinity),
            gpu_mem_bandwidth_gbps: Self::format_optional_locked_f32_list(
                &self.gpu_mem_bandwidth_gbps,
            )
            .await,
            gpu_compute_tflops_fp32: Self::format_optional_locked_f32_list(
                &self.gpu_compute_tflops_fp32,
            )
            .await,
            gpu_compute_tflops_fp16: Self::format_optional_locked_f32_list(
                &self.gpu_compute_tflops_fp16,
            )
            .await,
            inference_admission_state: activity_advertisement.admission_state,
        }
    }

    pub(crate) async fn plugin_inference_models(&self) -> Vec<String> {
        let plugin_manager = self.plugin_manager.lock().await.clone();
        let Some(plugin_manager) = plugin_manager else {
            return Vec::new();
        };
        plugin_manager
            .inference_models()
            .await
            .unwrap_or_else(|error| {
                tracing::debug!(%error, "failed to collect plugin inference models for gossip");
                Vec::new()
            })
    }

    /// One descriptor per plugin inference endpoint and model, tagged with the
    /// workload class that endpoint serves.
    ///
    /// A plugin advertises models by *name* and selects a serving endpoint by
    /// *protocol*, so one name can be served by several endpoints of different
    /// classes. Advertising only the OCR class made a shared name look
    /// OCR-only to every descriptor-based admission check — chat on this node
    /// and, because this is the gossip payload, chat on every peer too.
    pub(crate) async fn plugin_inference_descriptors(&self) -> Vec<ServedModelDescriptor> {
        let plugin_manager = self.plugin_manager.lock().await.clone();
        let Some(plugin_manager) = plugin_manager else {
            return Vec::new();
        };
        let Ok(endpoints) = plugin_manager.inference_endpoints().await else {
            return Vec::new();
        };
        endpoints
            .into_iter()
            .flat_map(|endpoint| {
                let workload_class = crate::plugin::inference_routing::endpoint_workload_class(
                    endpoint.protocol.as_deref(),
                );
                endpoint
                    .models
                    .into_iter()
                    .map(move |model_name| ServedModelDescriptor {
                        identity: ServedModelIdentity {
                            model_name,
                            ..Default::default()
                        },
                        capabilities_known: true,
                        capabilities: crate::models::ModelCapabilities {
                            audio: if workload_class
                                == crate::mesh::ModelWorkloadClass::SpeechRecognition
                            {
                                crate::models::CapabilityLevel::Supported
                            } else {
                                crate::models::CapabilityLevel::None
                            },
                            ..Default::default()
                        },
                        metadata: Some(ServedModelMetadata {
                            workload_class: Some(workload_class),
                            ..Default::default()
                        }),
                        ..Default::default()
                    })
            })
            .collect()
    }

    pub(crate) async fn local_descriptors_with_plugins(&self) -> Vec<ServedModelDescriptor> {
        let mut descriptors = self.served_model_descriptors.lock().await.clone();
        descriptors.extend(self.plugin_inference_descriptors().await);
        descriptors
    }

    pub(crate) async fn format_optional_locked_f32_list(
        values: &tokio::sync::Mutex<Option<Vec<f64>>>,
    ) -> Option<String> {
        values.lock().await.as_ref().map(|values| {
            values
                .iter()
                .map(|f| format!("{:.2}", f))
                .collect::<Vec<_>>()
                .join(",")
        })
    }

    /// Builds this node's own gossip announcement from freshly collected
    /// local data.
    pub(crate) fn build_local_announcement(&self, data: LocalAnnouncementData) -> PeerAnnouncement {
        PeerAnnouncement {
            addr: self.endpoint_addr_for_advertisement(),
            role: data.role,
            first_joined_mesh_ts: data.first_joined_mesh_ts,
            models: data.models,
            vram_bytes: self.vram_bytes,
            model_source: data.model_source,
            serving_models: data.serving_models,
            hosted_models: Some(data.hosted_models),
            available_models: data.available_models,
            requested_models: data.requested_models,
            explicit_model_interests: data.explicit_model_interests,
            version: Some(crate::VERSION.to_string()),
            model_demand: data.model_demand,
            mesh_id: data.mesh_id,
            mesh_policy_hash: data.mesh_policy_hash,
            gpu_name: self.enumerate_host.then(|| self.gpu_name.clone()).flatten(),
            hostname: self.enumerate_host.then(|| self.hostname.clone()).flatten(),
            is_soc: self.is_soc,
            gpu_vram: self.enumerate_host.then(|| self.gpu_vram.clone()).flatten(),
            gpu_reserved_bytes: self
                .enumerate_host
                .then(|| self.gpu_reserved_bytes.clone())
                .flatten(),
            memory: self.enumerate_host.then_some(self.advertised_memory),
            gpu_mem_bandwidth_gbps: data.gpu_mem_bandwidth_gbps,
            gpu_compute_tflops_fp32: data.gpu_compute_tflops_fp32,
            gpu_compute_tflops_fp16: data.gpu_compute_tflops_fp16,
            available_model_metadata: data.available_model_metadata,
            experts_summary: None,
            available_model_sizes: data.available_model_sizes,
            served_model_descriptors: data.served_model_descriptors,
            served_model_runtime: data.served_model_runtime,
            owner_attestation: data.owner_attestation,
            genesis_policy: data.signed_genesis_policy,
            release_attestation: data.release_attestation,
            direct_admission_proof: data.direct_admission_proof,
            artifact_transfer_supported: !self.full_model_only && data.artifact_transfer_supported,
            stage_protocol_generation_supported: !self.full_model_only && !self.is_membership,
            stage_status_list_supported: !self.full_model_only && !self.is_membership,
            local_gguf_content_id_supported: !self.full_model_only,
            // Full-model attachments cannot participate in stage scheduling.
            // Ordinary nodes support this independently of build features.
            decode_batch_policy_supported: !self.full_model_only,
            plugin_origin_signature_supported: true,
            persistent_lane_marker_supported: true,
            advertised_model_throughput: data.advertised_model_throughput,
            #[cfg(feature = "payments")]
            lightning_offers: data.lightning_offers.into(),
            cache_affinity: data.cache_affinity,
            latency_ms: None,
            latency_source: None,
            latency_age_ms: None,
            latency_observer_id: None,
            inference_admission_state: data.inference_admission_state,
            // No local claimed-log-head source is wired yet — this node never
            // advertises its own until a companion process is plumbed in.
            claimed_log_head: None,
        }
    }
}

#[cfg(test)]
mod plugin_descriptor_tests {
    use super::*;
    use crate::mesh::ModelWorkloadClass;
    use crate::plugin::{InferenceEndpointRoute, PluginManager};

    fn endpoint(
        plugin_name: &str,
        protocol: Option<&str>,
        models: &[&str],
    ) -> InferenceEndpointRoute {
        InferenceEndpointRoute {
            plugin_name: plugin_name.into(),
            endpoint_id: plugin_name.into(),
            protocol: protocol.map(str::to_string),
            address: "http://127.0.0.1:1".into(),
            models: models.iter().map(|model| (*model).to_string()).collect(),
        }
    }

    fn workload_classes(
        descriptors: &[ServedModelDescriptor],
        model: &str,
    ) -> Vec<ModelWorkloadClass> {
        descriptors
            .iter()
            .filter(|descriptor| descriptor.identity.model_name == model)
            .filter_map(|descriptor| {
                descriptor
                    .metadata
                    .as_ref()
                    .and_then(|metadata| metadata.workload_class)
            })
            .collect()
    }

    #[tokio::test]
    async fn plugin_endpoints_are_advertised_with_their_own_workload_class() {
        let node = Node::new_for_tests(NodeRole::Worker).await.expect("node");
        let manager = PluginManager::for_test_summaries(Vec::new());
        manager
            .set_test_inference_endpoints(vec![
                endpoint("onnx-ocr", Some("document_ocr_v1"), &["datalab/indra-ocr"]),
                endpoint("chat", Some("openai_compatible"), &["acme/chat"]),
                endpoint("legacy", None, &["acme/legacy"]),
            ])
            .await;
        node.set_plugin_manager(manager).await;

        let descriptors = node.local_descriptors_with_plugins().await;
        for (model, class) in [
            ("datalab/indra-ocr", ModelWorkloadClass::DocumentOcr),
            ("acme/chat", ModelWorkloadClass::CausalGeneration),
            ("acme/legacy", ModelWorkloadClass::CausalGeneration),
        ] {
            assert_eq!(
                workload_classes(&descriptors, model),
                vec![class],
                "{model} should advertise the class its endpoint serves"
            );
        }

        let data = node.snapshot_local_announcement_data().await;
        assert_eq!(
            workload_classes(&data.served_model_descriptors, "acme/chat"),
            vec![ModelWorkloadClass::CausalGeneration]
        );
        assert_eq!(
            workload_classes(&data.served_model_descriptors, "datalab/indra-ocr"),
            vec![ModelWorkloadClass::DocumentOcr]
        );
    }

    #[tokio::test]
    async fn shared_model_name_advertises_every_class_its_endpoints_serve() {
        let node = Node::new_for_tests(NodeRole::Worker).await.expect("node");
        let manager = PluginManager::for_test_summaries(Vec::new());
        manager
            .set_test_inference_endpoints(vec![
                endpoint("onnx-ocr", Some("document_ocr_v1"), &["acme/shared"]),
                endpoint("chat", Some("openai_compatible"), &["acme/shared"]),
            ])
            .await;
        node.set_plugin_manager(manager).await;

        let mut classes = workload_classes(
            &node
                .snapshot_local_announcement_data()
                .await
                .served_model_descriptors,
            "acme/shared",
        );
        classes.sort_by_key(|class| format!("{class:?}"));
        assert_eq!(
            classes,
            vec![
                ModelWorkloadClass::CausalGeneration,
                ModelWorkloadClass::DocumentOcr
            ],
            "a name served by both endpoints must advertise both classes, or \
             descriptor-based admission on this node and its peers sees an OCR-only model"
        );
    }
}
