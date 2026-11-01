//! One mesh-scoped catalog shared by native SDK and HTTP projections.

use crate::mesh::{self, ModelRuntimeDescriptor, ServedModelDescriptor};
use crate::network::openai::model_names::public_model_id;
use anyhow::Result;
pub use mesh_llm_types::mesh::ModelWorkloadClass;
pub use mesh_llm_types::models::discovery::{
    MeshModel, MeshModelAvailability, MeshModelKind, MeshModelPage, MeshModelProvider,
    MeshModelQuery,
};
use std::collections::{BTreeMap, BTreeSet};

mod query;
pub(crate) use query::{http_response, parse_params as parse_query_parameters};
pub use query::{lookup, query_models};

/// Snapshot locks must not block the runtime's command and shutdown polling.
pub(crate) fn respond(
    node: mesh::Node,
    mut resp: tokio::sync::oneshot::Sender<Result<Vec<MeshModel>>>,
) -> tokio::task::JoinHandle<()> {
    let shutdown = crate::sdk::embedded_shutdown::SHUTDOWN
        .try_with(Clone::clone)
        .ok();
    tokio::spawn(async move {
        let discover = async move {
            tokio::select! {
                biased;
                _ = crate::sdk::embedded_shutdown::cancelled() => {}
                _ = resp.closed() => {}
                result = collect(&node) => { let _ = resp.send(result); }
            }
        };
        // Spawned tasks do not inherit the runtime's task-local scope.
        if let Some(shutdown) = shutdown {
            crate::sdk::embedded_shutdown::SHUTDOWN
                .scope(shutdown, discover)
                .await;
        } else {
            discover.await;
        }
    })
}

/// Collect only this node's mesh advertisements. Downloads and other memberships
/// are deliberately not queried. Provider facts remain together throughout.
pub(crate) async fn collect(node: &mesh::Node) -> Result<Vec<MeshModel>> {
    let mut catalog = BTreeMap::new();
    collect_local(node, &mut catalog).await?;
    collect_peers(node, &mut catalog).await;
    collect_virtual(node, &mut catalog).await?;
    Ok(catalog.into_values().map(finish_model).collect())
}

async fn collect_local(node: &mesh::Node, catalog: &mut BTreeMap<String, MeshModel>) -> Result<()> {
    let descriptors = node.served_model_descriptors().await;
    let runtimes = node.model_runtime_descriptors.lock().await.clone();
    let hosted = node.hosted_models().await;
    let endpoints = match node.plugin_manager().await {
        Some(manager) => manager.inference_endpoints().await?,
        None => Vec::new(),
    };
    let local_available = !node
        .activity_policy_guard
        .advertisement_decision(node.public_mesh)
        .withdraw_model_availability;
    let mut names: BTreeSet<_> = node.serving_models().await.into_iter().collect();
    names.extend(hosted.iter().cloned());
    // Native models loaded on demand stay listed while their instance is
    // loaded (parity with the chat-compatible listing), not merely while the
    // host keeps them in its persistent hosted set.
    let lifecycle_names: BTreeSet<String> = node
        .native_model_lifecycle
        .get()
        .map(|lifecycle| lifecycle.model_names().cloned().collect())
        .unwrap_or_default();
    names.extend(lifecycle_names.iter().cloned());
    names.extend(
        descriptors
            .iter()
            .map(|descriptor| descriptor.identity.model_name.clone()),
    );
    for name in names {
        if endpoints
            .iter()
            .any(|endpoint| endpoint.models.contains(&name))
            && !descriptors
                .iter()
                .any(|descriptor| descriptor.identity.model_name == name)
            && !runtimes.iter().any(|runtime| runtime.model_name == name)
        {
            continue;
        }
        insert_observations(
            catalog,
            &name,
            &descriptors,
            &runtimes,
            &ProviderContext {
                provider_id: &node.id().to_string(),
                local: true,
                available: local_available
                    && (hosted.contains(&name) || lifecycle_names.contains(&name)),
                endpoint: None,
            },
        );
    }
    {
        for endpoint in endpoints {
            let class = if endpoint.protocol.as_deref() == Some("document_ocr_v1") {
                ModelWorkloadClass::DocumentOcr
            } else {
                ModelWorkloadClass::CausalGeneration
            };
            for name in endpoint.models {
                let descriptor = ServedModelDescriptor {
                    identity: mesh::ServedModelIdentity {
                        model_name: name.clone(),
                        ..Default::default()
                    },
                    metadata: Some(mesh::ServedModelMetadata {
                        workload_class: Some(class),
                        ..Default::default()
                    }),
                    ..Default::default()
                };
                insert_observations(
                    catalog,
                    &name,
                    &[descriptor],
                    &[],
                    &ProviderContext {
                        provider_id: &node.id().to_string(),
                        local: true,
                        available: local_available,
                        endpoint: Some((&endpoint.plugin_name, &endpoint.endpoint_id)),
                    },
                );
            }
        }
    }
    Ok(())
}

async fn collect_peers(node: &mesh::Node, catalog: &mut BTreeMap<String, MeshModel>) {
    let peers = {
        let state = node.state.lock().await;
        state
            .peers
            .values()
            .filter(|peer| peer.is_admitted())
            .map(|peer| (peer.clone(), state.peer_has_observed_liveness(peer)))
            .collect::<Vec<_>>()
    };
    for (peer, live) in peers {
        let mut names: BTreeSet<_> = peer.serving_models.iter().cloned().collect();
        names.extend(peer.hosted_models.iter().cloned());
        names.extend(
            peer.served_model_descriptors
                .iter()
                .map(|descriptor| descriptor.identity.model_name.clone()),
        );
        let paused = matches!(
            peer.inference_admission_state,
            Some(
                crate::proto::node::InferenceAdmissionState::RemotePaused
                    | crate::proto::node::InferenceAdmissionState::AllPaused
            )
        );
        for name in names {
            let available = live && !paused && mesh::routes_http_model(&peer, &name);
            insert_observations(
                catalog,
                &name,
                &peer.served_model_descriptors,
                &peer.served_model_runtime,
                &ProviderContext {
                    provider_id: &peer.id.to_string(),
                    local: false,
                    available,
                    endpoint: None,
                },
            );
        }
    }
}

async fn collect_virtual(
    node: &mesh::Node,
    catalog: &mut BTreeMap<String, MeshModel>,
) -> Result<()> {
    let Some(manager) = node.plugin_manager().await else {
        return Ok(());
    };
    let has_llm = catalog.values().any(MeshModel::is_available_llm);
    for route in manager.virtual_models().await? {
        let generation = route
            .output_modalities
            .iter()
            .any(|modality| modality == "text");
        let available = generation && (!route.requires_candidates || has_llm);
        let class = generation.then_some(ModelWorkloadClass::CausalGeneration);
        let provider = MeshModelProvider {
            provider_id: node.id().to_string(),
            local: true,
            endpoint_id: None,
            plugin: Some(route.plugin_name.clone()),
            workload_class: class,
            available,
            operations: operations(class),
            capabilities: None,
            metadata: None,
            runtime: None,
            thinking: None,
            requires: None,
        };
        catalog.insert(route.model_id.clone(), MeshModel {
            id: route.model_id.clone(), display_name: route.model_id, kind: MeshModelKind::Virtual,
            workload_class: class, availability: availability(available), operations: provider.operations.clone(),
            metadata: None, providers: vec![provider],
            virtual_model: Some(serde_json::json!({ "plugin": route.plugin_name,
                "input_modalities": route.input_modalities, "output_modalities": route.output_modalities,
                "supports_tools": route.supports_tools, "supports_streaming": route.supports_streaming })),
        });
    }
    Ok(())
}

fn normalized_id(name: &str, descriptor: Option<&ServedModelDescriptor>) -> String {
    let (base, profile) = name.rsplit_once('#').unwrap_or((name, ""));
    public_model_id(base, descriptor, profile)
}

struct ProviderContext<'a> {
    provider_id: &'a str,
    local: bool,
    available: bool,
    endpoint: Option<(&'a str, &'a str)>,
}

fn insert_observations(
    catalog: &mut BTreeMap<String, MeshModel>,
    name: &str,
    descriptors: &[ServedModelDescriptor],
    runtimes: &[ModelRuntimeDescriptor],
    context: &ProviderContext<'_>,
) {
    let matching: Vec<_> = descriptors
        .iter()
        .filter(|descriptor| {
            descriptor.identity.model_name == name
                || descriptor.identity.model_name
                    == name.rsplit_once('#').map_or(name, |(base, _)| base)
        })
        .collect();
    if matching.is_empty() {
        insert_provider(catalog, name, None, runtimes, context);
    } else {
        for descriptor in matching {
            insert_provider(catalog, name, Some(descriptor), runtimes, context);
        }
    }
}

fn insert_provider(
    catalog: &mut BTreeMap<String, MeshModel>,
    name: &str,
    descriptor: Option<&ServedModelDescriptor>,
    runtimes: &[ModelRuntimeDescriptor],
    context: &ProviderContext<'_>,
) {
    let ProviderContext {
        provider_id,
        local,
        available,
        endpoint,
    } = *context;
    let id = normalized_id(name, descriptor);
    let class = descriptor
        .and_then(|descriptor| descriptor.metadata.as_ref())
        .and_then(|metadata| metadata.workload_class);
    let runtime = runtimes
        .iter()
        .find(|runtime| {
            class != Some(ModelWorkloadClass::DocumentOcr)
                && runtime.model_name == name
                && descriptor.is_none_or(|descriptor| {
                    runtime.identity_hash.is_none()
                        || runtime.identity_hash == descriptor.identity.identity_hash
                })
        })
        .cloned()
        .map(|mut runtime| {
            runtime.model_name = id.clone();
            runtime
        });
    let available = available && runtime.as_ref().is_none_or(|runtime| runtime.ready);
    let provider = MeshModelProvider {
        provider_id: provider_id.to_string(),
        local,
        endpoint_id: endpoint.map(|(_, endpoint)| endpoint.to_string()),
        plugin: endpoint.map(|(plugin, _)| plugin.to_string()),
        workload_class: class,
        available,
        operations: descriptor_operations(descriptor),
        capabilities: descriptor
            .filter(|descriptor| descriptor.capabilities_known)
            .map(|descriptor| descriptor.capabilities),
        metadata: descriptor.and_then(|descriptor| descriptor.metadata.clone()),
        runtime,
        thinking: if local && endpoint.is_none() {
            crate::inference::skippy::local_thinking(name)
        } else {
            None
        },
        requires: descriptor.and_then(|descriptor| model_requirements_json(&descriptor.requires)),
    };
    let model = catalog.entry(id.clone()).or_insert_with(|| MeshModel {
        display_name: id.clone(),
        id,
        kind: MeshModelKind::Model,
        workload_class: class,
        availability: MeshModelAvailability::Unavailable,
        operations: Vec::new(),
        metadata: None,
        providers: Vec::new(),
        virtual_model: None,
    });
    if class == Some(ModelWorkloadClass::CausalGeneration) {
        model.kind = MeshModelKind::Llm;
    }
    if !model.providers.contains(&provider) {
        model.providers.push(provider);
    }
}

fn finish_model(mut model: MeshModel) -> MeshModel {
    model.providers.sort_by(|a, b| {
        a.provider_id
            .cmp(&b.provider_id)
            .then_with(|| a.endpoint_id.cmp(&b.endpoint_id))
            .then_with(|| format!("{:?}", a.workload_class).cmp(&format!("{:?}", b.workload_class)))
    });
    model.availability = availability(model.providers.iter().any(|provider| provider.available));
    model.operations = model
        .providers
        .iter()
        .flat_map(|provider| provider.operations.iter().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if let Some(first) = model.providers.first() {
        model.workload_class = model
            .providers
            .iter()
            .all(|provider| provider.workload_class == first.workload_class)
            .then_some(first.workload_class)
            .flatten();
        model.metadata = model
            .providers
            .iter()
            .all(|provider| provider.metadata == first.metadata)
            .then(|| first.metadata.clone())
            .flatten();
    }
    model
}

fn availability(available: bool) -> MeshModelAvailability {
    if available {
        MeshModelAvailability::Available
    } else {
        MeshModelAvailability::Unavailable
    }
}

fn operations(class: Option<ModelWorkloadClass>) -> Vec<String> {
    let names: &[&str] = match class {
        Some(ModelWorkloadClass::CausalGeneration) => &["chat", "completions", "responses"],
        Some(ModelWorkloadClass::Embedding) => &["embedding"],
        Some(ModelWorkloadClass::Rerank) => &["rerank"],
        Some(ModelWorkloadClass::SpeechSynthesis) => &["speech"],
        Some(ModelWorkloadClass::SpeechRecognition) => &["live_transcription"],
        Some(ModelWorkloadClass::DocumentOcr) => &["document_ocr"],
        Some(ModelWorkloadClass::ImageTextSimilarity) => &["image_text_similarity"],
        Some(ModelWorkloadClass::DepthEstimation) => &["depth_estimation"],
        Some(ModelWorkloadClass::Decision) => &[],
        Some(ModelWorkloadClass::EncoderDecoder | ModelWorkloadClass::Unknown) | None => &[],
    };
    names.iter().map(|name| (*name).to_string()).collect()
}

fn descriptor_operations(descriptor: Option<&ServedModelDescriptor>) -> Vec<String> {
    let class = descriptor
        .and_then(|descriptor| descriptor.metadata.as_ref())
        .and_then(|metadata| metadata.workload_class);
    let mut names = operations(class);
    if let Some(descriptor) = descriptor {
        if crate::network::openai::workload_routing::descriptor_supports_audio_upload(descriptor) {
            names.extend(["transcription".to_string(), "translation".to_string()]);
        }
        if crate::network::openai::workload_routing::descriptor_supports_system_one(descriptor) {
            names.push("system_one".to_string());
        }
    }
    names.sort();
    names.dedup();
    names
}

#[cfg(test)]
mod tests;

/// The plugin channels a model requires, with the terms its provider's plugin
/// attached, republished verbatim so clients and plugins can list and compare
/// what the mesh is offering. The host does not interpret the terms: they are
/// UTF-8 text when the plugin wrote text, otherwise base64 with
/// `terms_encoding: "base64"`. Absent when the model is ungated.
fn model_requirements_json(
    requires: &[mesh_llm_types::mesh::ChannelRequirement],
) -> Option<serde_json::Value> {
    use base64::Engine as _;
    if requires.is_empty() {
        return None;
    }
    let entries: Vec<serde_json::Value> = requires
        .iter()
        .map(|requirement| {
            let mut entry = serde_json::Map::new();
            entry.insert("channel".into(), requirement.channel.clone().into());
            if !requirement.terms.is_empty() {
                match std::str::from_utf8(&requirement.terms) {
                    Ok(text) => {
                        entry.insert("terms".into(), text.into());
                    }
                    Err(_) => {
                        entry.insert(
                            "terms".into(),
                            base64::engine::general_purpose::STANDARD
                                .encode(&requirement.terms)
                                .into(),
                        );
                        entry.insert("terms_encoding".into(), "base64".into());
                    }
                }
            }
            serde_json::Value::Object(entry)
        })
        .collect();
    Some(serde_json::Value::Array(entries))
}

#[cfg(test)]
mod requirement_tests {
    use super::model_requirements_json;
    use mesh_llm_types::mesh::ChannelRequirement;

    /// Channel requirements and the terms a plugin attached are republished
    /// verbatim per provider, so a client or plugin can list and compare offers
    /// without the host knowing what the terms mean. Ungated models carry none.
    #[test]
    fn requirements_are_republished_verbatim_and_absent_when_ungated() {
        let requires = model_requirements_json(&[
            ChannelRequirement {
                channel: "example.offer.v1".into(),
                terms: br#"{"unit":"token","rate":3}"#.to_vec(),
            },
            ChannelRequirement {
                channel: "example.binary.v1".into(),
                terms: vec![0xff, 0x00, 0x01],
            },
        ])
        .unwrap();
        assert_eq!(requires[0]["channel"], "example.offer.v1");
        assert_eq!(requires[0]["terms"], r#"{"unit":"token","rate":3}"#);
        assert!(requires[0].get("terms_encoding").is_none());
        assert_eq!(requires[1]["channel"], "example.binary.v1");
        assert_eq!(requires[1]["terms"], "/wAB");
        assert_eq!(requires[1]["terms_encoding"], "base64");
        assert!(model_requirements_json(&[]).is_none());
    }
}
