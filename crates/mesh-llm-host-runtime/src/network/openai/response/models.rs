use crate::mesh;
use crate::network::openai::request_parse::public_model_id;
use crate::network::openai::routing_rank::{capabilities_for_model, descriptor_for_model};
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;

pub async fn send_models_list_with_descriptors(
    mut stream: TcpStream,
    models: &[String],
    descriptors: &[mesh::ServedModelDescriptor],
    runtimes: &[mesh::ModelRuntimeDescriptor],
) -> std::io::Result<()> {
    let body = models_list_json(models, descriptors, runtimes).to_string();
    let resp = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}",
        body.len(),
        body
    );
    stream.write_all(resp.as_bytes()).await?;
    stream.shutdown().await?;
    Ok(())
}

fn models_list_json(
    models: &[String],
    descriptors: &[mesh::ServedModelDescriptor],
    runtimes: &[mesh::ModelRuntimeDescriptor],
) -> serde_json::Value {
    let mut seen = std::collections::HashSet::new();
    let mut data: Vec<serde_json::Value> = models
        .iter()
        .filter_map(|m| {
            let (base_model, profile) =
                crate::network::openai::ingress::parse_model_with_profile(m);
            let descriptor = descriptor_for_model(descriptors, base_model);
            let public_id = public_model_id(base_model, descriptor, profile);
            if !seen.insert(public_id.clone()) {
                return None;
            }
            let capabilities = capabilities_for_model(base_model, descriptors);
            let has_multimodal = capabilities.supports_multimodal_runtime();
            let has_vision = capabilities.supports_vision_runtime();
            let has_audio = capabilities.supports_audio_runtime();
            let mut caps = vec!["text"];
            if has_multimodal {
                caps.push("multimodal");
            }
            if has_vision {
                caps.push("vision");
            }
            if has_audio {
                caps.push("audio");
            }
            if capabilities.reasoning_label().is_some() {
                caps.push("reasoning");
            }
            let display_name = if public_id == *m {
                crate::models::installed_model_display_name(base_model)
            } else {
                public_id.clone()
            };
            let mut model = serde_json::json!({
                "id": public_id,
                "display_name": display_name,
                "object": "model",
                "owned_by": "mesh-llm",
                "capabilities": caps,
                "multimodal_status": capabilities.multimodal_status(),
                "vision_status": capabilities.vision_status(),
                "audio_status": capabilities.audio_status(),
                "reasoning_status": capabilities.reasoning_status(),
            });
            if let Some(metadata) = model_metadata_json(base_model, descriptor, runtimes)
                && let Some(object) = model.as_object_mut()
            {
                object.insert("metadata".to_string(), metadata);
            }
            Some(model)
        })
        .collect();

    if crate::network::openai::moa_gateway::context_selection::should_advertise_virtual_mesh(models)
        && seen.insert(mesh_mixture_of_agents::VIRTUAL_MODEL_NAME.to_string())
    {
        let mut model = serde_json::json!({
            "id": mesh_mixture_of_agents::VIRTUAL_MODEL_NAME,
            "display_name": "Mesh (MoA)",
            "object": "model",
            "owned_by": "mesh-llm",
            "capabilities": ["text"],
            "multimodal_status": "unsupported",
            "vision_status": "unsupported",
            "audio_status": "unsupported",
            "reasoning_status": "unknown",
        });
        if let Some(context_length) =
            crate::network::openai::moa_gateway::context_selection::virtual_mesh_context_length(
                models, runtimes,
            )
            && let Some(object) = model.as_object_mut()
        {
            object.insert(
                "metadata".to_string(),
                serde_json::json!({ "context_length": context_length }),
            );
        }
        data.push(model);
    }

    serde_json::json!({ "object": "list", "data": data })
}

fn model_metadata_json(
    model_name: &str,
    descriptor: Option<&mesh::ServedModelDescriptor>,
    runtimes: &[mesh::ModelRuntimeDescriptor],
) -> Option<serde_json::Value> {
    let mut metadata = serde_json::Map::new();
    let descriptor_metadata = descriptor.and_then(|descriptor| descriptor.metadata.as_ref());
    if let Some(value) = descriptor_metadata.and_then(|metadata| metadata.architecture.as_ref()) {
        metadata.insert("architecture".to_string(), serde_json::json!(value));
    }
    if let Some(value) = descriptor_metadata.and_then(|metadata| metadata.parameter_size.as_ref()) {
        metadata.insert("parameter_size".to_string(), serde_json::json!(value));
    }
    if let Some(value) = descriptor_metadata.and_then(|metadata| metadata.parameter_count_b)
        && value.is_finite()
    {
        metadata.insert("parameter_count_b".to_string(), serde_json::json!(value));
    }
    if let Some(value) = descriptor_metadata.and_then(|metadata| metadata.quant.as_ref()) {
        metadata.insert("quant".to_string(), serde_json::json!(value));
    }
    if let Some(contexts) = runtime_context_lengths_for_model(model_name, runtimes) {
        metadata.insert(
            "context_length".to_string(),
            serde_json::json!(contexts.min),
        );
        if contexts.max != contexts.min {
            metadata.insert(
                "max_context_length".to_string(),
                serde_json::json!(contexts.max),
            );
        }
    }
    if let Some(value) = descriptor_metadata.and_then(|metadata| metadata.native_context_length) {
        metadata.insert(
            "native_context_length".to_string(),
            serde_json::json!(value),
        );
    }
    if let Some(value) = descriptor_metadata.and_then(|metadata| metadata.tokenizer.as_ref()) {
        metadata.insert("tokenizer".to_string(), serde_json::json!(value));
    }
    if let Some(value) = descriptor_metadata.and_then(|metadata| metadata.layer_count) {
        metadata.insert("layer_count".to_string(), serde_json::json!(value));
    }
    if let Some(value) = descriptor_metadata.and_then(|metadata| metadata.embedding_size) {
        metadata.insert("embedding_size".to_string(), serde_json::json!(value));
    }
    if let Some(value) = descriptor_metadata.and_then(|metadata| metadata.head_count) {
        metadata.insert("head_count".to_string(), serde_json::json!(value));
    }
    if let Some(value) = descriptor_metadata.and_then(|metadata| metadata.kv_head_count) {
        metadata.insert("kv_head_count".to_string(), serde_json::json!(value));
    }
    if let Some(value) = descriptor_metadata.and_then(|metadata| metadata.expert_count) {
        metadata.insert("expert_count".to_string(), serde_json::json!(value));
    }
    if let Some(value) = descriptor_metadata.and_then(|metadata| metadata.active_expert_count) {
        metadata.insert("active_expert_count".to_string(), serde_json::json!(value));
    }
    (!metadata.is_empty()).then_some(serde_json::Value::Object(metadata))
}

struct RuntimeContextLengths {
    min: u32,
    max: u32,
}

fn runtime_context_lengths_for_model(
    model_name: &str,
    runtimes: &[mesh::ModelRuntimeDescriptor],
) -> Option<RuntimeContextLengths> {
    let mut lengths = runtimes
        .iter()
        .filter(|runtime| runtime.model_name == model_name)
        .filter_map(mesh::ModelRuntimeDescriptor::advertised_context_length);
    let first = lengths.next()?;
    let (min, max) = lengths.fold((first, first), |(min, max), value| {
        (min.min(value), max.max(value))
    });
    Some(RuntimeContextLengths { min, max })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hf_descriptor(model_name: &str) -> mesh::ServedModelDescriptor {
        mesh::ServedModelDescriptor {
            identity: mesh::ServedModelIdentity {
                model_name: model_name.to_string(),
                source_kind: mesh::ModelSourceKind::HuggingFace,
                repository: Some("tiiuae/Falcon-H1-1.5B-Instruct-GGUF".to_string()),
                revision: Some("0d3a6cfe25fb4eeab0153fb8623aac5b69d6bd0a".to_string()),
                artifact: Some("Falcon-H1-1.5B-Instruct-Q4_K_M.gguf".to_string()),
                canonical_ref: Some(
                    "tiiuae/Falcon-H1-1.5B-Instruct-GGUF@0d3a6cfe25fb4eeab0153fb8623aac5b69d6bd0a/Falcon-H1-1.5B-Instruct-Q4_K_M.gguf"
                        .to_string(),
                ),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn catalog_model_ref_descriptor(model_name: &str) -> mesh::ServedModelDescriptor {
        mesh::ServedModelDescriptor {
            identity: mesh::ServedModelIdentity {
                model_name: model_name.to_string(),
                source_kind: mesh::ModelSourceKind::Catalog,
                canonical_ref: Some("tiiuae/Falcon-H1-1.5B-Instruct-GGUF:Q4_K_M".to_string()),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn local_gguf_descriptor(model_name: &str) -> mesh::ServedModelDescriptor {
        mesh::ServedModelDescriptor {
            identity: mesh::ServedModelIdentity {
                model_name: model_name.to_string(),
                source_kind: mesh::ModelSourceKind::LocalGguf,
                local_file_name: Some(format!("{model_name}.gguf")),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn local_gguf_descriptor_with_capabilities(
        model_name: &str,
        capabilities: crate::models::ModelCapabilities,
    ) -> mesh::ServedModelDescriptor {
        mesh::ServedModelDescriptor {
            capabilities_known: true,
            capabilities,
            ..local_gguf_descriptor(model_name)
        }
    }
    #[test]
    fn models_list_uses_public_huggingface_model_ref_ids() {
        let models = vec!["Falcon-H1-1.5B-Instruct-Q4_K_M".to_string()];
        let descriptors = vec![hf_descriptor(&models[0])];

        let body = models_list_json(&models, &descriptors, &[]);

        assert_eq!(
            body["data"][0]["id"],
            "tiiuae/Falcon-H1-1.5B-Instruct-GGUF:Q4_K_M"
        );
        assert_eq!(
            body["data"][0]["display_name"],
            "tiiuae/Falcon-H1-1.5B-Instruct-GGUF:Q4_K_M"
        );
        assert_eq!(body["data"][0]["owned_by"], "mesh-llm");
    }

    #[test]
    fn models_list_id_preserves_quant_suffix_when_descriptor_has_no_artifact() {
        // Regression for PR #566 review feedback: the gateway's view of a
        // model's public ID must include enough information to route a
        // request back to that exact model. When a `ServedModelDescriptor`
        // for a HuggingFace model has no `artifact` field (because the
        // descriptor was built without inspecting the GGUF file on disk),
        // `public_huggingface_model_ref` collapses the public ID to just
        // the repo name — dropping the quant-tag suffix the internal
        // `model_name` carries. The model is then advertised in `/v1/models`
        // under a shorter ID than the resolver knows how to route.
        //
        // Symptom on a real 2-node mesh: the studio's Qwen3-0.6B-GGUF
        // shows as `unsloth/Qwen3-0.6B-GGUF:BF16` (descriptor has
        // artifact), but the gateway-local Qwen2.5-3B-Instruct-GGUF
        // shows as `Qwen/Qwen2.5-3B-Instruct-GGUF` (descriptor has no
        // artifact). A client doing the natural thing — read /v1/models,
        // call /v1/chat/completions with the listed id — then 404s on
        // remote models because the resolver doesn't know the short id.
        //
        // Acceptable behaviour: the public ID either round-trips to the
        // same model, OR includes the quant suffix the internal name
        // carries.
        let models = vec!["Qwen/Qwen2.5-3B-Instruct-GGUF:qwen2.5-3b-instruct-q4_k_m".to_string()];
        let descriptor = mesh::ServedModelDescriptor {
            identity: mesh::ServedModelIdentity {
                model_name: models[0].clone(),
                source_kind: mesh::ModelSourceKind::HuggingFace,
                repository: Some("Qwen/Qwen2.5-3B-Instruct-GGUF".to_string()),
                // No artifact — this is the field whose absence loses the
                // quant suffix.
                artifact: None,
                ..Default::default()
            },
            ..Default::default()
        };
        let descriptors = vec![descriptor];

        let body = models_list_json(&models, &descriptors, &[]);
        let public_id = body["data"][0]["id"].as_str().unwrap_or_default();

        // The public ID must NOT silently drop the quant suffix that the
        // internal model_name carries. Acceptable IDs:
        //   * the full internal name, OR
        //   * the repo with a quant tag we can route back to.
        assert!(
            public_id == models[0]
                || public_id
                    .strip_prefix("Qwen/Qwen2.5-3B-Instruct-GGUF:")
                    .is_some_and(|tag| !tag.is_empty()),
            "public id must keep enough information to route back; got {public_id:?}, \
             internal model_name was {:?}",
            models[0]
        );
    }

    #[test]
    fn models_list_uses_catalog_model_ref_ids() {
        let models = vec!["Falcon-H1-1.5B-Instruct-Q4_K_M".to_string()];
        let descriptors = vec![catalog_model_ref_descriptor(&models[0])];

        let body = models_list_json(&models, &descriptors, &[]);

        assert_eq!(
            body["data"][0]["id"],
            "tiiuae/Falcon-H1-1.5B-Instruct-GGUF:Q4_K_M"
        );
    }

    #[test]
    fn models_list_keeps_local_gguf_model_name_ids() {
        let models = vec!["smollm2-a".to_string()];
        let descriptors = vec![local_gguf_descriptor(&models[0])];

        let body = models_list_json(&models, &descriptors, &[]);

        assert_eq!(body["data"][0]["id"], "smollm2-a");
        assert_eq!(body["data"][0]["display_name"], "smollm2-a");
    }

    #[test]
    fn models_list_reports_model_metadata() {
        let models = vec!["Qwen3-32B-Q4_K_M".to_string()];
        let mut descriptor = local_gguf_descriptor(&models[0]);
        descriptor.metadata = Some(mesh::ServedModelMetadata {
            architecture: Some("qwen3".to_string()),
            parameter_size: Some("32B".to_string()),
            parameter_count_b: Some(32.0),
            quant: Some("Q4_K_M".to_string()),
            native_context_length: Some(32_768),
            tokenizer: Some("gpt2".to_string()),
            layer_count: Some(64),
            embedding_size: Some(5120),
            head_count: Some(40),
            kv_head_count: Some(8),
            expert_count: Some(128),
            active_expert_count: Some(8),
        });
        let runtimes = vec![mesh::ModelRuntimeDescriptor {
            model_name: models[0].clone(),
            identity_hash: None,
            context_length: Some(65_536),
            ready: true,
        }];

        let body = models_list_json(&models, &[descriptor], &runtimes);
        let metadata = &body["data"][0]["metadata"];

        assert_eq!(metadata["architecture"], "qwen3");
        assert_eq!(metadata["parameter_size"], "32B");
        assert_eq!(metadata["parameter_count_b"], 32.0);
        assert_eq!(metadata["quant"], "Q4_K_M");
        assert_eq!(metadata["context_length"], 65_536);
        assert_eq!(metadata["native_context_length"], 32_768);
        assert_eq!(metadata["tokenizer"], "gpt2");
        assert_eq!(metadata["layer_count"], 64);
        assert_eq!(metadata["embedding_size"], 5120);
        assert_eq!(metadata["head_count"], 40);
        assert_eq!(metadata["kv_head_count"], 8);
        assert_eq!(metadata["expert_count"], 128);
        assert_eq!(metadata["active_expert_count"], 8);
    }

    #[test]
    fn models_list_uses_route_safe_context_for_duplicate_runtimes() {
        let models = vec!["Qwen3.5-9B-Q4_K_M".to_string()];
        let runtimes = vec![
            mesh::ModelRuntimeDescriptor {
                model_name: models[0].clone(),
                identity_hash: None,
                context_length: Some(32_768),
                ready: true,
            },
            mesh::ModelRuntimeDescriptor {
                model_name: models[0].clone(),
                identity_hash: None,
                context_length: Some(131_072),
                ready: true,
            },
        ];

        let body = models_list_json(&models, &[], &runtimes);
        let metadata = &body["data"][0]["metadata"];

        assert_eq!(metadata["context_length"], 32_768);
        assert_eq!(metadata["max_context_length"], 131_072);
    }

    #[test]
    fn models_list_advertises_virtual_mesh_when_moa_has_two_models() {
        let models = vec!["fast-8b".to_string(), "strong-32b".to_string()];
        let runtimes = vec![
            mesh::ModelRuntimeDescriptor {
                model_name: "fast-8b".to_string(),
                identity_hash: None,
                context_length: Some(16_384),
                ready: true,
            },
            mesh::ModelRuntimeDescriptor {
                model_name: "strong-32b".to_string(),
                identity_hash: None,
                context_length: Some(65_536),
                ready: true,
            },
        ];

        let body = models_list_json(&models, &[], &runtimes);
        let mesh = body["data"]
            .as_array()
            .unwrap()
            .iter()
            .find(|model| model["id"] == mesh_mixture_of_agents::VIRTUAL_MODEL_NAME)
            .expect("virtual mesh model should be listed");

        assert_eq!(mesh["display_name"], "Mesh (MoA)");
        assert_eq!(mesh["metadata"]["context_length"], 16_384);
    }

    #[test]
    fn models_list_does_not_invent_virtual_mesh_context() {
        let models = vec!["unknown-a".to_string(), "unknown-b".to_string()];

        let body = models_list_json(&models, &[], &[]);
        let mesh = body["data"]
            .as_array()
            .unwrap()
            .iter()
            .find(|model| model["id"] == mesh_mixture_of_agents::VIRTUAL_MODEL_NAME)
            .expect("virtual mesh model should be listed");

        assert!(mesh.get("metadata").is_none());
    }

    #[test]
    fn models_list_uses_descriptor_capabilities_not_filename_heuristics() {
        let models = vec!["Qwen3VL-2B-Instruct-Q4_K_M".to_string()];
        let descriptors = vec![local_gguf_descriptor_with_capabilities(
            &models[0],
            crate::models::ModelCapabilities::default(),
        )];

        let body = models_list_json(&models, &descriptors, &[]);

        assert_eq!(body["data"][0]["capabilities"], serde_json::json!(["text"]));
        assert_eq!(body["data"][0]["vision_status"], "none");
        assert_eq!(body["data"][0]["multimodal_status"], "none");
    }

    #[test]
    fn models_list_uses_static_fallback_for_unknown_descriptor_capabilities() {
        let models = vec!["Qwen3VL-2B-Instruct-Q4_K_M".to_string()];
        let descriptors = vec![local_gguf_descriptor(&models[0])];

        let body = models_list_json(&models, &descriptors, &[]);
        let capabilities = body["data"][0]["capabilities"].as_array().unwrap();

        assert!(capabilities.iter().any(|cap| cap == "multimodal"));
        assert!(capabilities.iter().any(|cap| cap == "vision"));
        assert_eq!(body["data"][0]["vision_status"], "supported");
        assert_eq!(body["data"][0]["multimodal_status"], "supported");
    }

    #[test]
    fn models_list_reports_runtime_verified_projector_capabilities() {
        let models = vec!["Qwen3VL-2B-Instruct-Q4_K_M".to_string()];
        let descriptors = vec![local_gguf_descriptor_with_capabilities(
            &models[0],
            crate::models::ModelCapabilities {
                multimodal: true,
                vision: crate::models::CapabilityLevel::Supported,
                ..Default::default()
            },
        )];

        let body = models_list_json(&models, &descriptors, &[]);
        let capabilities = body["data"][0]["capabilities"].as_array().unwrap();

        assert!(capabilities.iter().any(|cap| cap == "multimodal"));
        assert!(capabilities.iter().any(|cap| cap == "vision"));
        assert_eq!(body["data"][0]["vision_status"], "supported");
        assert_eq!(body["data"][0]["multimodal_status"], "supported");
    }
}
