//! Hold native residency through the complete response, including SSE forwarding.
use super::*;

/// Defer a configured cold model's workload decision until its verified descriptor exists.
pub(super) fn named_model_route(
    node: &mesh::Node,
    request: &proxy::BufferedHttpRequest,
    model: &str,
    descriptors: &[mesh::ServedModelDescriptor],
) -> AutoRouteResolution {
    if node
        .native_model_lifecycle
        .get()
        .is_some_and(|service| service.manages(model))
    {
        AutoRouteResolution::Continue {
            effective_model: Some(model.to_owned()),
            classification: None,
        }
    } else {
        explicit_model_route(request, model, descriptors)
    }
}

/// Hold a managed request lease through dispatch while preserving membership peer isolation.
pub(super) async fn route_request(
    local_only: bool,
    stream: ClientStream,
    request: &mut proxy::BufferedHttpRequest,
    ctx: &IngressRouteContext<'_>,
    model: Option<&str>,
    required_tokens: Option<u32>,
    observer: OpenAiRouteObserver<'_>,
) -> proxy::RouteDispatchOutcome {
    let lifecycle = ctx
        .node
        .native_model_lifecycle
        .get()
        .filter(|lifecycle| model.is_some_and(|name| lifecycle.manages(name)));
    let (Some(_), Some(model_name)) = (lifecycle, model) else {
        return route_prepared_request(
            local_only,
            stream,
            request,
            ctx,
            model,
            required_tokens,
            observer,
        )
        .await;
    };
    let (target, excluded) = match parse_mesh_routing_headers(request)
        .and_then(|routing| twin_bracket_id(request).map(|_| routing))
    {
        Ok(routing) => routing,
        Err(message) => {
            return response_outcome(
                400,
                proxy::send_400_observed(stream, &message, observer).await,
            );
        }
    };
    if mesh_headers_force_remote(ctx.node.id(), target, &excluded) {
        return route_prepared_request(
            local_only,
            stream,
            request,
            ctx,
            model,
            required_tokens,
            observer,
        )
        .await;
    }
    if let Some(targets) = ready_targets(
        ctx,
        model_name,
        &request.client_path,
        local_only || target.is_some(),
        &excluded,
    )
    .await
    {
        let route = IngressRouteContext {
            targets: &targets,
            requested_by_node_id: ctx.requested_by_node_id.clone(),
            ..*ctx
        };
        return route_prepared_request(
            local_only,
            stream,
            request,
            &route,
            model,
            required_tokens,
            observer,
        )
        .await;
    }
    let stream = match try_plugin_when_no_mesh_candidate_serves(
        local_only, stream, request, ctx, model_name, &excluded, observer,
    )
    .await
    {
        Ok(outcome) => return outcome,
        Err(stream) => stream,
    };
    dispatch_cold(
        local_only,
        stream,
        request,
        ctx,
        model_name,
        required_tokens,
        observer,
    )
    .await
}

/// Consult workload-compatible hosts in this membership before demanding local memory.
async fn ready_targets(
    ctx: &IngressRouteContext<'_>,
    model: &str,
    path: &str,
    local_only: bool,
    excluded: &[iroh::EndpointId],
) -> Option<election::ModelTargets> {
    let candidates = workload_routing::ingress_candidates(ctx.node, model, path, ctx.targets)
        .await
        .into_iter()
        .filter(|candidate| match candidate {
            election::InferenceTarget::Local(_) => true,
            election::InferenceTarget::Remote(peer) => {
                !local_only
                    && !excluded.contains(peer)
                    && !ctx
                        .node
                        .peer_blocks
                        .is_blocked(peer, crate::network::peer_blocks::now_ms())
            }
            election::InferenceTarget::None => false,
        })
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        return None;
    }
    let mut targets = ctx.targets.clone();
    targets.targets.insert(model.to_owned(), candidates);
    Some(targets)
}

/// Load a cold named model, verify its descriptor, then dispatch to its owned instance.
async fn dispatch_cold(
    local_only: bool,
    stream: ClientStream,
    request: &mut proxy::BufferedHttpRequest,
    ctx: &IngressRouteContext<'_>,
    model_name: &str,
    required_tokens: Option<u32>,
    observer: OpenAiRouteObserver<'_>,
) -> proxy::RouteDispatchOutcome {
    let model = Some(model_name);
    let Some(lifecycle) = ctx.node.native_model_lifecycle.get() else {
        return route_prepared_request(
            local_only,
            stream,
            request,
            ctx,
            model,
            required_tokens,
            observer,
        )
        .await;
    };
    let _lease = match lifecycle.acquire(model_name).await {
        Ok(lease) => lease,
        Err(error) => {
            tracing::warn!(model = model_name, %error, "native model not ready");
            return cold_failure(
                local_only,
                stream,
                request,
                ctx,
                model_name,
                required_tokens,
                observer,
            )
            .await;
        }
    };
    let execution_model = lifecycle.execution_model(model_name);
    let mut targets = refresh_local_targets(
        ctx.targets,
        &lifecycle.targets_for(model_name),
        &execution_model,
    );
    let descriptors = ctx.node.all_served_model_descriptors().await;
    if let Some(workload) = required_request_workload(&request.client_path)
        && !model_satisfies_request(&execution_model, &request.client_path, &descriptors)
    {
        return send_workload_unsupported(stream, workload, &request.client_path, observer).await;
    }
    if local_only {
        for candidates in targets.targets.values_mut() {
            candidates.retain(|target| !matches!(target, election::InferenceTarget::Remote(_)));
        }
    }
    let route = IngressRouteContext {
        targets: &targets,
        requested_by_node_id: ctx.requested_by_node_id.clone(),
        ..*ctx
    };
    if execution_model != model_name {
        proxy::rewrite_model_field(request, &execution_model);
    }
    route_prepared_request(
        local_only,
        stream,
        request,
        &route,
        Some(&execution_model),
        required_tokens,
        observer,
    )
    .await
}

/// Keep eligible peers and plugins available after a failed local cold load.
pub(super) async fn cold_failure(
    local_only: bool,
    stream: ClientStream,
    request: &mut proxy::BufferedHttpRequest,
    ctx: &IngressRouteContext<'_>,
    model: &str,
    required_tokens: Option<u32>,
    observer: OpenAiRouteObserver<'_>,
) -> proxy::RouteDispatchOutcome {
    if let Ok((target, excluded)) = parse_mesh_routing_headers(request)
        && let Some(targets) = ready_targets(
            ctx,
            model,
            &request.client_path,
            local_only || target.is_some(),
            &excluded,
        )
        .await
    {
        let route = IngressRouteContext {
            targets: &targets,
            requested_by_node_id: ctx.requested_by_node_id.clone(),
            ..*ctx
        };
        return route_prepared_request(
            local_only,
            stream,
            request,
            &route,
            Some(model),
            required_tokens,
            observer,
        )
        .await;
    }
    execution::local_model_unavailable(stream, request, ctx, model, observer).await
}

/// Refresh local residency without importing peers from another mesh membership.
fn refresh_local_targets(
    existing: &election::ModelTargets,
    fresh: &election::ModelTargets,
    model: &str,
) -> election::ModelTargets {
    // Memberships share local residency, but their peer routing tables remain
    // separate. Refresh local targets only; never import primary-mesh peers.
    let mut targets = existing.clone();
    let candidates = targets.targets.entry(model.to_owned()).or_default();
    candidates.retain(|target| !matches!(target, election::InferenceTarget::Local(_)));
    candidates.extend(
        fresh
            .candidates(model)
            .into_iter()
            .filter(|target| matches!(target, election::InferenceTarget::Local(_))),
    );
    targets
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refreshing_residency_keeps_membership_peers_separate() {
        use election::{InferenceTarget, ModelTargets};
        let member_peer = iroh::SecretKey::from_bytes(&[1; 32]).public();
        let primary_peer = iroh::SecretKey::from_bytes(&[2; 32]).public();
        let mut membership = ModelTargets::default();
        membership.targets.insert(
            "test".into(),
            vec![
                InferenceTarget::Local(1111),
                InferenceTarget::Remote(member_peer),
            ],
        );
        let mut primary = ModelTargets::default();
        primary.targets.insert(
            "test".into(),
            vec![
                InferenceTarget::Local(2222),
                InferenceTarget::Remote(primary_peer),
            ],
        );
        let refreshed = refresh_local_targets(&membership, &primary, "test");
        assert_eq!(
            refreshed.candidates("test"),
            vec![
                InferenceTarget::Remote(member_peer),
                InferenceTarget::Local(2222)
            ]
        );
        let unloaded = refresh_local_targets(&refreshed, &ModelTargets::default(), "test");
        assert_eq!(
            unloaded.candidates("test"),
            vec![InferenceTarget::Remote(member_peer)]
        );
    }
    #[tokio::test]
    async fn discovered_membership_peer_is_used_without_an_elected_target() {
        let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
            .await
            .unwrap();
        let peer = super::super::tests::test_remote_peer(34, "cold");
        let peer_id = peer.id;
        node.insert_test_peer(peer).await;
        let targets = election::ModelTargets::default();
        let affinity = affinity::AffinityRouter::new();
        let ctx = IngressRouteContext {
            node: &node,
            targets: &targets,
            affinity: &affinity,
            plugin_manager: None,
            requested_by_node_id: None,
            exchange_channel: None,
        };
        let ready = ready_targets(&ctx, "cold", "/v1/chat/completions", false, &[])
            .await
            .unwrap();
        assert_eq!(
            ready.candidates("cold"),
            vec![election::InferenceTarget::Remote(peer_id)]
        );
        assert!(
            ready_targets(&ctx, "cold", "/v1/chat/completions", true, &[])
                .await
                .is_none()
        );
        assert!(
            ready_targets(&ctx, "cold", "/v1/chat/completions", false, &[peer_id])
                .await
                .is_none()
        );
        assert!(
            ready_targets(&ctx, "cold", "/v1/embeddings", false, &[])
                .await
                .is_none()
        );
        node.endpoint.close().await;
    }

    #[tokio::test]
    async fn cold_named_embeddings_load_before_admission_then_recheck_verified_workload() {
        for workload in [
            mesh::ModelWorkloadClass::Embedding,
            mesh::ModelWorkloadClass::CausalGeneration,
        ] {
            cold_embedding_roundtrip(workload).await;
        }
    }

    async fn cold_embedding_roundtrip(workload: mesh::ModelWorkloadClass) {
        use crate::runtime::native_model_lifecycle::NativeModelLifecycle;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::{TcpListener, TcpStream};
        let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
            .await
            .unwrap();
        let config: crate::plugin::MeshConfig = toml::from_str(
            r#"
[[models]]
model = "cold"
[models.lifecycle]
load = "on_demand"
[models.advanced.server]
alias = "public-alias"
"#,
        )
        .unwrap();
        let (service, mut intents, targets) = NativeModelLifecycle::test_service(&config);
        assert!(node.native_model_lifecycle.set(service.clone()).is_ok());
        let backend = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = backend.local_addr().unwrap().port();
        let backend_task = tokio::spawn(async move {
            let (mut stream, _) = backend.accept().await.unwrap();
            let request = proxy::read_http_request(&mut stream).await.unwrap();
            assert_eq!(request.model_name.as_deref(), Some("public-alias"));
            stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 11\r\nConnection: close\r\n\r\n{\"ok\":true}").await.unwrap();
        });
        let control = {
            let node = node.clone();
            tokio::spawn(async move {
                let crate::runtime::ModelIntent::Load {
                    completion: Some(completion),
                    config_model_id,
                    ..
                } = intents.recv().await.unwrap()
                else {
                    panic!("expected cold load")
                };
                assert_eq!(config_model_id.as_deref(), Some("cold"));
                node.set_hosted_models(vec!["public-alias".into()]).await;
                node.set_served_model_descriptors(vec![mesh::ServedModelDescriptor {
                    identity: mesh::ServedModelIdentity {
                        model_name: "public-alias".into(),
                        ..Default::default()
                    },
                    capabilities_known: true,
                    metadata: Some(mesh::ServedModelMetadata {
                        workload_class: Some(workload),
                        ..Default::default()
                    }),
                    ..Default::default()
                }])
                .await;
                targets.send_modify(|targets| {
                    targets.targets.insert(
                        "public-alias".into(),
                        vec![election::InferenceTarget::Local(port)],
                    );
                });
                service.record_loaded("cold", "test-instance", "public-alias", port);
                completion
                    .send(Ok(crate::api::RuntimeLoadResponse {
                        model_ref: "cold".into(),
                        model: "public-alias".into(),
                        instance_id: "test-instance".into(),
                        profile: String::new(),
                        backend: None,
                        context_length: None,
                    }))
                    .unwrap();
            })
        };
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let ingress = {
            let node = node.clone();
            tokio::spawn(async move {
                let (stream, _) = listener.accept().await.unwrap();
                handle_api_proxy_connection(
                    node,
                    stream.into(),
                    election::ModelTargets::default(),
                    affinity::AffinityRouter::new(),
                    crate::runtime::IngressType::LocalOpenAi,
                    None,
                )
                .await;
            })
        };
        let mut client = TcpStream::connect(address).await.unwrap();
        let body = r#"{"model":"public-alias","input":"hello"}"#;
        client.write_all(format!("POST /v1/embeddings HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes()).await.unwrap();
        let mut response = String::new();
        tokio::time::timeout(
            std::time::Duration::from_secs(5),
            client.read_to_string(&mut response),
        )
        .await
        .unwrap()
        .unwrap();
        let expected = if workload == mesh::ModelWorkloadClass::Embedding {
            "HTTP/1.1 200"
        } else {
            "HTTP/1.1 422"
        };
        assert!(response.starts_with(expected), "{response}");
        control.await.unwrap();
        ingress.await.unwrap();
        if workload == mesh::ModelWorkloadClass::Embedding {
            backend_task.await.unwrap();
        } else {
            backend_task.abort();
        }
        node.endpoint.close().await;
    }
}
