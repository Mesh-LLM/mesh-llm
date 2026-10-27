use super::*;
use mesh_llm_events::logging::identifiers::RequestId;

#[derive(Default)]
struct PreRoutingRejectionSink {
    requests: std::sync::Mutex<Vec<(Option<String>, usize, crate::network::metrics::RequestOutcome)>>,
}

impl crate::network::metrics::RoutingTelemetrySink for PreRoutingRejectionSink {
    fn observe_inflight_requests(&self, _current: u64) {}

    fn record_model_request(
        &self,
        model: Option<&str>,
        attempts: usize,
        outcome: crate::network::metrics::RequestOutcome,
    ) {
        self.requests
            .lock()
            .unwrap()
            .push((model.map(str::to_string), attempts, outcome));
    }

    fn record_route_attempt(
        &self,
        _model: Option<&str>,
        _target: &crate::network::metrics::AttemptTarget,
        _outcome: crate::network::metrics::AttemptOutcome,
    ) {
    }
}

#[tokio::test]
async fn a_model_rejected_before_routing_is_counted_in_request_telemetry() {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .expect("test node should start");
    let sink = std::sync::Arc::new(PreRoutingRejectionSink::default());
    node.set_routing_telemetry_sink(Some(sink.clone()));

    let model = "definitely-not-a-served-model";
    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "hello"}]
    })
    .to_string();
    let mut request = proxy::BufferedHttpRequest {
        raw: body.as_bytes().to_vec(),
        method: "POST".to_owned(),
        path: "/v1/chat/completions".to_owned(),
        client_path: "/v1/chat/completions".to_owned(),
        request_id: RequestId::default(),
        body_json: None,
        body_json_attempted: false,
        body_bytes: None,
        body_len_bytes: body.len(),
        completion_tokens: None,
        stream: None,
        model_name: Some(model.to_owned()),
        request_object_request_ids: Vec::new(),
        response_adapter: proxy::ResponseAdapter::OpenAiChatCompletionsJson,
        correlation_id: None,
        exchange_observation_id: None,
    };

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback listener");
    let addr = listener.local_addr().expect("local addr");
    let client_connect = tokio::net::TcpStream::connect(addr);
    let server_accept = async { listener.accept().await.map(|(s, _)| s) };
    let (_client_side, server_side) = tokio::join!(client_connect, server_accept);
    let tcp_stream = server_side.expect("accept server side");

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
    let lifecycle = OpenAiLifecycleAttachment::unowned();

    let outcome = route_missing_local_model(
        tcp_stream.into(),
        &mut request,
        &ctx,
        model,
        None,
        &[],
        None,
        lifecycle.route_observer(),
    )
    .await;

    assert!(
        matches!(outcome, proxy::RouteDispatchOutcome::Responded(404)),
        "an unknown model must be rejected with 404 before routing, got {outcome:?}"
    );

    let requests = sink.requests.lock().unwrap();
    assert_eq!(
        requests.len(),
        1,
        "a request rejected before routing must be counted in \
         mesh_llm_model_request_total"
    );
    let (recorded_model, _attempts, recorded_outcome) = &requests[0];
    assert!(
        recorded_model.is_some(),
        "the rejection must carry the bounded model ref"
    );
    assert!(
        matches!(
            recorded_outcome,
            crate::network::metrics::RequestOutcome::Rejected(_)
                | crate::network::metrics::RequestOutcome::Unavailable
        ),
        "a pre-routing rejection must carry a bounded outcome, got {recorded_outcome:?}"
    );
}
