//! Actual installed policy admission over planner and injected strong requests.
use super::lifecycle_live_tests::LiveHost;
use crate::network::openai::{client_stream::ClientStream, response};
use crate::plugin::{ExternalPluginSpec, PluginHostMode, PluginManager, ResolvedPlugins};
use mesh_llm_config::{OpenAiExchangeFailurePolicy, OpenAiExchangeGrant};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use tokio::{io::AsyncReadExt, net::TcpListener};

async fn select_policy(host: &mut LiveHost, deny_model: &str, virtual_echo: bool) {
    host.manager.shutdown().await;
    let mut args = vec!["--body".into(), "--admission".into()];
    if virtual_echo {
        args.push("--virtual-echo".into());
    }
    let spec = ExternalPluginSpec {
        name: "openai-exchange-observer".into(),
        command: host
            .installed_metadata
            .executable_path()
            .display()
            .to_string(),
        args,
        url: None,
        startup: Default::default(),
        web_ui_enabled: None,
        web_ui_primary_tab: None,
        env: BTreeMap::from([
            (
                "MESH_LLM_EXEMPLAR_DENY_SELECTED_MODEL".into(),
                deny_model.into(),
            ),
            (
                "MESH_LLM_EXEMPLAR_EVENT_LOG".into(),
                host.root.path().join("events.jsonl").display().to_string(),
            ),
            (
                "MESH_LLM_EXEMPLAR_VIRTUAL_LOG".into(),
                host.root.path().join("virtual.json").display().to_string(),
            ),
        ]),
        openai_exchange_grant: Some(Box::new(OpenAiExchangeGrant {
            endpoints: vec![
                "chat_completions".into(),
                "completions".into(),
                "responses".into(),
            ],
            phases: vec![
                "request_received".into(),
                "backend_selected".into(),
                "exchange_finished".into(),
            ],
            request_body: true,
            effective_request_body: true,
            response_body: true,
            admission: true,
            metadata: true,
            deadline_ms: 2000,
            max_body_bytes: 1048576,
            max_queue_bytes: 4194304,
            max_in_flight: 8,
            failure_policy: OpenAiExchangeFailurePolicy::Required,
            ..Default::default()
        })),
        installed_metadata: Some(host.installed_metadata.clone()),
    };
    let (tx, rx) = tokio::sync::mpsc::channel(8);
    host.manager = PluginManager::start(
        &ResolvedPlugins {
            externals: vec![spec],
            inactive: Vec::new(),
        },
        PluginHostMode {
            mesh_visibility: mesh_llm_plugin::MeshVisibility::Private,
        },
        tx,
    )
    .await
    .unwrap();
    assert_eq!(host.manager.list().await[0].status, "running");
    host.node.set_plugin_manager(host.manager.clone()).await;
    host.node.start_plugin_channel_forwarder(rx);
}

#[tokio::test]
#[ignore = "requires just package-openai-exchange-exemplar"]
async fn installed_lifecycle_pipeline_admits_each_prepared_dispatch_and_never_falls_back_on_deny() {
    for (denied_model, expected_requests) in [("planner-model", 0), ("allowed-model", 1), ("", 2)] {
        let mut host = LiveHost::start(true, true).await;
        select_policy(&mut host, denied_model, false).await;
        let port = match host.targets.targets["allowed-model"][0] {
            crate::inference::election::InferenceTarget::Local(port) => port,
            _ => panic!("fixture requires local target"),
        };
        let request_id = skippy_inference_api::generate_request_id();
        let exchange_id = request_id.as_uuid().to_string();
        let body = json!({"model":"allowed-model","messages":[{"role":"user","content":"prepare this task"}],"stream":false});
        let original = serde_json::to_vec(&body).unwrap();
        let event = crate::plugin::request_event(
            exchange_id.clone(),
            "chat_completions",
            "POST",
            "/v1/chat/completions",
            &original,
            Default::default(),
            false,
        );
        let (mut session, received) =
            crate::plugin::ExchangeSession::begin(&host.manager, event).await;
        assert!(received.error_status().is_none());
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let mut client = tokio::net::TcpStream::connect(listener.local_addr().unwrap())
            .await
            .unwrap();
        let (server, _) = listener.accept().await.unwrap();
        let mut stream = ClientStream::from(server).with_wire_bytes_observer(session.observer());
        let nonce = response::PipelineCapsuleNonce {
            observation_id: Some(session.observation_id().into()),
            exchange_id: Some(exchange_id.clone()),
            ..Default::default()
        };
        let result = response::pipeline_proxy_local(
            &mut stream,
            "/v1/chat/completions",
            body,
            port,
            "planner-model",
            port,
            &host.node,
            &nonce,
            response::RouteAttemptLoggingContext {
                exchange_id: None,
                request_id,
                response_adapter: crate::network::openai::request_normalize::ResponseAdapter::None,
                retry_policy: response::ResponseRetryPolicy::next_target_available(false),
                route_observer: crate::logging::OpenAiLifecycleAttachment::unowned()
                    .route_observer(),
                served_by: None,
                peer_capsule_id: None,
            },
        )
        .await;
        drop(stream);
        let mut wire = Vec::new();
        tokio::time::timeout(
            std::time::Duration::from_secs(5),
            client.read_to_end(&mut wire),
        )
        .await
        .unwrap()
        .unwrap();
        if denied_model.is_empty() {
            assert!(matches!(
                result,
                response::PipelineProxyResult::Responded(200)
                    | response::PipelineProxyResult::RespondedWithUsage {
                        status_code: 200,
                        ..
                    }
            ));
            session.finish("completed").await;
        } else {
            assert_eq!(result, response::PipelineProxyResult::PolicyDenied);
            assert!(wire.starts_with(b"HTTP/1.1 403 "));
            session.finish("policy_denied").await;
        }
        let dispatched = host.requests.lock().await.clone();
        assert_eq!(
            dispatched.len(),
            expected_requests,
            "denial must not fall back or invoke the rejected model"
        );
        let events = host.events();
        let selected: Vec<&Value> = events
            .iter()
            .filter(|event| {
                event["exchange_id"] == exchange_id && event["phase"] == "backend_selected"
            })
            .collect();
        assert_eq!(selected.len(), if expected_requests == 0 { 1 } else { 2 });
        assert_eq!(selected[0]["body"]["model"], "planner-model");
        assert_eq!(selected[0]["body"]["max_tokens"], 256);
        if expected_requests > 0 {
            assert!(
                selected[1]["body"]["messages"][0]["content"]
                    .as_str()
                    .unwrap()
                    .contains("hi")
            );
        }
        for (index, exact) in dispatched.iter().enumerate() {
            assert_eq!(selected[index]["effective_request_encoding"], "http_entity");
            assert_eq!(
                selected[index]["effective_request_wire_digest"]["sha256"],
                hex::encode(Sha256::digest(exact))
            );
            assert_eq!(
                selected[index]["effective_request_wire_digest"]["byte_count"],
                exact.len()
            );
        }
        host.stop().await;
    }
}

#[tokio::test]
#[ignore = "requires just package-openai-exchange-exemplar"]
async fn installed_lifecycle_virtual_admission_observes_final_invocation_before_backend_runs() {
    for (deny, path, endpoint) in [
        (true, "/v1/chat/completions", "chat_completions"),
        (false, "/v1/responses", "responses"),
    ] {
        let mut host = LiveHost::start(true, true).await;
        select_policy(&mut host, if deny { "exchange-echo" } else { "" }, true).await;
        let body: &[u8] = if path == "/v1/responses" {
            br#"{"model":"exchange-echo","input":"echo this","stream":false}"#
        } else {
            br#"{"model":"exchange-echo","messages":[{"role":"user","content":"echo this"}],"stream":false}"#
        };
        let response = host.request(path, body).await;
        let invocation_file = host.root.path().join("virtual.json");
        let events = host.events();
        let selected = events
            .iter()
            .find(|event| event["phase"] == "backend_selected")
            .expect("virtual dispatch must be observed");
        assert_eq!(selected["path"], path);
        assert_eq!(selected["endpoint"], endpoint);
        assert_eq!(
            selected["effective_request_encoding"],
            "plugin_invocation_json"
        );
        assert_eq!(selected["body"]["request"]["model"], "exchange-echo");
        assert!(selected["body"]["candidates"].is_array());
        if deny {
            assert!(response.starts_with(b"HTTP/1.1 403 "));
            assert!(
                !invocation_file.exists(),
                "denied virtual backend must not run"
            );
        } else {
            assert!(response.starts_with(b"HTTP/1.1 200 "));
            let independently_received = std::fs::read(&invocation_file).unwrap();
            assert_eq!(
                selected["effective_request_wire_digest"]["sha256"],
                hex::encode(Sha256::digest(&independently_received))
            );
            assert_eq!(
                selected["effective_request_wire_digest"]["byte_count"],
                independently_received.len()
            );
        }
        assert!(host.requests.lock().await.is_empty());
        host.stop().await;
    }
}
