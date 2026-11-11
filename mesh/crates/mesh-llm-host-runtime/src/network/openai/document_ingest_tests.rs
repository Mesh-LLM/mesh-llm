use super::*;
use crate::plugin;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

pub(crate) const FAKE_PLUGIN: &str = "fake-documents";
const PDF: &[u8] = b"%PDF-1.7\nfake body\n%%EOF\n";

pub(crate) fn pdf_data_url() -> String {
    format!(
        "data:application/pdf;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(PDF)
    )
}

/// What a fake `document-ingest.v1` provider answers `prepare_documents` with.
#[derive(Clone)]
pub(crate) enum FakeAnswer {
    /// One text part per document, `route_model` optional.
    Text {
        text: String,
        route_model: Option<String>,
    },
    /// The plugin's own `{error}` answer.
    Error { status: u16, code: String },
}

/// An in-memory provider: hands out a private staging directory, checks that
/// every staged file is a real PDF there, records each request and answers
/// as configured.
#[derive(Clone)]
pub(crate) struct FakeDocumentPlugin {
    pub(crate) staging: PathBuf,
    pub(crate) answer: FakeAnswer,
    pub(crate) requests: Arc<Mutex<Vec<Value>>>,
}

impl FakeDocumentPlugin {
    pub(crate) fn new(answer: FakeAnswer) -> Self {
        let staging =
            std::env::temp_dir().join(format!("mesh-doc-staging-{}", rand::random::<u64>()));
        std::fs::create_dir_all(&staging).unwrap();
        Self {
            staging,
            answer,
            requests: Arc::default(),
        }
    }

    pub(crate) async fn plugin_manager(&self) -> plugin::PluginManager {
        let manager =
            plugin::PluginManager::for_test_bridge(&[FAKE_PLUGIN], Arc::new(self.clone()));
        let mut manifests = BTreeMap::new();
        manifests.insert(
            FAKE_PLUGIN.to_string(),
            mesh_llm_plugin::proto::PluginManifest {
                capabilities: vec![DOCUMENT_INGEST_CAPABILITY.into()],
                ..Default::default()
            },
        );
        manager.set_test_manifests(manifests).await;
        manager
    }

    pub(crate) fn staged_files(&self) -> usize {
        std::fs::read_dir(&self.staging).unwrap().count()
    }

    pub(crate) fn requests(&self) -> Vec<Value> {
        self.requests.lock().unwrap().clone()
    }

    fn prepare(&self, arguments: Value) -> Value {
        let documents = arguments["documents"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        for document in &documents {
            let name = document["staged"].as_str().unwrap();
            assert!(name.starts_with(STAGED_NAME_PREFIX), "{name}");
            let bytes = std::fs::read(self.staging.join(name)).unwrap();
            assert_eq!(bytes, PDF);
        }
        self.requests.lock().unwrap().push(arguments);
        match &self.answer {
            FakeAnswer::Text { text, route_model } => json!({
                "parts": documents.iter().map(|document| json!({
                    "index": document["index"],
                    "mode": if route_model.is_some() { "rlm" } else { "inline" },
                    "text": text,
                })).collect::<Vec<_>>(),
                "route_model": route_model,
            }),
            FakeAnswer::Error { status, code } => json!({
                "error": {"status": status, "code": code, "message": format!("fake {code}")},
            }),
        }
    }
}

impl Drop for FakeDocumentPlugin {
    fn drop(&mut self) {
        if Arc::strong_count(&self.requests) == 1 {
            let _ = std::fs::remove_dir_all(&self.staging);
        }
    }
}

impl plugin::PluginRpcBridge for FakeDocumentPlugin {
    fn handle_request(
        &self,
        _plugin_name: String,
        method: String,
        params_json: String,
    ) -> plugin::BridgeFuture<Result<plugin::RpcResult, plugin::proto::ErrorResponse>> {
        let fake = self.clone();
        Box::pin(async move {
            assert_eq!(method, "tools/call");
            let request: Value = serde_json::from_str(&params_json).unwrap();
            let value = match request["name"].as_str().unwrap() {
                STAGING_DIR_OPERATION => json!({"dir": fake.staging}),
                PREPARE_DOCUMENTS_OPERATION => fake.prepare(request["arguments"].clone()),
                other => panic!("unexpected operation {other}"),
            };
            Ok(plugin::RpcResult {
                result_json: serde_json::to_string(&rmcp::model::CallToolResult::structured(value))
                    .unwrap(),
            })
        })
    }

    fn handle_notification(
        &self,
        _plugin_name: String,
        _method: String,
        _params_json: String,
    ) -> plugin::BridgeFuture<()> {
        Box::pin(async {})
    }
}

fn chat_with_pdf(block: Value) -> Value {
    json!({
        "model": "qwen",
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": [
                {"type": "text", "text": "summarise"},
                block,
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}},
            ]},
        ],
    })
}

fn local_loopback() -> RequestOrigin {
    RequestOrigin {
        local_listener: true,
        loopback: true,
    }
}

#[test]
fn finds_chat_and_responses_pdf_parts_but_not_other_files() {
    let url = pdf_data_url();
    let body = json!({"messages": [{"role": "user", "content": [
        {"type": "file", "file": {"file_data": url, "filename": "a.pdf"}},
        {"type": "input_file", "input_file": {"url": url, "file_name": "b.pdf"}},
        {"type": "file", "file": {"file_data": "data:text/plain;base64,aGk="}},
        {"type": "file", "file": {"file_id": "file-123"}},
        {"type": "file", "file": {"url": "https://example.com/c.pdf"}},
    ]}]});
    let parts = find_pdf_parts(&body).unwrap();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0].filename.as_deref(), Some("a.pdf"));
    assert_eq!((parts[1].message_index, parts[1].block_index), (0, 1));
    assert_eq!(parts[1].bytes, PDF);
}

#[test]
fn a_declared_pdf_that_is_not_one_is_rejected() {
    let body = json!({"messages": [{"role": "user", "content": [
        {"type": "file", "file": {"file_data": "data:application/pdf;base64,aGVsbG8="}},
    ]}]});
    let error = find_pdf_parts(&body).err().unwrap();
    assert_eq!((error.status, error.code.as_str()), (422, "invalid_pdf"));
}

#[test]
fn body_sniff_matches_mime_and_inline_magic() {
    assert!(body_may_contain_pdf(&pdf_data_url()));
    assert!(body_may_contain_pdf(
        r#"{"file_data":"data:;base64,JVBERi0xLjc="}"#
    ));
    assert!(!body_may_contain_pdf(r#"{"messages":[]}"#));
}

#[tokio::test]
async fn replaces_pdf_parts_in_place_and_cleans_staging() {
    let fake = FakeDocumentPlugin::new(FakeAnswer::Text {
        text: "PDF TEXT".into(),
        route_model: None,
    });
    let manager = fake.plugin_manager().await;
    let mut body = chat_with_pdf(
        json!({"type": "file", "file": {"file_data": pdf_data_url(), "filename": "a.pdf"}}),
    );
    body["tools"] = json!([{"type": "function", "function": {"name": "read_file"}}]);

    let rewrite = prepare_request_documents(&mut body, &manager, local_loopback(), Some("pointer"))
        .await
        .unwrap()
        .unwrap();

    assert_eq!(rewrite.documents, 1);
    assert_eq!(rewrite.route_model, None);
    let content = &body["messages"][1]["content"];
    assert_eq!(content[0], json!({"type": "text", "text": "summarise"}));
    assert_eq!(content[1], json!({"type": "text", "text": "PDF TEXT"}));
    assert_eq!(content[2]["type"], "image_url");
    assert_eq!(body["model"], "qwen");
    let request = &fake.requests()[0];
    assert_eq!(request["caller"]["loopback"], true);
    assert_eq!(request["caller"]["has_tools"], true);
    assert_eq!(request["caller"]["mode_header"], "pointer");
    assert_eq!(request["model"], "qwen");
    assert_eq!(request["documents"][0]["filename"], "a.pdf");
    assert_eq!(fake.staged_files(), 0, "staged PDFs must be removed");
}

#[tokio::test]
async fn route_model_from_the_plugin_replaces_the_requested_model() {
    let fake = FakeDocumentPlugin::new(FakeAnswer::Text {
        text: "<mesh-document token=\"t\">".into(),
        route_model: Some("mesh-docs".into()),
    });
    let manager = fake.plugin_manager().await;
    let mut body = chat_with_pdf(json!({"type": "file", "file": {"file_data": pdf_data_url()}}));

    let rewrite = prepare_request_documents(&mut body, &manager, local_loopback(), None)
        .await
        .unwrap()
        .unwrap();

    assert_eq!(rewrite.route_model.as_deref(), Some("mesh-docs"));
    assert_eq!(body["model"], "mesh-docs");
    assert_eq!(fake.requests()[0]["caller"]["has_tools"], false);
}

#[tokio::test]
async fn plugin_errors_end_the_request_with_their_status_and_clean_staging() {
    let fake = FakeDocumentPlugin::new(FakeAnswer::Error {
        status: 422,
        code: "pdf_no_text_layer".into(),
    });
    let manager = fake.plugin_manager().await;
    let mut body = chat_with_pdf(json!({"type": "file", "file": {"file_data": pdf_data_url()}}));

    let error = prepare_request_documents(&mut body, &manager, local_loopback(), None)
        .await
        .unwrap_err();

    assert_eq!(
        (error.status, error.code.as_str()),
        (422, "pdf_no_text_layer")
    );
    assert_eq!(fake.staged_files(), 0);
}

#[tokio::test]
async fn peer_requests_and_requests_without_a_provider_are_untouched() {
    let fake = FakeDocumentPlugin::new(FakeAnswer::Text {
        text: "unused".into(),
        route_model: None,
    });
    let manager = fake.plugin_manager().await;
    let original = chat_with_pdf(json!({"type": "file", "file": {"file_data": pdf_data_url()}}));

    let mut body = original.clone();
    let peer = prepare_request_documents(&mut body, &manager, RequestOrigin::default(), None)
        .await
        .unwrap();
    assert_eq!(peer, None);
    assert_eq!(body, original);
    assert!(fake.requests().is_empty());

    let no_provider = plugin::PluginManager::for_test_bridge(&[], Arc::new(fake.clone()));
    let mut body = original.clone();
    let none = prepare_request_documents(&mut body, &no_provider, local_loopback(), None)
        .await
        .unwrap();
    assert_eq!(none, None);
    assert_eq!(body, original);
}

#[tokio::test]
async fn more_than_the_pdf_limit_is_rejected_before_staging() {
    let fake = FakeDocumentPlugin::new(FakeAnswer::Text {
        text: "unused".into(),
        route_model: None,
    });
    let manager = fake.plugin_manager().await;
    let blocks: Vec<Value> = (0..=MAX_REQUEST_PDFS)
        .map(|_| json!({"type": "file", "file": {"file_data": pdf_data_url()}}))
        .collect();
    let mut body = json!({"model": "qwen", "messages": [{"role": "user", "content": blocks}]});

    let error = prepare_request_documents(&mut body, &manager, local_loopback(), None)
        .await
        .unwrap_err();

    assert_eq!(
        (error.status, error.code.as_str()),
        (413, "document_too_large")
    );
    assert!(fake.requests().is_empty());
    assert_eq!(fake.staged_files(), 0);
}

#[test]
fn origin_is_loopback_only_for_loopback_peers() {
    let loopback = RequestOrigin::local(Some("127.0.0.1:9".parse().unwrap()));
    assert!(loopback.local_listener && loopback.loopback);
    let lan = RequestOrigin::local(Some("192.168.1.5:9".parse().unwrap()));
    assert!(lan.local_listener && !lan.loopback);
    assert!(!RequestOrigin::local(None).loopback);
}
