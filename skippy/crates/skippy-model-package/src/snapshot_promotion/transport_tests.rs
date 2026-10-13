use super::super::hub_fixture::Server;
use super::*;

fn transport(endpoint: &str) -> HubTransport {
    let _ = skippy_model_hf::configure_hf_tls_provider();
    HubTransport {
        runtime: tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap(),
        client: hf_hub::HFClientBuilder::new()
            .endpoint(endpoint)
            .token("fixture-token")
            .build()
            .unwrap(),
        http: reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap(),
        repo: "fixture/model".into(),
        token: Some("fixture-token".into()),
    }
}

fn publication_fixture(receipt: Vec<u8>, lfs_hash: &str) -> (Server, PromotionPlan) {
    let manifest = super::super::fixtures::manifest("shared/weights.gguf", 999, "b".repeat(64));
    let plan =
        super::super::policy::promote(&manifest, "automation/republish-fixture", &"a".repeat(40))
            .unwrap();
    let info = serde_json::json!({"id":"fixture/model","sha":"d".repeat(40)});
    let paths = serde_json::json!([
        {"type":"file","oid":"e".repeat(40),"path":"shared/weights.gguf","size":999,
            "lfs":{"size":999,"sha256":lfs_hash,"pointerSize":120}},
        {"type":"file","oid":"f".repeat(40),"path":"model-package.json","size":manifest.len()}
    ]);
    let mut responses = vec![
        (200, serde_json::to_vec(&info).unwrap()),
        (200, serde_json::to_vec(&paths).unwrap()),
        (200, manifest),
    ];
    if lfs_hash == "b".repeat(64) {
        responses.push((200, receipt));
    }
    (Server::start(responses), plan)
}

#[test]
fn real_transport_pins_reads_and_sends_one_parent_bound_commit() {
    let (server, plan) = publication_fixture(
        serde_json::to_vec(&serde_json::json!({"commitOid":"c".repeat(40)})).unwrap(),
        &"b".repeat(64),
    );
    transport(&server.endpoint).publish(&plan).unwrap();
    let requests = server.finish();
    assert_eq!(requests.len(), 4);
    let request_text = requests
        .iter()
        .map(|bytes| String::from_utf8_lossy(bytes).into_owned())
        .collect::<Vec<_>>();
    assert!(request_text[1].contains(&format!("/paths-info/{}", "d".repeat(40))));
    assert!(request_text[2].contains(&format!("/resolve/{}/model-package.json", "d".repeat(40))));
    assert!(request_text[3].starts_with("POST /api/models/fixture/model/commit/main "));
    assert!(request_text[3].contains(&format!("\"parentCommit\":\"{}\"", "a".repeat(40))));
    assert!(request_text[3].contains("\"key\":\"lfsFile\""));
    assert!(request_text[3].contains("\"size\":999"));
    assert!(request_text[3].contains("\"key\":\"file\""));
    assert!(request_text.iter().all(|request| {
        request
            .to_ascii_lowercase()
            .contains("authorization: bearer fixture-token")
    }));
}

#[test]
fn real_transport_refuses_catalog_identity_mismatch_before_commit() {
    let (server, plan) = publication_fixture(Vec::new(), &"e".repeat(64));
    let error = transport(&server.endpoint).publish(&plan).unwrap_err();
    assert!(error.to_string().contains("identity differs"));
    assert_eq!(server.finish().len(), 3);
}

#[test]
fn real_transport_retains_staging_on_invalid_or_oversized_receipt_without_retry() {
    for receipt in [b"invalid json".to_vec(), vec![b' '; 65537], b"{}".to_vec()] {
        let (server, plan) = publication_fixture(receipt, &"b".repeat(64));
        let error = transport(&server.endpoint).publish(&plan).unwrap_err();
        assert!(error.to_string().contains("staging retained"));
        assert_eq!(server.finish().len(), 4);
    }
}
