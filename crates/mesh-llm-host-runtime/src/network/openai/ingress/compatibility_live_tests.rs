//! Real installed child compatibility against an old generation-3 host handshake.
#![cfg(unix)]
use mesh_llm_plugin::{LocalStream, PROTOCOL_VERSION, proto, read_envelope, write_envelope};
use mesh_llm_plugin_manager::{PluginInstallOptions, PluginTarget};
use std::time::Duration;

struct OldHost {
    _root: tempfile::TempDir,
    child: tokio::process::Child,
    stream: LocalStream,
}
impl OldHost {
    async fn start(legacy: bool) -> Self {
        let root = tempfile::tempdir_in("/tmp").unwrap();
        let archive = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../dist/openai-exchange-observer.tar.gz");
        assert!(
            archive.exists(),
            "run just package-openai-exchange-exemplar"
        );
        let options = PluginInstallOptions {
            store_root: root.path().join("store"),
            install_root: root.path().join("installed"),
            catalog_url: "unused".into(),
            target: PluginTarget::current().unwrap(),
        };
        let installed = mesh_llm_plugin_manager::install::install_plugin_archive(
            "openai-exchange-observer",
            "1.0.0",
            &archive,
            &options,
            &mut |_| {},
        )
        .unwrap()
        .metadata;
        let path = root.path().join("control.sock");
        let listener = tokio::net::UnixListener::bind(&path).unwrap();
        let mut command = tokio::process::Command::new(installed.executable_path());
        command
            .env("MESH_LLM_PLUGIN_ENDPOINT", &path)
            .env("MESH_LLM_PLUGIN_TRANSPORT", "unix")
            .kill_on_drop(true)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        if legacy {
            command.arg("--legacy-conformance");
        }
        let child = command.spawn().unwrap();
        let (stream, _) = tokio::time::timeout(Duration::from_secs(5), listener.accept())
            .await
            .unwrap()
            .unwrap();
        Self {
            _root: root,
            child,
            stream: LocalStream::Unix(stream),
        }
    }
    async fn exchange(&mut self, id: u64, payload: proto::envelope::Payload) -> proto::Envelope {
        write_envelope(
            &mut self.stream,
            &proto::Envelope {
                protocol_version: PROTOCOL_VERSION,
                plugin_id: "openai-exchange-observer".into(),
                request_id: id,
                payload: Some(payload),
            },
        )
        .await
        .unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let response = read_envelope(&mut self.stream).await.unwrap();
                if response.request_id == id {
                    return response;
                }
            }
        })
        .await
        .unwrap()
    }
    async fn initialize(&mut self) -> proto::Envelope {
        self.exchange(
            1,
            proto::envelope::Payload::InitializeRequest(proto::InitializeRequest {
                host_protocol_version: PROTOCOL_VERSION,
                host_version: "generation-3-without-lifecycle".into(),
                host_info_json: "{}".into(),
                mesh_visibility: proto::MeshVisibility::Private as i32,
                host_capabilities: Vec::new(),
            }),
        )
        .await
    }
    async fn stop(mut self) {
        let _ = self.child.kill().await;
        self.child.wait().await.unwrap();
    }
}

#[tokio::test]
#[ignore = "requires just package-openai-exchange-exemplar"]
async fn installed_generation3_child_without_lifecycle_manifest_initializes_and_operates() {
    assert_eq!(PROTOCOL_VERSION, 3);
    let mut host = OldHost::start(true).await;
    let initialized = host.initialize().await;
    let Some(proto::envelope::Payload::InitializeResponse(response)) = initialized.payload else {
        panic!("old declaration must initialize: {initialized:?}");
    };
    assert_eq!(response.plugin_protocol_version, 3);
    assert!(response.manifest.unwrap().openai_exchange_hook.is_none());
    let echoed = host
        .exchange(
            2,
            proto::envelope::Payload::InvokeServiceRequest(proto::InvokeServiceRequest {
                kind: proto::ServiceKind::Operation as i32,
                service_name: "legacy_echo".into(),
                input_json: "{\"sentinel\":\"legacy-operated\"}".into(),
            }),
        )
        .await;
    let Some(proto::envelope::Payload::InvokeServiceResponse(result)) = echoed.payload else {
        panic!("old operation must run: {echoed:?}");
    };
    assert!(!result.is_error);
    assert!(result.output_json.contains("legacy-operated"));
    host.stop().await;
}

#[tokio::test]
#[ignore = "requires just package-openai-exchange-exemplar"]
async fn installed_required_lifecycle_child_rejects_generation3_host_missing_capability() {
    let mut host = OldHost::start(false).await;
    let initialized = host.initialize().await;
    let Some(proto::envelope::Payload::ErrorResponse(error)) = initialized.payload else {
        panic!("required declaration must fail initialization: {initialized:?}");
    };
    assert!(
        error
            .message
            .contains("host does not support required openai_exchange.v1 contract")
    );
    host.stop().await;
}
