use super::*;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

const MODEL: &str = "audio-workload";

fn multipart(model: &str) -> Vec<u8> {
    [b"--test\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\nRIFF\0{\xff}\x80\r\n--test\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n".as_slice(), model.as_bytes(), b"\r\n--test--\r\n"].concat()
}

async fn audio_node() -> mesh::Node {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .unwrap();
    node.set_hosted_models(vec![MODEL.into()]).await;
    node.set_served_model_descriptors(vec![mesh::ServedModelDescriptor {
        identity: mesh::ServedModelIdentity {
            model_name: MODEL.into(),
            ..Default::default()
        },
        capabilities_known: true,
        capabilities: crate::models::ModelCapabilities {
            audio: crate::models::CapabilityLevel::Supported,
            multimodal: true,
            ..Default::default()
        },
        metadata: Some(mesh::ServedModelMetadata {
            workload_class: Some(mesh::ModelWorkloadClass::CausalGeneration),
            ..Default::default()
        }),
        ..Default::default()
    }])
    .await;
    node
}

async fn route_audio(path: &str, chunked: bool) {
    let backend = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let backend_port = backend.local_addr().unwrap().port();
    let backend_task = tokio::spawn(async move {
        let (mut stream, _) = backend.accept().await.unwrap();
        let request = proxy::read_http_request(&mut stream).await.unwrap();
        stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 13\r\nConnection: close\r\n\r\n{\"text\":\"ok\"}").await.unwrap();
        request
    });
    let node = audio_node().await;
    let mut targets = election::ModelTargets::default();
    targets.targets.insert(
        MODEL.into(),
        vec![election::InferenceTarget::Local(backend_port)],
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let ingress = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        handle_api_proxy_connection(
            node,
            stream.into(),
            targets,
            affinity::AffinityRouter::new(),
            crate::runtime::IngressType::LocalOpenAi,
        )
        .await;
    });
    let body = multipart("auto");
    let framing = if chunked {
        "Transfer-Encoding: chunked".into()
    } else {
        format!("Content-Length: {}", body.len())
    };
    let mut raw = format!("POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: multipart/form-data; boundary=test\r\n{framing}\r\n\r\n").into_bytes();
    if chunked {
        for chunk in body.chunks(11) {
            raw.extend_from_slice(format!("{:x}\r\n", chunk.len()).as_bytes());
            raw.extend_from_slice(chunk);
            raw.extend_from_slice(b"\r\n");
        }
        raw.extend_from_slice(b"0\r\n\r\n");
    } else {
        raw.extend_from_slice(&body);
    }
    let mut client = TcpStream::connect(address).await.unwrap();
    client.write_all(&raw).await.unwrap();
    let mut response = String::new();
    client.read_to_string(&mut response).await.unwrap();
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    let forwarded = backend_task.await.unwrap();
    ingress.await.unwrap();
    assert_eq!(forwarded.model_name.as_deref(), Some(MODEL));
    assert_eq!(
        forwarded.body_bytes.as_deref(),
        Some(multipart(MODEL).as_slice())
    );
    let header_end = forwarded
        .raw
        .windows(4)
        .position(|bytes| bytes == b"\r\n\r\n")
        .unwrap();
    let headers = String::from_utf8_lossy(&forwarded.raw[..header_end]).to_lowercase();
    assert!(!headers.contains("transfer-encoding:"));
    assert!(headers.contains(&format!("content-length: {}", multipart(MODEL).len())));
}

#[tokio::test]
async fn automatic_audio_reaches_http_backend_with_intact_binary_and_decoded_framing() {
    for path in ["/v1/audio/transcriptions", "/v1/audio/translations?trace=1"] {
        for chunked in [false, true] {
            tokio::time::timeout(
                std::time::Duration::from_secs(10),
                route_audio(path, chunked),
            )
            .await
            .expect("audio route must finish");
        }
    }
}
