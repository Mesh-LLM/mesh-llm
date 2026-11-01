//! Real SDK lifecycle regressions. Each scenario runs in a disposable process:
//! native state and environment are process-wide, and a stuck worker must not
//! hang the test runner or escape a successful stop assertion.
use super::*;
use std::net::{Ipv4Addr, TcpListener};
use std::process::{Command, Stdio};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::Notify;

const CHILD: &str = "MESH_EMBEDDED_LIFECYCLE_CHILD";
// Child processes inherit the parent environment at spawn; serialize with
// other environment fixtures so a concurrent test cannot supply their config.
#[test]
#[serial_test::serial]
#[ignore = "real embedded serving; requires an installed compatible native runtime"]
fn stop_during_download_releases_runtime_and_restarts() {
    isolated(
        "stop_during_download_releases_runtime_and_restarts",
        download_scenario,
    );
}

#[test]
#[serial_test::serial]
fn stop_during_join_releases_runtime_and_restarts() {
    isolated(
        "stop_during_join_releases_runtime_and_restarts",
        join_scenario,
    );
}

#[test]
#[serial_test::serial]
fn cancelling_startup_releases_runtime_and_restarts() {
    isolated(
        "cancelling_startup_releases_runtime_and_restarts",
        cancelled_startup_scenario,
    );
}

#[test]
#[serial_test::serial]
#[ignore = "native inference; requires MESH_EMBEDDED_TEST_GGUF and an installed native runtime"]
fn cancelled_download_can_restart_and_generate() {
    isolated(
        "cancelled_download_can_restart_and_generate",
        inference_scenario,
    );
}

async fn inference_scenario() {
    use sha2::Digest;
    let model =
        std::env::var("MESH_EMBEDDED_TEST_GGUF").expect("path to pinned stories15M-q4_0.gguf");
    let bytes = std::fs::read(&model).unwrap();
    assert_eq!(
        hex::encode(sha2::Sha256::digest(&bytes)),
        "66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739",
        "native acceptance uses the documented pinned model"
    );
    let mut config = cancel_download().await;
    config.serving.models = vec![model];
    config.network.enumerate_host = true;
    let handle = start_embedded_node(config)
        .await
        .expect("restart serving after cancellation");
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .unwrap();
    let model = tokio::time::timeout(Duration::from_secs(45), async {
        loop {
            let response: serde_json::Value = client
                .get(format!("{}/models", handle.api_base_url()))
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            if let Some(model) = response["data"].as_array().and_then(|models| {
                models
                    .iter()
                    .filter_map(|model| model["id"].as_str())
                    .find(|model| !matches!(*model, "auto" | "mesh"))
            }) {
                break model.to_string();
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .expect("restarted model becomes ready");
    let response = client.post(format!("{}/chat/completions", handle.api_base_url()))
        .json(&serde_json::json!({"model": model, "messages": [{"role": "user", "content": "Once upon a time"}], "max_tokens": 8, "temperature": 0}))
        .send().await.unwrap();
    let status = response.status();
    let body: serde_json::Value = response.json().await.unwrap();
    assert!(status.is_success(), "inference failed: {body}");
    assert!(
        body["choices"][0]["message"]["content"]
            .as_str()
            .is_some_and(|text| !text.trim().is_empty()),
        "no generated response: {body}"
    );
    assert!(
        body["usage"]["completion_tokens"]
            .as_u64()
            .is_some_and(|tokens| tokens > 0)
    );
    tokio::time::timeout(Duration::from_secs(5), handle.stop())
        .await
        .unwrap()
        .unwrap();
}

fn isolated<F: std::future::Future<Output = ()>>(name: &str, scenario: fn() -> F) {
    if std::env::var_os(CHILD).is_some() {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap()
            .block_on(scenario());
        return;
    }
    let temp = tempfile::tempdir().unwrap();
    // Keep an explicit installed-runtime cache for opt-in native tests; otherwise
    // resolve in the fixture without consulting HOME or platform cache paths.
    let runtime_cache = std::env::var_os("MESH_LLM_NATIVE_RUNTIME_CACHE_DIR")
        .filter(|path| !path.is_empty())
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| temp.path().join("native-runtimes"));
    let executable = std::env::current_exe().unwrap();
    #[cfg(target_os = "macos")]
    let executable = {
        // CoreFoundation scans the executable's directory during initialization.
        // A large Cargo deps directory can consume the lifecycle deadline before
        // the node starts. Run identical bytes from this isolated fixture instead.
        let isolated_executable = temp.path().join("host-runtime-tests");
        std::fs::hard_link(&executable, &isolated_executable)
            .or_else(|_| std::fs::copy(&executable, &isolated_executable).map(|_| ()))
            .expect("stage the unchanged child test executable");
        isolated_executable
    };
    let log = std::fs::File::create(temp.path().join("child.log")).unwrap();
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    drop(listener);
    let mut child = Command::new(executable)
        .args([
            "--exact",
            &format!("sdk::embedded_lifecycle_tests::{name}"),
            "--include-ignored",
            "--nocapture",
        ])
        .env(CHILD, temp.path())
        // Lifecycle barriers use local fixtures, not the availability of GitHub.
        .env(mesh_llm_config::MESH_LLM_NO_UPDATE_CHECK_ENV, "1")
        .env("HF_ENDPOINT", endpoint)
        .env("MESH_LLM_NATIVE_RUNTIME_CACHE_DIR", runtime_cache)
        .env("HF_HOME", temp.path().join("hf"))
        .env("HF_HUB_CACHE", temp.path().join("hf/hub"))
        .env("HF_XET_CACHE", temp.path().join("hf/xet"))
        .env("MESH_LLM_RUNTIME_ROOT", temp.path().join("runtime"))
        .env("XDG_CACHE_HOME", temp.path().join("cache"))
        .env_remove("HF_TOKEN")
        .env_remove("HUGGING_FACE_HUB_TOKEN")
        .stdout(Stdio::from(log.try_clone().unwrap()))
        .stderr(Stdio::from(log))
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(90);
    let result = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break Some(status);
        }
        if Instant::now() >= deadline {
            child.kill().unwrap();
            child.wait().unwrap();
            break None;
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let output = std::fs::read_to_string(temp.path().join("child.log")).unwrap();
    assert!(
        result.is_some_and(|status| status.success()),
        "child failed or leaked runtime threads: {result:?}\n{output}"
    );
}

async fn download_scenario() {
    restart_and_stop(cancel_download().await).await;
}

async fn cancel_download() -> EmbeddedMeshNodeConfig {
    let server = StalledHub::start().await;
    let mut config = isolated_config();
    config.mode = EmbeddedMeshNodeMode::Serve;
    config.serving.models = vec!["owner/repo@main:model.gguf".into()];
    let handle = start_embedded_node(config.clone())
        .await
        .expect("start real serving SDK");
    tokio::time::timeout(Duration::from_secs(30), server.downloading.notified())
        .await
        .expect("fixture must observe an active model transfer before cancellation");
    handle
        .status()
        .await
        .expect("status remains available during download");
    // The fixture NEVER completes its response. Only cancellation can release
    // this operation; a longer download timeout cannot make this test pass.
    tokio::time::timeout(Duration::from_secs(5), handle.stop())
        .await
        .expect("stop must interrupt the stalled transfer")
        .expect("runtime stopped cleanly");
    tokio::time::timeout(Duration::from_secs(2), server.disconnected.notified())
        .await
        .expect("stop closed the download connection");
    config
}

fn isolated_config() -> EmbeddedMeshNodeConfig {
    let temp = std::path::PathBuf::from(std::env::var_os(CHILD).unwrap());
    let api = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    let console = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    let mut config = EmbeddedMeshNodeConfig::default();
    config.http.api_port = api.local_addr().unwrap().port();
    config.http.console_port = console.local_addr().unwrap().port();
    config.network.disable_iroh_relays = true;
    config.network.enumerate_host = false;
    config.network.bind_ip = Some(Ipv4Addr::LOCALHOST.into());
    let quic = std::net::UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    config.network.bind_port = Some(quic.local_addr().unwrap().port());
    config.network.mesh_name = Some("sdk-cancellation-regression".into());
    config.admission.owner_key = Some(temp.join("owner.json"));
    drop((api, console, quic));
    config.mode = EmbeddedMeshNodeMode::Client;
    config
}

async fn restart_and_stop(mut config: EmbeddedMeshNodeConfig) {
    for port in [config.http.api_port, config.http.console_port] {
        TcpListener::bind((Ipv4Addr::LOCALHOST, port)).expect("stop released listener");
    }
    std::net::UdpSocket::bind((Ipv4Addr::LOCALHOST, config.network.bind_port.unwrap()))
        .expect("stop released QUIC listener");
    config.network.join_tokens.clear();
    config.mode = EmbeddedMeshNodeMode::Client;
    config.serving.models.clear();
    let restarted = start_embedded_node(config)
        .await
        .expect("restart on the same ports");
    restarted.status().await.expect("restarted SDK responds");
    tokio::time::timeout(Duration::from_secs(5), restarted.stop())
        .await
        .expect("restarted runtime stops promptly")
        .unwrap();
}

async fn unresponsive_peer() -> (tokio::net::UdpSocket, String) {
    use base64::Engine;
    let socket = tokio::net::UdpSocket::bind((Ipv4Addr::LOCALHOST, 0))
        .await
        .unwrap();
    let address = iroh::EndpointAddr::new(iroh::SecretKey::generate().public())
        .with_ip_addr(socket.local_addr().unwrap());
    let token = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(serde_json::to_vec(&address).unwrap());
    (socket, token)
}

async fn join_scenario() {
    let config = isolated_config();
    let (peer, token) = unresponsive_peer().await;
    let handle = start_embedded_node(config.clone()).await.unwrap();
    {
        let join = handle.join_token(token);
        tokio::pin!(join);
        let mut packet = [0; 2048];
        tokio::time::timeout(Duration::from_secs(10), async {
            tokio::select! {
                result = &mut join => panic!("join finished before the fixture gate: {result:?}"),
                packet = peer.recv(&mut packet) => { packet.unwrap(); }
            }
        })
        .await
        .expect("peer observed an actual dial");
        // Dropping the caller's join future does not withdraw the runtime's
        // queued command. Shutdown must interrupt that operation itself.
    }
    handle.status().await.unwrap();
    tokio::time::timeout(Duration::from_secs(5), handle.stop())
        .await
        .expect("shutdown preempts in-flight join")
        .unwrap();
    restart_and_stop(config).await;
}

async fn cancelled_startup_scenario() {
    let mut config = isolated_config();
    let (peer, token) = unresponsive_peer().await;
    config.network.join_tokens = vec![token];
    let mut startup = tokio::spawn(start_embedded_node(config.clone()));
    let mut packet = [0; 2048];
    tokio::time::timeout(Duration::from_secs(10), async {
        tokio::select! {
            result = &mut startup => match result {
                Ok(Err(error)) => panic!("startup failed before dialing the fixture: {error:#}"),
                Ok(Ok(_)) => panic!("startup returned before dialing the fixture"),
                Err(error) => panic!("startup task failed before dialing the fixture: {error}"),
            },
            packet = peer.recv(&mut packet) => { packet.unwrap(); }
        }
    })
    .await
    .expect("startup is dialing the fixture");
    assert!(
        !startup.is_finished(),
        "fixture must hold startup before it returns a handle"
    );
    startup.abort();
    assert!(matches!(startup.await, Err(error) if error.is_cancelled()));
    // Rebinding the QUIC port is a completion barrier for the runtime created
    // before the startup future was dropped, not just for that future.
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if std::net::UdpSocket::bind((Ipv4Addr::LOCALHOST, config.network.bind_port.unwrap()))
                .is_ok()
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("cancelled startup releases its runtime");
    restart_and_stop(config).await;
}

struct StalledHub {
    downloading: Arc<Notify>,
    disconnected: Arc<Notify>,
    task: tokio::task::JoinHandle<()>,
}

impl StalledHub {
    async fn start() -> Self {
        let endpoint = std::env::var("HF_ENDPOINT").unwrap();
        let listener = tokio::net::TcpListener::bind(endpoint.trim_start_matches("http://"))
            .await
            .unwrap();
        let downloading = Arc::new(Notify::new());
        let disconnected = Arc::new(Notify::new());
        let started = downloading.clone();
        let closed = disconnected.clone();
        let task = tokio::spawn(async move {
            let mut requests = tokio::task::JoinSet::new();
            loop {
                let (stream, _) = listener.accept().await.unwrap();
                requests.spawn(serve_hub_request(stream, started.clone(), closed.clone()));
            }
        });
        Self {
            downloading,
            disconnected,
            task,
        }
    }
}

impl Drop for StalledHub {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn serve_hub_request(
    mut stream: tokio::net::TcpStream,
    started: Arc<Notify>,
    closed: Arc<Notify>,
) {
    let mut request = Vec::new();
    while !request.ends_with(b"\r\n\r\n") {
        let mut byte = [0];
        if stream.read_exact(&mut byte).await.is_err() {
            return;
        }
        request.push(byte[0]);
    }
    let request = String::from_utf8(request).unwrap();
    if request.contains("/api/models/owner/repo") {
        let body = r#"{"id":"owner/repo","sha":"0123456789012345678901234567890123456789","siblings":[{"rfilename":"model.gguf","size":1048576}]}"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(response.as_bytes()).await.unwrap();
    } else if request.contains("/resolve/") && request.contains("model.gguf") {
        let header = "HTTP/1.1 200 OK\r\nContent-Length: 1048576\r\nETag: \"fixture-blob\"\r\nx-repo-commit: 0123456789012345678901234567890123456789\r\nConnection: close\r\n\r\n";
        stream.write_all(header.as_bytes()).await.unwrap();
        if request.starts_with("HEAD ") {
            return;
        }
        stream.write_all(b"GGUF").await.unwrap();
        started.notify_one();
        let mut byte = [0];
        let _ = stream.read(&mut byte).await;
        closed.notify_one();
    } else {
        stream
            .write_all(
                b"HTTP/1.1 404 Not Found\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}",
            )
            .await
            .unwrap();
    }
}

#[cfg(windows)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial_test::serial]
async fn embedded_listen_all_rejects_foreign_loopback_console() {
    let foreign = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind foreign console");
    let console_port = foreign.local_addr().expect("foreign address").port();
    let api = TcpListener::bind("127.0.0.1:0").expect("reserve API port");
    let api_port = api.local_addr().expect("API address").port();
    drop(api);

    let result = start_embedded_serve(EmbeddedServeConfig {
        mode: EmbeddedMeshNodeMode::Client,
        api_port,
        console_port,
        listen_all: true,
        startup_timeout: Duration::from_secs(30),
        ..EmbeddedServeConfig::default()
    })
    .await;

    if let Ok(handle) = result {
        let _ = handle.stop().await;
        panic!("embedded startup must reject a foreign loopback console");
    }
}
