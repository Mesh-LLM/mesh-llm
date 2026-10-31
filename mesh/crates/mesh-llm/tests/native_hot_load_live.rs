//! Real native cold-load / idle-unload / reload certification.
//!
//! Uses the shared product/SDK SmolLM2-135M Q8 fixture and a compatible packaged
//! runtime. No model downloads, fake backends or mocked load completions.
//! See mesh/docs/design/TESTING.md, "Native model hot loading" for the command.

use serde_json::{Value, json};
use std::fs::File;
use std::io::Read;
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const MODEL: &str = "native-hot-load-live";

#[tokio::test]
#[ignore = "requires the shared SmolLM2 fixture and a compatible packaged native runtime"]
async fn real_gguf_cold_load_idle_unload_and_reload() {
    let model = fixture_model();
    let model_bytes = model.metadata().unwrap().len();
    let mut server = Server::start(&model);
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap();

    let status_url = format!("http://127.0.0.1:{}/api/runtime", server.control);
    let models_url = format!("http://127.0.0.1:{}/v1/models", server.api);
    server
        .wait_for(&client, &models_url, Duration::from_secs(90), |list| {
            list["data"]
                .as_array()
                .is_some_and(|models| models.iter().any(|model| model["id"] == MODEL))
        })
        .await;
    let cold = server
        .wait_for(&client, &status_url, Duration::from_secs(90), |_| true)
        .await;
    assert!(
        models(&cold).is_empty(),
        "model must start unloaded: {cold}"
    );
    server.generate(&client).await;
    let first = server
        .wait_for(&client, &status_url, Duration::from_secs(10), |status| {
            serving(status).is_some()
        })
        .await;
    let (first_id, first_port) = serving(&first).unwrap();
    let backend: SocketAddr = format!("127.0.0.1:{first_port}").parse().unwrap();
    assert!(TcpStream::connect_timeout(&backend, Duration::from_secs(1)).is_ok());

    server
        .wait_for(&client, &status_url, Duration::from_secs(90), |status| {
            models(status).is_empty()
        })
        .await;
    let deadline = Instant::now() + Duration::from_secs(10);
    while TcpStream::connect_timeout(&backend, Duration::from_millis(200)).is_ok() {
        assert!(
            Instant::now() < deadline,
            "unloaded backend still accepts connections"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert_eq!(
        model.metadata().unwrap().len(),
        model_bytes,
        "disk weights must remain cached"
    );
    server.generate(&client).await;
    let second = server
        .wait_for(&client, &status_url, Duration::from_secs(10), |status| {
            serving(status).is_some()
        })
        .await;
    let (second_id, _) = serving(&second).unwrap();
    assert_ne!(
        first_id, second_id,
        "second request must create a new native instance"
    );
    server.assert_alive();
}

/// Reuse the shared product/SDK smoke artifact, including its canonical cache name.
fn fixture_model() -> PathBuf {
    let manifest: Value = serde_json::from_str(include_str!(
        "../../../../ci/model-artifacts/manifests/product-smoke.json"
    ))
    .unwrap();
    let artifact = manifest["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|artifact| artifact["id"] == "smollm2-q8-inference")
        .expect("shared SmolLM2 inference fixture");
    let model = std::env::var_os("MESH_LLM_HOT_LOAD_TEST_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let home = std::env::var_os("HOME")
                .or_else(|| std::env::var_os("USERPROFILE"))
                .expect("set MESH_LLM_HOT_LOAD_TEST_GGUF to the restored SmolLM2 fixture");
            PathBuf::from(home)
                .join(".models")
                .join(artifact["file"].as_str().unwrap())
        });
    assert!(model.is_absolute(), "GGUF path must be absolute");
    let model = model
        .canonicalize()
        .expect("restore the shared SmolLM2 fixture first; see TESTING.md");
    let mut magic = [0; 4];
    File::open(&model).unwrap().read_exact(&mut magic).unwrap();
    assert_eq!(&magic, b"GGUF", "fixture must be a real GGUF");
    assert_eq!(
        model.metadata().unwrap().len(),
        artifact["size_bytes"].as_u64().unwrap(),
        "use the shared smollm2-q8-inference artifact, not a different model"
    );
    model
}

fn models(status: &Value) -> &[Value] {
    status["models"]
        .as_array()
        .expect("/api/runtime models array")
}

fn serving(status: &Value) -> Option<(String, u16)> {
    let entry = models(status)
        .iter()
        .find(|entry| entry["name"] == MODEL && entry["status"] == "ready")?;
    Some((
        entry["instance_id"].as_str()?.to_owned(),
        u16::try_from(entry["port"].as_u64()?).ok()?,
    ))
}

struct Server {
    child: Child,
    root: tempfile::TempDir,
    api: u16,
    control: u16,
}

impl Server {
    fn start(model: &std::path::Path) -> Self {
        let root = tempfile::tempdir().unwrap();
        // Keep both reservations alive while choosing distinct ephemeral ports.
        let api = TcpListener::bind("127.0.0.1:0").unwrap();
        let control = TcpListener::bind("127.0.0.1:0").unwrap();
        let api_port = api.local_addr().unwrap().port();
        let control_port = control.local_addr().unwrap().port();
        let config = root.path().join("config.toml");
        let path = serde_json::to_string(model.to_str().expect("UTF-8 model path")).unwrap();
        std::fs::write(
            &config,
            format!(
                "version = 1\n[runtime]\nmode = \"serve\"\n[[models]]\nmodel = \"{MODEL}\"\n\
             [models.lifecycle]\nload = \"on_demand\"\nidle_timeout_secs = 15\nprefetch = false\n\
             [models.hardware]\nmodel_path = {path}\n\
             [models.skippy]\nsource_policy = \"local-required\"\n\
             [models.model_fit]\nctx_size = 512\n"
            ),
        )
        .unwrap();
        let log = File::create(root.path().join("server.log")).unwrap();
        drop((api, control));
        let child = Command::new(env!("CARGO_BIN_EXE_mesh-llm"))
            .args([
                "serve",
                "--headless",
                "--disable-iroh-relays",
                "--device",
                "CPU",
                "--bind-ip",
                "127.0.0.1",
                "--bind-port",
                "0",
                "--log-format",
                "json",
            ])
            .arg("--config")
            .arg(config)
            .arg("--port")
            .arg(api_port.to_string())
            .arg("--console")
            .arg(control_port.to_string())
            .env("HOME", root.path())
            .env("MESH_LLM_DATA_DIR", root.path().join("data"))
            .env("MESH_LLM_RUNTIME_ROOT", root.path().join("runtime"))
            .stdin(Stdio::null())
            .stdout(log.try_clone().unwrap())
            .stderr(log)
            .spawn()
            .expect("start shipped Mesh binary");
        Self {
            child,
            root,
            api: api_port,
            control: control_port,
        }
    }

    fn logs(&self) -> String {
        std::fs::read_to_string(self.root.path().join("server.log")).unwrap_or_default()
    }

    fn assert_alive(&mut self) {
        let exit = self.child.try_wait().unwrap();
        assert!(exit.is_none(), "Mesh exited {exit:?}:\n{}", self.logs());
    }

    async fn wait_for(
        &mut self,
        client: &reqwest::Client,
        url: &str,
        timeout: Duration,
        predicate: impl Fn(&Value) -> bool,
    ) -> Value {
        let deadline = Instant::now() + timeout;
        let mut last = Value::Null;
        loop {
            self.assert_alive();
            if let Ok(response) = client.get(url).send().await
                && response.status().is_success()
                && let Ok(status) = response.json::<Value>().await
            {
                if predicate(&status) {
                    return status;
                }
                last = status;
            }
            assert!(
                Instant::now() < deadline,
                "runtime state deadline; last={last}\n{}",
                self.logs()
            );
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }

    async fn generate(&mut self, client: &reqwest::Client) {
        self.assert_alive();
        let response = client
            .post(format!("http://127.0.0.1:{}/v1/completions", self.api))
            .timeout(Duration::from_secs(300))
            .json(&json!({"model": MODEL, "prompt": "Hello", "max_tokens": 4, "temperature": 0}))
            .send()
            .await
            .unwrap_or_else(|e| panic!("generation transport: {e}\n{}", self.logs()));
        let status = response.status();
        let body = response.text().await.unwrap();
        assert!(
            status.is_success(),
            "generation {status}: {body}\n{}",
            self.logs()
        );
        let output: Value = serde_json::from_str(&body).expect("native completion JSON");
        assert!(
            output["choices"]
                .as_array()
                .is_some_and(|choices| !choices.is_empty()),
            "{output}"
        );
        assert!(
            output["usage"]["completion_tokens"]
                .as_u64()
                .is_some_and(|count| count > 0),
            "{output}"
        );
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
