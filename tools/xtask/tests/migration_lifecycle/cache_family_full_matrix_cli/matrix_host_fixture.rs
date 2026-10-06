//! Inert subprocess-only protocol host; never linked to a model/runtime.
use serde_json::{Value, json};
use std::{
    io::Write as _,
    path::PathBuf,
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpStream,
    task::JoinSet,
};
static STOP: AtomicBool = AtomicBool::new(false);
extern "C" fn stop(_signal: i32) {
    STOP.store(true, Ordering::SeqCst);
}
struct Signal(libc::sighandler_t);
impl Signal {
    fn install() -> Self {
        let prior = unsafe { libc::signal(libc::SIGTERM, stop as *const () as libc::sighandler_t) };
        assert_ne!(prior, libc::SIG_ERR);
        Self(prior)
    }
}
impl Drop for Signal {
    fn drop(&mut self) {
        unsafe {
            libc::signal(libc::SIGTERM, self.0);
        }
    }
}
async fn request(socket: &mut TcpStream) -> Result<(String, Option<Value>), String> {
    let mut bytes = Vec::new();
    loop {
        let mut b = [0; 4096];
        let n = socket.read(&mut b).await.map_err(|e| e.to_string())?;
        if n == 0 {
            return Err("early request EOF".into());
        }
        bytes.extend_from_slice(&b[..n]);
        if bytes.len() > 128 * 1024 {
            return Err("oversized fixture request".into());
        }
        if let Some(end) = bytes.windows(4).position(|b| b == b"\r\n\r\n") {
            let h = std::str::from_utf8(&bytes[..end]).map_err(|e| e.to_string())?;
            let first = h.lines().next().ok_or("missing request")?.to_owned();
            let length = h
                .lines()
                .find_map(|l| {
                    l.to_ascii_lowercase()
                        .strip_prefix("content-length:")
                        .map(str::trim)
                        .map(str::to_owned)
                })
                .map(|s| s.parse::<usize>())
                .transpose()
                .map_err(|e| e.to_string())?
                .unwrap_or(0);
            if bytes.len() >= end + 4 + length {
                let body = if length == 0 {
                    None
                } else {
                    Some(
                        serde_json::from_slice(&bytes[end + 4..end + 4 + length])
                            .map_err(|e| e.to_string())?,
                    )
                };
                return Ok((first, body));
            }
        }
    }
}
fn declared(root: &std::path::Path) -> Value {
    std::fs::read(root.join("artifact-input.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
        .unwrap_or_else(|| json!({"model_id":"Qwen/Qwen3-0.6B:Q8_0","case_key":"qwen3_dense"}))
}
async fn respond(mut socket: TcpStream, root: PathBuf, native: bool) -> Result<(), String> {
    let declaration = declared(&root);
    let (line, body) = request(&mut socket).await?;
    let mode = std::fs::read_to_string(root.join("mode")).unwrap_or_default();
    let bytes = if line.starts_with("GET /health ") && native {
        serde_json::to_vec(&json!({"status":"ok"})).unwrap()
    } else if line.starts_with("GET /v1/models ") && !native {
        serde_json::to_vec(
            &json!({"data":[{"id":if mode=="bad-readiness"{"wrong-model"}else{declaration["model_id"].as_str().ok_or("declared model ID")?}}]}),
        )
        .unwrap()
    } else {
        let body = body.ok_or("missing POST body")?;
        if native {
            if !line.starts_with("POST /completion ")
                || body["prompt"] != "fixed prompt"
                || body["cache_prompt"] != true
                || body["temperature"] != 0
                || body["top_k"] != 1
            {
                return Err("native body mismatch".into());
            }
        } else if !line.starts_with("POST /v1/chat/completions ")
            || body["model"] != declaration["model_id"]
            || body["messages"] != json!([{"role":"user","content":"fixed prompt"}])
            || body["stream"] != true
            || body["stream_options"] != json!({"include_usage":true})
        {
            return Err("OpenAI body mismatch".into());
        }
        let side = std::env::var("CACHE_MATRIX_SIDE").unwrap();
        std::fs::write(root.join(format!("post-{side}")), b"typed-request")
            .map_err(|e| e.to_string())?;
        if mode == "sibling-drift" && side == "new" {
            let primary = PathBuf::from(
                declaration["model"]
                    .as_str()
                    .ok_or("declared shard primary")?,
            );
            let sibling = primary
                .parent()
                .ok_or("shard parent")?
                .join("MiniMax-M2.7-UD-Q2_K_XL-00002-of-00003.gguf");
            std::fs::write(&sibling, b"causal serving-time sibling mutation")
                .map_err(|e| e.to_string())?;
            std::fs::write(
                root.join("sibling-drift-observed"),
                b"new POST while host alive",
            )
            .map_err(|e| e.to_string())?;
        }
        if mode == "hold" && std::env::var("CACHE_MATRIX_SIDE").unwrap_or_default() == "new" {
            while !STOP.load(Ordering::SeqCst) {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            return Ok(());
        }
        if mode == "post-run-identity-failure" && side == "new" {
            let mut binary = std::fs::OpenOptions::new()
                .append(true)
                .open(root.join("new"))
                .map_err(|e| e.to_string())?;
            binary
                .write_all(b"\n# owned post-launch fixture byte drift\n")
                .map_err(|e| e.to_string())?;
            binary.flush().map_err(|e| e.to_string())?;
        }
        if native {
            let mut value = json!({"stop":true,"content":if body["stream"]==true{""}else{"hi"},"tokens_predicted":1,"tokens_evaluated":40,"tokens_cached":41,"truncated":false,"model":declaration["model_id"],"stop_type":"limit","timings":{"prompt_n":10,"cache_n":30,"predicted_n":1,"prompt_ms":2,"predicted_ms":3}});
            if mode == "later-failure"
                && std::env::var("CACHE_MATRIX_SIDE").unwrap_or_default() == "new"
            {
                value.as_object_mut().unwrap().remove("tokens_predicted");
            }
            if body["stream"] == true {
                format!(
                    "data: {}\n\ndata: {value}\n\n",
                    json!({"stop":false,"content":"hi"})
                )
                .into_bytes()
            } else {
                serde_json::to_vec(&value).unwrap()
            }
        } else {
            if mode == "later-failure"
                && std::env::var("CACHE_MATRIX_SIDE").unwrap_or_default() == "new"
            {
                return socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 14\r\nConnection: close\r\n\r\ndata: [DONE]\n\n").await.map_err(|e|e.to_string());
            }
            format!("data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",json!({"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}),json!({"usage":{"prompt_tokens":40,"completion_tokens":1,"prompt_tokens_details":{"cached_tokens":30}}})).into_bytes()
        }
    };
    socket
        .write_all(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                bytes.len()
            )
            .as_bytes(),
        )
        .await
        .map_err(|e| e.to_string())?;
    socket.write_all(&bytes).await.map_err(|e| e.to_string())?;
    Ok(())
}
async fn serve(root: PathBuf) -> Result<(), String> {
    let native = std::env::var("CACHE_FIXTURE_NATIVE").unwrap() == "1";
    let address: std::net::SocketAddr = std::env::var("CACHE_FIXTURE_BIND")
        .unwrap()
        .parse()
        .map_err(|_| "invalid bind")?;
    if !address.ip().is_loopback() || address.port() == 0 {
        return Err("invalid fixture loopback bind".into());
    }

    if native {
        let model =
            PathBuf::from(std::env::var_os("CACHE_FIXTURE_MODEL").ok_or("native model flag")?);
        if model != std::env::var_os("CACHE_MATRIX_MODEL").unwrap() || !model.is_file() {
            return Err("native canonical model argv mismatch".into());
        }
    } else {
        let path =
            PathBuf::from(std::env::var_os("CACHE_FIXTURE_CONFIG").ok_or("stage config flag")?);
        let config: Value =
            serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        if config["model_path"]
            != json!(PathBuf::from(
                std::env::var_os("CACHE_MATRIX_MODEL").unwrap()
            ))
            || config["layer_start"] != 0
            || config["layer_end"]
                != if declared(&root)["case_key"] == "minimax_m27" {
                    62
                } else {
                    28
                }
            || config["ctx_size"] != 512
            || config["lane_count"] != 1
            || config["n_gpu_layers"] != 0
            || config["load_mode"] != "runtime-slice"
        {
            return Err("actual stage configuration mismatch".into());
        }
    }
    let listener = tokio::net::TcpListener::bind(address)
        .await
        .map_err(|e| e.to_string())?;

    let side = std::env::var("CACHE_MATRIX_SIDE").map_err(|_| "fixture side")?;
    let mut launches = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(root.join(format!("host-launches-{side}")))
        .map_err(|e| e.to_string())?;
    writeln!(launches, "{}", std::process::id()).map_err(|e| e.to_string())?;
    let mode = std::fs::read_to_string(root.join("mode")).unwrap_or_default();
    if mode == "signal-stop" {
        unsafe {
            libc::signal(libc::SIGTERM, libc::SIG_DFL);
        }
    }
    if mode == "early-exit" {
        return Err("fixture early exit".into());
    }
    if native {
        println!("srv  llama_server: listening on http://{address}");
    } else {
        println!(
            "skippy-server listening: openai={address} model_id={} backend=fixture generation_concurrency=1 generation_queue_capacity=256 generation_admission_timeout_secs=30",
            declared(&root)["model_id"]
                .as_str()
                .ok_or("declared model ID")?
        );
    }
    std::io::stdout().flush().map_err(|e| e.to_string())?;
    let until = tokio::time::Instant::now() + Duration::from_secs(30);
    let mut tasks = JoinSet::new();
    let mut error = None;
    while !STOP.load(Ordering::SeqCst) && tokio::time::Instant::now() < until {
        tokio::select! {biased;
            result=tasks.join_next(),if !tasks.is_empty()=>{if !matches!(result,Some(Ok(Ok(())))){error=Some("fixture response failed".to_owned());break;}},
            result=listener.accept(),if tasks.len()<16=>{let(socket,_)=match result {Ok(value)=>value,Err(failure)=>{error=Some(failure.to_string());break;}};let root=root.clone();tasks.spawn(async move{tokio::time::timeout(Duration::from_secs(10),respond(socket,root,native)).await.map_err(|_|"fixture response deadline".to_owned())?});},
            ()=tokio::time::sleep(Duration::from_millis(5))=>{}
        }
    }
    tasks.abort_all();
    while tasks.join_next().await.is_some() {}
    if let Some(error) = error {
        return Err(error);
    }
    if !STOP.load(Ordering::SeqCst) {
        return Err("fixture host deadline".into());
    }
    Ok(())
}
pub(super) fn run() {
    STOP.store(false, Ordering::SeqCst);
    let _signal = Signal::install();
    let root = PathBuf::from(std::env::var_os("CACHE_FIXTURE_ROOT").unwrap());
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(serve(root)).unwrap();
}
