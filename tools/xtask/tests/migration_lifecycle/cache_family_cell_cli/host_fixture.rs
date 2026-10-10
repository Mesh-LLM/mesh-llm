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
async fn respond(mut socket: TcpStream, root: PathBuf, native: bool) -> Result<(), String> {
    let (line, body) = request(&mut socket).await?;
    let mode = std::fs::read_to_string(root.join("mode")).unwrap_or_default();
    let bytes = if line.starts_with("GET /health ") && native {
        serde_json::to_vec(&json!({"status":"ok"})).unwrap()
    } else if line.starts_with("GET /v1/models ") && !native {
        serde_json::to_vec(
            &json!({"data":[{"id":if mode=="bad-readiness"{"wrong-model"}else{"fixture"}}]}),
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
            || body["model"] != "fixture"
            || body["messages"] != json!([{"role":"user","content":"fixed prompt"}])
            || body["stream"] != true
            || body["stream_options"] != json!({"include_usage":true})
        {
            return Err("OpenAI body mismatch".into());
        }
        std::fs::write(root.join("post-admitted"), b"typed-request").map_err(|e| e.to_string())?;
        if mode == "hold" {
            while !STOP.load(Ordering::SeqCst) {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            return Ok(());
        }
        if native {
            let value = json!({"stop":true,"content":if body["stream"]==true{""}else{"hi"},"tokens_predicted":1,"tokens_evaluated":40,"tokens_cached":41,"truncated":false,"model":"fixture","stop_type":"limit","timings":{"prompt_n":10,"cache_n":30,"predicted_n":1,"prompt_ms":2,"predicted_ms":3}});
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
        if model != root.join("model.gguf") || !model.is_file() {
            return Err("native canonical model argv mismatch".into());
        }
    } else {
        let path =
            PathBuf::from(std::env::var_os("CACHE_FIXTURE_CONFIG").ok_or("stage config flag")?);
        let config: Value =
            serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        if config["model_path"] != json!(root.join("model.gguf"))
            || config["layer_start"] != 0
            || config["layer_end"] != 6
            || config["ctx_size"] != 128
            || config["lane_count"] != 1
            || config["n_gpu_layers"] != -1
            || config["load_mode"] != "runtime-slice"
        {
            return Err("actual stage configuration mismatch".into());
        }
    }
    let listener = tokio::net::TcpListener::bind(address)
        .await
        .map_err(|e| e.to_string())?;

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
    } else if std::env::var_os("CACHE_FIXTURE_CURRENT").is_some() {
        println!(
            "{}",
            serde_json::json!({"schema_version":1,"sequence":1,"type":"status","data":{"message":format!("skippy-serving listening: openai={address}"),"context":null}})
        );
    } else {
        println!(
            "skippy-server listening: openai={address} model_id=fixture backend=fixture generation_concurrency=1 generation_queue_capacity=256 generation_admission_timeout_secs=30"
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
