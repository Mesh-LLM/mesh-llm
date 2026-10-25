//! Inert executable boundary for competitive coordinator fixtures only.
use serde_json::{Value, json};
use std::{io::Write, path::Path, time::Duration};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    task::JoinSet,
    time::{Instant, timeout, timeout_at},
};

pub(super) fn selected(args: &[String]) -> bool {
    args.iter()
        .any(|arg| arg == "--pp" || arg == "serve-openai")
        || (args.first().is_some_and(|arg| arg == "serve")
            && args.iter().any(|arg| arg == "--model-id")
            && !args.iter().any(|arg| arg == "--console"))
        || (args.iter().any(|arg| arg == "--port") && !args.iter().any(|arg| arg == "--console"))
}
fn value<'a>(args: &'a [String], flag: &str) -> Result<&'a str, Box<dyn std::error::Error>> {
    args.windows(2)
        .find(|pair| pair[0] == flag)
        .map(|pair| pair[1].as_str())
        .ok_or_else(|| format!("fixture missing {flag}").into())
}
pub(super) fn run(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.iter().any(|arg| arg == "--pp") {
        return benchy(args);
    }
    let (port, model, root) = if args
        .iter()
        .any(|arg| arg == "serve-openai" || arg == "serve")
    {
        let stage: Value = serde_json::from_slice(&std::fs::read(value(args, "--config")?)?)?;
        let root = Path::new(stage["model_path"].as_str().ok_or("model path")?)
            .parent()
            .ok_or("root")?
            .to_path_buf();
        (
            value(args, "--bind-addr")?
                .rsplit(':')
                .next()
                .ok_or("bind")?
                .parse::<u16>()?,
            value(args, "--model-id")?.to_owned(),
            root,
        )
    } else {
        let model_path = value(args, "--model")?;
        (
            value(args, "--port")?.parse::<u16>()?,
            value(args, "--alias")?.to_owned(),
            Path::new(model_path).parent().ok_or("root")?.to_path_buf(),
        )
    };
    std::fs::write(root.join("server-argv.json"), serde_json::to_vec(args)?)?;
    crate::signals::install()?;
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?
        .block_on(serve(port, &root, &model))?;
    std::fs::write(root.join("server-stopped"), b"owned server stopped")?;
    Ok(())
}

// Accepts do not create OS threads or poll-sleep between connections. Task count
// stays bounded; every accepted socket has one whole-response deadline.
async fn serve(port: u16, root: &Path, model: &str) -> Result<(), Box<dyn std::error::Error>> {
    let listener = TcpListener::bind(("127.0.0.1", port)).await?;
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut responders = JoinSet::new();
    let result = accept_until_stop(&listener, root, model, deadline, &mut responders).await;
    if result.is_err() {
        responders.abort_all();
    }
    let drained = drain(&mut responders).await;
    result?;
    drained?;
    Ok(())
}
async fn accept_until_stop(
    listener: &TcpListener,
    root: &Path,
    model: &str,
    deadline: Instant,
    responders: &mut JoinSet<Result<(), String>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut stop = tokio::time::interval(Duration::from_millis(1));
    stop.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        if crate::signals::stopped() || Instant::now() >= deadline {
            return Ok(());
        }
        tokio::select! {
            // Admit a ready burst promptly, before running response completion
            // bookkeeping. The branch is disabled at the owned active-task cap.
            biased;
            accepted=listener.accept(), if responders.len()<512 => {
                let (stream,_)=accepted?;let root=root.to_path_buf();let model=model.to_owned();
                let response_deadline=Instant::now()+Duration::from_secs(1);
                responders.spawn(async move {
                    timeout_at(response_deadline,respond(stream,&root,&model)).await
                        .map_err(|_|"fixture whole-response deadline".to_owned())?
                        .map_err(|error|error.to_string())
                });
            }
            joined=responders.join_next(), if !responders.is_empty() => {
                if let Some(joined)=joined {joined.map_err(|_|"fixture responder panicked")?.map_err(|error|->Box<dyn std::error::Error>{error.into()})?;}
            }
            _=stop.tick()=>{}
        }
    }
}
async fn drain(
    responders: &mut JoinSet<Result<(), String>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let drained = timeout(Duration::from_secs(2), async {
        while let Some(joined) = responders.join_next().await {
            joined
                .map_err(|_| "fixture responder cancelled/panicked")?
                .map_err(|error| -> Box<dyn std::error::Error> { error.into() })?;
        }
        Ok::<_, Box<dyn std::error::Error>>(())
    })
    .await;
    if !matches!(&drained, Ok(Ok(()))) {
        responders.abort_all();
        while responders.join_next().await.is_some() {}
    }
    drained.map_err(|_| "fixture response drain deadline")??;
    Ok(())
}
async fn respond(
    mut stream: TcpStream,
    root: &Path,
    model: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut bytes = Vec::new();
    let mut buffer = [0; 4096];
    let end = loop {
        let count = stream.read(&mut buffer).await?;
        if count == 0 {
            return Ok(());
        }
        bytes.extend_from_slice(&buffer[..count]);
        if bytes.len() > 65536 {
            return Err("fixture request bound".into());
        }
        if let Some(end) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
            break end + 4;
        }
    };
    let headers = String::from_utf8_lossy(&bytes[..end]);
    let length = headers
        .lines()
        .find_map(|line| {
            line.to_ascii_lowercase()
                .strip_prefix("content-length:")
                .and_then(|value| value.trim().parse::<usize>().ok())
        })
        .unwrap_or(0);
    if length > 65536 {
        return Err("fixture body bound".into());
    }
    let models = headers.starts_with("GET /v1/models ");
    while bytes.len() < end + length {
        let count = stream.read(&mut buffer).await?;
        if count == 0 {
            return Ok(());
        }
        bytes.extend_from_slice(&buffer[..count]);
        if bytes.len() > end + 65536 + buffer.len() {
            return Err("fixture request/body bound".into());
        }
    }
    let payload = if models {
        serde_json::to_string(&json!({"data":[{"id":model}]}))?
    } else {
        let body: Value = serde_json::from_slice(&bytes[end..end + length])?;
        let mut log = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(root.join("captured-requests.jsonl"))?;
        log.write_all(format!("{body}\n").as_bytes())?;
        let tokens = if root.join("short-response").exists() {
            1
        } else {
            body["max_tokens"].as_u64().ok_or("output budget")?
        };
        format!(
            "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
            json!({"choices":[{"delta":{"content":"fixture"},"finish_reason":"stop"}]}),
            json!({"choices":[],"usage":{"prompt_tokens":32,"completion_tokens":tokens,"prompt_tokens_details":{"cached_tokens":24}}})
        )
    };
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
        payload.len()
    );
    stream.write_all(response.as_bytes()).await?;
    Ok(())
}

fn benchy(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if !args.iter().any(|arg| arg == "--save-result") {
        return Ok(());
    }
    let path = value(args, "--save-result")?;
    let progress = value(args, "--emit-progress")?;
    let concurrency = value(args, "--concurrency")?.parse::<u64>()?;
    let runs = value(args, "--runs")?.parse::<u64>()?;
    let tokens = value(args, "--tg")?.parse::<u64>()?;
    let root = Path::new(value(args, "--tokenizer")?)
        .parent()
        .ok_or("root")?;
    std::fs::write(root.join("benchy-argv.json"), serde_json::to_vec(args)?)?;
    let mut file = std::fs::File::create(progress)?;
    for _ in 0..concurrency.checked_mul(runs).ok_or("count overflow")? {
        writeln!(
            file,
            "{}",
            json!({"type":"request_end","total_tokens":tokens,"error":if root.join("benchy-hidden-error").exists(){Value::String("fixture error".into())}else{Value::Null}})
        )?;
    }
    std::fs::write(
        path,
        serde_json::to_vec(
            &json!({"benchmarks":[{"response_size":tokens,"tg_throughput":{"mean":12.0}}]}),
        )?,
    )?;
    Ok(())
}
