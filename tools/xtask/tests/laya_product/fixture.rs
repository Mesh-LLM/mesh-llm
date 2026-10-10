use std::io::{Read, Write};
use std::net::TcpListener;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<String> = std::env::args().collect();
    let port = arguments
        .windows(2)
        .find(|pair| pair[0] == "--port")
        .ok_or("missing fixture port")?[1]
        .parse::<u16>()?;
    let listener = TcpListener::bind(("127.0.0.1", port))?;
    if let Some(console) = arguments.windows(2).find(|pair| pair[0] == "--console") {
        let console = TcpListener::bind(("127.0.0.1", console[1].parse::<u16>()?))?;
        std::thread::spawn(move || {
            for stream in console.incoming() {
                let mut stream = stream.unwrap();
                stream
                    .set_read_timeout(Some(std::time::Duration::from_secs(2)))
                    .unwrap();
                let mut request = [0; 4096];
                assert!(stream.read(&mut request).unwrap() > 0);
                let body = br#"{"models":[{"context_length":131072}]}"#;
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                )
                .unwrap();
                stream.write_all(body).unwrap();
            }
        });
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../ci/llama-canary/fixtures/laya-golden");
    let fixtures = std::fs::read_dir(root)?
        .map(|entry| {
            let path = entry?.path();
            if path
                .extension()
                .is_some_and(|extension| extension == "json")
                && path.file_stem().is_some_and(|stem| stem != "manifest")
            {
                Ok(Some(serde_json::from_slice::<serde_json::Value>(
                    &std::fs::read(path)?,
                )?))
            } else {
                Ok(None)
            }
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    for stream in listener.incoming() {
        let mut stream = stream?;
        stream.set_read_timeout(Some(std::time::Duration::from_secs(2)))?;
        let mut bytes = Vec::new();
        let (header_end, length) = loop {
            let mut buffer = [0; 4096];
            let read = stream.read(&mut buffer)?;
            if read == 0 {
                return Err("incomplete fixture request".into());
            }
            bytes.extend_from_slice(&buffer[..read]);
            if let Some(end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
                let header = std::str::from_utf8(&bytes[..end])?;
                let length = header
                    .lines()
                    .find_map(|line| {
                        line.split_once(':')
                            .filter(|(key, _)| key.eq_ignore_ascii_case("content-length"))
                            .map(|(_, value)| value.trim().parse::<usize>())
                    })
                    .transpose()?
                    .unwrap_or(0);
                break (end + 4, length);
            }
        };
        while bytes.len() - header_end < length {
            let mut buffer = [0; 4096];
            let read = stream.read(&mut buffer)?;
            if read == 0 {
                return Err("incomplete fixture body".into());
            }
            bytes.extend_from_slice(&buffer[..read]);
        }
        if bytes.starts_with(b"POST /v1/chat/completions ") {
            let request: serde_json::Value = serde_json::from_slice(&bytes[header_end..])?;
            if request["prompt_cache_key"] == "s" && request["max_tokens"] != 1 {
                return Err("context probe did not request exactly one token".into());
            }
            if let Some(root) = std::env::var_os("MESH_LLM_RUNTIME_ROOT") {
                let directory = std::path::PathBuf::from(root).join("fixture/logs");
                std::fs::create_dir_all(&directory)?;
                let mut log = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(directory.join("skippy-native.log"))?;
                let count = request["messages"]
                    .as_array()
                    .ok_or("missing fixture messages")?
                    .len();
                let event = serde_json::json!({
                    "event":"stage.openai_kv_lookup_decision", "start_time_unix_nanos":count,
                    "attributes":{"openai.prompt_cache_key":request["prompt_cache_key"],
                        "skippy.kv.decision":if count > 1 { "exact_hit" } else { "miss" },
                        "skippy.exact_cache.payload_kind":"kv-recurrent", "skippy.exact_cache.restored_tokens":30}
                });
                serde_json::to_writer(&mut log, &event)?;
                log.write_all(b"\n")?;
            }
            if request["prompt_cache_key"] == "failed" {
                stream.write_all(b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")?;
                continue;
            }
            let body = b"data: {\"choices\":[{\"delta\":{\"content\":\"fixture answer\"},\"finish_reason\":\"stop\"}]}\n\ndata: {\"usage\":{\"prompt_tokens\":40,\"completion_tokens\":2,\"prompt_tokens_details\":{\"cached_tokens\":30}}}\n\ndata: [DONE]\n\n";
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            )?;
            stream.write_all(body)?;
            continue;
        }
        let response = if bytes.starts_with(b"GET /v1/models ") {
            serde_json::json!({"data":[{"id":"laya-fixture"}]})
        } else {
            let request: serde_json::Value = serde_json::from_slice(&bytes[header_end..])?;
            let fixture = fixtures
                .iter()
                .flatten()
                .find(|fixture| {
                    fixture["state"] == request["state"]
                        && fixture["questions"] == request["questions"]
                })
                .ok_or("unknown fixture request")?;
            serde_json::json!({"answers":fixture["answers"]})
        };
        let body = serde_json::to_vec(&response)?;
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        )?;
        stream.write_all(&body)?;
    }
    Ok(())
}
