use super::http::{self, Request, Response};
use crate::process::Cancellation;
use serde::Deserialize;
use std::{
    path::Path,
    time::{Duration, Instant},
};

#[derive(Deserialize)]
struct Page {
    items: Vec<Record>,
}
#[derive(Deserialize)]
struct Record {
    #[serde(rename = "requestId")]
    request_id: String,
}

pub(in crate::automation) enum Check {
    Ready,
    Persist,
    Restart(String),
    Access,
    Replay,
}

pub(in crate::automation) struct CheckContext<'a> {
    pub base: u16,
    pub root: &'a Path,
    pub wait: Duration,
    pub cancellation: &'a Cancellation,
}

pub(in crate::automation) fn execute(
    check: Check,
    context: CheckContext<'_>,
) -> Result<String, String> {
    let CheckContext {
        base,
        root,
        wait,
        cancellation,
    } = context;
    match check {
        Check::Ready => {
            let until = Instant::now() + wait;
            loop {
                if cancellation.is_cancelled() {
                    return Err("logging check cancelled".into());
                }
                if let Ok(response) = fetch(base + 1, "/api/status", Vec::new(), Vec::new(), false)
                    && response.status < 400
                {
                    write(root, "status.json", &response.body)?;
                    return Ok(String::new());
                }
                if Instant::now() >= until {
                    return Err("console readiness deadline".into());
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        }
        Check::Persist => {
            rejected(base, root, "persisted-request")?;
            let until = Instant::now() + wait;
            loop {
                if cancellation.is_cancelled() {
                    return Err("logging check cancelled".into());
                }
                if let Ok(response) = fetch(
                    base + 1,
                    "/api/logs/requests?limit=10",
                    Vec::new(),
                    Vec::new(),
                    false,
                ) && let Ok(page) = serde_json::from_slice::<Page>(&response.body)
                    && let Some(record) = page
                        .items
                        .into_iter()
                        .next()
                        .filter(|record| !record.request_id.is_empty())
                {
                    write(root, "list-before-restart.json", &response.body)?;
                    return Ok(record.request_id);
                }
                if Instant::now() >= until {
                    return Err("durable request deadline".into());
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        }
        Check::Restart(id) => {
            if !id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
            {
                return Err("invalid durable request identity".into());
            }
            let response = fetch(
                base + 1,
                &format!("/api/logs/requests/{id}"),
                Vec::new(),
                Vec::new(),
                false,
            )?;
            write(root, "persisted-detail.json", &response.body)?;
            if response.status >= 400 {
                return Err("durable detail missing after restart".into());
            }
            Ok(id)
        }
        Check::Access => {
            for (name, headers) in [
                (
                    "hostile-host",
                    vec![("host".into(), "attacker.example".into())],
                ),
                (
                    "hostile-origin",
                    vec![
                        ("host".into(), format!("localhost:{}", base + 1)),
                        ("origin".into(), "https://attacker.example".into()),
                    ],
                ),
            ] {
                let response = fetch(base + 1, "/api/logs/requests", Vec::new(), headers, false)?;
                write(root, &format!("{name}.json"), &response.body)?;
                if response.status != 403 {
                    return Err("hostile request was not forbidden".into());
                }
            }
            rejected(base, root, "replay-seed-one")?;
            rejected(base, root, "replay-seed-two")?;
            Ok(String::new())
        }
        Check::Replay => {
            let response = fetch(
                base + 1,
                "/api/logs/events?channel=requests&cursor=v1%3A0.0.0",
                Vec::new(),
                vec![
                    ("accept".into(), "text/event-stream".into()),
                    ("last-event-id".into(), "v1:0.0.0".into()),
                ],
                true,
            )?;
            write(root, "sse-replay.body", &response.body)?;
            if response.status != 200
                || !response
                    .body
                    .windows(b"event: replay_gap".len())
                    .any(|bytes| bytes == b"event: replay_gap")
                || !response
                    .body
                    .windows(b"/api/logs/requests".len())
                    .any(|bytes| bytes == b"/api/logs/requests")
            {
                return Err("dedicated SSE replay gap missing".into());
            }
            Ok(String::new())
        }
    }
}

fn rejected(base: u16, root: &Path, name: &str) -> Result<(), String> {
    let body = serde_json::to_vec(&serde_json::json!({"model":"qa-no-model","messages":[{"role":"user","content":"real console QA request"}]})).map_err(|_| "chat request encoding")?;
    let response = fetch(base, "/v1/chat/completions", body, Vec::new(), false)?;
    write(root, &format!("{name}.json"), &response.body)
}

fn fetch(
    port: u16,
    path: &str,
    body: Vec<u8>,
    headers: Vec<(String, String)>,
    partial: bool,
) -> Result<Response, String> {
    http::transfer(Request {
        port,
        path: path.into(),
        body,
        headers,
        timeout: Duration::from_secs(5),
        partial,
        method: None,
    })
}

fn write(root: &Path, name: &str, bytes: &[u8]) -> Result<(), String> {
    std::fs::write(root.join(name), bytes).map_err(|_| "logging evidence write failed".into())
}
