use crate::automation::logging_console::{
    checks::{self, CheckContext},
    http::{Request, Response, transfer},
};
use crate::process::Cancellation;
use serde::Deserialize;
use std::{path::Path, time::Duration};

pub(super) const MARKER: &str = "QA_PRIVATE_MARKER_NOT_FOR_PERSISTENCE";
pub(super) enum Check {
    Ready(u16),
    Persist(u16),
    Detail { base: u16, id: String },
    Access(u16),
    Delete { base: u16, id: String },
    Replay(u16),
    FailOpen { base: u16, model: Option<String> },
}
pub(super) struct Context<'a> {
    pub root: &'a Path,
    pub wait: Duration,
    pub cancellation: &'a Cancellation,
}

pub(super) fn execute(check: Check, context: Context<'_>) -> Result<String, String> {
    let shared = |check, base| {
        checks::execute(
            check,
            CheckContext {
                base,
                root: context.root,
                wait: context.wait,
                cancellation: context.cancellation,
            },
        )
    };
    match check {
        Check::Ready(base) => shared(checks::Check::Ready, base),
        Check::Persist(base) => {
            let body=serde_json::to_vec(&serde_json::json!({"model":"qa-no-model","messages":[{"role":"user","content":MARKER}]})).map_err(|_|"private request encoding")?;
            request(base, "/v1/chat/completions", body, None)?;
            let until = std::time::Instant::now() + context.wait;
            #[derive(Deserialize)]
            struct Page {
                items: Vec<Row>,
            }
            #[derive(Deserialize)]
            struct Row {
                #[serde(rename = "requestId")]
                id: String,
            }
            loop {
                if context.cancellation.is_cancelled() {
                    return Err("logging recovery cancelled".into());
                }
                if let Ok(response) =
                    request(base + 1, "/api/logs/requests?limit=10", Vec::new(), None)
                    && let Ok(page) = serde_json::from_slice::<Page>(&response.body)
                    && let Some(row) = page
                        .items
                        .into_iter()
                        .next()
                        .filter(|row| !row.id.is_empty())
                {
                    private(&response.body)?;
                    write(context.root, "list-before-restart.json", &response.body)?;
                    return Ok(row.id);
                }
                if std::time::Instant::now() >= until {
                    return Err("durable logging request prerequisite missing".into());
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        }
        Check::Detail { base, id } => {
            let response = request(
                base + 1,
                &format!("/api/logs/requests/{}", identity(&id)?),
                Vec::new(),
                None,
            )?;
            if response.status >= 400 {
                return Err("durable request missing after restart".into());
            }
            private(&response.body)?;
            write(context.root, "detail-after-restart.json", &response.body)?;
            Ok(id)
        }
        Check::Access(base) => {
            for (name, headers) in [
                (
                    "trusted-host",
                    vec![("host".into(), "attacker.example".into())],
                ),
                (
                    "trusted-origin",
                    vec![
                        ("host".into(), format!("localhost:{}", base + 1)),
                        ("origin".into(), "https://attacker.example".into()),
                    ],
                ),
            ] {
                let response = transfer(Request {
                    port: base + 1,
                    path: "/api/logs/events?channel=requests".into(),
                    body: Vec::new(),
                    headers,
                    timeout: Duration::from_secs(5),
                    partial: false,
                    method: None,
                })?;
                #[derive(Deserialize)]
                struct ErrorBody {
                    error: Option<ErrorCode>,
                    code: Option<String>,
                }
                #[derive(Deserialize)]
                struct ErrorCode {
                    code: String,
                }
                let body: ErrorBody = serde_json::from_slice(&response.body)
                    .map_err(|_| "forbidden response malformed")?;
                let code = body.error.map(|error| error.code).or(body.code);
                if response.status != 403 || code.as_deref() != Some("forbidden") {
                    return Err("trusted-local response not typed forbidden".into());
                }
                write(context.root, &format!("{name}.json"), &response.body)?;
            }
            Ok(String::new())
        }
        Check::Delete { base, id } => {
            let mut random = [0; 16];
            getrandom::fill(&mut random).map_err(|_| "delete operation entropy")?;
            let operation = uuid::Uuid::from_bytes(random).to_string();
            let body = serde_json::to_vec(
                &serde_json::json!({"operationId":operation,"reason":"qa retention cascade"}),
            )
            .map_err(|_| "delete encoding")?;
            let response = request(
                base + 1,
                &format!("/api/logs/requests/{}/delete", identity(&id)?),
                body,
                Some("POST"),
            )?;
            #[derive(Deserialize)]
            struct Receipt {
                #[serde(rename = "operationId")]
                operation_id: String,
                #[serde(rename = "requestId")]
                request_id: String,
            }
            let receipt: Receipt =
                serde_json::from_slice(&response.body).map_err(|_| "delete receipt malformed")?;
            if response.status >= 400
                || receipt.operation_id != operation
                || receipt.request_id != id
            {
                return Err("delete receipt identity mismatch".into());
            }
            write(context.root, "delete-receipt.json", &response.body)?;
            let detail = request(
                base + 1,
                &format!("/api/logs/requests/{}", identity(&id)?),
                Vec::new(),
                None,
            )?;
            if detail.status != 404 {
                return Err("deleted request remains queryable".into());
            }
            Ok(String::new())
        }
        Check::Replay(base) => {
            shared(checks::Check::Access, base)?;
            shared(checks::Check::Replay, base)?;
            let bytes = std::fs::read(context.root.join("sse-replay.body"))
                .map_err(|_| "SSE evidence missing")?;
            private(&bytes)?;
            if bytes
                .windows(b"private/operator".len())
                .any(|bytes| bytes == b"private/operator")
                || bytes.windows(6).any(|bytes| bytes == b"token=")
            {
                return Err("SSE privacy leakage".into());
            }
            Ok(String::new())
        }
        Check::FailOpen { base, model } => {
            let response = request(base + 1, "/api/logs/requests", Vec::new(), None)?;
            write(context.root, "fail-open-logs.json", &response.body)?;
            if response.status != 503 {
                return Err("failed logging root did not return unavailable".into());
            }
            if let Some(model) = model {
                let body=serde_json::to_vec(&serde_json::json!({"model":model,"messages":[{"role":"user","content":"return deterministic QA output"}]})).map_err(|_|"inference encoding")?;
                let response = request(base, "/v1/chat/completions", body, None)?;
                write(context.root, "fail-open-inference.json", &response.body)?;
                if !(200..300).contains(&response.status) {
                    return Err("fail-open inference failed".into());
                }
            }
            Ok(String::new())
        }
    }
}
fn identity(id: &str) -> Result<&str, String> {
    if id.is_empty()
        || !id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        Err("request identity invalid".into())
    } else {
        Ok(id)
    }
}
fn private(bytes: &[u8]) -> Result<(), String> {
    if bytes
        .windows(MARKER.len())
        .any(|bytes| bytes == MARKER.as_bytes())
    {
        Err("private request marker leaked".into())
    } else {
        Ok(())
    }
}
fn request(port: u16, path: &str, body: Vec<u8>, method: Option<&str>) -> Result<Response, String> {
    transfer(Request {
        port,
        path: path.into(),
        body,
        headers: Vec::new(),
        timeout: Duration::from_secs(5),
        partial: false,
        method: method.map(str::to_owned),
    })
}
fn write(root: &Path, name: &str, bytes: &[u8]) -> Result<(), String> {
    std::fs::write(root.join(name), bytes).map_err(|_| "logging recovery evidence write".into())
}
