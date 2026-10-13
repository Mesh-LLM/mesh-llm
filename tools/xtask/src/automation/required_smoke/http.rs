use super::{Check, Response, Transfer, evidence::RESPONSE_LIMIT};
use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper_util::rt::TokioIo;
use std::{net::Ipv4Addr, time::Instant};

pub(super) struct Request {
    pub check: Check,
    pub port: u16,
    pub model: String,
    pub deadline: Instant,
}

pub(super) enum ResultBody {
    Complete(u16, Vec<u8>),
    Failed,
    TimedOut,
    Oversized,
}
impl ResultBody {
    pub fn transfer(&self) -> Transfer<'_> {
        match self {
            Self::Complete(status, body) => Transfer::Complete(Response {
                status: *status,
                body,
            }),
            Self::Failed => Transfer::Failed,
            Self::TimedOut => Transfer::TimedOut,
            Self::Oversized => Transfer::Oversized,
        }
    }
}

pub(super) async fn transfer(request: Request) -> ResultBody {
    match tokio::time::timeout_at(request.deadline.into(), exchange(&request)).await {
        Ok(result) if Instant::now() < request.deadline => result,
        Ok(_) | Err(_) => ResultBody::TimedOut,
    }
}

async fn exchange(request: &Request) -> ResultBody {
    let result: Result<ResultBody, Box<dyn std::error::Error>> = async {
        let stream = tokio::net::TcpStream::connect((Ipv4Addr::LOCALHOST, request.port)).await?;
        let (mut sender, connection) = hyper::client::conn::http1::Builder::new()
            .max_buf_size(16_384)
            .max_headers(4096)
            .handshake::<_, Full<Bytes>>(TokioIo::new(stream))
            .await?;
        let (path, body) = payload(request)?;
        let message = hyper::Request::builder()
            .uri(path)
            .method(if body.is_empty() { "GET" } else { "POST" })
            .header("host", format!("127.0.0.1:{}", request.port))
            .header("connection", "close")
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(body)))?;
        let response = async {
            let mut response = sender.send_request(message).await?;
            let status = response.status().as_u16();
            let mut bytes = Vec::new();
            while let Some(frame) = response.body_mut().frame().await {
                if let Some(data) = frame?.data_ref() {
                    if data.len() > RESPONSE_LIMIT - bytes.len() {
                        return Ok::<_, Box<dyn std::error::Error>>(ResultBody::Oversized);
                    }
                    bytes.extend_from_slice(data);
                }
            }
            Ok(ResultBody::Complete(status, bytes))
        };
        tokio::pin!(response);
        tokio::select! {
            biased;
            result = &mut response => result,
            result = connection => { result?; response.await }
        }
    }
    .await;
    result.unwrap_or(ResultBody::Failed)
}

fn payload(request: &Request) -> Result<(&'static str, Vec<u8>), serde_json::Error> {
    let (prompt, model, stream) = match request.check {
        Check::Runtime
        | Check::RuntimeAttestation
        | Check::HeadlessStatus
        | Check::HeadlessAttestation => return Ok(("/api/status", Vec::new())),
        Check::Models | Check::HeadlessModels => return Ok(("/v1/models", Vec::new())),
        Check::Chat => (
            "Say hello in exactly 3 words.",
            request.model.as_str(),
            false,
        ),
        Check::Stream => ("Count from one to three.", request.model.as_str(), true),
        Check::Auto => ("Say hi.", "auto", false),
        Check::InspectAttestation => return Ok(("/api/status", Vec::new())),
    };
    #[derive(serde::Serialize)]
    struct Message<'a> {
        role: &'a str,
        content: &'a str,
    }
    #[derive(serde::Serialize)]
    struct Usage {
        include_usage: bool,
    }
    #[derive(serde::Serialize)]
    struct Chat<'a> {
        model: &'a str,
        messages: [Message<'a>; 1],
        max_tokens: u32,
        temperature: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        stream: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        stream_options: Option<Usage>,
    }
    serde_json::to_vec(&Chat {
        model,
        messages: [Message {
            role: "user",
            content: prompt,
        }],
        max_tokens: 4,
        temperature: 0,
        stream: stream.then_some(true),
        stream_options: stream.then_some(Usage {
            include_usage: true,
        }),
    })
    .map(|body| ("/v1/chat/completions", body))
}
