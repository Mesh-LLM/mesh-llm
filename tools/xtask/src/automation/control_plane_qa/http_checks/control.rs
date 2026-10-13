use super::{Context, Fact, decode, request, save};
use serde::Deserialize;
use std::time::Duration;

#[derive(Deserialize)]
struct Endpoint {
    endpoint: Option<String>,
}

pub(super) fn bootstrap(console: u16) -> Result<Fact, String> {
    #[derive(Deserialize)]
    struct Bootstrap {
        enabled: bool,
        endpoint: Option<String>,
        requires_explicit_remote_endpoint: bool,
    }
    let bootstrap: Bootstrap = decode(&request(
        console,
        "/api/runtime/control-bootstrap",
        Vec::new(),
        Duration::from_secs(5),
    )?)?;
    if !bootstrap.requires_explicit_remote_endpoint {
        return Err("bootstrap omitted explicit remote endpoint requirement".into());
    }
    match bootstrap
        .endpoint
        .filter(|endpoint| bootstrap.enabled && !endpoint.is_empty())
    {
        Some(endpoint) => Ok(Fact::Endpoint(endpoint)),
        None => Ok(Fact::Prerequisite("config-runtime-bootstrap")),
    }
}
pub(super) fn wrong_owner(
    console: u16,
    endpoint: &str,
    context: &Context<'_>,
) -> Result<Fact, String> {
    let response = request(
        console,
        "/api/runtime/control/scan-refresh",
        serde_json::to_vec(&serde_json::json!({"endpoint":endpoint}))
            .map_err(|_| "wrong owner encoding")?,
        Duration::from_secs(30),
    )?;
    #[derive(Deserialize)]
    struct Error {
        code: String,
        message: String,
    }
    #[derive(Deserialize)]
    struct Body {
        error: Error,
    }
    let body: Body = decode(&response)?;
    let error = format!("{} {}", body.error.code, body.error.message).to_ascii_lowercase();
    if !(400..500).contains(&response.status)
        || !["unauthorized", "owner", "handshake"]
            .iter()
            .any(|marker| error.contains(marker))
    {
        return Err("wrong-owner scan was not rejected".into());
    }
    save(
        context.directory,
        "wrong-owner-response.json",
        &response.body,
    )?;
    Ok(Fact::Done)
}
pub(super) fn lifecycle(console: u16, context: &Context<'_>) -> Result<Fact, String> {
    let bootstrap: Endpoint = decode(&request(
        console,
        "/api/runtime/control-bootstrap",
        Vec::new(),
        Duration::from_secs(5),
    )?)?;
    let Some(endpoint) = bootstrap.endpoint.filter(|endpoint| !endpoint.is_empty()) else {
        return Ok(Fact::Prerequisite("lifecycle-models"));
    };
    for operation in ["load", "unload", "ensure", "drain"] {
        if context.cancel.is_cancelled() {
            return Err("mixed-version check cancelled".into());
        }
        let body = serde_json::to_vec(
            &serde_json::json!({"endpoint":endpoint,"model":"qa.invalid/model@main:missing.gguf"}),
        )
        .map_err(|_| "lifecycle encoding")?;
        let response = request(
            console,
            &format!("/api/runtime/control/{operation}-model"),
            body,
            Duration::from_secs(10),
        )?;
        #[derive(Deserialize)]
        struct Accepted {
            accepted: bool,
            model: String,
            instance_id: Option<String>,
        }
        let accepted: Accepted = decode(&response)?;
        if response.status != 200
            || !accepted.accepted
            || accepted.model != "qa.invalid/model@main:missing.gguf"
            || accepted.instance_id.is_some()
        {
            return Err("lifecycle acceptance mismatch".into());
        }
        save(
            context.directory,
            &format!("lifecycle-{operation}.json"),
            &response.body,
        )?;
    }
    Ok(Fact::Done)
}
pub(super) fn legacy(current: u16, released: u16, context: &Context<'_>) -> Result<Fact, String> {
    let response = match request(
        released,
        "/api/runtime/control-bootstrap",
        Vec::new(),
        Duration::from_secs(5),
    ) {
        Ok(response) if response.status < 400 => response,
        _ => return Ok(Fact::Prerequisite("lifecycle-legacy-unsupported")),
    };
    let endpoint: Endpoint = decode(&response)?;
    let Some(endpoint) = endpoint.endpoint.filter(|endpoint| !endpoint.is_empty()) else {
        return Ok(Fact::Prerequisite("lifecycle-legacy-unsupported"));
    };
    let body = serde_json::to_vec(
        &serde_json::json!({"endpoint":endpoint,"model":"qa.invalid/model@main:missing.gguf"}),
    )
    .map_err(|_| "legacy request")?;
    let response = request(
        current,
        "/api/runtime/control/load-model",
        body,
        Duration::from_secs(30),
    )?;
    #[derive(Deserialize)]
    struct Error {
        code: String,
    }
    #[derive(Deserialize)]
    struct Body {
        error: Error,
    }
    let body: Body = decode(&response)?;
    if response.status != 503 || body.error.code != "control_unsupported" {
        return Err("legacy rejection was not typed unsupported".into());
    }
    save(context.directory, "legacy-unsupported.json", &response.body)?;
    Ok(Fact::Done)
}
