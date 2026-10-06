//! Bounded first-advertised model discovery for optional external benchmark callers.
use super::{command_interrupt::Interrupt, guardrail_corpus::transport};
use crate::{command::DynResult, process::Cancellation};
use std::time::{Duration, Instant};
const USAGE: &str = "automation endpoint-model-discovery --base-url HTTP_OR_HTTPS_V1 [--timeout-secs 1..60]; API_KEY is read privately from the environment, default EMPTY";
pub(super) fn endpoint(base: &str) -> DynResult<String> {
    let url = url::Url::parse(base).map_err(|_| "invalid model discovery endpoint")?;
    if !matches!(url.scheme(), "http" | "https")
        || url.host().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || url.path().trim_end_matches('/') != "/v1"
    {
        return Err(
            "model discovery requires credential-free HTTP(S) /v1 without query or fragment".into(),
        );
    }
    // Emit the admitted URL parser's canonical host so dispatch and transport
    // classify the same bytes, including shortened/numeric IPv4 spellings.
    Ok(url.as_str().trim_end_matches('/').to_owned())
}
fn authorization(token: &str) -> DynResult<()> {
    if token.len() > 4096 || token.chars().any(char::is_control) {
        return Err("model discovery authorization must be bounded single-line text".into());
    }
    Ok(())
}
fn first(body: &[u8], status: u16) -> DynResult<String> {
    if status != 200 {
        return Err("model discovery requires HTTP 200".into());
    }
    let value: serde_json::Value =
        serde_json::from_slice(body).map_err(|_| "model discovery response is not JSON")?;
    if value.get("error").is_some_and(|v| !v.is_null()) {
        return Err("model discovery server error".into());
    }
    let id = value["data"]
        .as_array()
        .and_then(|rows| rows.first())
        .and_then(|row| row["id"].as_str())
        .ok_or("no first model ID returned from /v1/models")?;
    if id.is_empty() || id.len() > 4096 || id.chars().any(char::is_control) {
        return Err("first model ID must be nonempty bounded single-line text".into());
    }
    Ok(id.to_owned())
}
async fn cancelled(token: &Cancellation) {
    while !token.is_cancelled() {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}
pub(super) async fn discover(
    base: &str,
    key: &str,
    deadline: Instant,
    token: &Cancellation,
) -> DynResult<String> {
    if token.is_cancelled() || Instant::now() >= deadline {
        return Err("model discovery cancelled or deadline before request".into());
    }
    let url = url::Url::parse(base)?;
    let curl = url.scheme() == "https" || !matches!(url.host(), Some(url::Host::Ipv4(_)));
    let future = transport::exchange_owned_with_private_authorization(
        base,
        "models",
        None,
        deadline,
        token,
        (!key.is_empty()).then_some(key),
    );
    let response = if curl {
        // Await the supervised child including its reserved cleanup; never abandon spawn_blocking.
        future
            .await
            .map_err(|_| "model discovery bounded curl transfer refused")?
    } else {
        tokio::select! {
            biased;
            _=cancelled(token)=>return Err("model discovery cancelled".into()),
            result=tokio::time::timeout_at(deadline.into(),future)=>
                result.map_err(|_| "model discovery request deadline")?
                    .map_err(|_| "model discovery HTTP transfer refused")?,
        }
    };
    if token.is_cancelled() || Instant::now() >= deadline {
        return Err("model discovery cancelled or deadline after request".into());
    }
    first(&response.body, response.status)
}
fn options(args: &[String]) -> DynResult<(String, u64)> {
    let (base, seconds) = match args {
        [flag, base] if flag == "--base-url" => (base, 10),
        [flag, base, time, seconds] if flag == "--base-url" && time == "--timeout-secs" => (
            base,
            seconds
                .parse::<u64>()
                .map_err(|_| "invalid discovery timeout")?,
        ),
        _ => return Err(USAGE.into()),
    };
    if !(1..=60).contains(&seconds) {
        return Err("discovery timeout must be 1..60 seconds".into());
    }
    Ok((endpoint(base)?, seconds))
}
pub(crate) fn run(args: &[String]) -> DynResult<()> {
    if args == ["--help"] {
        println!("{USAGE}");
        return Ok(());
    }
    let (base, seconds) = options(args)?;
    let key = match std::env::var("API_KEY") {
        Ok(value) => value,
        Err(std::env::VarError::NotPresent) => "EMPTY".to_owned(),
        Err(_) => return Err("model discovery authorization must be UTF-8".into()),
    };
    authorization(&key)?;
    let interrupt = Interrupt::install()?;
    let token = interrupt.cancellation();
    let deadline = Instant::now() + Duration::from_secs(seconds);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let result = runtime.block_on(discover(&base, &key, deadline, &token));
    let finish = interrupt.finish();
    finish?;
    if token.is_cancelled() || Instant::now() >= deadline {
        return Err("model discovery terminal cancellation/deadline".into());
    }
    println!("{}", result?);
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn discovery_first_row_and_exact_status_refuse_missing_typed_or_unsafe_identity() {
        assert_eq!(
            first(br#"{"data":[{"id":"first"},{"id":"second"}]}"#, 200).unwrap(),
            "first"
        );
        for status in [201, 204, 302, 401, 503] {
            assert!(first(br#"{"data":[{"id":"first"}]}"#, status).is_err());
        }
        for bytes in [
            b"{}".as_slice(),
            b"{\"data\":[]}",
            b"{\"data\":[{\"id\":1}]}",
            b"{\"data\":[{\"id\":\"\"}]}",
            b"{\"data\":[{\"id\":\"a\\nb\"}]}",
            b"not-json",
        ] {
            assert!(first(bytes, 200).is_err());
        }
        assert!(
            first(
                &serde_json::to_vec(&serde_json::json!({"data":[{"id":"x".repeat(4097)}]}))
                    .unwrap(),
                200
            )
            .is_err()
        );
    }
    #[test]
    fn discovery_admission_refuses_url_credentials_and_header_injection_before_transport() {
        assert_eq!(
            endpoint("http://127.0.0.1:9337/v1/").unwrap(),
            "http://127.0.0.1:9337/v1"
        );
        for alias in ["http://127.1/v1", "http://2130706433/v1"] {
            assert_eq!(endpoint(alias).unwrap(), "http://127.0.0.1/v1");
        }
        for base in [
            "http://u:p@host/v1",
            "http://host/v1?token=secret",
            "http://host/v1#frag",
            "file:///v1",
            "http://host/",
        ] {
            assert!(endpoint(base).is_err());
        }
        for key in ["secret\r\ninjected", "secret\0", "secret\n"] {
            assert!(authorization(key).is_err());
        }
        assert!(authorization(&"x".repeat(4097)).is_err());
        assert!(authorization("").is_ok());
    }
}
