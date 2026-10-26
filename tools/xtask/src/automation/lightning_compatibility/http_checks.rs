//! Local bounded HTTP observations; invites never leave this scoped worker/owner.
use crate::automation::logging_console::http::{Request, transfer_cancelled};
use crate::process::Cancellation;
use serde_json::{Value, json};
use std::time::{Duration, Instant};
pub(super) const MODEL: &str = "Payment-Compatibility-Smoke";
pub(super) enum Check {
    Status {
        port: u16,
        invite: bool,
    },
    Models(u16),
    Inference {
        port: u16,
        expected: u16,
        name: &'static str,
    },
    Pricing(u16),
}
pub(super) enum Fact {
    Ready,
    Invite(String),
    Case(Value),
}
fn request(
    port: u16,
    path: &str,
    body: Option<Value>,
    deadline: Instant,
    cancel: &Cancellation,
    timeout: Duration,
) -> Result<(u16, Value), String> {
    if cancel.is_cancelled()
        || deadline.saturating_duration_since(Instant::now()) <= Duration::from_secs(1)
    {
        return Err("compatibility cancelled/deadline".into());
    }
    let response = transfer_cancelled(
        Request {
            port,
            path: path.into(),
            body: body
                .map(|v| serde_json::to_vec(&v).unwrap())
                .unwrap_or_default(),
            headers: vec![],
            timeout,
            partial: false,
            method: None,
        },
        cancel,
        deadline,
    )?;
    let value =
        serde_json::from_slice(&response.body).map_err(|_| "compatibility malformed JSON")?;
    Ok((response.status, value))
}
fn wait(
    port: u16,
    path: &str,
    deadline: Instant,
    cancel: &Cancellation,
    accept: impl Fn(&Value) -> bool,
) -> Result<Value, String> {
    let until = deadline.min(Instant::now() + Duration::from_secs(90));
    loop {
        if cancel.is_cancelled() || Instant::now() + Duration::from_secs(1) >= until {
            return Err("compatibility readiness cancelled/deadline".into());
        }
        if let Ok((200, value)) =
            request(port, path, None, until, cancel, Duration::from_millis(250))
            && accept(&value)
        {
            return Ok(value);
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}
pub(super) fn execute(
    check: Check,
    deadline: Instant,
    cancel: &Cancellation,
) -> Result<Fact, String> {
    match check {
        Check::Status { port, invite } => {
            let value = wait(port, "/api/status", deadline, cancel, Value::is_object)?;
            if !invite {
                return Ok(Fact::Ready);
            }
            let token = value["token"]
                .as_str()
                .filter(|s| !s.is_empty() && s.len() <= 16384 && !s.contains(['\0', '\r', '\n']))
                .ok_or("compatibility invite absent/invalid")?;
            Ok(Fact::Invite(token.to_owned()))
        }
        Check::Models(port) => {
            wait(port, "/v1/models", deadline, cancel, |v| {
                v["data"]
                    .as_array()
                    .is_some_and(|a| a.iter().any(|m| m["id"] == MODEL))
            })?;
            Ok(Fact::Ready)
        }
        Check::Pricing(port) => {
            let (status, _) = request(
                port,
                "/api/wallet",
                Some(
                    json!({"command":"set_pricing","model":MODEL,"value":{"input_msat_per_million":10000000,"output_msat_per_million":30000000}}),
                ),
                deadline,
                cancel,
                Duration::from_secs(30),
            )?;
            if status != 200 {
                return Err("compatibility pricing status refused".into());
            }
            Ok(Fact::Ready)
        }
        Check::Inference {
            port,
            expected,
            name,
        } => {
            let (status, value) = request(
                port,
                "/v1/chat/completions",
                Some(
                    json!({"model":MODEL,"messages":[{"role":"user","content":"Say hello."}],"max_tokens":8,"temperature":0,"stream":false}),
                ),
                deadline,
                cancel,
                Duration::from_secs(30),
            )?;
            Ok(Fact::Case(observation(name, expected, status, &value)))
        }
    }
}
fn observation(name: &str, expected: u16, status: u16, value: &Value) -> Value {
    let content = value["choices"][0]["message"]["content"].as_str();
    let completion = value["usage"]["completion_tokens"].as_u64();
    let passed = status == expected
        && (expected != 200
            || (content.is_some_and(|s| !s.is_empty()) && completion.is_some_and(|n| n > 0)));
    json!({"case":name,"expected_status":expected,"status":status,"passed":passed,"content_present":content.is_some_and(|s|!s.is_empty()),"completion_tokens":completion,"response_sha256":hex::encode(sha2::Sha256::digest(serde_json::to_vec(value).unwrap())),"response_recording":"typed_observation_no_raw_response"})
}
use sha2::Digest as _;
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn free_cases_require_content_and_usage_paid_case_requires_402() {
        assert_eq!(
            observation(
                "free",
                200,
                200,
                &json!({"choices":[{"message":{"content":"hello"}}],"usage":{"completion_tokens":1}})
            )["passed"],
            true
        );
        for value in [
            json!({}),
            json!({"choices":[{"message":{"content":""}}],"usage":{"completion_tokens":1}}),
            json!({"choices":[{"message":{"content":"hello"}}],"usage":{"completion_tokens":0}}),
        ] {
            assert_eq!(observation("free", 200, 200, &value)["passed"], false);
        }
        assert_eq!(
            observation("paid", 402, 402, &json!({"error":{"message":"policy"}}))["passed"],
            true
        );
        assert_eq!(observation("paid", 402, 200, &json!({}))["passed"], false);
    }
}
