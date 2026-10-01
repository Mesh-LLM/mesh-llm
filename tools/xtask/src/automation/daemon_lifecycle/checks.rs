use crate::{
    automation::logging_console::http::{Request, Response, transfer},
    process::Cancellation,
};
use serde::Deserialize;
use std::{
    path::Path,
    time::{Duration, Instant},
};

pub(super) enum Check {
    Ready(u16),
    Models(u16),
    Intents(u16),
    Activity(u16),
}
pub(super) fn execute(
    check: Check,
    root: &Path,
    wait: Duration,
    cancel: &Cancellation,
) -> Result<(), String> {
    match check {
        Check::Ready(port) => {
            let until = Instant::now() + wait;
            loop {
                if cancel.is_cancelled() {
                    return Err("daemon check cancelled".into());
                }
                if let Ok(response) = request(port, "/api/status", None, Vec::new())
                    && response.status < 400
                {
                    std::fs::write(root.join(format!("status-{port}.json")), response.body)
                        .map_err(|_| "status evidence")?;
                    return Ok(());
                }
                if Instant::now() >= until {
                    return Err("daemon readiness deadline".into());
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        }
        Check::Models(port) => {
            let response = request(port, "/v1/models", None, Vec::new())?;
            if response.status >= 400 {
                return Err("zero-model listing failed".into());
            }
            std::fs::write(root.join("zero-model-list.json"), response.body)
                .map_err(|_| "model evidence")?;
            Ok(())
        }
        Check::Intents(port) => intents(port, root, wait, cancel),
        Check::Activity(port) => activity(port, root),
    }
}
fn request(port: u16, path: &str, method: Option<&str>, body: Vec<u8>) -> Result<Response, String> {
    transfer(Request {
        port,
        path: path.into(),
        body,
        method: method.map(str::to_owned),
        headers: Vec::new(),
        timeout: Duration::from_secs(5),
        partial: false,
    })
}
#[derive(Deserialize)]
struct Bootstrap {
    endpoint: Option<String>,
}
#[derive(Deserialize)]
struct Accepted {
    accepted: bool,
    intent_id: String,
    accepted_state: String,
    model: String,
    instance_id: Option<String>,
}
#[derive(Deserialize)]
struct Intents {
    intents: Vec<Intent>,
}
#[derive(Deserialize)]
struct Intent {
    intent_id: String,
    model_ref: String,
    source: String,
    desired_state: String,
}
const MODEL: &str = "qa.invalid/model@main:missing.gguf";
fn intents(port: u16, root: &Path, wait: Duration, cancel: &Cancellation) -> Result<(), String> {
    let response = request(port, "/api/runtime/control-bootstrap", None, Vec::new())?;
    let bootstrap: Bootstrap =
        serde_json::from_slice(&response.body).map_err(|_| "control bootstrap malformed")?;
    let endpoint = bootstrap
        .endpoint
        .filter(|endpoint| !endpoint.is_empty())
        .ok_or("owner-control prerequisite unavailable")?;
    for (operation, expected) in [
        ("load-model", "present"),
        ("unload-model", "absent"),
        ("ensure-model", "present"),
        ("drain-model", "draining"),
    ] {
        let body = serde_json::to_vec(&serde_json::json!({"endpoint":endpoint,"model":MODEL}))
            .map_err(|_| "control encoding")?;
        let response = request(
            port,
            &format!("/api/runtime/control/{operation}"),
            Some("POST"),
            body,
        )?;
        std::fs::write(
            root.join(format!("{operation}-response.json")),
            &response.body,
        )
        .map_err(|_| "control evidence")?;
        let accepted: Accepted =
            serde_json::from_slice(&response.body).map_err(|_| "control acceptance malformed")?;
        if response.status != 200
            || !accepted.accepted
            || accepted.intent_id.is_empty()
            || accepted.accepted_state != expected
            || accepted.model != MODEL
            || accepted.instance_id.is_some()
        {
            return Err("control acceptance identity mismatch".into());
        }
        let until = Instant::now() + wait;
        loop {
            if cancel.is_cancelled() {
                return Err("intent check cancelled".into());
            }
            let response = request(port, "/api/runtime/intents", None, Vec::new())?;
            let intents: Intents =
                serde_json::from_slice(&response.body).map_err(|_| "intents malformed")?;
            if intents.intents.iter().any(|intent| {
                intent.intent_id == accepted.intent_id
                    && intent.model_ref == MODEL
                    && intent.source == "owner_lifecycle"
                    && intent.desired_state == expected
            }) {
                std::fs::write(
                    root.join(format!("{operation}-intents.json")),
                    response.body,
                )
                .map_err(|_| "intent evidence")?;
                break;
            }
            if Instant::now() >= until {
                return Err("authoritative intent deadline".into());
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    }
    Ok(())
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Activity {
    effective_state: Effective,
    override_mode: Override,
    detector_category: Detector,
}
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Effective {
    Accepting,
    AcceptingDeprioritized,
    RemotePaused,
    AllPaused,
}
#[derive(Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
enum Override {
    Auto,
    Active,
    Idle,
}
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Detector {
    Active,
    Idle,
    Unavailable,
}
fn activity(port: u16, root: &Path) -> Result<(), String> {
    for (method, body, expected) in [
        ("PUT", b"\"active\"".to_vec(), Override::Active),
        ("DELETE", Vec::new(), Override::Auto),
    ] {
        let response = request(port, "/api/runtime/activity/override", Some(method), body)?;
        let activity: Activity =
            serde_json::from_slice(&response.body).map_err(|_| "activity override malformed")?;
        if response.status != 200 || activity.override_mode != expected {
            return Err("activity override mismatch".into());
        }
    }
    let response = request(port, "/api/runtime/activity", None, Vec::new())?;
    let activity: Activity =
        serde_json::from_slice(&response.body).map_err(|_| "activity privacy shape invalid")?;
    let Activity {
        effective_state,
        override_mode,
        detector_category,
    } = activity;
    match effective_state {
        Effective::Accepting
        | Effective::AcceptingDeprioritized
        | Effective::RemotePaused
        | Effective::AllPaused => (),
    }
    match override_mode {
        Override::Auto | Override::Active | Override::Idle => (),
    }
    match detector_category {
        Detector::Active | Detector::Idle | Detector::Unavailable => (),
    }
    std::fs::write(root.join("activity.json"), response.body).map_err(|_| "activity evidence")?;
    Ok(())
}
