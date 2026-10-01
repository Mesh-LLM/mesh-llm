use super::{options::Options, projection};
use crate::automation::logging_console::http::{Request, transfer};
use crate::process::{
    Cancellation,
    retained::{
        MemberId,
        recovery::{Observation, Recovery},
    },
};
use std::time::Duration;

pub(super) enum RequestKind {
    Invite,
    Startup,
    Chat(usize),
    Select(usize),
    Recovery {
        killed: String,
        old_run: String,
        stopped: usize,
    },
}
pub(super) enum Facts {
    Pending,
    Invite(String),
    Startup(usize),
    Chat,
    Selected {
        member: MemberId,
        index: usize,
        killed: String,
        old_run: String,
    },
    Recovery(Vec<Recovery>),
}

pub(super) fn work(
    kind: RequestKind,
    options: &Options,
    cancellation: &Cancellation,
) -> Result<Facts, String> {
    if cancellation.is_cancelled() {
        return Err("split HTTP work cancelled".into());
    }
    match kind {
        RequestKind::Invite => {
            let status =
                fetch::<projection::Status>(options.console, "/api/status").unwrap_or_default();
            if status.token.is_empty() {
                Ok(Facts::Pending)
            } else {
                Ok(Facts::Invite(status.token))
            }
        }
        RequestKind::Startup => {
            let seed =
                fetch::<projection::Status>(options.console, "/api/status").unwrap_or_default();
            if seed.peers.len() < options.workers {
                return Ok(Facts::Pending);
            }
            for index in 0..=options.workers {
                if cancellation.is_cancelled() {
                    return Err("split cancelled".into());
                }
                if let Ok(observation) = observe(options, index)
                    && observation.startup_ready()
                {
                    return Ok(Facts::Startup(index));
                }
            }
            Ok(Facts::Pending)
        }
        RequestKind::Chat(index) => {
            let models = fetch::<projection::Models>(port(options.api, index)?, "/v1/models")?;
            let model = models.data.first().ok_or("chat model missing")?;
            let body=serde_json::to_vec(&serde_json::json!({"model":model.id,"messages":[{"role":"user","content":"Say hello."}],"max_tokens":8,"temperature":0})).map_err(|_|"chat encoding")?;
            let response = transfer(Request {
                port: port(options.api, index)?,
                path: "/v1/chat/completions".into(),
                body,
                headers: Vec::new(),
                timeout: Duration::from_secs(120),
                partial: false,
                method: None,
            })?;
            #[derive(serde::Deserialize)]
            struct Chat {
                object: String,
                choices: Vec<serde_json::Value>,
            }
            let chat: Chat =
                serde_json::from_slice(&response.body).map_err(|_| "chat response malformed")?;
            if response.status >= 400
                || !crate::process::retained::recovery::chat_valid(&chat.object, chat.choices.len())
            {
                return Err("split chat failed".into());
            }
            Ok(Facts::Chat)
        }
        RequestKind::Select(driver) => {
            let mut workers = Vec::new();
            for index in 1..=options.workers {
                let status =
                    fetch::<projection::Status>(port(options.console, index)?, "/api/status")?;
                if status.node_id.is_empty() {
                    return Err("worker identity missing".into());
                }
                workers.push((
                    status.node_id,
                    options.member(index).map_err(|error| error.to_string())?,
                ));
            }
            let observation = observe(options, driver)?;
            let member = observation
                .downstream_worker(
                    &workers,
                    options.member(driver).map_err(|error| error.to_string())?,
                )
                .ok_or("downstream worker missing")?;
            let (index, (killed, _)) = workers
                .into_iter()
                .enumerate()
                .find(|(_, (_, id))| *id == member)
                .ok_or("selected identity missing")?;
            Ok(Facts::Selected {
                member,
                index: index + 1,
                killed,
                old_run: observation
                    .topologies
                    .first()
                    .map(|topology| topology.run_id.clone())
                    .unwrap_or_default(),
            })
        }
        RequestKind::Recovery {
            killed,
            old_run,
            stopped,
        } => {
            let mut recovery = Vec::new();
            for index in 0..=options.workers {
                if index == stopped {
                    continue;
                }
                if cancellation.is_cancelled() {
                    return Err("split cancelled".into());
                }
                recovery.push(match observe(options, index) {
                    Ok(observation) => observation.recovery(&killed, &old_run),
                    Err(_) => Recovery::Pending,
                });
            }
            Ok(Facts::Recovery(recovery))
        }
    }
}

fn observe(options: &Options, index: usize) -> Result<Observation, String> {
    let console = port(options.console, index)?;
    let api = port(options.api, index)?;
    Ok(projection::observation(
        fetch(console, "/api/runtime/stages")?,
        fetch(api, "/v1/models")?,
    ))
}
fn port(base: u16, index: usize) -> Result<u16, String> {
    base.checked_add(u16::try_from(index).map_err(|_| "port index")?)
        .ok_or_else(|| "port overflow".into())
}
fn fetch<T: serde::de::DeserializeOwned>(port: u16, path: &str) -> Result<T, String> {
    let response = transfer(Request {
        port,
        path: path.into(),
        body: Vec::new(),
        headers: Vec::new(),
        timeout: Duration::from_secs(5),
        partial: false,
        method: None,
    })?;
    if response.status >= 400 {
        return Err("split HTTP status".into());
    }
    serde_json::from_slice(&response.body).map_err(|_| "split HTTP response malformed".into())
}
