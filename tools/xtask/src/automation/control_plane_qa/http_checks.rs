mod control;
mod mesh;
mod validation;
use super::options::Options;
use crate::{
    automation::logging_console::http::{Request, Response, transfer},
    process::Cancellation,
};
use std::{
    path::{Path, PathBuf},
    time::Duration,
};

pub(super) enum Check {
    Ready {
        console: u16,
    },
    Invite {
        console: u16,
    },
    Pair {
        server: u16,
        client: u16,
        api: u16,
        chat: bool,
    },
    Public {
        console: u16,
        api: u16,
    },
    Bootstrap {
        console: u16,
    },
    WrongOwner {
        console: u16,
        endpoint: String,
    },
    Lifecycle {
        console: u16,
    },
    Legacy {
        current: u16,
        released: u16,
    },
    ValidateScan(PathBuf),
    Capabilities {
        current: PathBuf,
        released: PathBuf,
    },
}
pub(super) enum Fact {
    Done,
    Token(String),
    Endpoint(String),
    Prerequisite(&'static str),
    Capabilities { current: String, released: String },
}
pub(super) struct Context<'a> {
    pub options: &'a Options,
    pub directory: &'a Path,
    pub cancel: &'a Cancellation,
}

pub(super) fn execute(check: Check, context: Context<'_>) -> Result<Fact, String> {
    match check {
        Check::Ready { console } => mesh::ready(console, &context),
        Check::Invite { console } => mesh::invite(console, &context),
        Check::Pair {
            server,
            client,
            api,
            chat,
        } => mesh::pair((server, client, api), chat, &context),
        Check::Public { console, api } => mesh::public(console, api, &context),
        Check::Bootstrap { console } => control::bootstrap(console),
        Check::WrongOwner { console, endpoint } => {
            control::wrong_owner(console, &endpoint, &context)
        }
        Check::Lifecycle { console } => control::lifecycle(console, &context),
        Check::Legacy { current, released } => control::legacy(current, released, &context),
        Check::ValidateScan(path) => {
            let body = std::fs::read(path).map_err(|_| "scan evidence missing")?;
            validation::scan(&Response { status: 200, body })?;
            Ok(Fact::Done)
        }
        Check::Capabilities { current, released } => {
            let current =
                std::fs::read_to_string(current).map_err(|_| "current binary help unavailable")?;
            let released = std::fs::read_to_string(released)
                .map_err(|_| "released binary help unavailable")?;
            Ok(Fact::Capabilities { current, released })
        }
    }
}

fn request(port: u16, path: &str, body: Vec<u8>, timeout: Duration) -> Result<Response, String> {
    transfer(Request {
        port,
        path: path.into(),
        body,
        headers: Vec::new(),
        timeout,
        partial: false,
        method: None,
    })
}
fn decode<T: serde::de::DeserializeOwned>(response: &Response) -> Result<T, String> {
    serde_json::from_slice(&response.body).map_err(|_| "control response malformed".into())
}
fn save(directory: &Path, name: &str, bytes: &[u8]) -> Result<(), String> {
    std::fs::write(directory.join(name), bytes).map_err(|_| "mixed-version evidence write".into())
}
