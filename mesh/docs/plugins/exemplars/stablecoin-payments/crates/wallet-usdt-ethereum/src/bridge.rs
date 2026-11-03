//! Private stdio boundary to the pinned WDK runtime. No signing API listens on HTTP.
use anyhow::{Context, Result, bail, ensure};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::{path::PathBuf, process::Stdio, sync::Arc, time::Duration};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::{Child, ChildStdin, ChildStdout, Command},
    sync::Mutex,
};

use crate::wallet::Settings;

const MAX_RESPONSE: usize = 1024 * 1024;

#[async_trait]
pub trait Worker: Send + Sync {
    async fn call(&self, operation: &str, input: Value) -> Result<Value>;
}

struct Process {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
    poisoned: bool,
}

pub struct Bridge(Mutex<Process>);

impl Bridge {
    pub async fn spawn(settings: &Settings) -> Result<Arc<Self>> {
        let path = worker_path(settings)?;
        let mut child = Command::new(&settings.node_binary)
            .arg(&path)
            .env_remove("NODE_OPTIONS")
            .env_remove("NODE_PATH")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .context("starting WDK worker; install Node.js and the locked WDK dependencies")?;
        let stdin = child.stdin.take().context("missing WDK input pipe")?;
        let stdout = BufReader::new(child.stdout.take().context("missing WDK output pipe")?);
        Ok(Arc::new(Self(Mutex::new(Process {
            child,
            stdin,
            stdout,
            next_id: 1,
            poisoned: false,
        }))))
    }
}

fn worker_path(settings: &Settings) -> Result<PathBuf> {
    if let Some(path) = &settings.worker_path {
        ensure!(path.is_absolute(), "worker_path must be absolute");
        ensure!(path.is_file(), "WDK worker file does not exist");
        return Ok(path.clone());
    }
    let installed = std::env::current_exe()?
        .parent()
        .context("executable has no parent")?
        .join("wdk/worker.mjs");
    if installed.is_file() {
        return Ok(installed);
    }
    #[cfg(debug_assertions)]
    {
        let development = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../wallets/usdt-ethereum-wdk/worker.mjs");
        if development.is_file() {
            return Ok(development);
        }
    }
    bail!("missing packaged wdk/worker.mjs; use a complete plugin archive or set worker_path")
}

#[async_trait]
impl Worker for Bridge {
    async fn call(&self, operation: &str, input: Value) -> Result<Value> {
        let mut process = self.0.lock().await;
        ensure!(
            !process.poisoned,
            "WDK worker stopped; restart provider and reconcile the existing payment ID"
        );
        // Cancellation by the host must not let the next operation consume an
        // unfinished response. Only a complete protocol exchange clears this.
        process.poisoned = true;
        let result = tokio::time::timeout(
            Duration::from_secs(24),
            exchange(&mut process, operation, input),
        )
        .await;
        match result {
            Ok(result) => {
                if result.is_err() {
                    // A protocol failure must not leave a stale response in the stream.
                    process.poisoned = true;
                    let _ = process.child.kill().await;
                } else {
                    process.poisoned = false;
                }
                result
            }
            Err(_) => {
                process.poisoned = true;
                let _ = process.child.kill().await;
                bail!("WDK operation timed out; reconcile any journaled transaction after restart")
            }
        }
    }
}

async fn exchange(process: &mut Process, operation: &str, input: Value) -> Result<Value> {
    let id = process.next_id;
    process.next_id = id.checked_add(1).context("WDK request counter overflow")?;
    let mut encoded = serde_json::to_vec(&json!({"id":id,"operation":operation,"input":input}))?;
    ensure!(encoded.len() <= MAX_RESPONSE, "WDK request too large");
    encoded.push(b'\n');
    process.stdin.write_all(&encoded).await?;
    process.stdin.flush().await?;
    let mut bytes = Vec::new();
    // read_until on an unbounded stream could allocate before checking its size.
    loop {
        let available = process.stdout.fill_buf().await?;
        ensure!(
            !available.is_empty(),
            "WDK worker exited without a response"
        );
        let take = available
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(available.len(), |position| position + 1);
        ensure!(bytes.len() + take <= MAX_RESPONSE, "WDK response too large");
        let complete = available[take - 1] == b'\n';
        bytes.extend_from_slice(&available[..take]);
        process.stdout.consume(take);
        if complete {
            break;
        }
    }
    let response: Value =
        serde_json::from_slice(&bytes).context("invalid WDK protocol response")?;
    ensure!(
        response["id"].as_u64() == Some(id),
        "WDK response ID mismatch"
    );
    // Worker errors have already been scrubbed; never include request/response
    // bodies in diagnostics, because a signed payload is authorization material.
    if response.get("error").is_some() {
        let message = response["error"].as_str().unwrap_or("WDK operation failed");
        return Ok(json!({"__worker_error":message}));
    }
    response
        .get("result")
        .cloned()
        .context("missing WDK result")
}

pub async fn invoke(worker: &dyn Worker, operation: &str, input: Value) -> Result<Value> {
    let value = worker.call(operation, input).await?;
    if let Some(error) = value.get("__worker_error") {
        bail!("{}", error.as_str().unwrap_or("WDK operation failed"));
    }
    Ok(value)
}
