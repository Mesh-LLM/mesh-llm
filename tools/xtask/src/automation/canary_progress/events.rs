//! Cancellable event framing and bounded projections of oversized tool lifecycles.
use super::{AtomicBool, MAX_EVENT, Ordering, mpsc};
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    io::{self, Read, Seek, SeekFrom, Write},
    time::Duration,
};

#[cfg(unix)]
pub(super) struct NonblockingInput {
    file: File,
    flags: rustix::fs::OFlags,
}
#[cfg(unix)]
impl NonblockingInput {
    pub(super) fn stdin() -> io::Result<Self> {
        let file = File::from(rustix::io::dup(io::stdin())?);
        let flags = rustix::fs::fcntl_getfl(&file)?;
        rustix::fs::fcntl_setfl(&file, flags | rustix::fs::OFlags::NONBLOCK)?;
        Ok(Self { file, flags })
    }
}
#[cfg(unix)]
impl Read for NonblockingInput {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        self.file.read(bytes)
    }
}
#[cfg(unix)]
impl Drop for NonblockingInput {
    fn drop(&mut self) {
        let _ = rustix::fs::fcntl_setfl(&self.file, self.flags);
    }
}

pub(super) fn read_events(
    mut input: impl Read,
    send: mpsc::SyncSender<Vec<u8>>,
    cancelled: &AtomicBool,
) -> io::Result<()> {
    let mut event = Event::default();
    let mut buffer = [0; 8192];
    while !cancelled.load(Ordering::Relaxed) {
        let count = match input.read(&mut buffer) {
            Ok(0) => {
                if !event.bytes.is_empty() || event.spool.is_some() {
                    event.send(&send)?;
                }
                return Ok(());
            }
            Ok(count) => count,
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(50));
                continue;
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        };
        for part in buffer[..count].split_inclusive(|byte| *byte == b'\n') {
            event.append(part)?;
            if part.ends_with(b"\n") && !event.send(&send)? {
                return Ok(());
            }
        }
    }
    Ok(())
}

#[derive(Default)]
struct Event {
    bytes: Vec<u8>,
    spool: Option<File>,
}
impl Event {
    fn append(&mut self, part: &[u8]) -> io::Result<()> {
        if self.spool.is_none() && self.bytes.len() + part.len() > MAX_EVENT {
            let mut spool = tempfile::tempfile()?;
            spool.write_all(&self.bytes)?;
            self.bytes.clear();
            self.spool = Some(spool);
        }
        match &mut self.spool {
            Some(spool) => spool.write_all(part),
            None => {
                self.bytes.extend_from_slice(part);
                Ok(())
            }
        }
    }
    fn send(&mut self, send: &mpsc::SyncSender<Vec<u8>>) -> io::Result<bool> {
        let event = if let Some(mut spool) = self.spool.take() {
            spool.seek(SeekFrom::Start(0))?;
            // Unknown fields (including arbitrarily large tool bodies) are skipped
            // by serde's streaming reader instead of being allocated as a Value.
            match serde_json::from_reader::<_, Lifecycle>(io::BufReader::new(spool)) {
                Ok(event) => {
                    if send.send(serde_json::to_vec(&event)?).is_err() {
                        return Ok(false);
                    }
                    b"Goose event exceeds display limit; lifecycle retained; full event in agent.log\n".to_vec()
                }
                Err(_) => b"Goose event exceeds display limit; full event retained in agent.log\n"
                    .to_vec(),
            }
        } else {
            std::mem::take(&mut self.bytes)
        };
        Ok(send.send(event).is_ok())
    }
}

#[derive(Default, Deserialize, Serialize)]
#[serde(default)]
struct Lifecycle {
    r#type: String,
    message: Message,
}
#[derive(Default, Deserialize, Serialize)]
#[serde(default)]
struct Message {
    role: String,
    content: Vec<Part>,
}
#[derive(Default, Deserialize, Serialize)]
#[serde(default, rename_all = "camelCase")]
struct Part {
    r#type: String,
    id: String,
    tool_call: ToolCall,
    tool_result: ToolResult,
}
#[derive(Default, Deserialize, Serialize)]
#[serde(default)]
struct ToolCall {
    status: String,
}
#[derive(Default, Deserialize, Serialize)]
#[serde(default)]
struct ToolResult {
    status: String,
    value: ToolValue,
}
#[derive(Default, Deserialize, Serialize)]
#[serde(default, rename_all = "camelCase")]
struct ToolValue {
    is_error: bool,
}
