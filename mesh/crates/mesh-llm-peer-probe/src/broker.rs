use crate::protocol::{Challenge, Intent, ProcessBinding, ToolRecord, range_digest};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::PathBuf,
};
use sysinfo::{Pid, System};

pub trait Broker: Send {
    fn observe(&mut self, challenge: &Challenge, tool: Intent) -> Result<ToolRecord>;
}

pub struct LocalBroker {
    executable: PathBuf,
    pid: u32,
    started: u64,
}

impl LocalBroker {
    pub fn current() -> Result<Self> {
        let pid = std::process::id();
        let system = System::new_all();
        let started = system
            .process(Pid::from_u32(pid))
            .context("current process not found")?
            .start_time();
        Ok(Self {
            executable: std::env::current_exe()?,
            pid,
            started,
        })
    }

    pub fn reference_bytes(&self) -> Result<Vec<u8>> {
        Ok(std::fs::read(&self.executable)?)
    }

    fn handle(&self) -> String {
        format!("probe-exe-{}-{}", self.pid, self.started)
    }

    fn binding(&self) -> ProcessBinding {
        ProcessBinding {
            pid: u64::from(self.pid),
            start_time: self.started,
            executable_handle: self.handle(),
        }
    }
}

impl Broker for LocalBroker {
    fn observe(&mut self, challenge: &Challenge, tool: Intent) -> Result<ToolRecord> {
        let (supported, source, data) = match tool {
            Intent::Process => (
                true,
                "local_os",
                json!({"pid":self.pid,"start_time":self.started,"executable_handle":self.handle(),"scope":"probe_component_process"}),
            ),
            Intent::Executable => {
                if challenge
                    .process_binding
                    .as_ref()
                    .is_some_and(|binding| binding != &self.binding())
                {
                    return Ok(ToolRecord {
                        tool,
                        supported: false,
                        source: "local_os".into(),
                        data: json!({"reason":"process_changed_since_prior_round"}),
                    });
                }
                let mut file = File::open(&self.executable)?;
                let end = challenge
                    .offset
                    .checked_add(u64::from(challenge.length))
                    .context("range overflow")?;
                ensure!(end <= file.metadata()?.len(), "sample outside executable");
                file.seek(SeekFrom::Start(challenge.offset))?;
                let mut bytes = vec![0; challenge.length as usize];
                file.read_exact(&mut bytes)?;
                let digest = range_digest(&bytes, challenge.offset, &challenge.nonce)?;
                (
                    true,
                    "local_os",
                    json!({"pid":self.pid,"start_time":self.started,"executable_handle":self.handle(),"sha256":digest,"offset":challenge.offset,"length":challenge.length}),
                )
            }
            Intent::Model => (
                false,
                "mesh_integration_missing",
                json!({"reason":"No authenticated serving exchange or active-model handle is connected to this probe component."}),
            ),
        };
        Ok(ToolRecord {
            tool,
            supported,
            source: source.into(),
            data,
        })
    }
}

#[derive(Clone)]
pub struct FixtureBroker {
    pub bytes: Vec<u8>,
    pub process_available: bool,
    pub executable_available: bool,
    pub model_metadata: Option<Value>,
    pub digest_override: Option<String>,
    pub injected_note: Option<String>,
    pub pid: u64,
    pub started: u64,
    pub executable_handle: String,
}

impl FixtureBroker {
    pub fn honest(bytes: &[u8]) -> Self {
        Self {
            bytes: bytes.to_vec(),
            process_available: true,
            executable_available: true,
            model_metadata: Some(
                json!({"architecture":"qwen3","tensor_count":350,"file_size":559423488u64}),
            ),
            digest_override: None,
            injected_note: None,
            pid: 4182,
            started: 1234567890,
            executable_handle: "fixture-exe-1".into(),
        }
    }

    fn binding(&self) -> ProcessBinding {
        ProcessBinding {
            pid: self.pid,
            start_time: self.started,
            executable_handle: self.executable_handle.clone(),
        }
    }
}

impl Broker for FixtureBroker {
    fn observe(&mut self, challenge: &Challenge, tool: Intent) -> Result<ToolRecord> {
        let missing = |reason: &str| ToolRecord {
            tool,
            supported: false,
            source: "simulated_fixture".into(),
            data: json!({"reason":reason}),
        };
        Ok(match tool {
            Intent::Process if self.process_available => {
                let mut data = json!({"pid":self.pid,"start_time":self.started,"executable_handle":self.executable_handle,"scope":"probe_component_process"});
                if let Some(note) = &self.injected_note {
                    data["note"] = json!(note);
                }
                ToolRecord {
                    tool,
                    supported: true,
                    source: "simulated_fixture".into(),
                    data,
                }
            }
            Intent::Process => missing("process observation unavailable"),
            Intent::Executable if self.executable_available && self.process_available => {
                if challenge
                    .process_binding
                    .as_ref()
                    .is_some_and(|binding| binding != &self.binding())
                {
                    return Ok(missing("process_changed_since_prior_round"));
                }
                let start = usize::try_from(challenge.offset)?;
                let end = start
                    .checked_add(challenge.length as usize)
                    .context("range overflow")?;
                ensure!(end <= self.bytes.len(), "sample outside fixture");
                let digest = match &self.digest_override {
                    Some(digest) => digest.clone(),
                    None => {
                        range_digest(&self.bytes[start..end], challenge.offset, &challenge.nonce)?
                    }
                };
                ToolRecord {
                    tool,
                    supported: true,
                    source: "simulated_fixture".into(),
                    data: json!({"pid":self.pid,"start_time":self.started,"executable_handle":self.executable_handle,"sha256":digest,"offset":challenge.offset,"length":challenge.length}),
                }
            }
            Intent::Executable => missing("executable or process observation unavailable"),
            Intent::Model => match &self.model_metadata {
                Some(meta) => {
                    let mut data = meta.clone();
                    if let Some(note) = &self.injected_note {
                        data["note"] = json!(note);
                    }
                    ToolRecord {
                        tool,
                        supported: true,
                        source: "simulated_fixture_self_report".into(),
                        data,
                    }
                }
                None => missing("active model metadata unavailable"),
            },
        })
    }
}
