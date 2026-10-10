use crate::{command::DynResult, process::retained::recovery::Expected};
use std::{num::NonZeroUsize, path::PathBuf, time::Duration};

pub(super) struct Options {
    pub binary: PathBuf,
    pub model: String,
    pub workers: usize,
    pub expected: Expected,
    pub startup: Duration,
    pub recovery: Duration,
    pub stable: NonZeroUsize,
    pub api: u16,
    pub console: u16,
    pub bind: u16,
    pub discovery: String,
    pub seed_vram: String,
    pub worker_vram: Vec<String>,
    pub context: u32,
    pub device: String,
    pub inference: bool,
    pub keep: bool,
    pub work: Option<PathBuf>,
    pub process_root: Option<PathBuf>,
}

impl Options {
    pub fn parse(args: &[String]) -> DynResult<Self> {
        let [binary, model] = args else {
            return Err("usage: automation startup-recovery BINARY MODEL".into());
        };
        let number = |key, default| -> DynResult<u64> {
            let number = std::env::var(key).map_or(Ok(default), |value| value.parse())?;
            if number == 0 || number > 86400 {
                return Err("split budget or port outside positive bound".into());
            }
            Ok(number)
        };
        let workers = usize::try_from(number("MESH_SPLIT_CERT_WORKERS", 2)?)?;
        if !(2..16).contains(&workers) {
            return Err("split recovery requires 2..=15 workers".into());
        }
        let expected = match std::env::var("MESH_SPLIT_CERT_EXPECT")
            .unwrap_or_else(|_| "replacement".into())
            .as_str()
        {
            "replacement" => Expected::Replacement,
            "local-fallback" => Expected::LocalFallback,
            "withdraw" => Expected::Withdraw,
            "any" => Expected::Any,
            _ => return Err("unknown split recovery outcome".into()),
        };
        let text = |key: &str, default: &str| std::env::var(key).unwrap_or_else(|_| default.into());
        let api = u16::try_from(number("MESH_SPLIT_CERT_BASE_API_PORT", 9460)?)?;
        let console = u16::try_from(number("MESH_SPLIT_CERT_BASE_CONSOLE_PORT", 3260)?)?;
        let bind = u16::try_from(number("MESH_SPLIT_CERT_BASE_BIND_PORT", 54600)?)?;
        for port in [api, console, bind] {
            port.checked_add(u16::try_from(workers)?)
                .ok_or("split port range overflow")?;
        }
        if model.is_empty() {
            return Err("split model must not be empty".into());
        }
        Ok(Self {
            binary: PathBuf::from(binary).canonicalize()?,
            model: model.clone(),
            workers,
            expected,
            startup: Duration::from_secs(number("MESH_SPLIT_CERT_MAX_WAIT", 420)?),
            recovery: Duration::from_secs(number("MESH_SPLIT_CERT_RECOVERY_MAX_WAIT", 240)?),
            stable: NonZeroUsize::new(usize::try_from(number(
                "MESH_SPLIT_CERT_STABLE_PROBES",
                2,
            )?)?)
            .ok_or("stable count")?,
            api,
            console,
            bind,
            discovery: text("MESH_SPLIT_CERT_DISCOVERY_MODE", ""),
            seed_vram: text("MESH_SPLIT_CERT_SEED_MAX_VRAM", "10"),
            worker_vram: text("MESH_SPLIT_CERT_WORKER_MAX_VRAM", "10")
                .split(',')
                .map(str::to_owned)
                .collect(),
            context: u32::try_from(number("MESH_SPLIT_CERT_CTX_SIZE", 1024)?)?,
            device: text("MESH_SPLIT_CERT_DEVICE", ""),
            inference: text("MESH_SPLIT_CERT_RUN_INFERENCE", "0") == "1",
            keep: text("MESH_SPLIT_CERT_KEEP_LOGS", "0") == "1",
            work: std::env::var_os("MESH_SPLIT_CERT_WORK_DIR")
                .filter(|value| !value.is_empty())
                .map(PathBuf::from),
            process_root: std::env::var_os("MESH_SPLIT_CERT_PROCESS_ROOT")
                .filter(|value| !value.is_empty())
                .map(PathBuf::from),
        })
    }
    pub fn member(&self, index: usize) -> DynResult<crate::process::retained::MemberId> {
        Ok(if index == 0 {
            crate::process::retained::MemberId::Seed
        } else {
            crate::process::retained::MemberId::new(&format!("worker-{index}"), 0)?
        })
    }
}
