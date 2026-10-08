use crate::command::DynResult;
use serde::Serialize;
use std::{path::PathBuf, time::Duration};

#[derive(Serialize)]
pub(super) struct Options {
    pub binary: PathBuf,
    pub evidence_root: PathBuf,
    pub base_port: u16,
    pub max_wait_seconds: u64,
    pub keep_state: bool,
    pub print_plan: bool,
}

impl Options {
    pub fn parse(args: &[String]) -> DynResult<Self> {
        let mut options = Self {
            binary: PathBuf::new(),
            evidence_root: ".sisyphus/evidence".into(),
            base_port: std::env::var("MESH_QA_BASE_PORT")
                .map_or(Ok(20960), |value| value.parse())?,
            max_wait_seconds: std::env::var("MESH_QA_MAX_WAIT")
                .map_or(Ok(45), |value| value.parse())?,
            keep_state: false,
            print_plan: false,
        };
        let mut arguments = args.iter();
        while let Some(flag) = arguments.next() {
            match flag.as_str() {
                "--keep-state" => options.keep_state = true,
                "--print-plan" => options.print_plan = true,
                "--current-binary" => {
                    options.binary = arguments.next().ok_or("missing binary")?.into()
                }
                "--evidence-dir" => {
                    options.evidence_root = arguments.next().ok_or("missing evidence root")?.into()
                }
                "--base-port" => {
                    options.base_port = arguments.next().ok_or("missing port")?.parse()?
                }
                "--max-wait" => {
                    options.max_wait_seconds = arguments.next().ok_or("missing wait")?.parse()?
                }
                _ => return Err("unknown logging console option".into()),
            }
        }
        if options.binary.as_os_str().is_empty()
            || !(1025..=65533).contains(&options.base_port)
            || !(1..=3600).contains(&options.max_wait_seconds)
        {
            return Err("logging console requires binary, valid base port and bounded wait".into());
        }
        if !options.print_plan {
            options.binary = options.binary.canonicalize()?;
        }
        Ok(options)
    }
    pub const fn wait(&self) -> Duration {
        Duration::from_secs(self.max_wait_seconds)
    }
}
