use crate::command::DynResult;
use std::{path::PathBuf, time::Duration};
pub(super) struct Options {
    pub binary: PathBuf,
    pub evidence: PathBuf,
    pub base: u16,
    pub wait: Duration,
    pub keep: bool,
    pub plan: bool,
    pub endpoint: String,
    pub model: String,
}
impl Options {
    pub fn parse(args: &[String]) -> DynResult<Self> {
        let mut options = Self {
            binary: PathBuf::new(),
            evidence: ".sisyphus/evidence".into(),
            base: std::env::var("MESH_QA_BASE_PORT").map_or(Ok(19860), |value| value.parse())?,
            wait: Duration::from_secs(
                std::env::var("MESH_QA_MAX_WAIT").map_or(Ok(45), |value| value.parse())?,
            ),
            keep: false,
            plan: false,
            endpoint: String::new(),
            model: "qa-deterministic".into(),
        };
        let mut arguments = args.iter();
        while let Some(flag) = arguments.next() {
            match flag.as_str() {
                "--current-binary" => {
                    options.binary = arguments.next().ok_or("binary missing")?.into()
                }
                "--evidence-dir" => {
                    options.evidence = arguments.next().ok_or("evidence missing")?.into()
                }
                "--base-port" => options.base = arguments.next().ok_or("port missing")?.parse()?,
                "--max-wait" => {
                    options.wait =
                        Duration::from_secs(arguments.next().ok_or("wait missing")?.parse()?)
                }
                "--deterministic-openai-endpoint" => {
                    options.endpoint = arguments.next().ok_or("endpoint missing")?.clone()
                }
                "--deterministic-openai-model" => {
                    options.model = arguments.next().ok_or("model missing")?.clone()
                }
                "--keep-logs" => options.keep = true,
                "--print-plan" => options.plan = true,
                _ => return Err("unknown logging recovery option".into()),
            }
        }
        if options.binary.as_os_str().is_empty()
            || options.base < 1025
            || options.base > 65500
            || options.wait.is_zero()
            || options.wait > Duration::from_secs(3600)
        {
            return Err("invalid logging recovery inputs".into());
        }
        if !options.plan {
            options.binary = options.binary.canonicalize()?;
        }
        Ok(options)
    }
}
