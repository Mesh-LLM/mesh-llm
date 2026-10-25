use crate::command::DynResult;
use std::{path::PathBuf, time::Duration};
pub(super) struct Options {
    pub current: PathBuf,
    pub released: PathBuf,
    pub evidence: PathBuf,
    pub local: bool,
    pub config: bool,
    pub cargo: bool,
    pub public_required: bool,
    pub plan: bool,
    pub keep: bool,
    pub current_model: String,
    pub released_model: String,
    pub base: u16,
    pub wait: Duration,
    pub stable: usize,
    pub chat: Duration,
    pub context: u32,
}
impl Options {
    pub fn parse(args: &[String]) -> DynResult<Self> {
        let mut options = Self {
            current: PathBuf::new(),
            released: PathBuf::new(),
            evidence: ".sisyphus/evidence".into(),
            local: false,
            config: false,
            cargo: true,
            public_required: false,
            plan: false,
            keep: false,
            current_model: String::new(),
            released_model: String::new(),
            base: 19640,
            wait: Duration::from_secs(180),
            stable: 3,
            chat: Duration::from_secs(120),
            context: 512,
        };
        for (key, target) in [
            ("MESH_QA_BASE_PORT", 0),
            ("MESH_QA_MAX_WAIT", 1),
            ("MESH_QA_STABLE_PROBES", 2),
            ("MESH_QA_CHAT_MAX_TIME", 3),
            ("MESH_QA_CTX_SIZE", 4),
        ] {
            if let Ok(value) = std::env::var(key) {
                match target {
                    0 => options.base = value.parse()?,
                    1 => options.wait = Duration::from_secs(value.parse()?),
                    2 => options.stable = value.parse()?,
                    3 => options.chat = Duration::from_secs(value.parse()?),
                    4 => options.context = value.parse()?,
                    _ => unreachable!(),
                }
            }
        }
        let mut model = String::new();
        let mut arguments = args.iter();
        while let Some(flag) = arguments.next() {
            match flag.as_str() {
                "--current-binary" => {
                    options.current = arguments.next().ok_or("current binary missing")?.into()
                }
                "--released-binary" => {
                    options.released = arguments.next().ok_or("released binary missing")?.into()
                }
                "--evidence-dir" => {
                    options.evidence = arguments.next().ok_or("evidence missing")?.into()
                }
                "--model" => model = arguments.next().ok_or("model missing")?.clone(),
                "--current-model" => {
                    options.current_model = arguments.next().ok_or("current model missing")?.clone()
                }
                "--released-model" => {
                    options.released_model =
                        arguments.next().ok_or("released model missing")?.clone()
                }
                "--base-port" => options.base = arguments.next().ok_or("port missing")?.parse()?,
                "--max-wait" => {
                    options.wait =
                        Duration::from_secs(arguments.next().ok_or("wait missing")?.parse()?)
                }
                "--stable-probes" => {
                    options.stable = arguments.next().ok_or("stable probes missing")?.parse()?
                }
                "--chat-max-time" => {
                    options.chat =
                        Duration::from_secs(arguments.next().ok_or("chat budget missing")?.parse()?)
                }
                "--ctx-size" => {
                    options.context = arguments.next().ok_or("context missing")?.parse()?
                }
                "--local-only" => options.local = true,
                "--config-only" => {
                    options.local = true;
                    options.config = true;
                }
                "--skip-cargo-tests" => options.cargo = false,
                "--require-public" => options.public_required = true,
                "--print-plan" => options.plan = true,
                "--keep-logs" => options.keep = true,
                _ => return Err("unknown mixed-version option".into()),
            }
        }
        if options.current_model.is_empty() {
            options.current_model = model.clone();
        }
        if options.released_model.is_empty() {
            options.released_model = model;
        }
        if options.current.as_os_str().is_empty()
            || options.released.as_os_str().is_empty()
            || options.base < 1025
            || options.base > 65440
            || options.wait.is_zero()
            || options.wait > Duration::from_secs(3600)
            || options.chat.is_zero()
            || options.chat > Duration::from_secs(3600)
            || options.stable == 0
            || options.stable > 100
            || options.context == 0
        {
            return Err("invalid mixed-version inputs".into());
        }
        if !options.plan {
            options.current = options.current.canonicalize()?;
            options.released = options.released.canonicalize()?;
        }
        Ok(options)
    }
}
