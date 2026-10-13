use super::{Model, Stack, Variant};
use crate::command::DynResult;
use std::{collections::BTreeMap, path::PathBuf, time::Duration};

pub(super) struct Options {
    pub binary: PathBuf,
    pub model: String,
    pub native: PathBuf,
    pub parent: PathBuf,
    pub device: String,
    pub projector: Option<PathBuf>,
    pub key: Option<PathBuf>,
    pub expected: String,
    pub variant: Variant,
    pub readiness: Duration,
    pub shutdown: Duration,
    pub context_size: Option<u32>,
    pub batch_sizes: Option<(u32, u32)>,
    pub endpoints: Option<[(u16, u16); 2]>,
}

impl Options {
    pub(super) fn parse(args: &[String]) -> DynResult<Self> {
        let mut values = BTreeMap::new();
        let mut args = args.iter();
        while let Some(flag) = args.next() {
            let value = args.next().ok_or("missing option value")?;
            if !matches!(
                flag.as_str(),
                "--binary"
                    | "--model"
                    | "--native-runtime-root"
                    | "--state-parent"
                    | "--device"
                    | "--mmproj"
                    | "--public-key-file"
                    | "--expected-attestation"
                    | "--model-class"
                    | "--stack"
                    | "--ready-max-wait"
                    | "--shutdown-max-wait"
                    | "--ctx-size"
                    | "--batch-size"
                    | "--ubatch-size"
                    | "--api-port"
                    | "--console-port"
                    | "--headless-api-port"
                    | "--headless-console-port"
            ) {
                return Err("unknown smoke option".into());
            }
            if value.starts_with("--") || values.insert(flag.as_str(), value.as_str()).is_some() {
                return Err("missing value or duplicate smoke option".into());
            }
        }
        let required = |flag| {
            values
                .get(flag)
                .copied()
                .ok_or("required smoke option missing")
        };
        let path = |value: &str| -> DynResult<PathBuf> {
            let path = PathBuf::from(value);
            if !path.is_absolute() {
                return Err("smoke paths must be absolute".into());
            }
            Ok(path.canonicalize()?)
        };
        let binary = path(required("--binary")?)?;
        if !binary.is_file() {
            return Err("binary must be a file".into());
        }
        let native = path(required("--native-runtime-root")?)?;
        let parent = match values.get("--state-parent") {
            Some(value) => path(value)?,
            None => std::env::temp_dir().canonicalize()?,
        };
        if !native.is_dir() || !parent.is_dir() {
            return Err("expected directory".into());
        }
        let model = required("--model")?.to_owned();
        if model.is_empty() {
            return Err("empty model".into());
        }
        let model_class = match values.get("--model-class").copied().unwrap_or("dense") {
            "dense" => Model::Dense,
            "recurrent" => Model::Recurrent,
            _ => return Err("model class must be dense or recurrent".into()),
        };
        let stack = match values.get("--stack").copied().unwrap_or("default") {
            "default" => Stack::Default,
            "constrained" => Stack::Constrained,
            _ => return Err("stack must be default or constrained".into()),
        };
        let seconds = |flag, default| -> DynResult<Duration> {
            let seconds = values
                .get(flag)
                .map_or(Ok(default), |value| value.parse::<u64>())?;
            if !(1..=3600).contains(&seconds) {
                return Err("budget must be 1..=3600 seconds".into());
            }
            Ok(Duration::from_secs(seconds))
        };
        let positive = |flag| -> DynResult<Option<u32>> {
            values
                .get(flag)
                .map(|value| {
                    let number = value.parse::<u32>()?;
                    if number == 0 {
                        return Err("size must be positive".into());
                    }
                    Ok(number)
                })
                .transpose()
        };
        let batch_sizes = match (positive("--batch-size")?, positive("--ubatch-size")?) {
            (Some(batch), Some(ubatch)) => Some((batch, ubatch)),
            (None, None) => None,
            _ => return Err("batch and ubatch must be set together".into()),
        };
        let port_keys = [
            "--api-port",
            "--console-port",
            "--headless-api-port",
            "--headless-console-port",
        ];
        let endpoints = if port_keys.iter().any(|key| values.contains_key(key)) {
            let port = |key| -> DynResult<u16> {
                let value = required(key)?.parse::<u16>()?;
                if value == 0 {
                    return Err("explicit port must be nonzero".into());
                }
                Ok(value)
            };
            let ports = [
                port(port_keys[0])?,
                port(port_keys[1])?,
                port(port_keys[2])?,
                port(port_keys[3])?,
            ];
            for (index, value) in ports.iter().enumerate() {
                if ports[..index].contains(value) {
                    return Err("smoke ports must be distinct".into());
                }
            }
            Some([(ports[0], ports[1]), (ports[2], ports[3])])
        } else {
            None
        };
        Ok(Self {
            binary,
            model,
            native,
            parent,
            device: values.get("--device").copied().unwrap_or("CPU").into(),
            projector: values
                .get("--mmproj")
                .map(|value| path(value))
                .transpose()?,
            key: values
                .get("--public-key-file")
                .map(|value| path(value))
                .transpose()?,
            expected: values
                .get("--expected-attestation")
                .copied()
                .unwrap_or("valid")
                .into(),
            variant: Variant::REQUIRED
                .into_iter()
                .find(|variant| variant.model == model_class && variant.stack == stack)
                .ok_or("unknown smoke variant")?,
            readiness: seconds("--ready-max-wait", 180)?,
            shutdown: seconds("--shutdown-max-wait", 15)?,
            context_size: positive("--ctx-size")?,
            batch_sizes,
            endpoints,
        })
    }
}
