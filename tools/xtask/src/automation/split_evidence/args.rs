use super::Error;
use std::path::PathBuf;

pub(super) const NAMES: [&str; 6] = [
    "seed_status",
    "seed_stages",
    "seed_models",
    "worker_status",
    "worker_stages",
    "worker_models",
];

pub(super) enum Mode {
    Output(PathBuf),
    Verify(PathBuf),
}

pub(super) struct Request {
    pub(super) paths: [PathBuf; 6],
    pub(super) model_label: String,
    pub(super) mode: Mode,
}

pub(super) fn parse(args: &[String]) -> Result<Request, Error> {
    let mut paths: [Option<PathBuf>; 6] = std::array::from_fn(|_| None);
    let mut label = None;
    let mut mode = None;
    let mut args = args.iter();
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .filter(|value| !value.starts_with("--"))
            .ok_or_else(|| Error::Contract(format!("missing value for {flag}")))?;
        match flag.as_str() {
            "--model-label" => label = Some(value.clone()),
            "--output" | "--verify" => {
                if mode.is_some() {
                    return Err(Error::Contract(
                        "exactly one of --output or --verify is required".into(),
                    ));
                }
                mode = Some(if flag == "--output" {
                    Mode::Output(value.into())
                } else {
                    Mode::Verify(value.into())
                });
            }
            _ => {
                let index = NAMES
                    .iter()
                    .position(|name| format!("--{}", name.replace('_', "-")) == *flag)
                    .ok_or_else(|| Error::Contract(format!("unrecognized argument: {flag}")))?;
                paths[index] = Some(value.into());
            }
        }
    }
    let paths = paths
        .into_iter()
        .zip(NAMES)
        .map(|(path, name)| {
            path.ok_or_else(|| Error::Contract(format!("missing --{}", name.replace('_', "-"))))
        })
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| Error::Contract("six snapshots required".into()))?;
    let model_label = label.ok_or_else(|| Error::Contract("missing --model-label".into()))?;
    let mode = mode
        .ok_or_else(|| Error::Contract("exactly one of --output or --verify is required".into()))?;
    Ok(Request {
        paths,
        model_label,
        mode,
    })
}
