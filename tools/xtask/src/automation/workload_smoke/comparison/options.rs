use crate::command::DynResult;
use std::{collections::BTreeMap, path::PathBuf};
pub(super) enum Reference {
    Server(String),
    Completion {
        executable: PathBuf,
        model_path: PathBuf,
    },
}
pub(super) struct Options {
    pub candidate: String,
    pub model: String,
    pub class: String,
    pub reference: Reference,
}
impl Options {
    pub fn parse(args: &[String]) -> DynResult<Self> {
        if !args.len().is_multiple_of(2) {
            return Err("oracle requires named option values".into());
        }
        let mut values = BTreeMap::new();
        for pair in args.as_chunks::<2>().0 {
            if !matches!(
                pair[0].as_str(),
                "--candidate-url"
                    | "--oracle-url"
                    | "--oracle-completion"
                    | "--model-path"
                    | "--model"
                    | "--class"
            ) || pair[1].is_empty()
                || values.insert(pair[0].as_str(), pair[1].as_str()).is_some()
            {
                return Err("oracle option is unknown, duplicate or empty".into());
            }
        }
        let required = |key| {
            values
                .get(key)
                .copied()
                .ok_or_else(|| format!("missing {key}"))
        };
        let class = required("--class")?.to_owned();
        let candidate = endpoint(required("--candidate-url")?)?;
        let reference = match class.as_str() {
            "embedding"|"rerank" if !values.contains_key("--oracle-completion") && !values.contains_key("--model-path") => Reference::Server(endpoint(required("--oracle-url")?)?),
            "encoder_decoder" if !values.contains_key("--oracle-url") => Reference::Completion{executable:required("--oracle-completion")?.into(),model_path:required("--model-path")?.into()},
            _ => return Err("embedding/rerank require --oracle-url only; encoder_decoder requires --oracle-completion and --model-path only".into()),
        };
        Ok(Self {
            candidate,
            model: required("--model")?.into(),
            class,
            reference,
        })
    }
}
fn endpoint(text: &str) -> DynResult<String> {
    let text = text.trim_end_matches('/');
    let uri: hyper::Uri = text.parse()?;
    if uri.scheme_str() != Some("http") || uri.host().is_none() {
        return Err("oracle requires HTTP endpoint with host".into());
    }
    Ok(text.into())
}
