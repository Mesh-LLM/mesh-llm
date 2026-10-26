use crate::command::DynResult;
use std::{collections::BTreeMap, path::PathBuf};
pub(super) struct Options {
    pub root: PathBuf,
    pub oracle: PathBuf,
    pub model_path: PathBuf,
    pub projector: PathBuf,
    pub model: String,
    pub layer_end: u32,
    pub work: PathBuf,
}
impl Options {
    pub fn parse(args: &[String]) -> DynResult<Self> {
        if !args.len().is_multiple_of(2) {
            return Err("TTS requires named option values".into());
        }
        let mut values = BTreeMap::new();
        for pair in args.as_chunks::<2>().0 {
            if !matches!(
                pair[0].as_str(),
                "--root"
                    | "--oracle-cli"
                    | "--model-path"
                    | "--projector-path"
                    | "--model"
                    | "--layer-end"
                    | "--work-dir"
            ) || pair[1].is_empty()
                || values.insert(pair[0].as_str(), pair[1].as_str()).is_some()
            {
                return Err("unknown, duplicate or empty TTS option".into());
            }
        }
        let required = |key| {
            values
                .get(key)
                .copied()
                .ok_or_else(|| format!("missing {key}"))
        };
        let root = PathBuf::from(required("--root")?).canonicalize()?;
        let layer_end = required("--layer-end")?.parse::<u32>()?;
        if layer_end == 0 {
            return Err("positive TTS layer count required".into());
        }
        Ok(Self {
            root,
            oracle: PathBuf::from(required("--oracle-cli")?).canonicalize()?,
            model_path: PathBuf::from(required("--model-path")?).canonicalize()?,
            projector: PathBuf::from(required("--projector-path")?).canonicalize()?,
            model: required("--model")?.into(),
            layer_end,
            work: std::path::absolute(required("--work-dir")?)?,
        })
    }
}
