use std::{collections::BTreeMap, path::PathBuf};

use super::selection::Selection;
use crate::command::DynResult;

pub(super) struct Options {
    pub dataset_file: PathBuf,
    pub dataset_revision: String,
    pub output: PathBuf,
    pub requests_per_family: usize,
    pub selection: Selection,
}

impl Options {
    pub fn parse(args: &[String]) -> DynResult<Self> {
        let mut values = BTreeMap::new();
        let mut sources = Vec::new();
        let mut remaining = args;
        while !remaining.is_empty() {
            let [flag, value, rest @ ..] = remaining else {
                return Err("prompt manifest option requires a value".into());
            };
            if flag == "--source-dataset" {
                sources.push(value.clone());
            } else if matches!(
                flag.as_str(),
                "--dataset-file"
                    | "--dataset-revision"
                    | "--output"
                    | "--families"
                    | "--requests-per-family"
                    | "--min-isl"
                    | "--max-isl"
                    | "--min-turns"
            ) {
                if values.insert(flag.as_str(), value.as_str()).is_some() {
                    return Err(format!("duplicate prompt manifest option {flag}").into());
                }
            } else {
                return Err(format!("unknown prompt manifest option {flag}").into());
            }
            remaining = rest;
        }
        let selection = Selection {
            sources,
            families: values.get("--families").copied().unwrap_or("8").parse()?,
            min_isl: values.get("--min-isl").copied().unwrap_or("8192").parse()?,
            max_isl_exclusive: values
                .get("--max-isl")
                .copied()
                .unwrap_or("12000")
                .parse()?,
            min_turns: values.get("--min-turns").copied().unwrap_or("20").parse()?,
        };
        selection.validate()?;
        let requests_per_family = values
            .get("--requests-per-family")
            .copied()
            .unwrap_or("2")
            .parse()?;
        if requests_per_family == 0 {
            return Err("requests per family must be positive".into());
        }
        let required = |name| -> DynResult<&str> {
            values
                .get(name)
                .copied()
                .filter(|value| !value.trim().is_empty())
                .ok_or_else(|| format!("missing nonempty prompt manifest option {name}").into())
        };
        Ok(Self {
            dataset_file: required("--dataset-file")?.into(),
            dataset_revision: required("--dataset-revision")?.into(),
            output: required("--output")?.into(),
            requests_per_family,
            selection,
        })
    }
}
