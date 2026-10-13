//! Finite standalone whole-trajectory command; outputs only after full admission.
use super::{Selection, document, input};
use crate::DynResult;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    io::{Read, Write},
    path::{Path, PathBuf},
};
const OPTIONS: [&str; 11] = [
    "--dataset-file",
    "--dataset-revision",
    "--output",
    "--cohort",
    "--framework",
    "--source-dataset",
    "--sessions-per-cohort",
    "--trajectories-per-framework",
    "--min-isl",
    "--max-isl",
    "--min-turns",
];
pub const USAGE: &str = "trajectory-reader cohorts --dataset-file FILE --dataset-revision COMMIT --output FILE --cohort NAME... --framework NAME... --source-dataset NAME... (--sessions-per-cohort N | --trajectories-per-framework N) [--min-isl N] [--max-isl N] [--min-turns N]";
fn parse(args: &[String]) -> DynResult<BTreeMap<&str, Vec<&str>>> {
    if !args.len().is_multiple_of(2) {
        return Err(USAGE.into());
    }
    let mut values = BTreeMap::<&str, Vec<&str>>::new();
    for pair in args.as_chunks::<2>().0 {
        let key = pair[0].as_str();
        if !OPTIONS.contains(&key) {
            return Err(format!("unknown option {key}").into());
        }
        let bucket = values.entry(key).or_default();
        if !bucket.is_empty() && !["--cohort", "--framework", "--source-dataset"].contains(&key) {
            return Err(format!("duplicate option {key}").into());
        }
        bucket.push(&pair[1]);
    }
    Ok(values)
}
fn digest(path: &Path) -> DynResult<String> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NONBLOCK);
    }
    let mut file = options.open(path)?;
    if !file.metadata()?.is_file() {
        return Err("dataset must be a regular file".into());
    }
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(hash
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}
pub fn run(args: &[String]) -> DynResult<()> {
    let values = parse(args)?;
    let one = |key| -> DynResult<&str> {
        values
            .get(key)
            .and_then(|v| v.first().copied())
            .ok_or_else(|| format!("missing {key}").into())
    };
    let names = |key| {
        values
            .get(key)
            .into_iter()
            .flatten()
            .map(|s| (*s).to_owned())
            .collect()
    };
    let count = |key| -> DynResult<Option<usize>> {
        values
            .get(key)
            .map(|v| v[0].parse::<usize>().map_err(Into::into))
            .transpose()
    };
    let number = |key, default| -> DynResult<u64> {
        values
            .get(key)
            .map_or(Ok(default), |v| v[0].parse().map_err(Into::into))
    };
    let selection = Selection {
        cohorts: names("--cohort"),
        frameworks: names("--framework"),
        sources: names("--source-dataset"),
        trajectories_per_framework: count("--trajectories-per-framework")?,
        sessions_per_cohort: count("--sessions-per-cohort")?,
        min_isl: number("--min-isl", 8192)?,
        max_isl_exclusive: number("--max-isl", 65536)?,
        min_turns: number("--min-turns", 5)?,
    };
    selection.allocation()?;
    let revision = one("--dataset-revision")?;
    document::manifest(Default::default(), &selection, revision, "")?;
    let dataset = std::path::absolute(PathBuf::from(one("--dataset-file")?))?;
    let output = std::path::absolute(PathBuf::from(one("--output")?))?;
    if !dataset.is_absolute() || !output.is_absolute() {
        return Err("dataset and output paths must be absolute".into());
    }
    if output.try_exists()? {
        return Err("trajectory manifest output already exists".into());
    }
    let before = digest(&dataset)?;
    let cohorts = input::select(&dataset, &selection)?;
    if digest(&dataset)? != before {
        return Err("dataset changed during selection".into());
    }
    let document = document::manifest(cohorts, &selection, revision, &before)?;
    let bytes = serde_json::to_vec_pretty(&document)?;
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output)?;
    let result = (|| -> DynResult<()> {
        file.write_all(&bytes)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        Ok(())
    })();
    if result.is_err() {
        drop(file);
        std::fs::remove_file(&output)?;
    }
    result
}
