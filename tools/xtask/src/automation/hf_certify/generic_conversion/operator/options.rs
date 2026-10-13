use super::Input;
use crate::command::DynResult;
use std::{collections::BTreeMap, path::PathBuf};
pub(super) struct Options {
    pub input: PathBuf,
    pub output: PathBuf,
    values: BTreeMap<String, String>,
    upload: bool,
    dry: bool,
    confirm: bool,
    pub xet: bool,
}
pub(super) fn parse(args: &[String]) -> DynResult<Options> {
    let mut values = BTreeMap::new();
    let mut flags = std::collections::BTreeSet::new();
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if [
            "--upload-only",
            "--dry-run",
            "--confirm-publication",
            "--xet-high-performance",
        ]
        .contains(&arg.as_str())
        {
            if !flags.insert(arg.clone()) {
                return Err("duplicate operator flag".into());
            }
            i += 1;
            continue;
        }
        let (key, value) = if let Some((k, v)) = arg.split_once('=') {
            (k.to_string(), v.to_string())
        } else {
            let value = args.get(i + 1).ok_or("operator flag value absent")?.clone();
            i += 1;
            (arg.clone(), value)
        };
        if ![
            "--input",
            "--output-directory",
            "--source",
            "--source-repo",
            "--target-repo",
            "--target-prefix",
            "--output-basename",
            "--expected-splits",
            "--split-max-size",
            "--max-memory",
            "--work-dir",
            "--mesh-repo",
            "--mesh-revision",
        ]
        .contains(&key.as_str())
            || values.insert(key, value).is_some()
        {
            return Err("unknown/duplicate operator option".into());
        }
        i += 1;
    }
    if values
        .get("--mesh-repo")
        .is_some_and(|v| v != "https://github.com/Mesh-LLM/mesh-llm.git")
    {
        return Err("observed G3 owns canonical mesh repository only".into());
    }
    Ok(Options {
        input: values
            .remove("--input")
            .ok_or("generic-job requires --input prepared manifest")?
            .into(),
        output: values
            .remove("--output-directory")
            .ok_or("generic-job requires --output-directory fresh evidence")?
            .into(),
        values,
        upload: flags.contains("--upload-only"),
        dry: flags.contains("--dry-run"),
        confirm: flags.contains("--confirm-publication"),
        xet: flags.contains("--xet-high-performance"),
    })
}
impl Options {
    pub(super) fn apply(&self, input: &mut Input) -> DynResult<()> {
        for (key, value) in &self.values {
            match key.as_str() {
                "--source" => input.source = value.into(),
                "--work-dir" => input.work_directory = value.into(),
                "--source-repo" => input.source_repo = value.clone(),
                "--target-repo" => input.target_repo = value.clone(),
                "--target-prefix" => input.target_prefix = value.clone(),
                "--output-basename" => input.output_basename = value.clone(),
                "--expected-splits" => input.expected_splits = value.parse()?,
                "--split-max-size" => input.split_max_size = value.clone(),
                "--max-memory" => input.max_memory = value.clone(),
                "--mesh-revision" => input.mesh_revision = value.clone(),
                "--mesh-repo" => (),
                _ => return Err("closed operator options".into()),
            }
        }
        input.upload_only = self.upload;
        input.dry_run = self.dry;
        input.publish_confirmed = self.confirm;
        if input.upload_only {
            input.binary = None;
            input.source_files.clear();
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn generic_operator_original_flags_preserve_defaults_literal_values_and_closed_bootstrap_repo()
    {
        let root = std::env::current_dir().unwrap();
        let mut input:Input=serde_json::from_value(serde_json::json!({"schema_version":1,"source_repo":"fixture/source","target_repo":"fixture/result","mesh_revision":"a".repeat(40),"output_basename":"model","binary":null,"source_files":[],"timeout_seconds":40})).unwrap();
        let args = [
            "--input",
            "/prepared/request.json",
            "--output-directory",
            "/fresh/evidence",
            "--source-repo=fixture/source",
            "--target-repo",
            "fixture/result",
            "--output-basename",
            "model",
            "--mesh-revision",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "--mesh-repo",
            "https://github.com/Mesh-LLM/mesh-llm.git",
        ]
        .map(String::from);
        let parsed = parse(&args).unwrap();
        parsed.apply(&mut input).unwrap();
        assert_eq!(input.source, PathBuf::from("/mnt/checkpoint"));
        assert_eq!(input.work_directory, PathBuf::from("/data/skippy-convert"));
        assert_eq!(input.target_prefix, "BF16");
        assert_eq!(input.expected_splits, 1);
        assert_eq!(input.split_max_size, "50G");
        assert_eq!(input.max_memory, "24G");
        assert!(!input.publish_confirmed);
        let literal = [
            "--input",
            "/prepared/request.json",
            "--output-directory",
            "/fresh/evidence",
            "--max-memory",
            "24G",
            "--source",
            "/literal/space $(never executed)",
            "--upload-only",
            "--xet-high-performance",
        ]
        .map(String::from);
        let opts = parse(&literal).unwrap();
        opts.apply(&mut input).unwrap();
        assert!(input.upload_only && opts.xet);
        assert_eq!(
            input.source,
            PathBuf::from("/literal/space $(never executed)")
        );
        assert!(input.source_files.is_empty());
        let bad = [
            "--input",
            "/i",
            "--output-directory",
            "/o",
            "--mesh-repo",
            "https://foreign.invalid/source",
        ]
        .map(String::from);
        assert!(parse(&bad).is_err());
        assert!(parse(&["--upload-only=true".into()]).is_err());
        input.source = root.clone();
        input.work_directory = root;
        input.upload_only = false;
        assert!(input.validate_for_bootstrap().is_err());
    }
}
