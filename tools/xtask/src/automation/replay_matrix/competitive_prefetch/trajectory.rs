//! Existing approved native trajectory owner; no tokenizer/parser substitution.
use crate::{
    automation::{hf_certify::execution, waiting_prefix::adaptive_identity},
    command::DynResult,
    process::{
        self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, RawCaptureOptions,
        Readiness, Value,
    },
};
use serde_json::{Value as Json, json};
use sha2::{Digest as _, Sha256};
use std::{
    collections::BTreeMap,
    num::NonZeroUsize,
    path::Path,
    time::{Duration, Instant},
};
pub(super) struct Context<'a> {
    pub root: &'a Path,
    pub deadline: Instant,
    pub cancel: &'a Cancellation,
    pub reader: Option<&'a Path>,
    pub reader_sha256: Option<&'a str>,
}
impl Context<'_> {
    fn phase(&self, args: Vec<String>, label: &str) -> DynResult<process::RawProcessReport> {
        super::check(self.deadline, self.cancel)?;
        if args.first().is_none_or(|s| s != "models") {
            super::pin(
                self.reader
                    .ok_or("reader required for dataset materialization")?,
                self.reader_sha256.ok_or("reader SHA required")?,
                self.deadline,
                self.cancel,
            )?;
        }
        let execution = self
            .deadline
            .checked_duration_since(Instant::now())
            .and_then(|d| d.checked_sub(Duration::from_secs(3)))
            .filter(|d| !d.is_zero())
            .ok_or("reader phase lacks execution allowance")?;
        let mut environment: BTreeMap<_, _> = ["PATH", "SYSTEMROOT", "WINDIR"]
            .into_iter()
            .filter_map(|k| std::env::var_os(k).map(|v| (k.into(), Value::Public(v))))
            .collect();
        if args.first().is_none_or(|s| s != "models") {
            environment.insert(
                "MESH_LLM_TRAJECTORY_READER_BIN".into(),
                Value::Public(self.reader.ok_or("reader required")?.as_os_str().into()),
            );
        }
        let spec = ProcessSpec {
            executable: std::env::current_exe()?,
            arguments: args.into_iter().map(|a| Value::Public(a.into())).collect(),
            cwd: self.root.into(),
            environment,
        };
        Ok(process::supervise_raw_with_files(
            &spec,
            &Limits {
                execution,
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(1),
                retained_bytes_per_stream: 1024 * 1024,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            self.cancel,
            OutputFiles {
                stdout: Some(self.root.join(format!("{label}-stdout.log"))),
                stderr: Some(self.root.join(format!("{label}-stderr.log"))),
            },
            RawCaptureOptions {
                stdout: NonZeroUsize::new(1024 * 1024),
                stderr: NonZeroUsize::new(1024 * 1024),
            },
        )?)
    }
    pub(super) fn preflight(&self, evidence: &mut Json) -> DynResult<()> {
        let report = self.phase(
            vec![
                "automation".into(),
                "agentic-prompt-manifest".into(),
                "check-reader".into(),
            ],
            "reader-preflight",
        )?;
        evidence["reader_preflight_clean"] = json!(execution::clean(&report));
        if !execution::clean(&report) {
            return Err("native reader preflight failed before acquisition".into());
        }
        super::check(self.deadline, self.cancel)
    }
    pub(super) fn models(&self, input: &Json, evidence: &mut Json) -> DynResult<()> {
        let manifest = Path::new(
            input["model_manifest"]
                .as_str()
                .ok_or("pinned model manifest required")?,
        );
        let expected = input["model_manifest_sha256"]
            .as_str()
            .ok_or("manifest SHA required")?;
        let bytes = adaptive_identity::bounded(manifest, 8 * 1024 * 1024)?;
        if hex::encode(Sha256::digest(&bytes)) != expected {
            return Err("model manifest pin refused".into());
        }
        let config = Path::new(input["config"].as_str().ok_or("config required")?);
        let raw = adaptive_identity::bounded(config, 1024 * 1024)?;
        if json!(hex::encode(Sha256::digest(&raw))) != input["config_sha256"] {
            return Err("config pin refused".into());
        }
        let config: Json = serde_json::from_slice(&raw)?;
        let filter = input["model_keys"].as_array();
        let mut admitted = Vec::new();
        for row in config["models"].as_array().ok_or("models required")? {
            let key = row["key"].as_str().ok_or("model key")?;
            if filter.is_some_and(|f| !f.is_empty() && !f.iter().any(|v| v.as_str() == Some(key))) {
                continue;
            }
            let report = self.phase(
                vec![
                    "models".into(),
                    "resolve".into(),
                    manifest.to_str().ok_or("manifest Unicode")?.into(),
                    "--artifact-id".into(),
                    row["artifact_id"]
                        .as_str()
                        .ok_or("artifact ID required")?
                        .into(),
                    "--cadence".into(),
                    "manual".into(),
                    "--require-single-file".into(),
                ],
                &format!("model-authority-{}", admitted.len()),
            )?;
            if !execution::clean(&report) {
                return Err("native model authority resolution refused".into());
            }
            let selected: Json = serde_json::from_slice(
                report
                    .stdout
                    .as_ref()
                    .ok_or("model authority stdout")?
                    .as_bytes(),
            )?;
            for (native, configured) in [
                ("repo", "repo"),
                ("revision", "revision"),
                ("file", "filename"),
                ("sha256", "sha256"),
            ] {
                if selected[native] != row[configured] {
                    return Err("model config differs from native artifact authority".into());
                }
            }
            admitted.push(json!({"key":key,"artifact_id":row["artifact_id"],"selection":selected}));
        }
        if admitted.is_empty() {
            return Err("empty native model authority roster".into());
        }
        if adaptive_identity::bounded(manifest, 8 * 1024 * 1024)? != bytes
            || adaptive_identity::bounded(
                Path::new(input["config"].as_str().unwrap()),
                1024 * 1024,
            )? != raw
        {
            return Err("model authority source changed".into());
        }
        evidence["model_authority"] = json!({"manifest_sha256":expected,"resolved":admitted});
        super::check(self.deadline, self.cancel)
    }
    pub(super) fn materialize(&self, input: &Json, evidence: &mut Json) -> DynResult<()> {
        let config_path = Path::new(input["config"].as_str().ok_or("config required")?);
        let bytes = adaptive_identity::bounded(config_path, 1024 * 1024)?;
        if json!(hex::encode(Sha256::digest(&bytes))) != input["config_sha256"] {
            return Err("reader config pin mismatch".into());
        }
        let config: Json = serde_json::from_slice(&bytes)?;
        let output = Path::new(
            input["output_directory"]
                .as_str()
                .ok_or("output required")?,
        );
        let mut args = arguments(&config, output)?;
        args.splice(
            0..0,
            ["automation".into(), "agentic-prompt-manifest".into()],
        );
        let report = self.phase(args, "trajectory-manifest")?;
        evidence["trajectory_process_clean"] = json!(execution::clean(&report));
        if !execution::clean(&report) {
            return Err("native deterministic manifest producer failed".into());
        }
        let path = output.join("thoughtworks/manifest.json");
        let produced = adaptive_identity::bounded(&path, 8 * 1024 * 1024)?;
        let pin = hex::encode(Sha256::digest(&produced));
        evidence["manifest_sha256"] = json!(&pin);
        if json!(&pin) != config["thoughtworks"]["selection"]["manifest_sha256"] {
            return Err(
                "native manifest differs from original pinned deterministic selection".into(),
            );
        }
        super::pin(
            self.reader.ok_or("reader required")?,
            self.reader_sha256.ok_or("reader SHA required")?,
            self.deadline,
            self.cancel,
        )?;
        if adaptive_identity::bounded(config_path, 1024 * 1024)? != bytes {
            return Err("manifest config changed".into());
        }
        super::check(self.deadline, self.cancel)
    }
}
fn arguments(config: &Json, output: &Path) -> DynResult<Vec<String>> {
    let thought = &config["thoughtworks"];
    let dataset = &thought["dataset"];
    let selection = &thought["selection"];
    let file = dataset["filename"]
        .as_str()
        .filter(|s| !s.contains('/') && !s.contains('\\') && *s != "." && *s != "..")
        .ok_or("dataset filename refused")?;
    let revision = dataset["revision"].as_str().ok_or("dataset revision")?;
    let mut args = vec![
        "--dataset-file".into(),
        output
            .join("thoughtworks")
            .join(file)
            .to_str()
            .ok_or("dataset Unicode")?
            .into(),
        "--dataset-revision".into(),
        revision.into(),
        "--output".into(),
        output
            .join("thoughtworks/manifest.json")
            .to_str()
            .ok_or("manifest Unicode")?
            .into(),
    ];
    for (flag, key) in [
        ("--families", "families"),
        ("--requests-per-family", "requests_per_family"),
        ("--min-isl", "min_isl"),
        ("--max-isl", "max_isl_exclusive"),
        ("--min-turns", "min_turns"),
    ] {
        let n = selection[key].as_u64().ok_or("selection numeric field")?;
        args.extend([flag.into(), n.to_string()]);
    }
    for source in selection["sources"].as_array().ok_or("selection sources")? {
        args.extend([
            "--source-dataset".into(),
            source.as_str().ok_or("source dataset name")?.into(),
        ]);
    }
    Ok(args)
}
#[cfg(all(test, unix))]
mod tests {
    use super::*;
    #[test]
    fn native_trajectory_bridge_forwards_exact_original_selection_without_default_substitution() {
        let config = json!({"thoughtworks":{"dataset":{"filename":"sessions.parquet","revision":"a".repeat(40)},"selection":{"families":8,"requests_per_family":32,"min_isl":3072,"max_isl_exclusive":4096,"min_turns":5,"sources":["swe-smith-claude-3-7-sonnet"]}}});
        let args = arguments(&config, Path::new("/owned")).unwrap();
        assert_eq!(
            args,
            vec![
                "--dataset-file",
                "/owned/thoughtworks/sessions.parquet",
                "--dataset-revision",
                &"a".repeat(40),
                "--output",
                "/owned/thoughtworks/manifest.json",
                "--families",
                "8",
                "--requests-per-family",
                "32",
                "--min-isl",
                "3072",
                "--max-isl",
                "4096",
                "--min-turns",
                "5",
                "--source-dataset",
                "swe-smith-claude-3-7-sonnet"
            ]
        );
    }
}
