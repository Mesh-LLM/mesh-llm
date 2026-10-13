use super::manifest_preflight::{Manifest, Requirements};
use crate::command::DynResult;
use crate::repository::{check_args::Grammar, check_report::CheckReport};
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
pub(super) struct Input {
    pub manifest: PathBuf,
    pub requirements: Requirements,
    #[serde(default)]
    pub binary: PathBuf,
    #[serde(default)]
    pub native_runtime_root: PathBuf,
    pub external: Option<super::external_probe::Verified>,
    #[serde(default)]
    pub replay_mode: super::replay_profile::Mode,
    #[serde(default)]
    pub hf_home: Option<PathBuf>,
    pub model: String,
    pub label: String,
    #[serde(rename = "ref")]
    pub reference: String,
    pub commit: String,
    pub pass: u32,
    pub max_output_tokens: u64,
    pub request_timeout_seconds: u64,
    pub startup_timeout_seconds: u64,
    pub timeout_seconds: u64,
    pub port: u16,
    pub output: PathBuf,
    pub qualification: Option<Qualification>,
}

#[derive(Deserialize)]
pub(super) struct Qualification {
    pub model_sha256: String,
    pub minimum_context_tokens: u64,
    pub minimum_session_prompt_tokens: u64,
    pub require_recurrent_restores: bool,
    pub prompt_tokens_by_cohort:
        std::collections::BTreeMap<String, std::collections::BTreeMap<String, u64>>,
}

pub(in crate::automation) fn run(root: Option<&Path>, args: &[String]) -> DynResult<()> {
    const GRAMMAR: Grammar = Grammar {
        usage: "cargo xtool automation replay-matrix arm-pass --input PATH --output PATH",
        values: &["--input", "--output"],
        flags: &["--help"],
    };
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    if parsed.flag("--help") {
        return CheckReport::success(format!("{}\n", GRAMMAR.usage)).emit();
    }
    if !parsed.positionals.is_empty() {
        return GRAMMAR.error("unexpected positional arguments").emit();
    }
    let input: Input = serde_json::from_slice(&std::fs::read(
        parsed.last("--input").ok_or("missing --input")?,
    )?)?;
    let result_path = std::path::absolute(parsed.last("--output").ok_or("missing --output")?)?;
    execute(root, input, &result_path)
}

fn execute(root: Option<&Path>, input: Input, result_path: &Path) -> DynResult<()> {
    validate(&input)?;
    if result_path.try_exists()? || input.output.try_exists()? {
        return Err("arm-pass outputs already exist".into());
    }
    let manifest: Manifest = serde_json::from_slice(&std::fs::read(&input.manifest)?)?;
    super::manifest_preflight::validate(&manifest, &input.requirements)?;
    let started = std::time::Instant::now();
    let mut progress = super::progress::Record::boundary(
        super::progress::Phase::Pass,
        super::progress::Event::Started,
        format!("{}:pass-{}", input.label, input.pass),
    );
    super::progress::emit(&progress)?;
    let (workload, levels) = super::arm_pass_workload::prepare(&input, manifest)?;
    std::fs::create_dir_all(&input.output)?;
    let workload_path = input.output.join("workload.json");
    crate::command::write_json_file(&workload_path, &workload)?;
    let requests = input.output.join("warmup-requests.jsonl");
    let warmup_summary = input.output.join("warmup.json");
    let log = input.output.join(if input.external.is_some() {
        "server.log"
    } else {
        "mesh.log"
    });
    let execution = super::arm_pass_launch::execute(
        root,
        &input,
        &workload_path,
        &requests,
        &warmup_summary,
        &log,
    );
    let passed = super::arm_pass_result::retain(
        &input,
        (result_path, &log, &warmup_summary),
        &levels,
        &execution,
    )?;
    progress.event = super::progress::Event::Completed;
    progress.elapsed_seconds = started.elapsed().as_secs_f64();
    progress.outcome = Some(if passed {
        super::progress::Outcome::Success
    } else {
        super::progress::Outcome::Error
    });
    super::progress::emit(&progress)?;
    execution
}

fn validate(input: &Input) -> DynResult<()> {
    if input
        .hf_home
        .as_ref()
        .is_some_and(|home| !home.is_absolute())
        || (input.replay_mode != super::replay_profile::Mode::All && input.qualification.is_some())
    {
        return Err("selected replay profiles cannot claim full-session qualification".into());
    }
    if input.pass == 0
        || input.port == 0
        || !input.output.is_absolute()
        || input.label.is_empty()
        || [".", ".."].contains(&input.label.as_str())
        || !input
            .label
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"_.-".contains(&byte))
    {
        return Err(
            "arm pass requires a positive pass/port, safe label and absolute artifact directory"
                .into(),
        );
    }
    if let Some(build) = &input.external {
        if input.qualification.is_some() {
            return Err("runtime context qualification currently requires mesh arms".into());
        }
        if input.label != build.label
            || input.reference != build.reference
            || input.commit != build.commit
        {
            return Err("external arm pass identity does not match its verified build".into());
        }
    }
    Ok(())
}
