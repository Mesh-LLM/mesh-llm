use super::{l3_contract::Run, l3_execution::DiskRoots, run_workload::BuildJob};
use crate::{
    command::DynResult,
    repository::{check_args::Grammar, check_report::CheckReport},
};
use std::{path::Path, time::Duration};
const GRAMMAR: Grammar = Grammar {
    usage: "cargo xtool automation replay-matrix l3-run --ref LABEL=GIT_REF --model MODEL --trajectory-manifest PATH --output PATH --require-source-dataset SOURCE... [--skip-build]",
    values: &[
        "--repo",
        "--ref",
        "--model",
        "--backend",
        "--trajectory-manifest",
        "--lifecycle-cohort",
        "--require-source-dataset",
        "--concurrency",
        "--prompt-token-range",
        "--cold-samples",
        "--restart-samples",
        "--identical-repeats",
        "--max-output-tokens",
        "--disk-budget",
        "--low-space-disk-budget",
        "--minimum-free",
        "--low-space-minimum-free",
        "--max-l3-ttft-ratio",
        "--max-payload-write-amplification",
        "--max-decode-p99-regression-pct",
        "--output",
        "--worktree-root",
        "--hf-home",
        "--startup-timeout",
        "--request-timeout",
    ],
    flags: &["--help", "--skip-build"],
};
fn duration(value: &str) -> DynResult<Duration> {
    let seconds = value.parse::<u64>()?;
    if !(1..=86400).contains(&seconds) {
        return Err("disk-L3 timeout must be in 1..=86400 seconds".into());
    }
    Ok(Duration::from_secs(seconds))
}
pub(in crate::automation) fn run(root: Option<&Path>, args: &[String]) -> DynResult<()> {
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
    let config = super::l3_plan::config(&parsed)?;
    config.validate(true)?;
    let repo =
        crate::repository::RepositoryRoot::resolve(parsed.last("--repo").map(Path::new).or(root))?;
    let reference = parsed.last("--ref").ok_or("missing --ref")?;
    let (label, reference) = reference
        .split_once('=')
        .ok_or("ref must be LABEL=GIT_REF")?;
    if label.is_empty() || reference.is_empty() || reference.starts_with('-') {
        return Err("invalid ref specification".into());
    }
    let output = std::path::absolute(parsed.last("--output").ok_or("missing --output")?)?;
    let source = Path::new(
        parsed
            .last("--trajectory-manifest")
            .ok_or("missing --trajectory-manifest")?,
    )
    .canonicalize()?;
    let startup = duration(parsed.last("--startup-timeout").unwrap_or("1800"))?;
    let request_timeout = duration(parsed.last("--request-timeout").unwrap_or("900"))?;
    let hf_home = parsed
        .last("--hf-home")
        .map(std::path::absolute)
        .transpose()?;
    let worktree_root = parsed
        .last("--worktree-root")
        .map(std::path::absolute)
        .transpose()?
        .unwrap_or_else(|| {
            repo.as_path()
                .parent()
                .unwrap_or(repo.as_path())
                .join(".agentic-replay-worktrees")
        });
    std::fs::create_dir(&output)?;
    let plan = super::l3_plan::document(Some(repo.as_path()), &parsed)?;
    crate::command::write_json_file(&output.join("plan.json"), &plan)?;
    let (manifest, selected, inputs) = super::l3_manifest::import(&source, &output, &config)?;
    let job = BuildJob {
        repo: repo.as_path().into(),
        worktree_root,
        label: label.into(),
        reference: reference.into(),
        backend: config.backend.clone(),
        git: super::executable_resolution::tool("git")?,
        just: super::executable_resolution::tool("just")?,
        timeout_seconds: 86400,
        logs: output.join("logs"),
        skip_build: parsed.flag("--skip-build"),
    };
    let build = super::run_builds::prepare(Some(repo.as_path()), &[job], &output)?
        .pop()
        .ok_or("missing disk-L3 build")?;
    let build = match build {
        super::run_workload::Build::Mesh(build) => build,
        super::run_workload::Build::External(_) => {
            return Err("disk-L3 certification requires a Mesh build".into());
        }
    };
    let plan_sha256 = super::cohort_identity::digest(&plan)?;
    let mut run = Run {
        schema_version: 1,
        kind: "disk-l3-lifecycle".into(),
        config: config.clone(),
        build: serde_json::to_value(&build)?,
        inputs,
        phases: Default::default(),
        completed_at: None,
        gates: None,
        evidence: Default::default(),
    };
    run.evidence
        .insert("plan_sha256".into(), plan_sha256.into());
    run.evidence
        .insert("owner".into(), "rust-disk-l3-v1".into());
    run.evidence.insert(
        "started_at".into(),
        crate::ci_operations::ci_metrics_time::isoformat(
            crate::ci_operations::ci_metrics_time::Instant::now(),
        )
        .into(),
    );
    run.evidence.insert("host".into(),serde_json::json!({"hostname":super::l3_identity::hostname()?,"platform":std::env::consts::OS}));
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let cancellation = interrupt.cancellation();
    let mut roots = DiskRoots::create(&std::env::temp_dir())?;
    let result = std::thread::scope(|scope| -> DynResult<()> {
        let mut driver = super::l3_driver::Runtime::new(
            scope,
            super::l3_driver::Input {
                config,
                build,
                manifest,
                selected,
                output: output.clone(),
                hf_home,
                startup,
                request_timeout,
                execution: Duration::from_secs(86400),
                cancellation,
            },
        );
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        let result = runtime.block_on(async {
            tokio::time::timeout(
                Duration::from_secs(86400),
                super::l3_execution::execute(&mut driver, &mut roots, &mut run),
            )
            .await
        });
        match result {
            Ok(result) => result,
            Err(_) => {
                runtime.block_on(super::l3_execution::Driver::stop(&mut driver))?;
                crate::command::write_json_file(&output.join("run.json"), &run)?;
                Err("disk-L3 certification execution deadline exceeded".into())
            }
        }
    });
    let interrupted = interrupt.finish();
    let cleanup = roots.finish();
    let report = if run.gates.is_some() {
        Some(super::l3_report::write(&output, &run)?)
    } else {
        None
    };
    interrupted?;
    cleanup?;
    result?;
    CheckReport::success(format!(
        "{}\n",
        report.ok_or("missing disk-L3 report")?.display()
    ))
    .emit()
}
