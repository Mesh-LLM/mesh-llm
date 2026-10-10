//! Serial generic matrix execution with complete rosters and correlation-checked resume.
use crate::{
    command::DynResult,
    process::{self, Value as Argument},
};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
#[derive(Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    #[serde(default)]
    pub source_context: Option<super::competitive_prepare::SourceContext>,
    pub config: PathBuf,
    pub config_sha256: String,
    pub platform: String,
    pub models: Vec<super::competitive_roster::Model>,
    pub workloads: Vec<String>,
    pub optional_arms: Vec<String>,
    pub required_comparisons: Vec<String>,
    pub adaptive: bool,
    pub manifest: Option<PathBuf>,
    pub benchy: Option<super::competitive_launch::Artifact>,
    pub output: PathBuf,
    pub timeout_seconds: u64,
    pub cell_timeout_seconds: u64,
    pub request_timeout_seconds: u64,
    pub resume: bool,
    pub force: bool,
}
pub(in crate::automation) fn run(args: &[String]) -> DynResult<()> {
    use crate::repository::{check_args::Grammar, check_report::CheckReport};
    const GRAMMAR: Grammar = Grammar {
        usage: "cargo xtool automation replay-matrix competitive-run --input PATH",
        values: &["--input"],
        flags: &["--help"],
    };
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    if parsed.flag("--help") {
        return CheckReport::success(format!(
            "{}
",
            GRAMMAR.usage
        ))
        .emit();
    }
    if !parsed.positionals.is_empty() {
        return GRAMMAR.error("unexpected positional arguments").emit();
    }
    let started = Instant::now();
    let bytes = super::competitive_cell::read(
        Path::new(parsed.last("--input").ok_or("input")?),
        8 * 1024 * 1024,
    )?;
    let input: Input = serde_json::from_slice(&bytes)?;
    execute(&input, started)
}
fn execute(input: &Input, started: Instant) -> DynResult<()> {
    let (config, config_bytes, roster) = admit(input)?;
    if input.output.exists() {
        if !input.resume && !input.force || !std::fs::symlink_metadata(&input.output)?.is_dir() {
            return Err(
                "existing matrix output requires explicit resume/force and regular owned directory"
                    .into(),
            );
        }
    } else {
        std::fs::create_dir(&input.output)?;
    }
    let lease = Lease::acquire(&input.output)?;
    preflight(
        &config,
        input,
        started + Duration::from_secs(input.timeout_seconds),
    )?;
    let invocation = input.output.join(format!(
        "invocation-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));
    std::fs::create_dir(&invocation)?;
    let plan = json!({"schema_version":2,"scope":"competitive_native_matrix","config_sha256":input.config_sha256,"source_context":input.source_context,"platform":input.platform,"planner_linux":cfg!(target_os="linux"),"selection":{"workloads":input.workloads,"optional_arms":input.optional_arms,"required_comparisons":input.required_comparisons,"adaptive":input.adaptive},"cells":roster.cells,"availability":roster.availability,"prepared_inputs":input.models,"manifest":input.manifest,"benchy":input.benchy});
    bind_source(input, &config_bytes)?;
    bind_plan(input, &plan)?;
    let deadline = started + Duration::from_secs(input.timeout_seconds);
    let mut accepted = Vec::new();
    for (index, cell) in roster.cells.iter().enumerate() {
        let model = input
            .models
            .iter()
            .find(|model| cell["model"] == model.key)
            .ok_or("selected model")?;
        let provenance = super::competitive_resume::provenance(&config, cell, model)?;
        let directory = cell_directory(&input.output, cell)?;
        ensure_directory(&input.output, directory.parent().ok_or("cell parent")?)?;
        if let Ok(metadata) = std::fs::symlink_metadata(&directory)
            && !metadata.is_dir()
        {
            return Err("competitive cell path must be an owned regular directory".into());
        }
        if input.resume
            && super::competitive_resume::completed(
                &directory,
                cell,
                &input.config_sha256,
                &provenance,
            )?
        {
            accepted.push(json!({"cell":cell,"directory":directory,"resumed":true}));
            continue;
        }
        if directory.exists() {
            if input.force {
                super::competitive_resume::quarantine(&input.output, &directory)?;
            } else {
                return Err(
                    "incomplete cell cannot be resumed; explicit force preserves it in quarantine"
                        .into(),
                );
            }
        }
        ensure_directory(&input.output, directory.parent().ok_or("cell parent")?)?;
        let budget = deadline
            .saturating_duration_since(Instant::now())
            .checked_sub(Duration::from_secs(3))
            .ok_or("matrix lacks final child cleanup reserve")?
            .min(Duration::from_secs(input.cell_timeout_seconds));
        if budget < Duration::from_secs(10) {
            return Err("matrix deadline exhausted before next cell".into());
        }
        let prepared = cell_input(input, model, cell, &directory, budget)?;
        let path = invocation.join(format!("cell-input-{index}.json"));
        super::competitive_synthetic::write_new(&path, &prepared)?;
        supervise_cell(input, &path, budget)?;
        if !super::competitive_resume::completed(
            &directory,
            cell,
            &input.config_sha256,
            &provenance,
        )? {
            return Err("child exited without correlated complete cell".into());
        }
        accepted.push(json!({"cell":cell,"directory":directory,"resumed":false}));
        crate::command::write_json_file(
            &input.output.join("matrix-progress.json"),
            &json!({"completed":false,"accepted":accepted,"planned_cells":roster.cells.len()}),
        )?;
    }
    if Instant::now() >= deadline {
        return Err("matrix deadline reached; full completion withheld".into());
    }
    crate::command::write_json_file(
        &input.output.join("matrix-results.json"),
        &json!({"schema_version":1,"scope":"competitive_matrix_cell_execution","completed":true,"config_sha256":input.config_sha256,"cells":accepted,"availability":roster.availability}),
    )?;
    let reporting = super::competitive_report::write(
        &input.output,
        &roster.cells,
        &input.config_sha256,
        deadline,
        false,
    );
    drop(lease);
    reporting
}
fn cell_input(
    input: &Input,
    model: &super::competitive_roster::Model,
    cell: &Value,
    directory: &Path,
    budget: Duration,
) -> DynResult<Value> {
    Ok(
        json!({"config":input.config,"config_sha256":input.config_sha256,"cell":cell,"model":model.model,"backend":model.backends.get(cell["arm"].as_str().ok_or("arm")?).ok_or("backend")?,"manifest":input.manifest,"benchy":input.benchy,"output":directory,"timeout_seconds":budget.as_secs(),"request_timeout_seconds":input.request_timeout_seconds.min(budget.as_secs()-1)}),
    )
}
fn supervise_cell(input: &Input, path: &Path, budget: Duration) -> DynResult<()> {
    let spec = process::ProcessSpec {
        executable: std::env::current_exe()?,
        arguments: [
            "automation",
            "replay-matrix",
            "competitive-run-cell",
            "--input",
        ]
        .into_iter()
        .map(|value| Argument::Public(value.into()))
        .chain(std::iter::once(Argument::Public(
            path.as_os_str().to_owned(),
        )))
        .collect(),
        cwd: input.output.clone(),
        environment: super::competitive_launch::environment(None, false),
    };
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let result = process::supervise_raw(
        &spec,
        &super::competitive_run_cell::limits(budget),
        &interrupt.cancellation(),
        process::RawCaptureOptions {
            stdout: std::num::NonZeroUsize::new(8 * 1024 * 1024),
            stderr: std::num::NonZeroUsize::new(8 * 1024 * 1024),
        },
    );
    let finish = interrupt.finish();
    let report = result?;
    finish?;
    std::fs::write(
        path.with_extension("stdout.log"),
        report.stdout.as_ref().ok_or("cell stdout")?.as_bytes(),
    )?;
    std::fs::write(
        path.with_extension("stderr.log"),
        report.stderr.as_ref().ok_or("cell stderr")?.as_bytes(),
    )?;
    if !report.process.success() {
        return Err(
            "competitive child failed or required forced/incomplete cleanup; evidence retained"
                .into(),
        );
    }
    Ok(())
}
fn bind_plan(input: &Input, plan: &Value) -> DynResult<()> {
    let path = input.output.join("matrix-plan.json");
    if path.exists() {
        let previous: Value =
            serde_json::from_slice(&super::competitive_cell::read(&path, 64 * 1024 * 1024)?)?;
        if previous != *plan {
            return Err("refusing to mix a changed competitive plan/availability".into());
        }
    } else {
        super::competitive_synthetic::write_new(&path, plan)?;
    }
    Ok(())
}
fn safe(component: &str) -> DynResult<()> {
    if component.is_empty()
        || component.len() > 256
        || [".", ".."].contains(&component)
        || !component
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"_.-".contains(&byte))
    {
        return Err("matrix model/arm path must be a safe component".into());
    }
    Ok(())
}
pub(super) fn cell_directory(root: &Path, cell: &Value) -> DynResult<PathBuf> {
    let model = cell["model"].as_str().ok_or("model")?;
    let arm = cell["arm"].as_str().ok_or("arm")?;
    safe(model)?;
    safe(arm)?;
    let workload = cell["workload"].as_str().ok_or("workload")?;
    if !["synthetic", "thoughtworks"].contains(&workload) {
        return Err("unreviewed cell workload".into());
    }
    if !["cuda", "metal", "rocm"].contains(&cell["platform"].as_str().ok_or("platform")?) {
        return Err("unreviewed cell platform".into());
    }
    let concurrency = cell["concurrency"].as_u64().ok_or("concurrency")?;
    let output = cell["output_tokens"].as_u64().ok_or("output")?;
    Ok(root
        .join("cells")
        .join(model)
        .join(workload)
        .join(format!("tg-{output}-c-{concurrency}"))
        .join(arm))
}
struct Lease(PathBuf);
impl Lease {
    fn acquire(root: &Path) -> DynResult<Self> {
        let path = root.join(".matrix-active");
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        use std::io::Write;
        writeln!(file, "{}", std::process::id())?;
        Ok(Self(path))
    }
}
impl Drop for Lease {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

pub(super) fn preflight(config: &Value, input: &Input, deadline: Instant) -> DynResult<()> {
    if let Some(context) = &input.source_context {
        if config["baseline"]["llama_cpp_revision"] != context.llama_head {
            return Err("prepared raw source differs from configured baseline".into());
        }
        super::competitive_prepare::check_context(context, deadline)?;
    }
    for model in &input.models {
        let source = config["models"]
            .as_array()
            .ok_or("models")?
            .iter()
            .find(|source| source["key"] == model.key)
            .ok_or("source")?;
        if super::competitive_launch::file(&model.model)?
            != source["sha256"].as_str().ok_or("model pin")?
        {
            return Err("model pin differs".into());
        }
        for backend in model.backends.values() {
            super::competitive_launch::file(&backend.executable)?;
            for artifact in [
                &backend.runtime,
                &backend.tokenizer,
                &backend.comparison_model,
            ]
            .into_iter()
            .flatten()
            {
                super::competitive_launch::tree(artifact)?;
            }
            if let Some(tokenizer) = &backend.tokenizer
                && source["tokenizer_sha256"].as_str() != Some(tokenizer.sha256.as_str())
            {
                return Err("tokenizer source differs".into());
            }
            if let Some(config) = &backend.hf_config {
                super::competitive_launch::hf_config_directory(config)?;
                if source["vllm_hf_config"]["sha256"].as_str() != Some(config.sha256.as_str()) {
                    return Err("HF config source differs".into());
                }
            }
        }
    }
    if let Some(benchy) = &input.benchy {
        super::competitive_launch::file(benchy)?;
    }
    if input.workloads.iter().any(|value| value == "thoughtworks") {
        let manifest = input.manifest.as_ref().ok_or("trace manifest")?;
        let bytes = super::competitive_cell::read(manifest, 64 * 1024 * 1024)?;
        if config["thoughtworks"]["selection"]["manifest_sha256"].as_str()
            != Some(hex::encode(Sha256::digest(bytes)).as_str())
        {
            return Err("trace manifest differs".into());
        }
    }
    Ok(())
}

pub(super) fn ensure_directory(root: &Path, path: &Path) -> DynResult<()> {
    let mut current = root.to_path_buf();
    for component in path.strip_prefix(root)?.components() {
        if !matches!(component, std::path::Component::Normal(_)) {
            return Err("owned output path cannot escape root".into());
        }
        current.push(component);
        match std::fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.is_dir() => (),
            Ok(_) => return Err("owned output parent refuses links/special files".into()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                std::fs::create_dir(&current)?
            }
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

fn bind_source(input: &Input, bytes: &[u8]) -> DynResult<()> {
    let path = input.output.join("benchmark-config.source.json");
    if path.exists() {
        let previous = super::competitive_cell::read(&path, 8 * 1024 * 1024)?;
        if previous != bytes {
            return Err("refusing changed source configuration snapshot".into());
        }
    } else {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        use std::io::Write;
        file.write_all(bytes)?;
        file.flush()?;
    }
    Ok(())
}

pub(super) fn admit(
    input: &Input,
) -> DynResult<(Value, Vec<u8>, super::competitive_roster::Roster)> {
    if !(13..=86400).contains(&input.timeout_seconds)
        || !(10..=86400).contains(&input.cell_timeout_seconds)
        || input.request_timeout_seconds == 0
        || input.request_timeout_seconds >= input.cell_timeout_seconds
        || input.resume && input.force
        || !input.output.is_absolute()
    {
        return Err(
            "invalid competitive matrix budgets/output or mutually exclusive resume/force".into(),
        );
    }
    let config_bytes = super::competitive_cell::read(&input.config, 8 * 1024 * 1024)?;
    if hex::encode(Sha256::digest(&config_bytes)) != input.config_sha256 {
        return Err("matrix config pin differs".into());
    }
    let config: Value = serde_json::from_slice(&config_bytes)?;
    let roster = super::competitive_roster::select(&config, &config_bytes, input)?;
    for model in &input.models {
        safe(&model.key)?;
    }
    Ok((config, config_bytes, roster))
}

pub(super) fn existing_parents(root: &Path, path: &Path) -> DynResult<()> {
    if !std::fs::symlink_metadata(root)?.is_dir() {
        return Err("archive root must be regular directory".into());
    }
    let mut current = root.to_path_buf();
    for component in path.strip_prefix(root)?.components() {
        if !matches!(component, std::path::Component::Normal(_)) {
            return Err("archive path escapes root".into());
        }
        current.push(component);
        match std::fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.is_dir() => (),
            Ok(_) => return Err("archive parent refuses symlink/special files".into()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "competitive_matrix_tests.rs"]
mod tests;
