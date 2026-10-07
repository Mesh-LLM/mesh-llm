//! Optional MCP SDK preparation and source/environment custody, never benchmark qualification.
use super::*;
use crate::cli::EvalPrepareMcpArgs;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
#[path = "mcp_environment/custody.rs"]
mod custody;
#[cfg(test)]
#[path = "mcp_environment/tests.rs"]
mod tests;

const PREPARATION_SECONDS: u64 = 600;
const CAPTURE_LIMIT: u64 = 1048576;
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    schema_version: u8,
    source: String,
    uv: PathBuf,
    python: PathBuf,
    tool_pins: BTreeMap<PathBuf, String>,
    environment_pins: BTreeMap<PathBuf, String>,
    probe_modules: BTreeMap<String, PathBuf>,
    benchmark_qualified: bool,
}
fn base(root: &Path) -> PathBuf {
    root.join("mcp-sdk-v1")
}
pub(super) fn runtime_python(root: &Path) -> PathBuf {
    base(root).join("environment/bin/python")
}
fn project(root: &Path) -> PathBuf {
    harness_dir(root, registry::definition(EvalId::McpAtlas)).join("services/mcp_eval")
}

pub(super) fn admit(root: &Path) -> Result<()> {
    if !cfg!(unix) {
        bail!("MCP prepared SDK currently supports Unix hosts only");
    }
    let bytes = custody::read(&base(root).join("receipt.json"), 8 * 1048576)
        .context("MCP SDK is unprepared; run eval prepare-mcp explicitly")?;
    let receipt: Receipt = serde_json::from_slice(&bytes)?;
    if receipt.schema_version != 1
        || receipt.source != registry::MCP_ATLAS_REF
        || receipt.benchmark_qualified
    {
        bail!("MCP prepared receipt contract refused");
    }
    custody::source(&project(root))?;
    admit_environment(root, &receipt)
}
fn admit_environment(root: &Path, receipt: &Receipt) -> Result<()> {
    custody::tools(receipt)?;
    if custody::environment(&base(root).join("environment"), &receipt.python)?
        != receipt.environment_pins
    {
        bail!("MCP prepared environment changed");
    }
    validate_modules(root, &receipt.probe_modules)?;
    Ok(())
}
fn validate_modules(root: &Path, modules: &BTreeMap<String, PathBuf>) -> Result<()> {
    let site = base(root)
        .join("environment/lib/python3.12/site-packages")
        .canonicalize()?;
    let source = project(root).canonicalize()?;
    let expected = [
        "mcp_completion",
        "mcp_completion.main",
        "pandas",
        "numpy",
        "aiohttp",
        "aiofiles",
        "aiocsv",
        "requests",
        "dotenv",
        "datasets",
        "litellm",
        "tenacity",
        "tqdm",
        "matplotlib",
        "nest_asyncio",
        "uvicorn",
        "fastapi",
        "pydantic",
        "httpx",
    ];
    if modules.len() != expected.len() || expected.iter().any(|name| !modules.contains_key(*name)) {
        bail!("MCP probe module roster differs");
    }
    for (name, path) in modules {
        let actual = path.canonicalize()?;
        let admitted = if name == "mcp_completion" {
            source.join("mcp_completion/__init__.py")
        } else if name == "mcp_completion.main" {
            source.join("mcp_completion/main.py")
        } else {
            if !actual.starts_with(&site) {
                bail!("MCP SDK import escaped environment");
            }
            actual.clone()
        };
        if actual != admitted {
            bail!("MCP SDK package location differs");
        }
    }
    Ok(())
}
fn isolated(spec: CommandSpec, root: &Path) -> CommandSpec {
    spec.isolated()
        .env("PATH", "/usr/bin:/bin")
        .env("HOME", base(root).join("home").display().to_string())
        .env("TMPDIR", base(root).join("tmp").display().to_string())
        .env(
            "UV_CACHE_DIR",
            base(root).join("cache").display().to_string(),
        )
        .env(
            "UV_PROJECT_ENVIRONMENT",
            base(root).join("environment").display().to_string(),
        )
        .env("UV_PYTHON_DOWNLOADS", "never")
}
fn execute(spec: &CommandSpec, root: &Path, phase: &str, deadline: Instant) -> Result<String> {
    let remaining = deadline.saturating_duration_since(Instant::now());
    let execution = remaining
        .checked_sub(Duration::from_secs(6))
        .filter(|d| !d.is_zero())
        .context("MCP preparation cleanup reserve unavailable")?;
    let stdout = base(root).join(format!("logs/{phase}.stdout"));
    let stderr = base(root).join(format!("logs/{phase}.stderr"));
    // Dependency: corrected eval process owner, with finite capture polling and unconditional cleanup.
    let outcome = capture(spec, execution, &stdout, &stderr)?;
    if !outcome.success || outcome.timed_out || Instant::now() >= deadline {
        bail!("MCP preparation phase {phase} failed; logs retained");
    }
    Ok(String::from_utf8(custody::read(&stdout, CAPTURE_LIMIT)?)?)
}
fn capture(
    spec: &CommandSpec,
    execution: Duration,
    stdout: &Path,
    stderr: &Path,
) -> Result<CommandOutcome> {
    super::sdk_environment::capture(spec, execution, stdout, stderr)
}

fn sync_spec(root: &Path, uv: &Path, python: &Path) -> CommandSpec {
    isolated(
        CommandSpec::new(uv.display().to_string()).args([
            "--no-config".to_owned(),
            "sync".to_owned(),
            "--locked".to_owned(),
            "--project".to_owned(),
            project(root).display().to_string(),
            "--python".to_owned(),
            python.display().to_string(),
            "--no-python-downloads".to_owned(),
        ]),
        root,
    )
}
pub(super) fn prepare(args: EvalPrepareMcpArgs) -> Result<()> {
    let root = absolute_path(cache_root(args.cache_root)?)?;
    if !cfg!(unix) {
        bail!("MCP preparation currently supports Unix hosts only");
    }
    if [&root, &args.uv, &args.python]
        .iter()
        .any(|p| p.to_str().is_none_or(|s| s.chars().any(char::is_control)))
    {
        bail!("MCP paths need control-free UTF-8");
    }
    if !args.uv.is_absolute() || !args.python.is_absolute() {
        bail!("MCP preparation needs explicit absolute uv/Python paths");
    }
    if args.dry_run {
        println!(
            "{} (pending preparation; unqualified)",
            sync_spec(&root, &args.uv, &args.python).display()
        );
        return Ok(());
    }
    harness_source::admit_run(&root, registry::definition(EvalId::McpAtlas))?;
    custody::source(&project(&root))?;
    let uv = args.uv.canonicalize()?;
    let python = args.python.canonicalize()?;
    let tool_pins = custody::capture_tools(&uv, &python)?;
    let destination = base(&root);
    fs::create_dir(&destination).context("MCP preparation destination must be fresh; preserve failed evidence or choose another cache root")?;
    for name in [
        "home",
        "tmp",
        "cache",
        "logs",
        "probe",
        "probe/completion_results",
        "probe/mpl",
    ] {
        fs::create_dir_all(destination.join(name))?;
    }
    let deadline = Instant::now() + Duration::from_secs(PREPARATION_SECONDS);
    let version = execute(
        &isolated(
            CommandSpec::new(python.display().to_string()).args(["-I", "-B", "--version"]),
            &root,
        ),
        &root,
        "python-version",
        deadline,
    )?;
    if !version.trim().starts_with("Python 3.12.") {
        bail!("MCP prepared profile requires existing CPython3.12");
    }
    execute(
        &isolated(
            CommandSpec::new(uv.display().to_string()).args(["--version"]),
            &root,
        ),
        &root,
        "uv-version",
        deadline,
    )?;
    execute(
        &sync_spec(&root, &uv, &python),
        &root,
        "locked-sync",
        deadline,
    )?;
    let modules = probe(&root, deadline)?;
    validate_modules(&root, &modules)?;
    let receipt = Receipt {
        schema_version: 1,
        source: registry::MCP_ATLAS_REF.into(),
        uv,
        python: python.clone(),
        tool_pins,
        environment_pins: custody::environment(&destination.join("environment"), &python)?,
        probe_modules: modules,
        benchmark_qualified: false,
    };
    harness_source::admit_run(&root, registry::definition(EvalId::McpAtlas))?;
    custody::source(&project(&root))?;
    admit_environment(&root, &receipt)?;
    if Instant::now() >= deadline {
        bail!("MCP preparation terminal deadline");
    }
    let bytes = serde_json::to_vec_pretty(&receipt)?;
    if bytes.len() > 8 * 1048576 {
        bail!("MCP receipt exceeds finite bound");
    }
    use std::io::Write as _;
    let mut output = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(destination.join("receipt.pending"))?;
    output.write_all(&bytes)?;
    output.sync_all()?;
    fs::rename(
        destination.join("receipt.pending"),
        destination.join("receipt.json"),
    )?;
    println!(
        "MCP SDK prepared at {}; benchmark qualification remains false",
        destination.display()
    );
    Ok(())
}
fn probe(root: &Path, deadline: Instant) -> Result<BTreeMap<String, PathBuf>> {
    // Fixed SDK import leaf, no task main, dataset load, service listener or model request.
    let completion = serde_json::to_string(
        &project(root)
            .join("mcp_completion_script.py")
            .display()
            .to_string(),
    )?;
    let scorer = serde_json::to_string(
        &project(root)
            .join("mcp_evals_scores.py")
            .display()
            .to_string(),
    )?;
    let code = format!(
        "import sys,json,runpy,mcp_completion,mcp_completion.main; runpy.run_path({completion},run_name='_mcp_import_only'); runpy.run_path({scorer},run_name='_mcp_import_only'); names=['mcp_completion','mcp_completion.main','pandas','numpy','aiohttp','aiofiles','aiocsv','requests','dotenv','datasets','litellm','tenacity','tqdm','matplotlib','nest_asyncio','uvicorn','fastapi','pydantic','httpx']; print(json.dumps({{n:sys.modules[n].__file__ for n in names}}))"
    );
    let spec = isolated(
        CommandSpec::new(runtime_python(root).display().to_string()).args([
            "-I".into(),
            "-B".into(),
            "-c".into(),
            code,
        ]),
        root,
    )
    .cwd(base(root).join("probe"))
    .env("PYTHON_DOTENV_DISABLED", "1")
    .env("HF_HUB_OFFLINE", "1")
    .env("HF_DATASETS_OFFLINE", "1")
    .env("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    .env("MPLBACKEND", "Agg")
    .env(
        "MPLCONFIGDIR",
        base(root).join("probe/mpl").display().to_string(),
    );
    Ok(serde_json::from_str(
        execute(&spec, root, "sdk-import-only", deadline)?.trim(),
    )?)
}
