//! Local prepared-layout adapter; checkout/byte observations do not attest build custody.
use crate::command::DynResult;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    config: PathBuf,
    platform: String,
    model_keys: Vec<String>,
    workloads: Vec<String>,
    model_root: PathBuf,
    tokenizer_root: PathBuf,
    mesh_root: PathBuf,
    llama_root: PathBuf,
    mesh_binary: PathBuf,
    llama_binary: PathBuf,
    native_runtime: PathBuf,
    manifest: Option<PathBuf>,
    benchy: Option<PathBuf>,
    optional_backends: BTreeMap<String, Optional>,
    required_comparisons: Vec<String>,
    adaptive: bool,
    output: PathBuf,
    prepared_output: PathBuf,
    timeout_seconds: u64,
    cell_timeout_seconds: u64,
    request_timeout_seconds: u64,
    preparation_timeout_seconds: u64,
    resume: bool,
    force: bool,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Optional {
    executable: PathBuf,
    cwd: PathBuf,
    hf_config_root: Option<PathBuf>,
    comparison_model_root: Option<PathBuf>,
    match_kv_capacity: bool,
}
#[derive(Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SourceContext {
    pub mesh_root: PathBuf,
    pub mesh_head: String,
    pub llama_root: PathBuf,
    pub llama_head: String,
    pub custody_scope: String,
}
pub(super) fn check_context(context: &SourceContext, deadline: Instant) -> DynResult<()> {
    if context.custody_scope != "local_checkout_and_artifact_observation_no_build_attestation" {
        return Err("unknown prepared source custody scope".into());
    }
    for (root, expected) in [
        (&context.mesh_root, &context.mesh_head),
        (&context.llama_root, &context.llama_head),
    ] {
        if head(root, deadline)? != *expected {
            return Err("prepared checkout HEAD changed".into());
        }
    }
    Ok(())
}
fn head(root: &Path, deadline: Instant) -> DynResult<String> {
    if !root.is_absolute() {
        return Err("prepared checkout root must be absolute".into());
    }
    let head = super::git_head::git_head_with_budget(
        root,
        deadline
            .saturating_duration_since(Instant::now())
            .checked_sub(Duration::from_secs(3))
            .filter(|budget| !budget.is_zero())
            .ok_or("prepared source lacks cleanup budget")?
            .min(Duration::from_secs(5)),
    )?
    .trim_end_matches(['\r', '\n'])
    .to_owned();
    if head.len() != 40
        || !head
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err("prepared checkout SHA is not canonical".into());
    }
    Ok(head)
}
fn file(path: &Path) -> DynResult<Value> {
    if !path.is_absolute() {
        return Err("prepared file path must be absolute".into());
    }
    let canonical = path.canonicalize()?;
    let path = canonical.as_path();
    if !std::fs::symlink_metadata(path)?.is_file() {
        return Err("prepared input is not a regular file".into());
    }
    let digest = crate::product::digest::file_sha256(path).map_err(|error| error.error)?;
    Ok(json!({"path":path,"sha256":digest}))
}
fn tree(path: &Path) -> DynResult<Value> {
    if !path.is_absolute() {
        return Err("prepared tree path must be absolute".into());
    }
    let canonical = path.canonicalize()?;
    let path = canonical.as_path();
    let artifact = super::competitive_launch::Artifact {
        path: path.into(),
        sha256: crate::product::digest::tree_sha256(path).map_err(|error| error.error)?,
    };
    let digest = super::competitive_launch::tree(&artifact)?;
    Ok(json!({"path":path,"sha256":digest}))
}
fn backend(
    executable: &Path,
    cwd: &Path,
    runtime: Option<Value>,
    deadline: Instant,
    name: &str,
) -> DynResult<Value> {
    let executable = file(executable)?;
    let mut backend: super::competitive_launch::Backend = serde_json::from_value(
        json!({"executable":executable,"version_sha256":"0".repeat(64),"cwd":cwd,"runtime":runtime,"tokenizer":null,"hf_config":null,"comparison_model":null,"match_kv_capacity":false}),
    )?;
    let version = super::competitive_run_cell::observe_version(&backend, name, deadline)?;
    backend.version_sha256 = hex::encode(Sha256::digest(version.as_bytes()));
    // Probe execution is an observation; recheck actual bytes after it completes.
    super::competitive_launch::file(&backend.executable)?;
    serde_json::to_value(backend).map_err(Into::into)
}
fn models(
    request: &Request,
    config: &Value,
    raw: &Value,
    mesh: &Value,
    deadline: Instant,
) -> DynResult<Vec<Value>> {
    let sources = config["models"].as_array().ok_or("configured models")?;
    let mut result = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for key in &request.model_keys {
        if !seen.insert(key) {
            return Err("duplicate prepared model".into());
        }
        component(key)?;
        let source = sources
            .iter()
            .find(|source| source["key"] == *key)
            .ok_or("unknown prepared model")?;
        component(source["filename"].as_str().ok_or("model filename")?)?;
        let artifact = file(
            &request
                .model_root
                .join(key)
                .join(source["filename"].as_str().ok_or("model filename")?),
        )?;
        if artifact["sha256"] != source["sha256"] {
            return Err("prepared model source pin differs".into());
        }
        let tokenizer = if request.workloads.iter().any(|w| w == "synthetic")
            || (cfg!(target_os = "linux")
                && request.platform == "cuda"
                && request
                    .optional_backends
                    .keys()
                    .any(|name| source["comparison_support"][name]["available"] != false))
        {
            Some(tree(&request.tokenizer_root.join(key))?)
        } else {
            None
        };
        if tokenizer
            .as_ref()
            .is_some_and(|t| t["sha256"] != source["tokenizer_sha256"])
        {
            return Err("prepared tokenizer source pin differs".into());
        }
        let mut backends = BTreeMap::from([
            ("llama".to_owned(), raw.clone()),
            ("mesh".to_owned(), mesh.clone()),
        ]);
        if request.adaptive {
            backends.insert("mesh-adaptive".into(), mesh.clone());
        }
        for value in backends.values_mut() {
            value["tokenizer"] = json!(tokenizer);
        }
        for (name, options) in &request.optional_backends {
            if !["vllm", "sglang"].contains(&name.as_str()) {
                return Err("unknown optional prepared backend".into());
            }
            if !cfg!(target_os = "linux")
                || request.platform != "cuda"
                || source["comparison_support"][name]["available"] == false
            {
                continue;
            }
            let mut value = backend(&options.executable, &options.cwd, None, deadline, name)?;
            value["tokenizer"] = json!(tokenizer);
            value["match_kv_capacity"] = options.match_kv_capacity.into();
            if let Some(root) = &options.hf_config_root {
                value["hf_config"] = file(&root.join(key).join("config.json"))?;
            }
            if let Some(root) = &options.comparison_model_root {
                value["comparison_model"] = tree(&root.join(key))?;
            }
            backends.insert(name.clone(), value);
        }
        result.push(json!({"key":key,"model":artifact,"backends":backends}));
    }
    Ok(result)
}
fn prepare(request: &Request) -> DynResult<()> {
    if !(30..=3600).contains(&request.preparation_timeout_seconds) {
        return Err("prepared adapter budget must be30..3600 seconds".into());
    }
    if std::fs::symlink_metadata(&request.prepared_output).is_ok() {
        return Err("prepared output already exists".into());
    }
    let deadline = Instant::now() + Duration::from_secs(request.preparation_timeout_seconds);
    let bytes = super::competitive_cell::read(&request.config, 8 * 1024 * 1024)?;
    let config: Value = serde_json::from_slice(&bytes)?;
    super::competitive_plan::config::admit(&config)?;
    let context = SourceContext {
        mesh_root: request.mesh_root.clone(),
        mesh_head: head(&request.mesh_root, deadline)?,
        llama_root: request.llama_root.clone(),
        llama_head: head(&request.llama_root, deadline)?,
        custody_scope: "local_checkout_and_artifact_observation_no_build_attestation".into(),
    };
    if config["baseline"]["llama_cpp_revision"] != context.llama_head {
        return Err("raw llama checkout revision differs from configured baseline".into());
    }
    let raw = backend(
        &request.llama_binary,
        &request.llama_root,
        None,
        deadline,
        "llama",
    )?;
    let mesh = backend(
        &request.mesh_binary,
        &request.mesh_root,
        Some(tree(&request.native_runtime)?),
        deadline,
        "mesh",
    )?;
    let models = models(request, &config, &raw, &mesh, deadline)?;
    let input: super::competitive_matrix::Input = serde_json::from_value(
        json!({"config":request.config,"config_sha256":hex::encode(Sha256::digest(&bytes)),"source_context":context,"platform":request.platform,"models":models,"workloads":request.workloads,"optional_arms":request.optional_backends.keys().collect::<Vec<_>>(),"required_comparisons":request.required_comparisons,"adaptive":request.adaptive,"manifest":request.manifest,"benchy":request.benchy.as_ref().map(|path|file(path)).transpose()?,"output":request.output,"timeout_seconds":request.timeout_seconds,"cell_timeout_seconds":request.cell_timeout_seconds,"request_timeout_seconds":request.request_timeout_seconds,"resume":request.resume,"force":request.force}),
    )?;
    super::competitive_matrix::admit(&input)?;
    super::competitive_matrix::preflight(&config, &input, deadline)?;
    if Instant::now() >= deadline {
        return Err("prepared adapter deadline reached".into());
    }
    if !request.prepared_output.is_absolute() {
        return Err("prepared output path must be absolute".into());
    }
    let parent = request
        .prepared_output
        .parent()
        .ok_or("prepared output parent")?;
    super::competitive_report_output::admit(parent, &request.prepared_output)?;
    super::competitive_synthetic::write_new(
        &request.prepared_output,
        &serde_json::to_value(&input)?,
    )
}
pub(in crate::automation) fn run(args: &[String]) -> DynResult<()> {
    use crate::repository::{check_args::Grammar, check_report::CheckReport};
    const GRAMMAR: Grammar = Grammar {
        usage: "cargo xtool automation replay-matrix competitive-prepare --input PATH",
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
    let bytes = super::competitive_cell::read(
        Path::new(parsed.last("--input").ok_or("input")?),
        8 * 1024 * 1024,
    )?;
    prepare(&serde_json::from_slice(&bytes)?)
}

fn component(value: &str) -> DynResult<()> {
    if value.is_empty()
        || [".", ".."].contains(&value)
        || value
            .bytes()
            .any(|b| !b.is_ascii_alphanumeric() && !b"_.-".contains(&b))
    {
        return Err("prepared model layout requires safe component".into());
    }
    Ok(())
}
