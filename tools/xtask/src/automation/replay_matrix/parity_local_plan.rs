//! Local immutable candidate admission and certification invocation planning.
//! This is an observation/plan, never native-source or model certification proof.
use crate::{command::DynResult, repository::check_report::CheckReport};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeSet,
    fs,
    path::{Component, Path, PathBuf},
};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    cache_root: PathBuf,
    pub(super) source_root: PathBuf,
    candidates: Vec<Candidate>,
    statuses: Vec<String>,
    families: Vec<String>,
    llama_models: Vec<String>,
    priorities: Vec<String>,
    limit: Option<usize>,
    policy: Policy,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Candidate {
    family: String,
    llama_model: String,
    status: String,
    priority: String,
    model_pin: Pin,
    #[serde(default)]
    observation_only: bool,
    layer_end: Option<u64>,
    split_layer: Option<u64>,
    splits: Option<Vec<u64>>,
    recurrent_all: bool,
    recurrent_ranges: Vec<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Pin {
    repo: String,
    revision: String,
    file: String,
    blob_sha256: String,
    size_bytes: u64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Policy {
    ctx_size: u64,
    n_gpu_layers: i64,
    prompt: String,
    run_id: String,
    skip_build: bool,
    skip_state: bool,
    state_payload_kind: Option<String>,
    #[serde(default = "dense_kind")]
    dense_state_payload_kind: String,
    prefix_token_count: Option<u64>,
    cache_hit_repeats: Option<u64>,
    borrow_resident_hits: bool,
    cache_decoded_result_hits: bool,
    startup_timeout_secs: Option<u64>,
}
pub(in crate::automation) fn run(args: &[String]) -> DynResult<()> {
    match args {
        [help] if help == "--help" => CheckReport::success("cargo xtool automation replay-matrix parity-local-plan --input PATH\nObserves exact immutable HF snapshot files and plans family-certify argument arrays. No download, build or model run; no source admission claim.\n".into()).emit(),
        [flag, path] if flag == "--input" => {
            let bytes = document(Path::new(path))?;
            let input: Input = serde_json::from_slice(&bytes)?;
            let output = plan(&input)?;
            CheckReport::success(format!("{}\n",serde_json::to_string_pretty(&output)?)).emit()
        }
        _ => Err("parity-local-plan requires --input PATH".into()),
    }
}
fn label(text: &str) -> bool {
    !text.is_empty()
        && text
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
        && text != "."
        && text != ".."
}
fn hex(text: &str, length: usize) -> bool {
    text.len() == length
        && text
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn names(values: &[String]) -> bool {
    values.iter().all(|v| label(v)) && values.iter().collect::<BTreeSet<_>>().len() == values.len()
}
fn admit(input: &Input) -> DynResult<()> {
    if !input.cache_root.is_absolute()
        || !input.source_root.is_absolute()
        || !input.cache_root.is_dir()
        || !input.source_root.is_dir()
        || input.candidates.len() > 4096
        || input.limit == Some(0)
        || !names(&input.statuses)
        || !names(&input.families)
        || !names(&input.llama_models)
        || !names(&input.priorities)
    {
        return Err("invalid local parity roots/selection".into());
    }
    let p = &input.policy;
    if ![
        "resident-kv",
        "kv-recurrent",
        "full-state",
        "kv-only",
        "recurrent-only",
    ]
    .contains(&p.dense_state_payload_kind.as_str())
    {
        return Err("invalid dense cache default".into());
    }
    if p.ctx_size == 0
        || p.n_gpu_layers < -1
        || !label(&p.run_id)
        || p.prompt.contains('\0')
        || p.prefix_token_count == Some(0)
        || p.cache_hit_repeats == Some(0)
        || p.startup_timeout_secs == Some(0)
        || p.state_payload_kind.as_ref().is_some_and(|s| {
            ![
                "resident-kv",
                "kv-recurrent",
                "full-state",
                "kv-only",
                "recurrent-only",
            ]
            .contains(&s.as_str())
        })
    {
        return Err("invalid certification policy".into());
    }
    Ok(())
}
fn selected(input: &Input, c: &Candidate) -> bool {
    let statuses: Vec<&str> = if input.statuses.is_empty() {
        vec![
            "candidate",
            "candidate_stateful",
            "candidate_multimodal",
            "certified",
        ]
    } else {
        input.statuses.iter().map(String::as_str).collect()
    };
    statuses.contains(&c.status.as_str())
        && (input.families.is_empty() || input.families.contains(&c.family))
        && (input.llama_models.is_empty() || input.llama_models.contains(&c.llama_model))
        && (input.priorities.is_empty() || input.priorities.contains(&c.priority))
}
fn candidate_path(root: &Path, c: &Candidate) -> DynResult<PathBuf> {
    let parts: Vec<_> = c.model_pin.repo.split('/').collect();
    if !label(&c.family)
        || !label(&c.llama_model)
        || !label(&c.status)
        || !["p0", "p1", "p2"].contains(&c.priority.as_str())
        || parts.len() != 2
        || !parts.iter().all(|p| label(p))
        || !hex(&c.model_pin.revision, 40)
        || !hex(&c.model_pin.blob_sha256, 64)
        || c.model_pin.size_bytes == 0
        || c.model_pin.file.contains('\\')
        || c.model_pin.file.chars().any(char::is_control)
        || Path::new(&c.model_pin.file)
            .components()
            .any(|p| !matches!(p, Component::Normal(_)))
        || crate::model_registry::serving_entry::rank(&c.model_pin.file).is_none()
    {
        return Err("candidate requires safe immutable serving-file pin".into());
    }
    Ok(root
        .join(format!("models--{}--{}", parts[0], parts[1]))
        .join("snapshots")
        .join(&c.model_pin.revision)
        .join(&c.model_pin.file))
}
fn observe(
    root: &Path,
    c: &Candidate,
    guard: &mut impl FnMut() -> std::io::Result<()>,
) -> DynResult<Value> {
    let named = candidate_path(root, c)?;
    if !named.try_exists()? {
        return Ok(json!({"status":"missing","local_path":null}));
    }
    let path = named.canonicalize()?;
    if !fs::symlink_metadata(&path)?.is_file() {
        return Err("cached candidate target must be a regular file".into());
    }
    // HF snapshot links commonly point into the owning cache's blobs directory.
    if !path.starts_with(root.canonicalize()?) {
        return Err("cached candidate target escapes selected cache".into());
    }
    if fs::metadata(&path)?.len() != c.model_pin.size_bytes {
        return Err("cached candidate size differs from source pin".into());
    }
    let actual =
        crate::product::digest::file_sha256_with_guard(&path, guard).map_err(|e| e.error)?;
    if actual != c.model_pin.blob_sha256 {
        return Err("cached candidate digest differs from source pin".into());
    }
    let shape = super::model_preflight::dimensions::inspect(&path)?
        .ok_or("cached GGUF has no model dimensions")?;
    let layers = c.layer_end.unwrap_or(shape.block_count);
    if layers < 3 || layers > shape.block_count || shape.activation_width == 0 {
        return Err("certification requires at least three admitted layers".into());
    }
    let split = c.split_layer.unwrap_or(layers / 2);
    let splits = c
        .splits
        .clone()
        .unwrap_or_else(|| vec![layers / 3, 2 * layers / 3]);
    if split == 0
        || split >= layers
        || splits.is_empty()
        || splits.iter().any(|s| *s == 0 || *s >= layers)
        || splits.windows(2).any(|s| s[0] >= s[1])
    {
        return Err("certification splits must be increasing interior boundaries".into());
    }
    Ok(
        json!({"status":"local_verified","local_path":path,"gguf_arch":shape.architecture,"layer_end":layers,"activation_width":shape.activation_width,"split_layer":split,"splits":splits,"blob_sha256":actual}),
    )
}
fn invocation(root: &Path, c: &Candidate, row: &Value, p: &Policy) -> DynResult<Value> {
    if c.recurrent_all && !c.recurrent_ranges.is_empty() {
        return Err("recurrent-all and recurrent-ranges conflict".into());
    }
    for range in &c.recurrent_ranges {
        let (a, b) = range
            .split_once("..")
            .ok_or("recurrent range requires start..end")?;
        let (a, b) = (a.parse::<u64>()?, b.parse::<u64>()?);
        if a >= b || b > row["layer_end"].as_u64().ok_or("layer end")? {
            return Err("recurrent range escapes admitted layers".into());
        }
    }
    let prefix = p.prefix_token_count.unwrap_or(0);
    let context = p.ctx_size.max(if prefix > 0 {
        prefix
            .checked_mul(2)
            .and_then(|n| n.checked_add(32))
            .ok_or("prefix context overflow")?
    } else {
        0
    });
    let mut args = vec![
        "--family".into(),
        c.family.clone(),
        "--target-model".into(),
        row["local_path"].as_str().ok_or("local path")?.into(),
        "--model-id".into(),
        format!(
            "{}@{}/{}",
            c.model_pin.repo, c.model_pin.revision, c.model_pin.file
        ),
        "--layer-end".into(),
        row["layer_end"].to_string(),
        "--split-layer".into(),
        row["split_layer"].to_string(),
        "--splits".into(),
        row["splits"]
            .as_array()
            .ok_or("splits")?
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join(","),
        "--activation-width".into(),
        row["activation_width"].to_string(),
        "--ctx-size".into(),
        context.to_string(),
        "--n-gpu-layers".into(),
        p.n_gpu_layers.to_string(),
        "--prompt".into(),
        p.prompt.clone(),
        "--run-id".into(),
        p.run_id.clone(),
    ];
    for (flag, enabled) in [
        ("--skip-build", p.skip_build),
        ("--skip-state", p.skip_state),
        ("--borrow-resident-hits", p.borrow_resident_hits),
        ("--cache-decoded-result-hits", p.cache_decoded_result_hits),
        ("--recurrent-all", c.recurrent_all),
    ] {
        if enabled {
            args.push(flag.into());
        }
    }
    if !c.recurrent_ranges.is_empty() {
        args.extend(["--recurrent-ranges".into(), c.recurrent_ranges.join(",")]);
    }
    let kind = p.state_payload_kind.clone().or_else(|| {
        (prefix > 0).then(|| {
            if c.recurrent_all || !c.recurrent_ranges.is_empty() {
                "kv-recurrent"
            } else {
                &p.dense_state_payload_kind
            }
            .into()
        })
    });
    if let Some(kind) = kind {
        args.extend(["--state-payload-kind".into(), kind]);
    }
    for (flag, value) in [
        ("--prefix-token-count", p.prefix_token_count),
        ("--cache-hit-repeats", p.cache_hit_repeats),
        ("--startup-timeout-secs", p.startup_timeout_secs),
    ] {
        if let Some(value) = value {
            args.extend([flag.into(), value.to_string()]);
        }
    }
    Ok(
        json!({"executable":root.join("scripts/family-certify.sh"),"arguments":args,"cwd":root,"scope":"planned_only_no_certification"}),
    )
}
pub(super) fn plan(input: &Input) -> DynResult<Value> {
    plan_with_guard(input, &mut || Ok(()))
}
pub(super) fn plan_with_guard(
    input: &Input,
    guard: &mut impl FnMut() -> std::io::Result<()>,
) -> DynResult<Value> {
    guard()?;
    admit(input)?;
    let root = input.source_root.canonicalize()?;
    let mut rows = Vec::new();
    let mut invocations = Vec::new();
    for c in &input.candidates {
        guard()?;
        // Pin syntax/path errors refuse the entire command even for filtered rows.
        candidate_path(&input.cache_root, c)?;
        let mut row = json!({"family":c.family,"llama_model":c.llama_model,"classification":c.status,"priority":c.priority,"identity_scope":if c.observation_only{"observed_offline_cache_not_provider_pin"}else{"declared_immutable_source_pin"}});
        match observe(&input.cache_root, c, guard) {
            Ok(observation) => {
                for (key, value) in observation.as_object().ok_or("observation")? {
                    row[key] = value.clone();
                }
            }
            Err(error) => {
                guard()?;
                row["status"] = "inspect_error".into();
                let named = candidate_path(&input.cache_root, c)?;
                row["local_path"] = if named.try_exists()? {
                    json!(named)
                } else {
                    Value::Null
                };
                row["inspect_error"] = error.to_string().into();
            }
        }
        if row["status"] == "local_verified"
            && selected(input, c)
            && input.limit.is_none_or(|limit| invocations.len() < limit)
        {
            invocations.push(invocation(&root, c, &row, &input.policy)?);
        }
        rows.push(row);
    }
    Ok(
        json!({"schema_version":1,"scope":"local_file_identity_and_invocation_plan_not_source_admission_or_certification","rows":rows,"invocations":invocations}),
    )
}
pub(super) fn document(path: &Path) -> DynResult<Vec<u8>> {
    use std::io::Read as _;
    if !fs::symlink_metadata(path)?.is_file() {
        return Err("parity input must be regular".into());
    }
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let file = options.open(path)?;
    if !file.metadata()?.is_file() {
        return Err("opened parity input must be regular".into());
    }
    let mut bytes = Vec::new();
    file.take(8 * 1024 * 1024 + 1).read_to_end(&mut bytes)?;
    if bytes.len() > 8 * 1024 * 1024 {
        return Err("parity input exceeds eight MiB".into());
    }
    Ok(bytes)
}

fn dense_kind() -> String {
    "resident-kv".into()
}
