//! One supplied native window, immutable publication, then same-owned output cleanup.
#[path = "quantization_window/contract.rs"]
pub(super) mod contract;
#[path = "quantization_window/helper.rs"]
pub(super) mod helper;
#[cfg(test)]
#[path = "quantization_window/tests.rs"]
mod tests;
use super::{admission, bootstrap, execution, publication};
use crate::{command::DynResult, process::Cancellation};
use contract::Input;
use serde_json::{Value, json};
use std::{path::Path, time::Instant};
fn check(until: Instant, cancel: &Cancellation) -> DynResult<()> {
    if Instant::now() >= until || cancel.is_cancelled() {
        return Err("quant window inherited deadline/cancellation".into());
    }
    Ok(())
}
pub(in crate::automation::hf_certify) fn pin(
    a: &admission::Artifact,
    until: Instant,
    cancel: &Cancellation,
) -> DynResult<()> {
    use sha2::{Digest as _, Sha256};
    use std::io::Read as _;
    if a.path.canonicalize()? != a.path {
        return Err("quant window canonical immutable pin required".into());
    }
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let mut file = options.open(&a.path)?;
    let size = file.metadata()?.len();
    if !file.metadata()?.is_file() || size == 0 || size > 1024_u64.pow(4) {
        return Err("quant immutable size/type refused".into());
    }
    let mut h = Sha256::new();
    let mut bytes = [0; 65536];
    let mut seen = 0_u64;
    loop {
        check(until, cancel)?;
        let n = file.read(&mut bytes)?;
        if n == 0 {
            break;
        }
        seen = seen
            .checked_add(n as u64)
            .ok_or("quant immutable size overflow")?;
        if seen > size {
            return Err("quant immutable grew".into());
        }
        h.update(&bytes[..n]);
    }
    let observed: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
    if seen != size || file.metadata()?.len() != size || observed != a.sha256 {
        return Err("quant immutable bytes changed".into());
    }
    check(until, cancel)
}
fn custody(input: &Input, until: Instant, cancel: &Cancellation) -> DynResult<()> {
    for a in [
        &input.tool,
        &input.tool_source,
        &input.runtime,
        &input.manifest,
        &input.recipe,
        &input.helper,
        &input.helper_source,
    ]
    .into_iter()
    {
        pin(a, until, cancel)?;
    }
    for (index, a) in input.source_parts.iter().enumerate() {
        check(until, cancel)?;
        if index == 0 || index + 1 == input.ordinal as usize {
            pin(a, until, cancel)?;
        } else if a.path.canonicalize()? != a.path || !std::fs::symlink_metadata(&a.path)?.is_file()
        {
            return Err("quant nonresident source identity changed".into());
        }
    }
    Ok(())
}
fn manifest(input: &Input) -> DynResult<()> {
    let bytes = admission::read(&input.manifest.path, 1048576)?;
    if admission::digest(&bytes) != input.manifest.sha256 {
        return Err("bounded quant manifest pin changed".into());
    }
    let v: Value = serde_json::from_slice(&bytes)?;
    if v["schema_version"] != 1
        || v["kind"] != "QUANTIZE_GGUF"
        || v["expected_splits"] != input.expected_splits
        || v["window_size"] != 1
        || v["source"] != input.source_root.to_string_lossy().as_ref()
        || v["source_prefix"] != input.source_prefix
        || v["target"] != input.target_root.to_string_lossy().as_ref()
        || v["target_prefix"] != input.target_prefix
        || v["output_basename"] != input.basename
        || v["tensor_type_file"] != input.recipe.path.to_string_lossy().as_ref()
        || v["quant"] != input.quant
    {
        return Err("quant window complete manifest identity refused".into());
    }
    Ok(())
}
fn regular_header(path: &Path) -> DynResult<()> {
    use std::io::Read as _;
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let mut file = options.open(path)?;
    if !file.metadata()?.is_file()
        || file.metadata()?.len() < 24
        || file.metadata()?.len() > 1024_u64.pow(4)
    {
        return Err("quant output GGUF type/size refused".into());
    }
    let mut header = [0; 24];
    file.read_exact(&mut header)?;
    if header[..4] != *b"GGUF" || u32::from_le_bytes(header[4..8].try_into()?) != 3 {
        return Err("quant output GGUF framing refused".into());
    }
    Ok(())
}
fn clean_stage(input: &Input, until: Instant, cancel: &Cancellation) -> DynResult<()> {
    // Only the literal source-window subtree in this fresh owned workspace is removable.
    let stage = input.work_root.join("source-window");
    let prefix = stage.join(&input.source_prefix);
    if stage.canonicalize()? != stage || prefix.canonicalize()? != prefix {
        return Err("quant staging directory identity refused".into());
    }
    let mut actual = std::collections::BTreeSet::new();
    for entry in std::fs::read_dir(&prefix)? {
        check(until, cancel)?;
        actual.insert(entry?.path());
        if actual.len() > input.expected_splits as usize {
            return Err("quant staging roster exceeds declared bound".into());
        }
    }
    let expected: std::collections::BTreeSet<_> = input
        .source_parts
        .iter()
        .map(|a| prefix.join(a.path.file_name().unwrap()))
        .collect();
    if actual != expected {
        return Err("quant staging contains foreign paths; cleanup refused".into());
    }
    for (i, a) in input.source_parts.iter().enumerate() {
        let staged = prefix.join(a.path.file_name().unwrap());
        let metadata = std::fs::symlink_metadata(&staged)?;
        if i + 1 == input.ordinal as usize {
            pin(
                &admission::Artifact {
                    path: staged.clone(),
                    sha256: a.sha256.clone(),
                },
                until,
                cancel,
            )?;
        } else if !metadata.file_type().is_symlink() || staged.canonicalize()? != a.path {
            return Err("quant nonresident source links changed; cleanup refused".into());
        }
        check(until, cancel)?;
        std::fs::remove_file(staged)?;
    }
    std::fs::remove_dir(&prefix)?;
    std::fs::remove_dir(stage)?;
    std::fs::remove_dir(&input.work_root)?;
    std::fs::remove_dir(input.target_root.join(&input.target_prefix))?;
    std::fs::remove_dir(&input.target_root)?;
    Ok(())
}
pub(in crate::automation::hf_certify) fn execute(
    input: &Input,
    root: &Path,
    until: Instant,
    cancel: &Cancellation,
    evidence: &mut Value,
) -> DynResult<()> {
    input.validate()?;
    check(until, cancel)?;
    custody(input, until, cancel)?;
    manifest(input)?;
    if let Some(resume) = &input.resume {
        pin(&resume.record, until, cancel)?;
        pin(&resume.shard, until, cancel)?;
        let record: Value =
            serde_json::from_slice(&admission::read(&resume.record.path, 1048576)?)?;
        if record["schema_version"] != 1
            || record["artifact"]["byte_size"]
                != std::fs::symlink_metadata(&resume.shard.path)?.len()
            || record["context_sha256"] != input.context_sha256()?
            || record["ordinal"] != input.ordinal
            || record["artifact"]["sha256"] != resume.shard.sha256
            || record["relative_path"] != input.remote_path()
            || record["window_uploaded"] != true
        {
            return Err("remote resume source/recipe/complete window correlation refused".into());
        }
        helper::verify(
            input,
            &resume.record,
            &resume.commit,
            &input.record_path(),
            helper::Context {
                root,
                until,
                cancel,
                evidence,
                label: "resume-record",
            },
        )?;
        helper::verify(
            input,
            &resume.shard,
            &resume.commit,
            &input.remote_path(),
            helper::Context {
                root,
                until,
                cancel,
                evidence,
                label: "resume-shard",
            },
        )?;
        custody(input, until, cancel)?;
        evidence["window_uploaded"] = json!(true);
        evidence["resumed_immutable_commit"] = json!(resume.commit);
        evidence["record_commit"] = json!(resume.commit);
        evidence["artifact"] = record["artifact"].clone();
        return check(until, cancel);
    }
    for directory in [&input.work_root, &input.target_root] {
        if std::fs::symlink_metadata(directory).is_ok()
            || directory.parent().ok_or("window parent")?.canonicalize()?
                != directory.parent().unwrap()
        {
            return Err("quant window work/target must be fresh canonical-parent paths".into());
        }
    }
    let p = execution::run_process(
        &input.tool.path,
        input.quant_args(),
        root,
        "quant-window",
        until,
        cancel,
    )?;
    evidence["quant-window"] = publication::process_observation(&p);
    if !execution::clean(&p) {
        return Err("quant window failed; no full-model fallback".into());
    }
    check(until, cancel)?;
    custody(input, until, cancel)?;
    regular_header(&input.output_path())?;
    let mut outputs = std::fs::read_dir(input.target_root.join(&input.target_prefix))?;
    if outputs.next().transpose()?.map(|e| e.path()) != Some(input.output_path())
        || outputs.next().is_some()
    {
        return Err("quant one-window output roster refused".into());
    }
    let publication = helper::upload(
        input,
        &input.output_path(),
        &input.remote_path(),
        true,
        helper::Context {
            root,
            until,
            cancel,
            evidence,
            label: "upload-shard",
        },
    )?;
    if input.output_path().exists() {
        return Err("verified unlink absent; preserve staging".into());
    }
    evidence["artifact"] = publication["identity"].clone();
    evidence["window_uploaded"] = json!(true);
    // The record binds actual observed output bytes to immutable source/recipe/tool context.
    let record = json!({"schema_version":1,"context_sha256":input.context_sha256()?,"ordinal":input.ordinal,"relative_path":input.remote_path(),"window_uploaded":true,"artifact":publication["identity"],"shard_commit":helper::commit(&publication)?});
    let record_path = root.join("window-record.json");
    admission::publish(&record_path, &record)?;
    let published_record = helper::upload(
        input,
        &record_path,
        &input.record_path(),
        false,
        helper::Context {
            root,
            until,
            cancel,
            evidence,
            label: "upload-record",
        },
    )?;
    evidence["record_commit"] = json!(helper::commit(&published_record)?);
    check(until, cancel)?;
    custody(input, until, cancel)?;
    clean_stage(input, until, cancel)?;
    evidence["staged_input_cleaned"] = json!(true);
    check(until, cancel)
}
pub(super) fn run(args: &[String]) -> DynResult<()> {
    let [a, path, b, output] = args else {
        return Err("quantization-window --input FILE --output-directory FRESH".into());
    };
    if a != "--input" || b != "--output-directory" {
        return Err("quant window closed flags".into());
    }
    let input: Input = serde_json::from_slice(&admission::read(Path::new(path), 8 * 1048576)?)?;
    input.validate()?;
    let root = std::path::absolute(output)?;
    if root.parent().ok_or("quant output parent")?.canonicalize()? != root.parent().unwrap()
        || [&input.source_root, &input.target_root, &input.work_root]
            .iter()
            .any(|p| !contract::disjoint(&root, p))
        || input.pins().iter().any(|a| a.path.starts_with(&root))
        || input.credential_file.starts_with(&root)
    {
        return Err("quant evidence/source/work disjointness refused".into());
    }
    std::fs::create_dir(&root)?;
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let cancel = interrupt.cancellation();
    let until = Instant::now() + std::time::Duration::from_secs(input.timeout_seconds);
    let mut evidence = json!({"schema_version":1,"status":"WINDOW_OBSERVATIONS_ONLY","context_sha256":input.context_sha256()?,"window_uploaded":false,"completed_job":false,"tool_profile_qualified":false,"workflow_qualified":false,"ordinal":input.ordinal});
    let result = execute(&input, &root, until, &cancel, &mut evidence);
    super::quantization_probe::finish_observations(
        &mut evidence,
        &root.join("observations.json"),
        until,
        interrupt,
        result,
    )
}
