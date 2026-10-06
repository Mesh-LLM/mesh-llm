//! Bounded terminal references retain detailed observations in their owning directories.
use super::*;
pub(super) fn retain(root: &Path, label: &str, value: &Value) -> DynResult<Value> {
    let path = root.join(format!("{label}-observations.json"));
    admission::publish(&path, value)?;
    let bytes = admission::read(&path, 1048576)?;
    Ok(json!({"path":path,"sha256":admission::digest(&bytes),"byte_size":bytes.len()}))
}
#[derive(serde::Serialize)]
struct RepositoryOptions<'a> {
    repo: &'a str,
    credential_file: &'a Path,
    output_directory: &'a Path,
    timeout_seconds: u64,
    confirm: bool,
}
pub(super) fn repository(
    input: &Input,
    repo: &str,
    context: window::helper::Context<'_>,
) -> DynResult<()> {
    let window::helper::Context {
        root,
        until,
        cancel,
        evidence,
        label,
    } = context;
    let w = &input.window_template;
    let output = root.join(label);
    let seconds = seconds(until, cancel)?.min(1200);
    let opts = RepositoryOptions {
        repo,
        credential_file: &w.credential_file,
        output_directory: &output,
        timeout_seconds: seconds,
        confirm: true,
    };
    let hash = admission::digest(&serde_json::to_vec(&opts)?);
    window::pin(&w.helper_source, until, cancel)?;
    child(
        &w.helper,
        vec![
            "ensure-repo".into(),
            "--repo".into(),
            repo.into(),
            "--credential-file".into(),
            w.credential_file.to_string_lossy().into(),
            "--output-directory".into(),
            output.to_string_lossy().into(),
            "--timeout-seconds".into(),
            seconds.to_string(),
            "--confirm".into(),
        ],
        root,
        label,
        until,
        cancel,
        evidence,
    )?;
    window::pin(&w.helper_source, until, cancel)?;
    let value: Value =
        serde_json::from_slice(&admission::read(&output.join("repository.json"), 1048576)?)?;
    evidence[format!("{label}-receipt")] = value.clone();
    if value["request_sha256"] != hash
        || value["status"] != "REPOSITORY_READY"
        || value["repository"]["repo"] != repo
        || value["repository"]["completed"] != true
        || !value["repository"]["error"].is_null()
        || !value["repository"]["observed_parent"]
            .as_str()
            .is_some_and(|s| bootstrap::contract::hex(s, 40))
    {
        return Err("quant repository correlation refused".into());
    }
    check(until, cancel)
}
