use super::contract::{Identity, Input};
use crate::{
    automation::hf_certify::{admission, execution},
    command::DynResult,
    process::Cancellation,
};
use std::{path::Path, time::Instant};
pub(super) fn worker(args: &[String]) -> DynResult<()> {
    let [a, path, b, output, phase] = args else {
        return Err("compose identity-worker flags/phase".into());
    };
    if a != "--input"
        || b != "--output"
        || !["before", "composed", "after"].contains(&phase.as_str())
    {
        return Err("compose identity closed input/phase".into());
    }
    let mut input: Input = serde_json::from_slice(&admission::read(Path::new(path), 262144)?)?;
    input.validate()?;
    let request_sha256 = admission::digest(&serde_json::to_vec(&input)?);
    admission::admit(&mut input.binary, false)?;
    admission::admit(&mut input.mtp_gguf, true)?;
    for part in &mut input.target_parts {
        admission::admit(part, true)?;
    }
    input.validate()?;
    let root = Path::new(output)
        .parent()
        .ok_or("identity output parent")?
        .canonicalize()?;
    let mut outputs = Vec::new();
    if phase != "before" {
        for i in [0, input.expected_parts - 1] {
            outputs.push(admission::observe(&root.join(input.remote_name(i)), true)?);
        }
    }
    admission::publish(
        Path::new(output),
        &Identity {
            schema_version: 1,
            request_sha256,
            admitted: input,
            outputs,
        },
    )
}
pub(super) fn observe(
    input: &Input,
    root: &Path,
    phase: &str,
    deadline: Instant,
    cancel: &Cancellation,
) -> DynResult<Identity> {
    let source = root.join(format!("{phase}-input.json"));
    let output = root.join(format!("{phase}-identity.json"));
    admission::publish(&source, input)?;
    let args = vec![
        "automation".into(),
        "hf-mtp-compose".into(),
        "identity-worker".into(),
        "--input".into(),
        source.to_str().ok_or("identity Unicode")?.into(),
        "--output".into(),
        output.to_str().ok_or("identity Unicode")?.into(),
        phase.into(),
    ];
    let raw = execution::run_process(
        &std::env::current_exe()?,
        args,
        root,
        &format!("identity-{phase}"),
        deadline,
        cancel,
    )?;
    if !execution::clean(&raw) {
        return Err("compose pinned identity worker did not complete cleanly".into());
    }
    let value: Identity = serde_json::from_slice(&admission::read(&output, 1048576)?)?;
    if value.schema_version != 1
        || value.request_sha256 != admission::digest(&serde_json::to_vec(input)?)
        || (phase == "before" && !value.outputs.is_empty())
        || (phase != "before" && value.outputs.len() != 2)
    {
        return Err("compose identity receipt correlation".into());
    }
    value.admitted.validate()?;
    Ok(value)
}
