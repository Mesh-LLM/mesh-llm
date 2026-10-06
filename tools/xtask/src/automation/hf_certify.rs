//! Local native certification caller. No download, build or publication adapter.
pub(in crate::automation) mod admission;
pub(in crate::automation) mod execution;
use crate::{automation::command_interrupt::Interrupt, command::DynResult};
use serde_json::json;
use std::{
    path::Path,
    time::{Duration, Instant},
};
pub(crate) fn run(args: &[String]) -> DynResult<()> {
    if let [verb, rest @ ..] = args
        && verb == "identity-worker"
    {
        return admission::worker(rest);
    }
    if args == ["--help"] {
        println!(
            "automation hf-certify --input FILE --output-directory FRESH_DIRECTORY; local standalone CPU product validation only"
        );
        return Ok(());
    }
    let [a, path, b, output] = args else {
        return Err("hf-certify --input FILE --output-directory FRESH_DIRECTORY".into());
    };
    if a != "--input" || b != "--output-directory" {
        return Err("hf-certify closed flags".into());
    }
    let input: admission::Input =
        serde_json::from_slice(&admission::read(Path::new(path), 262144)?)?;
    input.validate()?;
    let requested = std::path::absolute(output)?;
    let parent = requested.parent().ok_or("output parent")?.canonicalize()?;
    let root = parent.join(requested.file_name().ok_or("output leaf")?);
    std::fs::create_dir(&root)?;
    let interrupt = Interrupt::install()?;
    let cancel = interrupt.cancellation();
    let deadline = Instant::now() + Duration::from_secs(input.timeout_secs);
    let mut evidence = json!({"schema_version":1,"status":"FAILED","request_sha256":admission::digest(&serde_json::to_vec(&input)?),"admitted":null,"native_report":null,"source_unchanged":false,"error":null,"profile_scope":"supplied standalone static CPU profile; real native loader/ABI acceptance belongs to product validation","host":{"os":std::env::consts::OS,"arch":std::env::consts::ARCH}});
    let result = execution::execute(&input, &root, deadline, &cancel, &mut evidence);
    let interruption = interrupt.finish();
    match (result, interruption) {
        (Ok(()), Ok(())) => evidence["status"] = json!("PASS"),
        (result, signal) => {
            evidence["error"] = json!(format!(
                "{}{}",
                result.err().map_or(String::new(), |e| e.to_string()),
                signal.err().map_or(String::new(), |e| format!("; {e}"))
            ))
        }
    }
    admission::publish(&root.join("report.json"), &evidence)?;
    println!("{}", root.join("report.json").display());
    if evidence["status"] == "PASS" {
        Ok(())
    } else {
        Err("HF certification failed; owned evidence retained".into())
    }
}

#[cfg(test)]
#[path = "hf_certify/tests.rs"]
mod tests;
