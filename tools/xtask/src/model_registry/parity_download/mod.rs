//! Manual parity discovery and immutable candidate input materialization.
mod materialize;
mod selection;
use crate::{command::DynResult, repository::check_report::CheckReport};
use std::path::PathBuf;

const USAGE: &str = "usage: cargo xtool models parity-download --cadence manual
    --manifest <candidates.json> --model-manifest <artifacts.json>
    --hf-command <absolute-path> [--dry-run] [--status <CSV>] [--priority <CSV>]
    [--timeout-secs <1..86400>]

Select manual parity inputs before starting downloads. Immutable inputs verify
every selected file. Unpinned patterns remain discovery inputs. Ordinary HF
failures are reported as skipped; cancellation, deadlines and integrity failures
stop without verified claims or completion. Timeout defaults to 3600 seconds
per target and covers its download and verification.\n";

struct Request {
    manifest: PathBuf,
    registry: PathBuf,
    statuses: String,
    priorities: String,
    hf: PathBuf,
    dry_run: bool,
    timeout: u64,
}
fn request(args: &[String]) -> DynResult<Request> {
    let mut request = Request {
        manifest: PathBuf::new(),
        registry: PathBuf::new(),
        hf: PathBuf::new(),
        statuses: "needs_candidate,candidate_multimodal,package_or_remote_only".into(),
        priorities: "p0,p1".into(),
        dry_run: false,
        timeout: 3600,
    };
    let mut manual_cadence = false;
    let mut args = args.iter();
    while let Some(option) = args.next() {
        if option == "--dry-run" {
            request.dry_run = true;
            continue;
        }
        let value = args.next().ok_or("option requires a value")?;
        match option.as_str() {
            "--cadence" if value == "manual" => manual_cadence = true,
            "--cadence" => return Err("parity downloads are manual-cadence only".into()),
            "--manifest" => request.manifest = value.into(),
            "--model-manifest" => request.registry = value.into(),
            "--hf-command" => request.hf = value.into(),
            "--status" | "--statuses" => request.statuses = value.clone(),
            "--priority" | "--priorities" => request.priorities = value.clone(),
            "--timeout-secs" => request.timeout = value.parse()?,
            _ => return Err(format!("unknown parity-download option: {option}").into()),
        }
    }
    if !manual_cadence {
        return Err("models parity-download requires explicit --cadence manual".into());
    }
    if request.manifest.as_os_str().is_empty()
        || request.registry.as_os_str().is_empty()
        || !request.hf.is_absolute()
        || !request.hf.is_file()
        || request.timeout == 0
        || request.timeout > 86400
    {
        return Err("models parity-download requires --manifest, --model-manifest and absolute --hf-command; timeout must be positive".into());
    }
    Ok(request)
}
pub(super) fn run(args: &[String]) -> CheckReport {
    if args
        .iter()
        .any(|arg| matches!(arg.as_str(), "-h" | "--help"))
    {
        return CheckReport::success(USAGE.to_owned());
    }
    let mut stdout = String::new();
    let result: DynResult<()> = (|| {
        let request = request(args)?;
        let plan = selection::plan(&request)?;
        materialize::execute(&request, &plan, &mut stdout)
    })();
    match result {
        Ok(()) => CheckReport::success(stdout),
        Err(error) => CheckReport {
            stdout,
            stderr: format!("parity inputs: {error}\n"),
            code: 2,
        },
    }
}
