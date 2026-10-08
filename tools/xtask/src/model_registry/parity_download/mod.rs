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
/// A manual generated-registry declaration pin; local cache bytes verify separately.
pub(crate) fn source_pin(
    registry: &serde_json::Value,
    id: &str,
) -> crate::command::DynResult<serde_json::Value> {
    use crate::ci_plan::document::Json;
    let bytes = serde_json::to_vec(registry)?;
    let document = Json::parse(&bytes)?;
    let artifact = super::manifest::resolve(
        &document,
        &super::manifest::Selection {
            artifact_id: Some(id),
            cadence: "manual",
        },
    )
    .map_err(|error| error.to_string())?;
    let revision = artifact
        .row
        .iter()
        .find(|(name, _)| name == "revision")
        .and_then(|(_, v)| v.as_str())
        .ok_or("registry revision")?;
    let repo = artifact
        .row
        .iter()
        .find(|(name, _)| name == "repo")
        .and_then(|(_, v)| v.as_str())
        .ok_or("registry repo")?;
    if revision.len() != 40
        || !revision
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err("manual source pin requires immutable revision".into());
    }
    let file = artifact
        .files
        .iter()
        .filter_map(|file| super::serving_entry::rank(&file.name).map(|rank| (rank, file)))
        .min_by_key(|(rank, _)| *rank)
        .map(|(_, file)| file)
        .ok_or("manual registry has no serving GGUF entry")?;
    Ok(
        serde_json::json!({"repo":repo,"revision":revision,"file":file.name,"blob_sha256":file.sha256,"size_bytes":file.size_bytes}),
    )
}

#[cfg(test)]
mod source_pin_tests {
    #[test]
    fn manual_registry_pin_uses_serving_entry_and_refuses_cadence_revision_drift() {
        use serde_json::json;
        let mut registry = json!({"manifest_kind":"test-model-artifacts","artifacts":[{"id":"fixture","repo":"fixture/model","revision":"a".repeat(40),"selector":"fixture","model_ref":"fixture/model","cadences":["manual"],"files":["model-00002-of-00002.gguf","model-00001-of-00002.gguf"],"urls":["https://example.invalid/later","https://example.invalid/first"],"file_integrity":{"model-00002-of-00002.gguf":{"blob_id":"2".repeat(64),"size_bytes":2},"model-00001-of-00002.gguf":{"blob_id":"1".repeat(64),"size_bytes":1}}}]});
        let pin = super::source_pin(&registry, "fixture").unwrap();
        assert_eq!(pin["file"], "model-00001-of-00002.gguf");
        assert_eq!(pin["blob_sha256"], "1".repeat(64));
        assert_eq!(pin["size_bytes"], 1);
        registry["artifacts"][0]["revision"] = "main".into();
        assert!(
            super::source_pin(&registry, "fixture")
                .unwrap_err()
                .to_string()
                .contains("immutable revision")
        );
        registry["artifacts"][0]["revision"] = "a".repeat(40).into();
        registry["artifacts"][0]["cadences"] = json!(["pull_request"]);
        assert!(
            super::source_pin(&registry, "fixture")
                .unwrap_err()
                .to_string()
                .contains("not allowed at cadence")
        );
    }
}
