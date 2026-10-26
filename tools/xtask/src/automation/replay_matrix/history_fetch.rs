use crate::{
    automation::private_state::PrivateState,
    command::DynResult,
    process,
    repository::{check_args::Grammar, check_report::CheckReport},
};
use std::{path::Path, time::Duration};
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Receipt {
    Downloaded,
    Bootstrap,
}
pub(super) fn fetch(
    repo: &str,
    output: &Path,
    hf: &Path,
    curl: &Path,
    timeout: Duration,
    cancellation: &process::Cancellation,
) -> DynResult<Receipt> {
    super::history_hub::repository(repo)?;
    if !output.is_absolute() || output.try_exists()? || std::fs::symlink_metadata(output).is_ok() {
        return Err("history output must be an absent absolute path".into());
    }
    let parent = output.parent().ok_or("history output needs a parent")?;
    std::fs::create_dir_all(parent)?;
    let state = PrivateState::create(parent, "replay-history-fetch")?;
    state.prepare()?;
    let temporary = state.root().join("download");
    let result: DynResult<Receipt> = (|| {
        let report = super::history_hub::execute(
            hf,
            vec![
                "download".into(),
                repo.into(),
                "--repo-type".into(),
                "dataset".into(),
                "--local-dir".into(),
                temporary
                    .to_str()
                    .ok_or("non-Unicode history staging path")?
                    .into(),
                "--exclude".into(),
                "*.md".into(),
            ],
            &state,
            None,
            timeout,
            cancellation,
        )?;
        if !super::history_hub::terminal(&report) {
            return Err("history download supervision failed; bootstrap forbidden".into());
        }
        if report.success() {
            if !std::fs::symlink_metadata(&temporary)?.file_type().is_dir() {
                return Err("history download did not produce an owned directory".into());
            }
            reject_links(&temporary)?;
            std::fs::rename(&temporary, output)?;
            return Ok(Receipt::Downloaded);
        }
        let status = super::history_hub::lookup(
            curl,
            &super::history_hub::lookup_url(repo)?,
            &state,
            cancellation,
        )?;
        if status == 404 {
            Ok(Receipt::Bootstrap)
        } else {
            Err(format!(
                "history download failed; anonymous dataset lookup HTTP {status} forbids bootstrap"
            )
            .into())
        }
    })();
    state
        .finish(result)
        .map_err(|error| format!("history fetch/finalization failed: {error:?}").into())
}
fn reject_links(root: &Path) -> DynResult<()> {
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        if kind.is_symlink() || (!kind.is_dir() && !kind.is_file()) {
            return Err("history download contains unsupported filesystem entries".into());
        }
        if kind.is_dir() {
            reject_links(&entry.path())?;
        }
    }
    Ok(())
}
pub(in crate::automation) fn run(args: &[String]) -> DynResult<()> {
    const GRAMMAR: Grammar = Grammar {
        usage: "cargo xtool automation replay-matrix history-fetch --dataset-repo OWNER/NAME --output PATH [--hf PATH] [--curl PATH] [--timeout SECONDS]",
        values: &["--dataset-repo", "--output", "--hf", "--curl", "--timeout"],
        flags: &["--help"],
    };
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    if parsed.flag("--help") {
        return CheckReport::success(format!("{}\n", GRAMMAR.usage)).emit();
    }
    if !parsed.positionals.is_empty() {
        return GRAMMAR.error("unexpected positional arguments").emit();
    }
    let repo = parsed
        .last("--dataset-repo")
        .ok_or("missing --dataset-repo")?;
    let output = std::path::absolute(parsed.last("--output").ok_or("missing --output")?)?;
    let hf = super::history_hub::tool(parsed.last("--hf"), "hf")?;
    let curl = super::history_hub::tool(parsed.last("--curl"), "curl")?;
    let timeout = super::history_hub::seconds(parsed.last("--timeout").unwrap_or("540"))?;
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let result = fetch(
        repo,
        &output,
        &hf,
        &curl,
        timeout,
        &interrupt.cancellation(),
    );
    let interrupted = interrupt.finish();
    interrupted?;
    let message = match result? {
        Receipt::Downloaded => "history downloaded anonymously",
        Receipt::Bootstrap => "no history yet (verified HTTP 404 bootstrap)",
    };
    CheckReport::success(format!("{message}\n")).emit()
}
