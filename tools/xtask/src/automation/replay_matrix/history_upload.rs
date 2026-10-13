use crate::{
    automation::private_state::PrivateState,
    command::DynResult,
    process,
    repository::{check_args::Grammar, check_report::CheckReport},
};
use std::{
    ffi::OsString,
    path::{Path, PathBuf},
    time::Duration,
};
#[derive(Clone, Copy)]
pub(super) enum Kind {
    Shard,
    Card,
}
pub(super) struct Input<'a> {
    pub repo: &'a str,
    pub output: &'a Path,
    pub kind: Kind,
    pub date: &'a str,
    pub run_id: Option<&'a str>,
    pub source_sha: Option<&'a str>,
}
fn destination(input: &Input<'_>) -> DynResult<(PathBuf, String, String)> {
    super::history_hub::repository(input.repo)?;
    let date = input.date;
    if date.len() != 10
        || date.as_bytes()[4] != b'-'
        || date.as_bytes()[7] != b'-'
        || date
            .bytes()
            .enumerate()
            .any(|(index, byte)| index != 4 && index != 7 && !byte.is_ascii_digit())
    {
        return Err("publication date must be YYYY-MM-DD".into());
    }
    let timestamp = format!("{date}T00:00:00Z");
    if crate::ci_operations::ci_metrics_time::timestamp(&timestamp).is_err() {
        return Err("publication date is invalid".into());
    }
    if !std::fs::symlink_metadata(input.output)?
        .file_type()
        .is_dir()
    {
        return Err("publication output must be a non-symlink directory".into());
    }
    let root = input.output.canonicalize()?;
    let (relative, destination, message) = match input.kind {
        Kind::Shard => {
            let id = input.run_id.ok_or("shard publication requires run id")?;
            let sha = input
                .source_sha
                .ok_or("shard publication requires source SHA")?;
            if id.is_empty()
                || !id.bytes().all(|byte| byte.is_ascii_digit())
                || sha.len() != 40
                || !sha
                    .bytes()
                    .all(|byte| matches!(byte,b'0'..=b'9'|b'a'..=b'f'))
            {
                return Err("invalid immutable shard run/source identity".into());
            }
            (
                "summary/history.jsonl",
                format!("data/runs/{date}/{id}.jsonl"),
                format!("run {date} {id} @ {sha}"),
            )
        }
        Kind::Card => {
            if input.run_id.is_some() || input.source_sha.is_some() {
                return Err("card publication does not accept shard identities".into());
            }
            ("card.md", "card.md".into(), format!("card refresh {date}"))
        }
    };
    let path = root.join(relative);
    if !std::fs::symlink_metadata(&path)?.file_type().is_file() || path.canonicalize()? != path {
        return Err(
            "publication source must be a regular file inside the canonical output directory"
                .into(),
        );
    }
    let length = std::fs::metadata(&path)?.len();
    if length == 0 || length > 64 * 1024 * 1024 {
        return Err("publication source is empty or exceeds 64 MiB".into());
    }
    if matches!(input.kind, Kind::Shard) {
        let rows = super::history_artifacts::records(&path)?;
        if rows.is_empty()
            || rows.iter().any(|row| {
                row["schema_version"] != 3
                    || row["complete"] != true
                    || row["source_sha"] != input.source_sha.unwrap_or_default()
            })
        {
            return Err(
                "only complete current-source typed history shards may be published".into(),
            );
        }
    }
    Ok((path, destination, message))
}
pub(super) fn upload(
    input: &Input<'_>,
    hf: &Path,
    token: OsString,
    timeout: Duration,
    cancellation: &process::Cancellation,
) -> DynResult<()> {
    if token.is_empty() {
        return Err("publication requires an explicit HF_TOKEN".into());
    }
    let (source, destination, message) = destination(input)?;
    let state = PrivateState::create(&std::env::temp_dir(), "replay-history-publish")?;
    state.prepare()?;
    let result: DynResult<()> = (|| {
        let report = super::history_hub::execute(
            hf,
            vec![
                "upload".into(),
                input.repo.into(),
                source
                    .to_str()
                    .ok_or("non-Unicode publication source")?
                    .into(),
                destination,
                "--repo-type".into(),
                "dataset".into(),
                "--commit-message".into(),
                message,
            ],
            &state,
            Some(token),
            timeout,
            cancellation,
        )?;
        if !report.success() || !super::history_hub::terminal(&report) {
            return Err("history publication failed; no success receipt retained".into());
        }
        Ok(())
    })();
    state
        .finish(result)
        .map_err(|error| format!("history publication/finalization failed: {error:?}").into())
}
pub(in crate::automation) fn run(args: &[String]) -> DynResult<()> {
    const GRAMMAR: Grammar = Grammar {
        usage: "cargo xtool automation replay-matrix history-upload --dataset-repo OWNER/NAME --output-dir PATH --kind shard|card [--run-date YYYY-MM-DD] [--run-id ID --source-sha SHA] [--hf PATH] [--timeout SECONDS]",
        values: &[
            "--dataset-repo",
            "--output-dir",
            "--kind",
            "--run-date",
            "--run-id",
            "--source-sha",
            "--hf",
            "--timeout",
        ],
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
    let kind = match parsed.last("--kind").ok_or("missing --kind")? {
        "shard" => Kind::Shard,
        "card" => Kind::Card,
        _ => return Err("publication kind must be shard or card".into()),
    };
    let today = crate::ci_operations::ci_metrics_time::isoformat(
        crate::ci_operations::ci_metrics_time::Instant::now(),
    );
    let input = Input {
        repo: parsed
            .last("--dataset-repo")
            .ok_or("missing --dataset-repo")?,
        output: Path::new(parsed.last("--output-dir").ok_or("missing --output-dir")?),
        kind,
        date: parsed.last("--run-date").unwrap_or(&today[..10]),
        run_id: parsed.last("--run-id"),
        source_sha: parsed.last("--source-sha"),
    };
    let hf = super::history_hub::tool(parsed.last("--hf"), "hf")?;
    let token = std::env::var_os("HF_TOKEN").ok_or("publication step requires HF_TOKEN")?;
    let timeout = super::history_hub::seconds(parsed.last("--timeout").unwrap_or("570"))?;
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let result = upload(&input, &hf, token, timeout, &interrupt.cancellation());
    let interrupted = interrupt.finish();
    interrupted?;
    result?;
    CheckReport::success("history publication succeeded\n".into()).emit()
}
