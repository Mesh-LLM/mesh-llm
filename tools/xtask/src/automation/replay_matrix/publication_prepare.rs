//! Prepare data for the separate trusted repair publisher; never publish here.
use crate::{
    command::DynResult,
    repository::{check_args::Grammar, check_report::CheckReport},
};
use std::path::Path;

const GRAMMAR: Grammar = Grammar {
    usage: "cargo xtool automation replay-matrix publication-prepare --publication-dir PATH --run-id ID --run-attempt N --base-sha SHA --run-date DATE --server-url URL --dataset-repo OWNER/NAME --fix-summary TEXT --files-changed TEXT --history-outcome success --rerun-outcome success [--template PATH] [--regressing-cohorts TEXT] [--bootstrap-state TEXT]",
    values: &[
        "--publication-dir",
        "--run-id",
        "--run-attempt",
        "--base-sha",
        "--run-date",
        "--server-url",
        "--dataset-repo",
        "--fix-summary",
        "--files-changed",
        "--history-outcome",
        "--rerun-outcome",
        "--template",
        "--regressing-cohorts",
        "--bootstrap-state",
    ],
    flags: &["--help"],
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Outcome {
    Success,
    Failure,
    Cancelled,
    Skipped,
}
impl Outcome {
    fn parse(value: &str) -> DynResult<Self> {
        match value {
            "success" => Ok(Self::Success),
            "failure" => Ok(Self::Failure),
            "cancelled" => Ok(Self::Cancelled),
            "skipped" => Ok(Self::Skipped),
            _ => Err("invalid replay step outcome".into()),
        }
    }
}
fn admit(history: Outcome, rerun: Outcome) -> DynResult<()> {
    if history != Outcome::Success || rerun != Outcome::Success {
        return Err(
            "repair publication requires successful complete replay and history gate".into(),
        );
    }
    Ok(())
}
const BODY: &str = "The nightly agentic replay regressed; Goose analyzed the evidence and this fix passes the re-run benchmark.\n\nResults and repair logs are retained in the replay-artifacts workflow artifact.";

fn render(template: Option<&str>, values: &[(&str, String)]) -> DynResult<String> {
    let Some(template) = template else {
        return Ok(format!("{BODY}\n"));
    };
    if template.len() > 1024 * 1024 || template.contains('\0') {
        return Err("repair template is invalid or exceeds 1 MiB".into());
    }
    // Replace source template spans once. Replacement text is data, including
    // sed metacharacters or strings resembling another template placeholder.
    let mut output = String::new();
    let mut remaining = template;
    while let Some(start) = remaining.find("{{") {
        output.push_str(&remaining[..start]);
        let tail = &remaining[start + 2..];
        let Some(end) = tail.find("}}") else {
            output.push_str(&remaining[start..]);
            remaining = "";
            break;
        };
        let key = &tail[..end];
        if let Some((_, value)) = values.iter().find(|(name, _)| *name == key) {
            output.push_str(value);
        } else {
            output.push_str(&remaining[start..start + end + 4]);
        }
        remaining = &tail[end + 2..];
    }
    output.push_str(remaining);
    output.push_str("\n---\n");
    output.push_str(BODY);
    output.push('\n');
    if output.len() > 1024 * 1024 || output.contains('\0') {
        return Err("repair body is invalid or exceeds 1 MiB".into());
    }
    Ok(output)
}

fn read_template(path: &str) -> DynResult<String> {
    if std::fs::metadata(path)?.len() > 1024 * 1024 {
        return Err("repair template exceeds 1 MiB".into());
    }
    Ok(std::fs::read_to_string(path)?)
}

pub(in crate::automation) fn run(args: &[String]) -> DynResult<()> {
    let parsed = match GRAMMAR.parse(args) {
        Ok(value) => value,
        Err(report) => return report.emit(),
    };
    if parsed.flag("--help") {
        return CheckReport::success(format!("{}\n", GRAMMAR.usage)).emit();
    }
    if !parsed.positionals.is_empty() {
        return GRAMMAR.error("unexpected positional arguments").emit();
    }
    let required = |key| parsed.last(key).ok_or_else(|| format!("missing {key}"));
    admit(
        Outcome::parse(required("--history-outcome")?)?,
        Outcome::parse(required("--rerun-outcome")?)?,
    )?;
    let directory = Path::new(required("--publication-dir")?);
    let date = required("--run-date")?;
    crate::ci_operations::ci_metrics_time::timestamp(&format!("{date}T00:00:00Z"))
        .map_err(|_| "invalid publication date")?;
    let dataset = required("--dataset-repo")?;
    super::history_hub::repository(dataset)?;
    let run = required("--run-id")?;
    let attempt = required("--run-attempt")?;
    let base = required("--base-sha")?;
    let server = required("--server-url")?.trim_end_matches('/');
    if !server.is_empty()
        && (!server.starts_with("https://") || server.chars().any(char::is_control))
    {
        return Err("invalid repair workflow server URL".into());
    }
    let values = [
        ("RESOLUTION_STATUS", "fix verified".into()),
        (
            "RUN_URL",
            format!("{server}/Mesh-LLM/mesh-llm/actions/runs/{run}/attempts/{attempt}"),
        ),
        ("RUN_DATE", date.into()),
        ("RUN_ID", run.into()),
        ("SOURCE_SHA", base.into()),
        ("DATASET_REPO", dataset.into()),
        (
            "REGRESSING_COHORTS",
            parsed
                .last("--regressing-cohorts")
                .unwrap_or("unavailable")
                .into(),
        ),
        ("GATE_OUTPUT", "see run artifacts".into()),
        ("DIAGNOSIS", "see Goose output in repair.log".into()),
        ("FIX_SUMMARY", required("--fix-summary")?.into()),
        ("FILES_CHANGED", required("--files-changed")?.into()),
        ("RATIONALE", "automated repair attempt".into()),
        ("RESULT_ROWS", "see history-repair.jsonl artifact".into()),
        ("RERUN_GATE_RESULT", "pass".into()),
        (
            "BOOTSTRAP_STATE",
            parsed
                .last("--bootstrap-state")
                .unwrap_or("unavailable")
                .into(),
        ),
    ];
    let template = parsed.last("--template").map(read_template).transpose()?;
    let body = render(template.as_deref(), &values)?;
    if !std::fs::symlink_metadata(directory)?.file_type().is_dir() {
        return Err("repair publication must be a non-symlink directory".into());
    }
    use std::io::Write;
    let body_path = directory.join("pr-body.md");
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&body_path)?;
    file.write_all(body.as_bytes())?;
    super::publication::prepare_status(directory, base, run, attempt)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_successful_rerun_and_history_qualify_publication() {
        for history in [
            Outcome::Success,
            Outcome::Failure,
            Outcome::Cancelled,
            Outcome::Skipped,
        ] {
            for rerun in [
                Outcome::Success,
                Outcome::Failure,
                Outcome::Cancelled,
                Outcome::Skipped,
            ] {
                assert_eq!(
                    admit(history, rerun).is_ok(),
                    history == Outcome::Success && rerun == Outcome::Success
                );
            }
        }
        assert!(Outcome::parse("true").is_err());
    }
    #[test]
    fn template_values_are_literal_nonrecursive_data_and_fallback_is_bounded() {
        let text = render(
            Some("{{FIX_SUMMARY}} | {{SOURCE_SHA}}"),
            &[
                ("FIX_SUMMARY", "fix & | \\ {{SOURCE_SHA}}".into()),
                ("SOURCE_SHA", "base".into()),
            ],
        )
        .unwrap();
        assert!(text.starts_with("fix & | \\ {{SOURCE_SHA}} | base"));
        assert!(
            render(Some("{{UNEXPECTED}}"), &[])
                .unwrap()
                .starts_with("{{UNEXPECTED}}")
        );
        assert!(render(Some("\0"), &[]).is_err());
        assert!(render(None, &[]).unwrap().contains("re-run benchmark"));
    }
}
