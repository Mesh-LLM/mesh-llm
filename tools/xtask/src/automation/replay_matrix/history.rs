//! Normalize qualified replay artifacts into immutable history shards and gates.
use crate::command::DynResult;
use crate::repository::{check_args::Grammar, check_report::CheckReport};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    io::Write,
    path::{Path, PathBuf},
};

pub(in crate::automation) fn run(args: &[String]) -> DynResult<()> {
    const GRAMMAR: Grammar = Grammar {
        usage: "cargo xtool automation replay-matrix history --matrix PATH --replay-dir PATH --label LABEL --hardware PATH --source-sha SHA --replay PATH --output PATH [--backend-binary-sha256 SHA] [--baseline PATH] [--gate] [--github-output PATH]",
        values: &[
            "--matrix",
            "--replay-dir",
            "--summary-dir",
            "--label",
            "--hardware",
            "--source-sha",
            "--replay",
            "--output",
            "--backend-binary-sha256",
            "--baseline",
            "--github-output",
        ],
        flags: &["--help", "--gate"],
    };
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    if parsed.flag("--help") {
        return CheckReport::success(format!("{}\n", GRAMMAR.usage)).emit();
    }
    let github = parsed.last("--github-output").map(PathBuf::from);
    github_output(github.as_deref(), false)?;
    if !parsed.positionals.is_empty() {
        return GRAMMAR.error("unexpected positional arguments").emit();
    }
    let required = |name| parsed.last(name).ok_or_else(|| format!("missing {name}"));
    let root = parsed
        .last("--replay-dir")
        .or_else(|| parsed.last("--summary-dir"))
        .ok_or("missing --replay-dir")?;
    let input = super::history_input::load(
        Path::new(required("--matrix")?),
        Path::new(required("--replay")?),
        Path::new(required("--hardware")?),
        PathBuf::from(root),
        parsed.last("--label").unwrap_or("pr").into(),
        required("--source-sha")?.into(),
        parsed.last("--backend-binary-sha256").map(str::to_owned),
    )?;
    let mut rows = Vec::new();
    let mut integrity = Vec::new();
    let mut shared = None;
    let mut backend = None;
    for (model, original) in &input.models {
        match super::history_artifacts::load(&input, model) {
            Ok(family) => {
                integrity.extend(
                    family
                        .problems
                        .iter()
                        .map(|problem| format!("{}: {problem}", model.family)),
                );
                if !family.backend.is_empty() {
                    if backend
                        .as_ref()
                        .is_some_and(|digest| digest != &family.backend)
                    {
                        integrity.push("models used different backend binaries".into());
                    } else {
                        backend = Some(family.backend.clone());
                    }
                }
                match super::history_rows::build(&input, (model, original), &family) {
                    Ok(mut family_rows) => {
                        let identities = family_rows
                            .iter()
                            .map(|row| {
                                (
                                    row["replay"]["concurrency"].to_string(),
                                    row["session_cohort_sha256"].clone(),
                                )
                            })
                            .collect::<BTreeMap<_, _>>();
                        if shared
                            .as_ref()
                            .is_some_and(|cohorts| cohorts != &identities)
                        {
                            integrity.push(format!(
                                "{}: models used different session cohorts",
                                model.family
                            ));
                            for row in &mut family_rows {
                                row["complete"] = false.into();
                                row["artifact_result"] = "incomplete".into();
                            }
                        } else if family_rows.iter().all(|row| row["complete"] == true) {
                            shared = Some(identities);
                        }
                        rows.extend(family_rows);
                    }
                    Err(error) => integrity.push(format!("{}: {error}", model.family)),
                }
            }
            Err(error) => integrity.push(format!("{}: {error}", model.family)),
        }
    }
    for row in &rows {
        if row["complete"] != true {
            integrity.push(format!(
                "{}: incomplete run",
                super::history_baseline::key(row)?
            ));
        }
    }
    if rows.is_empty() {
        integrity.push("no replay rows were produced".into());
    }
    write_rows(Path::new(required("--output")?), &rows)?;
    let mut regressions = Vec::new();
    if let Some(baseline) = parsed.last("--baseline") {
        let prior = super::history_baseline::load(Path::new(baseline))?;
        for row in rows.iter().filter(|row| row["complete"] == true) {
            let key = super::history_baseline::key(row)?;
            regressions.extend(super::history_baseline::compare(
                row,
                prior.get(&key).map_or(&[], Vec::as_slice),
            )?);
        }
    }
    let gate = parsed.flag("--gate");
    if gate && integrity.is_empty() && !regressions.is_empty() {
        github_output(github.as_deref(), true)?;
    }
    for problem in integrity.iter().chain(&regressions) {
        writeln!(crate::cli_output::stderr(), "history: {problem}")?;
    }
    if !integrity.is_empty() || (gate && !regressions.is_empty()) {
        return Err("history qualification or regression gate failed; shard retained".into());
    }
    writeln!(
        crate::cli_output::stdout(),
        "wrote {} history rows to {}",
        rows.len(),
        required("--output")?
    )?;
    Ok(())
}
fn github_output(path: Option<&Path>, repair: bool) -> DynResult<()> {
    if let Some(path) = path {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        writeln!(file, "repair_required={repair}")?;
        file.flush()?;
    }
    Ok(())
}
fn write_rows(path: &Path, rows: &[Value]) -> DynResult<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::File::create(path)?;
    for row in rows {
        serde_json::to_writer(&mut file, row)?;
        file.write_all(b"\n")?;
    }
    file.sync_all()?;
    Ok(())
}
