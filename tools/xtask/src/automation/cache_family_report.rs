//! Read-only cache-family measurement rendering; never runs a benchmark/model.
mod input;
mod measurement;
mod render;
use crate::{
    command::DynResult,
    repository::{check_args::Grammar, check_report::CheckReport},
};
use std::{fs, path::Path};

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    const GRAMMAR: Grammar = Grammar {
        usage: "cargo xtool automation cache-family-report --input PATH... [--output PATH] [--use-case-corpus PATH]",
        values: &["--input", "--output", "--use-case-corpus"],
        flags: &["--help"],
    };
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    if parsed.flag("--help") {
        return CheckReport::success(format!("{}\n", GRAMMAR.usage)).emit();
    }
    let inputs = parsed.all("--input");
    if !parsed.positionals.is_empty() || inputs.is_empty() || inputs.len() > 32 {
        return GRAMMAR
            .error("provide 1..=32 input files and no positional arguments")
            .emit();
    }
    let mut rows = Vec::<input::Row>::new();
    for path in inputs {
        let mut given = input::load::<Vec<input::Row>>(Path::new(path))?;
        if rows.len() + given.len() > 100_000 {
            return Err("cache report exceeds 100000 rows".into());
        }
        for row in &given {
            input::validate(row)?;
        }
        rows.append(&mut given);
    }
    let default = std::env::current_dir()?.join("evals/skippy-usecase-corpus.json");
    let corpus = input::load::<input::Corpus>(
        parsed
            .last("--use-case-corpus")
            .map(Path::new)
            .unwrap_or(&default),
    )?;
    if corpus.use_cases.len() > 10_000 {
        return Err("cache source corpus exceeds 10000 entries".into());
    }
    let text = render::report(&rows, &corpus)?;
    if let Some(path) = parsed.last("--output") {
        let path = Path::new(path);
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, text)?;
        CheckReport::success(format!("Wrote {}\n", path.display())).emit()
    } else {
        CheckReport::success(format!("{text}\n")).emit()
    }
}

/// Existing typed renderer boundary for an owned native benchmark producer.
/// Supplied rows are observations, not a promotion/real-platform certificate.
pub(in crate::automation) fn producer(
    rows: &serde_json::Value,
    corpus: &serde_json::Value,
) -> DynResult<String> {
    let rows: Vec<input::Row> = serde_json::from_value(rows.clone())?;
    if rows.len() > 100_000 {
        return Err("cache producer report exceeds100000rows".into());
    }
    for row in &rows {
        input::validate(row)?;
    }
    let corpus: input::Corpus = serde_json::from_value(corpus.clone())?;
    if corpus.use_cases.len() > 10_000 {
        return Err("cache producer corpus exceeds10000entries".into());
    }
    render::producer(&rows, &corpus)
}
