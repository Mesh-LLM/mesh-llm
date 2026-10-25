use super::report_chart::Metric;
use super::report_input::Document;
use super::report_row::Row;
use crate::command::DynResult;
use crate::repository::{check_args::Grammar, check_report::CheckReport};
use std::path::{Path, PathBuf};

pub(in crate::automation) fn run(args: &[String]) -> DynResult<()> {
    const GRAMMAR: Grammar = Grammar {
        usage: "cargo xtool automation replay-matrix report --artifact PATH",
        values: &["--artifact"],
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
    let output = Path::new(parsed.last("--artifact").ok_or("missing --artifact")?);
    let document: Document = serde_json::from_slice(&std::fs::read(output.join("run.json"))?)?;
    let report = write(output, &document)?;
    CheckReport::success(format!("{}\n", report.display())).emit()
}

pub(super) fn write(output: &Path, document: &Document) -> DynResult<PathBuf> {
    if document
        .gates
        .as_ref()
        .is_some_and(|gates| gates.evaluated && gates.passed.is_none())
    {
        return Err("evaluated acceptance gates require a boolean outcome".into());
    }
    let rows: Vec<Row> = super::pooled_rows::pool(&document.results)?
        .into_iter()
        .map(serde_json::from_value)
        .collect::<Result<_, _>>()?;
    let markdown = super::report_markdown::render(document, &rows)?;
    let summary = output.join("summary");
    let charts = summary.join("charts");
    std::fs::create_dir_all(&charts)?;
    super::report_csv::write(
        &mut std::fs::File::create(summary.join("comparison.csv"))?,
        &rows,
    )?;
    crate::command::write_json_file(&summary.join("comparison.json"), &rows)?;
    for metric in [Metric::Decode, Metric::Output, Metric::Ttft] {
        std::fs::write(
            charts.join(metric.artifact()),
            super::report_chart::render(&metric, (&rows, &document.builds))?,
        )?;
    }
    let path = summary.join("REPORT.md");
    std::fs::write(&path, markdown)?;
    inventory(output)?;
    Ok(path)
}

pub(super) fn inventory(root: &Path) -> DynResult<()> {
    let mut files = Vec::new();
    collect(root, &mut files)?;
    files.sort();
    let mut inventory = String::new();
    for path in files {
        if path
            .file_name()
            .is_some_and(|name| name == "artifact-sha256.txt")
        {
            continue;
        }
        let digest = crate::product::digest::file_sha256(&path).map_err(|error| error.error)?;
        let relative = path
            .strip_prefix(root)?
            .components()
            .map(|component| {
                component
                    .as_os_str()
                    .to_str()
                    .ok_or("artifact path is not UTF-8")
            })
            .collect::<Result<Vec<_>, _>>()?
            .join("/");
        inventory.push_str(&format!("{digest}  {relative}\n"));
    }
    std::fs::write(root.join("artifact-sha256.txt"), inventory)?;
    Ok(())
}

fn collect(directory: &Path, files: &mut Vec<PathBuf>) -> DynResult<()> {
    for entry in std::fs::read_dir(directory)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        if kind.is_symlink() {
            return Err("artifact inventory rejects symbolic links".into());
        }
        if kind.is_dir() {
            collect(&entry.path(), files)?;
        } else if kind.is_file() {
            files.push(entry.path());
        }
    }
    Ok(())
}
