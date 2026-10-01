use crate::command::DynResult;
use crate::repository::check_args::Grammar;
use crate::repository::check_report::CheckReport;
use serde::Deserialize;
use std::collections::BTreeSet;
use std::path::Path;

const GRAMMAR: Grammar = Grammar {
    usage: "cargo xtool automation replay-matrix pins --matrix <path> --canonical <path> --models-output <path> --dataset-output <path>",
    values: &[
        "--matrix",
        "--canonical",
        "--models-output",
        "--dataset-output",
    ],
    flags: &["--help"],
};

#[derive(Deserialize)]
struct Matrix {
    models: Vec<Model>,
    replay: ReplayDataset,
}

#[derive(Deserialize)]
struct Model {
    family: String,
    repo: String,
    revision: String,
    file: String,
    sha256: String,
}

#[derive(Deserialize)]
struct ReplayDataset {
    dataset: String,
    dataset_revision: String,
    dataset_file: String,
    dataset_sha256: String,
}

#[derive(Deserialize)]
struct Canonical {
    thoughtworks: Thoughtworks,
}

#[derive(Deserialize)]
struct Thoughtworks {
    dataset: Dataset,
}

#[derive(Deserialize, PartialEq, Eq)]
struct Dataset {
    repo: String,
    revision: String,
    filename: String,
    sha256: String,
}

pub(in crate::automation) fn run(args: &[String]) -> DynResult<()> {
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    if parsed.flag("--help") {
        return CheckReport::success(format!("usage: {}\n", GRAMMAR.usage)).emit();
    }
    if !parsed.positionals.is_empty() {
        return GRAMMAR.error("unexpected positional arguments").emit();
    }
    let required = |key| parsed.last(key).ok_or_else(|| format!("missing {key}"));
    let matrix_path = required("--matrix")?;
    super::input::load(Path::new(matrix_path)).map_err(|error| error.to_string())?;
    let matrix: Matrix = serde_json::from_slice(&std::fs::read(matrix_path)?)?;
    let canonical: Canonical = serde_json::from_slice(&std::fs::read(required("--canonical")?)?)?;
    let (models, dataset) = project(matrix, &canonical.thoughtworks.dataset)?;
    let models_output = required("--models-output")?;
    let dataset_output = required("--dataset-output")?;
    std::fs::write(models_output, models)?;
    std::fs::write(dataset_output, dataset)?;
    Ok(())
}

fn project(matrix: Matrix, canonical: &Dataset) -> DynResult<(String, String)> {
    if matrix.models.is_empty() {
        return Err("replay model roster must not be empty".into());
    }
    let mut families = BTreeSet::new();
    let mut models = String::new();
    for model in matrix.models {
        text(&model.family)?;
        text(&model.repo)?;
        text(&model.file)?;
        if !families.insert(model.family.clone()) {
            return Err("replay model families must be unique".into());
        }
        hex_pin(&model.revision, 40)?;
        hex_pin(&model.sha256, 64)?;
        models.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\n",
            model.family, model.repo, model.revision, model.file, model.sha256
        ));
    }
    let dataset = Dataset {
        repo: matrix.replay.dataset,
        revision: matrix.replay.dataset_revision,
        filename: matrix.replay.dataset_file,
        sha256: matrix.replay.dataset_sha256,
    };
    text(&dataset.repo)?;
    text(&dataset.filename)?;
    hex_pin(&dataset.revision, 40)?;
    hex_pin(&dataset.sha256, 64)?;
    if &dataset != canonical {
        return Err("nightly dataset pin must match the canonical replay harness pin".into());
    }
    Ok((
        models,
        format!(
            "{}\t{}\t{}\t{}\n",
            dataset.repo, dataset.revision, dataset.filename, dataset.sha256
        ),
    ))
}

fn text(value: &str) -> DynResult<()> {
    if value.is_empty() || value.chars().any(char::is_control) {
        return Err("replay pin text must be nonempty and contain no control characters".into());
    }
    Ok(())
}

fn hex_pin(value: &str, length: usize) -> DynResult<()> {
    if value.len() != length
        || !value
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
    {
        return Err(format!("replay pin must be {length} lowercase hexadecimal characters").into());
    }
    Ok(())
}
