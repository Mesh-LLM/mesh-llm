//! `models restore-inputs`: the Rust owner of the `resolve-model` step in
//! `.github/actions/restore-test-model/action.yml`. Exactly one source feeds
//! the cache/download steps: a direct URL plus safe filename, a
//! cadence-authorized single-file manifest artifact, or no model at all
//! (empty outputs, which skip every later step).

use super::manifest::Selection;
use super::resolve::{Request, execute};
use crate::repository::check_args::Grammar;
use crate::repository::check_report::CheckReport;
use std::fs::OpenOptions;
use std::io::Write;

const GRAMMAR: Grammar = Grammar {
    usage: "cargo xtool models restore-inputs --github-output <path> [--model-url <url>] \
            [--model-file <name>] [--model-manifest <path>] [--model-artifact-id <id>] \
            [--model-cadence <cadence>]",
    values: &[
        "--model-url",
        "--model-file",
        "--model-manifest",
        "--model-artifact-id",
        "--model-cadence",
        "--github-output",
    ],
    flags: &[],
};

/// The action inputs, where `''` means "not supplied".
struct Inputs<'a> {
    url: &'a str,
    file: &'a str,
    manifest: &'a str,
    artifact_id: &'a str,
    cadence: &'a str,
}

pub(super) fn run(args: &[String]) -> CheckReport {
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) if parsed.positionals.is_empty() => parsed,
        Ok(parsed) => {
            let message = format!("unrecognized arguments: {}", parsed.positionals.join(" "));
            return GRAMMAR.error(&message);
        }
        Err(report) => return report,
    };
    let Some(output) = parsed.last("--github-output") else {
        return GRAMMAR.error("the following arguments are required: --github-output");
    };
    let value = |name: &str| parsed.last(name).unwrap_or_default();
    let inputs = Inputs {
        url: value("--model-url"),
        file: value("--model-file"),
        manifest: value("--model-manifest"),
        artifact_id: value("--model-artifact-id"),
        cadence: value("--model-cadence"),
    };
    restore(&inputs, output)
}

fn restore(inputs: &Inputs<'_>, output: &str) -> CheckReport {
    if !inputs.url.is_empty() || !inputs.file.is_empty() {
        return direct(inputs, output);
    }
    if inputs.manifest.is_empty() {
        return append(output, &["url=", "file=", "sha256=", "size_bytes="]);
    }
    if inputs.cadence.is_empty() {
        return rejected("model_cadence is required with model_manifest");
    }
    execute(&Request {
        manifest: inputs.manifest,
        selection: Selection {
            artifact_id: Some(inputs.artifact_id).filter(|id| !id.is_empty()),
            cadence: inputs.cadence,
        },
        require_single_file: true,
        github_output: Some(output),
        output_prefix: "",
        verify_root: None,
    })
}

/// `^[A-Za-z0-9][A-Za-z0-9._-]*$`: one path component, never hidden.
fn safe_filename(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|first| first.is_ascii_alphanumeric())
        && chars.all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-'))
}

fn direct(inputs: &Inputs<'_>, output: &str) -> CheckReport {
    if inputs.url.is_empty() || inputs.file.is_empty() {
        return rejected("model_url and model_file must be supplied together");
    }
    if !safe_filename(inputs.file) {
        return rejected("model_file must be a safe single filename");
    }
    let url = format!("url={}", inputs.url);
    let file = format!("file={}", inputs.file);
    append(output, &[&url, &file, "sha256=", "size_bytes="])
}

fn rejected(message: &str) -> CheckReport {
    CheckReport {
        stdout: String::new(),
        stderr: format!("{message}\n"),
        code: 2,
    }
}

fn append(path: &str, lines: &[&str]) -> CheckReport {
    let written = OpenOptions::new()
        .append(true)
        .create(true)
        .open(path)
        .and_then(|mut output| lines.iter().try_for_each(|line| writeln!(output, "{line}")));
    match written {
        Ok(()) => CheckReport::success(String::new()),
        Err(error) => CheckReport {
            stdout: String::new(),
            stderr: format!("cannot append GitHub outputs to {path}: {error}\n"),
            code: 1,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::safe_filename;

    #[test]
    fn migration_models_direct_filename_is_one_visible_component() {
        for good in ["m.gguf", "Model-1_2.gguf", "9"] {
            assert!(safe_filename(good), "{good}");
        }
        for bad in [
            "",
            ".m",
            "../m",
            "a/b",
            "a\\b",
            "m.gguf\nsha256=x",
            "-m",
            "é",
        ] {
            assert!(!safe_filename(bad), "{bad:?}");
        }
    }
}
