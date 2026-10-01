mod options;
mod output;
#[cfg(test)]
mod refusal_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod whitespace_tests;

use super::{Error, Interrupt, invocation, parent, publish_count, scope};
use crate::automation::rewriter_report::generator::{self, Pass, Summary};
use crate::command::DynResult;
use crate::process::Cancellation;
use crate::repository::python_text::strip;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

pub(super) const USAGE: &str = "cargo xtool automation native-generator generate --source-root <dir> --build-dir <dir> --rewriter <absolute-executable> --git <absolute-executable> --report <json> --output <patch> --max-diff-bytes <positive-integer> [--diff-base <revision>] [--timeout <seconds>] [--extra-arg <argument>]... [--shard-output-dir <dir> --family-source-map <json> --family-manifest <json>]";

#[derive(Debug)]
struct Generated {
    output: PathBuf,
    first_summary: Summary,
    second_summary: Summary,
    shards: usize,
}

pub(super) fn run(arguments: &[String]) -> DynResult<()> {
    if arguments == ["--help"] {
        println!("{USAGE}");
        return Ok(());
    }
    let options = options::parse(arguments)?;
    let generated = scope::run(|interrupt| generate(&options, interrupt))?;
    println!("{}", output::encode(&generated)?);
    Ok(())
}

fn generate(options: &options::Options, interrupt: &Interrupt) -> Result<Generated, Error> {
    execute(options, &interrupt.cancellation(), |diff| {
        interrupt.check()?;
        publish_count(diff, &options.publication)
    })
}

fn execute(
    options: &options::Options,
    cancellation: &Cancellation,
    publish: impl FnOnce(&[u8]) -> Result<usize, Error>,
) -> Result<Generated, Error> {
    let input = &options.publication.git;
    let status = invocation::git(
        input,
        &["status", "--porcelain", "--untracked-files=no"],
        cancellation,
    )?;
    if !strip(std::str::from_utf8(status.as_bytes())?).is_empty() {
        return Err(Error::Dirty);
    }
    let marker = input.source_root.join(".mesh-llm-upstream-sha");
    let identity = if marker.try_exists()? {
        strip(&std::fs::read_to_string(marker)?).to_owned()
    } else {
        let head = invocation::git(input, &["rev-parse", "HEAD"], cancellation)?;
        strip(std::str::from_utf8(head.as_bytes())?).to_owned()
    };
    let sources = sources(&input.source_root)?;
    std::fs::create_dir_all(parent(&options.report))?;
    let first = PassInput {
        report: &options.report,
        identity: &identity,
        sources: &sources,
        pass: Pass::First,
    };
    rewrite(options, &first, cancellation)?;
    let first_summary = generator::load(&options.report, Pass::First)?;
    let second_report = second_report(&options.report)?;
    let second = PassInput {
        report: &second_report,
        identity: &identity,
        sources: &sources,
        pass: Pass::Second,
    };
    rewrite(options, &second, cancellation)?;
    let second_summary = generator::load(&second_report, Pass::Second)?;
    let diff = invocation::git(
        input,
        &[
            "diff",
            "--no-ext-diff",
            "--binary",
            "--full-index",
            &input.base,
            "--",
            "src/models",
        ],
        cancellation,
    )?;
    let shards = publish(diff.as_bytes())?;
    Ok(Generated {
        output: options.publication.output.clone(),
        first_summary,
        second_summary,
        shards,
    })
}

struct PassInput<'a> {
    report: &'a Path,
    identity: &'a str,
    sources: &'a [PathBuf],
    pass: Pass,
}

fn rewrite(
    options: &options::Options,
    input: &PassInput<'_>,
    cancellation: &Cancellation,
) -> Result<(), Error> {
    invocation::execute(
        &options.publication.git,
        invocation::Invocation {
            executable: &options.rewriter,
            arguments: argv(options, input),
            phase: match input.pass {
                Pass::First => "first rewriter",
                Pass::Second => "second rewriter",
            },
            capture: false,
        },
        cancellation,
    )?;
    Ok(())
}

fn argv(options: &options::Options, input: &PassInput<'_>) -> Vec<OsString> {
    let mut arguments = vec![
        "--source-root".into(),
        options.publication.git.source_root.as_os_str().to_owned(),
        "--llama-commit".into(),
        input.identity.into(),
        "--report".into(),
        input.report.as_os_str().to_owned(),
        "-p".into(),
        options.build.as_os_str().to_owned(),
    ];
    match input.pass {
        Pass::First => arguments.push("--apply".into()),
        Pass::Second => {}
    }
    for argument in &options.extra {
        arguments.push("--extra-arg".into());
        arguments.push(argument.into());
    }
    arguments.extend(
        input
            .sources
            .iter()
            .map(|source| source.as_os_str().to_owned()),
    );
    arguments
}

fn sources(root: &Path) -> Result<Vec<PathBuf>, Error> {
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(root.join("src/models"))? {
        let path = entry?.path();
        if path.extension().is_some_and(|extension| extension == "cpp") {
            paths.push(path);
        }
    }
    paths.sort();
    if paths.is_empty() {
        return Err(Error::Sources);
    }
    Ok(paths)
}

fn second_report(first: &Path) -> Result<PathBuf, Error> {
    let stem = first
        .file_stem()
        .ok_or(Error::Arguments("report requires filename"))?;
    let mut name = stem.to_os_string();
    name.push("-second.json");
    Ok(first.with_file_name(name))
}
