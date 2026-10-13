//! Product composition arguments: exact named inputs, optional byte verification,
//! and help. Usage failures exit with status 2.

use crate::ci_operations::build_cache_options::{Kind, classify, help_flag, is_option_like};
use crate::repository::check_report::CheckReport;

const PROG: &str = "product compose";
const USAGE: &str = "\
usage: product compose [-h] --bundle BUNDLE --host HOST
                                 --runtime RUNTIME --version VERSION
                                 --backend BACKEND [--check]
";
const HELP: &str = "
options:
  -h, --help         show this help message and exit
  --bundle BUNDLE
  --host HOST
  --runtime RUNTIME
  --version VERSION
  --backend BACKEND
  --check            Validate the existing product manifest without rewriting
                     it.
";
const OPTIONS: [&str; 8] = [
    "-h",
    "--help",
    "--bundle",
    "--host",
    "--runtime",
    "--version",
    "--backend",
    "--check",
];
const REQUIRED: [&str; 5] = ["--bundle", "--host", "--runtime", "--version", "--backend"];

/// The parsed command line; paths stay as given, like `type=Path`.
pub(super) struct Args {
    pub(super) bundle: String,
    pub(super) host: String,
    pub(super) runtime: String,
    pub(super) version: String,
    pub(super) backend: String,
    pub(super) check: bool,
}

fn fail(message: &str) -> CheckReport {
    CheckReport {
        stdout: String::new(),
        stderr: format!("{USAGE}{PROG}: error: {message}\n"),
        code: 2,
    }
}

/// Parses the composition inputs or returns a help/usage report.
pub(super) fn parse(args: &[String]) -> Result<Args, CheckReport> {
    let mut values: [Option<String>; 5] = Default::default();
    let mut check = false;
    let mut extras: Vec<&str> = Vec::new();
    let mut positional_only = false;
    let mut index = 0;
    while let Some(arg) = args.get(index) {
        index += 1;
        if positional_only || arg == "--" {
            positional_only = true;
            extras.push(arg);
            continue;
        }
        match classify(arg, &OPTIONS) {
            Kind::Positional | Kind::Unknown => extras.push(arg),
            Kind::Known("-h" | "--help", explicit, sep) => {
                help_flag(explicit, sep, "-h/--help", &fail)?;
                return Err(CheckReport::success(format!("{USAGE}{HELP}")));
            }
            Kind::Known("--check", explicit, sep) => {
                help_flag(explicit, sep, "--check", &fail)?;
                check = true;
            }
            Kind::Known(option, explicit, _) => {
                let value = take_value(explicit, args, &mut index, option)?;
                if let Some(slot) = REQUIRED.iter().position(|name| *name == option) {
                    values[slot] = Some(value);
                }
            }
        }
    }
    finish(values, check, &extras)
}

fn take_value(
    explicit: Option<String>,
    args: &[String],
    index: &mut usize,
    option: &str,
) -> Result<String, CheckReport> {
    if let Some(value) = explicit {
        return Ok(value);
    }
    match args.get(*index) {
        Some(next) if !is_option_like(next, &OPTIONS) => {
            *index += 1;
            Ok(next.clone())
        }
        _ => Err(fail(&format!("argument {option}: expected one argument"))),
    }
}

fn finish(values: [Option<String>; 5], check: bool, extras: &[&str]) -> Result<Args, CheckReport> {
    let missing: Vec<&str> = REQUIRED
        .iter()
        .zip(&values)
        .filter_map(|(name, value)| value.is_none().then_some(*name))
        .collect();
    if !missing.is_empty() {
        return Err(fail(&format!(
            "the following arguments are required: {}",
            missing.join(", ")
        )));
    }
    if !extras.is_empty() {
        return Err(fail(&format!(
            "unrecognized arguments: {}",
            extras.join(" ")
        )));
    }
    let [bundle, host, runtime, version, backend] = values.map(Option::unwrap_or_default);
    Ok(Args {
        bundle,
        host,
        runtime,
        version,
        backend,
        check,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(args: &[&str]) -> Vec<String> {
        args.iter().map(|arg| (*arg).to_owned()).collect()
    }

    #[test]
    fn migration_product_argv_prefixes_and_check() {
        let parsed = parse(&strings(&[
            "--bundle=b",
            "--host",
            "h",
            "--runtime",
            "r",
            "--version",
            "-1",
            "--backend",
            "cpu",
            "--check",
        ]))
        .unwrap_or_else(|report| panic!("{}", report.stderr));
        assert_eq!(parsed.bundle, "b");
        assert_eq!(parsed.version, "-1");
        assert!(parsed.check);
        let report = parse(&strings(&["--check=1"])).err().map(|r| r.stderr);
        let expected = "argument --check: ignored explicit argument \"1\"";
        assert!(report.is_some_and(|text| text.contains(expected)));
    }
}
