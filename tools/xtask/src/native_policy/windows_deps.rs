//! `native windows-runtime-deps`: the Rust owner of
//! `scripts/windows-native-runtime-deps.py`. Keeps its argparse surface (a
//! required `{collect,verify}` subcommand), streams and statuses: handled
//! failures print one line to stderr and exit 1. PE imports are read
//! in-process as the legacy script did; `shutil.which` for the compiler
//! search directory goes through the [`Toolchain`] adapter.

use super::argv::{Grammar, Opt, Parsed};
use super::linux_deps_elf::file_name;
use super::toolchain::Toolchain;
use super::windows_deps_policy::Runtime;
use crate::ci_operations::build_cache_options::{Kind, classify, help_flag};
use crate::ci_operations::runner_identity_argv::error;
use crate::ci_plan::catalog::python_path_display;
use crate::repository::check_report::CheckReport;
use crate::repository::text::repr;
use std::path::Path;

const PROG: &str = "windows-native-runtime-deps.py";
const USAGE: &str = "usage: windows-native-runtime-deps.py [-h] {collect,verify} ...\n";
const HELP: &str = "\
usage: windows-native-runtime-deps.py [-h] {collect,verify} ...

Collect and verify non-system DLL dependencies in Windows native runtimes.

positional arguments:
  {collect,verify}

options:
  -h, --help        show this help message and exit
";

const COLLECT_USAGE: &str = "\
usage: windows-native-runtime-deps.py collect [-h] --lib-dir LIB_DIR
                                              [--search-dir SEARCH_DIR]
                                              [--scan-dir SCAN_DIR]
";
const VERIFY_USAGE: &str = "\
usage: windows-native-runtime-deps.py verify [-h] --lib-dir LIB_DIR
                                             [--scan-dir SCAN_DIR]
";

const COLLECT: Grammar = Grammar {
    prog: "windows-native-runtime-deps.py collect",
    usage: COLLECT_USAGE,
    help: concat!(
        "usage: windows-native-runtime-deps.py collect [-h] --lib-dir LIB_DIR\n",
        "                                              [--search-dir SEARCH_DIR]\n",
        "                                              [--scan-dir SCAN_DIR]\n",
        "\noptions:\n",
        "  -h, --help            show this help message and exit\n",
        "  --lib-dir LIB_DIR\n  --search-dir SEARCH_DIR\n  --scan-dir SCAN_DIR\n",
    ),
    options: &[
        Opt::required("--lib-dir"),
        Opt::value("--search-dir"),
        Opt::value("--scan-dir"),
    ],
    positional: None,
};

const VERIFY: Grammar = Grammar {
    prog: "windows-native-runtime-deps.py verify",
    usage: VERIFY_USAGE,
    help: concat!(
        "usage: windows-native-runtime-deps.py verify [-h] --lib-dir LIB_DIR\n",
        "                                             [--scan-dir SCAN_DIR]\n",
        "\noptions:\n",
        "  -h, --help           show this help message and exit\n",
        "  --lib-dir LIB_DIR\n  --scan-dir SCAN_DIR\n",
    ),
    options: &[Opt::required("--lib-dir"), Opt::value("--scan-dir")],
    positional: None,
};

fn top_fail(message: &str) -> CheckReport {
    error(USAGE, PROG, message)
}

/// The top-level parser: options before the subcommand are only `-h`
/// (others are extras), the first positional word (after one optional
/// `--`) selects the subcommand, and every later word belongs to it.
/// Returns whether `collect` was selected.
fn parse(args: &[String]) -> Result<(bool, Parsed), CheckReport> {
    let names = ["-h", "--help"];
    let mut extras: Vec<String> = Vec::new();
    let mut index = 0;
    while let Some(arg) = args.get(index) {
        let kind = classify(arg, &names);
        if arg == "--" || matches!(kind, Kind::Positional) {
            let start = index + usize::from(arg == "--");
            let Some(command) = args.get(start) else {
                break;
            };
            return subcommand(command, &args[start + 1..], extras);
        }
        index += 1;
        match kind {
            Kind::Known(_, explicit, sep) => {
                help_flag(explicit, sep, "-h/--help", &top_fail)?;
                return Err(CheckReport::success(HELP.to_owned()));
            }
            _ => extras.push(arg.clone()),
        }
    }
    Err(top_fail("the following arguments are required: command"))
}

fn subcommand(
    command: &str,
    rest: &[String],
    mut extras: Vec<String>,
) -> Result<(bool, Parsed), CheckReport> {
    let collect = match command {
        "collect" => true,
        "verify" => false,
        _ => {
            return Err(top_fail(&format!(
                "argument command: invalid choice: {} (choose from 'collect', 'verify')",
                repr(command)
            )));
        }
    };
    let grammar = if collect { &COLLECT } else { &VERIFY };
    let (parsed, unknown) = grammar.parse_known(rest)?;
    extras.extend(unknown);
    if !extras.is_empty() {
        return Err(top_fail(&format!(
            "unrecognized arguments: {}",
            extras.join(" ")
        )));
    }
    Ok((collect, parsed))
}

fn path_text(value: &str) -> String {
    python_path_display(Path::new(value))
}

pub(super) fn run(args: &[String], tools: &dyn Toolchain) -> CheckReport {
    let (collect, parsed) = match parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report,
    };
    let lib_dir = path_text(parsed.value("--lib-dir").unwrap_or_default());
    let mut scan_dirs = vec![lib_dir.clone()];
    scan_dirs.extend(parsed.values("--scan-dir").into_iter().map(path_text));
    let runtime = Runtime { lib_dir, scan_dirs };
    let outcome = if collect {
        let search: Vec<String> = parsed
            .values("--search-dir")
            .into_iter()
            .map(path_text)
            .collect();
        runtime.collect(&search, tools).map(|copied| {
            copied
                .iter()
                .map(|path| format!("bundled Windows runtime dependency: {}\n", file_name(path)))
                .collect()
        })
    } else {
        runtime.verify().map(|()| String::new())
    };
    match outcome {
        Ok(stdout) => CheckReport::success(stdout),
        Err(line) => CheckReport::failure(String::new(), format!("{line}\n")),
    }
}
