//! `native linux-runtime-deps`: the Rust owner of
//! `scripts/linux-native-runtime-deps.py`. Keeps its argparse surface (a
//! required `{collect,verify,order}` subcommand), streams and statuses:
//! handled failures print one line to stderr and exit 1. `readelf` runs only
//! through the [`Toolchain`] adapter.

use super::argv::{Grammar, Opt, Parsed};
use super::linux_deps_elf::file_name;
use super::linux_deps_order::dependency_order;
use super::linux_deps_policy::Package;
use super::toolchain::Toolchain;
use crate::ci_operations::build_cache_options::{Kind, classify, help_flag};
use crate::ci_operations::runner_identity_argv::error;
use crate::ci_plan::catalog::python_path_display;
use crate::repository::check_report::CheckReport;
use crate::repository::python_text::repr;
use std::path::Path;

const PROG: &str = "linux-native-runtime-deps.py";
const USAGE: &str = "usage: linux-native-runtime-deps.py [-h] {collect,verify,order} ...\n";
const HELP: &str = "\
usage: linux-native-runtime-deps.py [-h] {collect,verify,order} ...

Collect and verify redistributable Linux ELF runtime dependencies.

positional arguments:
  {collect,verify,order}

options:
  -h, --help            show this help message and exit
";
const ARCHES: &[&str] = &["x86_64", "aarch64", "arm"];

const COLLECT_USAGE: &str = "\
usage: linux-native-runtime-deps.py collect [-h] --lib-dir LIB_DIR
                                            [--scan-dir SCAN_DIR]
                                            [--arch {x86_64,aarch64,arm}]
                                            [--search-dir SEARCH_DIR]
                                            --cuda-major {12,13}
";
const VERIFY_USAGE: &str = "\
usage: linux-native-runtime-deps.py verify [-h] --lib-dir LIB_DIR
                                           [--scan-dir SCAN_DIR]
                                           [--arch {x86_64,aarch64,arm}]
";
const ORDER_USAGE: &str = "\
usage: linux-native-runtime-deps.py order [-h] --lib-dir LIB_DIR
                                          [--scan-dir SCAN_DIR]
                                          [--arch {x86_64,aarch64,arm}]
                                          --primary PRIMARY
";
const COLLECT: Grammar = Grammar {
    prog: "linux-native-runtime-deps.py collect",
    usage: COLLECT_USAGE,
    help: concat!(
        "usage: linux-native-runtime-deps.py collect [-h] --lib-dir LIB_DIR\n",
        "                                            [--scan-dir SCAN_DIR]\n",
        "                                            [--arch {x86_64,aarch64,arm}]\n",
        "                                            [--search-dir SEARCH_DIR]\n",
        "                                            --cuda-major {12,13}\n",
        "\noptions:\n",
        "  -h, --help            show this help message and exit\n",
        "  --lib-dir LIB_DIR\n  --scan-dir SCAN_DIR\n  --arch {x86_64,aarch64,arm}\n",
        "  --search-dir SEARCH_DIR\n  --cuda-major {12,13}\n",
    ),
    options: &[
        Opt::required("--lib-dir"),
        Opt::value("--scan-dir"),
        Opt::choice("--arch", ARCHES),
        Opt::value("--search-dir"),
        Opt::int_choice("--cuda-major", &["12", "13"]),
    ],
    positional: None,
};

const VERIFY: Grammar = Grammar {
    prog: "linux-native-runtime-deps.py verify",
    usage: VERIFY_USAGE,
    help: concat!(
        "usage: linux-native-runtime-deps.py verify [-h] --lib-dir LIB_DIR\n",
        "                                           [--scan-dir SCAN_DIR]\n",
        "                                           [--arch {x86_64,aarch64,arm}]\n",
        "\noptions:\n",
        "  -h, --help            show this help message and exit\n",
        "  --lib-dir LIB_DIR\n  --scan-dir SCAN_DIR\n  --arch {x86_64,aarch64,arm}\n",
    ),
    options: &[
        Opt::required("--lib-dir"),
        Opt::value("--scan-dir"),
        Opt::choice("--arch", ARCHES),
    ],
    positional: None,
};

const ORDER: Grammar = Grammar {
    prog: "linux-native-runtime-deps.py order",
    usage: ORDER_USAGE,
    help: concat!(
        "usage: linux-native-runtime-deps.py order [-h] --lib-dir LIB_DIR\n",
        "                                          [--scan-dir SCAN_DIR]\n",
        "                                          [--arch {x86_64,aarch64,arm}]\n",
        "                                          --primary PRIMARY\n",
        "\noptions:\n",
        "  -h, --help            show this help message and exit\n",
        "  --lib-dir LIB_DIR\n  --scan-dir SCAN_DIR\n  --arch {x86_64,aarch64,arm}\n",
        "  --primary PRIMARY\n",
    ),
    options: &[
        Opt::required("--lib-dir"),
        Opt::value("--scan-dir"),
        Opt::choice("--arch", ARCHES),
        Opt::required("--primary"),
    ],
    positional: None,
};

/// The selected subcommand.
#[derive(Clone, Copy)]
enum Action {
    Collect,
    Verify,
    Order,
}

fn top_fail(message: &str) -> CheckReport {
    error(USAGE, PROG, message)
}

const fn grammar(action: Action) -> &'static Grammar {
    match action {
        Action::Collect => &COLLECT,
        Action::Verify => &VERIFY,
        Action::Order => &ORDER,
    }
}

/// The top-level parser: options before the subcommand are only `-h`
/// (others are extras), the first positional word (after one optional
/// `--`) selects the subcommand, and every later word belongs to it.
fn parse(args: &[String]) -> Result<(Action, Parsed), CheckReport> {
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
) -> Result<(Action, Parsed), CheckReport> {
    let action = match command {
        "collect" => Action::Collect,
        "verify" => Action::Verify,
        "order" => Action::Order,
        _ => {
            return Err(top_fail(&format!(
                "argument command: invalid choice: {} (choose from 'collect', 'verify', 'order')",
                repr(command)
            )));
        }
    };
    let (parsed, unknown) = grammar(action).parse_known(rest)?;
    extras.extend(unknown);
    if !extras.is_empty() {
        return Err(top_fail(&format!(
            "unrecognized arguments: {}",
            extras.join(" ")
        )));
    }
    Ok((action, parsed))
}

fn path_text(value: &str) -> String {
    python_path_display(Path::new(value))
}

pub(super) fn run(args: &[String], tools: &dyn Toolchain) -> CheckReport {
    let (action, parsed) = match parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report,
    };
    let lib_dir = path_text(parsed.value("--lib-dir").unwrap_or_default());
    let mut scan_dirs = vec![lib_dir.clone()];
    scan_dirs.extend(parsed.values("--scan-dir").into_iter().map(path_text));
    let package = Package {
        tools,
        lib_dir,
        scan_dirs,
        arch: parsed.value("--arch").map(str::to_owned),
    };
    let outcome = match action {
        Action::Collect => {
            let search: Vec<String> = parsed
                .values("--search-dir")
                .into_iter()
                .map(path_text)
                .collect();
            let major = parsed.value("--cuda-major").unwrap_or_default();
            package.collect(&search, major).map(|copied| {
                copied
                    .iter()
                    .map(|path| format!("bundled Linux runtime dependency: {}\n", file_name(path)))
                    .collect()
            })
        }
        Action::Verify => package.verify().map(|()| String::new()),
        Action::Order => {
            let primary = parsed.value("--primary").unwrap_or_default();
            dependency_order(&package, primary).map(|ordered| {
                ordered
                    .iter()
                    .map(|path| format!("{}\n", file_name(path)))
                    .collect()
            })
        }
    };
    match outcome {
        Ok(stdout) => CheckReport::success(stdout),
        Err(line) => CheckReport::failure(String::new(), format!("{line}\n")),
    }
}
