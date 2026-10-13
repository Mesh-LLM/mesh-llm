//! argparse usage and help text for `scripts/manage-build-cache.py`, as
//! Python 3.13 renders it at the 80-column fallback width used when stdout
//! is not a terminal.

pub(crate) const PROG: &str = "manage-build-cache.py";

pub(crate) const TOP_USAGE: &str = "usage: manage-build-cache.py [-h] {status,prune,build} ...\n";

pub(crate) const STATUS_USAGE: &str = "\
usage: manage-build-cache.py status [-h] [--workspace WORKSPACE]
                                    [--target-dir TARGET_DIR]
                                    [--max-size MAX_SIZE] [--max-age MAX_AGE]
                                    [--json]
";

pub(crate) const PRUNE_USAGE: &str = "\
usage: manage-build-cache.py prune [-h] [--workspace WORKSPACE]
                                   [--target-dir TARGET_DIR]
                                   [--max-size MAX_SIZE] [--max-age MAX_AGE]
                                   [--json] [--execute]
";

pub(crate) const BUILD_USAGE: &str = "\
usage: manage-build-cache.py build [-h] [--workspace WORKSPACE]
                                   [--target-dir TARGET_DIR]
                                   ...
";

const HELP_LINE: &str = "  -h, --help            show this help message and exit\n";

pub(crate) fn top_help() -> String {
    format!("{TOP_USAGE}\npositional arguments:\n  {{status,prune,build}}\n\noptions:\n{HELP_LINE}")
}

pub(crate) fn status_help(execute: bool) -> String {
    let usage = if execute { PRUNE_USAGE } else { STATUS_USAGE };
    let mut text = format!(
        "{usage}\noptions:\n{HELP_LINE}  --workspace WORKSPACE\n  --target-dir TARGET_DIR\n  \
         --max-size MAX_SIZE\n  --max-age MAX_AGE\n  --json\n"
    );
    if execute {
        text.push_str("  --execute\n");
    }
    text
}

pub(crate) fn build_help() -> String {
    format!(
        "{BUILD_USAGE}\npositional arguments:\n  build_command\n\noptions:\n{HELP_LINE}  \
         --workspace WORKSPACE\n  --target-dir TARGET_DIR\n"
    )
}
