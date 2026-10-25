//! Help text for `ci-ops build-cache`.

pub(crate) const PROG: &str = "cargo xtool ci-ops build-cache";

pub(crate) const TOP_USAGE: &str =
    "usage: cargo xtool ci-ops build-cache <status|prune|build> [options]\n";

pub(crate) const STATUS_USAGE: &str = "usage: cargo xtool ci-ops build-cache status [--workspace <path>] \
     [--target-dir <path>] [--max-size <size>] [--max-age <days>] [--json]\n";

pub(crate) const PRUNE_USAGE: &str = "usage: cargo xtool ci-ops build-cache prune [--workspace <path>] \
     [--target-dir <path>] [--max-size <size>] [--max-age <days>] [--json] [--execute]\n";

pub(crate) const BUILD_USAGE: &str = "usage: cargo xtool ci-ops build-cache build [--workspace <path>] \
     [--target-dir <path>] -- <build command>...\n";

fn options_header() -> &'static str {
    "\nOptions:\n  -h, --help\n"
}

pub(crate) fn top_help() -> String {
    format!(
        "{TOP_USAGE}\nCommands:\n  status\n  prune\n  build\n{}",
        options_header()
    )
}

pub(crate) fn status_help(execute: bool) -> String {
    let usage = if execute { PRUNE_USAGE } else { STATUS_USAGE };
    let mut text = format!(
        "{usage}{}  --workspace <path>\n  --target-dir <path>\n  \
         --max-size <size>\n  --max-age <days>\n  --json\n",
        options_header()
    );
    if execute {
        text.push_str("  --execute\n");
    }
    text
}

pub(crate) fn build_help() -> String {
    format!(
        "{BUILD_USAGE}{}  --workspace <path>\n  --target-dir <path>\n",
        options_header()
    )
}
