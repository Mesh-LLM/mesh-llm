//! Native usage and help text for `ci-ops runner-identity`.

pub(crate) const PROG: &str = "cargo xtool ci-ops runner-identity";

pub(crate) const TOP_USAGE: &str = "usage: cargo xtool ci-ops runner-identity [--root <path>] [--catalog <path>] \
     <validate|check|diagnose|lookup|seed-key|bind> [options]\n";

const DESCRIPTION: &str = "Inspect checked-in runner identities using local repository files.

Examples:
  cargo xtool ci-ops runner-identity check
  cargo xtool ci-ops runner-identity lookup ui-quality --field reference
  cargo xtool ci-ops runner-identity seed-key --recipe-hash <hash>
  cargo xtool ci-ops runner-identity diagnose

Historical images have no verified tool observations or provenance. A null
receipt means unknown, not verified or compatible. Compiler workload coverage
is independent of image compatibility; this catalog never authorizes a restore.

Workflow inspection covers .yml and .yaml files and supports the existing
block-style job, container and CUDA matrix declarations. Unsupported runner
references fail closed. This is not a general YAML parser or workflow generator.
Planner rows are checked through the repository's typed native planner. No Cargo
metadata, builds, network access or cache operations are performed. The root
must be a trusted executable checkout, not an untrusted PR manifest directory.
This tool is not wired into the protected planner workflow.
";

pub(crate) fn top_help() -> String {
    format!(
        "{TOP_USAGE}\n{DESCRIPTION}\nCommands:\n  validate\n  check\n  diagnose\n  lookup\n  seed-key\n  bind\n\nOptions:\n  -h, --help\n  --root <path>\n  --catalog <path>\n"
    )
}

pub(crate) fn sub_usage(command: &str) -> String {
    match command {
        "lookup" => format!(
            "usage: {PROG} lookup <role> [--field <reference|native_toolchain_epoch|receipt|provenance>]\n"
        ),
        "seed-key" => format!("usage: {PROG} seed-key --recipe-hash <sha256>\n"),
        "bind" => format!(
            "usage: {PROG} bind --image-id <id> --cohort <path> --anchor <path> --output <path>\n"
        ),
        other => format!("usage: {PROG} {other}\n"),
    }
}

pub(crate) fn sub_help(command: &str) -> String {
    let body = match command {
        "lookup" => {
            "\nOptions:\n  -h, --help\n  --field <reference|native_toolchain_epoch|receipt|provenance>\n"
        }
        "seed-key" => "\nOptions:\n  -h, --help\n  --recipe-hash <sha256>\n",
        "bind" => {
            "\nOptions:\n  -h, --help\n  --image-id <id>\n  --cohort <path>\n  --anchor <path>\n  --output <path>\n"
        }
        _ => "\nOptions:\n  -h, --help\n",
    };
    format!("{}{body}", sub_usage(command))
}
