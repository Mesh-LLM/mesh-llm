//! argparse usage and help text of `scripts/runner-image-identity.py` at the
//! default 80-column width, keyed by the legacy program name.

pub(crate) const PROG: &str = "runner-image-identity.py";

pub(crate) const TOP_USAGE: &str =
    "usage: runner-image-identity.py [-h] [--root ROOT] [--catalog CATALOG]
                                {validate,check,diagnose,lookup,seed-key,bind} ...
";

const DESCRIPTION: &str = "Inspect checked-in runner identities without changing CI execution.

Examples:
  python3 scripts/runner-image-identity.py check
  python3 scripts/runner-image-identity.py lookup ui-quality --field reference
  python3 scripts/runner-image-identity.py seed-key --recipe-hash <hash>
  python3 scripts/runner-image-identity.py diagnose

Historical images have no verified tool observations or provenance. A null
receipt means unknown, not verified or compatible. Compiler workload coverage
is independent of image compatibility; this catalog never authorizes a restore.

Workflow inspection covers .yml and .yaml files and deliberately supports only
the existing block-style job, container and CUDA matrix declarations. Runner
references in unsupported declarations fail closed. This is not a general YAML
parser or a workflow generator. Planner
rows are checked through the repository's real, stdlib-only selection function;
no Cargo metadata, builds, network access, or cache operations are performed.
Consequently --root must be a trusted executable checkout, not an untrusted PR
manifest directory. This tool is not wired into the protected planner workflow.
";

pub(crate) fn top_help() -> String {
    format!(
        "{TOP_USAGE}\n{DESCRIPTION}\npositional arguments:\n  {{validate,check,diagnose,lookup,seed-key,bind}}\n\noptions:\n  -h, --help            show this help message and exit\n  --root ROOT\n  --catalog CATALOG\n"
    )
}

pub(crate) fn sub_usage(command: &str) -> String {
    match command {
        "lookup" => "usage: runner-image-identity.py lookup [-h]
                                       [--field {reference,native_toolchain_epoch,receipt,provenance}]
                                       role
"
        .to_owned(),
        "seed-key" => "usage: runner-image-identity.py seed-key [-h] --recipe-hash RECIPE_HASH\n".to_owned(),
        "bind" => "usage: runner-image-identity.py bind [-h] --image-id IMAGE_ID --cohort COHORT
                                     --anchor ANCHOR --output OUTPUT
"
        .to_owned(),
        other => format!("usage: runner-image-identity.py {other} [-h]\n"),
    }
}

pub(crate) fn sub_help(command: &str) -> String {
    let body = match command {
        "lookup" => {
            "positional arguments:
  role

options:
  -h, --help            show this help message and exit
  --field {reference,native_toolchain_epoch,receipt,provenance}
"
        }
        "seed-key" => {
            "options:
  -h, --help            show this help message and exit
  --recipe-hash RECIPE_HASH
"
        }
        "bind" => {
            "options:
  -h, --help           show this help message and exit
  --image-id IMAGE_ID
  --cohort COHORT
  --anchor ANCHOR
  --output OUTPUT
"
        }
        _ => {
            "options:
  -h, --help  show this help message and exit
"
        }
    };
    format!("{}\n{body}", sub_usage(command))
}
