//! `product canonical-inputs` and `product runtime-version` parity with the
//! two inline snippets in `scripts/ci-compose-product-input.sh`.

use crate::packaging::{Snippet, run_cases};
use crate::packaging_cases::{Case, write};
use crate::support::TestResult;
use std::os::unix::fs::symlink;
use std::path::Path;

const CANONICAL: Snippet = Snippet {
    script: "scripts/ci-compose-product-input.sh",
    heredoc: 0,
    subcommand: "canonical-inputs",
    extractor_at: None,
};

const VERSION: Snippet = Snippet {
    script: "scripts/ci-compose-product-input.sh",
    heredoc: 1,
    subcommand: "runtime-version",
    extractor_at: None,
};

fn inputs(root: &Path) -> TestResult {
    write(root, "ws/host/mesh-llm", "host\n")?;
    write(root, "ws/runtime/rt/manifest.json", "{}")
}

fn linked(root: &Path) -> TestResult {
    inputs(root)?;
    symlink("host", root.join("ws/host-link"))?;
    symlink("../outside", root.join("ws/escape-link"))?;
    Ok(())
}

fn host_file(root: &Path) -> TestResult {
    write(root, "ws/host", "not a directory\n")?;
    write(root, "ws/runtime/rt/manifest.json", "{}")
}

const fn args4(
    host: &'static str,
    runtime: &'static str,
    output: &'static str,
) -> [&'static str; 4] {
    ["{scratch}/ws", host, runtime, output]
}

const CANONICAL_CASES: &[Case] = &[
    Case {
        name: "relative_inputs",
        setup: inputs,
        args: &args4("host", "runtime", "out/product"),
    },
    Case {
        name: "absolute_and_dotted_inputs",
        setup: inputs,
        args: &args4("{scratch}/ws/host", "./runtime/../runtime", "out//product/"),
    },
    Case {
        name: "symlinked_host_resolves",
        setup: linked,
        args: &args4("host-link", "runtime", "out"),
    },
    Case {
        name: "symlink_escape",
        setup: linked,
        args: &args4("escape-link", "runtime", "out"),
    },
    Case {
        name: "dotdot_escape",
        setup: inputs,
        args: &args4("../ws-other", "runtime", "out"),
    },
    Case {
        name: "missing_host_input",
        setup: inputs,
        args: &args4("absent", "runtime", "out"),
    },
    Case {
        name: "host_input_is_file",
        setup: host_file,
        args: &args4("host", "runtime", "out"),
    },
    Case {
        name: "output_is_workspace",
        setup: inputs,
        args: &args4("host", "runtime", "."),
    },
    Case {
        name: "output_inside_host",
        setup: inputs,
        args: &args4("host", "runtime", "host/out"),
    },
    Case {
        name: "output_contains_runtime",
        setup: inputs,
        args: &args4("host", "runtime/rt", "runtime"),
    },
    Case {
        name: "output_sibling_prefix",
        setup: inputs,
        args: &args4("host", "runtime", "hostile"),
    },
    Case {
        name: "missing_workspace",
        setup: inputs,
        args: &["{scratch}/nope", "host", "runtime", "out"],
    },
    Case {
        name: "too_few_arguments",
        setup: inputs,
        args: &["{scratch}/ws", "host"],
    },
];

fn manifest_file(root: &Path) -> TestResult {
    write(
        root,
        "manifest.json",
        "{\"runtime\": {\"mesh_version\": \"v0.70.1\"}}",
    )
}

fn numeric_version(root: &Path) -> TestResult {
    write(
        root,
        "manifest.json",
        "{\"runtime\": {\"mesh_version\": 7}}",
    )
}

fn missing_key(root: &Path) -> TestResult {
    write(root, "manifest.json", "{\"runtime\": {\"id\": \"rt\"}}")
}

fn not_json(root: &Path) -> TestResult {
    write(root, "manifest.json", "{\"runtime\":")
}

const VERSION_CASES: &[Case] = &[
    Case {
        name: "prints_mesh_version",
        setup: manifest_file,
        args: &["manifest.json"],
    },
    Case {
        name: "numeric_version",
        setup: numeric_version,
        args: &["manifest.json"],
    },
    Case {
        name: "missing_mesh_version",
        setup: missing_key,
        args: &["manifest.json"],
    },
    Case {
        name: "malformed_json",
        setup: not_json,
        args: &["manifest.json"],
    },
    Case {
        name: "missing_manifest",
        setup: manifest_file,
        args: &["absent.json"],
    },
];

#[test]
fn migration_product_canonical_inputs_resolves_and_rejects() -> TestResult {
    run_cases(&CANONICAL, "canonical_inputs_goldens.json", CANONICAL_CASES)
}

#[test]
fn migration_product_runtime_version_reads_manifest() -> TestResult {
    run_cases(&VERSION, "runtime_version_goldens.json", VERSION_CASES)
}
