//! `product runtime-release-manifest` parity with the inline Python in
//! `scripts/generate-native-runtime-release-manifest.sh`.

use crate::packaging::{Snippet, run_cases};
use crate::packaging_cases::{Case, LINUX, MAC, manifest, runtime, tar_gz, write};
use crate::support::TestResult;
use std::path::Path;

const SNIPPET: Snippet = Snippet {
    script: "scripts/generate-native-runtime-release-manifest.sh",
    heredoc: 0,
    subcommand: "runtime-release-manifest",
    extractor_at: Some(4),
};

const TWO: &[&str] = &[
    "{scratch}/out/release.json",
    "Mesh-LLM/mesh-llm",
    "v1.2.3",
    "{scratch}/tmp",
    "dist/rt-linux.tar.gz",
    "dist/rt-a-mac.tar.gz",
];
const ONE: &[&str] = &[
    "{scratch}/out/release.json",
    "Mesh-LLM/mesh-llm",
    "v1.2.3",
    "{scratch}/tmp",
    "dist/rt-linux.tar.gz",
];

fn linux(root: &Path) -> TestResult {
    runtime(
        root,
        "rt-linux",
        &manifest("rt-linux", "v1.2.3", "7", LINUX),
        "linux-bytes\n",
    )
}

fn two(root: &Path) -> TestResult {
    linux(root)?;
    runtime(
        root,
        "rt-a-mac",
        &manifest("rt-a-mac", "v1.2.3", "7", MAC),
        "mac-bytes\n",
    )
}

fn stale(root: &Path) -> TestResult {
    runtime(
        root,
        "rt-linux",
        &manifest("rt-linux", "v1.2.3", "7", LINUX),
        "rebuilt-bytes\n",
    )
}

fn wrong_version(root: &Path) -> TestResult {
    runtime(
        root,
        "rt-linux",
        &manifest("rt-linux", "v1.2.4", "7", LINUX),
        "x\n",
    )
}

fn mixed_abi(root: &Path) -> TestResult {
    linux(root)?;
    runtime(
        root,
        "rt-a-mac",
        &manifest("rt-a-mac", "v1.2.3", "8", MAC),
        "x\n",
    )
}

fn mixed_version(root: &Path) -> TestResult {
    linux(root)?;
    runtime(
        root,
        "rt-a-mac",
        &manifest("rt-a-mac", "1.2.3", "7", MAC),
        "x\n",
    )
}

fn missing_platform(root: &Path) -> TestResult {
    let text = "{\"runtime\": {\"id\": \"rt-linux\", \"mesh_version\": \"1.2.3\", \
                \"skippy_abi\": 7, \"backend\": {}, \"files\": []}}";
    runtime(root, "rt-linux", text, "x\n")
}

fn no_runtime(root: &Path) -> TestResult {
    runtime(root, "rt-linux", "{\"runtime\": [1]}", "x\n")
}

fn two_manifests(root: &Path) -> TestResult {
    let text = manifest("rt-linux", "1.2.3", "7", LINUX);
    tar_gz(
        &root.join("dist/rt-linux.tar.gz"),
        &[
            ("a/manifest.json", text.as_str()),
            ("b/manifest.json", text.as_str()),
        ],
    )
}

fn unsafe_member(root: &Path) -> TestResult {
    tar_gz(
        &root.join("dist/rt-linux.tar.gz"),
        &[("../escape.txt", "x\n")],
    )
}

fn nothing(root: &Path) -> TestResult {
    write(root, "dist/.keep", "")
}

const HAPPY: &[Case] = &[
    Case {
        name: "two_runtimes_sorted_by_id",
        setup: two,
        args: TWO,
    },
    Case {
        name: "single_runtime",
        setup: linux,
        args: ONE,
    },
    Case {
        name: "rebuilt_archive_changes_digest",
        setup: stale,
        args: ONE,
    },
];

const REJECT: &[Case] = &[
    Case {
        name: "tag_version_mismatch",
        setup: wrong_version,
        args: ONE,
    },
    Case {
        name: "mixed_skippy_abi",
        setup: mixed_abi,
        args: TWO,
    },
    Case {
        name: "mixed_mesh_version_prefix",
        setup: mixed_version,
        args: TWO,
    },
    Case {
        name: "missing_platform_field",
        setup: missing_platform,
        args: ONE,
    },
    Case {
        name: "runtime_not_object",
        setup: no_runtime,
        args: ONE,
    },
    Case {
        name: "two_manifests",
        setup: two_manifests,
        args: ONE,
    },
    Case {
        name: "unsafe_archive_member",
        setup: unsafe_member,
        args: ONE,
    },
    Case {
        name: "missing_archive_never_rebuilt",
        setup: nothing,
        args: ONE,
    },
    Case {
        name: "empty_tag_version",
        setup: nothing,
        args: &[
            "{scratch}/out/release.json",
            "Mesh-LLM/mesh-llm",
            "v",
            "{scratch}/tmp",
        ],
    },
    Case {
        name: "no_archives",
        setup: nothing,
        args: &[
            "{scratch}/out/release.json",
            "Mesh-LLM/mesh-llm",
            "v1.2.3",
            "{scratch}/tmp",
        ],
    },
];

#[test]
fn migration_product_release_manifest_writes_sorted_digests() -> TestResult {
    run_cases(&SNIPPET, "release_manifest_goldens_happy.json", HAPPY)
}

#[test]
fn migration_product_release_manifest_rejects_mismatched_archives() -> TestResult {
    run_cases(&SNIPPET, "release_manifest_goldens_reject.json", REJECT)
}
