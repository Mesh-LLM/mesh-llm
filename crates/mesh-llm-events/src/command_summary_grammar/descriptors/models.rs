use super::{Descriptor, JSON, NONE, RawKind, YES_JSON, descriptor, descriptor_with_conflicts};

const MODEL_PACKAGE_FLAGS: &[&str] = &[
    "--experimental",
    "--dry-run",
    "--confirm",
    "--follow",
    "--list",
    "--update-script",
    "--json",
];
const SEARCH_FLAGS: &[&str] = &["--gguf", "--mlx", "--catalog", "--json"];
const MODEL_CERTIFY_FLAGS: &[&str] = &["--json", "--package-only"];
const MODEL_SEARCH_CONFLICTS: &[&[&str]] = &[&["--gguf", "--mlx"]];

pub(super) const DESCRIPTORS: &[Descriptor] = &[
    descriptor(
        &["mesh-llm", "models", "package"],
        MODEL_PACKAGE_FLAGS,
        &[
            "source_repo",
            "--quant",
            "--target",
            "--model-id",
            "--flavor",
            "--timeout",
            "--mesh-llm-ref",
            "--status",
            "--logs",
            "--cancel",
        ],
        false,
        RawKind::None,
    ),
    descriptor(
        &["mesh-llm", "models", "recommended"],
        JSON,
        NONE,
        false,
        RawKind::None,
    ),
    descriptor(
        &["mesh-llm", "models", "installed"],
        JSON,
        NONE,
        false,
        RawKind::None,
    ),
    descriptor(
        &["mesh-llm", "models", "cleanup"],
        YES_JSON,
        &["--unused-since"],
        false,
        RawKind::None,
    ),
    descriptor(
        &["mesh-llm", "models", "prune"],
        YES_JSON,
        NONE,
        false,
        RawKind::None,
    ),
    descriptor(
        &["mesh-llm", "models", "certify"],
        MODEL_CERTIFY_FLAGS,
        &[
            "model",
            "--report-out",
            "--api-base",
            "--prompt",
            "--max-tokens",
        ],
        false,
        RawKind::None,
    ),
    descriptor(
        &["mesh-llm", "models", "list"],
        JSON,
        NONE,
        false,
        RawKind::None,
    ),
    descriptor_with_conflicts(
        &["mesh-llm", "models", "search"],
        SEARCH_FLAGS,
        &["query", "--limit", "--sort"],
        MODEL_SEARCH_CONFLICTS,
        false,
        RawKind::None,
    ),
    descriptor(
        &["mesh-llm", "models", "show"],
        JSON,
        &["model"],
        false,
        RawKind::None,
    ),
    descriptor(
        &["mesh-llm", "models", "download"],
        &["--draft", "--direct", "--json"],
        &["model"],
        false,
        RawKind::None,
    ),
    descriptor(
        &["mesh-llm", "models", "updates"],
        &["--all", "--check", "--json"],
        &["repo"],
        false,
        RawKind::None,
    ),
    descriptor(
        &["mesh-llm", "models", "delete"],
        YES_JSON,
        &["model"],
        false,
        RawKind::None,
    ),
];
