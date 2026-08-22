use super::{Descriptor, JSON, NONE, RawKind, descriptor, descriptor_with_conflicts};

const RUNTIME_LIST_FLAGS: &[&str] = &["--available", "--installed", "--json"];
const RUNTIME_PRUNE_FLAGS: &[&str] = &["--active-only", "--json"];
const RUNTIME_LIST_CONFLICTS: &[&[&str]] = &[&["--available", "--installed"]];
const REMOTE_MODEL_CONFLICTS: &[&[&str]] = &[&["--model", "--instance-id"]];

pub(super) const DESCRIPTORS: &[Descriptor] = &[
    descriptor(&["mesh-llm", "runtime"], NONE, NONE, false, RawKind::None),
    descriptor(
        &["mesh-llm", "runtime", "status"],
        NONE,
        NONE,
        true,
        RawKind::None,
    ),
    descriptor(
        &["mesh-llm", "runtime", "load"],
        NONE,
        &["name"],
        true,
        RawKind::None,
    ),
    descriptor(
        &["mesh-llm", "runtime", "unload"],
        NONE,
        &["name"],
        true,
        RawKind::None,
    ),
    descriptor(
        &["mesh-llm", "runtime", "guardrails"],
        JSON,
        NONE,
        true,
        RawKind::Mode,
    ),
    descriptor(
        &["mesh-llm", "runtime", "bootstrap"],
        JSON,
        NONE,
        true,
        RawKind::None,
    ),
    descriptor_with_conflicts(
        &["mesh-llm", "runtime", "list"],
        RUNTIME_LIST_FLAGS,
        &["--manifest", "--bundle-dir", "--cache-dir"],
        RUNTIME_LIST_CONFLICTS,
        false,
        RawKind::None,
    ),
    descriptor(
        &["mesh-llm", "runtime", "install"],
        JSON,
        &["runtime_ref", "--manifest", "--bundle-dir", "--cache-dir"],
        false,
        RawKind::None,
    ),
    descriptor(
        &["mesh-llm", "runtime", "remove"],
        JSON,
        &["native_runtime_id", "--mesh-version", "--cache-dir"],
        false,
        RawKind::None,
    ),
    descriptor(
        &["mesh-llm", "runtime", "prune"],
        RUNTIME_PRUNE_FLAGS,
        &["--mesh-version", "--cache-dir"],
        false,
        RawKind::None,
    ),
    descriptor(
        &["mesh-llm", "runtime", "remote"],
        JSON,
        &["--endpoint"],
        true,
        RawKind::None,
    ),
    descriptor_with_conflicts(
        &["mesh-llm", "runtime", "remote-model"],
        JSON,
        &["--endpoint", "--model", "--profile", "--instance-id"],
        REMOTE_MODEL_CONFLICTS,
        true,
        RawKind::None,
    ),
    descriptor(
        &["mesh-llm", "runtime", "apply-config"],
        JSON,
        &["--endpoint", "--expected-revision", "--config"],
        true,
        RawKind::None,
    ),
];
