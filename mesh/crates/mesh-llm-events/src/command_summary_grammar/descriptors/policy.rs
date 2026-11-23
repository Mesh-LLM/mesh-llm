use super::{Descriptor, JSON, NONE, RawKind, descriptor};

pub(super) const DESCRIPTORS: &[Descriptor] = &[
    descriptor(&["mesh-llm", "policy"], NONE, NONE, false, RawKind::None),
    descriptor(
        &["mesh-llm", "policy", "set"],
        JSON,
        &["payload"],
        true,
        RawKind::None,
    ),
    descriptor(
        &["mesh-llm", "policy", "get"],
        JSON,
        NONE,
        true,
        RawKind::None,
    ),
    descriptor(
        &["mesh-llm", "policy", "clear"],
        JSON,
        NONE,
        true,
        RawKind::None,
    ),
];
