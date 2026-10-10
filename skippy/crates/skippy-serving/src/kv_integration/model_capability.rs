#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum ModelKvCapability {
    KnownDense,
    KnownRecurrent,
    Unknown(String),
}

/// The admitted graph is the only authority for the representation shape.
/// An absent or unrecognized summary may use full-state snapshots only.
pub(super) fn graph_model_kv_capability(state: &str) -> ModelKvCapability {
    match state {
        "dense" => ModelKvCapability::KnownDense,
        "recurrent" => ModelKvCapability::KnownRecurrent,
        _ => ModelKvCapability::Unknown("graph state is not partial-snapshot safe".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admitted_graph_state_selects_representation() {
        for (state, expected) in [
            ("dense", ModelKvCapability::KnownDense),
            ("recurrent", ModelKvCapability::KnownRecurrent),
        ] {
            assert_eq!(graph_model_kv_capability(state), expected);
        }
    }

    #[test]
    fn unsupported_or_missing_graph_fails_closed() {
        assert!(matches!(
            graph_model_kv_capability("full-state"),
            ModelKvCapability::Unknown(_)
        ));
        assert!(matches!(
            graph_model_kv_capability(""),
            ModelKvCapability::Unknown(_)
        ));
    }
}
