use crate::TopologyPlanRequest;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanError {
    EmptyLayers,
    EmptyNodes,
    NonContiguousLayers {
        expected: u32,
        found: u32,
    },
    InvalidSplitBoundary {
        boundary: u32,
        layer_start: u32,
        layer_end: u32,
    },
    NonAscendingSplitBoundary {
        previous: u32,
        boundary: u32,
    },
    NotEnoughNodesForSplits {
        stages: usize,
        nodes: usize,
    },
    FamilyLayerCountMismatch {
        family_id: String,
        expected: u32,
        found: u32,
    },
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyLayers => write!(f, "topology plan requires at least one layer"),
            Self::EmptyNodes => write!(f, "topology plan requires at least one node"),
            Self::NonContiguousLayers { expected, found } => write!(
                f,
                "layers must be sorted and contiguous: expected layer {expected}, found {found}"
            ),
            Self::InvalidSplitBoundary {
                boundary,
                layer_start,
                layer_end,
            } => write!(
                f,
                "invalid split boundary {boundary}; expected {layer_start} < boundary < {layer_end}"
            ),
            Self::NonAscendingSplitBoundary { previous, boundary } => write!(
                f,
                "split boundaries must be strictly ascending: previous {previous}, found {boundary}"
            ),
            Self::NotEnoughNodesForSplits { stages, nodes } => write!(
                f,
                "split plan requires {stages} nodes but only {nodes} were provided"
            ),
            Self::FamilyLayerCountMismatch {
                family_id,
                expected,
                found,
            } => write!(
                f,
                "family capability {family_id} expects {expected} layers, found {found}"
            ),
        }
    }
}

impl std::error::Error for PlanError {}

pub(crate) fn validate_request(request: &TopologyPlanRequest) -> Result<(), PlanError> {
    if request.layers.is_empty() {
        return Err(PlanError::EmptyLayers);
    }
    if request.nodes.is_empty() {
        return Err(PlanError::EmptyNodes);
    }
    if let Some(family) = &request.family {
        let found = request.layers.len() as u32;
        if family.layer_count != found {
            return Err(PlanError::FamilyLayerCountMismatch {
                family_id: family.family_id.clone(),
                expected: family.layer_count,
                found,
            });
        }
    }

    for (expected, layer) in (request.layers[0].index..).zip(request.layers.iter()) {
        if layer.index != expected {
            return Err(PlanError::NonContiguousLayers {
                expected,
                found: layer.index,
            });
        }
    }

    Ok(())
}
