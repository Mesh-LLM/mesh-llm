use super::SplitParticipant;
use mesh_llm_transport::protocol::StageLinkObservation;
use skippy_coordinator::topology::StageLink;

impl SplitParticipant {
    pub(in crate::runtime) fn with_stage_link(
        mut self,
        source: iroh::EndpointId,
        observation: Option<StageLinkObservation>,
    ) -> Self {
        self.stage_link = observation.map(|observation| (source, observation));
        self
    }
}

pub(super) fn planning_links(participants: &[SplitParticipant]) -> Vec<StageLink> {
    participants
        .iter()
        .filter_map(|participant| {
            let (source, observation) = participant.stage_link?;
            Some(StageLink {
                source_node_id: source.to_string(),
                destination_node_id: participant.node_id.to_string(),
                acknowledgement_latency: observation.acknowledgement_latency,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn participant_measurements_become_directed_planning_edges_without_rounding() {
        let source = iroh::SecretKey::generate().public();
        let destination = iroh::SecretKey::generate().public();
        let observation = StageLinkObservation {
            acknowledgement_latency: std::time::Duration::from_micros(150),
            payload_bytes_per_second: 10_000_000,
            samples: 3,
            age: std::time::Duration::ZERO,
        };
        let participants = [
            SplitParticipant::new(source, 1, None),
            SplitParticipant::new(destination, 1, None).with_stage_link(source, Some(observation)),
        ];
        assert_eq!(
            planning_links(&participants),
            vec![StageLink {
                source_node_id: source.to_string(),
                destination_node_id: destination.to_string(),
                acknowledgement_latency: observation.acknowledgement_latency,
            }]
        );
    }
}
