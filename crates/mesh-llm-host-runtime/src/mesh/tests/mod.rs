use crate::mesh::artifact_transfer_io::{PartialArtifactGuard, write_artifact_transfer_chunk};
use crate::mesh::heartbeat::HeartbeatFailurePolicy;
use crate::mesh::node::LocalRequestMetricsSampler;
use mesh_llm_types::mesh::{DEMAND_TTL_SECS, merge_demand};

mod control_listener;

include!("connections.rs");
include!("admission/helpers.rs");
include!("protocol_compat.rs");
include!("owner_control.rs");
include!("stage_transport.rs");
include!("peer_state.rs");
include!("protocol_frames.rs");
include!("admission.rs");
include!("requirements.rs");

mod split_coverage {
    use super::super::*;

    mod cases {
        use super::super::{
            make_test_endpoint_id, open_owner_control_stream, read_owner_control_envelope,
            start_owner_control_test_server, test_owner_keypair,
        };
        use super::*;
        use crate::mesh::node_requirements::preflight_pushed_config_for_current_node_with_gpus;
        use prost::Message;

        include!("announcement_unique.rs");
        include!("control_plane_unique.rs");
        include!("stage_control.rs");
    }
}
