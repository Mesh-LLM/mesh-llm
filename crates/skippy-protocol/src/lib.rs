/// Compatibility namespace for tokenizer contracts.
///
/// New consumers should depend on `skippy-tokenizer` directly. Keeping this
/// re-export avoids breaking older protocol users while the contract moves
/// out of the wire-protocol crate.
pub mod tokenizer {
    pub use skippy_tokenizer::*;
}

pub mod binary;
pub mod proto {
    pub mod stage {
        include!(concat!(env!("OUT_DIR"), "/skippy.stage.v1.rs"));
    }
}

mod config;
mod messages;
mod validation;

pub use config::{
    ActivationDType, ActivationDescriptor, ActivationLayout, FlashAttentionType, GlmDsaPolicy,
    LoadMode, PeerConfig, SplitMode, StageConfig, StageDevice, StageIdentity, StageKvCacheConfig,
    StageKvCacheMode, StageKvCachePayload, StageTopology, StageTopologyEntry,
};
pub use messages::{
    AckMessage, DecodeTokenMessage, ErrorMessage, FinalPrefillChunkMessage, MessageBase,
    MessageKind, PrefillChunkMessage, ReadyMessage, StageMessage, StateExportMessage,
    StateImportMessage, StopMessage, TokenReplyMessage,
};
pub use validation::{
    MAX_STAGE_FRAME_BYTES, MAX_VERIFY_WINDOW_PIPELINE_DEPTH, SCHEMA_VERSION, STAGE_ALPN_V2,
    STAGE_PROTOCOL_GENERATION, STAGE_STREAM_ARTIFACT_TRANSFER, STAGE_STREAM_CONTROL,
    STAGE_STREAM_TRANSPORT, STAGE_SUBPROTOCOL_FEATURE_ARTIFACT_TRANSFER,
    STAGE_SUBPROTOCOL_FEATURE_STAGE_CONTROL, STAGE_SUBPROTOCOL_FEATURE_STAGE_GENERATION,
    STAGE_SUBPROTOCOL_FEATURE_STAGE_PROTOCOL_GENERATION_V5, STAGE_SUBPROTOCOL_FEATURE_STATUS_LIST,
    STAGE_SUBPROTOCOL_MAJOR, STAGE_SUBPROTOCOL_NAME, StageFrameError,
    validate_stage_artifact_transfer_request, validate_stage_artifact_transfer_response,
    validate_stage_control_request, validate_stage_control_response, validate_stage_transport_open,
};

#[cfg(test)]
mod tests {
    use prost::Message as _;

    use super::proto::stage::{
        CancelPrepareStage, GetLayerInventory, GetStageStatus, LayerInventory, LayerRange,
        LoadStage, PrepareStage, PrepareStageAccepted, SourceModelKind,
        StageArtifactTransferRequest, StageArtifactTransferResponse, StageControlRequest,
        StageControlResponse, StagePreparationState, StagePreparationStatus, StageReady,
        StageRuntimeState, StageStatus, StageStatusAck, StageStatusList, StageStatusUpdate,
        StageTransportOpen, StopStage, stage_control_request, stage_control_response,
    };
    use super::{
        STAGE_PROTOCOL_GENERATION, STAGE_SUBPROTOCOL_FEATURE_STAGE_PROTOCOL_GENERATION_V5,
        StageFrameError, validate_stage_artifact_transfer_request,
        validate_stage_artifact_transfer_response, validate_stage_control_request,
        validate_stage_control_response, validate_stage_transport_open,
    };

    #[test]
    fn stage_protocol_generation_feature_names_current_generation() {
        assert_eq!(
            STAGE_SUBPROTOCOL_FEATURE_STAGE_PROTOCOL_GENERATION_V5,
            format!("stage-generation-{STAGE_PROTOCOL_GENERATION}")
        );
    }

    #[test]
    fn stage_control_request_validates_generation_sender_and_command() {
        let frame = StageControlRequest {
            r#gen: STAGE_PROTOCOL_GENERATION,
            requester_id: vec![9u8; 32],
            command: Some(stage_control_request::Command::GetStageStatus(
                GetStageStatus {
                    topology_id: Some("topology-a".to_string()),
                    run_id: Some("run-a".to_string()),
                    stage_id: Some("stage-0".to_string()),
                },
            )),
        };
        validate_stage_control_request(&frame).unwrap();

        let load = StageControlRequest {
            command: Some(stage_control_request::Command::LoadStage(LoadStage {
                topology_id: "topology-a".to_string(),
                run_id: "run-a".to_string(),
                model_id: "qwen".to_string(),
                backend: "skippy".to_string(),
                package_ref: "hf://repo/model".to_string(),
                manifest_sha256: "a5".repeat(32),
                stage_id: "stage-0".to_string(),
                layer_end: 16,
                activation_width: 4096,
                projector_path: Some("/models/mmproj.gguf".to_string()),
                ..Default::default()
            })),
            ..frame.clone()
        };
        let decoded = StageControlRequest::decode(load.encode_to_vec().as_slice()).unwrap();
        match decoded.command {
            Some(stage_control_request::Command::LoadStage(load)) => {
                assert_eq!(load.projector_path.as_deref(), Some("/models/mmproj.gguf"));
            }
            other => panic!("expected LoadStage, got {other:?}"),
        }

        let stop = StageControlRequest {
            command: Some(stage_control_request::Command::StopStage(StopStage {
                topology_id: "topology-a".to_string(),
                run_id: "run-a".to_string(),
                stage_id: "stage-0".to_string(),
                shutdown_generation: 7,
                coordinator_term: 7,
            })),
            ..frame.clone()
        };
        validate_stage_control_request(&stop).unwrap();

        let inventory = StageControlRequest {
            command: Some(stage_control_request::Command::GetLayerInventory(
                GetLayerInventory {
                    model_id: "qwen".to_string(),
                    package_ref: "hf://repo/model".to_string(),
                    manifest_sha256: "a5".repeat(32),
                },
            )),
            ..frame.clone()
        };
        validate_stage_control_request(&inventory).unwrap();

        let prepare = StageControlRequest {
            command: Some(stage_control_request::Command::PrepareStage(PrepareStage {
                load_stage: Some(LoadStage {
                    topology_id: "topology-a".to_string(),
                    run_id: "run-a".to_string(),
                    model_id: "qwen".to_string(),
                    backend: "skippy".to_string(),
                    package_ref: "gguf:///model.gguf".to_string(),
                    manifest_sha256: "direct-gguf:1:model.gguf".to_string(),
                    stage_id: "stage-1".to_string(),
                    layer_start: 8,
                    layer_end: 16,
                    ..Default::default()
                }),
                coordinator_id: Some(vec![8u8; 32]),
            })),
            ..frame.clone()
        };
        validate_stage_control_request(&prepare).unwrap();

        let status_update = StageControlRequest {
            command: Some(stage_control_request::Command::StageStatusUpdate(
                StageStatusUpdate {
                    status: Some(StagePreparationStatus {
                        topology_id: "topology-a".to_string(),
                        run_id: "run-a".to_string(),
                        model_id: "qwen".to_string(),
                        backend: "skippy".to_string(),
                        package_ref: "gguf:///model.gguf".to_string(),
                        manifest_sha256: "direct-gguf:1:model.gguf".to_string(),
                        stage_id: "stage-1".to_string(),
                        stage_index: 1,
                        layer_start: 8,
                        layer_end: 16,
                        state: StagePreparationState::Loading as i32,
                        bytes_done: Some(10),
                        bytes_total: Some(20),
                        shutdown_generation: 7,
                        ..Default::default()
                    }),
                },
            )),
            ..frame.clone()
        };
        validate_stage_control_request(&status_update).unwrap();

        let cancel = StageControlRequest {
            command: Some(stage_control_request::Command::CancelPrepareStage(
                CancelPrepareStage {
                    topology_id: "topology-a".to_string(),
                    run_id: "run-a".to_string(),
                    stage_id: "stage-1".to_string(),
                    shutdown_generation: 8,
                },
            )),
            ..frame.clone()
        };
        validate_stage_control_request(&cancel).unwrap();

        let missing_command = StageControlRequest {
            command: None,
            ..frame.clone()
        };
        assert!(matches!(
            validate_stage_control_request(&missing_command),
            Err(StageFrameError::MissingStageControlCommand)
        ));

        let wrong_gen = StageControlRequest { r#gen: 1, ..frame };
        assert!(matches!(
            validate_stage_control_request(&wrong_gen),
            Err(StageFrameError::BadGeneration { got: 1 })
        ));
    }

    #[test]
    fn stage_control_response_validates_generation_and_response() {
        let frame = StageControlResponse {
            r#gen: STAGE_PROTOCOL_GENERATION,
            response: Some(stage_control_response::Response::StageReady(StageReady {
                accepted: true,
                status: Some(StageStatus {
                    topology_id: "topology-a".to_string(),
                    run_id: "run-a".to_string(),
                    model_id: "qwen".to_string(),
                    backend: "skippy".to_string(),
                    stage_id: "stage-0".to_string(),
                    stage_index: 0,
                    layer_start: 0,
                    layer_end: 16,
                    state: StageRuntimeState::Ready as i32,
                    bind_addr: "127.0.0.1:0".to_string(),
                    activation_width: 4096,
                    shutdown_generation: 7,
                    ctx_size: 8192,
                    lane_count: 2,
                    projector_path: Some("/models/mmproj.gguf".to_string()),
                    ..Default::default()
                }),
                error: None,
            })),
        };
        let decoded = StageControlResponse::decode(frame.encode_to_vec().as_slice()).unwrap();
        validate_stage_control_response(&decoded).unwrap();
        match decoded.response {
            Some(stage_control_response::Response::StageReady(ready)) => {
                let status = ready.status.expect("stage-ready status");
                assert_eq!(
                    status.projector_path.as_deref(),
                    Some("/models/mmproj.gguf")
                );
                assert_eq!(status.lane_count, 2);
            }
            other => panic!("expected StageReady, got {other:?}"),
        }

        let inventory_response = StageControlResponse {
            response: Some(stage_control_response::Response::LayerInventory(
                LayerInventory {
                    model_id: "qwen".to_string(),
                    package_ref: "hf://repo/model".to_string(),
                    manifest_sha256: "a5".repeat(32),
                    layer_count: 16,
                    source_model_path: Some("/model.gguf".to_string()),
                    source_model_bytes: Some(1024),
                    source_model_kind: SourceModelKind::PlainGguf as i32,
                    ready_ranges: vec![LayerRange {
                        layer_start: 0,
                        layer_end: 8,
                    }],
                    ..Default::default()
                },
            )),
            ..frame.clone()
        };
        validate_stage_control_response(&inventory_response).unwrap();

        let prepare_response = StageControlResponse {
            response: Some(stage_control_response::Response::PrepareStageAccepted(
                PrepareStageAccepted {
                    accepted: true,
                    status: Some(StagePreparationStatus {
                        topology_id: "topology-a".to_string(),
                        run_id: "run-a".to_string(),
                        model_id: "qwen".to_string(),
                        backend: "skippy".to_string(),
                        package_ref: "hf://repo/model".to_string(),
                        manifest_sha256: "a5".repeat(32),
                        stage_id: "stage-1".to_string(),
                        stage_index: 1,
                        layer_start: 8,
                        layer_end: 16,
                        state: StagePreparationState::Assigned as i32,
                        shutdown_generation: 7,
                        ..Default::default()
                    }),
                    error: None,
                },
            )),
            ..frame.clone()
        };
        validate_stage_control_response(&prepare_response).unwrap();

        let ack_response = StageControlResponse {
            response: Some(stage_control_response::Response::StageStatusAck(
                StageStatusAck {
                    accepted: true,
                    error: None,
                },
            )),
            ..frame.clone()
        };
        validate_stage_control_response(&ack_response).unwrap();

        let status_list_response = StageControlResponse {
            response: Some(stage_control_response::Response::StageStatuses(
                StageStatusList {
                    statuses: vec![StageStatus {
                        topology_id: "topology-a".to_string(),
                        run_id: "run-a".to_string(),
                        model_id: "qwen".to_string(),
                        backend: "skippy".to_string(),
                        stage_id: "stage-0".to_string(),
                        stage_index: 0,
                        layer_start: 0,
                        layer_end: 16,
                        state: StageRuntimeState::Ready as i32,
                        bind_addr: "127.0.0.1:51234".to_string(),
                        activation_width: 4096,
                        shutdown_generation: 7,
                        ctx_size: 8192,
                        lane_count: 2,
                        ..Default::default()
                    }],
                },
            )),
            ..frame.clone()
        };
        validate_stage_control_response(&status_list_response).unwrap();

        let missing_response = StageControlResponse {
            response: None,
            ..frame.clone()
        };
        assert!(matches!(
            validate_stage_control_response(&missing_response),
            Err(StageFrameError::MissingStageControlResponse)
        ));

        let wrong_gen = StageControlResponse { r#gen: 1, ..frame };
        assert!(matches!(
            validate_stage_control_response(&wrong_gen),
            Err(StageFrameError::BadGeneration { got: 1 })
        ));
    }

    #[test]
    fn stage_transport_open_validates_generation_sender_and_target() {
        let frame = StageTransportOpen {
            r#gen: STAGE_PROTOCOL_GENERATION,
            requester_id: vec![7u8; 32],
            topology_id: "topology-a".to_string(),
            run_id: "run-a".to_string(),
            stage_id: "stage-1".to_string(),
        };
        validate_stage_transport_open(&frame).unwrap();

        let missing_target = StageTransportOpen {
            stage_id: String::new(),
            ..frame.clone()
        };
        assert!(matches!(
            validate_stage_transport_open(&missing_target),
            Err(StageFrameError::MissingStageTransportTarget)
        ));

        let wrong_gen = StageTransportOpen { r#gen: 1, ..frame };
        assert!(matches!(
            validate_stage_transport_open(&wrong_gen),
            Err(StageFrameError::BadGeneration { got: 1 })
        ));
    }

    #[test]
    fn stage_artifact_transfer_frames_validate_skippy_owned_contract() {
        let request = StageArtifactTransferRequest {
            r#gen: STAGE_PROTOCOL_GENERATION,
            requester_id: vec![7u8; 32],
            topology_id: "topology-a".to_string(),
            run_id: "run-a".to_string(),
            stage_id: "stage-0".to_string(),
            package_ref: "hf://meshllm/demo-layers@abc123".to_string(),
            manifest_sha256: "a".repeat(64),
            relative_path: "layers/layer-000.gguf".to_string(),
            offset: 0,
            expected_size: Some(8),
            expected_sha256: Some("b".repeat(64)),
        };
        let decoded =
            StageArtifactTransferRequest::decode(request.encode_to_vec().as_slice()).unwrap();
        validate_stage_artifact_transfer_request(&decoded).unwrap();
        assert_eq!(decoded.stage_id, "stage-0");

        let mut unsafe_path = request.clone();
        unsafe_path.relative_path = "../layer.gguf".to_string();
        assert!(matches!(
            validate_stage_artifact_transfer_request(&unsafe_path),
            Err(StageFrameError::InvalidArtifactPath)
        ));

        let mut bad_offset = request.clone();
        bad_offset.offset = 9;
        assert!(matches!(
            validate_stage_artifact_transfer_request(&bad_offset),
            Err(StageFrameError::InvalidArtifactOffset)
        ));

        let mut missing_target = request.clone();
        missing_target.topology_id.clear();
        assert!(matches!(
            validate_stage_artifact_transfer_request(&missing_target),
            Err(StageFrameError::MissingStageArtifactTarget)
        ));

        let response = StageArtifactTransferResponse {
            r#gen: STAGE_PROTOCOL_GENERATION,
            accepted: true,
            total_size: 8,
            sha256: Some("b".repeat(64)),
            error: None,
        };
        let decoded =
            StageArtifactTransferResponse::decode(response.encode_to_vec().as_slice()).unwrap();
        validate_stage_artifact_transfer_response(&decoded).unwrap();

        let bad_response_sha = StageArtifactTransferResponse {
            sha256: Some("not-a-sha".to_string()),
            ..response
        };
        assert!(matches!(
            validate_stage_artifact_transfer_response(&bad_response_sha),
            Err(StageFrameError::InvalidArtifactDigestLength { .. })
        ));
    }
}
