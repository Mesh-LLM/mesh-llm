pub(super) fn is_retryable_split_start_failure(message: &str) -> bool {
    split_participants_are_still_converging(message)
        || split_control_transport_failed(message)
        || split_stage_source_preparation_timed_out(message)
}

fn split_participants_are_still_converging(message: &str) -> bool {
    message.contains("at least two participating nodes")
        || message.contains("at least two stage participants")
        || message.contains("split_capacity_shortfall")
        || (message.contains("split topology lock stage")
            && message.contains("matched 0 eligible nodes"))
}

fn split_control_transport_failed(message: &str) -> bool {
    let is_control_operation = message.contains("load split stage")
        || message.contains("prepare split stage")
        || message.contains("stage_control_unreachable");
    let is_transport_failure = message.contains("connection lost")
        || message.contains("stream finished early")
        || message.contains("timeout waiting for stage control response");
    is_control_operation && is_transport_failure
}

fn split_stage_source_preparation_timed_out(message: &str) -> bool {
    message.contains("stage_source_prepare_timeout")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn participant_shortage_is_retryable() {
        assert!(is_retryable_split_start_failure(
            "split runtime needs at least two participating nodes for model"
        ));
        assert!(is_retryable_split_start_failure(
            "split runtime needs at least two stage participants"
        ));
    }

    #[test]
    fn transient_capacity_shortfall_is_retryable() {
        assert!(is_retryable_split_start_failure(
            "split_capacity_shortfall: unable to plan split topology: max_placeable_layers_at_evaluated_shape=21/66"
        ));
    }

    #[test]
    fn missing_locked_participant_is_retryable() {
        assert!(is_retryable_split_start_failure(
            "split topology lock stage 3 selector \"worker\" matched 0 eligible nodes; available: local"
        ));
    }

    #[test]
    fn ambiguous_or_invalid_lock_is_not_retryable() {
        assert!(!is_retryable_split_start_failure(
            "split topology lock stage 3 selector \"worker\" matched 2 eligible nodes"
        ));
        assert!(!is_retryable_split_start_failure(
            "split topology lock manifest abc does not match resolved package manifest def"
        ));
    }

    #[test]
    fn stage_control_transport_failure_is_retryable() {
        assert!(is_retryable_split_start_failure(
            "load split stage stage-1: connection lost: closed"
        ));
        assert!(is_retryable_split_start_failure(
            "prepare split stage stage-2: stage_control_unreachable: stream finished early"
        ));
        assert!(is_retryable_split_start_failure(
            "load split stage stage-3: timeout waiting for stage control response"
        ));
    }

    #[test]
    fn stage_source_preparation_timeout_is_retryable() {
        assert!(is_retryable_split_start_failure(
            "prepare split stage stage-1: stage_source_prepare_timeout: timed out waiting for stage source availability after 30m"
        ));
    }

    #[test]
    fn runtime_or_unrelated_transport_failure_is_not_retryable() {
        assert!(!is_retryable_split_start_failure(
            "load skippy stage 0 runtime: unable to allocate CUDA0 buffer"
        ));
        assert!(!is_retryable_split_start_failure(
            "artifact transfer connection lost: closed"
        ));
    }
}
