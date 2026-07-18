#[cfg(test)]
mod tests {
    use anyhow::Result;
    use serde_json::Value;

    use super::{
        ChatReasoningFormat, ChatTemplateJsonOptions, ChatTemplateMessage, FlashAttentionType,
        GGML_TYPE_F16, ModelInfo, NativeMtpDraft, RuntimeConfig, RuntimeLoadMode, SamplingConfig,
        StageModel, StageSession, Status, TensorRole, format_skippy_error,
    };
    use std::{env, path::PathBuf};

    fn correctness_model() -> Option<PathBuf> {
        env::var_os("SKIPPY_CORRECTNESS_MODEL").map(PathBuf::from)
    }

    fn infer_layer_end(path: &PathBuf) -> anyhow::Result<u32> {
        let info = ModelInfo::open(path)?;
        let layer_end = info
            .tensors()?
            .into_iter()
            .filter(|tensor| tensor.role == TensorRole::Layer)
            .filter_map(|tensor| tensor.layer_index)
            .max()
            .map(|layer| layer + 1)
            .unwrap_or(1);
        Ok(layer_end)
    }

    #[test]
    fn invalid_selected_backend_device_fails_before_model_open() {
        let _native_log_guard = crate::logging::native_log_test_guard();
        let config = RuntimeConfig {
            selected_backend_device: Some("definitely-not-a-device".to_string()),
            ..RuntimeConfig::default()
        };

        let error = match StageModel::open("/definitely/missing/model.gguf", &config) {
            Ok(_) => panic!("invalid device should fail before model load"),
            Err(error) => error.to_string(),
        };

        assert!(
            error.contains("unknown selected backend device: definitely-not-a-device"),
            "unexpected error: {error}"
        );
    }

    fn open_correctness_model(model_path: &PathBuf) -> anyhow::Result<StageModel> {
        let layer_end = infer_layer_end(model_path)?;
        let config = RuntimeConfig {
            stage_index: 0,
            layer_start: 0,
            layer_end,
            ctx_size: 256,
            lane_count: 1,
            n_batch: None,
            n_ubatch: None,
            n_threads: None,
            n_threads_batch: None,
            n_gpu_layers: 0,
            mmap: None,
            mlock: false,
            selected_backend_device: None,
            cache_type_k: GGML_TYPE_F16,
            cache_type_v: GGML_TYPE_F16,
            flash_attn_type: FlashAttentionType::Auto,
            load_mode: RuntimeLoadMode::RuntimeSlice,
            projector_path: None,
            include_embeddings: true,
            include_output: true,
            filter_tensors_on_load: false,
        };
        StageModel::open(model_path, &config)
    }

    #[test]
    fn chat_template_applies_when_model_is_configured() -> anyhow::Result<()> {
        let Some(model_path) = correctness_model() else {
            eprintln!("skipping chat template smoke: SKIPPY_CORRECTNESS_MODEL is not set");
            return Ok(());
        };
        let model = open_correctness_model(&model_path)?;
        let prompt = model.apply_chat_template(
            &[
                ChatTemplateMessage::new("system", "You are concise."),
                ChatTemplateMessage::new("user", "Template smoke prompt."),
            ],
            true,
        )?;
        assert!(prompt.contains("Template smoke prompt."));
        assert!(prompt.len() >= "Template smoke prompt.".len());
        Ok(())
    }

    // Requires SKIPPY_CORRECTNESS_MODEL to point at a reasoning-capable model
    // family whose chat parser extracts <think> blocks (e.g. Qwen3).
    #[test]
    fn chat_reasoning_markers_are_stripped_and_extracted_when_model_is_configured()
    -> anyhow::Result<()> {
        let Some(model_path) = correctness_model() else {
            eprintln!("skipping chat reasoning smoke: SKIPPY_CORRECTNESS_MODEL is not set");
            return Ok(());
        };
        let model = open_correctness_model(&model_path)?;
        let rendered = model.apply_chat_template_json(
            r#"[{"role":"user","content":"Say hi."}]"#,
            ChatTemplateJsonOptions {
                reasoning_format: Some(ChatReasoningFormat::Hidden),
                ..ChatTemplateJsonOptions::default()
            },
        )?;
        let metadata: Value = serde_json::from_str(&rendered.metadata_json)?;
        assert_eq!(
            metadata.get("reasoning_format").and_then(Value::as_str),
            Some("auto"),
        );

        // The generation prompt may already open the thought block, in which
        // case the model output continues inside it without the opening tag.
        let generation_prompt = metadata
            .get("generation_prompt")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let generated = if generation_prompt.contains("<think>") {
            "Consider the greeting.</think>Hi there!"
        } else {
            "<think>Consider the greeting.</think>Hi there!"
        };
        let parsed = model.parse_chat_response_json(generated, &rendered.metadata_json, false)?;
        let message: Value = serde_json::from_str(&parsed)?;
        let content = message
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or_default();
        assert!(
            !content.contains("<think>") && !content.contains("</think>"),
            "reasoning markers must be stripped from content: {content:?}"
        );
        assert!(
            content.contains("Hi there!"),
            "visible content must survive reasoning extraction: {content:?}"
        );
        assert_eq!(
            message.get("reasoning_content").and_then(Value::as_str),
            Some("Consider the greeting."),
            "reasoning content must be extracted from the thought block"
        );
        Ok(())
    }

    #[test]
    fn format_skippy_error_omits_abi_envelope() {
        let err = format_skippy_error(Status::RuntimeError, "something broke");
        assert!(
            !err.contains("skippy ABI call failed"),
            "error format must not contain the old ABI envelope prefix: {err}"
        );
        assert!(
            err.contains("RuntimeError"),
            "error must contain the status variant"
        );
        assert!(
            err.contains("something broke"),
            "error must contain the message"
        );
    }

    #[test]
    fn format_skippy_error_works_without_message() {
        let err = format_skippy_error(Status::Unsupported, "");
        assert!(!err.contains("skippy ABI call failed"));
        assert!(err.contains("Unsupported"));
    }

    #[test]
    fn format_skippy_error_covers_all_status_variants() {
        for status in [
            Status::Error,
            Status::InvalidArgument,
            Status::Unsupported,
            Status::BufferTooSmall,
            Status::IoError,
            Status::ModelError,
            Status::RuntimeError,
        ] {
            let err = format_skippy_error(status, "test");
            assert!(
                !err.contains("skippy ABI call failed"),
                "error must not contain ABI envelope for {status:?}: {err}"
            );
            assert!(err.contains("test"));
        }
    }

    #[test]
    fn configure_chat_sampling_survives_bad_metadata_json() -> anyhow::Result<()> {
        let Some(model_path) = correctness_model() else {
            eprintln!("skipping: SKIPPY_CORRECTNESS_MODEL is not set");
            return Ok(());
        };
        let model = open_correctness_model(&model_path)?;
        let mut session = model.create_session()?;
        let sampling = SamplingConfig {
            temperature: 0.0,
            ..Default::default()
        };
        // Send deliberately malformed JSON — the C++ catch blocks
        // should clear chat sampling and return success instead of
        // surfacing the parse error as a fatal status.
        let result = session.configure_chat_sampling("this is not valid json", 0, Some(&sampling));
        assert!(
            result.is_ok(),
            "configure_chat_sampling should return Ok even with bad metadata: {result:?}"
        );
        Ok(())
    }

    #[test]
    fn stage_session_exposes_non_frame_native_mtp_decode_api() {
        type DecodeStepSampledMtp = fn(
            &mut StageSession,
            i32,
            Option<&SamplingConfig>,
            usize,
        ) -> Result<(i32, Option<NativeMtpDraft>)>;

        let _decode: DecodeStepSampledMtp = StageSession::decode_step_sampled_mtp;
    }
}

#[cfg(test)]
#[test]
fn model_open_events_success() {
    runtime_events::tests::assert_model_open_events_success();
}

#[cfg(test)]
#[test]
fn model_open_events_handled_failure() {
    runtime_events::tests::assert_model_open_events_handled_failure();
}

#[cfg(test)]
#[test]
fn model_open_events_missing_terminal_callback_uses_return() {
    runtime_events::tests::assert_model_open_events_missing_terminal_callback_uses_return();
}

#[cfg(test)]
#[test]
fn model_open_events_forwarded_before_open_returns() {
    runtime_events::tests::assert_model_open_events_forwarded_before_open_returns();
}

#[cfg(test)]
#[test]
fn model_open_events_feature_missing_falls_back() {
    runtime_events::tests::assert_model_open_events_feature_missing_falls_back();
}
