use std::{
    env,
    path::{Path, PathBuf},
};

use serde_json::Value;

use super::{
    ChatReasoningFormat, ChatTemplateJsonOptions, FlashAttentionType, GGML_TYPE_F16, ModelInfo,
    RuntimeConfig, RuntimeLoadMode, StageModel, TensorRole,
};

fn open_model(model_path: &Path) -> anyhow::Result<StageModel> {
    let info = ModelInfo::open(model_path)?;
    let layer_end = info
        .tensors()?
        .into_iter()
        .filter(|tensor| tensor.role == TensorRole::Layer)
        .filter_map(|tensor| tensor.layer_index)
        .max()
        .map(|layer| layer + 1)
        .unwrap_or(1);
    StageModel::open(
        model_path,
        &RuntimeConfig {
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
        },
    )
}

#[test]
fn command_a_disabled_thinking_response_parses_when_model_is_configured() -> anyhow::Result<()> {
    let Some(model_path) = env::var_os("SKIPPY_COMMAND_A_MODEL").map(PathBuf::from) else {
        eprintln!("skipping Command-A chat smoke: SKIPPY_COMMAND_A_MODEL is not set");
        return Ok(());
    };
    let model = open_model(&model_path)?;
    let rendered = model.apply_chat_template_json(
        r#"[{"role":"user","content":"Reply with exactly LOCAL_FIXED"}]"#,
        ChatTemplateJsonOptions {
            enable_thinking: Some(false),
            reasoning_format: Some(ChatReasoningFormat::Hidden),
            ..ChatTemplateJsonOptions::default()
        },
    )?;
    assert!(!rendered.prompt.contains("## Reasoning"));

    let parsed = model.parse_chat_response_json(
        "<|START_RESPONSE|>LOCAL_FIXED<|END_RESPONSE|>",
        &rendered.metadata_json,
        false,
    )?;
    let message: Value = serde_json::from_str(&parsed)?;
    assert_eq!(
        message.get("content").and_then(Value::as_str),
        Some("LOCAL_FIXED")
    );
    Ok(())
}

#[test]
fn command_r_tool_result_history_uses_tool_template_when_model_is_configured() -> anyhow::Result<()>
{
    let Some(model_path) = env::var_os("SKIPPY_COMMAND_R_MODEL").map(PathBuf::from) else {
        eprintln!("skipping Command-R tool-result smoke: SKIPPY_COMMAND_R_MODEL is not set");
        return Ok(());
    };
    let model = open_model(&model_path)?;
    let rendered = model.apply_chat_template_json(
        r#"[
            {"role":"user","content":"Look up the codeword."},
            {"role":"assistant","content":null,"tool_calls":[{
                "id":"call_0","type":"function","function":{
                    "name":"lookup_fixture_fact","arguments":"{\"key\":\"codeword\"}"
                }
            }]},
            {"role":"tool","tool_call_id":"call_0","name":"lookup_fixture_fact",
             "content":"{\"key\":\"codeword\",\"value\":\"signal-7429\"}"}
        ]"#,
        ChatTemplateJsonOptions {
            enable_thinking: Some(false),
            reasoning_format: Some(ChatReasoningFormat::Hidden),
            ..ChatTemplateJsonOptions::default()
        },
    )?;
    assert!(rendered.prompt.contains("<|START_TOOL_RESULT|>"));
    assert!(rendered.prompt.contains("signal-7429"));
    let metadata: Value = serde_json::from_str(&rendered.metadata_json)?;
    assert_eq!(
        metadata.get("generation_prompt").and_then(Value::as_str),
        Some("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>")
    );

    let parsed = model.parse_chat_response_json(
        "<|START_RESPONSE|>signal-7429<|END_RESPONSE|>",
        &rendered.metadata_json,
        false,
    )?;
    let message: Value = serde_json::from_str(&parsed)?;
    assert_eq!(
        message.get("content").and_then(Value::as_str),
        Some("signal-7429")
    );
    Ok(())
}
