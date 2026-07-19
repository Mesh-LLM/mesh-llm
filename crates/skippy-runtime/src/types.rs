use anyhow::{Result, anyhow};
use skippy_ffi::{
    ActivationDType, ActivationDesc as RawActivationDesc, ActivationLayout,
    GenerationSignalWindow as RawGenerationSignalWindow, KvPageDesc as RawKvPageDesc,
    LogitBias as RawLogitBias, SamplingConfig as RawSamplingConfig, TensorRole,
    TokenSignal as RawTokenSignal,
};

pub const MAX_LOGIT_BIAS: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub layer_index: Option<u32>,
    pub role: TensorRole,
    pub ggml_type: u32,
    pub byte_size: u64,
    pub element_count: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivationDesc {
    pub version: u32,
    pub dtype: ActivationDType,
    pub layout: ActivationLayout,
    pub producer_stage_index: i32,
    pub layer_start: i32,
    pub layer_end: i32,
    pub token_count: u32,
    pub sequence_count: u32,
    pub payload_bytes: u64,
    pub flags: u64,
}

impl ActivationDesc {
    pub(crate) fn as_raw(&self) -> RawActivationDesc {
        RawActivationDesc {
            version: self.version,
            dtype: self.dtype,
            layout: self.layout,
            producer_stage_index: self.producer_stage_index,
            layer_start: self.layer_start,
            layer_end: self.layer_end,
            token_count: self.token_count,
            sequence_count: self.sequence_count,
            payload_bytes: self.payload_bytes,
            flags: self.flags,
        }
    }
}

impl From<RawActivationDesc> for ActivationDesc {
    fn from(raw: RawActivationDesc) -> Self {
        Self {
            version: raw.version,
            dtype: raw.dtype,
            layout: raw.layout,
            producer_stage_index: raw.producer_stage_index,
            layer_start: raw.layer_start,
            layer_end: raw.layer_end,
            token_count: raw.token_count,
            sequence_count: raw.sequence_count,
            payload_bytes: raw.payload_bytes,
            flags: raw.flags,
        }
    }
}

pub(crate) fn empty_raw_activation_desc() -> RawActivationDesc {
    RawActivationDesc {
        version: 0,
        dtype: ActivationDType::Unknown,
        layout: ActivationLayout::Opaque,
        producer_stage_index: -1,
        layer_start: 0,
        layer_end: 0,
        token_count: 0,
        sequence_count: 0,
        payload_bytes: 0,
        flags: 0,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActivationFrame {
    pub desc: ActivationDesc,
    pub payload: Vec<u8>,
}

impl ActivationFrame {
    pub(crate) fn validate_payload_len(&self) -> Result<()> {
        let payload_len = u64::try_from(self.payload.len())
            .map_err(|_| anyhow!("activation payload length exceeds u64"))?;
        if self.desc.payload_bytes != payload_len {
            return Err(anyhow!(
                "activation payload length {} does not match descriptor payload_bytes {}",
                self.payload.len(),
                self.desc.payload_bytes
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeKvPageDesc {
    pub version: u32,
    pub layer_start: i32,
    pub layer_end: i32,
    pub token_start: u64,
    pub token_count: u64,
    pub layer_count: u32,
    pub k_type: u32,
    pub v_type: u32,
    pub k_row_bytes: u32,
    pub v_row_bytes: u32,
    pub v_element_bytes: u32,
    pub payload_bytes: u64,
    pub flags: u64,
}

impl RuntimeKvPageDesc {
    pub(crate) fn as_raw(&self) -> RawKvPageDesc {
        RawKvPageDesc {
            version: self.version,
            layer_start: self.layer_start,
            layer_end: self.layer_end,
            token_start: self.token_start,
            token_count: self.token_count,
            layer_count: self.layer_count,
            k_type: self.k_type,
            v_type: self.v_type,
            k_row_bytes: self.k_row_bytes,
            v_row_bytes: self.v_row_bytes,
            v_element_bytes: self.v_element_bytes,
            payload_bytes: self.payload_bytes,
            flags: self.flags,
        }
    }
}

impl From<RawKvPageDesc> for RuntimeKvPageDesc {
    fn from(raw: RawKvPageDesc) -> Self {
        Self {
            version: raw.version,
            layer_start: raw.layer_start,
            layer_end: raw.layer_end,
            token_start: raw.token_start,
            token_count: raw.token_count,
            layer_count: raw.layer_count,
            k_type: raw.k_type,
            v_type: raw.v_type,
            k_row_bytes: raw.k_row_bytes,
            v_row_bytes: raw.v_row_bytes,
            v_element_bytes: raw.v_element_bytes,
            payload_bytes: raw.payload_bytes,
            flags: raw.flags,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeKvPage {
    pub desc: RuntimeKvPageDesc,
    pub payload: Vec<u8>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TokenSignal {
    pub entropy: f32,
    pub top_logprob: f32,
    pub second_logprob: f32,
    pub margin: f32,
    pub top_token: i32,
    pub second_token: i32,
}

impl From<RawTokenSignal> for TokenSignal {
    fn from(raw: RawTokenSignal) -> Self {
        Self {
            entropy: raw.entropy,
            top_logprob: raw.top_logprob,
            second_logprob: raw.second_logprob,
            margin: raw.margin,
            top_token: raw.top_token,
            second_token: raw.second_token,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GenerationSignalWindow {
    pub token_count: u32,
    pub mean_entropy: f32,
    pub max_entropy: f32,
    pub mean_margin: f32,
    pub min_margin: f32,
    pub high_entropy_count: u32,
    pub repetition_count: u32,
}

impl From<RawGenerationSignalWindow> for GenerationSignalWindow {
    fn from(raw: RawGenerationSignalWindow) -> Self {
        Self {
            token_count: raw.token_count,
            mean_entropy: raw.mean_entropy,
            max_entropy: raw.max_entropy,
            mean_margin: raw.mean_margin,
            min_margin: raw.min_margin,
            high_entropy_count: raw.high_entropy_count,
            repetition_count: raw.repetition_count,
        }
    }
}

pub struct DecodeFrameBatchOutput {
    pub predicted_token: i32,
    pub output: ActivationFrame,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaInput {
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MediaPrefill {
    pub token_count: usize,
    pub position: u64,
    pub first_token: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaPrefillChunkFrame {
    pub token_count: usize,
    pub positions: Vec<i32>,
    pub output: ActivationFrame,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaPrefillFrame {
    pub token_count: usize,
    pub position: u64,
    pub positions: Vec<i32>,
    pub output: ActivationFrame,
    pub chunks: Vec<MediaPrefillChunkFrame>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct LogitBias {
    pub token_id: i32,
    pub bias: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SamplingConfig {
    pub enabled: bool,
    pub seed: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub min_p: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub repeat_penalty: f32,
    pub penalty_last_n: i32,
    pub logit_bias: Vec<LogitBias>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            seed: 0,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repeat_penalty: 1.0,
            penalty_last_n: -1,
            logit_bias: Vec::new(),
        }
    }
}

impl SamplingConfig {
    pub(crate) fn as_raw(&self) -> RawSamplingConfig {
        let mut logit_bias = [RawLogitBias {
            token_id: 0,
            bias: 0.0,
        }; MAX_LOGIT_BIAS];
        for (target, source) in logit_bias.iter_mut().zip(
            self.logit_bias
                .iter()
                .take(self.logit_bias.len().min(MAX_LOGIT_BIAS)),
        ) {
            *target = RawLogitBias {
                token_id: source.token_id,
                bias: source.bias,
            };
        }
        RawSamplingConfig {
            version: 1,
            flags: u32::from(self.enabled),
            seed: self.seed,
            top_k: self.top_k,
            penalty_last_n: self.penalty_last_n,
            temperature: self.temperature,
            top_p: self.top_p,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            repeat_penalty: self.repeat_penalty,
            logit_bias_count: self.logit_bias.len().min(MAX_LOGIT_BIAS) as u32,
            min_p: self.min_p,
            logit_bias,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTemplateMessage {
    pub role: String,
    pub content: String,
}

impl ChatTemplateMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatReasoningFormat {
    Auto,
    None,
    Deepseek,
    DeepseekLegacy,
    Hidden,
}

impl ChatReasoningFormat {
    pub const fn parser_name(self) -> &'static str {
        match self {
            Self::Auto | Self::Hidden => "auto",
            Self::None => "none",
            Self::Deepseek => "deepseek",
            Self::DeepseekLegacy => "deepseek-legacy",
        }
    }

    pub const fn parses_reasoning(self) -> bool {
        !matches!(self, Self::None)
    }

    pub const fn exposes_reasoning(self) -> bool {
        !matches!(self, Self::None | Self::Hidden)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChatTemplateOptions {
    pub add_assistant: bool,
    pub enable_thinking: Option<bool>,
    pub reasoning_format: Option<ChatReasoningFormat>,
}

impl Default for ChatTemplateOptions {
    fn default() -> Self {
        Self {
            add_assistant: true,
            enable_thinking: None,
            reasoning_format: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTemplateJsonOptions {
    pub add_assistant: bool,
    pub enable_thinking: Option<bool>,
    pub reasoning_format: Option<ChatReasoningFormat>,
    pub tools_json: Option<String>,
    pub tool_choice_json: Option<String>,
    pub parallel_tool_calls: bool,
}

impl Default for ChatTemplateJsonOptions {
    fn default() -> Self {
        Self {
            add_assistant: true,
            enable_thinking: None,
            reasoning_format: None,
            tools_json: None,
            tool_choice_json: None,
            parallel_tool_calls: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTemplateJsonResult {
    pub prompt: String,
    pub metadata_json: String,
}
