//! System One reads on a Laya decision model.
//!
//! Laya scores each option at its own `[MASK]` marker in one encoder pass per
//! question, so there is no chat template or answer canvas: each question
//! becomes `[CLS] <type> question: <instructions> [SEP] ([MASK] option)* [SEP]
//! state [SEP]`, matching the reference `build_sequence`. Options and `state`
//! are rendered in request order with Python `json.dumps` separators, as the
//! checkpoint was trained on.

use std::sync::Arc;

use async_trait::async_trait;
use openai_frontend::{
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionStream, ModelObject,
    OpenAiBackend, OpenAiError, OpenAiRequestContext, OpenAiResult, SystemOneJson,
    SystemOneQuestion, SystemOneRequest, SystemOneResponse, SystemOneUsage,
};
use serde_json::Value;
use skippy_runtime::{LayaModel, LayaQuestionType, LayaSequence};
use tokio::task;

use super::{
    ChoiceOrder, PreparedQuestion, PreparedQuestionKind, answers, prepare_questions,
    validate_request_fields,
};

/// Option text is capped per option before the head budget is applied.
const MAX_OPTION_TOKENS: usize = 48;
/// Smallest option region left for the head text before options are trimmed.
const MIN_OPTION_BUDGET: usize = 16;
const MIN_HEAD_TOKENS: usize = 8;
const NOUL_FALSE_DEFAULT: &str = "no, the statement does not hold";
const NOUL_TRUE_DEFAULT: &str = "yes, the statement holds";

/// Serves `POST /systemone` from a loaded Laya model. Every other OpenAI
/// surface is refused: Laya generates no text.
#[derive(Clone)]
pub struct LayaSystemOneBackend {
    model_id: String,
    model: Arc<LayaModel>,
}

impl LayaSystemOneBackend {
    pub fn new(model_id: impl Into<String>, model: Arc<LayaModel>) -> Self {
        Self {
            model_id: model_id.into(),
            model,
        }
    }

    fn run_system_one(&self, request: SystemOneRequest) -> OpenAiResult<SystemOneResponse> {
        validate_request_fields(&request, &self.model_id)?;
        let state_text = python_text(&request.state);
        let questions_in = request.questions.clone();
        let questions = prepare_questions(request.questions, ChoiceOrder::Request)?;
        let max_markers = self.model.info().max_markers;
        let mut sequences = Vec::with_capacity(questions.len());
        for question in &questions {
            let source = questions_in
                .get(&question.key)
                .expect("prepared questions keep their request keys");
            let options = render_options(source);
            if options.len() > max_markers {
                return Err(OpenAiError::invalid_request(format!(
                    "question {:?}: Laya scores at most {max_markers} options",
                    question.key
                )));
            }
            let instructions = match source {
                SystemOneQuestion::Noul { instructions, .. }
                | SystemOneQuestion::Choice { instructions, .. }
                | SystemOneQuestion::Score { instructions, .. } => instructions
                    .as_ref()
                    .map(|value| python_text(&SystemOneJson::from(value.clone())))
                    .unwrap_or_default(),
            };
            sequences.push(build_sequence(
                &self.model,
                question_type(question),
                &instructions,
                &options,
                &state_text,
            )?);
        }

        let outputs = self.model.read(&sequences).map_err(laya_error)?;
        let info = self.model.info();
        let mut probabilities = Vec::with_capacity(questions.len());
        for ((question, sequence), output) in questions.iter().zip(&sequences).zip(&outputs) {
            let temperature =
                effective_temperature(info.temperature[sequence.question_type.index()]);
            let distribution = softmax(&output.logits, temperature);
            probabilities.push(match question.kind {
                // Laya lists `false` then `true`; the shared mapping reads the
                // `yes` probability first.
                PreparedQuestionKind::Noul { .. } => vec![distribution[1], distribution[0]],
                _ => distribution,
            });
        }
        let input_tokens = sequences
            .iter()
            .map(|sequence| sequence.tokens.len())
            .sum::<usize>();
        Ok(SystemOneResponse {
            model: request.model,
            answers: answers(&questions, &probabilities)?,
            usage: SystemOneUsage {
                input_tokens: u32::try_from(input_tokens).unwrap_or(u32::MAX),
                output_tokens: 0,
            },
        })
    }
}

#[async_trait]
impl OpenAiBackend for LayaSystemOneBackend {
    async fn models(&self) -> OpenAiResult<Vec<ModelObject>> {
        Ok(vec![ModelObject::new(self.model_id.clone())])
    }

    async fn system_one(&self, request: SystemOneRequest) -> OpenAiResult<SystemOneResponse> {
        let backend = self.clone();
        task::spawn_blocking(move || backend.run_system_one(request))
            .await
            .map_err(|error| {
                OpenAiError::backend(format!("System One execution task failed: {error}"))
            })?
    }

    async fn chat_completion(
        &self,
        _request: ChatCompletionRequest,
    ) -> OpenAiResult<ChatCompletionResponse> {
        Err(decision_only())
    }

    async fn chat_completion_stream(
        &self,
        _request: ChatCompletionRequest,
        _context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionStream> {
        Err(decision_only())
    }
}

fn decision_only() -> OpenAiError {
    OpenAiError::unsupported("this Laya decision model only serves POST /systemone")
}

fn laya_error(error: anyhow::Error) -> OpenAiError {
    OpenAiError::backend(format!("Laya read failed: {error:#}"))
}

fn question_type(question: &PreparedQuestion) -> LayaQuestionType {
    match question.kind {
        PreparedQuestionKind::Noul { .. } => LayaQuestionType::Noul,
        PreparedQuestionKind::Choice { .. } => LayaQuestionType::Choice,
        PreparedQuestionKind::Score { .. } => LayaQuestionType::Score,
    }
}

fn type_name(question_type: LayaQuestionType) -> &'static str {
    match question_type {
        LayaQuestionType::Choice => "choice",
        LayaQuestionType::Score => "score",
        LayaQuestionType::Noul => "noul",
    }
}

/// Option lines in marker order, as the reference `render_options` builds them.
fn render_options(question: &SystemOneQuestion) -> Vec<String> {
    match question {
        SystemOneQuestion::Choice { criteria, .. } => criteria
            .iter()
            .map(|(name, description)| {
                let description = python_text(description);
                if description.is_empty() {
                    name.to_string()
                } else {
                    format!("{name}: {description}")
                }
            })
            .collect(),
        SystemOneQuestion::Score { criteria, .. } => criteria
            .iter()
            .enumerate()
            .map(|(index, level)| format!("level {index}: {}", python_text(level)))
            .collect(),
        SystemOneQuestion::Noul { criteria, .. } => {
            let describe = |value: Option<&Value>, default: &str| {
                let text = value
                    .map(|value| python_text(&SystemOneJson::from(value.clone())))
                    .unwrap_or_default();
                if text.is_empty() {
                    default.to_string()
                } else {
                    text
                }
            };
            let criteria = criteria.as_ref();
            vec![
                format!(
                    "false: {}",
                    describe(
                        criteria.and_then(|c| c.r#false.as_ref()),
                        NOUL_FALSE_DEFAULT
                    )
                ),
                format!(
                    "true: {}",
                    describe(criteria.and_then(|c| c.r#true.as_ref()), NOUL_TRUE_DEFAULT)
                ),
            ]
        }
    }
}

fn build_sequence(
    model: &LayaModel,
    question_type: LayaQuestionType,
    instructions: &str,
    options: &[String],
    state: &str,
) -> OpenAiResult<LayaSequence> {
    let info = model.info();
    assemble_sequence(
        SequenceLayout {
            max_len: info.max_len,
            head_max_len: info.head_max_len,
            cls: info.cls_token_id,
            sep: info.sep_token_id,
            mask: info.mask_token_id,
        },
        question_type,
        instructions,
        options,
        state,
        |text| model.tokenize(text).map_err(laya_error),
    )
}

#[derive(Clone, Copy)]
struct SequenceLayout {
    max_len: usize,
    head_max_len: usize,
    cls: i32,
    sep: i32,
    mask: i32,
}

/// Port of the reference `build_sequence`, with the tokenizer injected.
fn assemble_sequence(
    layout: SequenceLayout,
    question_type: LayaQuestionType,
    instructions: &str,
    options: &[String],
    state: &str,
    mut tokenize: impl FnMut(&str) -> OpenAiResult<Vec<i32>>,
) -> OpenAiResult<LayaSequence> {
    let instructions = instructions.replace("[MASK]", " ");
    let mut head = tokenize(&format!(
        "{} question: {instructions}",
        type_name(question_type)
    ))?;

    let mut option_ids = Vec::with_capacity(options.len());
    for option in options {
        let mut ids = vec![layout.mask];
        let mut text = tokenize(&format!(" {}", option.replace("[MASK]", " ")))?;
        text.truncate(MAX_OPTION_TOKENS);
        ids.extend(text);
        option_ids.push(ids);
    }

    let option_tokens = |option_ids: &[Vec<i32>]| option_ids.iter().map(Vec::len).sum::<usize>();
    let mut budget = layout.head_max_len as isize - option_tokens(&option_ids) as isize;
    if budget < MIN_OPTION_BUDGET as isize {
        let per_option = ((layout.head_max_len as isize - MIN_OPTION_BUDGET as isize)
            / option_ids.len().max(1) as isize)
            .max(4) as usize;
        for ids in &mut option_ids {
            ids.truncate(per_option);
        }
        budget = layout.head_max_len as isize - option_tokens(&option_ids) as isize;
    }
    head.truncate(budget.max(MIN_HEAD_TOKENS as isize) as usize);

    let mut tokens = vec![layout.cls];
    tokens.extend(head);
    tokens.push(layout.sep);
    let mut markers = Vec::with_capacity(option_ids.len());
    for ids in option_ids {
        markers.push(tokens.len());
        tokens.extend(ids);
    }
    tokens.push(layout.sep);

    let room = layout.max_len.saturating_sub(tokens.len() + 1);
    let mut state_ids = tokenize(&state.replace("[MASK]", " "))?;
    state_ids.truncate(room);
    tokens.extend(state_ids);
    tokens.push(layout.sep);
    tokens.truncate(layout.max_len);

    let markers = markers
        .into_iter()
        .filter(|marker| *marker < layout.max_len)
        .map(|marker| u32::try_from(marker).unwrap_or(u32::MAX))
        .collect::<Vec<_>>();
    if markers.is_empty() {
        return Err(OpenAiError::invalid_request(
            "question options do not fit in the Laya sequence budget",
        ));
    }
    Ok(LayaSequence {
        tokens,
        question_type,
        markers,
    })
}

/// A missing GGUF temperature is 1; a present one is clamped to [0.5, 5].
fn effective_temperature(temperature: f32) -> f32 {
    if temperature == 0.0 || !temperature.is_finite() {
        1.0
    } else {
        temperature.clamp(0.5, 5.0)
    }
}

fn softmax(logits: &[f32], temperature: f32) -> Vec<f32> {
    let scaled = logits
        .iter()
        .map(|logit| logit / temperature)
        .collect::<Vec<_>>();
    let max = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps = scaled
        .iter()
        .map(|logit| (logit - max).exp())
        .collect::<Vec<_>>();
    let sum = exps.iter().sum::<f32>();
    exps.into_iter().map(|value| value / sum).collect()
}

/// Strings pass through; everything else is Python `json.dumps` with
/// `ensure_ascii=False`, which is how the reference serializes state and criteria.
fn python_text(value: &SystemOneJson) -> String {
    match value {
        SystemOneJson::String(text) => text.clone(),
        other => {
            let mut out = String::new();
            python_dump(other, &mut out);
            out
        }
    }
}

fn python_dump(value: &SystemOneJson, out: &mut String) {
    match value {
        SystemOneJson::Null => out.push_str("null"),
        SystemOneJson::Bool(value) => out.push_str(if *value { "true" } else { "false" }),
        SystemOneJson::Number(number) => {
            match (number.as_i64(), number.as_u64(), number.as_f64()) {
                (Some(value), _, _) => out.push_str(&value.to_string()),
                (None, Some(value), _) => out.push_str(&value.to_string()),
                (None, None, Some(value)) => out.push_str(&python_float(value)),
                _ => out.push_str(&number.to_string()),
            }
        }
        SystemOneJson::String(text) => python_string(text, out),
        SystemOneJson::Array(values) => {
            out.push('[');
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                python_dump(value, out);
            }
            out.push(']');
        }
        SystemOneJson::Object(object) => {
            out.push('{');
            for (index, (key, value)) in object.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                python_string(key, out);
                out.push_str(": ");
                python_dump(value, out);
            }
            out.push('}');
        }
    }
}

fn python_string(text: &str, out: &mut String) {
    out.push('"');
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            control if (control as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", control as u32));
            }
            other => out.push(other),
        }
    }
    out.push('"');
}

/// Python `repr(float)`: shortest round-trip digits, always with a fraction or
/// exponent, and exponents written as `e+NN`/`e-NN` outside [1e-4, 1e16).
fn python_float(value: f64) -> String {
    let magnitude = value.abs();
    if magnitude != 0.0 && !(1e-4..1e16).contains(&magnitude) {
        let formatted = format!("{value:e}");
        let (mantissa, exponent) = formatted
            .split_once('e')
            .expect("exponent formatting always has an exponent");
        let exponent = exponent.parse::<i32>().unwrap_or(0);
        let sign = if exponent < 0 { '-' } else { '+' };
        return format!("{mantissa}e{sign}{:02}", exponent.abs());
    }
    let formatted = value.to_string();
    if formatted.contains('.') {
        formatted
    } else {
        format!("{formatted}.0")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(json: &str) -> SystemOneJson {
        serde_json::from_str(json).expect("valid JSON")
    }

    #[test]
    fn python_text_matches_json_dumps_separators_and_order() {
        let state = parse(
            r#"{"dialogue":[{"role":"user","content":"hi \"there\"\n"}],"n":3,"x":2.5,"ok":true,"none":null}"#,
        );
        assert_eq!(
            python_text(&state),
            r#"{"dialogue": [{"role": "user", "content": "hi \"there\"\n"}], "n": 3, "x": 2.5, "ok": true, "none": null}"#
        );
    }

    #[test]
    fn python_text_keeps_non_ascii_and_passes_strings_through() {
        assert_eq!(python_text(&parse(r#""满意""#)), "满意");
        assert_eq!(python_text(&parse(r#"{"k":"满意"}"#)), r#"{"k": "满意"}"#);
    }

    #[test]
    fn python_float_matches_repr() {
        assert_eq!(python_float(1.0), "1.0");
        assert_eq!(python_float(0.1), "0.1");
        assert_eq!(python_float(-2.5), "-2.5");
        assert_eq!(python_float(1e20), "1e+20");
        assert_eq!(python_float(1.5e-7), "1.5e-07");
    }

    #[test]
    fn noul_options_list_false_then_true_with_defaults() {
        let question: SystemOneQuestion =
            serde_json::from_str(r#"{"type":"noul","instructions":"Ready?"}"#).expect("noul");
        assert_eq!(
            render_options(&question),
            vec![
                format!("false: {NOUL_FALSE_DEFAULT}"),
                format!("true: {NOUL_TRUE_DEFAULT}"),
            ]
        );
    }

    #[test]
    fn choice_and_score_options_follow_request_order() {
        let choice: SystemOneQuestion =
            serde_json::from_str(r#"{"type":"choice","criteria":{"Model B":"","Model A":"fast"}}"#)
                .expect("choice");
        assert_eq!(render_options(&choice), vec!["Model B", "Model A: fast"]);
        let score: SystemOneQuestion =
            serde_json::from_str(r#"{"type":"score","criteria":["low",{"hi":1}]}"#).expect("score");
        assert_eq!(
            render_options(&score),
            vec!["level 0: low", r#"level 1: {"hi": 1}"#]
        );
    }

    fn char_tokens(text: &str) -> OpenAiResult<Vec<i32>> {
        Ok(text.chars().map(|character| character as i32).collect())
    }

    const LAYOUT: SequenceLayout = SequenceLayout {
        max_len: 64,
        head_max_len: 24,
        cls: -1,
        sep: -2,
        mask: -3,
    };

    #[test]
    fn sequence_places_markers_before_each_option_and_state_last() {
        let sequence = assemble_sequence(
            LAYOUT,
            LayaQuestionType::Noul,
            "q",
            &["a".to_string(), "b".to_string()],
            "s",
            char_tokens,
        )
        .expect("sequence");
        // Two options of three tokens leave an 18-token head budget, so the
        // 16-token head fits whole.
        let head = char_tokens("noul question: q").unwrap();
        let mut expected = vec![LAYOUT.cls];
        expected.extend(&head);
        expected.push(LAYOUT.sep);
        expected.extend([LAYOUT.mask, ' ' as i32, 'a' as i32]);
        expected.extend([LAYOUT.mask, ' ' as i32, 'b' as i32]);
        expected.push(LAYOUT.sep);
        expected.push('s' as i32);
        expected.push(LAYOUT.sep);
        assert_eq!(sequence.tokens, expected);
        assert_eq!(sequence.markers, vec![18, 21]);
    }

    #[test]
    fn long_options_are_trimmed_to_share_the_head_budget() {
        let long = "x".repeat(100);
        let sequence = assemble_sequence(
            LAYOUT,
            LayaQuestionType::Choice,
            "q",
            &[long.clone(), long],
            "",
            char_tokens,
        )
        .expect("sequence");
        // (24 - 16) / 2 = 4 tokens per option leaves 16 tokens for the
        // 18-token head.
        assert_eq!(sequence.markers, vec![18, 22]);
    }

    #[test]
    fn state_is_truncated_to_the_sequence_budget() {
        let sequence = assemble_sequence(
            LAYOUT,
            LayaQuestionType::Noul,
            "",
            &["a".to_string(), "b".to_string()],
            &"z".repeat(500),
            char_tokens,
        )
        .expect("sequence");
        assert_eq!(sequence.tokens.len(), LAYOUT.max_len);
        assert_eq!(*sequence.tokens.last().unwrap(), LAYOUT.sep);
    }

    #[test]
    fn temperature_defaults_and_clamps() {
        assert_eq!(effective_temperature(0.0), 1.0);
        assert_eq!(effective_temperature(0.1), 0.5);
        assert_eq!(effective_temperature(9.0), 5.0);
        assert_eq!(effective_temperature(1.3), 1.3);
    }

    #[test]
    fn softmax_applies_temperature() {
        let flat = softmax(&[1.0, 1.0], 1.0);
        assert!((flat[0] - 0.5).abs() < 1e-6);
        let sharp = softmax(&[2.0, 0.0], 0.5);
        let soft = softmax(&[2.0, 0.0], 2.0);
        assert!(sharp[0] > soft[0]);
    }
}
