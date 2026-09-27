//! System One reads on a Laya decision model.
//!
//! Laya scores each option at its own `[MASK]` marker in one encoder pass per
//! question, so there is no chat template or answer canvas: each question
//! becomes `[CLS] <type> question: <instructions> [SEP] ([MASK] option)* [SEP]
//! state [SEP]`, matching the reference `build_sequence`. Options and `state`
//! are rendered in request order with Python `json.dumps` separators, as the
//! checkpoint was trained on.

use super::{
    DecisionError, DecisionModel, DecisionOutput, DecisionQuestion, DecisionQuestionKind,
    DecisionRequest, DecisionValue, check_output,
};
use crate::laya::{LayaModel, LayaQuestionType, LayaSequence};

/// Option text is capped per option before the head budget is applied.
const MAX_OPTION_TOKENS: usize = 48;
/// Smallest option region left for the head text before options are trimmed.
const MIN_OPTION_BUDGET: usize = 16;
const MIN_HEAD_TOKENS: usize = 8;
const NOUL_FALSE_DEFAULT: &str = "no, the statement does not hold";
const NOUL_TRUE_DEFAULT: &str = "yes, the statement holds";

impl DecisionModel for LayaModel {
    fn decide(&self, request: &DecisionRequest) -> Result<DecisionOutput, DecisionError> {
        let info = self.info();
        let layout = SequenceLayout {
            max_len: info.max_len,
            head_max_len: info.head_max_len,
            cls: info.cls_token_id,
            sep: info.sep_token_id,
            mask: info.mask_token_id,
        };
        let state = python_text(&request.state);
        let mut sequences = Vec::with_capacity(request.questions.len());
        for question in &request.questions {
            let options = render_options(question);
            if options.len() > info.max_markers {
                return Err(DecisionError::InvalidRequest(format!(
                    "question {:?}: Laya scores at most {} options",
                    question.key, info.max_markers
                )));
            }
            let instructions = question
                .instructions
                .as_ref()
                .map(python_text)
                .unwrap_or_default();
            sequences.push(assemble_sequence(
                layout,
                question_type(question),
                &instructions,
                &options,
                &state,
                |text| Ok(self.tokenize(text)?),
            )?);
        }

        let outputs = self.read(&sequences)?;
        let probabilities = sequences
            .iter()
            .zip(&outputs)
            .map(|(sequence, output)| {
                let temperature =
                    effective_temperature(info.temperature[sequence.question_type.index()]);
                let distribution = softmax(&output.logits, temperature);
                match sequence.question_type {
                    // Laya lists `false` then `true`; decisions report `true` first.
                    LayaQuestionType::Noul => vec![distribution[1], distribution[0]],
                    _ => distribution,
                }
            })
            .collect::<Vec<_>>();
        check_output(request, &probabilities)?;
        Ok(DecisionOutput {
            probabilities,
            input_tokens: sequences.iter().map(|sequence| sequence.tokens.len()).sum(),
        })
    }
}

fn question_type(question: &DecisionQuestion) -> LayaQuestionType {
    match question.kind {
        DecisionQuestionKind::Noul { .. } => LayaQuestionType::Noul,
        DecisionQuestionKind::Choice { .. } => LayaQuestionType::Choice,
        DecisionQuestionKind::Score { .. } => LayaQuestionType::Score,
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
fn render_options(question: &DecisionQuestion) -> Vec<String> {
    match &question.kind {
        DecisionQuestionKind::Choice { options } => options
            .iter()
            .map(|(name, description)| {
                let description = python_text(description);
                if description.is_empty() {
                    name.clone()
                } else {
                    format!("{name}: {description}")
                }
            })
            .collect(),
        DecisionQuestionKind::Score { levels } => levels
            .iter()
            .enumerate()
            .map(|(index, level)| format!("level {index}: {}", python_text(level)))
            .collect(),
        DecisionQuestionKind::Noul {
            when_true,
            when_false,
        } => {
            let describe = |value: Option<&DecisionValue>, default: &str| {
                let text = value.map(python_text).unwrap_or_default();
                if text.is_empty() {
                    default.to_string()
                } else {
                    text
                }
            };
            vec![
                format!(
                    "false: {}",
                    describe(when_false.as_ref(), NOUL_FALSE_DEFAULT)
                ),
                format!("true: {}", describe(when_true.as_ref(), NOUL_TRUE_DEFAULT)),
            ]
        }
    }
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
    mut tokenize: impl FnMut(&str) -> Result<Vec<i32>, DecisionError>,
) -> Result<LayaSequence, DecisionError> {
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
        return Err(DecisionError::InvalidRequest(
            "question options do not fit in the Laya sequence budget".into(),
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
fn python_text(value: &DecisionValue) -> String {
    match value {
        DecisionValue::String(text) => text.clone(),
        other => {
            let mut out = String::new();
            python_dump(other, &mut out);
            out
        }
    }
}

fn python_dump(value: &DecisionValue, out: &mut String) {
    match value {
        DecisionValue::Null => out.push_str("null"),
        DecisionValue::Bool(value) => out.push_str(if *value { "true" } else { "false" }),
        DecisionValue::Number(number) => {
            match (number.as_i64(), number.as_u64(), number.as_f64()) {
                (Some(value), _, _) => out.push_str(&value.to_string()),
                (None, Some(value), _) => out.push_str(&value.to_string()),
                (None, None, Some(value)) => out.push_str(&python_float(value)),
                _ => out.push_str(&number.to_string()),
            }
        }
        DecisionValue::String(text) => python_string(text, out),
        DecisionValue::Array(values) => {
            out.push('[');
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                python_dump(value, out);
            }
            out.push(']');
        }
        DecisionValue::Object(entries) => {
            out.push('{');
            for (index, (key, value)) in entries.iter().enumerate() {
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

    fn parse(json: &str) -> DecisionValue {
        serde_json::from_str(json).expect("valid JSON")
    }

    fn question(kind: DecisionQuestionKind) -> DecisionQuestion {
        DecisionQuestion {
            key: "q".into(),
            instructions: None,
            kind,
        }
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
        assert_eq!(python_text(&DecisionValue::String("满意".into())), "满意");
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
        let noul = question(DecisionQuestionKind::Noul {
            when_true: None,
            when_false: None,
        });
        assert_eq!(
            render_options(&noul),
            vec![
                format!("false: {NOUL_FALSE_DEFAULT}"),
                format!("true: {NOUL_TRUE_DEFAULT}"),
            ]
        );
    }

    #[test]
    fn choice_and_score_options_follow_request_order() {
        let choice = question(DecisionQuestionKind::Choice {
            options: vec![
                ("Model B".into(), DecisionValue::String(String::new())),
                ("Model A".into(), DecisionValue::String("fast".into())),
            ],
        });
        assert_eq!(render_options(&choice), vec!["Model B", "Model A: fast"]);
        let score = question(DecisionQuestionKind::Score {
            levels: vec![DecisionValue::String("low".into()), parse(r#"{"hi":1}"#)],
        });
        assert_eq!(
            render_options(&score),
            vec!["level 0: low", r#"level 1: {"hi": 1}"#]
        );
    }

    fn char_tokens(text: &str) -> Result<Vec<i32>, DecisionError> {
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
