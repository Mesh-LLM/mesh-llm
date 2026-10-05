//! System One reads on DiffusionGemma: one zero-self-conditioning diffusion
//! step over a fixed answer canvas, restricted to each question's labels.

use serde_json::Value;

use super::{
    DecisionError, DecisionModel, DecisionOutput, DecisionQuestion, DecisionQuestionKind,
    DecisionRequest, DecisionValue, check_output,
};
use crate::{ChatTemplateMessage, ChatTemplateOptions, StageModel, SystemOneReadSlot};

const TURN_CLOSE_TOKEN: i32 = 106;
const PAD_TOKEN: i32 = 0;
/// Length of the deterministic word drawn from the model tokenizer for each
/// randomized canvas slot.
const FILLER_WORD_LEN: usize = 8;
const SCAFFOLD: &str = "<|channel>thought\n<channel|>";
const CHOICE_LABELS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ";
/// Questions beyond this many switch the answer line to the indexed format.
const LINE_FORMAT_MAX_QUESTIONS: usize = 10;

/// A question as the canvas lays it out: an id, its label strings, and choice
/// options sorted by name, which is how the canvas assigns letters.
struct CanvasQuestion {
    key: String,
    id: String,
    instructions: String,
    labels: Vec<String>,
    kind: CanvasKind,
}

enum CanvasKind {
    Noul {
        true_description: String,
        false_description: String,
    },
    Choice {
        /// `(name, description)` sorted by name.
        choices: Vec<(String, String)>,
        /// For each sorted choice, its index in request order.
        request_index: Vec<usize>,
    },
    Score {
        legend: Vec<String>,
    },
}

#[derive(Clone, Copy)]
enum AnswerFormat {
    Lines,
    Indexed,
}

impl DecisionModel for StageModel {
    fn decide(&self, request: &DecisionRequest) -> Result<DecisionOutput, DecisionError> {
        let questions = canvas_questions(&request.questions)?;
        let format = if questions.len() <= LINE_FORMAT_MAX_QUESTIONS {
            AnswerFormat::Lines
        } else {
            AnswerFormat::Indexed
        };
        let prompt = self.apply_chat_template_with_options(
            &[
                ChatTemplateMessage::new("system", system_text(&questions, format)),
                ChatTemplateMessage::new("user", value_text(&request.state.to_json())),
            ],
            ChatTemplateOptions {
                add_assistant: true,
                enable_thinking: Some(false),
                ..ChatTemplateOptions::default()
            },
        )?;
        let prompt_tokens = self.tokenize(&prompt, true)?;
        if prompt_tokens.is_empty() {
            return Err(DecisionError::InvalidRequest(
                "System One prompt produced no tokens".into(),
            ));
        }
        let canvas_token_count = self.system_one_canvas_length()?;
        let (canvas, slots) =
            build_canvas(self, &questions, format, request.seed, canvas_token_count)?;
        let label_probabilities = self.system_one_read(&prompt_tokens, &canvas, &slots)?;
        let probabilities = questions
            .iter()
            .zip(label_probabilities)
            .map(|(question, distribution)| in_request_order(question, distribution))
            .collect::<Vec<_>>();
        check_output(request, &probabilities)?;
        Ok(DecisionOutput {
            probabilities,
            input_tokens: prompt_tokens.len(),
        })
    }
}

/// Maps a label distribution back to the question's request order. Only
/// choices differ: the canvas lists them sorted by name.
fn in_request_order(question: &CanvasQuestion, distribution: Vec<f32>) -> Vec<f32> {
    match &question.kind {
        CanvasKind::Choice { request_index, .. } => {
            let mut ordered = vec![0.0; distribution.len()];
            for (sorted, probability) in distribution.into_iter().enumerate() {
                ordered[request_index[sorted]] = probability;
            }
            ordered
        }
        _ => distribution,
    }
}

fn value_text(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::String(text) => text.trim().to_string(),
        _ => serde_json::to_string(value).unwrap_or_default(),
    }
}

fn optional_text(value: Option<&DecisionValue>) -> String {
    value
        .map(|value| value_text(&value.to_json()))
        .unwrap_or_default()
}

fn canvas_questions(questions: &[DecisionQuestion]) -> Result<Vec<CanvasQuestion>, DecisionError> {
    questions
        .iter()
        .enumerate()
        .map(|(index, question)| {
            let (labels, kind) = match &question.kind {
                DecisionQuestionKind::Noul {
                    when_true,
                    when_false,
                } => (
                    vec!["yes".to_string(), "no".to_string()],
                    CanvasKind::Noul {
                        true_description: optional_text(when_true.as_ref()),
                        false_description: optional_text(when_false.as_ref()),
                    },
                ),
                DecisionQuestionKind::Choice { options } => {
                    if options.len() > CHOICE_LABELS.len() {
                        return Err(DecisionError::InvalidRequest(format!(
                            "question {:?}: DiffusionGemma reads at most {} choices",
                            question.key,
                            CHOICE_LABELS.len()
                        )));
                    }
                    let mut request_index = (0..options.len()).collect::<Vec<_>>();
                    request_index.sort_by(|left, right| options[*left].0.cmp(&options[*right].0));
                    let choices = request_index
                        .iter()
                        .map(|index| {
                            let (name, description) = &options[*index];
                            (name.clone(), value_text(&description.to_json()))
                        })
                        .collect();
                    let labels = CHOICE_LABELS[..options.len()]
                        .iter()
                        .map(|label| char::from(*label).to_string())
                        .collect();
                    (
                        labels,
                        CanvasKind::Choice {
                            choices,
                            request_index,
                        },
                    )
                }
                DecisionQuestionKind::Score { levels } => (
                    (0..levels.len()).map(|index| index.to_string()).collect(),
                    CanvasKind::Score {
                        legend: levels
                            .iter()
                            .map(|level| value_text(&level.to_json()))
                            .collect(),
                    },
                ),
            };
            Ok(CanvasQuestion {
                key: question.key.clone(),
                id: format!("q{}", index + 1),
                instructions: optional_text(question.instructions.as_ref()),
                labels,
                kind,
            })
        })
        .collect()
}

fn system_text(questions: &[CanvasQuestion], format: AnswerFormat) -> String {
    let mut text = String::from(
        "Answer a fixed set of questions about the state the user provides. Each question lists its allowed answers; reply with exactly one label per question.\n",
    );
    for question in questions {
        let instruction = if question.instructions.is_empty() {
            "Answer about the state."
        } else {
            &question.instructions
        };
        text.push_str(&format!("\nQuestion {}: {instruction}\n", question.id));
        match &question.kind {
            CanvasKind::Noul {
                true_description,
                false_description,
            } => {
                for (label, description) in [
                    ("yes", true_description.as_str()),
                    ("no", false_description.as_str()),
                ] {
                    text.push_str(&format!("  {label}"));
                    if !description.is_empty() {
                        text.push_str(&format!(": {description}"));
                    }
                    text.push('\n');
                }
            }
            CanvasKind::Choice { choices, .. } => {
                for ((name, description), label) in choices.iter().zip(&question.labels) {
                    text.push_str(&format!("  {label}: {name}"));
                    if !description.is_empty() {
                        text.push_str(&format!(" ({description})"));
                    }
                    text.push('\n');
                }
            }
            CanvasKind::Score { legend } => {
                for (index, description) in legend.iter().enumerate() {
                    text.push_str(&format!("  {index}: {description}\n"));
                }
            }
        }
    }
    text.push('\n');
    text.push_str(match format {
        AnswerFormat::Lines => {
            "Reply with one line per question, in this order, formatted as \"id: label\"."
        }
        AnswerFormat::Indexed => {
            "Reply on one line with each question's id immediately followed by its label, separated by single spaces."
        }
    });
    text
}

fn answer_text(
    questions: &[CanvasQuestion],
    label_indexes: &[usize],
    format: AnswerFormat,
) -> String {
    questions
        .iter()
        .zip(label_indexes)
        .map(|(question, label_index)| match format {
            AnswerFormat::Lines => format!("{}: {}", question.id, question.labels[*label_index]),
            AnswerFormat::Indexed => format!("{}{}", question.id, question.labels[*label_index]),
        })
        .collect::<Vec<_>>()
        .join(match format {
            AnswerFormat::Lines => "\n",
            AnswerFormat::Indexed => " ",
        })
}

fn build_canvas(
    model: &StageModel,
    questions: &[CanvasQuestion],
    format: AnswerFormat,
    seed: [u8; 32],
    canvas_token_count: usize,
) -> Result<(Vec<i32>, Vec<SystemOneReadSlot>), DecisionError> {
    let scaffold = model.tokenize(SCAFFOLD, false)?;
    let base_indexes = vec![0usize; questions.len()];
    let mut canvas = scaffold.clone();
    canvas.extend(model.tokenize(&answer_text(questions, &base_indexes, format), false)?);
    if canvas.len() + 1 > canvas_token_count {
        return Err(DecisionError::InvalidRequest(format!(
            "System One answer template uses {} tokens; this PoC canvas holds {}",
            canvas.len() + 1,
            canvas_token_count
        )));
    }

    let mut slots = Vec::with_capacity(questions.len());
    for (question_index, question) in questions.iter().enumerate() {
        slots.push(answer_slot(
            model,
            questions,
            question_index,
            question,
            format,
            &scaffold,
            &canvas,
        )?);
    }

    canvas.push(TURN_CLOSE_TOKEN);
    canvas.resize(canvas_token_count, PAD_TOKEN);
    let mut rng = u64::from_be_bytes(seed[..8].try_into().expect("SHA-256 seed is eight bytes"));
    let filler = filler_tokens(slots.len(), &mut rng, |text| {
        Ok(model.tokenize(text, false)?)
    })?;
    for (slot, token) in slots.iter().zip(filler) {
        canvas[slot.canvas_position as usize] = token;
    }
    Ok((canvas, slots))
}

/// Finds the one canvas position where a question's labels differ and the
/// token each label puts there.
fn answer_slot(
    model: &StageModel,
    questions: &[CanvasQuestion],
    question_index: usize,
    question: &CanvasQuestion,
    format: AnswerFormat,
    scaffold: &[i32],
    canvas: &[i32],
) -> Result<SystemOneReadSlot, DecisionError> {
    let base_indexes = vec![0usize; questions.len()];
    let mut canvas_position = None;
    let mut label_token_ids = vec![0i32; question.labels.len()];
    for (label_index, label_token_id) in label_token_ids.iter_mut().enumerate().skip(1) {
        let mut indexes = base_indexes.clone();
        indexes[question_index] = label_index;
        let mut candidate = scaffold.to_vec();
        candidate.extend(model.tokenize(&answer_text(questions, &indexes, format), false)?);
        let differences = candidate
            .iter()
            .zip(canvas)
            .enumerate()
            .filter_map(|(index, (left, right))| (left != right).then_some(index))
            .collect::<Vec<_>>();
        if candidate.len() != canvas.len()
            || differences.len() != 1
            || canvas_position.is_some_and(|position| position != differences[0])
        {
            return Err(DecisionError::InvalidRequest(format!(
                "question {:?}: labels do not share one tokenizer slot",
                question.key
            )));
        }
        canvas_position = Some(differences[0]);
        *label_token_id = candidate[differences[0]];
    }
    let canvas_position = canvas_position.ok_or_else(|| {
        DecisionError::InvalidRequest(format!(
            "question {:?}: could not resolve its answer slot",
            question.key
        ))
    })?;
    label_token_ids[0] = canvas[canvas_position];
    Ok(SystemOneReadSlot {
        canvas_position: u32::try_from(canvas_position).unwrap_or(u32::MAX),
        label_token_ids,
    })
}

/// Draws one canvas filler token per randomized slot.
///
/// Every canvas slot outside the answer position must hold a token the loaded
/// vocabulary can decode, and the value must be reproducible from the request
/// seed so a leaked diffusion state still changes the read. Minting the fillers
/// from the model's own tokenizer satisfies both for any vocabulary size, so
/// the canvas never needs a fixed vocabulary bound.
fn filler_tokens(
    count: usize,
    rng: &mut u64,
    mut tokenize: impl FnMut(&str) -> Result<Vec<i32>, DecisionError>,
) -> Result<Vec<i32>, DecisionError> {
    let mut tokens = Vec::with_capacity(count);
    while tokens.len() < count {
        let mut word = String::with_capacity(FILLER_WORD_LEN);
        for _ in 0..FILLER_WORD_LEN {
            word.push(char::from(b'a' + (next_random(rng) % 26) as u8));
        }
        let mut drawn = tokenize(&word)?;
        if drawn.is_empty() {
            return Err(DecisionError::Backend(anyhow::anyhow!(
                "tokenizer produced no filler tokens for the System One canvas"
            )));
        }
        drawn.truncate(count - tokens.len());
        tokens.extend(drawn);
    }
    Ok(tokens)
}

/// One xorshift64 step. Shared by the canvas filler so its layout stays a pure
/// function of the request seed.
fn next_random(rng: &mut u64) -> u64 {
    *rng ^= *rng << 13;
    *rng ^= *rng >> 7;
    *rng ^= *rng << 17;
    *rng
}

#[cfg(test)]
mod tests {
    use super::*;

    fn choice(options: &[&str]) -> DecisionQuestion {
        DecisionQuestion {
            key: "team".into(),
            instructions: None,
            kind: DecisionQuestionKind::Choice {
                options: options
                    .iter()
                    .map(|name| (name.to_string(), DecisionValue::Null))
                    .collect(),
            },
        }
    }

    #[test]
    fn choices_are_lettered_in_sorted_order_and_mapped_back_to_request_order() {
        let questions = canvas_questions(&[choice(&["support", "billing", "sales"])]).unwrap();
        let CanvasKind::Choice { choices, .. } = &questions[0].kind else {
            unreachable!()
        };
        let names = choices
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["billing", "sales", "support"]);
        assert_eq!(questions[0].labels, vec!["A", "B", "C"]);
        // Canvas order is billing, sales, support; request order is support, billing, sales.
        assert_eq!(
            in_request_order(&questions[0], vec![0.5, 0.2, 0.3]),
            vec![0.3, 0.5, 0.2]
        );
    }

    #[test]
    fn system_text_matches_the_published_layout() {
        let questions = canvas_questions(&[DecisionQuestion {
            key: "billing".into(),
            instructions: Some(DecisionValue::String(" Is this billing? ".into())),
            kind: DecisionQuestionKind::Noul {
                when_true: Some(DecisionValue::String("a charge".into())),
                when_false: None,
            },
        }])
        .unwrap();
        assert_eq!(
            system_text(&questions, AnswerFormat::Lines),
            "Answer a fixed set of questions about the state the user provides. Each question lists its allowed answers; reply with exactly one label per question.\n\nQuestion q1: Is this billing?\n  yes: a charge\n  no\n\nReply with one line per question, in this order, formatted as \"id: label\"."
        );
    }

    #[test]
    fn too_many_choices_are_refused() {
        let names = (0..27)
            .map(|index| format!("o{index:02}"))
            .collect::<Vec<_>>();
        let names = names.iter().map(String::as_str).collect::<Vec<_>>();
        assert!(matches!(
            canvas_questions(&[choice(&names)]),
            Err(DecisionError::InvalidRequest(_))
        ));
    }

    /// A stand-in tokenizer whose token depends on every byte of the word,
    /// so the expected sequence below pins the xorshift walk itself.
    fn word_hash(text: &str) -> Result<Vec<i32>, DecisionError> {
        Ok(vec![
            text.bytes().fold(7_i32, |acc, byte| {
                acc.wrapping_mul(31).wrapping_add(i32::from(byte))
            }) & 0xffff,
        ])
    }

    #[test]
    fn filler_tokens_follow_the_seeded_sequence() {
        let mut rng = 0x0123_4567_89ab_cdef_u64;
        let tokens = filler_tokens(5, &mut rng, word_hash).expect("fillers");
        assert_eq!(tokens, vec![32609, 5868, 46964, 38019, 44864]);
        let mut again = 0x0123_4567_89ab_cdef_u64;
        assert_eq!(filler_tokens(5, &mut again, word_hash).unwrap(), tokens);
        let mut other = 0x0123_4567_89ab_cdee_u64;
        assert_ne!(filler_tokens(5, &mut other, word_hash).unwrap(), tokens);
    }

    #[test]
    fn filler_tokens_stop_at_the_requested_count() {
        let mut rng = 7_u64;
        let tokens = filler_tokens(3, &mut rng, |text| {
            Ok(text.chars().map(|character| character as i32).collect())
        })
        .expect("fillers");
        assert_eq!(tokens.len(), 3);
    }

    #[test]
    fn filler_tokens_reject_a_tokenizer_that_produces_nothing() {
        let mut rng = 11_u64;
        let error = filler_tokens(1, &mut rng, |_| Ok(Vec::new())).expect_err("no fillers");
        assert!(
            error.to_string().contains("no filler tokens"),
            "unexpected error: {error}"
        );
    }
}
