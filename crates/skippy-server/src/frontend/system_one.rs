use std::collections::BTreeMap;

use openai_frontend::{
    OpenAiError, OpenAiResult, SystemOneAnswer, SystemOneQuestion, SystemOneRequest,
    SystemOneResponse, SystemOneUsage,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use skippy_runtime::{ChatTemplateMessage, ChatTemplateOptions, SystemOneReadSlot};

use crate::frontend::{OpenAiBackendMode, StageOpenAiBackend, openai_backend_error};

const SYSTEM_ONE_TURN_CLOSE_TOKEN: i32 = 106;
const SYSTEM_ONE_PAD_TOKEN: i32 = 0;
const DIFFUSION_GEMMA_VOCAB_SIZE: u64 = 262_144;
const SYSTEM_ONE_SCAFFOLD: &str = "<|channel>thought\n<channel|>";

#[derive(Clone)]
struct PreparedQuestion {
    key: String,
    id: String,
    instructions: String,
    labels: Vec<String>,
    kind: PreparedQuestionKind,
}

#[derive(Clone)]
enum PreparedQuestionKind {
    Noul {
        true_description: String,
        false_description: String,
    },
    Choice {
        choices: Vec<(String, String)>,
    },
    Score {
        legend: Vec<Value>,
    },
}

impl StageOpenAiBackend {
    pub(super) fn run_system_one(
        &self,
        request: SystemOneRequest,
    ) -> OpenAiResult<SystemOneResponse> {
        self.validate_system_one_request(&request)?;
        let request_seed =
            serde_json::to_vec(&(&request.state, &request.questions)).map_err(|error| {
                OpenAiError::invalid_request(format!("serialize System One request: {error}"))
            })?;
        let questions = prepare_questions(request.questions)?;
        let format = if questions.len() <= 10 {
            AnswerFormat::Lines
        } else {
            AnswerFormat::Indexed
        };
        let system_text = system_text(&questions, format);
        let state_text = value_text(&request.state);
        let seed: [u8; 32] = Sha256::digest(request_seed).into();

        let read_questions = questions.clone();
        let outcome =
            self.iteration_scheduler
                .execute_runtime("system-one-read", move |runtime| {
                    let prompt = runtime
                        .model
                        .apply_chat_template_with_options(
                            &[
                                ChatTemplateMessage::new("system", &system_text),
                                ChatTemplateMessage::new("user", &state_text),
                            ],
                            ChatTemplateOptions {
                                add_assistant: true,
                                enable_thinking: Some(false),
                                ..ChatTemplateOptions::default()
                            },
                        )
                        .map_err(openai_backend_error)?;
                    let prompt_tokens = runtime
                        .model
                        .tokenize(&prompt, true)
                        .map_err(openai_backend_error)?;
                    if prompt_tokens.is_empty() {
                        return Err(OpenAiError::invalid_request(
                            "System One prompt produced no tokens",
                        ));
                    }
                    let canvas_token_count = runtime
                        .model
                        .system_one_canvas_length()
                        .map_err(openai_backend_error)?;
                    let (canvas, slots) = build_canvas(
                        &runtime.model,
                        &read_questions,
                        format,
                        seed,
                        canvas_token_count,
                    )?;
                    let probabilities = runtime
                        .model
                        .system_one_read(&prompt_tokens, &canvas, &slots)
                        .map_err(openai_backend_error)?;
                    Ok((prompt_tokens.len(), probabilities))
                })?;

        let input_tokens = u32::try_from(outcome.0).unwrap_or(u32::MAX);
        Ok(SystemOneResponse {
            model: request.model,
            answers: answers(&questions, &outcome.1)?,
            usage: SystemOneUsage {
                input_tokens,
                output_tokens: 0,
            },
        })
    }

    fn validate_system_one_request(&self, request: &SystemOneRequest) -> OpenAiResult<()> {
        const ALIASES: &[&str] = &["openjev-latest", "openjev-0.1", "jev-latest", "jev-preview"];
        if request.model != self.model_id && !ALIASES.contains(&request.model.as_str()) {
            return Err(OpenAiError::invalid_request(format!(
                "model {:?} is not loaded; use {:?} or openjev-latest",
                request.model, self.model_id
            )));
        }
        if request.questions.is_empty() {
            return Err(OpenAiError::invalid_request(
                "System One needs at least one question",
            ));
        }
        if request
            .images
            .as_ref()
            .is_some_and(|images| !images.is_empty())
            || request.steps.is_some_and(|steps| steps != 1)
            || request.samples.is_some_and(|samples| samples != 1)
            || request.think.is_some_and(|think| think != 0)
            || request.sequential.unwrap_or(false)
        {
            return Err(OpenAiError::unsupported(
                "this PoC supports one text-only System One read; images, multiple steps/samples, thinking, and sequential reads are not yet supported",
            ));
        }
        match &self.mode {
            OpenAiBackendMode::LocalRuntime => Ok(()),
            OpenAiBackendMode::EmbeddedStageZero { config, .. } if config.downstream.is_none() => {
                Ok(())
            }
            OpenAiBackendMode::EmbeddedStageZero { .. } => Err(OpenAiError::unsupported(
                "System One reads currently require a complete model on one Skippy worker",
            )),
        }
    }
}

fn value_text(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::String(text) => text.trim().to_string(),
        _ => serde_json::to_string(value).unwrap_or_default(),
    }
}

fn optional_value_text(value: Option<&Value>) -> String {
    value.map(value_text).unwrap_or_default()
}

fn prepare_questions(
    questions: BTreeMap<String, SystemOneQuestion>,
) -> OpenAiResult<Vec<PreparedQuestion>> {
    const CHOICE_LABELS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    questions
        .into_iter()
        .enumerate()
        .map(|(index, (key, question))| {
            let id = format!("q{}", index + 1);
            match question {
                SystemOneQuestion::Noul {
                    instructions,
                    criteria,
                } => {
                    let true_description = criteria
                        .as_ref()
                        .map(|criteria| optional_value_text(criteria.r#true.as_ref()))
                        .unwrap_or_default();
                    let false_description = criteria
                        .as_ref()
                        .map(|criteria| optional_value_text(criteria.r#false.as_ref()))
                        .unwrap_or_default();
                    Ok(PreparedQuestion {
                        key,
                        id,
                        instructions: optional_value_text(instructions.as_ref()),
                        labels: vec!["yes".to_string(), "no".to_string()],
                        kind: PreparedQuestionKind::Noul {
                            true_description,
                            false_description,
                        },
                    })
                }
                SystemOneQuestion::Choice {
                    instructions,
                    criteria,
                } => {
                    if !(2..=CHOICE_LABELS.len()).contains(&criteria.len()) {
                        return Err(OpenAiError::invalid_request(format!(
                            "question {key:?}: choice criteria must contain 2 to {} options",
                            CHOICE_LABELS.len()
                        )));
                    }
                    let choices = criteria
                        .into_iter()
                        .map(|(name, description)| (name, value_text(&description)))
                        .collect::<Vec<_>>();
                    let labels = CHOICE_LABELS[..choices.len()]
                        .iter()
                        .map(|label| char::from(*label).to_string())
                        .collect();
                    Ok(PreparedQuestion {
                        key,
                        id,
                        instructions: optional_value_text(instructions.as_ref()),
                        labels,
                        kind: PreparedQuestionKind::Choice { choices },
                    })
                }
                SystemOneQuestion::Score {
                    instructions,
                    criteria,
                } => {
                    if !(2..=10).contains(&criteria.len()) {
                        return Err(OpenAiError::invalid_request(format!(
                            "question {key:?}: score criteria must contain 2 to 10 levels"
                        )));
                    }
                    let labels = (0..criteria.len()).map(|index| index.to_string()).collect();
                    Ok(PreparedQuestion {
                        key,
                        id,
                        instructions: optional_value_text(instructions.as_ref()),
                        labels,
                        kind: PreparedQuestionKind::Score { legend: criteria },
                    })
                }
            }
        })
        .collect()
}

#[derive(Clone, Copy)]
enum AnswerFormat {
    Lines,
    Indexed,
}

fn system_text(questions: &[PreparedQuestion], format: AnswerFormat) -> String {
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
            PreparedQuestionKind::Noul {
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
            PreparedQuestionKind::Choice { choices } => {
                for ((name, description), label) in choices.iter().zip(&question.labels) {
                    text.push_str(&format!("  {label}: {name}"));
                    if !description.is_empty() {
                        text.push_str(&format!(" ({description})"));
                    }
                    text.push('\n');
                }
            }
            PreparedQuestionKind::Score { legend } => {
                for (index, description) in legend.iter().enumerate() {
                    text.push_str(&format!("  {index}: {}\n", value_text(description)));
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
    questions: &[PreparedQuestion],
    label_indexes: &[usize],
    format: AnswerFormat,
) -> String {
    questions
        .iter()
        .zip(label_indexes)
        .map(|(question, label_index)| match format {
            AnswerFormat::Lines => {
                format!("{}: {}", question.id, question.labels[*label_index])
            }
            AnswerFormat::Indexed => {
                format!("{}{}", question.id, question.labels[*label_index])
            }
        })
        .collect::<Vec<_>>()
        .join(match format {
            AnswerFormat::Lines => "\n",
            AnswerFormat::Indexed => " ",
        })
}

fn build_canvas(
    model: &skippy_runtime::StageModel,
    questions: &[PreparedQuestion],
    format: AnswerFormat,
    seed: [u8; 32],
    canvas_token_count: usize,
) -> OpenAiResult<(Vec<i32>, Vec<SystemOneReadSlot>)> {
    let scaffold = model
        .tokenize(SYSTEM_ONE_SCAFFOLD, false)
        .map_err(openai_backend_error)?;
    let base_indexes = vec![0usize; questions.len()];
    let mut canvas = scaffold.clone();
    canvas.extend(
        model
            .tokenize(&answer_text(questions, &base_indexes, format), false)
            .map_err(openai_backend_error)?,
    );
    if canvas.len() + 1 > canvas_token_count {
        return Err(OpenAiError::invalid_request(format!(
            "System One answer template uses {} tokens; this PoC canvas holds {}",
            canvas.len() + 1,
            canvas_token_count
        )));
    }

    let mut slots = Vec::with_capacity(questions.len());
    for (question_index, question) in questions.iter().enumerate() {
        let mut canvas_position = None;
        let mut label_token_ids = vec![0i32; question.labels.len()];
        for (label_index, label_token_id) in label_token_ids.iter_mut().enumerate().skip(1) {
            let mut indexes = base_indexes.clone();
            indexes[question_index] = label_index;
            let mut candidate = scaffold.clone();
            candidate.extend(
                model
                    .tokenize(&answer_text(questions, &indexes, format), false)
                    .map_err(openai_backend_error)?,
            );
            let differences = candidate
                .iter()
                .zip(&canvas)
                .enumerate()
                .filter_map(|(index, (left, right))| (left != right).then_some(index))
                .collect::<Vec<_>>();
            if candidate.len() != canvas.len()
                || differences.len() != 1
                || canvas_position.is_some_and(|position| position != differences[0])
            {
                return Err(OpenAiError::invalid_request(format!(
                    "question {:?}: labels do not share one tokenizer slot",
                    question.key
                )));
            }
            canvas_position = Some(differences[0]);
            *label_token_id = candidate[differences[0]];
        }
        let canvas_position = canvas_position.ok_or_else(|| {
            OpenAiError::invalid_request(format!(
                "question {:?}: could not resolve its answer slot",
                question.key
            ))
        })?;
        label_token_ids[0] = canvas[canvas_position];
        slots.push(SystemOneReadSlot {
            canvas_position: u32::try_from(canvas_position).unwrap_or(u32::MAX),
            label_token_ids,
        });
    }

    canvas.push(SYSTEM_ONE_TURN_CLOSE_TOKEN);
    canvas.resize(canvas_token_count, SYSTEM_ONE_PAD_TOKEN);
    let mut rng = u64::from_be_bytes(seed[..8].try_into().expect("SHA-256 seed is eight bytes"));
    for slot in &slots {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        canvas[slot.canvas_position as usize] = (rng % DIFFUSION_GEMMA_VOCAB_SIZE) as i32;
    }
    Ok((canvas, slots))
}

fn confidence(probabilities: &[f32]) -> f32 {
    let entropy = -probabilities
        .iter()
        .filter(|probability| **probability > 0.0)
        .map(|probability| probability * probability.ln())
        .sum::<f32>();
    (1.0 - entropy / (probabilities.len() as f32).ln()).clamp(0.0, 1.0)
}

fn answers(
    questions: &[PreparedQuestion],
    probabilities: &[Vec<f32>],
) -> OpenAiResult<BTreeMap<String, SystemOneAnswer>> {
    if questions.len() != probabilities.len() {
        return Err(OpenAiError::backend(
            "native System One output did not match the question count",
        ));
    }
    questions
        .iter()
        .zip(probabilities)
        .map(|(question, probabilities)| {
            if probabilities.len() != question.labels.len() {
                return Err(OpenAiError::backend(format!(
                    "native System One output for {:?} did not match its label count",
                    question.key
                )));
            }
            let answer = match &question.kind {
                PreparedQuestionKind::Noul { .. } => SystemOneAnswer::Noul {
                    noul: probabilities[0],
                },
                PreparedQuestionKind::Choice { choices } => {
                    let selected = probabilities
                        .iter()
                        .enumerate()
                        .max_by(|left, right| left.1.total_cmp(right.1))
                        .map(|(index, _)| index)
                        .unwrap_or(0);
                    SystemOneAnswer::Choice {
                        choice: choices[selected].0.clone(),
                        probabilities: choices
                            .iter()
                            .zip(probabilities)
                            .map(|((name, _), probability)| (name.clone(), *probability))
                            .collect(),
                        confidence: confidence(probabilities),
                    }
                }
                PreparedQuestionKind::Score { legend } => SystemOneAnswer::Score {
                    score: probabilities
                        .iter()
                        .enumerate()
                        .map(|(index, probability)| index as f32 * probability)
                        .sum(),
                    legend: legend
                        .iter()
                        .enumerate()
                        .map(|(index, value)| (index.to_string(), value.clone()))
                        .collect(),
                    probabilities: probabilities
                        .iter()
                        .enumerate()
                        .map(|(index, probability)| (index.to_string(), *probability))
                        .collect(),
                    confidence: confidence(probabilities),
                },
            };
            Ok((question.key.clone(), answer))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confidence_is_zero_for_uniform_distribution() {
        assert!(confidence(&[0.5, 0.5]).abs() < f32::EPSILON);
    }

    #[test]
    fn confidence_is_one_for_certain_distribution() {
        assert!((confidence(&[1.0, 0.0]) - 1.0).abs() < f32::EPSILON);
    }
}
