//! `POST /systemone`: Jev-compatible System One reads.
//!
//! The frontend owns the Jev contract — validation, aliases, and mapping
//! probabilities to typed answers. The model behind it is any
//! [`DecisionModel`]; how each one reads the questions lives in
//! `skippy_runtime::decision`.

use std::collections::BTreeMap;

use openai_frontend::{
    OpenAiError, OpenAiResult, SystemOneAnswer, SystemOneJson, SystemOneQuestion, SystemOneRequest,
    SystemOneResponse, SystemOneUsage,
};
use sha2::{Digest, Sha256};
use skippy_runtime::{
    DecisionError, DecisionModel, DecisionOutput, DecisionQuestion, DecisionQuestionKind,
    DecisionRequest, DecisionValue,
};

use crate::frontend::{OpenAiBackendMode, StageOpenAiBackend};

mod laya;

pub use laya::LayaSystemOneBackend;

/// Jev's bounds on options per question.
const MAX_CHOICES: usize = 26;
const MAX_SCORE_LEVELS: usize = 10;

impl StageOpenAiBackend {
    pub(super) fn run_system_one(
        &self,
        request: SystemOneRequest,
    ) -> OpenAiResult<SystemOneResponse> {
        self.validate_system_one_request(&request)?;
        let decision = decision_request(&request)?;
        let output = self
            .iteration_scheduler
            .execute_runtime("system-one-read", move |runtime| {
                runtime.model.decide(&decision).map_err(decision_error)
            })?;
        response(request, output)
    }

    fn validate_system_one_request(&self, request: &SystemOneRequest) -> OpenAiResult<()> {
        validate_request_fields(request, &self.model_id)?;
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

/// Runs a request on a model that holds no scheduler, such as Laya.
fn run_on_model(
    model: &dyn DecisionModel,
    model_id: &str,
    request: SystemOneRequest,
) -> OpenAiResult<SystemOneResponse> {
    validate_request_fields(&request, model_id)?;
    let decision = decision_request(&request)?;
    let output = model.decide(&decision).map_err(decision_error)?;
    response(request, output)
}

/// Checks the request fields every System One backend shares: the model name
/// or a Jev alias, a non-empty question set, and the one-read text-only subset.
fn validate_request_fields(request: &SystemOneRequest, model_id: &str) -> OpenAiResult<()> {
    const ALIASES: &[&str] = &["openjev-latest", "openjev-0.1", "jev-latest", "jev-preview"];
    if request.model != model_id && !ALIASES.contains(&request.model.as_str()) {
        return Err(OpenAiError::invalid_request(format!(
            "model {:?} is not loaded; use {:?} or openjev-latest",
            request.model, model_id
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
    Ok(())
}

/// Translates a Jev request into the model-neutral decision request, enforcing
/// Jev's option bounds. Questions keep the frontend's key order.
fn decision_request(request: &SystemOneRequest) -> OpenAiResult<DecisionRequest> {
    let canonical = serde_json::to_vec(&(&request.state, &request.questions)).map_err(|error| {
        OpenAiError::invalid_request(format!("serialize System One request: {error}"))
    })?;
    let questions = request
        .questions
        .iter()
        .map(|(key, question)| decision_question(key, question))
        .collect::<OpenAiResult<Vec<_>>>()?;
    Ok(DecisionRequest {
        state: decision_value(&request.state),
        questions,
        seed: Sha256::digest(canonical).into(),
    })
}

fn decision_question(key: &str, question: &SystemOneQuestion) -> OpenAiResult<DecisionQuestion> {
    let (instructions, kind) = match question {
        SystemOneQuestion::Noul {
            instructions,
            criteria,
        } => (
            instructions,
            DecisionQuestionKind::Noul {
                when_true: criteria
                    .as_ref()
                    .and_then(|criteria| criteria.r#true.as_ref())
                    .map(decision_value),
                when_false: criteria
                    .as_ref()
                    .and_then(|criteria| criteria.r#false.as_ref())
                    .map(decision_value),
            },
        ),
        SystemOneQuestion::Choice {
            instructions,
            criteria,
        } => {
            if !(2..=MAX_CHOICES).contains(&criteria.len()) {
                return Err(OpenAiError::invalid_request(format!(
                    "question {key:?}: choice criteria must contain 2 to {MAX_CHOICES} options"
                )));
            }
            (
                instructions,
                DecisionQuestionKind::Choice {
                    options: criteria
                        .iter()
                        .map(|(name, description)| (name.to_string(), decision_value(description)))
                        .collect(),
                },
            )
        }
        SystemOneQuestion::Score {
            instructions,
            criteria,
        } => {
            if !(2..=MAX_SCORE_LEVELS).contains(&criteria.len()) {
                return Err(OpenAiError::invalid_request(format!(
                    "question {key:?}: score criteria must contain 2 to {MAX_SCORE_LEVELS} levels"
                )));
            }
            (
                instructions,
                DecisionQuestionKind::Score {
                    levels: criteria.iter().map(decision_value).collect(),
                },
            )
        }
    };
    Ok(DecisionQuestion {
        key: key.to_string(),
        instructions: instructions.as_ref().map(decision_value),
        kind,
    })
}

fn decision_value(value: &SystemOneJson) -> DecisionValue {
    match value {
        SystemOneJson::Null => DecisionValue::Null,
        SystemOneJson::Bool(value) => DecisionValue::Bool(*value),
        SystemOneJson::Number(value) => DecisionValue::Number(value.clone()),
        SystemOneJson::String(value) => DecisionValue::String(value.clone()),
        SystemOneJson::Array(values) => {
            DecisionValue::Array(values.iter().map(decision_value).collect())
        }
        SystemOneJson::Object(object) => DecisionValue::Object(
            object
                .iter()
                .map(|(key, value)| (key.to_string(), decision_value(value)))
                .collect(),
        ),
    }
}

fn decision_error(error: DecisionError) -> OpenAiError {
    match error {
        DecisionError::InvalidRequest(message) => OpenAiError::invalid_request(message),
        DecisionError::Unsupported(message) => OpenAiError::unsupported(message),
        DecisionError::Backend(error) => OpenAiError::backend(format!("{error:#}")),
    }
}

fn response(request: SystemOneRequest, output: DecisionOutput) -> OpenAiResult<SystemOneResponse> {
    Ok(SystemOneResponse {
        answers: answers(&request.questions, &output.probabilities)?,
        model: request.model,
        usage: SystemOneUsage {
            input_tokens: u32::try_from(output.input_tokens).unwrap_or(u32::MAX),
            output_tokens: 0,
        },
    })
}

fn confidence(probabilities: &[f32]) -> f32 {
    let entropy = -probabilities
        .iter()
        .filter(|probability| **probability > 0.0)
        .map(|probability| probability * probability.ln())
        .sum::<f32>();
    (1.0 - entropy / (probabilities.len() as f32).ln()).clamp(0.0, 1.0)
}

/// Maps each question's distribution (in the order `DecisionQuestionKind`
/// documents) to its Jev answer.
fn answers(
    questions: &BTreeMap<String, SystemOneQuestion>,
    probabilities: &[Vec<f32>],
) -> OpenAiResult<BTreeMap<String, SystemOneAnswer>> {
    if questions.len() != probabilities.len() {
        return Err(OpenAiError::backend(
            "System One output did not match the question count",
        ));
    }
    questions
        .iter()
        .zip(probabilities)
        .map(|((key, question), probabilities)| {
            let answer = match question {
                SystemOneQuestion::Noul { .. } => SystemOneAnswer::Noul {
                    noul: probabilities[0],
                },
                SystemOneQuestion::Choice { criteria, .. } => {
                    let names = criteria.iter().map(|(name, _)| name).collect::<Vec<_>>();
                    let selected = probabilities
                        .iter()
                        .enumerate()
                        .max_by(|left, right| left.1.total_cmp(right.1))
                        .map(|(index, _)| index)
                        .unwrap_or(0);
                    SystemOneAnswer::Choice {
                        choice: names[selected].to_string(),
                        probabilities: names
                            .iter()
                            .zip(probabilities)
                            .map(|(name, probability)| (name.to_string(), *probability))
                            .collect(),
                        confidence: confidence(probabilities),
                    }
                }
                SystemOneQuestion::Score { criteria, .. } => SystemOneAnswer::Score {
                    score: probabilities
                        .iter()
                        .enumerate()
                        .map(|(index, probability)| index as f32 * probability)
                        .sum(),
                    legend: criteria
                        .iter()
                        .enumerate()
                        .map(|(index, level)| (index.to_string(), level.to_value()))
                        .collect(),
                    probabilities: probabilities
                        .iter()
                        .enumerate()
                        .map(|(index, probability)| (index.to_string(), *probability))
                        .collect(),
                    confidence: confidence(probabilities),
                },
            };
            Ok((key.clone(), answer))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(json: &str) -> SystemOneRequest {
        serde_json::from_str(json).expect("request")
    }

    #[test]
    fn confidence_is_zero_for_uniform_distribution() {
        assert!(confidence(&[0.5, 0.5]).abs() < f32::EPSILON);
    }

    #[test]
    fn confidence_is_one_for_certain_distribution() {
        assert!((confidence(&[1.0, 0.0]) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn decision_request_keeps_request_order_for_options_and_state() {
        let request = request(
            r#"{"model":"m","state":{"zeta":1,"alpha":2},"questions":{"team":{"type":"choice","criteria":{"support":"","billing":""}}}}"#,
        );
        let decision = decision_request(&request).unwrap();
        let DecisionValue::Object(state) = &decision.state else {
            panic!("object state")
        };
        assert_eq!(state[0].0, "zeta");
        let DecisionQuestionKind::Choice { options } = &decision.questions[0].kind else {
            panic!("choice")
        };
        assert_eq!(options[0].0, "support");
    }

    /// Nested objects in unsorted order, a float, criteria given as objects,
    /// and every question kind.
    const SEED_REQUEST: &str = r#"{"model":"m","state":{"zeta":{"b":[1,2.5,"x"],"a":null},"alpha":"hi"},"questions":{"team":{"type":"choice","instructions":{"y":1,"x":2},"criteria":{"support":"faults","billing":{"z":true,"a":false}}},"billing":{"type":"noul","instructions":"Billing?","criteria":{"true":"a charge","false":{"k":2,"j":1}}},"urgency":{"type":"score","criteria":["low",{"m":1,"l":0},"high"]}}}"#;

    #[test]
    fn the_seed_is_the_digest_diffusiongemma_has_always_used() {
        // DiffusionGemma's canvas filler derives from this seed, so it must
        // not move. Captured from origin/main (7d571a91d) by serializing
        // `(&request.state, &request.questions)` for SEED_REQUEST and taking
        // its SHA-256.
        let seed = decision_request(&request(SEED_REQUEST)).unwrap().seed;
        let hex = seed
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(
            hex,
            "3e7216c3eca8bf70ed5ac23016188815058b30d8155e1adcb745e70a5c8d98dd"
        );
    }

    #[test]
    fn jev_bounds_are_enforced_before_any_backend() {
        let one_choice = request(
            r#"{"model":"m","state":"s","questions":{"q":{"type":"choice","criteria":{"only":""}}}}"#,
        );
        assert!(decision_request(&one_choice).is_err());
    }

    #[test]
    fn answers_map_request_order_probabilities() {
        let request = request(
            r#"{"model":"m","state":"s","questions":{"billing":{"type":"noul"},"team":{"type":"choice","criteria":{"support":"","billing":""}}}}"#,
        );
        let answers = answers(&request.questions, &[vec![0.8, 0.2], vec![0.3, 0.7]]).unwrap();
        assert_eq!(answers["billing"], SystemOneAnswer::Noul { noul: 0.8 });
        let SystemOneAnswer::Choice {
            choice,
            probabilities,
            ..
        } = &answers["team"]
        else {
            panic!("choice answer")
        };
        assert_eq!(choice, "billing");
        assert_eq!(probabilities["support"], 0.3);
    }
}
