//! OpenAI Decisions wire contract with model-specific System One readouts.
//! <https://developers.openai.com/api/docs/guides/decisions>

use std::collections::{BTreeMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::{
    errors::InferenceError,
    system_one::{
        SystemOneAnswer, SystemOneJson, SystemOneJsonObject, SystemOneQuestion, SystemOneRequest,
        SystemOneResponse, SystemOneUsage,
    },
};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DecisionsRequest {
    model: String,
    input: DecisionInput,
    questions: Vec<DecisionsQuestion>,
    #[serde(default)]
    safety_identifier: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum DecisionInput {
    Text(String),
    Messages(Vec<DecisionInputMessage>),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DecisionInputMessage {
    role: String,
    #[serde(default, deserialize_with = "optional_string_without_null")]
    r#type: Option<String>,
    content: DecisionContent,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum DecisionContent {
    Text(String),
    Parts(Vec<DecisionInputPart>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum DecisionInputPart {
    InputText {
        text: String,
    },
    InputImage {
        image_url: String,
        detail: Option<String>,
    },
}

impl DecisionInput {
    fn text(&self) -> Result<String, InferenceError> {
        match self {
            Self::Text(text) => Ok(text.clone()),
            Self::Messages(messages) => {
                let mut text = Vec::new();
                for message in messages {
                    if message.role != "user"
                        || message
                            .r#type
                            .as_deref()
                            .is_some_and(|kind| kind != "message")
                    {
                        return Err(InferenceError::invalid_request(
                            "Decisions input supports only user messages",
                        ));
                    }
                    match &message.content {
                        DecisionContent::Text(content) => text.push(content.clone()),
                        DecisionContent::Parts(parts) => {
                            for part in parts {
                                match part {
                                    DecisionInputPart::InputText { text: part } => {
                                        text.push(part.clone())
                                    }
                                    DecisionInputPart::InputImage { image_url, detail } => {
                                        let _ = (image_url, detail);
                                        return Err(InferenceError::unsupported(
                                            "Decisions image input is not supported by the local System One backend",
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(text.join("\n\n"))
            }
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
enum DecisionsQuestion {
    Predicate {
        #[serde(default, deserialize_with = "optional_string_without_null")]
        name: Option<String>,
        instructions: String,
    },
    Choice {
        #[serde(default, deserialize_with = "optional_string_without_null")]
        name: Option<String>,
        instructions: String,
        choices: Vec<ChoiceOption>,
    },
    Score {
        #[serde(default, deserialize_with = "optional_string_without_null")]
        name: Option<String>,
        instructions: String,
        levels: Vec<ScoreLevel>,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, Hash)]
#[serde(untagged)]
enum ChoiceValue {
    String(String),
    Boolean(bool),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChoiceOption {
    value: ChoiceValue,
    #[serde(default, deserialize_with = "optional_string_without_null")]
    description: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ScoreLevel {
    label: String,
    #[serde(default, deserialize_with = "optional_string_without_null")]
    description: Option<String>,
}

fn optional_string_without_null<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    String::deserialize(deserializer).map(Some)
}

impl DecisionsQuestion {
    fn name(&self) -> Option<&str> {
        match self {
            Self::Predicate { name, .. } | Self::Choice { name, .. } | Self::Score { name, .. } => {
                name.as_deref()
            }
        }
    }

    fn to_system_one(&self) -> Result<SystemOneQuestion, InferenceError> {
        match self {
            Self::Predicate { instructions, .. } => Ok(SystemOneQuestion::Noul {
                instructions: Some(SystemOneJson::from(instructions.as_str())),
                criteria: None,
            }),
            Self::Choice { choices, .. } => {
                let keys = choice_keys(choices)?;
                Ok(SystemOneQuestion::Choice {
                    instructions: Some(SystemOneJson::from(self.instructions())),
                    criteria: SystemOneJsonObject::from_entries(choices.iter().zip(keys).map(
                        |(choice, key)| {
                            let description = choice.description.as_deref().unwrap_or_else(|| {
                                match &choice.value {
                                    ChoiceValue::String(value) => value,
                                    ChoiceValue::Boolean(true) => "true",
                                    ChoiceValue::Boolean(false) => "false",
                                }
                            });
                            (key, SystemOneJson::from(description))
                        },
                    )),
                })
            }
            Self::Score { levels, .. } => {
                let mut seen = HashSet::new();
                if levels.is_empty() || levels.iter().any(|level| !seen.insert(&level.label)) {
                    return Err(InferenceError::invalid_request(
                        "score levels must have distinct labels",
                    ));
                }
                Ok(SystemOneQuestion::Score {
                    instructions: Some(SystemOneJson::from(self.instructions())),
                    criteria: levels
                        .iter()
                        .map(|level| {
                            SystemOneJson::from(
                                level.description.as_deref().unwrap_or(&level.label),
                            )
                        })
                        .collect(),
                })
            }
        }
    }

    fn instructions(&self) -> &str {
        match self {
            Self::Predicate { instructions, .. }
            | Self::Choice { instructions, .. }
            | Self::Score { instructions, .. } => instructions,
        }
    }
}

fn choice_keys(choices: &[ChoiceOption]) -> Result<Vec<String>, InferenceError> {
    let mut values = HashSet::new();
    if choices.is_empty()
        || choices
            .iter()
            .any(|choice| !values.insert(choice.value.clone()))
    {
        return Err(InferenceError::invalid_request(
            "choices must have distinct values",
        ));
    }
    let strings = choices
        .iter()
        .filter_map(|choice| match &choice.value {
            ChoiceValue::String(value) => Some(value.as_str()),
            ChoiceValue::Boolean(_) => None,
        })
        .collect::<HashSet<_>>();
    Ok(choices
        .iter()
        .map(|choice| match &choice.value {
            ChoiceValue::String(value) => value.clone(),
            ChoiceValue::Boolean(value) => {
                let mut key = format!("__boolean_{value}");
                while strings.contains(key.as_str()) {
                    key.push('_');
                }
                key
            }
        })
        .collect())
}

impl DecisionsRequest {
    pub(crate) fn to_system_one(&self) -> Result<SystemOneRequest, InferenceError> {
        if self.model.trim().is_empty() || self.questions.is_empty() {
            return Err(InferenceError::invalid_request(
                "Decisions needs a model and at least one question",
            ));
        }
        if self
            .safety_identifier
            .as_ref()
            .is_some_and(|value| value.chars().count() > 128)
        {
            return Err(InferenceError::invalid_request(
                "safety_identifier exceeds 128 characters",
            ));
        }
        let mut questions = BTreeMap::new();
        for (index, question) in self.questions.iter().enumerate() {
            questions.insert(format!("question_{index}"), question.to_system_one()?);
        }
        Ok(SystemOneRequest {
            state: SystemOneJson::from(self.input.text()?.as_str()),
            model: self.model.clone(),
            questions,
            images: None,
            steps: None,
            samples: None,
            think: None,
            sequential: None,
        })
    }

    /// Map internal decision results and evidence back to the caller's question order.
    pub(crate) fn response(
        &self,
        result: SystemOneResponse,
    ) -> Result<DecisionsResponse, InferenceError> {
        let answers = self.questions.iter().enumerate().map(|(index, question)| {
            let key = format!("question_{index}");
            let answer = result.answers.get(&key).ok_or_else(|| InferenceError::backend(format!("Decisions backend omitted answer {key:?}")))?;
            let name = question.name().map(str::to_owned);
            match (question, answer) {
                (DecisionsQuestion::Predicate { .. }, SystemOneAnswer::Noul { noul }) =>
                    Ok(DecisionsAnswer::Predicate { name, probability: *noul }),
                (DecisionsQuestion::Choice { choices, .. }, SystemOneAnswer::Choice { choice, probabilities, confidence }) => {
                    let keys = choice_keys(choices)?;
                    let selected = keys.iter().position(|key| key == choice).ok_or_else(|| InferenceError::backend("Decisions backend returned unknown choice"))?;
                    let probabilities = choices.iter().zip(keys).map(|(option, key)| Ok(ChoiceProbability {
                        value: option.value.clone(),
                        probability: *probabilities.get(&key).ok_or_else(|| InferenceError::backend(format!("Decisions backend omitted probability for {key:?}")))?,
                    })).collect::<Result<Vec<_>, InferenceError>>()?;
                    Ok(DecisionsAnswer::Choice { name, choice: choices[selected].value.clone(), probabilities, confidence: *confidence })
                }
                (DecisionsQuestion::Score { levels, .. }, SystemOneAnswer::Score { score, probabilities, confidence, .. }) => {
                    let probabilities = levels.iter().enumerate().map(|(index, level)| Ok(ScoreProbability {
                        value: index, label: level.label.clone(),
                        probability: *probabilities.get(&index.to_string()).ok_or_else(|| InferenceError::backend(format!("Decisions backend omitted score probability {index}")))?,
                    })).collect::<Result<Vec<_>, InferenceError>>()?;
                    Ok(DecisionsAnswer::Score { name, score: *score, probabilities, confidence: *confidence })
                }
                _ => Err(InferenceError::backend(format!("Decisions backend returned wrong answer type for {key:?}"))),
            }
        }).collect::<Result<Vec<_>, InferenceError>>()?;
        Ok(DecisionsResponse {
            model: result.model,
            answers: answers
                .into_iter()
                .enumerate()
                .map(|(index, answer)| {
                    let key = format!("question_{index}");
                    let readout = result.readouts.get(&key).cloned();
                    if readout.is_none()
                        && (result.readout_usage.is_some() || !result.readouts.is_empty())
                    {
                        return Err(InferenceError::backend(format!(
                            "Decisions backend omitted readout {key:?}"
                        )));
                    }
                    Ok(DecisionsAnswerEntry { answer, readout })
                })
                .collect::<Result<Vec<_>, InferenceError>>()?,
            usage: DecisionsUsage::from(result.usage),
            readout_usage: result.readout_usage,
        })
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct DecisionsResponse {
    model: String,
    answers: Vec<DecisionsAnswerEntry>,
    usage: DecisionsUsage,
    #[serde(skip_serializing_if = "Option::is_none")]
    readout_usage: Option<crate::system_one::DecisionReadoutUsage>,
}

#[derive(Debug, Serialize)]
struct DecisionsAnswerEntry {
    #[serde(flatten)]
    answer: DecisionsAnswer,
    #[serde(skip_serializing_if = "Option::is_none")]
    readout: Option<crate::system_one::DecisionReadout>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum DecisionsAnswer {
    Predicate {
        name: Option<String>,
        probability: f32,
    },
    Choice {
        name: Option<String>,
        choice: ChoiceValue,
        probabilities: Vec<ChoiceProbability>,
        confidence: f32,
    },
    Score {
        name: Option<String>,
        score: f32,
        probabilities: Vec<ScoreProbability>,
        confidence: f32,
    },
}

#[derive(Debug, Serialize)]
struct ChoiceProbability {
    value: ChoiceValue,
    probability: f32,
}

#[derive(Debug, Serialize)]
struct ScoreProbability {
    value: usize,
    label: String,
    probability: f32,
}

#[derive(Debug, Serialize)]
struct DecisionsUsage {
    input_tokens: u32,
    input_tokens_details: InputTokensDetails,
    output_tokens: u32,
    output_tokens_details: OutputTokensDetails,
    total_tokens: u32,
}

#[derive(Debug, Serialize)]
struct InputTokensDetails {
    cached_tokens: u32,
    cache_write_tokens: u32,
}

#[derive(Debug, Serialize)]
struct OutputTokensDetails {
    reasoning_tokens: u32,
}

impl From<SystemOneUsage> for DecisionsUsage {
    /// Project backend usage without losing its measured prefix-cache reuse.
    fn from(usage: SystemOneUsage) -> Self {
        Self {
            input_tokens: usage.input_tokens,
            input_tokens_details: InputTokensDetails {
                cached_tokens: usage.cached_tokens,
                // Snapshot persistence is not token evaluation; no write-token
                // measurement is available from these adapters.
                cache_write_tokens: 0,
            },
            output_tokens: usage.output_tokens,
            output_tokens_details: OutputTokensDetails {
                reasoning_tokens: 0,
            },
            total_tokens: usage.input_tokens.saturating_add(usage.output_tokens),
        }
    }
}

#[cfg(test)]
mod readout_tests {
    use super::*;
    use crate::system_one::{DecisionReadout, DecisionReadoutUsage};

    /// Causal execution must report evidence for every answer, including an empty map.
    #[test]
    fn missing_causal_readout_is_a_backend_error() {
        for partial in [false, true] {
            let request: DecisionsRequest = serde_json::from_value(serde_json::json!({
                "model": "test", "input": "state", "questions": [
                    {"type": "predicate", "instructions": "first?"},
                    {"type": "predicate", "instructions": "second?"}
                ]
            }))
            .unwrap();
            let result = SystemOneResponse {
                model: "test".into(),
                answers: (0..2)
                    .map(|i| {
                        (
                            format!("question_{i}"),
                            SystemOneAnswer::Noul { noul: 0.75 },
                        )
                    })
                    .collect(),
                usage: SystemOneUsage {
                    input_tokens: 5,
                    output_tokens: 0,
                    cached_tokens: 0,
                },
                readout_usage: Some(DecisionReadoutUsage::default()),
                readouts: if partial {
                    [("question_0".into(), DecisionReadout {
                        method: "selected_token_logits", normalization: "sum_log_probabilities",
                        probability_semantics: "relative_candidate_probability_not_calibrated_truth",
                        candidate_log_probabilities: vec![-1.0, -2.0], candidate_token_counts: vec![1, 1],
                        evaluated_tokens: 5, cached_tokens: 0,
                    })].into()
                } else {
                    BTreeMap::new()
                },
            };
            let error = request.response(result).unwrap_err();
            assert_eq!(error.status(), axum::http::StatusCode::BAD_GATEWAY);
            let missing = usize::from(partial);
            assert!(
                error
                    .message()
                    .contains(&format!("omitted readout \"question_{missing}\""))
            );
        }
    }

    /// Answers and readout evidence retain question order even beyond ten questions.
    #[test]
    fn restores_original_question_order_for_answers_and_evidence() {
        let request: DecisionsRequest = serde_json::from_value(serde_json::json!({
            "model": "test", "input": "shared state", "questions": (0..12).map(|i| serde_json::json!({
                "type": "predicate", "name": format!("q{i}"), "instructions": "Is this relevant?"
            })).collect::<Vec<_>>()
        })).unwrap();
        let usage = DecisionReadoutUsage {
            evaluated_tokens: 120,
            cached_tokens: 60,
            cache_preparation_tokens: 10,
        };
        let result = SystemOneResponse {
            model: "test".into(),
            answers: (0..12).map(|i| (format!("question_{i}"), SystemOneAnswer::Noul { noul: i as f32 / 12.0 })).collect(),
            usage: SystemOneUsage { input_tokens: 180, output_tokens: 0, cached_tokens: 60 },
            readout_usage: Some(usage),
            readouts: (0..12).map(|i| (format!("question_{i}"), DecisionReadout {
                method: "selected_token_logits", normalization: "sum_log_probabilities", probability_semantics: "relative_candidate_probability_not_calibrated_truth",
                candidate_log_probabilities: vec![-1.0, -2.0], candidate_token_counts: vec![1, 1], evaluated_tokens: i, cached_tokens: 5,
            })).collect(),
        };
        let json = serde_json::to_value(request.response(result).unwrap()).unwrap();
        for i in 0..12 {
            assert_eq!(json["answers"][i]["name"], format!("q{i}"));
            assert_eq!(json["answers"][i]["readout"]["evaluated_tokens"], i);
        }
        assert_eq!(json["readout_usage"], serde_json::to_value(usage).unwrap());
        assert_eq!(json["usage"]["output_tokens"], 0);
        assert_eq!(json["usage"]["input_tokens_details"]["cached_tokens"], 60);
    }

    /// Legacy decision responses omit evidence that only causal execution can produce.
    #[test]
    fn legacy_adapter_responses_do_not_gain_causal_evidence() {
        let request: DecisionsRequest = serde_json::from_value(serde_json::json!({
            "model": "legacy", "input": "state", "questions": [{"type": "predicate", "instructions": "yes?"}]
        })).unwrap();
        let result = SystemOneResponse {
            model: "legacy".into(),
            answers: [("question_0".into(), SystemOneAnswer::Noul { noul: 0.75 })].into(),
            usage: SystemOneUsage {
                input_tokens: 5,
                output_tokens: 0,
                cached_tokens: 0,
            },
            readouts: BTreeMap::new(),
            readout_usage: None,
        };
        let json = serde_json::to_value(request.response(result).unwrap()).unwrap();
        assert!(json.get("readout_usage").is_none());
        assert!(json["answers"][0].get("readout").is_none());
        assert_eq!(json["answers"][0]["probability"], 0.75);
    }
}
