//! Preview OpenAI Decisions wire shape, translated to the existing System One backend.
//! Based on the recorded POST /v1/decisions exchange in RubyLLM PR #1008.

use std::collections::{BTreeMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::{
    errors::OpenAiError,
    system_one::{
        SystemOneAnswer, SystemOneJson, SystemOneJsonObject, SystemOneQuestion, SystemOneRequest,
        SystemOneResponse, SystemOneUsage,
    },
};

#[derive(Debug, Deserialize)]
pub(crate) struct DecisionsRequest {
    model: String,
    input: String,
    questions: Vec<DecisionsQuestion>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum DecisionsQuestion {
    Predicate {
        name: String,
        instructions: Option<String>,
    },
    Choice {
        name: String,
        instructions: Option<String>,
        choices: Vec<ChoiceOption>,
    },
    Score {
        name: String,
        instructions: Option<String>,
        levels: Vec<ScoreLevel>,
    },
}

#[derive(Debug, Deserialize)]
struct ChoiceOption {
    value: String,
    description: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ScoreLevel {
    label: String,
    description: Option<String>,
}

impl DecisionsQuestion {
    fn name(&self) -> &str {
        match self {
            Self::Predicate { name, .. } | Self::Choice { name, .. } | Self::Score { name, .. } => {
                name
            }
        }
    }

    fn to_system_one(&self) -> Result<SystemOneQuestion, OpenAiError> {
        let instruction = |value: &Option<String>| value.as_deref().map(SystemOneJson::from);
        match self {
            Self::Predicate { instructions, .. } => Ok(SystemOneQuestion::Noul {
                instructions: instruction(instructions),
                criteria: None,
            }),
            Self::Choice {
                name,
                instructions,
                choices,
            } => {
                let mut seen = HashSet::new();
                if choices.is_empty()
                    || choices
                        .iter()
                        .any(|choice| choice.value.is_empty() || !seen.insert(&choice.value))
                {
                    return Err(OpenAiError::invalid_request(format!(
                        "question {name:?} must have distinct, nonempty choice values"
                    )));
                }
                Ok(SystemOneQuestion::Choice {
                    instructions: instruction(instructions),
                    criteria: SystemOneJsonObject::from_entries(choices.iter().map(|choice| {
                        (
                            choice.value.clone(),
                            SystemOneJson::from(choice.description.as_deref().unwrap_or("")),
                        )
                    })),
                })
            }
            Self::Score {
                name,
                instructions,
                levels,
            } => {
                let mut seen = HashSet::new();
                if levels.is_empty()
                    || levels
                        .iter()
                        .any(|level| level.label.is_empty() || !seen.insert(&level.label))
                {
                    return Err(OpenAiError::invalid_request(format!(
                        "question {name:?} must have distinct, nonempty level labels"
                    )));
                }
                Ok(SystemOneQuestion::Score {
                    instructions: instruction(instructions),
                    criteria: levels
                        .iter()
                        .map(|level| {
                            SystemOneJson::from(level.description.as_deref().unwrap_or(""))
                        })
                        .collect(),
                })
            }
        }
    }
}

impl DecisionsRequest {
    pub(crate) fn to_system_one(&self) -> Result<SystemOneRequest, OpenAiError> {
        if self.model.trim().is_empty() || self.questions.is_empty() {
            return Err(OpenAiError::invalid_request(
                "Decisions needs a model and at least one question",
            ));
        }
        let mut questions = BTreeMap::new();
        for question in &self.questions {
            let name = question.name();
            if name.is_empty()
                || questions
                    .insert(name.to_string(), question.to_system_one()?)
                    .is_some()
            {
                return Err(OpenAiError::invalid_request(
                    "Decisions question names must be distinct and nonempty",
                ));
            }
        }
        Ok(SystemOneRequest {
            state: SystemOneJson::from(self.input.as_str()),
            model: self.model.clone(),
            questions,
            images: None,
            steps: None,
            samples: None,
            think: None,
            sequential: None,
        })
    }

    pub(crate) fn response(
        &self,
        result: SystemOneResponse,
    ) -> Result<DecisionsResponse, OpenAiError> {
        let answers = self.questions.iter().map(|question| {
            let name = question.name();
            let answer = result.answers.get(name).ok_or_else(|| OpenAiError::backend(format!("Decisions backend omitted answer {name:?}")))?;
            match (question, answer) {
                (DecisionsQuestion::Predicate { .. }, SystemOneAnswer::Noul { noul }) =>
                    Ok(DecisionsAnswer::Predicate { name: name.to_string(), probability: *noul }),
                (DecisionsQuestion::Choice { choices, .. }, SystemOneAnswer::Choice { choice, probabilities, confidence }) => {
                    if !choices.iter().any(|option| option.value == *choice) {
                        return Err(OpenAiError::backend(format!("Decisions backend returned unknown choice for {name:?}")));
                    }
                    let probabilities = choices.iter().map(|option| Ok(ChoiceProbability {
                        value: option.value.clone(),
                        probability: *probabilities.get(&option.value).ok_or_else(|| OpenAiError::backend(format!("Decisions backend omitted probability for {:?}", option.value)))?,
                    })).collect::<Result<Vec<_>, OpenAiError>>()?;
                    Ok(DecisionsAnswer::Choice { name: name.to_string(), choice: choice.clone(), probabilities, confidence: *confidence })
                }
                (DecisionsQuestion::Score { levels, .. }, SystemOneAnswer::Score { score, probabilities, confidence, .. }) => {
                    let probabilities = levels.iter().enumerate().map(|(index, level)| Ok(ScoreProbability {
                        value: index,
                        label: level.label.clone(),
                        probability: *probabilities.get(&index.to_string()).ok_or_else(|| OpenAiError::backend(format!("Decisions backend omitted score probability {index}")))?,
                    })).collect::<Result<Vec<_>, OpenAiError>>()?;
                    Ok(DecisionsAnswer::Score { name: name.to_string(), score: *score, probabilities, confidence: *confidence })
                }
                _ => Err(OpenAiError::backend(format!("Decisions backend returned wrong answer type for {name:?}"))),
            }
        }).collect::<Result<Vec<_>, OpenAiError>>()?;
        Ok(DecisionsResponse {
            model: result.model,
            answers,
            usage: DecisionsUsage::from(result.usage),
        })
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct DecisionsResponse {
    model: String,
    answers: Vec<DecisionsAnswer>,
    usage: DecisionsUsage,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum DecisionsAnswer {
    Predicate {
        name: String,
        probability: f32,
    },
    Choice {
        name: String,
        choice: String,
        probabilities: Vec<ChoiceProbability>,
        confidence: f32,
    },
    Score {
        name: String,
        score: f32,
        probabilities: Vec<ScoreProbability>,
        confidence: f32,
    },
}

#[derive(Debug, Serialize)]
struct ChoiceProbability {
    value: String,
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
    output_tokens: u32,
    total_tokens: u32,
}

impl From<SystemOneUsage> for DecisionsUsage {
    fn from(usage: SystemOneUsage) -> Self {
        Self {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            total_tokens: usage.input_tokens.saturating_add(usage.output_tokens),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_preserves_choice_order_and_text_input() {
        let request: DecisionsRequest = serde_json::from_str(
            r#"{"model":"laya","input":"route this","questions":[{"type":"choice","name":"team","choices":[{"value":"zeta","description":"Last"},{"value":"alpha","description":"First"}]}]}"#,
        )
        .unwrap();
        let converted = request.to_system_one().unwrap();
        assert_eq!(converted.state.as_str(), Some("route this"));
        let SystemOneQuestion::Choice { criteria, .. } = &converted.questions["team"] else {
            panic!("expected choice");
        };
        assert_eq!(
            criteria.iter().map(|(name, _)| name).collect::<Vec<_>>(),
            vec!["zeta", "alpha"]
        );
    }
}
