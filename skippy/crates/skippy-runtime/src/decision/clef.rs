//! System One reads on a Clef decision model.
//!
//! Clef answers every question of a request in one pass: the native read
//! renders the GGUF's `systemone` template over the state and the questions,
//! marks each question and option token with its decision order, and returns
//! one head score per option in prompt order. This adapter builds that request
//! the way the upstream server does — `true` before `false` for noul, choice
//! options sorted by key, score levels numbered from `0` — then applies the
//! GGUF's calibration (`<arch>.decision.temperature.*`, `score_bias.*`) and
//! hands the distributions back in request order.

use serde_json::{Value, json};

use super::{
    DecisionError, DecisionModel, DecisionOutput, DecisionQuestion, DecisionQuestionKind,
    DecisionRequest, check_output,
};
use crate::clef::ClefModel;

impl DecisionModel for ClefModel {
    fn decide(&self, request: &DecisionRequest) -> Result<DecisionOutput, DecisionError> {
        let questions = request
            .questions
            .iter()
            .map(prompt_question)
            .collect::<Result<Vec<_>, _>>()?;
        let option_count = questions.iter().map(|q| q.option_keys.len()).sum::<usize>();
        let body = json!({
            "state": request.state.to_json(),
            "questions": questions.iter().map(|q| q.json.clone()).collect::<Vec<_>>(),
        })
        .to_string();

        let (tokens, _orders) = self.plan(&body).map_err(invalid_or_backend)?;
        let scores = self.read(&body, option_count).map_err(invalid_or_backend)?;

        let calibration = Calibration::from_metadata(&self.info().decision_metadata);
        let mut offset = 0;
        let mut probabilities = Vec::with_capacity(questions.len());
        for (question, prompt) in request.questions.iter().zip(&questions) {
            let n = prompt.option_keys.len();
            let logits = &scores[offset..offset + n];
            offset += n;
            if logits.iter().any(|score| !score.is_finite()) {
                return Err(DecisionError::Backend(anyhow::anyhow!(
                    "question {:?}: the Clef head returned no score for an option",
                    question.key
                )));
            }
            probabilities.push(request_order(
                question,
                prompt,
                &calibration.distribution(question, logits),
            ));
        }
        check_output(request, &probabilities)?;
        Ok(DecisionOutput {
            probabilities,
            input_tokens: tokens.len(),
        })
    }
}

/// One question as the native read receives it, with its options in prompt
/// order so the scores can be mapped back to the request's order.
struct PromptQuestion {
    json: Value,
    /// Option keys in prompt order.
    option_keys: Vec<String>,
    /// For each prompt option, its index in the request's option order.
    request_index: Vec<usize>,
}

fn prompt_question(question: &DecisionQuestion) -> Result<PromptQuestion, DecisionError> {
    let Some(instructions) = question.instructions.as_ref() else {
        return Err(DecisionError::InvalidRequest(format!(
            "question {:?}: Clef needs instructions",
            question.key
        )));
    };
    // (key, description, request index) in prompt order
    let options: Vec<(String, Value, usize)> = match &question.kind {
        DecisionQuestionKind::Noul {
            when_true,
            when_false,
        } => vec![
            ("true".to_string(), optional_json(when_true.as_ref()), 0),
            ("false".to_string(), optional_json(when_false.as_ref()), 1),
        ],
        DecisionQuestionKind::Choice { options } => {
            if options.len() < 2 {
                return Err(DecisionError::InvalidRequest(format!(
                    "question {:?}: a choice needs at least 2 options",
                    question.key
                )));
            }
            let mut sorted = options
                .iter()
                .enumerate()
                .map(|(index, (key, description))| (key.clone(), description.to_json(), index))
                .collect::<Vec<_>>();
            sorted.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
            sorted
        }
        DecisionQuestionKind::Score { levels } => {
            if !(2..=10).contains(&levels.len()) {
                return Err(DecisionError::InvalidRequest(format!(
                    "question {:?}: a score needs 2 to 10 levels",
                    question.key
                )));
            }
            levels
                .iter()
                .enumerate()
                .map(|(index, level)| (index.to_string(), level.to_json(), index))
                .collect()
        }
    };
    let json = json!({
        "id": question.key,
        "type": type_name(question),
        "instructions": instructions.to_json(),
        "options": options
            .iter()
            .map(|(key, description, _)| json!({"key": key, "description": description}))
            .collect::<Vec<_>>(),
    });
    Ok(PromptQuestion {
        json,
        option_keys: options.iter().map(|(key, _, _)| key.clone()).collect(),
        request_index: options.iter().map(|(_, _, index)| *index).collect(),
    })
}

fn optional_json(value: Option<&super::DecisionValue>) -> Value {
    value.map_or(Value::Null, super::DecisionValue::to_json)
}

fn type_name(question: &DecisionQuestion) -> &'static str {
    match question.kind {
        DecisionQuestionKind::Noul { .. } => "noul",
        DecisionQuestionKind::Choice { .. } => "choice",
        DecisionQuestionKind::Score { .. } => "score",
    }
}

/// Reorders a prompt-order distribution into the request's option order.
fn request_order(question: &DecisionQuestion, prompt: &PromptQuestion, probs: &[f32]) -> Vec<f32> {
    let mut ordered = vec![0f32; question.option_count()];
    for (prob, &index) in probs.iter().zip(&prompt.request_index) {
        ordered[index] = *prob;
    }
    ordered
}

/// `<arch>.decision.*` calibration of a Clef GGUF, as the upstream server
/// applies it: a temperature per question type (optionally per option-count
/// bucket) and an additive bias per score level for a given level count.
#[derive(Debug, Default, PartialEq)]
struct Calibration {
    /// (`<type>` or `<type>.<bucket>`, temperature)
    temperatures: Vec<(String, f32)>,
    /// (level count, biases by level)
    score_bias: Vec<(usize, Vec<f32>)>,
}

impl Calibration {
    fn from_metadata(metadata: &[(String, String)]) -> Self {
        let mut calibration = Self::default();
        for (key, value) in metadata {
            if let Some(name) = key.strip_prefix("temperature.") {
                if let Ok(temperature) = value.trim().parse::<f32>()
                    && temperature > 0.0
                {
                    calibration
                        .temperatures
                        .push((name.to_string(), temperature));
                }
            } else if let Some(rest) = key.strip_prefix("score_bias.") {
                let mut parts = rest.split('.');
                let (Some(Ok(count)), Some(Ok(level))) = (
                    parts.next().map(str::parse::<usize>),
                    parts.next().map(str::parse::<usize>),
                ) else {
                    continue;
                };
                let Ok(bias) = value.trim().parse::<f32>() else {
                    continue;
                };
                if level >= count {
                    continue;
                }
                let entry = match calibration.score_bias.iter_mut().find(|(n, _)| *n == count) {
                    Some(entry) => entry,
                    None => {
                        calibration.score_bias.push((count, vec![0.0; count]));
                        calibration.score_bias.last_mut().expect("just pushed")
                    }
                };
                entry.1[level] = bias;
            }
        }
        calibration
    }

    fn temperature(&self, question: &DecisionQuestion) -> f32 {
        let n = question.option_count();
        let bucket = if n <= 2 {
            "2"
        } else if n <= 5 {
            "3_5"
        } else if n <= 10 {
            "6_10"
        } else {
            "11"
        };
        let type_name = type_name(question);
        [format!("{type_name}.{bucket}"), type_name.to_string()]
            .iter()
            .find_map(|name| {
                self.temperatures
                    .iter()
                    .find(|(key, _)| key == name)
                    .map(|(_, temperature)| *temperature)
            })
            .unwrap_or(1.0)
    }

    /// Softmax of the head scores in prompt order, with this question's
    /// temperature and, for a score, its level biases.
    fn distribution(&self, question: &DecisionQuestion, logits: &[f32]) -> Vec<f32> {
        let temperature = self.temperature(question);
        let mut logits = logits.to_vec();
        if matches!(question.kind, DecisionQuestionKind::Score { .. })
            && let Some((_, bias)) = self.score_bias.iter().find(|(n, _)| *n == logits.len())
        {
            for (logit, bias) in logits.iter_mut().zip(bias) {
                *logit += bias;
            }
        }
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps = logits
            .iter()
            .map(|logit| ((logit - max) / temperature).exp())
            .collect::<Vec<_>>();
        let sum = exps.iter().sum::<f32>();
        exps.into_iter().map(|value| value / sum).collect()
    }
}

fn invalid_or_backend(error: anyhow::Error) -> DecisionError {
    // The native read rejects malformed requests with an invalid-argument
    // status that the FFI layer reports as an error message.
    let message = format!("{error:#}");
    if message.contains("invalid argument") || message.contains("clef request") {
        DecisionError::InvalidRequest(message)
    } else {
        DecisionError::Backend(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision::DecisionValue;

    fn text(value: &str) -> DecisionValue {
        DecisionValue::String(value.into())
    }

    fn choice(options: &[&str]) -> DecisionQuestion {
        DecisionQuestion {
            key: "route".into(),
            instructions: Some(text("Where does this go?")),
            kind: DecisionQuestionKind::Choice {
                options: options
                    .iter()
                    .map(|name| (name.to_string(), DecisionValue::Null))
                    .collect(),
            },
        }
    }

    #[test]
    fn choice_options_are_sorted_in_the_prompt_and_restored_for_the_answer() {
        let question = choice(&["sales", "billing", "tech"]);
        let prompt = prompt_question(&question).unwrap();
        assert_eq!(prompt.option_keys, ["billing", "sales", "tech"]);
        assert_eq!(prompt.request_index, [1, 0, 2]);
        let ordered = request_order(&question, &prompt, &[0.7, 0.2, 0.1]);
        assert_eq!(ordered, [0.2, 0.7, 0.1]);
    }

    #[test]
    fn noul_lists_true_first_and_score_numbers_levels() {
        let noul = DecisionQuestion {
            key: "angry".into(),
            instructions: Some(text("Is the user angry?")),
            kind: DecisionQuestionKind::Noul {
                when_true: Some(text("yes")),
                when_false: None,
            },
        };
        let prompt = prompt_question(&noul).unwrap();
        assert_eq!(prompt.option_keys, ["true", "false"]);
        assert_eq!(prompt.json["options"][1]["description"], Value::Null);

        let score = DecisionQuestion {
            key: "urgency".into(),
            instructions: Some(text("How urgent?")),
            kind: DecisionQuestionKind::Score {
                levels: vec![text("low"), text("mid"), text("high")],
            },
        };
        let prompt = prompt_question(&score).unwrap();
        assert_eq!(prompt.option_keys, ["0", "1", "2"]);
        assert_eq!(prompt.json["type"], "score");
    }

    #[test]
    fn missing_instructions_is_an_invalid_request() {
        let mut question = choice(&["a", "b"]);
        question.instructions = None;
        assert!(matches!(
            prompt_question(&question),
            Err(DecisionError::InvalidRequest(_))
        ));
    }

    #[test]
    fn calibration_reads_temperatures_and_score_bias() {
        let calibration = Calibration::from_metadata(&[
            ("type".into(), "clef".into()),
            ("temperature.choice.3_5".into(), "0.5".into()),
            ("temperature.score".into(), "2".into()),
            ("score_bias.3.0".into(), "1".into()),
            ("score_bias.3.2".into(), "-1".into()),
        ]);
        assert_eq!(calibration.temperature(&choice(&["a", "b", "c"])), 0.5);
        assert_eq!(calibration.temperature(&choice(&["a", "b"])), 1.0);
        let score = DecisionQuestion {
            key: "s".into(),
            instructions: Some(text("?")),
            kind: DecisionQuestionKind::Score {
                levels: vec![text("0"), text("1"), text("2")],
            },
        };
        assert_eq!(calibration.temperature(&score), 2.0);
        let probs = calibration.distribution(&score, &[0.0, 0.0, 0.0]);
        // biases [1, 0, -1] at temperature 2 → logits 0.5, 0, -0.5
        assert!(probs[0] > probs[1] && probs[1] > probs[2]);
        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    /// Reads the reference request against a real Clef GGUF named by
    /// `SKIPPY_CLEF_MODEL`, with the runtime bundle from
    /// `MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR`. The expected probabilities are
    /// what the upstream `llama-server /v1/systemone` answered for the same
    /// request on `ggml-org/Clef-Flash-GGUF` Q4_K_M; `SKIPPY_CLEF_EXPECTED`
    /// overrides them as `refund,urgency0..3,route0..2`.
    #[test]
    #[ignore = "needs a Clef GGUF and a native runtime bundle"]
    fn real_model_matches_the_upstream_server() {
        crate::test_support::require_native_runtime();
        let path = std::env::var("SKIPPY_CLEF_MODEL").expect("SKIPPY_CLEF_MODEL");
        let model = ClefModel::open(&path, 8192, 8, None).unwrap();
        let request = DecisionRequest {
            state: text(
                "Customer message: I was charged twice for my order last week and nobody has replied. I want a refund now.",
            ),
            questions: vec![
                DecisionQuestion {
                    key: "refund".into(),
                    instructions: Some(text("Is the customer asking for a refund?")),
                    kind: DecisionQuestionKind::Noul {
                        when_true: None,
                        when_false: None,
                    },
                },
                DecisionQuestion {
                    key: "urgency".into(),
                    instructions: Some(text("How urgent is this?")),
                    kind: DecisionQuestionKind::Score {
                        levels: ["low", "medium", "high", "critical"]
                            .iter()
                            .map(|level| text(level))
                            .collect(),
                    },
                },
                DecisionQuestion {
                    key: "route".into(),
                    instructions: Some(text("Which team should handle this?")),
                    kind: DecisionQuestionKind::Choice {
                        options: ["billing", "shipping", "technical"]
                            .iter()
                            .map(|name| (name.to_string(), DecisionValue::Null))
                            .collect(),
                    },
                },
            ],
            seed: [0; 32],
        };
        let output = model.decide(&request).unwrap();
        let flat = output
            .probabilities
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        eprintln!(
            "clef probabilities: {flat:?} input_tokens={}",
            output.input_tokens
        );
        let expected = std::env::var("SKIPPY_CLEF_EXPECTED")
            .ok()
            .map(|csv| {
                csv.split(',')
                    .map(|v| v.trim().parse::<f32>().unwrap())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| {
                vec![
                    0.97709, 0.02291, 0.013462, 0.040932, 0.623534, 0.322072, 0.890486, 0.057542,
                    0.051972,
                ]
            });
        // the server answered input_tokens=330 for this request
        assert_eq!(output.input_tokens, 330);
        for (index, (got, want)) in flat.iter().zip(&expected).enumerate() {
            assert!(
                (got - want).abs() < 2e-3,
                "probability {index}: skippy {got} vs server {want}"
            );
        }
    }

    #[test]
    fn uncalibrated_distribution_is_a_plain_softmax() {
        let probs = Calibration::default().distribution(&choice(&["a", "b"]), &[1.0, 1.0]);
        assert!((probs[0] - 0.5).abs() < 1e-6);
    }
}
