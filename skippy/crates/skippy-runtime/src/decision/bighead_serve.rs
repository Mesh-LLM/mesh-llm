//! BigHead serving model: the trained pointer-head backend plugged into skippy's
//! `DecisionModel` dispatch, alongside causal, laya, and diffusion.
//!
//! The scoring math lives in [`super::bighead`]; this module is the serving
//! seam: it renders the BigHead prompt layout, obtains per-token hidden states
//! from the torso, and assembles a [`DecisionOutput`].
//!
//! ## Hidden-state source
//!
//! Per-token final hidden states come from a scoped llama session. The runtime
//! loader attaches a BigHead artifact to `StageModel`; its decision dispatch
//! creates the session and provides the hidden-state reader for one call.

use super::bighead::{BigHeadConfig, BigHeadWeights, kind_probabilities, pointer_logits};
use super::{DecisionError, DecisionModel, DecisionOutput, DecisionRequest};

/// Renders one question's prompt and records where hidden states are read.
pub struct RenderedQuestion {
    /// Prompt text forwarded to the torso.
    pub prompt: String,
    /// Token index of the `<answer>` position (query side).
    pub answer_pos: usize,
    /// Token index of the last token of each option's line, in option order.
    pub option_positions: Vec<usize>,
    /// Token count with model-added special tokens, filled by the provider.
    pub prompt_token_count: usize,
    /// Question kind ("noul" | "choice" | "score").
    pub kind: String,
    /// Exact option lines, including labels and descriptions from training.
    pub option_lines: Vec<String>,
    /// Number of offered options.
    pub n_options: usize,
}

/// Renders the reference implementation's state/question/options layout so the
/// hidden states are read at the same token positions used during training.
pub fn render(
    request: &DecisionRequest,
    answer_marker: &str,
) -> Result<Vec<RenderedQuestion>, DecisionError> {
    let state = format!("<state>\n{}\n</state>\n", render_content(&request.state));
    let mut out = Vec::new();
    for question in &request.questions {
        let (kind, header, options) = match &question.kind {
            super::DecisionQuestionKind::Noul {
                when_true,
                when_false,
            } => (
                "noul",
                "Decide whether the statement is true of the state.",
                vec![
                    (
                        "false".to_string(),
                        when_false.as_ref().map_or_else(
                            || "the statement does not hold for this state".to_string(),
                            render_content,
                        ),
                    ),
                    (
                        "true".to_string(),
                        when_true.as_ref().map_or_else(
                            || "the statement holds for this state".to_string(),
                            render_content,
                        ),
                    ),
                ],
            ),
            super::DecisionQuestionKind::Choice { options } => (
                "choice",
                "Select exactly one option.",
                options
                    .iter()
                    .map(|(name, description)| (name.clone(), render_content(description)))
                    .collect(),
            ),
            super::DecisionQuestionKind::Score { levels } => (
                "score",
                "Rate the state against the ordered levels below (lowest first).",
                levels
                    .iter()
                    .enumerate()
                    .map(|(index, description)| (index.to_string(), render_content(description)))
                    .collect(),
            ),
        };
        let instructions = question
            .instructions
            .as_ref()
            .map_or_else(String::new, render_content);
        let mut prompt =
            format!("{state}<question type=\"{kind}\">\n{header}\n{instructions}\n<options>\n");
        let mut option_lines = Vec::with_capacity(options.len());
        for (index, (name, description)) in options.iter().enumerate() {
            let description = description.split_whitespace().collect::<Vec<_>>().join(" ");
            let line = if description.is_empty() {
                format!("{}. {name}", index + 1)
            } else {
                format!("{}. {name} — {description}", index + 1)
            };
            prompt.push_str(&line);
            prompt.push('\n');
            option_lines.push(line);
        }
        prompt.push_str("</options>\n</question>\n");
        prompt.push_str(answer_marker);
        out.push(RenderedQuestion {
            prompt,
            answer_pos: usize::MAX,
            option_positions: (0..options.len()).map(|_| usize::MAX).collect(),
            prompt_token_count: 0,
            kind: kind.to_string(),
            option_lines,
            n_options: options.len(),
        });
    }
    Ok(out)
}

fn render_content(value: &super::DecisionValue) -> String {
    match value {
        super::DecisionValue::Null => String::new(),
        super::DecisionValue::String(text) => text.trim().to_string(),
        _ => render_json(value, 0),
    }
}

fn render_json(value: &super::DecisionValue, depth: usize) -> String {
    let indent = |level: usize| " ".repeat(level * 2);
    match value {
        super::DecisionValue::Object(entries) if !entries.is_empty() => {
            let rendered = entries
                .iter()
                .map(|(key, value)| {
                    format!(
                        "{}{}: {}",
                        indent(depth + 1),
                        serde_json::to_string(key).unwrap_or_default(),
                        render_json(value, depth + 1)
                    )
                })
                .collect::<Vec<_>>();
            format!("{{\n{}\n{}}}", rendered.join(",\n"), indent(depth))
        }
        super::DecisionValue::Array(values) if !values.is_empty() => {
            let rendered = values
                .iter()
                .map(|value| format!("{}{}", indent(depth + 1), render_json(value, depth + 1)))
                .collect::<Vec<_>>();
            format!("[\n{}\n{}]", rendered.join(",\n"), indent(depth))
        }
        super::DecisionValue::Object(_) => "{}".into(),
        super::DecisionValue::Array(_) => "[]".into(),
        super::DecisionValue::Null => "null".into(),
        super::DecisionValue::Bool(value) => value.to_string(),
        super::DecisionValue::Number(value) => value.to_string(),
        super::DecisionValue::String(value) => serde_json::to_string(value).unwrap_or_default(),
    }
}

/// Supplies hidden states for a rendered prompt. Implementations tokenize,
/// prefill, and read per-token states (llama_get_embeddings_ith). Positions in
/// [`RenderedQuestion`] are filled from this side.
pub trait HiddenStateProvider {
    fn hidden_states(
        &mut self,
        rendered: &mut [RenderedQuestion],
    ) -> Result<Vec<Vec<Vec<f32>>>, DecisionError>;
}

impl HiddenStateProvider
    for fn(&mut [RenderedQuestion]) -> Result<Vec<Vec<Vec<f32>>>, DecisionError>
{
    fn hidden_states(
        &mut self,
        rendered: &mut [RenderedQuestion],
    ) -> Result<Vec<Vec<Vec<f32>>>, DecisionError> {
        (*self)(rendered)
    }
}

/// A loaded BigHead ready to serve.
pub struct BigHeadModel {
    config: BigHeadConfig,
    weights: BigHeadWeights,
    states: Option<std::sync::Mutex<Box<dyn HiddenStateProvider + Send>>>,
    /// Probability mass semantics reported alongside readouts.
    pub semantics: &'static str,
}

impl BigHeadModel {
    pub const SEMANTICS: &'static str = "trained_pointer_head_probability_measure_v1";

    pub fn new(config: BigHeadConfig, weights: BigHeadWeights) -> Self {
        Self {
            config,
            weights,
            states: None,
            semantics: Self::SEMANTICS,
        }
    }

    pub fn with_provider(mut self, provider: Box<dyn HiddenStateProvider + Send>) -> Self {
        self.states = Some(std::sync::Mutex::new(provider));
        self
    }

    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    /// Scores the request with a provider whose model/session borrow is scoped
    /// to this call. Production uses this path so raw llama handles cannot
    /// outlive their owning `StageModel` or `StageSession`.
    pub fn decide_with_provider(
        &self,
        request: &DecisionRequest,
        provider: &mut dyn HiddenStateProvider,
    ) -> Result<DecisionOutput, DecisionError> {
        let mut rendered = render(request, "<answer>")?;
        let states = provider.hidden_states(&mut rendered)?;
        let mut probabilities = Vec::with_capacity(rendered.len());
        let mut input_tokens = 0;
        for (q, hidden) in rendered.iter().zip(states) {
            if hidden.len() != q.n_options + 1 {
                return Err(DecisionError::Backend(anyhow::anyhow!(
                    "BigHead expected {} option states + 1 answer state, got {}",
                    q.n_options,
                    hidden.len()
                )));
            }
            let (answer, options) = hidden.split_last().expect("checked above");
            let option_refs: Vec<&[f32]> = options.iter().map(|v| v.as_slice()).collect();
            let logits = pointer_logits(&self.config, &self.weights, answer, &option_refs);
            probabilities.push(kind_probabilities(&self.config, &q.kind, &logits));
            input_tokens += q.prompt_token_count;
        }
        super::check_output(request, &probabilities)?;
        Ok(DecisionOutput {
            probabilities,
            input_tokens,
        })
    }

    /// Serving without a hidden-state provider (the llama plumbing is the
    /// remaining seam): construct the description of what is missing.
    pub fn unavailable(&self) -> DecisionError {
        DecisionError::Unsupported(
            "a standalone BigHead model needs the hidden-state provider supplied by a loaded StageModel"
                .into(),
        )
    }
}

impl DecisionModel for BigHeadModel {
    fn decide(&self, request: &DecisionRequest) -> Result<DecisionOutput, DecisionError> {
        let state = self.states.as_ref().ok_or_else(|| self.unavailable())?;
        let mut provider = state.lock().map_err(|_| {
            DecisionError::Backend(anyhow::anyhow!("BigHead provider lock poisoned"))
        })?;
        self.decide_with_provider(request, provider.as_mut())
    }
}

#[cfg(test)]
mod tests {
    use super::super::bighead::{BigHeadConfig, BigHeadWeights};
    use super::*;

    fn synthetic_states(
        rendered: &mut [RenderedQuestion],
    ) -> Result<Vec<Vec<Vec<f32>>>, DecisionError> {
        // One hidden state per option + the answer, deterministic: option k's
        // state has a 1 in component k, the answer state is all-ones — with the
        // identity head the highest logit lands on the last option.
        Ok(rendered
            .iter_mut()
            .map(|q| {
                q.prompt_token_count = q.prompt.split_whitespace().count();
                let mut rows: Vec<Vec<f32>> = (0..q.n_options)
                    .map(|k| {
                        let mut v = vec![0.0f32; 4];
                        v[k % 4] = 1.0;
                        v
                    })
                    .collect();
                rows.push(vec![1.0; 4]);
                rows
            })
            .collect())
    }

    fn model() -> BigHeadModel {
        let h = 4;
        let d = 2;
        let mut q = vec![0.0; d * h];
        let mut k = vec![0.0; d * h];
        for i in 0..d {
            q[i * h + i] = 1.0;
            k[i * h + i] = 1.0;
        }
        BigHeadModel::new(
            BigHeadConfig {
                schema_version: Some("1".into()),
                hidden_size: h,
                pointer_dim: d,
                kinds: Default::default(),
            },
            BigHeadWeights {
                norm_weight: vec![1.0; h],
                norm_bias: vec![0.0; h],
                q,
                k,
                q_bias: None,
                k_bias: None,
            },
        )
        .with_provider(Box::new(
            synthetic_states
                as fn(&mut [RenderedQuestion]) -> Result<Vec<Vec<Vec<f32>>>, DecisionError>,
        ))
    }

    #[test]
    fn render_matches_training_prompt_and_keeps_option_descriptions() {
        let dv = |value: &str| super::super::DecisionValue::String(value.into());
        let request = super::super::DecisionRequest {
            state: dv("the state"),
            questions: vec![super::super::DecisionQuestion {
                key: "choice".into(),
                instructions: Some(dv("Pick one")),
                kind: super::super::DecisionQuestionKind::Choice {
                    options: vec![
                        ("a".into(), dv("first option")),
                        ("b".into(), dv("second option")),
                    ],
                },
            }],
            seed: [0; 32],
        };
        let rendered = render(&request, "<answer>").unwrap();
        assert_eq!(
            rendered[0].prompt,
            "<state>\nthe state\n</state>\n<question type=\"choice\">\nSelect exactly one option.\nPick one\n<options>\n1. a — first option\n2. b — second option\n</options>\n</question>\n<answer>"
        );
        assert_eq!(
            rendered[0].option_lines,
            ["1. a — first option", "2. b — second option"]
        );
    }

    #[test]
    fn serves_all_question_kinds() {
        let dv = |s: &str| super::super::DecisionValue::String(s.into());
        let request = super::super::DecisionRequest {
            state: dv("the sky is blue"),
            questions: vec![
                super::super::DecisionQuestion {
                    key: "yes".into(),
                    instructions: Some(dv("Is it blue?")),
                    kind: super::super::DecisionQuestionKind::Noul {
                        when_true: None,
                        when_false: None,
                    },
                },
                super::super::DecisionQuestion {
                    key: "team".into(),
                    instructions: Some(dv("Which team?")),
                    kind: super::super::DecisionQuestionKind::Choice {
                        options: vec![("a".into(), dv("alpha")), ("b".into(), dv("beta"))],
                    },
                },
                super::super::DecisionQuestion {
                    key: "rate".into(),
                    instructions: Some(dv("How much?")),
                    kind: super::super::DecisionQuestionKind::Score {
                        levels: vec![dv("low"), dv("high"), dv("max")],
                    },
                },
            ],
            seed: [0u8; 32],
        };
        let out = model().decide(&request).unwrap();
        assert_eq!(out.probabilities.len(), 3);
        let expected_tokens = render(&request, "<answer>")
            .unwrap()
            .iter()
            .map(|q| q.prompt.split_whitespace().count())
            .sum::<usize>();
        assert_eq!(out.input_tokens, expected_tokens);
        for (dist, n) in out.probabilities.iter().zip([2, 2, 3]) {
            assert_eq!(dist.len(), n);
            assert!((dist.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn fails_closed_without_provider() {
        let request = super::super::DecisionRequest {
            state: super::super::DecisionValue::String("s".into()),
            questions: vec![super::super::DecisionQuestion {
                key: "q".into(),
                instructions: Some(super::super::DecisionValue::String("i?".into())),
                kind: super::super::DecisionQuestionKind::Noul {
                    when_true: None,
                    when_false: None,
                },
            }],
            seed: [0u8; 32],
        };
        let bare = BigHeadModel::new(
            BigHeadConfig {
                schema_version: None,
                hidden_size: 4,
                pointer_dim: 2,
                kinds: Default::default(),
            },
            BigHeadWeights {
                norm_weight: vec![1.0; 4],
                norm_bias: vec![0.0; 4],
                q: vec![0.0; 8],
                k: vec![0.0; 8],
                q_bias: None,
                k_bias: None,
            },
        );
        assert!(bare.decide(&request).is_err());
    }
}
