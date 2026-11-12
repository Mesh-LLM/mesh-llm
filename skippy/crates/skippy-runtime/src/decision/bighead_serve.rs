//! BigHead serving model: the trained pointer-head backend plugged into skippy's
//! `DecisionModel` dispatch, alongside causal, laya, and diffusion.
//!
//! The scoring math lives in [`super::bighead`]; this module is the serving
//! seam: it renders the BigHead prompt layout, obtains per-token hidden states
//! from the torso, and assembles a [`DecisionOutput`].
//!
//! ## Hidden-state source
//!
//! Per-token final hidden states are the only torso output BigHead needs, and
//! `llama_get_embeddings_ith` is already bound in skippy-ffi. The provider is
//! injected as a closure so this layer is testable without a loaded model, and
//! so the llama plumbing lands as one focused change: prefill the rendered
//! prompt, then call the provider for the hidden states at the recorded
//! positions. Until that plumbing lands, [`BigHeadModel::unavailable`]
//! describes the gap and `decide` fails closed with a clear error.

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
    /// Question kind ("noul" | "choice" | "score").
    pub kind: String,
    /// Option labels in request order, exactly as rendered on their lines.
    /// The hidden-state provider tokenizes `- {label}\n` to locate each
    /// option line's final token.
    pub option_labels: Vec<String>,
    /// Number of offered options.
    pub n_options: usize,
}

/// Renders the BigHead prompt layout for a whole request. The layout follows
/// the reference implementation (strands-decider `prompting.py`): the shared
/// state, then the question with each offered answer on its own line, closed by
/// the `<answer>` marker. Option lines appear in request order — the head is
/// permutation-generic by construction, so no canonical reordering here.
pub fn render(
    request: &DecisionRequest,
    answer_marker: &str,
) -> Result<Vec<RenderedQuestion>, DecisionError> {
    let state = request.state.to_json().to_string();
    let mut out = Vec::new();
    for question in &request.questions {
        let (kind, options) = match &question.kind {
            super::DecisionQuestionKind::Noul { .. } => {
                ("noul", vec!["true".to_string(), "false".to_string()])
            }
            super::DecisionQuestionKind::Choice { options } => (
                "choice",
                options.iter().map(|(name, _)| name.clone()).collect(),
            ),
            super::DecisionQuestionKind::Score { levels } => (
                "score",
                levels
                    .iter()
                    .enumerate()
                    .map(|(i, _)| i.to_string())
                    .collect(),
            ),
        };
        let instructions = question
            .instructions
            .as_ref()
            .map(|i| i.to_json().as_str().unwrap_or_default().to_string())
            .unwrap_or_default();
        let mut prompt = format!("{state}\n\n{instructions}\n");
        for opt in &options {
            prompt.push_str(&format!("- {opt}\n"));
        }
        prompt.push_str(answer_marker);
        // Token positions are resolved by the hidden-state provider after
        // tokenization; here we carry order, and the provider maps it.
        let n = options.len();
        out.push(RenderedQuestion {
            prompt,
            answer_pos: usize::MAX, // filled by the provider
            option_positions: (0..n).map(|_| usize::MAX).collect(),
            kind: kind.to_string(),
            option_labels: options,
            n_options: n,
        });
    }
    Ok(out)
}

/// Supplies hidden states for a rendered prompt. Implementations tokenize,
/// prefill, and read per-token states (llama_get_embeddings_ith). Positions in
/// [`RenderedQuestion`] are filled from this side.
pub trait HiddenStateProvider {
    fn hidden_states(
        &self,
        rendered: &mut [RenderedQuestion],
    ) -> Result<Vec<Vec<Vec<f32>>>, DecisionError>;
}

impl HiddenStateProvider
    for fn(&mut [RenderedQuestion]) -> Result<Vec<Vec<Vec<f32>>>, DecisionError>
{
    fn hidden_states(
        &self,
        rendered: &mut [RenderedQuestion],
    ) -> Result<Vec<Vec<Vec<f32>>>, DecisionError> {
        (*self)(rendered)
    }
}

/// A loaded BigHead ready to serve.
pub struct BigHeadModel {
    config: BigHeadConfig,
    weights: BigHeadWeights,
    states: Option<Box<dyn HiddenStateProvider + Send + Sync>>,
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

    pub fn with_provider(mut self, provider: Box<dyn HiddenStateProvider + Send + Sync>) -> Self {
        self.states = Some(provider);
        self
    }

    /// Serving without a hidden-state provider (the llama plumbing is the
    /// remaining seam): construct the description of what is missing.
    pub fn unavailable(&self) -> DecisionError {
        DecisionError::Unsupported(
            "BigHead serving needs the hidden-state provider (llama_get_embeddings_ith \
             plumbing); math, schema, and dispatch are in place"
                .into(),
        )
    }
}

impl DecisionModel for BigHeadModel {
    fn decide(&self, request: &DecisionRequest) -> Result<DecisionOutput, DecisionError> {
        let provider = self.states.as_ref().ok_or_else(|| self.unavailable())?;
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
            input_tokens += q.prompt.split_whitespace().count();
        }
        Ok(DecisionOutput {
            probabilities,
            input_tokens,
        })
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
            .iter()
            .map(|q| {
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
