//! Independent causal decision sequences, with explicit candidate likelihoods.
//!
//! Each question receives the same state prefix and its own question suffix.
//! A candidate is scored in the actual prompt tokenizer context. No sampled
//! answer, another question's answer, or model-family name affects the plan.

use anyhow::{Result, ensure};
use serde_json::json;

use super::{DecisionQuestion, DecisionQuestionKind, DecisionRequest};
use crate::{StageModel, StageModelReader};

pub const MAX_QUESTIONS: usize = 64;
pub const MAX_CANDIDATE_TOKENS: usize = 64;
pub const MAX_PROMPT_TOKENS: usize = 32_768;
pub const MAX_CHOICES: usize = 26;

/// The rule for combining conditional token log probabilities. No implicit
/// length penalty, temperature, or cross-question normalization is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadoutMethod {
    SelectedTokenLogits,
    CandidateLogLikelihood,
}

impl ReadoutMethod {
    /// Return the stable API identifier for this scoring method.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SelectedTokenLogits => "selected_token_logits",
            Self::CandidateLogLikelihood => "candidate_log_likelihood",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuestionPlan {
    pub key: String,
    pub prompt: Vec<i32>,
    /// Candidates remain in caller order, including when two share a token.
    pub candidates: Vec<Vec<i32>>,
    pub method: ReadoutMethod,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionPlan {
    pub questions: Vec<QuestionPlan>,
    pub shared_prefix_tokens: usize,
}

impl DecisionPlan {
    /// Plan independent questions using the loaded model's immutable vocabulary.
    pub fn for_model(model: &StageModel, request: &DecisionRequest) -> Result<Self> {
        Self::for_reader(&model.reader(), request)
    }

    /// Tokenize independent questions without holding a runtime or session lock.
    ///
    /// The reader retains the loaded model while planning. Capability admission,
    /// token limits, label boundaries, and candidate ordering match `for_model`.
    pub fn for_reader(reader: &StageModelReader, request: &DecisionRequest) -> Result<Self> {
        ensure!(
            reader.supports_causal_readout(),
            "loaded model does not support causal decision reads"
        );
        Self::tokenize(request, |text| {
            reader
                .tokenize_bounded(text, true, MAX_PROMPT_TOKENS)?
                .ok_or_else(|| anyhow::anyhow!("decision prompt exceeds the token limit"))
        })
    }

    /// Tokenization is injected so boundary behavior can be tested without
    /// substituting a guessed tokenizer for a loaded production model.
    fn tokenize(
        request: &DecisionRequest,
        mut tokenize: impl FnMut(&str) -> Result<Vec<i32>>,
    ) -> Result<Self> {
        ensure!(
            (1..=MAX_QUESTIONS).contains(&request.questions.len()),
            "decisions require 1 to {MAX_QUESTIONS} independent questions"
        );
        let state = serde_json::to_string(&request.state.to_json())?;
        let prefix = format!(
            "Read the shared state and answer only the question below. Select exactly one offered label.\nShared state: {state}\n"
        );
        let mut questions = Vec::with_capacity(request.questions.len());
        for question in &request.questions {
            let options = options(question)?;
            let instructions = question
                .instructions
                .as_ref()
                .map(super::DecisionValue::to_json);
            let prompt = format!(
                "{prefix}Question: {}\nAnswer:\n",
                json!({"instructions": instructions, "options": options})
            );
            let tokens = tokenize(&prompt)?;
            ensure!(
                !tokens.is_empty() && tokens.len() <= MAX_PROMPT_TOKENS,
                "decision prompt is empty or exceeds the token limit"
            );
            let mut candidates = Vec::with_capacity(options.len());
            for (label, _) in &options {
                let completed = tokenize(&format!("{prompt}{label}"))?;
                ensure!(
                    completed.starts_with(&tokens),
                    "question {:?}: label {label:?} changes the prompt token boundary",
                    question.key
                );
                let suffix = completed[tokens.len()..].to_vec();
                ensure!(
                    (1..=MAX_CANDIDATE_TOKENS).contains(&suffix.len()),
                    "question {:?}: candidate must contain 1 to {MAX_CANDIDATE_TOKENS} tokens",
                    question.key
                );
                ensure!(
                    !candidates.contains(&suffix),
                    "question {:?}: two labels tokenize identically",
                    question.key
                );
                candidates.push(suffix);
            }
            let method = if candidates.iter().all(|candidate| candidate.len() == 1) {
                ReadoutMethod::SelectedTokenLogits
            } else {
                ReadoutMethod::CandidateLogLikelihood
            };
            questions.push(QuestionPlan {
                key: question.key.clone(),
                prompt: tokens,
                candidates,
                method,
            });
        }
        let shared_prefix_tokens =
            questions
                .iter()
                .skip(1)
                .fold(questions[0].prompt.len(), |length, question| {
                    questions[0].prompt[..length]
                        .iter()
                        .zip(&question.prompt)
                        .take_while(|(a, b)| a == b)
                        .count()
                });
        Ok(Self {
            questions,
            shared_prefix_tokens,
        })
    }
}

/// Validate and expand question kinds into ordered label-description pairs.
fn options(question: &DecisionQuestion) -> Result<Vec<(String, serde_json::Value)>> {
    let options = match &question.kind {
        DecisionQuestionKind::Noul {
            when_true,
            when_false,
        } => vec![
            (
                "true".into(),
                when_true
                    .as_ref()
                    .map_or(json!("true"), super::DecisionValue::to_json),
            ),
            (
                "false".into(),
                when_false
                    .as_ref()
                    .map_or(json!("false"), super::DecisionValue::to_json),
            ),
        ],
        DecisionQuestionKind::Choice { options } => options
            .iter()
            .map(|(label, description)| (label.clone(), description.to_json()))
            .collect(),
        DecisionQuestionKind::Score { levels } => levels
            .iter()
            .enumerate()
            .map(|(index, description)| (index.to_string(), description.to_json()))
            .collect(),
    };
    ensure!(
        (2..=MAX_CHOICES).contains(&options.len()),
        "each question requires 2 to {MAX_CHOICES} options"
    );
    for (index, (label, _)) in options.iter().enumerate() {
        ensure!(
            !label.trim().is_empty() && !label.chars().any(char::is_control),
            "choice labels must be nonempty and contain no control characters"
        );
        ensure!(
            !options[..index].iter().any(|(earlier, _)| earlier == label),
            "choice labels must be distinct"
        );
    }
    Ok(options)
}

/// Relative mass over the offered candidates, not calibrated truth confidence.
/// Multi-token candidates use the sum of full-vocabulary conditional log
/// probabilities (length normalization: none).
pub fn normalize_candidate_scores(scores: &[f64]) -> Result<Vec<f32>> {
    ensure!(
        scores.len() >= 2 && scores.iter().all(|score| score.is_finite()),
        "candidate scores must be finite and include at least two choices"
    );
    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights = scores
        .iter()
        .map(|score| (score - maximum).exp())
        .collect::<Vec<_>>();
    let sum: f64 = weights.iter().sum();
    Ok(weights
        .into_iter()
        .map(|weight| (weight / sum) as f32)
        .collect())
}

#[cfg(test)]
mod native_tests;
#[cfg(test)]
mod tests;
