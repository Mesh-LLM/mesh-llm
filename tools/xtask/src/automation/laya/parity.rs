use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Deserialize)]
pub(super) struct Answer {
    pub noul: Option<f64>,
    #[serde(default)]
    pub probabilities: BTreeMap<String, f64>,
    pub choice: Option<serde_json::Value>,
}

impl Answer {
    fn probabilities(&self) -> DynResult<BTreeMap<String, f64>> {
        let probabilities = match self.noul {
            Some(probability) => BTreeMap::from([("true".into(), probability)]),
            None => self.probabilities.clone(),
        };
        if probabilities.is_empty() || probabilities.values().any(|value| !value.is_finite()) {
            return Err("answer requires finite probabilities".into());
        }
        Ok(probabilities)
    }
}

#[derive(Deserialize)]
pub(super) struct Question {
    pub input_ids: Vec<u64>,
}

#[derive(Deserialize)]
pub(super) struct Golden {
    pub state: crate::ci_plan::document::Json,
    pub questions: crate::ci_plan::document::Json,
    pub answers: BTreeMap<String, Answer>,
    pub per_question: BTreeMap<String, Question>,
}

#[derive(Deserialize)]
pub(super) struct Response {
    pub answers: BTreeMap<String, Answer>,
    pub per_question: Option<BTreeMap<String, Question>>,
}

#[derive(Serialize)]
pub(super) struct Comparison {
    pub fixture: String,
    pub failures: Vec<String>,
    pub max_abs_diff: f64,
    pub allowed: f64,
}

pub(super) fn allowance(name: &str) -> f64 {
    let upstream = match name {
        "choice_single_zh" => 0.0039,
        "choice_multi_zh" => 0.0087,
        "score_zh" => 0.0021,
        "noul_zh" => 0.0579,
        "choice_single_en" => 0.0027,
        "score_en" => 0.0131,
        "noul_en" => 0.0012,
        _ => 0.0,
    };
    upstream + 0.005
}

pub(super) fn compare(name: &str, golden: &Golden, response: &Response) -> DynResult<Comparison> {
    let mut failures = Vec::new();
    let mut worst = 0.0_f64;
    for (key, expected) in &golden.answers {
        let actual = response.answers.get(key).ok_or("missing question answer")?;
        let probabilities = actual.probabilities()?;
        for (option, expected_probability) in expected.probabilities()? {
            let actual_probability = probabilities.get(&option).ok_or("missing answer option")?;
            worst = worst.max((actual_probability - expected_probability).abs());
        }
        if expected.choice.is_some() && actual.choice != expected.choice {
            failures.push(format!("{key}: choice differs from the golden"));
        }
    }
    let allowed = allowance(name);
    if worst > allowed {
        failures.insert(0, format!("max |dp| {worst:.4} exceeds {allowed:.4}"));
    }
    if let Some(questions) = &response.per_question {
        for (key, expected) in &golden.per_question {
            if questions.get(key).map(|question| &question.input_ids) != Some(&expected.input_ids) {
                failures.push(format!("{key}: token ids differ from the golden"));
            }
        }
    }
    Ok(Comparison {
        fixture: name.into(),
        failures,
        max_abs_diff: (worst * 10_000.0).round() / 10_000.0,
        allowed: (allowed * 10_000.0).round() / 10_000.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn golden(name: &str) -> (Golden, Response) {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../ci/llama-canary/fixtures/laya-golden")
            .join(format!("{name}.json"));
        let bytes = std::fs::read(path).unwrap();
        (
            serde_json::from_slice(&bytes).unwrap(),
            serde_json::from_slice(&bytes).unwrap(),
        )
    }

    #[test]
    fn vendored_goldens_pass_with_exact_tokens() {
        for name in [
            "choice_single_zh",
            "choice_multi_zh",
            "score_zh",
            "noul_zh",
            "choice_single_en",
            "score_en",
            "noul_en",
        ] {
            let (golden, response) = golden(name);
            let result = compare(name, &golden, &response).unwrap();
            assert!(result.failures.is_empty(), "{name}");
        }
    }

    #[test]
    fn upstream_noul_error_within_budget_passes() {
        let (golden, mut response) = golden("noul_zh");
        *response
            .answers
            .values_mut()
            .next()
            .unwrap()
            .noul
            .as_mut()
            .unwrap() -= 0.058;
        let result = compare("noul_zh", &golden, &response).unwrap();
        assert!(result.failures.is_empty());
    }

    #[test]
    fn excess_probability_error_fails() {
        let (golden, mut response) = golden("choice_multi_zh");
        *response
            .answers
            .values_mut()
            .next()
            .unwrap()
            .probabilities
            .values_mut()
            .next()
            .unwrap() += 0.05;
        let result = compare("choice_multi_zh", &golden, &response).unwrap();
        assert!(
            result
                .failures
                .iter()
                .any(|failure| failure.contains("exceeds"))
        );
    }

    #[test]
    fn changed_choice_and_tokens_fail_even_with_matching_probabilities() {
        let (golden, mut response) = golden("choice_single_en");
        response.answers.values_mut().next().unwrap().choice = Some(serde_json::json!("wrong"));
        response
            .per_question
            .as_mut()
            .unwrap()
            .values_mut()
            .next()
            .unwrap()
            .input_ids
            .pop();
        let result = compare("choice_single_en", &golden, &response).unwrap();
        assert_eq!(result.failures.len(), 2);
    }

    #[test]
    fn missing_answers_are_environment_errors_not_passing_comparisons() {
        let (golden, mut response) = golden("score_en");
        response.answers.clear();
        assert!(compare("score_en", &golden, &response).is_err());
    }
}
