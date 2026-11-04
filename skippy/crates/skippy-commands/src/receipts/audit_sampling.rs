//! How `skippy audit-receipt` asks the node to choose its tokens.
//!
//! Without a temperature the request is greedy: the best logit at every
//! token, and nothing that could change the choice, which the head check
//! judges in full. With one, the node samples, and the request sets the
//! seed, so the node states a draw this client can replay with a seed it
//! did not choose.
//!
//! The request lets a thinking model reason without a limit unless it
//! names a reasoning budget. With one, the node's sampler forces the end
//! of the reasoning when the budget runs out, and the head check replays
//! which tokens it forced.

use skippy_receipts::sampler_request::RequestedSampling;

/// The sampling options of `skippy audit-receipt`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AuditSampling {
    /// 0 asks for greedy decoding.
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub top_k: Option<i64>,
    pub min_p: Option<f64>,
    /// The seed of a sampled turn. Without one the audit picks one.
    pub seed: Option<u32>,
    /// The reasoning tokens the request allows. Without a budget the
    /// model reasons without a limit.
    pub reasoning_budget: Option<u32>,
}

impl AuditSampling {
    fn sampled(&self) -> bool {
        self.temperature > 0.0
    }

    /// The options with a seed for a sampled turn: the one given, else a
    /// random one above 0, since a node reads seed 0 as none.
    pub(super) fn with_seed(mut self) -> Self {
        if self.sampled() && self.seed.is_none_or(|seed| seed == 0) {
            self.seed = Some(rand::RngExt::random_range(&mut rand::rng(), 1..=u32::MAX));
        }
        self
    }

    /// Sets the request's sampling fields in `body`.
    pub(super) fn apply(&self, body: &mut serde_json::Value) {
        body["temperature"] = self.temperature.max(0.0).into();
        // a node gives a thinking model a budget of its own (half the
        // tokens) when the request names none: the request always names
        // what it allows, so the comparison with the statement is exact
        body["reasoning_budget"] = self.requested_budget().into();
        if !self.sampled() {
            return;
        }
        if let Some(top_p) = self.top_p {
            body["top_p"] = top_p.into();
        }
        if let Some(top_k) = self.top_k {
            body["top_k"] = top_k.into();
        }
        if let Some(min_p) = self.min_p {
            body["min_p"] = min_p.into();
        }
        if let Some(seed) = self.seed {
            body["seed"] = seed.into();
        }
    }

    /// The reasoning tokens the request allows; -1 for no limit.
    fn requested_budget(&self) -> i64 {
        self.reasoning_budget.map_or(-1, i64::from)
    }

    /// What the request set, for the comparison with the node's sampler
    /// statement.
    pub(super) fn requested(&self) -> RequestedSampling {
        let greedy = RequestedSampling {
            reasoning_budget: Some(self.requested_budget()),
            ..RequestedSampling::greedy()
        };
        if !self.sampled() {
            return greedy;
        }
        RequestedSampling {
            temperature: Some(self.temperature),
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: self.min_p,
            seed: self.seed.map(u64::from),
            ..greedy
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn a_request_without_a_temperature_is_greedy() {
        let sampling = AuditSampling::default().with_seed();
        assert_eq!(sampling.seed, None);
        let mut body = json!({});
        sampling.apply(&mut body);
        assert_eq!(body, json!({ "temperature": 0.0, "reasoning_budget": -1 }));
        assert_eq!(sampling.requested(), RequestedSampling::greedy());
        // filters without a temperature are not sent: the turn is greedy
        let filtered = AuditSampling {
            top_k: Some(40),
            seed: Some(7),
            ..AuditSampling::default()
        };
        let mut body = json!({});
        filtered.with_seed().apply(&mut body);
        assert!(body.get("top_k").is_none() && body.get("seed").is_none());
    }

    #[test]
    fn a_sampled_request_sets_its_filters_and_a_seed() {
        let sampling = AuditSampling {
            temperature: 0.7,
            top_p: Some(0.95),
            top_k: Some(40),
            min_p: None,
            seed: None,
            reasoning_budget: None,
        }
        .with_seed();
        let seed = sampling.seed.expect("a seed");
        assert_ne!(seed, 0);
        let mut body = json!({ "model": "m" });
        sampling.apply(&mut body);
        assert_eq!(body["temperature"], 0.7);
        assert_eq!(body["top_p"], 0.95);
        assert_eq!(body["top_k"], 40);
        assert_eq!(body["seed"], seed);
        assert!(body.get("min_p").is_none());
        let requested = sampling.requested();
        assert_eq!(requested.seed, Some(u64::from(seed)));
        assert_eq!(requested.temperature, Some(0.7));
        assert_eq!(requested.top_k, Some(40));
        assert_eq!(requested.grammar, Some(false));
        assert_eq!(requested.reasoning_budget, Some(-1));
        // a given seed is kept, and seed 0 is replaced
        let given = AuditSampling {
            seed: Some(1234),
            ..sampling
        };
        assert_eq!(given.with_seed().seed, Some(1234));
        let zero = AuditSampling {
            seed: Some(0),
            ..sampling
        };
        assert_ne!(zero.with_seed().seed, Some(0));
    }

    #[test]
    fn a_request_with_a_reasoning_budget_names_it() {
        // greedy or sampled, the budget is sent and is what the node must state
        for temperature in [0.0, 0.7] {
            let sampling = AuditSampling {
                temperature,
                reasoning_budget: Some(64),
                ..AuditSampling::default()
            }
            .with_seed();
            let mut body = json!({});
            sampling.apply(&mut body);
            assert_eq!(body["reasoning_budget"], 64);
            assert_eq!(sampling.requested().reasoning_budget, Some(64));
        }
        // a budget of no tokens is a budget
        let none = AuditSampling {
            reasoning_budget: Some(0),
            ..AuditSampling::default()
        };
        let mut body = json!({});
        none.apply(&mut body);
        assert_eq!(body["reasoning_budget"], 0);
        assert_eq!(none.requested().reasoning_budget, Some(0));
    }
}
