//! What a client asked the sampler for, against what the node says it ran.
//!
//! A receipt's sampler statement is the node's own text
//! ([`crate::turn::TurnBinding::sampler`]), and it decides how the head
//! check judges the turn's tokens ([`crate::head::sampling`]): a grammar, a
//! penalty or a sampler the check does not know leaves them unjudged, a
//! logit bias moves any token into the candidates, and a chain without
//! filters keeps every token. A node that wrote tokens of its own choosing
//! would state one of those. So a client compares the statement with the
//! request it sent: a parameter it set must be stated with its value and
//! applied, and a constraint it did not send must not be claimed.
//!
//! The request is not part of a receipt, so a contradiction convinces only
//! the client that sent it: it is a failure no evidence shows.
//!
//! The seed is such a parameter: a node that names a draw a client can
//! replay ([`crate::head::draw`]) is held to the token the stated seed
//! picks, so a client that set the seed knows the node did not choose it.
//!
//! What stays open: a parameter the client left unset takes the node's
//! default, which the client cannot know.

use serde_json::Value;

use crate::head::sampling::neutral;
use crate::turn;

/// The sampling parameters of a client's request. A field is `None` when
/// the request did not set it, or the caller does not know.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RequestedSampling {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<i64>,
    pub min_p: Option<f64>,
    /// The seed the request set, in its body or in its receipt request
    /// (`x-receipt-seed`). A seed of 0 asks for a random one, like none.
    pub seed: Option<u64>,
    /// Reasoning tokens the request allowed; negative for no limit.
    pub reasoning_budget: Option<i64>,
    /// Whether the request carried a grammar, or a response format or
    /// tools that make one.
    pub grammar: Option<bool>,
    /// Whether the request set a penalty (repeat, presence, frequency,
    /// DRY) or another sampler beyond temperature, top-k, top-p and min-p
    /// (typical-p, top-n-sigma, XTC, mirostat, a dynamic temperature).
    pub penalties: Option<bool>,
    /// Whether the request carried logit biases.
    pub logit_bias: Option<bool>,
    /// Whether the request named its own sampler chain (`samplers` or
    /// `sampler_sequence`). Without one, the node runs its default chain,
    /// which holds every filter.
    pub sampler_chain: Option<bool>,
}

/// The samplers that filter the candidates by a parameter of their own.
const FILTERS: [&str; 5] = ["top_k", "top_p", "min_p", "temperature", "temp"];

impl RequestedSampling {
    /// A request for the best logit at every token, with nothing that
    /// could change the choice: what `skippy audit-receipt` sends.
    pub fn greedy() -> Self {
        Self {
            temperature: Some(0.0),
            reasoning_budget: Some(-1),
            grammar: Some(false),
            penalties: Some(false),
            logit_bias: Some(false),
            sampler_chain: Some(false),
            ..Self::default()
        }
    }

    /// The seed the request fixed. A node asked for no seed, or for seed 0,
    /// chooses its own.
    pub fn client_seed(&self) -> Option<u64> {
        self.seed.filter(|seed| *seed != 0)
    }

    /// How `statement`, a receipt's sampler statement, contradicts the
    /// request; `None` when it does not.
    pub fn contradiction(&self, statement: &str) -> Option<String> {
        if statement == turn::GREEDY {
            return self.against_greedy();
        }
        let Some(stated) = turn::sampler_parameters(statement) else {
            return Some("the sampler statement is neither greedy nor a list of parameters".into());
        };
        self.against_values(&stated)
            .or_else(|| self.against_chain(&stated))
            .or_else(|| self.against_constraints(&stated))
    }

    /// A node decodes greedily at a temperature of 0, and when a request
    /// turns every filter off at a temperature of 1.
    fn against_greedy(&self) -> Option<String> {
        let temperature = self.temperature.filter(|t| *t > 0.0)?;
        let filters_off = same(temperature, 1.0)
            && self.top_p.is_some_and(|p| p >= 1.0)
            && self.top_k == Some(0)
            && self.min_p.is_some_and(|p| p <= 0.0);
        (!filters_off).then(|| {
            format!("the request set temperature {temperature}; the node says it decoded greedily")
        })
    }

    /// Every parameter the request set is stated with its value.
    fn against_values(&self, stated: &Value) -> Option<String> {
        let numbers = [
            ("temperature", self.temperature),
            ("top_p", self.top_p),
            ("top_k", self.top_k.map(|k| k as f64)),
            ("min_p", self.min_p),
            ("seed", self.client_seed().map(|s| s as f64)),
        ];
        for (name, asked) in numbers {
            let Some(asked) = asked else { continue };
            match stated[name].as_f64() {
                Some(value) if same(value, asked) => {}
                Some(value) => {
                    return Some(format!(
                        "the request set {name} {asked}; the node says {value}"
                    ));
                }
                None => return Some(format!("the request set {name}; the node states none")),
            }
        }
        let budget = stated["reasoning_budget"].as_i64();
        match (self.reasoning_budget, budget) {
            (Some(asked), Some(stated)) if asked < 0 || stated > asked => Some(format!(
                "the request allowed {} reasoning tokens; the node says it allowed {stated}",
                if asked < 0 {
                    "any number of".to_owned()
                } else {
                    asked.to_string()
                }
            )),
            (Some(asked), None) if asked >= 0 => Some(format!(
                "the request allowed {asked} reasoning tokens; the node states no limit"
            )),
            _ => None,
        }
    }

    /// A filter the request turned on is in the chain the node says it
    /// ran, when the request left the chain to the node. A chain without
    /// its filters keeps every token.
    fn against_chain(&self, stated: &Value) -> Option<String> {
        if self.sampler_chain != Some(false) {
            return None;
        }
        let chain: Vec<&str> = stated["samplers"]
            .as_array()
            .map(|names| names.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        let on = [
            ("top_k", self.top_k.is_some_and(|k| k > 0)),
            ("top_p", self.top_p.is_some_and(|p| p < 1.0)),
            ("min_p", self.min_p.is_some_and(|p| p > 0.0)),
        ];
        let missing = on
            .iter()
            .find(|(name, on)| *on && !chain.contains(name))
            .map(|(name, _)| *name);
        let temperature = self.temperature.is_some()
            && !chain
                .iter()
                .any(|name| ["temperature", "temp"].contains(name));
        missing
            .or(temperature.then_some("temperature"))
            .map(|name| format!("the request set {name}; the node's sampler chain leaves it out"))
    }

    /// Nothing the request did not send is claimed.
    fn against_constraints(&self, stated: &Value) -> Option<String> {
        let unsent = |what: &str| format!("the node says {what}; the request sent none");
        if self.grammar == Some(false) && stated["grammar"].as_str().is_some_and(|g| !g.is_empty())
        {
            return Some(unsent("a grammar constrained the tokens"));
        }
        let biased = stated["logit_bias"]
            .as_array()
            .is_some_and(|b| !b.is_empty());
        if self.logit_bias == Some(false) && biased {
            return Some(unsent("logit biases moved the tokens"));
        }
        if self.penalties != Some(false) {
            return None;
        }
        if stated["mirostat"]["mode"].as_i64().unwrap_or(0) != 0 {
            return Some(unsent("mirostat replaced the sampler chain"));
        }
        if stated["dynatemp_range"].as_f64().unwrap_or(0.0) > 0.0 {
            return Some(unsent("the temperature was dynamic"));
        }
        let chain = stated["samplers"].as_array()?;
        chain
            .iter()
            .filter_map(Value::as_str)
            .filter(|name| !FILTERS.contains(name))
            .find(|name| neutral(name, stated) != Some(true))
            .map(|name| unsent(&format!("the {name} sampler was on")))
    }
}

/// Whether a stated value is the one asked, allowing for the node holding
/// it as an f32.
fn same(stated: f64, asked: f64) -> bool {
    (stated - asked).abs() <= 1e-6 * asked.abs().max(1.0)
}

#[cfg(test)]
mod tests;
