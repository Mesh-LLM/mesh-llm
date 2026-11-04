//! What a client asked the sampler for, against what the node says it ran.
//!
//! A receipt's sampler statement is the node's own text
//! ([`crate::turn::TurnBinding::sampler`]), and it decides how the head
//! check judges the turn's tokens ([`crate::head::sampling`]): a grammar, a
//! penalty or a sampler the check does not know leaves them unjudged, a
//! logit bias moves any token into the candidates, a chain without filters
//! keeps every token, and a statement that names no draw leaves the draw
//! unreplayed. A node that wrote tokens of its own choosing would state one
//! of those. So a client compares the statement with the request it sent:
//! a parameter it set must be stated with its exact value and applied, and
//! nothing that leaves the tokens or the draw unjudged may be stated
//! unless the request asked for it ([`crate::head::sampling::Gap`]).
//!
//! The request is not part of a receipt, so a contradiction convinces only
//! the client that sent it: it is a failure no evidence shows. A recheck of
//! evidence has no request, and reads the statement leniently: what the
//! statement leaves unjudged stays so, with a note.
//!
//! The seed is such a parameter: a node that names a draw a client can
//! replay ([`crate::head::draw`]) is held to the token the stated seed
//! picks, so a client that set the seed knows the node did not choose it.
//!
//! What stays open: a parameter the client left unset takes the node's
//! default, which the client knows only as far as
//! [`RequestedSampling::unset_filters_on`] says.

use serde_json::Value;

use crate::head::sampling::{self, Gap};
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
    /// Whether the request asked the node to ignore the end of generation
    /// (`ignore_eos`).
    pub ignore_eos: Option<bool>,
    /// Whether a filter the request left unset (top-k, top-p, min-p) runs
    /// at a default that is on, as Skippy's are (top-k 40, top-p 0.95,
    /// min-p 0.05). A node whose operator configured a default that turns
    /// a filter off states it off, which then contradicts the request.
    pub unset_filters_on: Option<bool>,
}

/// The ranges Skippy's request validation allows, so an honest node states
/// no value outside them (`sampling_config` in
/// `skippy-serving/src/frontend/request.rs`).
const RANGES: [(&str, f64, f64); 5] = [
    ("temperature", 0.0, 100.0),
    ("top_p", 0.0, 1.0),
    ("min_p", 0.0, 1.0),
    ("typical_p", 0.0, 1.0),
    ("dynatemp_range", 0.0, f64::MAX),
];

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
            ignore_eos: Some(false),
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
        let Some(stated) = turn::sampler_parameters(statement).filter(Value::is_object) else {
            return Some("the sampler statement is neither greedy nor a list of parameters".into());
        };
        let gaps = match sampling::gaps(&stated) {
            Ok(gaps) => gaps,
            Err(malformed) => {
                return Some(format!("the sampler statement is malformed: {malformed}"));
            }
        };
        against_range(&stated)
            .or_else(|| self.against_values(&stated))
            .or_else(|| self.against_budget(&stated))
            .or_else(|| self.against_chain(&stated))
            .or_else(|| self.against_unset_filters(&stated))
            .or_else(|| self.against_gaps(&gaps))
            .or_else(|| self.against_constraints(&stated))
    }

    /// A request the node decodes greedily however it states its sampler:
    /// a temperature of 0, or every filter turned off at a temperature of 1
    /// with no seed, which Skippy reads as sampling turned off.
    fn decodes_greedily(&self) -> bool {
        let Some(temperature) = self.temperature else {
            return false;
        };
        temperature <= 0.0
            || (same(temperature, 1.0)
                && self.top_p.is_some_and(|p| p >= 1.0)
                && self.top_k == Some(0)
                && self.min_p.is_some_and(|p| p <= 0.0))
    }

    /// A node decodes greedily at a temperature of 0, and when a request
    /// turns every filter off at a temperature of 1.
    fn against_greedy(&self) -> Option<String> {
        let temperature = self.temperature.filter(|t| *t > 0.0)?;
        (!self.decodes_greedily()).then(|| {
            format!("the request set temperature {temperature}; the node says it decoded greedily")
        })
    }

    /// Every parameter the request set is stated with its exact value: the
    /// node holds the numbers as f32, and the seed and top-k as integers.
    fn against_values(&self, stated: &Value) -> Option<String> {
        let numbers = [
            ("temperature", self.temperature),
            ("top_p", self.top_p),
            ("min_p", self.min_p),
        ];
        for (name, asked) in numbers {
            let Some(asked) = asked else { continue };
            match stated[name].as_f64() {
                Some(value) if same(value, asked) => {}
                // a request the runtime decodes greedily runs at 0
                Some(0.0) if name == "temperature" && self.decodes_greedily() => {}
                Some(value) => {
                    return Some(format!(
                        "the request set {name} {asked}; the node says {value}"
                    ));
                }
                None => return Some(format!("the request set {name}; the node states none")),
            }
        }
        let integers = [
            ("top_k", self.top_k),
            ("seed", self.client_seed().map(|s| s as i64)),
        ];
        for (name, asked) in integers {
            let Some(asked) = asked else { continue };
            match stated[name].as_i64() {
                Some(value) if value == asked => {}
                Some(value) => {
                    return Some(format!(
                        "the request set {name} {asked}; the node says {value}"
                    ));
                }
                None => return Some(format!("the request set {name}; the node states none")),
            }
        }
        None
    }

    /// The stated reasoning budget is at most the one the request allowed.
    fn against_budget(&self, stated: &Value) -> Option<String> {
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

    /// The filters the request turned on, or left at a default that is on.
    fn filters_on(&self) -> [(&'static str, bool); 3] {
        let unset_on = self.unset_filters_on == Some(true);
        [
            ("top_k", self.top_k.map_or(unset_on, |k| k > 0)),
            ("top_p", self.top_p.map_or(unset_on, |p| p < 1.0)),
            ("min_p", self.min_p.map_or(unset_on, |p| p > 0.0)),
        ]
    }

    /// A filter the request turned on is in the chain the node says it
    /// ran, when the request left the chain to the node. A chain without
    /// its filters keeps every token.
    fn against_chain(&self, stated: &Value) -> Option<String> {
        if self.sampler_chain != Some(false) {
            return None;
        }
        let chain = stated_chain(stated);
        let missing = self
            .filters_on()
            .into_iter()
            .find(|(name, on)| *on && !chain.iter().any(|c| c == name))
            .map(|(name, _)| name);
        let temperature = self.temperature.is_some()
            && !chain
                .iter()
                .any(|name| ["temperature", "temp"].contains(&name.as_str()));
        missing
            .or(temperature.then_some("temperature"))
            .map(|name| format!("the request set {name}; the node's sampler chain leaves it out"))
    }

    /// A filter the request left at a default that is on is not stated off.
    fn against_unset_filters(&self, stated: &Value) -> Option<String> {
        if self.unset_filters_on != Some(true) {
            return None;
        }
        let off = [
            (
                "top_k",
                self.top_k.is_none() && stated["top_k"].as_i64().is_some_and(|k| k <= 0),
            ),
            (
                "top_p",
                self.top_p.is_none() && stated["top_p"].as_f64().is_some_and(|p| p >= 1.0),
            ),
            (
                "min_p",
                self.min_p.is_none() && stated["min_p"].as_f64().is_some_and(|p| p <= 0.0),
            ),
        ];
        off.into_iter()
            .find(|(_, off)| *off)
            .map(|(name, _)| format!("the request left {name} on; the node says it was off"))
    }

    /// What leaves the tokens or the draw unjudged is something the
    /// request asked for.
    fn against_gaps(&self, gaps: &[Gap]) -> Option<String> {
        gaps.iter().find(|gap| self.unexplained(gap)).map(|gap| {
            format!(
                "the node says {}; the request asked for no such thing",
                gap.describe()
            )
        })
    }

    /// Whether the request leaves `gap` unexplained.
    fn unexplained(&self, gap: &Gap) -> bool {
        match gap {
            Gap::Grammar => self.grammar == Some(false),
            Gap::Mirostat | Gap::Sampler(_) | Gap::DynamicTemperature => {
                self.penalties == Some(false)
            }
            Gap::Unknown(_) => self.sampler_chain == Some(false) || self.penalties == Some(false),
            // a node that issues a receipt for a turn that draws names the
            // draw it replays
            Gap::NoDraw => true,
        }
    }

    /// Nothing the request did not send is claimed.
    fn against_constraints(&self, stated: &Value) -> Option<String> {
        let unsent = |what: &str| format!("the node says {what}; the request sent none");
        let biased = stated["logit_bias"]
            .as_array()
            .is_some_and(|b| !b.is_empty());
        if self.logit_bias == Some(false) && biased {
            return Some(unsent("logit biases moved the tokens"));
        }
        let ignores_end = stated["ignore_eos"].as_bool() == Some(true);
        if self.ignore_eos == Some(false) && ignores_end {
            return Some(unsent("the turn ignored its end"));
        }
        // the runtime builds a reasoning sampler only for a turn with a
        // finite budget, with or without a grammar (`reasoning_sampler` in
        // skippy-serving); the budget itself is held against the request
        // by `against_budget`
        let budgeted = stated["reasoning_budget"].as_i64().is_some_and(|b| b >= 0);
        if !stated[turn::REASONING].is_null() && !budgeted {
            return Some(
                "the node says a reasoning sampler forced tokens; it states no reasoning budget"
                    .to_owned(),
            );
        }
        None
    }
}

/// The names of the chain a statement says it ran, the runtime's default
/// chain for an empty one.
fn stated_chain(stated: &Value) -> Vec<String> {
    let names: Vec<String> = stated["samplers"]
        .as_array()
        .map(|names| {
            names
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    if !names.is_empty() {
        return names;
    }
    let number = |key: &str, default: f64| stated[key].as_f64().unwrap_or(default);
    sampling::default_chain(number("typical_p", 1.0), number("top_p", 1.0))
        .into_iter()
        .map(str::to_owned)
        .collect()
}

/// A stated value outside the range a request may set.
fn against_range(stated: &Value) -> Option<String> {
    let outside = RANGES.into_iter().find_map(|(name, low, high)| {
        let value = stated[name].as_f64()?;
        (!(low..=high).contains(&value))
            .then(|| format!("the stated {name} {value} is out of range"))
    });
    let top_k = stated["top_k"]
        .as_i64()
        .filter(|k| *k < 0)
        .map(|k| format!("the stated top_k {k} is out of range"));
    let mode = stated["mirostat"]["mode"]
        .as_i64()
        .filter(|m| !(0..=2).contains(m))
        .map(|m| format!("the stated mirostat mode {m} is out of range"));
    outside.or(top_k).or(mode)
}

/// Whether a stated value is the one asked: the node holds both as f32.
fn same(stated: f64, asked: f64) -> bool {
    stated as f32 == asked as f32
}

#[cfg(test)]
mod tests;
