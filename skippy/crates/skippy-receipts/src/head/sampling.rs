//! What a turn's sampler statement lets the head check judge of its tokens.
//!
//! A greedy turn must take the best logit. A sampled turn draws from the
//! candidates its filters keep, with the seed of the client's request; the
//! check does not replay the draw, but the token must be one the stated
//! filters keep, judged on the checked logits. That catches a node that
//! writes tokens of its own choosing. Penalties, a grammar and the other
//! samplers depend on more than the logits, so a turn that uses them gets
//! no token judgement; its logits are still checked.

use anyhow::{Context, Result};

use crate::turn;

/// How the head check judges the token a turn took at a position.
#[derive(Clone, Debug, PartialEq)]
pub enum TokenRule {
    /// The best logit, within the tie band.
    Greedy,
    /// A token the filters keep, applied in order to the logits plus
    /// `biases`.
    Filtered {
        filters: Vec<Filter>,
        biases: Vec<(i32, f64)>,
    },
    /// The sampler is beyond the check, for this reason.
    Unjudged(String),
}

/// One candidate filter of the sampler chain, in the stated order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Filter {
    TopK(u32),
    TopP(f64),
    MinP(f64),
    /// Divides the logits the later filters see.
    Temperature(f64),
}

/// The samplers a chain may name that leave the candidates alone when
/// their parameters are neutral, with whether they are.
fn neutral(name: &str, p: &serde_json::Value) -> Option<bool> {
    let f = |key: &str| p[key].as_f64();
    Some(match name {
        "penalties" => {
            f("repeat_penalty")? == 1.0
                && f("frequency_penalty")? == 0.0
                && f("presence_penalty")? == 0.0
        }
        "dry" => p["dry"]["multiplier"].as_f64()? == 0.0,
        "top_n_sigma" => f("top_nsigma")? <= 0.0,
        "typical_p" | "typ_p" => f("typical_p")? >= 1.0,
        "xtc" => p["xtc"]["probability"].as_f64()? <= 0.0,
        _ => return None,
    })
}

impl TokenRule {
    /// The rule a receipt's sampler statement gives.
    pub fn from_sampler(sampler: &str) -> Self {
        if sampler == turn::GREEDY {
            return Self::Greedy;
        }
        match turn::sampler_parameters(sampler) {
            Some(parameters) => Self::from_parameters(&parameters)
                .unwrap_or_else(|reason| Self::Unjudged(format!("{reason:#}"))),
            None => Self::Unjudged("the sampler statement names no parameters".into()),
        }
    }

    fn from_parameters(p: &serde_json::Value) -> Result<Self> {
        let unjudged = |reason: &str| Ok(Self::Unjudged(reason.into()));
        if p["grammar"].as_str().is_some_and(|g| !g.is_empty()) {
            return unjudged("a grammar constrains the tokens");
        }
        if p["mirostat"]["mode"].as_i64().unwrap_or(0) != 0 {
            return unjudged("mirostat replaces the sampler chain");
        }
        if p["reasoning_budget"].as_i64().is_some_and(|b| b >= 0) {
            return unjudged("a reasoning budget can force a token");
        }
        let number = |key: &str| {
            p[key]
                .as_f64()
                .with_context(|| format!("the sampler statement names no {key}"))
        };
        let temperature = number("temperature")?;
        let dynamic = p["dynatemp_range"].as_f64().unwrap_or(0.0) > 0.0;
        let mut filters = Vec::new();
        let samplers = p["samplers"]
            .as_array()
            .context("the sampler statement names no sampler chain")?;
        for name in samplers.iter().filter_map(serde_json::Value::as_str) {
            let filter = match name {
                "top_k" => Filter::TopK(u32::try_from(number("top_k")?.max(0.0) as i64)?),
                "top_p" => Filter::TopP(number("top_p")?),
                "min_p" => Filter::MinP(number("min_p")?),
                // a temperature of 0 keeps only the best logit
                "temperature" | "temp" if temperature <= 0.0 => Filter::TopK(1),
                "temperature" | "temp" if dynamic => {
                    // only the draw sees it last; a filter after it would not
                    // know the scale
                    filters.push(Filter::Temperature(f64::NAN));
                    continue;
                }
                "temperature" | "temp" => Filter::Temperature(temperature),
                other => match neutral(other, p) {
                    Some(true) => continue,
                    Some(false) => return unjudged(&format!("the {other} sampler is on")),
                    None => return unjudged(&format!("the {other} sampler is unknown")),
                },
            };
            filters.push(filter);
        }
        // a dynamic temperature after every filter changes only the draw
        while filters
            .last()
            .is_some_and(|f| matches!(f, Filter::Temperature(_)))
        {
            filters.pop();
        }
        if filters
            .iter()
            .any(|f| matches!(f, Filter::Temperature(t) if t.is_nan()))
        {
            return unjudged("a dynamic temperature comes before a filter");
        }
        let biases = p["logit_bias"]
            .as_array()
            .map(|pairs| {
                pairs
                    .iter()
                    .filter_map(|pair| Some((pair[0].as_i64()? as i32, pair[1].as_f64()?)))
                    .collect()
            })
            .unwrap_or_default();
        Ok(Self::Filtered { filters, biases })
    }

    /// The rule in words, for a report.
    pub fn describe(&self) -> String {
        match self {
            Self::Greedy => "the greedy choice".into(),
            Self::Filtered { filters, biases } => {
                let mut parts: Vec<String> = filters
                    .iter()
                    .map(|f| match f {
                        Filter::TopK(k) => format!("top_k {k}"),
                        Filter::TopP(p) => format!("top_p {p}"),
                        Filter::MinP(p) => format!("min_p {p}"),
                        Filter::Temperature(t) => format!("temperature {t}"),
                    })
                    .collect();
                if !biases.is_empty() {
                    parts.insert(0, format!("{} logit biases", biases.len()));
                }
                format!("kept by the sampler ({})", parts.join(", "))
            }
            Self::Unjudged(reason) => format!("not judged: {reason}"),
        }
    }

    /// The rule for logits the check computed without the transforms a
    /// model applies after the output matrix: only the greedy rule still
    /// holds on them, because a scale or a cap changes the probabilities.
    pub fn on_untransformed_logits(self, transformed: bool) -> Self {
        match self {
            Self::Filtered { .. } if transformed => Self::Unjudged(
                "the model scales or caps its logits, and the node opened none".into(),
            ),
            rule => rule,
        }
    }

    /// Whether `sampled` is a token the filters keep from `logits`, with
    /// `band` of slack in logits for the node's rounding.
    pub fn keeps(
        filters: &[Filter],
        biases: &[(i32, f64)],
        logits: &[f64],
        sampled: usize,
        band: f64,
    ) -> bool {
        let mut logits = logits.to_vec();
        for &(token, bias) in biases {
            if let Some(l) = usize::try_from(token).ok().and_then(|t| logits.get_mut(t)) {
                *l += bias;
            }
        }
        let mut band = band;
        for filter in filters {
            let s = logits[sampled];
            let kept = match *filter {
                Filter::TopK(0) => true,
                Filter::TopK(k) => logits.iter().filter(|&&l| l > s + band).count() < k as usize,
                Filter::TopP(p) if p >= 1.0 => true,
                Filter::TopP(p) => mass_above(&logits, s + band) < p,
                Filter::MinP(p) if p <= 0.0 => true,
                Filter::MinP(p) => {
                    let best = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    s >= best + p.ln() - band
                }
                Filter::Temperature(t) => {
                    for l in &mut logits {
                        *l /= t;
                    }
                    band /= t;
                    true
                }
            };
            if !kept {
                return false;
            }
        }
        true
    }
}

/// The softmax mass of the logits above `threshold`, over every logit. A
/// filter before this one keeps fewer candidates, which raises the mass,
/// so this is the most lenient reading.
fn mass_above(logits: &[f64], threshold: f64) -> f64 {
    let best = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let (above, total) = logits.iter().fold((0.0, 0.0), |(above, total), &l| {
        let e = (l - best).exp();
        (above + if l > threshold { e } else { 0.0 }, total + e)
    });
    above / total
}

#[cfg(test)]
mod tests;
