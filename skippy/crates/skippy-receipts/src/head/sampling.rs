//! What a turn's sampler statement lets the head check judge of its tokens.
//!
//! A greedy turn must take the best logit. A sampled turn draws from the
//! candidates its filters keep: the token must be one the stated filters
//! keep, judged on the checked logits, and when the statement names a draw
//! a client can replay ([`DrawRule`]) it must be the one that draw picks
//! ([`super::draw`]). That catches a node that writes tokens of its own
//! choosing. Penalties, a grammar and the other samplers depend on more
//! than the logits, so a turn that uses them gets no token judgement; its
//! logits are still checked. What leaves a turn's tokens or its draw
//! unjudged is a [`Gap`], and a client holds each one against its request
//! ([`crate::sampler_request`]): a node may not opt out of the judgement.
//!
//! The rule follows the runtime (`src/skippy/sampling_config.cpp`): an empty
//! chain is its default chain, a temperature of 0 keeps the best logit, a
//! top-p of 0 keeps the best token, and the model's suppressed tokens and,
//! for a turn that ignores its end, the end-of-generation tokens are taken
//! out of every candidate list before the chain runs ([`TokenRule::without`]).
//!
//! A reasoning budget forces the tag that ends the reasoning, and leaves
//! every other token to this rule. The head check replays which tokens it
//! forced ([`super::reasoning_budget`]).
//!
//! A statement whose fields do not have the types an honest node writes is
//! the node's fault ([`check_statement`]): the node signed it.

use anyhow::Result;
use serde_json::Value;

use super::draw::IndexedDraw;
use super::reasoning_budget::BudgetStatement;
use crate::node_fault::node_fault;
use crate::turn;

/// How the head check judges the token a turn took at a position.
#[derive(Clone, Debug, PartialEq)]
pub enum TokenRule {
    /// The best logit, within the tie band.
    Greedy,
    /// A token the filters keep, applied in order to the logits plus
    /// `biases`, and the one the draw picks among them when it can be
    /// replayed.
    Filtered {
        filters: Vec<Filter>,
        biases: Vec<(i32, f64)>,
        draw: DrawRule,
    },
    /// The sampler is beyond the check, for this reason.
    Unjudged(String),
}

/// Whether the check can replay which of the kept candidates a turn drew.
#[derive(Clone, Debug, PartialEq)]
pub enum DrawRule {
    /// The statement names the indexed draw and its seed.
    Indexed(IndexedDraw),
    /// The chain keeps the best logit alone, so the turn draws nothing:
    /// the kept-set rule is the whole judgement.
    Single,
    /// The draw is beyond the check, for this reason.
    Unjudged(String),
}

/// What a sampler statement leaves beyond the token judgement. Each is
/// something a client asks for; a node that states one its client did not
/// ask for contradicts the request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Gap {
    /// A grammar constrains the tokens.
    Grammar,
    /// Mirostat replaces the sampler chain.
    Mirostat,
    /// A sampler the check does not replay is on: a penalty, DRY,
    /// typical-p, top-n-sigma or XTC.
    Sampler(String),
    /// The temperature is dynamic.
    DynamicTemperature,
    /// The chain names a sampler the check does not know.
    Unknown(String),
    /// The turn draws from its chain and names no draw a client can
    /// replay.
    NoDraw,
}

impl Gap {
    pub fn describe(&self) -> String {
        match self {
            Self::Grammar => "a grammar constrains the tokens".into(),
            Self::Mirostat => "mirostat replaces the sampler chain".into(),
            Self::Sampler(name) => format!("the {name} sampler is on"),
            Self::DynamicTemperature => "the temperature is dynamic".into(),
            Self::Unknown(name) => format!("the {name} sampler is unknown"),
            Self::NoDraw => "the node names no draw a client can replay".into(),
        }
    }
}

/// A draw the check replays: the candidates are the tokens `filters` keep
/// of the logits plus `biases`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ReplayedDraw<'a> {
    pub(crate) draw: IndexedDraw,
    pub(crate) filters: &'a [Filter],
    pub(crate) biases: &'a [(i32, f64)],
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

/// The chain the runtime runs when a request names none of its own, as
/// `sampling_params` of `src/skippy/sampling_config.cpp` builds it from
/// the stated typical-p and top-p.
pub(crate) fn default_chain(typical_p: f64, top_p: f64) -> Vec<&'static str> {
    let between = |p: f64| p > 0.0 && p < 1.0;
    [
        "penalties",
        "dry",
        "top_n_sigma",
        "top_k",
        "typical_p",
        "top_p",
        "min_p",
        "xtc",
        "temperature",
    ]
    .into_iter()
    .filter(|name| match *name {
        "typical_p" => between(typical_p),
        "top_p" => between(top_p),
        _ => true,
    })
    .collect()
}

/// The samplers a chain may name that leave the candidates alone when
/// their parameters are neutral, with whether they are. `None` for a name
/// that is no such sampler; an error for a statement without the sampler's
/// parameters.
pub(crate) fn neutral(name: &str, p: &Value) -> Option<Result<bool, String>> {
    let f = |value: &Value, key: &str| {
        value
            .as_f64()
            .ok_or_else(|| format!("the statement names the {name} sampler and no number {key}"))
    };
    let neutral = || -> Result<bool, String> {
        Ok(match name {
            "penalties" => {
                f(&p["repeat_penalty"], "repeat_penalty")? == 1.0
                    && f(&p["frequency_penalty"], "frequency_penalty")? == 0.0
                    && f(&p["presence_penalty"], "presence_penalty")? == 0.0
            }
            "dry" => f(&p["dry"]["multiplier"], "dry.multiplier")? == 0.0,
            "top_n_sigma" => f(&p["top_nsigma"], "top_nsigma")? <= 0.0,
            "typical_p" | "typ_p" => f(&p["typical_p"], "typical_p")? >= 1.0,
            _ => f(&p["xtc"]["probability"], "xtc.probability")? <= 0.0,
        })
    };
    matches!(
        name,
        "penalties" | "dry" | "top_n_sigma" | "typical_p" | "typ_p" | "xtc"
    )
    .then(neutral)
}

/// A sampled statement read with the types an honest node writes.
struct Read {
    filters: Vec<Filter>,
    biases: Vec<(i32, f64)>,
    draw: DrawRule,
    /// What leaves every token unjudged.
    whole: Vec<Gap>,
    /// What leaves the draw unjudged.
    draw_gaps: Vec<Gap>,
}

/// The number `p[key]`, which the statement must hold.
fn number(p: &Value, key: &str) -> Result<f64, String> {
    p[key]
        .as_f64()
        .ok_or_else(|| format!("the sampler statement names no number {key}"))
}

/// The number `p[key]`, or `default` when the statement leaves it out.
fn optional_number(p: &Value, key: &str, default: f64) -> Result<f64, String> {
    match &p[key] {
        Value::Null => Ok(default),
        value => value
            .as_f64()
            .ok_or_else(|| format!("the stated {key} is not a number")),
    }
}

/// The chain's names: the stated chain, or the runtime's default one when
/// the statement names an empty chain, as the runtime reads it.
fn chain(p: &Value) -> Result<Vec<String>, String> {
    let samplers = p["samplers"]
        .as_array()
        .ok_or("the sampler statement names no sampler chain")?;
    let names = samplers
        .iter()
        .map(|name| {
            name.as_str()
                .map(str::to_owned)
                .ok_or_else(|| "the stated sampler chain holds a name that is not text".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    if !names.is_empty() {
        return Ok(names);
    }
    let typical = optional_number(p, "typical_p", 1.0)?;
    let top_p = number(p, "top_p")?;
    Ok(default_chain(typical, top_p)
        .into_iter()
        .map(str::to_owned)
        .collect())
}

/// The stated logit biases.
fn biases(p: &Value) -> Result<Vec<(i32, f64)>, String> {
    let malformed = || "the stated logit biases are not pairs of a token and a number".to_owned();
    match &p["logit_bias"] {
        Value::Null => Ok(Vec::new()),
        Value::Array(pairs) => pairs
            .iter()
            .map(|pair| {
                let token = pair[0].as_i64().and_then(|t| i32::try_from(t).ok());
                match (token, pair[1].as_f64(), pair.as_array().map(Vec::len)) {
                    (Some(token), Some(bias), Some(2)) => Ok((token, bias)),
                    _ => Err(malformed()),
                }
            })
            .collect(),
        _ => Err(malformed()),
    }
}

/// The types of the fields that do not shape the filters: what an honest
/// node writes, or nothing.
fn check_types(p: &Value) -> Result<(), String> {
    let typed = |key: &str, fits: fn(&Value) -> bool, what: &str| {
        if p[key].is_null() || fits(&p[key]) {
            Ok(())
        } else {
            Err(format!("the stated {key} is not {what}"))
        }
    };
    typed("draw", Value::is_string, "text")?;
    typed("ignore_eos", Value::is_boolean, "true or false")?;
    typed("grammar", Value::is_string, "text")?;
    typed("reasoning_budget", Value::is_i64, "a whole number")?;
    typed("mirostat", Value::is_object, "an object")?;
    if !p["mirostat"]["mode"].is_null() && !p["mirostat"]["mode"].is_i64() {
        return Err("the stated mirostat mode is not a whole number".into());
    }
    let Some(seed) = p["seed"].as_u64() else {
        return Err("the sampler statement names no seed".into());
    };
    u32::try_from(seed).map_err(|_| format!("the stated seed {seed} is past any seed"))?;
    BudgetStatement::check_types(p)
}

/// The stated filter of `name`, or `None` for a name that is no filter.
fn filter(p: &Value, name: &str) -> Result<Option<Filter>, String> {
    Ok(Some(match name {
        "top_k" => {
            let k = p["top_k"]
                .as_i64()
                .ok_or("the stated top_k is not a whole number")?;
            let k = i32::try_from(k).map_err(|_| format!("the stated top_k {k} is past any"))?;
            // llama.cpp ignores a top-k of 0 or below
            Filter::TopK(u32::try_from(k).unwrap_or(0))
        }
        "top_p" => Filter::TopP(number(p, "top_p")?),
        "min_p" => Filter::MinP(number(p, "min_p")?),
        _ => return Ok(None),
    }))
}

/// The filters of a stated chain, in order, and what in it the check
/// cannot judge.
struct Chain {
    filters: Vec<Filter>,
    gaps: Vec<Gap>,
    /// How many temperature samplers the chain holds.
    temperatures: i32,
}

/// The filter a temperature sampler of the chain is. llama.cpp's dynamic
/// temperature runs at any base temperature, 0 included
/// (`llama_sampler_temp_ext_apply`); a temperature of 0 that is not
/// dynamic keeps the best logit (`llama_sampler_temp_impl`).
fn temperature_filter(temperature: f64, dynamic: bool) -> Filter {
    if dynamic {
        Filter::Temperature(f64::NAN)
    } else if temperature <= 0.0 {
        Filter::TopK(1)
    } else {
        Filter::Temperature(temperature)
    }
}

/// Reads the stated chain of `p` at `temperature`.
fn read_chain(p: &Value, temperature: f64, dynamic: bool) -> Result<Chain, String> {
    let mut read = Chain {
        filters: Vec::new(),
        gaps: Vec::new(),
        temperatures: 0,
    };
    for name in chain(p)? {
        if let Some(found) = filter(p, &name)? {
            read.filters.push(found);
        } else if matches!(name.as_str(), "temperature" | "temp") {
            read.temperatures += 1;
            read.filters.push(temperature_filter(temperature, dynamic));
        } else {
            match neutral(&name, p) {
                Some(Ok(true)) => {}
                Some(Ok(false)) => read.gaps.push(Gap::Sampler(name)),
                Some(Err(malformed)) => return Err(malformed),
                None => read.gaps.push(Gap::Unknown(name)),
            }
        }
    }
    // a temperature after every filter changes only the draw
    while matches!(read.filters.last(), Some(Filter::Temperature(_))) {
        read.filters.pop();
    }
    // a dynamic temperature before a filter changes what the filter keeps
    if read
        .filters
        .iter()
        .any(|f| matches!(f, Filter::Temperature(t) if t.is_nan()))
    {
        read.gaps.push(Gap::DynamicTemperature);
    }
    Ok(read)
}

/// The draw a statement `p` of `chain` names, and what leaves it unjudged.
/// `whole` is what leaves every token unjudged already.
fn read_draw(
    p: &Value,
    chain: &Chain,
    temperature: f64,
    dynamic: bool,
    whole: &[Gap],
) -> (DrawRule, Vec<Gap>) {
    // a temperature of 0 or a top-k of 1 leaves one candidate, which
    // `common_sampler_init` takes without a draw when the chain ends in it
    if chain.filters.contains(&Filter::TopK(1)) {
        return (DrawRule::Single, Vec::new());
    }
    if dynamic && chain.temperatures > 0 {
        let gap = Gap::DynamicTemperature;
        return (DrawRule::Unjudged(gap.describe()), vec![gap]);
    }
    if p["draw"].as_str() == Some(turn::INDEXED_DRAW) {
        let seed = p["seed"].as_u64().and_then(|s| u32::try_from(s).ok());
        let draw = IndexedDraw {
            seed: seed.unwrap_or_default(),
            // every temperature of the chain divides the logits the draw sees
            temperature: (0..chain.temperatures).fold(1.0, |product, _| product * temperature),
        };
        return (DrawRule::Indexed(draw), Vec::new());
    }
    // mirostat draws by itself: the whole turn is already unjudged
    let gaps = if whole.is_empty() {
        vec![Gap::NoDraw]
    } else {
        Vec::new()
    };
    (DrawRule::Unjudged(Gap::NoDraw.describe()), gaps)
}

/// Reads the parameters `p` of a sampled statement, or says how they are
/// malformed.
fn read(p: &Value) -> Result<Read, String> {
    check_types(p)?;
    let temperature = number(p, "temperature")?;
    let dynamic = optional_number(p, "dynatemp_range", 0.0)? > 0.0;
    let mut whole = Vec::new();
    if p["grammar"].as_str().is_some_and(|g| !g.is_empty()) {
        whole.push(Gap::Grammar);
    }
    if p["mirostat"]["mode"].as_i64().unwrap_or(0) != 0 {
        whole.push(Gap::Mirostat);
    }
    let chain = read_chain(p, temperature, dynamic)?;
    whole.extend(chain.gaps.iter().cloned());
    let (draw, draw_gaps) = read_draw(p, &chain, temperature, dynamic, &whole);
    Ok(Read {
        filters: chain.filters,
        biases: biases(p)?,
        draw,
        whole,
        draw_gaps,
    })
}

/// Checks that `sampler`, a receipt's sampler statement, is one an honest
/// node writes: [`turn::GREEDY`], or a sampled statement whose fields have
/// the types the node writes them with. A statement that is not is the
/// node's fault: it signed it. A statement that is well formed and states
/// something the check cannot judge is not ([`gaps`]).
pub fn check_statement(sampler: &str) -> Result<()> {
    if sampler == turn::GREEDY {
        return Ok(());
    }
    let Some(parameters) = turn::sampler_parameters(sampler).filter(Value::is_object) else {
        node_fault!("the sampler statement is neither greedy nor a list of parameters");
    };
    if let Err(malformed) = read(&parameters) {
        node_fault!("the sampler statement is malformed: {malformed}");
    }
    Ok(())
}

/// What the sampled statement `p` leaves beyond the token judgement, or
/// how it is malformed.
pub fn gaps(p: &Value) -> Result<Vec<Gap>, String> {
    let read = read(p)?;
    Ok(read.whole.into_iter().chain(read.draw_gaps).collect())
}

/// Whether the statement `sampler` says the turn never picked an
/// end-of-generation token (`ignore_eos`).
pub fn ignores_end_tokens(sampler: &str) -> bool {
    turn::sampler_parameters(sampler).is_some_and(|p| p["ignore_eos"].as_bool() == Some(true))
}

impl TokenRule {
    /// The rule a receipt's sampler statement gives.
    pub(crate) fn from_sampler(sampler: &str) -> Self {
        if sampler == turn::GREEDY {
            return Self::Greedy;
        }
        let Some(parameters) = turn::sampler_parameters(sampler) else {
            return Self::Unjudged("the sampler statement names no parameters".into());
        };
        match read(&parameters) {
            Ok(read) => match read.whole.first() {
                Some(gap) => Self::Unjudged(gap.describe()),
                None => Self::Filtered {
                    filters: read.filters,
                    biases: read.biases,
                    draw: read.draw,
                },
            },
            Err(malformed) => Self::Unjudged(format!("the statement is malformed: {malformed}")),
        }
    }

    /// The rule in words, for a report.
    pub fn describe(&self) -> String {
        match self {
            Self::Greedy => "the greedy choice".into(),
            Self::Filtered {
                filters,
                biases,
                draw,
            } => {
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
                let kept = format!("kept by the sampler ({})", parts.join(", "));
                match draw {
                    DrawRule::Indexed(draw) => {
                        format!("{kept} and picked by the draw of seed {}", draw.seed)
                    }
                    DrawRule::Single => kept,
                    DrawRule::Unjudged(reason) => format!("{kept}; draw not replayed: {reason}"),
                }
            }
            Self::Unjudged(reason) => format!("not judged: {reason}"),
        }
    }

    /// The rule for a turn whose runtime took `excluded` out of every
    /// candidate list before its chain ran: the model's suppressed tokens,
    /// and for a turn that ignores its end the end-of-generation tokens.
    /// llama.cpp gives them a logit bias of minus infinity
    /// (`common_sampler_init` in `common/sampling.cpp`, `chat_params` in
    /// `src/skippy/sampling_config.cpp`). An error is why the client cannot
    /// read which tokens those are: the token judgement is then the
    /// client's gap, and no node's.
    pub fn without(self, excluded: Result<&[i32], &str>) -> Self {
        let excluded = match excluded {
            Ok([]) => return self,
            Ok(tokens) => tokens,
            Err(reason) => {
                return match self {
                    Self::Unjudged(_) => self,
                    _ => Self::Unjudged(reason.to_owned()),
                };
            }
        };
        let removed = excluded.iter().map(|&token| (token, f64::NEG_INFINITY));
        match self {
            // the greedy choice among the tokens left
            Self::Greedy => Self::Filtered {
                filters: vec![Filter::TopK(1)],
                biases: removed.collect(),
                draw: DrawRule::Single,
            },
            Self::Filtered {
                filters,
                mut biases,
                draw,
            } => {
                biases.extend(removed);
                Self::Filtered {
                    filters,
                    biases,
                    draw,
                }
            }
            Self::Unjudged(_) => self,
        }
    }

    /// The draw the rule replays, and what it draws over.
    pub(crate) fn replayed_draw(&self) -> Option<ReplayedDraw<'_>> {
        match self {
            Self::Filtered {
                filters,
                biases,
                draw: DrawRule::Indexed(draw),
            } => Some(ReplayedDraw {
                draw: *draw,
                filters,
                biases,
            }),
            _ => None,
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
        let mut logits = biased(logits, biases);
        // a token the runtime took out has no weight in any draw
        if logits[sampled] == f64::NEG_INFINITY && logits.iter().any(|l| l.is_finite()) {
            return false;
        }
        let mut band = band;
        for filter in filters {
            let s = logits[sampled];
            let best_alone = || logits.iter().filter(|&&l| l > s + band).count() == 0;
            let kept = match *filter {
                Filter::TopK(0) => true,
                Filter::TopK(k) => logits.iter().filter(|&&l| l > s + band).count() < k as usize,
                Filter::TopP(p) if p >= 1.0 => true,
                // llama.cpp's top-p keeps the best token whatever the p
                Filter::TopP(p) if p <= 0.0 => best_alone(),
                Filter::TopP(p) => mass_above(&logits, s + band) < p,
                Filter::MinP(p) if p <= 0.0 => true,
                // no token reaches p of the best: llama.cpp keeps the best
                Filter::MinP(p) if p > 1.0 => best_alone(),
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

/// `logits` with the turn's logit biases added, one after the other in
/// f32, as llama.cpp's logit-bias sampler adds them
/// (`llama_sampler_logit_bias_apply`).
pub(crate) fn biased(logits: &[f64], biases: &[(i32, f64)]) -> Vec<f64> {
    let mut logits = logits.to_vec();
    for &(token, bias) in biases {
        if let Some(l) = usize::try_from(token).ok().and_then(|t| logits.get_mut(t)) {
            *l = f64::from(*l as f32 + bias as f32);
        }
    }
    logits
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
