//! How a turn chose its tokens, as its receipt states it.
//!
//! The statement goes into the signed turn root, so a node that samples
//! otherwise than it says signs a false statement, and a client checks a
//! turn that says it decoded greedily against the output head.

use skippy_receipts::turn;

/// The grammar a turn's chat metadata constrains it with, if any.
fn grammar(chat_metadata: Option<&str>) -> Option<String> {
    let metadata: serde_json::Value = serde_json::from_str(chat_metadata?).ok()?;
    let grammar = metadata.get("grammar")?.as_str()?;
    (!grammar.is_empty()).then(|| grammar.to_owned())
}

/// What the runtime builds a turn's reasoning-budget sampler from, as the
/// chat metadata gives it: the tag that starts the reasoning, the tags
/// that end it, and the text the chat template placed after the prompt,
/// which the sampler reads first. `None` when the runtime builds no such
/// sampler: the turn has no budget, or the metadata the runtime configures
/// the sampler with names no tags (`chat_params` in the patch's
/// `src/skippy/sampling_config.cpp`). The runtime does not configure a
/// turn without a grammar with its whole metadata
/// (`skippy_runtime::chat_sampling_metadata_for_native`): it gives the
/// reasoning tags only when the turn has a finite budget and the template
/// did not already close the reasoning block. So the statement reads the
/// metadata through that function, and a turn that the runtime gave no
/// tags states no reasoning-budget sampler, whatever budget it states.
///
/// A client replays the sampler from these to know which tokens it forced
/// (`skippy_receipts::head::reasoning_budget`), so they are the texts the
/// runtime was given, unchanged. A statement that named a sampler the
/// runtime did not build would fail an honest turn whose reasoning runs
/// past the budget.
fn reasoning_sampler(
    sampling: &skippy_runtime::SamplingConfig,
    chat_metadata: Option<&str>,
) -> Option<serde_json::Value> {
    let budget = reasoning_budget_tokens(sampling.reasoning_budget)?;
    // the runtime passes the budget as an i32, and so does this
    let budget = i32::try_from(budget).unwrap_or(i32::MAX);
    let given = skippy_runtime::chat_sampling_metadata_for_native(chat_metadata?, Some(budget));
    let metadata: serde_json::Value = serde_json::from_str(&given).ok()?;
    let start = metadata.get("thinking_start_tag")?.as_str()?;
    let ends: Vec<&str> = metadata
        .get("thinking_end_tags")?
        .as_array()?
        .iter()
        .filter_map(serde_json::Value::as_str)
        .filter(|end| !end.is_empty())
        .collect();
    if start.is_empty() || ends.is_empty() {
        return None;
    }
    let prefill = metadata
        .get("generation_prompt")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    Some(serde_json::json!({ "start": start, "ends": ends, "prefill": prefill }))
}

/// Whether the runtime replaces the turn's chain by a temperature of 0, which
/// keeps the best logit: sampling is off, or nothing but a temperature of 0
/// shapes the choice. This is `skippy_sampling_is_greedy_equivalent` of the
/// patch's `src/skippy/sampling.cpp`, which `sampling_params` of
/// `src/skippy/sampling_config.cpp` reads; a turn that ignores its end is not
/// one, since the runtime takes the end tokens out of its candidates.
fn runs_greedy_chain(sampling: &skippy_runtime::SamplingConfig) -> bool {
    let repeat = if sampling.repeat_penalty == 0.0 {
        1.0
    } else {
        sampling.repeat_penalty
    };
    !sampling.enabled
        || (sampling.temperature <= 0.0
            && !sampling.ignore_eos
            && sampling.presence_penalty == 0.0
            && sampling.frequency_penalty == 0.0
            && repeat == 1.0
            && sampling.logit_bias.is_empty())
}

/// Whether a turn took the best logit at every token: no temperature,
/// penalty, logit bias, ignored end or grammar changed the choice. The
/// receipt says so, and a client checks greedy turns against the output
/// head.
pub(crate) fn greedy_sampling(
    sampling: &skippy_runtime::SamplingConfig,
    chat_metadata: Option<&str>,
) -> bool {
    let unconstrained = grammar(chat_metadata).is_none();
    unconstrained
        && runs_greedy_chain(sampling)
        && reasoning_budget_tokens(sampling.reasoning_budget).is_none()
}

/// The runtime's default chain, which it runs when a request names none
/// (`sampling_params` of `src/skippy/sampling_config.cpp`): typical-p and
/// top-p only when they are between 0 and 1.
fn default_chain(sampling: &skippy_runtime::SamplingConfig) -> Vec<String> {
    let between = |p: f32| p > 0.0 && p < 1.0;
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
        "typical_p" => between(sampling.typical_p),
        "top_p" => between(sampling.top_p),
        _ => true,
    })
    .map(str::to_owned)
    .collect()
}

/// The chain the runtime runs for the turn: the request's, or its default
/// one.
fn chain_run(sampling: &skippy_runtime::SamplingConfig) -> Vec<String> {
    if sampling.samplers.is_empty() {
        default_chain(sampling)
    } else {
        sampling.samplers.clone()
    }
}

/// Whether a turn's last sampler is the stream draw that the indexed draw
/// replaces: the turn does not take the best logit, and mirostat, which picks
/// its token itself, is off.
pub(crate) fn draws_from_the_chain(
    sampling: &skippy_runtime::SamplingConfig,
    chat_metadata: Option<&str>,
) -> bool {
    !greedy_sampling(sampling, chat_metadata) && sampling.mirostat_mode == 0
}

/// Whether the last sampler of the turn's chain draws at random, as
/// `common_sampler_init` of `common/sampling.cpp` decides: it takes the best
/// logit after a chain that ends in a temperature of 0 that is not dynamic,
/// or in a top-k of 1, and draws after any other. A chain the runtime
/// replaced by a temperature of 0 ([`runs_greedy_chain`]) takes the best
/// logit, also in a turn that is not greedy because a reasoning budget or a
/// grammar constrains it. The runtime installs no indexed draw on a chain
/// that does not draw, so the statement names none.
fn last_sampler_draws(sampling: &skippy_runtime::SamplingConfig) -> bool {
    if runs_greedy_chain(sampling) {
        return false;
    }
    match chain_run(sampling).last().map(String::as_str) {
        Some("temperature") => sampling.temperature > 0.0 || sampling.dynatemp_range > 0.0,
        Some("top_k") => sampling.top_k != 1,
        _ => true,
    }
}

/// Makes the draw of a sampled turn one a client can replay: the indexed
/// draw, with the seed the receipt will state. A request that set no seed
/// takes `client_seed`, the one its receipt request named, else a random
/// one, so the node never states a seed it did not use.
/// A turn with a receipt request draws with the seed the request names: the
/// seed of its body, else the one of its receipt request (`x-receipt-seed`),
/// else an operator's default. `body_seed` is the body's, before the
/// defaults filled it in. A client that set the receipt seed is held to the
/// token that seed picks, so a default must not override it.
pub(crate) fn prefer_receipt_seed(
    mut sampling: skippy_runtime::SamplingConfig,
    body_seed: Option<u64>,
    receipt_seed: Option<u32>,
) -> skippy_runtime::SamplingConfig {
    let asked = receipt_seed.filter(|seed| *seed != 0);
    // a seed of 0 is none: the draw takes the receipt seed later
    if let (None, Some(seed), true) = (body_seed, asked, sampling.seed != 0) {
        sampling.seed = seed;
    }
    sampling
}

pub(crate) fn make_draw_replayable(
    sampling: &mut skippy_runtime::SamplingConfig,
    chat_metadata: Option<&str>,
    client_seed: Option<u32>,
) {
    if !draws_from_the_chain(sampling, chat_metadata) {
        return;
    }
    sampling.indexed_draw = last_sampler_draws(sampling);
    if sampling.seed == 0 {
        sampling.seed = client_seed
            .filter(|seed| *seed != 0)
            .unwrap_or_else(|| rand::RngExt::random_range(&mut rand::rng(), 1..=u32::MAX));
    }
}

/// How a turn chose its tokens, as its receipt states it: [`turn::GREEDY`]
/// when it took the best logit at every token, else the sampler's
/// parameters and seed, so a node that samples otherwise than it says signs
/// a false statement. A grammar is named by its BLAKE3 hash.
/// `draw` names the rule the last sampler drew by when a client can replay
/// it ([`turn::INDEXED_DRAW`]), and [`turn::REASONING`] what the turn's
/// reasoning-budget sampler was built from ([`reasoning_sampler`]).
pub(crate) fn sampler_statement(
    sampling: &skippy_runtime::SamplingConfig,
    chat_metadata: Option<&str>,
) -> String {
    if greedy_sampling(sampling, chat_metadata) {
        return turn::GREEDY.to_owned();
    }
    let grammar =
        grammar(chat_metadata).map(|grammar| blake3::hash(grammar.as_bytes()).to_hex().to_string());
    let s = sampling;
    turn::sampled(&serde_json::json!({
        "seed": s.seed,
        "draw": s.indexed_draw.then_some(turn::INDEXED_DRAW),
        // what the runtime ran: its default chain for a request that named
        // none, and a temperature of 0 where sampling is off
        "samplers": chain_run(s),
        "temperature": if s.enabled { s.temperature } else { 0.0 },
        "dynatemp_range": s.dynatemp_range,
        "dynatemp_exponent": s.dynatemp_exponent,
        "top_k": s.top_k,
        "top_p": s.top_p,
        "min_p": s.min_p,
        "typical_p": s.typical_p,
        "top_nsigma": s.top_nsigma,
        "presence_penalty": s.presence_penalty,
        "frequency_penalty": s.frequency_penalty,
        "repeat_penalty": s.repeat_penalty,
        "penalty_last_n": s.penalty_last_n,
        "dry": {
            "multiplier": s.dry.multiplier,
            "base": s.dry.base,
            "allowed_length": s.dry.allowed_length,
            "penalty_last_n": s.dry.penalty_last_n,
            "sequence_breakers": s.dry.sequence_breakers,
        },
        "xtc": { "probability": s.xtc.probability, "threshold": s.xtc.threshold },
        "mirostat": {
            "mode": s.mirostat_mode,
            "entropy": s.mirostat_entropy,
            "learning_rate": s.mirostat_learning_rate,
        },
        "logit_bias": s.logit_bias.iter().map(|b| (b.token_id, b.bias)).collect::<Vec<_>>(),
        "ignore_eos": s.ignore_eos,
        "grammar": grammar,
        "reasoning_budget": reasoning_budget_tokens(s.reasoning_budget),
        (turn::REASONING): reasoning_sampler(s, chat_metadata),
    }))
}

/// The tokens a reasoning budget allows, `None` when it is unrestricted. A
/// budget can force the end of reasoning, so it is part of the statement.
fn reasoning_budget_tokens(budget: skippy_runtime::ReasoningBudget) -> Option<i64> {
    use skippy_runtime::ReasoningBudget;
    match budget {
        ReasoningBudget::Unrestricted => None,
        ReasoningBudget::Explicit(tokens) | ReasoningBudget::Capped(tokens) => {
            Some(i64::from(tokens))
        }
        ReasoningBudget::Resolved(tokens) => (tokens >= 0).then_some(i64::from(tokens)),
    }
}

#[cfg(test)]
mod tests {
    use skippy_runtime::{LogitBias, SamplingConfig};

    use super::*;

    #[test]
    fn only_a_turn_that_takes_the_best_logit_is_greedy() {
        let disabled = SamplingConfig::default();
        assert!(greedy_sampling(&disabled, None));
        let mut zero = SamplingConfig {
            enabled: true,
            temperature: 0.0,
            ..SamplingConfig::default()
        };
        assert!(greedy_sampling(&zero, Some(r#"{"grammar":""}"#)));
        assert!(!greedy_sampling(
            &zero,
            Some(r#"{"grammar":"root ::= \"a\""}"#)
        ));
        zero.logit_bias.push(LogitBias {
            token_id: 5,
            bias: 1.0,
        });
        assert!(!greedy_sampling(&zero, None));
        let warm = SamplingConfig {
            enabled: true,
            temperature: 0.7,
            ..SamplingConfig::default()
        };
        assert!(!greedy_sampling(&warm, None));
    }

    #[test]
    fn a_sampled_turn_states_its_parameters_and_seed() {
        assert_eq!(
            sampler_statement(&SamplingConfig::default(), None),
            turn::GREEDY
        );
        let warm = SamplingConfig {
            enabled: true,
            temperature: 0.7,
            seed: 1234,
            ..SamplingConfig::default()
        };
        let grammar = Some(r#"{"grammar":"root ::= \"a\""}"#);
        let statement = sampler_statement(&warm, grammar);
        let parameters = turn::sampler_parameters(&statement).expect("parameters");
        assert_eq!(parameters["seed"], 1234);
        assert!((parameters["temperature"].as_f64().unwrap() - 0.7).abs() < 1e-6);
        assert_eq!(parameters["grammar"].as_str().map(str::len), Some(64));
        let other = SamplingConfig { seed: 99, ..warm };
        assert_ne!(sampler_statement(&other, grammar), statement);
    }

    fn warm() -> SamplingConfig {
        SamplingConfig {
            enabled: true,
            temperature: 0.7,
            ..SamplingConfig::default()
        }
    }

    #[test]
    fn a_sampled_turn_with_a_receipt_draws_by_the_indexed_draw() {
        // the request set a seed: it is the one stated and used
        let mut seeded = SamplingConfig {
            seed: 1234,
            ..warm()
        };
        make_draw_replayable(&mut seeded, None, Some(99));
        assert!(seeded.indexed_draw);
        assert_eq!(seeded.seed, 1234);
        let parameters = turn::sampler_parameters(&sampler_statement(&seeded, None)).unwrap();
        assert_eq!(parameters["draw"], turn::INDEXED_DRAW);
        assert_eq!(parameters["seed"], 1234);
        // no seed in the request: the receipt request's seed
        let mut asked = warm();
        make_draw_replayable(&mut asked, None, Some(99));
        assert_eq!((asked.indexed_draw, asked.seed), (true, 99));
        // none there either, or 0: a random one above 0, which is stated
        for client_seed in [None, Some(0)] {
            let mut unseeded = warm();
            make_draw_replayable(&mut unseeded, None, client_seed);
            assert!(unseeded.indexed_draw);
            assert_ne!(unseeded.seed, 0);
            let parameters = turn::sampler_parameters(&sampler_statement(&unseeded, None)).unwrap();
            assert_eq!(parameters["seed"], unseeded.seed);
        }
    }

    #[test]
    fn a_turn_that_takes_the_best_logit_under_a_budget_states_no_draw() {
        // a request that turns every filter off at a temperature of 1 is
        // decoded greedily; a reasoning budget makes its statement a
        // sampled one, which must not name a draw the chain never made
        let disabled = SamplingConfig {
            enabled: false,
            temperature: 1.0,
            ..budgeted(16)
        };
        let cold = SamplingConfig {
            temperature: 0.0,
            ..budgeted(16)
        };
        for mut best in [disabled, cold] {
            make_draw_replayable(&mut best, None, Some(99));
            assert!(!best.indexed_draw);
            // the receipt request's seed is still the one stated
            assert_eq!(best.seed, 99);
            let parameters = turn::sampler_parameters(&sampler_statement(&best, None)).unwrap();
            assert!(parameters["draw"].is_null(), "{parameters}");
            assert_eq!(parameters["reasoning_budget"], 16);
        }
        // a budgeted turn that samples does name it
        let mut warm = SamplingConfig {
            temperature: 0.7,
            ..budgeted(16)
        };
        make_draw_replayable(&mut warm, None, Some(99));
        assert!(warm.indexed_draw);
    }

    #[test]
    fn a_turn_that_does_not_draw_from_the_chain_is_left_alone() {
        // a greedy turn has no draw
        let mut greedy = SamplingConfig::default();
        make_draw_replayable(&mut greedy, None, Some(99));
        assert_eq!(greedy, SamplingConfig::default());
        // mirostat picks its token itself
        let mut mirostat = SamplingConfig {
            mirostat_mode: 2,
            ..warm()
        };
        make_draw_replayable(&mut mirostat, None, Some(99));
        assert!(!mirostat.indexed_draw);
        assert_eq!(mirostat.seed, 0);
        // a turn without a receipt states no draw
        let parameters = turn::sampler_parameters(&sampler_statement(&warm(), None)).unwrap();
        assert!(parameters["draw"].is_null());
    }

    // The runtime configures the sampler of a turn with a grammar from the
    // whole metadata, and so builds its reasoning-budget sampler.
    const THINKING: &str = r#"{"grammar":"root ::= \"a\"","generation_prompt":"<|im_start|>assistant\n","thinking_start_tag":"<think>","thinking_end_tags":["</think>","",7],"supports_thinking":true}"#;

    fn budgeted(tokens: i32) -> SamplingConfig {
        SamplingConfig {
            reasoning_budget: skippy_runtime::ReasoningBudget::Resolved(tokens),
            ..warm()
        }
    }

    #[test]
    fn a_turn_with_a_reasoning_budget_states_what_its_sampler_was_built_from() {
        let statement = sampler_statement(&budgeted(256), Some(THINKING));
        let parameters = turn::sampler_parameters(&statement).unwrap();
        assert_eq!(parameters["reasoning_budget"], 256);
        // the tags as the runtime reads them: an end tag that is no text,
        // or empty, builds nothing
        assert_eq!(
            parameters[turn::REASONING],
            serde_json::json!({
                "start": "<think>",
                "ends": ["</think>"],
                "prefill": "<|im_start|>assistant\n",
            })
        );
        // a greedy request with a budget is a sampled statement too
        let greedy = SamplingConfig {
            temperature: 0.0,
            ..budgeted(0)
        };
        let parameters =
            turn::sampler_parameters(&sampler_statement(&greedy, Some(THINKING))).unwrap();
        assert_eq!(parameters[turn::REASONING]["start"], "<think>");
    }

    #[test]
    fn a_turn_whose_runtime_builds_no_reasoning_sampler_states_none() {
        let stated = |sampling: &SamplingConfig, metadata: Option<&str>| {
            turn::sampler_parameters(&sampler_statement(sampling, metadata)).unwrap()
                [turn::REASONING]
                .clone()
        };
        // no budget
        assert!(stated(&budgeted(-1), Some(THINKING)).is_null());
        assert!(stated(&warm(), Some(THINKING)).is_null());
        // a budget, and metadata that names no tags, or no metadata
        assert!(stated(&budgeted(64), Some(r#"{"grammar":""}"#)).is_null());
        assert!(stated(&budgeted(64), None).is_null());
        let no_start = r#"{"thinking_start_tag":"","thinking_end_tags":["</think>"]}"#;
        assert!(stated(&budgeted(64), Some(no_start)).is_null());
        let no_end = r#"{"thinking_start_tag":"<think>","thinking_end_tags":[""]}"#;
        assert!(stated(&budgeted(64), Some(no_end)).is_null());
    }

    #[test]
    fn a_turn_without_a_grammar_states_the_reasoning_sampler_the_runtime_builds() {
        // The statement follows what the runtime gives native sampling. A
        // statement that named a sampler the runtime did not build made a
        // client expect the end tag where the budget ran out, and fail an
        // honest turn that reasoned on. One that left out a sampler the
        // runtime did build would fail the honest turn it cut short.
        let open = r#"{"grammar":"","generation_prompt":"<|im_start|>assistant\n","thinking_start_tag":"<think>","thinking_end_tags":["</think>"],"supports_thinking":true}"#;
        let given = skippy_runtime::chat_sampling_metadata_for_native(open, Some(16));
        assert!(given.contains("thinking_start_tag"), "{given}");
        let parameters =
            turn::sampler_parameters(&sampler_statement(&budgeted(16), Some(open))).unwrap();
        assert_eq!(
            parameters[turn::REASONING],
            serde_json::json!({
                "start": "<think>",
                "ends": ["</think>"],
                "prefill": "<|im_start|>assistant\n",
            })
        );
        assert_eq!(parameters["reasoning_budget"], 16);

        // the template closed the reasoning block: the runtime builds no
        // budget sampler, and the statement names none
        let closed = r#"{"grammar":"","generation_prompt":"<|im_start|>assistant\n<think>\n\n</think>\n\n","thinking_start_tag":"<think>","thinking_end_tags":["</think>"],"supports_thinking":true}"#;
        assert_eq!(
            skippy_runtime::chat_sampling_metadata_for_native(closed, Some(16)),
            r#"{"grammar":""}"#
        );
        let parameters =
            turn::sampler_parameters(&sampler_statement(&budgeted(16), Some(closed))).unwrap();
        assert!(parameters[turn::REASONING].is_null(), "{parameters}");
        // the budget the request resolved to is still stated
        assert_eq!(parameters["reasoning_budget"], 16);
    }

    #[test]
    fn a_turn_that_ignores_its_end_is_not_greedy() {
        // the runtime takes the end tokens out of its candidates, so the
        // best logit may not be the token it took
        let cold = SamplingConfig {
            enabled: true,
            temperature: 0.0,
            ignore_eos: true,
            ..SamplingConfig::default()
        };
        assert!(!greedy_sampling(&cold, None));
        let mut drawn = cold.clone();
        make_draw_replayable(&mut drawn, None, Some(99));
        // a temperature of 0 at the end of the chain takes the best logit
        assert!(!drawn.indexed_draw);
        let parameters = turn::sampler_parameters(&sampler_statement(&drawn, None)).unwrap();
        assert_eq!(parameters["ignore_eos"], true);
        assert!(parameters["draw"].is_null());
    }

    #[test]
    fn a_turn_states_the_chain_the_runtime_ran() {
        // a request that names an empty chain runs the default one
        let empty = SamplingConfig {
            samplers: Vec::new(),
            top_p: 1.0,
            ..warm()
        };
        let parameters = turn::sampler_parameters(&sampler_statement(&empty, None)).unwrap();
        assert_eq!(
            parameters["samplers"],
            serde_json::json!([
                "penalties",
                "dry",
                "top_n_sigma",
                "top_k",
                "min_p",
                "xtc",
                "temperature"
            ])
        );
        // a turn with sampling off ran at a temperature of 0
        let off = SamplingConfig {
            enabled: false,
            temperature: 1.0,
            ..budgeted(16)
        };
        let parameters = turn::sampler_parameters(&sampler_statement(&off, None)).unwrap();
        assert_eq!(parameters["temperature"], 0.0);
    }

    #[test]
    fn the_last_sampler_draws_as_the_runtime_decides() {
        let with = |samplers: &[&str], temperature: f32, top_k: i32| SamplingConfig {
            samplers: samplers.iter().map(|s| (*s).to_owned()).collect(),
            temperature,
            top_k,
            logit_bias: vec![skippy_runtime::LogitBias {
                token_id: 3,
                bias: 1.0,
            }],
            ..warm()
        };
        assert!(last_sampler_draws(&with(
            &["top_k", "temperature"],
            0.7,
            40
        )));
        assert!(!last_sampler_draws(&with(
            &["top_k", "temperature"],
            0.0,
            40
        )));
        // a chain that ends in a top-k of 1 takes the best logit
        assert!(!last_sampler_draws(&with(
            &["temperature", "top_k"],
            0.7,
            1
        )));
        // a chain without a temperature draws at any stated temperature
        assert!(last_sampler_draws(&with(&["top_k"], 0.0, 40)));
        // a dynamic temperature draws at a base of 0
        let dynamic = SamplingConfig {
            dynatemp_range: 0.5,
            ..with(&["top_k", "temperature"], 0.0, 40)
        };
        assert!(last_sampler_draws(&dynamic));
    }

    #[test]
    fn the_receipt_seed_beats_a_default_and_the_body_beats_both() {
        let defaulted = SamplingConfig { seed: 5, ..warm() };
        // the operator's default seed 5 filled in a body without one
        assert_eq!(
            prefer_receipt_seed(defaulted.clone(), None, Some(99)).seed,
            99
        );
        // the body named 5 itself
        assert_eq!(
            prefer_receipt_seed(defaulted.clone(), Some(5), Some(99)).seed,
            5
        );
        // no receipt seed, or seed 0, leaves the default
        assert_eq!(prefer_receipt_seed(defaulted.clone(), None, None).seed, 5);
        assert_eq!(prefer_receipt_seed(defaulted, None, Some(0)).seed, 5);
        // a turn without any seed takes the receipt's when it draws
        let mut unseeded = prefer_receipt_seed(warm(), None, Some(99));
        assert_eq!(unseeded.seed, 0);
        make_draw_replayable(&mut unseeded, None, Some(99));
        assert_eq!(unseeded.seed, 99);
    }
}
