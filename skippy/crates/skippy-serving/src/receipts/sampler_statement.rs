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
/// `src/skippy/sampling_config.cpp`). The runtime configures a turn
/// without a grammar with the grammar field alone
/// (`skippy_runtime::chat_sampling_metadata_for_native`), so such a turn
/// has no reasoning-budget sampler and no token of it is forced, whatever
/// budget it states.
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
    reasoning_budget_tokens(sampling.reasoning_budget)?;
    let given = skippy_runtime::chat_sampling_metadata_for_native(chat_metadata?);
    let metadata: serde_json::Value = serde_json::from_str(given).ok()?;
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

/// Whether a turn took the best logit at every token: no temperature,
/// penalty, logit bias or grammar changed the choice. The receipt says so,
/// and a client checks greedy turns against the output head.
pub(crate) fn greedy_sampling(
    sampling: &skippy_runtime::SamplingConfig,
    chat_metadata: Option<&str>,
) -> bool {
    let unconstrained = grammar(chat_metadata).is_none();
    let best_logit = !sampling.enabled
        || (sampling.temperature <= 0.0
            && sampling.presence_penalty == 0.0
            && sampling.frequency_penalty == 0.0
            && sampling.repeat_penalty == 1.0
            && sampling.dry.multiplier == 0.0
            && sampling.logit_bias.is_empty());
    unconstrained && best_logit && reasoning_budget_tokens(sampling.reasoning_budget).is_none()
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

/// Whether the last sampler of the turn's chain draws at random. A disabled
/// sampler and a temperature of 0 take the best logit, also in a turn that
/// is not greedy because a reasoning budget or a grammar constrains it. The
/// runtime installs no indexed draw on such a chain, so the statement names
/// none: a client that replayed a draw there would fail an honest turn.
fn last_sampler_draws(sampling: &skippy_runtime::SamplingConfig) -> bool {
    sampling.enabled && sampling.temperature > 0.0
}

/// Makes the draw of a sampled turn one a client can replay: the indexed
/// draw, with the seed the receipt will state. A request that set no seed
/// takes `client_seed`, the one its receipt request named, else a random
/// one, so the node never states a seed it did not use.
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
        "samplers": s.samplers,
        "temperature": s.temperature,
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
    fn a_turn_without_a_grammar_states_no_reasoning_sampler() {
        // The runtime configures such a turn with the grammar field alone,
        // so it builds no reasoning-budget sampler and forces no token. A
        // statement that named one made a client expect the end tag where
        // the budget ran out, and fail an honest turn that reasoned on.
        let plain = r#"{"grammar":"","generation_prompt":"<|im_start|>assistant\n","thinking_start_tag":"<think>","thinking_end_tags":["</think>"],"supports_thinking":true}"#;
        assert_eq!(
            skippy_runtime::chat_sampling_metadata_for_native(plain),
            r#"{"grammar":""}"#
        );
        let parameters =
            turn::sampler_parameters(&sampler_statement(&budgeted(16), Some(plain))).unwrap();
        assert!(parameters[turn::REASONING].is_null(), "{parameters}");
        // the budget the request resolved to is still stated
        assert_eq!(parameters["reasoning_budget"], 16);
    }
}
