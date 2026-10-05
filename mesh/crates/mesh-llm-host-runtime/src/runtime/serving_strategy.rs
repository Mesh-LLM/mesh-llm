//! Compose a serving intent into the settings it needs.
//!
//! Every performance feature for splits shipped as its own flag, so reaching a
//! measured operating point meant knowing which of a dozen to set — and two of
//! them were environment variables that appeared nowhere outside Rust source.
//! `--strategy` names the intent instead and fills in what it implies.
//!
//! Two rules keep this safe to default on:
//!
//! 1. **It only fills in values nobody has stated.** Every write checks for
//!    `None` first, so an explicit config-file value survives, and the CLI
//!    mechanism overrides that run after this (`apply_runtime_cli_*`) win
//!    outright. Adopting a strategy cannot change an existing deployment.
//!
//!    Everything it writes lands in `[defaults]`, never on a model, so a
//!    `[models.*]` block still wins at resolution. The report names the full
//!    `defaults.*` path for that reason: an unscoped axis name would read as
//!    though the effective per-model value had changed. The global default is
//!    still written when some model overrides it, because the models that do
//!    not override it need it.
//! 2. **It composes only axes that exist.** The remaining ones in #2112 —
//!    min-sum placement, the RTT-derived run-ahead budget, the speculation
//!    break-even gate — are not wired yet, and a strategy that silently
//!    promised them would be the same defect this flag exists to remove. The
//!    report says which axes it actually set.

use crate::plugin;
use mesh_llm_config::{BoolOrAuto, SpeculativeConfig, ThroughputConfig};

/// What the deployment is optimising for.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ServingStrategy {
    /// Today's defaults, unchanged.
    Balanced,
    /// Single-stream and agentic work: speculation amortises the split hop.
    Interactive,
    /// Fleet tokens per second, accepting higher per-request latency.
    Throughput,
}

impl ServingStrategy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Balanced => "balanced",
            Self::Interactive => "interactive",
            Self::Throughput => "throughput",
        }
    }
}

/// One axis the strategy decided, and why it decided it that way.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StrategyDecision {
    pub axis: &'static str,
    pub value: String,
    pub because: &'static str,
}

/// What a strategy did, for the startup log and `doctor split`.
///
/// `declined` records an axis the strategy wanted but did not take, because
/// something more specific had already stated it. Surfacing those is the point:
/// an operator who sets one flag by hand should be able to see that it survived
/// rather than guess.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StrategyPlan {
    pub strategy: Option<&'static str>,
    pub applied: Vec<StrategyDecision>,
    pub declined: Vec<StrategyDecision>,
}

impl StrategyPlan {
    fn note_applied(
        &mut self,
        axis: &'static str,
        value: impl Into<String>,
        because: &'static str,
    ) {
        self.applied.push(StrategyDecision {
            axis,
            value: value.into(),
            because,
        });
    }

    fn note_declined(&mut self, axis: &'static str, because: &'static str) {
        self.declined.push(StrategyDecision {
            axis,
            value: "kept your value".to_string(),
            because,
        });
    }
}

/// Whether the split-only axes can be composed at all.
///
/// A strategy must not turn a single-node deployment into a split one: that
/// changes where the model runs, which is a much bigger decision than an
/// optimisation intent should make on an operator's behalf.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StrategyContext {
    pub split: bool,
    /// Already requested on the command line, so the strategy leaves it alone.
    pub auto_balance_requested: bool,
}

/// Fill in what the strategy implies, and report what it did.
///
/// Returns the plan rather than logging here so the caller can log it once
/// alongside the rest of its startup state, and so this stays testable without
/// capturing a subscriber.
pub(in crate::runtime) fn apply_serving_strategy(
    config: &mut plugin::MeshConfig,
    strategy: Option<ServingStrategy>,
    context: StrategyContext,
) -> StrategyPlan {
    let mut plan = StrategyPlan::default();
    let Some(strategy) = strategy else {
        return plan;
    };
    plan.strategy = Some(strategy.as_str());

    match strategy {
        // Explicitly a no-op. It exists so the default is nameable, and so
        // `--strategy balanced` is a way to say "do not compose anything" that
        // survives a future change to the default.
        ServingStrategy::Balanced => {
            plan.note_applied(
                "all",
                "unchanged",
                "balanced is today's defaults; nothing is composed",
            );
        }
        ServingStrategy::Interactive => compose_interactive(config, &mut plan),
        ServingStrategy::Throughput => compose_throughput(config, &mut plan, context),
    }

    plan
}

/// Speculation carries the split hop.
///
/// #1409 measured run-ahead admission at 108.3 tok/s against 99.6 for the best
/// fixed depth, and it beat every fixed depth under jitter too, so the budget
/// is the admission policy to reach for rather than a depth. Workload caveat
/// from #1037 and #1581: the N-gram proposers pay off on input-grounded output
/// and can lose on freeform prose. Picking them here is a statement about the
/// *workload the operator named*, not a claim they always win — the measured
/// gate that would decide it per-request is #2112 workstream 5.
fn compose_interactive(config: &mut plugin::MeshConfig, plan: &mut StrategyPlan) {
    let speculative = speculative_defaults(config);

    if speculative.strategy.is_none() {
        speculative.strategy = Some("ngram-suffix".to_string());
        plan.note_applied(
            "defaults.speculative.strategy",
            "ngram-suffix",
            "input-grounded output is mostly a copy of the context, which the suffix matcher drafts in long spans",
        );
    } else {
        plan.note_declined("defaults.speculative.strategy", "you set it explicitly");
    }

    if speculative.verify_window_runahead_tokens.is_none() {
        speculative.verify_window_runahead_tokens = Some(RUNAHEAD_TOKENS);
        plan.note_applied(
            "defaults.speculative.verify_window_runahead_tokens",
            RUNAHEAD_TOKENS.to_string(),
            "run-ahead admission beat every fixed depth in #1409, clean and jittered",
        );
    } else {
        plan.note_declined(
            "defaults.speculative.verify_window_runahead_tokens",
            "you set it explicitly",
        );
    }

    // Batching the final stage would suppress the drafts this strategy depends
    // on, so state it rather than leave the two to collide.
    let throughput = throughput_defaults(config);
    if throughput.last_stage_decode_batch.is_none() {
        throughput.last_stage_decode_batch = Some(BoolOrAuto::Bool(false));
        plan.note_applied(
            "defaults.throughput.last_stage_decode_batch",
            "false",
            "the batched final stage produces no native MTP drafts, so it cannot coexist with speculation",
        );
    } else {
        plan.note_declined(
            "defaults.throughput.last_stage_decode_batch",
            "you set it explicitly",
        );
    }
}

/// Fleet tokens per second.
///
/// #1935's hour-long runs on the two-mini lab: the speed-balanced cut alone
/// measured 15.90 tok/s against a stock split's 20.6, and only reached 29.16
/// once the final stage batched decode across lanes and the wave was split into
/// groups. Neither half pays off alone, which is the whole reason this is one
/// named strategy rather than three flags.
fn compose_throughput(
    config: &mut plugin::MeshConfig,
    plan: &mut StrategyPlan,
    context: StrategyContext,
) {
    let throughput = throughput_defaults(config);

    if throughput.last_stage_decode_batch.is_none() {
        throughput.last_stage_decode_batch = Some(BoolOrAuto::Bool(true));
        plan.note_applied(
            "defaults.throughput.last_stage_decode_batch",
            "true",
            "moving layers onto the last stage only helps once it batches decode across lanes (#1935)",
        );
    } else {
        plan.note_declined(
            "defaults.throughput.last_stage_decode_batch",
            "you set it explicitly",
        );
    }

    if throughput.pipeline_decode_groups.is_none() {
        throughput.pipeline_decode_groups = Some(DECODE_GROUPS);
        plan.note_applied(
            "defaults.throughput.pipeline_decode_groups",
            DECODE_GROUPS.to_string(),
            "keeps more than one batch in flight across the pipeline; 4 lanes / 2 groups is the measured arm",
        );
    } else {
        plan.note_declined(
            "defaults.throughput.pipeline_decode_groups",
            "you set it explicitly",
        );
    }

    // Speculation and last-stage batching are mutually exclusive by
    // construction, not by preference: the batched path emits no MTP drafts.
    // Under concurrency the lanes already hide the hop, so batching is the
    // better half of the trade here.
    let speculative = speculative_defaults(config);
    if speculative.strategy.is_none() {
        speculative.strategy = Some("disabled".to_string());
        plan.note_applied(
            "defaults.speculative.strategy",
            "disabled",
            "cannot coexist with the batched final stage this strategy needs",
        );
    } else {
        plan.note_declined("defaults.speculative.strategy", "you set it explicitly");
    }

    if context.auto_balance_requested {
        plan.note_declined("auto_balance", "already requested on the command line");
    } else if !context.split {
        plan.note_declined(
            "auto_balance",
            "needs --split; a strategy will not turn a single-node deployment into a split one",
        );
    }
}

/// Run-ahead speculative-token budget for `interactive`.
///
/// 96 is the budget #1409 measured at 108.3 tok/s; the native
/// checkpoint-retention bound caps the window count above it.
const RUNAHEAD_TOKENS: u32 = 96;

/// Decode-wave groups for `throughput`. 4 lanes in 2 groups is #1935's arm; its
/// own 6 lanes / 3 groups measurement was slower, at 27.66.
const DECODE_GROUPS: u32 = 2;

fn throughput_defaults(config: &mut plugin::MeshConfig) -> &mut ThroughputConfig {
    config
        .defaults
        .get_or_insert_with(plugin::ModelConfigDefaults::default)
        .throughput
        .get_or_insert_with(ThroughputConfig::default)
}

fn speculative_defaults(config: &mut plugin::MeshConfig) -> &mut SpeculativeConfig {
    config
        .defaults
        .get_or_insert_with(plugin::ModelConfigDefaults::default)
        .speculative
        .get_or_insert_with(SpeculativeConfig::default)
}

/// Whether the strategy wants closed-loop rebalancing turned on.
///
/// Kept separate from [`apply_serving_strategy`] because `auto_balance` is a
/// launch option rather than a config key, so the caller owns it.
pub fn strategy_requests_auto_balance(
    strategy: Option<ServingStrategy>,
    context: StrategyContext,
) -> bool {
    matches!(strategy, Some(ServingStrategy::Throughput)) && context.split
}

/// Log what the strategy composed, so an operator can see the settings a single
/// flag turned into — and, just as importantly, which of their own values it
/// left alone.
pub(in crate::runtime) fn log_strategy_plan(plan: &StrategyPlan) {
    let Some(strategy) = plan.strategy else {
        return;
    };
    for decision in &plan.applied {
        tracing::info!(
            strategy,
            axis = decision.axis,
            value = %decision.value,
            because = decision.because,
            "strategy set a global default"
        );
    }
    for decision in &plan.declined {
        tracing::info!(
            strategy,
            axis = decision.axis,
            because = decision.because,
            "strategy deferred to your setting"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context(split: bool) -> StrategyContext {
        StrategyContext {
            split,
            auto_balance_requested: false,
        }
    }

    fn defaults(config: &plugin::MeshConfig) -> (&ThroughputConfig, &SpeculativeConfig) {
        let defaults = config.defaults.as_ref().expect("defaults");
        (
            defaults.throughput.as_ref().expect("throughput"),
            defaults.speculative.as_ref().expect("speculative"),
        )
    }

    /// `MeshConfig` has no `PartialEq`, and comparing the rendered TOML is the
    /// stronger check anyway: it catches a default materialised into an empty
    /// table, which a field-by-field assertion would miss.
    fn rendered(config: &plugin::MeshConfig) -> String {
        mesh_llm_config::config_to_toml(config).expect("render config")
    }

    #[test]
    fn no_strategy_touches_nothing() {
        let mut config = plugin::MeshConfig::default();
        let before = rendered(&config);

        let plan = apply_serving_strategy(&mut config, None, context(true));

        assert_eq!(plan, StrategyPlan::default());
        assert_eq!(rendered(&config), before);
    }

    /// The parity gate from #2112: naming the default must be a no-op.
    #[test]
    fn balanced_leaves_the_configuration_identical() {
        let mut config = plugin::MeshConfig::default();
        let before = rendered(&config);

        let plan =
            apply_serving_strategy(&mut config, Some(ServingStrategy::Balanced), context(true));

        assert_eq!(rendered(&config), before);
        assert_eq!(plan.strategy, Some("balanced"));
        assert!(plan.declined.is_empty());
    }

    #[test]
    fn throughput_batches_the_last_stage_and_groups_the_wave() {
        let mut config = plugin::MeshConfig::default();
        apply_serving_strategy(
            &mut config,
            Some(ServingStrategy::Throughput),
            context(true),
        );

        let (throughput, speculative) = defaults(&config);
        assert_eq!(
            throughput.last_stage_decode_batch,
            Some(BoolOrAuto::Bool(true))
        );
        assert_eq!(throughput.pipeline_decode_groups, Some(2));
        // The pairing is the point: batching suppresses MTP drafts, so leaving
        // speculation on would silently cancel the batching.
        assert_eq!(speculative.strategy.as_deref(), Some("disabled"));
    }

    #[test]
    fn interactive_speculates_with_a_runahead_budget_and_no_batching() {
        let mut config = plugin::MeshConfig::default();
        apply_serving_strategy(
            &mut config,
            Some(ServingStrategy::Interactive),
            context(true),
        );

        let (throughput, speculative) = defaults(&config);
        assert_eq!(speculative.strategy.as_deref(), Some("ngram-suffix"));
        assert_eq!(speculative.verify_window_runahead_tokens, Some(96));
        assert_eq!(
            throughput.last_stage_decode_batch,
            Some(BoolOrAuto::Bool(false))
        );
    }

    /// The precedence rule that makes this safe to ship on by default.
    #[test]
    fn an_explicit_value_survives_the_strategy_and_is_reported() {
        let mut config = plugin::MeshConfig::default();
        config
            .defaults
            .get_or_insert_with(plugin::ModelConfigDefaults::default)
            .throughput
            .get_or_insert_with(ThroughputConfig::default)
            .pipeline_decode_groups = Some(8);

        let plan = apply_serving_strategy(
            &mut config,
            Some(ServingStrategy::Throughput),
            context(true),
        );

        let (throughput, _) = defaults(&config);
        assert_eq!(throughput.pipeline_decode_groups, Some(8));
        assert!(
            plan.declined
                .iter()
                .any(|decision| decision.axis == "defaults.throughput.pipeline_decode_groups"),
            "a kept value must be reported, not silently overridden: {plan:?}"
        );
        assert!(
            !plan
                .applied
                .iter()
                .any(|decision| decision.axis == "defaults.throughput.pipeline_decode_groups")
        );
    }

    /// Everything is written to `[defaults]`, so a model block still wins. The
    /// global default is written anyway — the models that do not override it
    /// need it — and the report says `defaults.*` so the log cannot be read as
    /// a claim about a model's effective value.
    #[test]
    fn a_model_override_is_untouched_and_the_report_names_the_defaults_scope() {
        let mut config = plugin::MeshConfig::default();
        config.models.push(plugin::ModelConfigEntry {
            model: "Qwen/Qwen3-0.6B:Q4_K_M".to_string(),
            speculative: Some({
                // `SpeculativeConfig` has a private field, so build it by
                // mutation rather than a struct literal with a spread.
                let mut speculative = SpeculativeConfig::default();
                speculative.strategy = Some("mtp".to_string());
                speculative
            }),
            ..plugin::ModelConfigEntry::default()
        });

        let plan = apply_serving_strategy(
            &mut config,
            Some(ServingStrategy::Interactive),
            context(true),
        );

        // The model keeps its own choice.
        assert_eq!(
            config.models[0]
                .speculative
                .as_ref()
                .and_then(|speculative| speculative.strategy.as_deref()),
            Some("mtp")
        );
        // And the global default is still written, for models without one.
        let (_, speculative) = defaults(&config);
        assert_eq!(speculative.strategy.as_deref(), Some("ngram-suffix"));

        let decision = plan
            .applied
            .iter()
            .find(|decision| decision.axis.ends_with("speculative.strategy"))
            .expect("the speculative strategy must be reported");
        assert_eq!(
            decision.axis, "defaults.speculative.strategy",
            "an unscoped axis would read as a per-model claim"
        );
    }

    #[test]
    fn throughput_requests_auto_balance_only_on_a_split() {
        assert!(strategy_requests_auto_balance(
            Some(ServingStrategy::Throughput),
            context(true)
        ));
        assert!(!strategy_requests_auto_balance(
            Some(ServingStrategy::Throughput),
            context(false)
        ));
        assert!(!strategy_requests_auto_balance(
            Some(ServingStrategy::Interactive),
            context(true)
        ));
        assert!(!strategy_requests_auto_balance(None, context(true)));
    }

    /// A single-node deployment must not be silently turned into a split one.
    #[test]
    fn throughput_without_split_says_why_it_declined_auto_balance() {
        let mut config = plugin::MeshConfig::default();
        let plan = apply_serving_strategy(
            &mut config,
            Some(ServingStrategy::Throughput),
            context(false),
        );

        let declined = plan
            .declined
            .iter()
            .find(|decision| decision.axis == "auto_balance")
            .expect("auto_balance must be reported as declined");
        assert!(declined.because.contains("--split"), "{declined:?}");
    }
}
