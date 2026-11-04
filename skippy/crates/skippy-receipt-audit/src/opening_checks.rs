//! Every check that needs the opened values, on an opening that was
//! verified against the receipt and validated against the claimed model:
//! each picked layer's glue and weight rows, the input of layer 0, the
//! output head, the tokens, and the recurrent regions. An audit and a
//! recheck of its evidence share this.
//!
//! No check here stops the others. Each one's result is a verdict, a
//! defect of the opening, or a reason it could not run on this machine
//! ([`Check`]); the opening is already known to be whole, so what can still
//! stop a check is the client's own: a weight read, a layer that does not
//! load.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use skippy_receipts::attention::AttentionDescriptor;
use skippy_receipts::audit::{self, AuditReport, Routers};
use skippy_receipts::cache_types;
use skippy_receipts::claimed_reads;
use skippy_receipts::glue::{self, GlueReport};
use skippy_receipts::head::reasoning_budget::{self, Vocabulary};
use skippy_receipts::head::sampling::TokenRule;
use skippy_receipts::head::{self, HeadVerdict};
use skippy_receipts::opened_values::ValidatedOpening;
use skippy_receipts::weights::WeightFile;
use skippy_receipts::wire::{LayerOpening, MixingKind};

use crate::chat_reply::ChatReply;
use crate::claimed_model::{CacheTypes, ClaimedModel};
use crate::outcome::Check;
use crate::output::{
    AuditOutput, CostOutput, GlueOutput, HeadOutput, LayerOutput, MatrixOutput, ReasoningOutput,
    RoleOutput,
};
use crate::recurrent_audit::{self, KnownLayers};
use crate::{head_key_file, token_check};

/// What the checks of an opening need beyond it.
pub(crate) struct Checks<'a> {
    pub(crate) timing_runs: u32,
    /// The prompt and the response, when the auditor received them.
    pub(crate) response: Option<(&'a str, &'a ChatReply)>,
    /// The client's head key file, which checks the node's logits. A
    /// recheck has none, and checks them against every row of the output
    /// matrix.
    pub(crate) head_key: Option<PathBuf>,
}

/// Each picked layer's attention ops, as the auditor's graph describes
/// them, for the layers whose glue ran.
pub(crate) type Layouts = BTreeMap<u32, Vec<AttentionDescriptor>>;

/// Runs every check of the opened values and records its result in
/// `output`. Returns the attention ops of the layers whose glue ran.
pub(crate) fn check_opening(
    checks: &Checks<'_>,
    model: &dyn ClaimedModel,
    weights: &WeightFile,
    opening: &ValidatedOpening<'_>,
    rows_seed: &[u8],
    output: &mut AuditOutput,
) -> Layouts {
    let picks = &opening.challenge().picks;
    // the layers load whole: what they read must be here first
    let layers: Vec<u32> = picks.iter().map(|p| p.layer).collect();
    let fetched = claimed_reads::glue_ranges(weights, opening.opening(), &layers)
        .and_then(|ranges| weights.prefetch(&ranges))
        .context("fetch the tensors the glue check reads");
    let mut reports = Vec::new();
    let mut layouts = Layouts::new();
    let mut state_layers = BTreeSet::new();
    for pick in picks {
        let Some((layer, backend)) = opening.layer(pick.layer) else {
            continue;
        };
        let checked = match &fetched {
            Ok(()) => check_layer(
                checks.timing_runs,
                model,
                weights,
                layer,
                rows_seed,
                backend,
            ),
            Err(error) => CheckedLayer::unavailable(layer, &format!("{error:#}")),
        };
        output.layers.push(checked.output);
        reports.extend(checked.rows);
        if let Some(glue) = checked.glue {
            layouts.insert(pick.layer, glue.attention);
            if glue.states > 0 {
                state_layers.insert(pick.layer);
            }
        }
    }
    output.roles = audit::pool_roles(&reports)
        .iter()
        .map(RoleOutput::from)
        .collect();
    output.anchors = match &fetched {
        Ok(()) => check_anchors(model, opening),
        Err(error) => Check::Unavailable(format!("{error:#}")),
    };
    output.head = check_head(
        model,
        weights,
        opening,
        checks.head_key.as_deref(),
        rows_seed,
    );
    note_unjudged_tokens(output);
    output.tokens = checks.response.map(|(prompt, reply)| {
        Check::of(token_check::check_tokens(
            model,
            opening.receipt(),
            opening.prefix_tokens(),
            prompt,
            reply,
        ))
    });
    let evaluated: BTreeSet<u32> = layouts.keys().copied().collect();
    let known = KnownLayers {
        evaluated: &evaluated,
        state: &state_layers,
    };
    output.recurrent = recurrent_audit::check_recurrent(model, opening, &known);
    layouts
}

/// A sampler statement that leaves the tokens unjudged is a gap of the
/// audit, which the report names.
fn note_unjudged_tokens(output: &mut AuditOutput) {
    let Some(Check::Verdict(head)) = &output.head else {
        return;
    };
    if head.positions.iter().any(|p| !p.judged) {
        output.notes.push(format!(
            "not checked: the sampled tokens are {}; their logits were checked",
            head.rule
        ));
    }
}

/// The input of layer 0, recomputed from the receipt's tokens.
fn check_anchors(model: &dyn ClaimedModel, opening: &ValidatedOpening<'_>) -> Check<GlueOutput> {
    let mut verdict = None;
    let checked = model.with_layer(0, CacheTypes::default(), &mut |evaluator| {
        verdict = Some(glue::check_anchors(evaluator, opening.anchors())?);
        Ok(())
    });
    Check::of(checked.and_then(|()| {
        let verdict = verdict.context("the anchors were not checked")?;
        Ok(GlueOutput::from(&verdict))
    }))
}

/// The sampled tokens at the head positions, judged on the logits the node
/// opened. An audit holds a head key and checks those logits with it. A
/// recheck has no key and checks them against every row of the claimed
/// output matrix. The two accept the same logits, so they give one
/// verdict. A node that left its logits out fails both.
///
/// A model whose logits are not the output matrix's product is the
/// exception: the check computes its logits from the opened final
/// residuals, with or without a key.
///
/// A turn with a reasoning budget has tokens its sampler forced. The
/// check replays that sampler with the claimed model's tokenizer, so a
/// forced token must be the forced one and every other token is judged
/// as usual ([`reasoning_budget`]).
fn check_head(
    model: &dyn ClaimedModel,
    weights: &WeightFile,
    opening: &ValidatedOpening<'_>,
    head_key: Option<&Path>,
    rows_seed: &[u8],
) -> Option<Check<HeadOutput>> {
    let (backend, openings) = opening.head()?;
    if openings.is_empty() {
        return None;
    }
    let receipt = opening.receipt();
    let started = Instant::now();
    let opened = head::logits_are_linear(weights);
    let keyed = head_key.filter(|_| opened);
    let rule = head::token_rule_for(weights, receipt, opened);
    let forcing = reasoning_budget::replay(receipt, Some(&ClaimedVocabulary(model)));
    let checked: Result<(HeadVerdict, Option<f64>)> = match keyed {
        Some(path) => head_key_file::load_or_build(path, weights).and_then(|(key, built)| {
            let verdict = head::key::check_head_with_key(
                weights, &key, receipt, openings, backend, rows_seed, &forcing,
            );
            // whether the key accepted the logits is a bit the node learns
            // about it: a key that refused some is not used again
            let refused = verdict.as_ref().is_ok_and(|v| {
                v.positions
                    .iter()
                    .any(|p| p.logits.is_some_and(|l| !l.pass))
            });
            if refused {
                // a key that cannot be removed must not hide the verdict
                let _ = head_key_file::discard(path);
            }
            Ok((verdict?, built))
        }),
        None => head::check_head(weights, receipt, openings, backend, rows_seed, &forcing)
            .map(|verdict| (verdict, None)),
    };
    Some(Check::of(checked.map(|(verdict, built)| {
        let seconds = started.elapsed().as_secs_f64() - built.unwrap_or(0.0);
        let mut output = HeadOutput::new(&verdict, seconds);
        output.rule = match forcing.unjudged() {
            Some(reason) => TokenRule::Unjudged(reason.to_owned()).describe(),
            None => rule.describe(),
        };
        output.reasoning = ReasoningOutput::of(&forcing, &verdict);
        output.key_build_seconds = built;
        output
    })))
}

/// The claimed model's tokenizer, for the replay of a reasoning budget.
struct ClaimedVocabulary<'a>(&'a dyn ClaimedModel);

impl Vocabulary for ClaimedVocabulary<'_> {
    fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
        self.0.tokenize(text)
    }

    fn piece(&self, token: i32) -> Result<Vec<u8>> {
        self.0.token_piece(token)
    }
}

/// One picked layer, checked.
struct CheckedLayer {
    output: LayerOutput,
    /// The row check's report, when it ran.
    rows: Option<AuditReport>,
    /// The glue check's report, when it ran.
    glue: Option<GlueReport>,
}

impl CheckedLayer {
    /// A layer no check of which could run here.
    fn unavailable(opening: &LayerOpening, reason: &str) -> Self {
        Self {
            output: LayerOutput {
                layer: opening.layer,
                matrices: Vec::new(),
                rows_fault: None,
                rows_error: Some(reason.to_owned()),
                glue: Vec::new(),
                glue_pass: false,
                glue_fault: None,
                glue_error: Some(reason.to_owned()),
                unchecked_weights: Vec::new(),
                cost: CostOutput {
                    positions: opening.leaves.len(),
                    ..CostOutput::default()
                },
            },
            rows: None,
            glue: None,
        }
    }
}

/// The glue check on the layer loaded alone, then the row checks. The
/// glue's evaluation gives the layer's plan: which weight each matmul
/// reads and which are routers. The row check takes those from it, never
/// from the opening; where the glue could not run it takes no matmul for
/// a router.
fn check_layer(
    timing_runs: u32,
    model: &dyn ClaimedModel,
    weights: &WeightFile,
    opening: &LayerOpening,
    rows_seed: &[u8],
    backend: &str,
) -> CheckedLayer {
    let started = Instant::now();
    let mut glue_eval_seconds = Vec::new();
    let mut glue_check_seconds = 0.0;
    let mut report = None;
    let loaded = model.with_layer(
        opening.layer,
        layer_cache_types(opening),
        &mut |evaluator| {
            let check_started = Instant::now();
            let checked = glue::check_glue(evaluator, opening)?;
            glue_check_seconds = check_started.elapsed().as_secs_f64();
            glue_eval_seconds.push(checked.eval_seconds);
            let request = glue::glue_request(opening)?;
            for _ in 1..timing_runs {
                let run_started = Instant::now();
                evaluator.evaluate(&request)?;
                glue_eval_seconds.push(run_started.elapsed().as_secs_f64());
            }
            report = Some(checked);
            Ok(())
        },
    );
    let layer_seconds = started.elapsed().as_secs_f64();
    let glue: Check<GlueReport> =
        Check::of(loaded.and_then(|()| report.ok_or_else(|| anyhow!("the glue was not checked"))));
    let started = Instant::now();
    let check_rows = |routers| audit::check_rows(weights, opening, routers, rows_seed, backend);
    let rows: Check<AuditReport> = match &glue {
        Check::Verdict(glue) => Check::of(check_rows(Routers::Planned(&glue.matmuls))),
        Check::NodeFault(_) => Check::Unavailable(
            "not checked: the glue check found the opening of the layer defective".to_owned(),
        ),
        // without the layer's plan no matmul is known for a router: each
        // is checked on a sample of its rows. A column that fails is still
        // a failure, and the glue that could not run keeps the layer from
        // counting as checked.
        Check::Unavailable(_) => Check::of(check_rows(Routers::Unknown)),
    };
    let row_check_seconds = started.elapsed().as_secs_f64();
    let glue_seconds: f64 = glue_check_seconds + glue_eval_seconds.iter().skip(1).sum::<f64>();
    let ran = glue.verdict();
    let output = LayerOutput {
        layer: opening.layer,
        matrices: rows
            .verdict()
            .map(|r| r.matrices.iter().map(MatrixOutput::from).collect())
            .unwrap_or_default(),
        rows_fault: rows.node_fault().map(str::to_owned),
        rows_error: rows.unavailable().map(str::to_owned),
        glue: ran
            .map(|g| g.verdicts.iter().map(GlueOutput::from).collect())
            .unwrap_or_default(),
        glue_pass: ran.is_some_and(|g| g.pass),
        glue_fault: glue.node_fault().map(str::to_owned),
        glue_error: glue.unavailable().map(str::to_owned),
        unchecked_weights: ran.map(|g| g.unchecked_weights.clone()).unwrap_or_default(),
        cost: CostOutput {
            positions: opening.leaves.len(),
            layer_load_seconds: (layer_seconds - glue_seconds).max(0.0),
            glue_eval_seconds,
            glue_check_seconds,
            row_check_seconds,
        },
    };
    CheckedLayer {
        output,
        rows: match rows {
            Check::Verdict(report) => Some(report),
            _ => None,
        },
        glue: match glue {
            Check::Verdict(report) => Some(report),
            _ => None,
        },
    }
}

/// The KV cache types of the rows `opening`'s layer stored: its first
/// stored row is the key, its second the value. The auditor builds its
/// layer with them, so they reach the native runtime: only a type the
/// auditor checks is taken from the opening, and any other keeps the
/// default.
fn layer_cache_types(opening: &LayerOpening) -> CacheTypes {
    let stored = |index| {
        opening
            .leaves
            .iter()
            .flat_map(|leaf| &leaf.mixing)
            .find(|m| m.kind == MixingKind::Stored && m.index == index && m.values.is_some())
            .and_then(|m| cache_types::audited(m.ggml_type))
    };
    let k = stored(0);
    CacheTypes {
        k,
        v: stored(1).or(k),
    }
}

#[cfg(test)]
mod tests {
    use skippy_receipts::wire::{LeafOpening, TensorOpening};

    use super::*;

    fn layer(stored: &[(u32, i32)]) -> LayerOpening {
        let mixing = stored
            .iter()
            .map(|&(index, ggml_type)| {
                TensorOpening::new(MixingKind::Stored, index, Some((ggml_type, &[0u8; 4])))
            })
            .collect();
        LayerOpening {
            layer: 3,
            leaves: vec![LeafOpening {
                position: 0,
                residual_in: String::new(),
                residual_out: None,
                residual_in_path: Vec::new(),
                mixing_path: Vec::new(),
                residual_out_path: Vec::new(),
                mixing,
                matrices: Vec::new(),
                selections: Vec::new(),
            }],
        }
    }

    #[test]
    fn only_a_cache_type_an_audit_covers_reaches_the_layer_load() {
        // f16 keys and f32 values
        let types = layer_cache_types(&layer(&[(0, 1), (1, 0)]));
        assert_eq!((types.k, types.v), (Some(1), Some(0)));
        // one stored row: the value has the key's type
        let types = layer_cache_types(&layer(&[(0, 1)]));
        assert_eq!((types.k, types.v), (Some(1), Some(1)));
        // no audit covers a quantized cache (q8_0 is 8, q4_0 is 2), so its
        // types never reach the load
        for quantized in [8, 2] {
            let types = layer_cache_types(&layer(&[(0, quantized), (1, quantized)]));
            assert_eq!(types, CacheTypes::default(), "{quantized}");
        }
        // 39 is no ggml type, and a negative id is none either: llama.cpp
        // would abort on them, so the load keeps its default
        for unknown in [39, 3, -1, i32::MAX] {
            let types = layer_cache_types(&layer(&[(0, unknown), (1, unknown)]));
            assert_eq!(types, CacheTypes::default(), "{unknown}");
        }
        assert_eq!(layer_cache_types(&layer(&[])), CacheTypes::default());
    }
}
