//! The output hea check. The final stage opens the residual leaving the
//! model's final layer at each head position, and the logits it computed
//! from it ([`crate::wire::HeadOpening`]). The client checks those logits
//! against the output norm and the output matrix of the claimed GGUF, with
//! the rounding of the node's backend, and then checks on them that the
//! token the turn sampled from that position is the greedy choice, or for
//! a sampled turn one its filters keep ([`sampling`]) and, when the turn's
//! draw can be replayed, the one the draw picks ([`draw`]). A token that
//! the turn's reasoning budget forced must be the forced one
//! ([`reasoning_budget`]).
//!
//! Without this check a node could run every claimed layer and still pick
//! other tokens. The residual is committed and chained to the checked
//! layers, so the node cannot change it either.
//!
//! The logits are checked in one of two ways, which accept the same
//! logits and judge the token on the same values, so they give one
//! verdict. An audit holds a secret head key and reads a few rows of the
//! matrix ([`key`]). A recheck of evidence has no key and reads every row
//! ([`every_row`]). Judging the token on the node's own logits matters: a
//! client that computed them without the node's activation rounding would
//! be off by more than a near-tie, and would refuse an honest node.
//!
//! A token whose logit is within [`TIE_BAND`] of the best one is accepted,
//! and counted: a node can steer near-ties inside the band, so a turn with
//! more of them than chance allows is flagged.
//!
//! A model that soft-caps its logits opens logits that are not the
//! matrix's product, so no check here can tell the node's logits from
//! others. Logits the client computes itself lack the node's activation
//! rounding, by more than the tie band, so judging a token on them could
//! fail an honest node. Such a model's tokens are not judged
//! ([`token_rule_for`]); [`check_computed_head`] still checks that its
//! final residuals fit the head.

use anyhow::{Context, Result, bail};

use crate::node_fault::{BlameNode, ensure_node, node_fault};
use crate::weights::{MetaValue, WeightFile};
use crate::wire::{HeadOpening, Receipt};

pub mod draw;
pub mod every_row;
pub mod key;
pub mod opened_logits;
pub mod reasoning_budget;
pub mod sampling;

use reasoning_budget::Forcing;
use sampling::TokenRule;

/// The tie band, relative to the root mean square of a position's logits.
///
/// The token is judged on the logits the node opened, which its replay
/// computed again. Where a replay does not repeat the serving kernel bit
/// for bit, they differ from the logits the node sampled from by that
/// kernel's rounding, and the band leaves room for it.
///
/// The band does not cover logits the client computes without the node's
/// activation rounding. On CPU with Q8_0 weights those were up to 4e-2 of
/// the RMS from an honest node's over the vocabulary, and about 2e-2 among
/// the likeliest tokens. So no token is judged on such logits
/// ([`token_rule_for`]).
pub(crate) const TIE_BAND: f64 = 1e-2;

/// The band of a replayed draw, relative to the root mean square of the
/// logits the node opened, which the draw is replayed on.
///
/// The node drew from its own logits, so the replay follows it exactly up
/// to the last bits of `exp`, which the node's and the client's libraries
/// may round apart. Every honest draw measured was exact with no band:
/// about 130 on CPU, 120 on CUDA (Q8_0 and Q4_K_M), and 160 on Vulkan
/// before it lost its rounding profiles. A wider band would only let a
/// node steer the draws near a boundary.
pub(crate) const DRAW_BAND: f64 = 1e-4;

fn root_mean_square(logits: &[f64]) -> f64 {
    if logits.is_empty() {
        return 0.0;
    }
    (logits.iter().map(|v| v * v).sum::<f64>() / logits.len() as f64).sqrt()
}

/// Tokens accepted within the tie band that a turn of `positions` checked
/// positions may have before it is flagged.
pub(crate) fn allowed_ties(positions: usize) -> usize {
    1 + positions / 16
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum NormKind {
    /// Scales by the root mean square.
    Rms,
    /// Subtracts the mean and scales by the standard deviation.
    Layer,
}

#[derive(Clone, Debug)]
struct OutputNorm {
    kind: NormKind,
    eps: f64,
    weight: Vec<f64>,
    bias: Option<Vec<f64>>,
}

impl OutputNorm {
    /// The norm of an opened residual. A residual of another width, or one
    /// that holds a value that is not a number, is the node's fault.
    fn apply(&self, x: &[f32]) -> Result<Vec<f64>> {
        ensure_node!(
            x.len() == self.weight.len(),
            "the output residual has {} values, the output norm {}",
            x.len(),
            self.weight.len()
        );
        ensure_node!(
            crate::error_bound::all_finite(x),
            "the output residual holds a value that is not a number"
        );
        let n = x.len() as f64;
        let x: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
        let mean = match self.kind {
            NormKind::Rms => 0.0,
            NormKind::Layer => x.iter().sum::<f64>() / n,
        };
        let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
        let scale = 1.0 / (var + self.eps).sqrt();
        Ok(x.iter()
            .zip(&self.weight)
            .enumerate()
            .map(|(i, (v, w))| (v - mean) * scale * w + self.bias.as_ref().map_or(0.0, |b| b[i]))
            .collect())
    }
}

/// The claimed model's output head, read from its GGUF.
pub(crate) struct OutputHead {
    /// The output matrix: `output.weight`, or the token embedding when the
    /// model ties them.
    weight: String,
    bias: Option<Vec<f64>>,
    norm: OutputNorm,
}

/// Whether the logits a node opens are the output matrix's product up to a
/// uniform scale: no soft-capping comes after it. A check then compares
/// them with the matrix and judges the token on them. Otherwise it
/// computes the logits itself ([`check_computed_head`]).
pub fn logits_are_linear(weights: &WeightFile) -> bool {
    !weights
        .metadata
        .keys()
        .any(|k| k.ends_with(".final_logit_softcapping"))
}

/// The output matrix of the claimed model: `output.weight`, or the token
/// embedding when the model ties them.
pub(crate) fn head_weight(weights: &WeightFile) -> &'static str {
    if weights.tensors.contains_key("output.weight") {
        "output.weight"
    } else {
        "token_embd.weight"
    }
}

impl OutputHead {
    /// The bias of the logit of token `row`, 0 for a head without one.
    pub(crate) fn bias_of(&self, row: usize) -> f64 {
        self.bias.as_ref().map_or(0.0, |bias| bias[row])
    }

    pub(crate) fn from_weights(weights: &WeightFile) -> Result<Self> {
        let arch = weights.metadata_str("general.architecture")?;
        let float = |key: String| match weights.metadata.get(&key) {
            Some(MetaValue::Float(value)) => Some(*value),
            _ => None,
        };
        let rms = float(format!("{arch}.attention.layer_norm_rms_epsilon"));
        let layer = float(format!("{arch}.attention.layer_norm_epsilon"));
        let (kind, eps) = match (rms, layer) {
            (Some(eps), None) => (NormKind::Rms, eps),
            (None, Some(eps)) => (NormKind::Layer, eps),
            (Some(_), Some(_)) => {
                bail!("the claimed GGUF names both an RMS and a layer norm epsilon")
            }
            (None, None) => bail!("the claimed GGUF names no norm epsilon"),
        };
        let optional = |name: &str| {
            weights
                .tensors
                .contains_key(name)
                .then(|| weights.vector(name))
                .transpose()
        };
        let weight = head_weight(weights);
        let head = Self {
            weight: weight.to_owned(),
            bias: optional("output.bias")?,
            norm: OutputNorm {
                kind,
                eps,
                weight: weights.vector("output_norm.weight")?,
                bias: optional("output_norm.bias")?,
            },
        };
        let t = weights.tensor(&head.weight)?;
        if t.row_len() as usize != head.norm.weight.len() {
            bail!(
                "{} rows have {} values, the output norm {}",
                head.weight,
                t.row_len(),
                head.norm.weight.len()
            );
        }
        if let Some(bias) = &head.bias
            && bias.len() as u64 != t.rows()
        {
            bail!("output.bias does not match {}", head.weight);
        }
        Ok(head)
    }

    /// The logits of every token at each residual. Reads the whole output
    /// matrix once.
    pub fn logits(&self, weights: &WeightFile, residuals: &[Vec<f32>]) -> Result<Vec<Vec<f64>>> {
        let inputs = residuals
            .iter()
            .map(|x| self.norm.apply(x))
            .collect::<Result<Vec<_>>>()?;
        let mut logits = self.products(weights, &inputs)?;
        if self.bias.is_some() {
            for values in &mut logits {
                for (row, value) in values.iter_mut().enumerate() {
                    *value += self.bias_of(row);
                }
            }
        }
        Ok(logits)
    }

    /// Fetches everything the head reads at once, and not row by row: the
    /// norm, the bias and the whole output matrix.
    pub(crate) fn prefetch(weights: &WeightFile) -> Result<()> {
        let ranges = [
            "output.weight",
            "output.bias",
            "output_norm.weight",
            "output_norm.bias",
        ]
        .into_iter()
        .chain([head_weight(weights)])
        .filter(|name| weights.tensors.contains_key(*name))
        .map(|name| weights.tensor_range(name))
        .collect::<Result<Vec<_>>>()?;
        weights.prefetch(&ranges)
    }

    /// The product of the output matrix with each of `inputs`, one value
    /// per row and without the bias. Reads the whole matrix once, split
    /// over the machine's cores.
    pub(crate) fn products(
        &self,
        weights: &WeightFile,
        inputs: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>> {
        let rows = weights.tensor(&self.weight)?.rows();
        let threads = std::thread::available_parallelism()
            .map_or(1, std::num::NonZero::get)
            .min(16) as u64;
        let span = rows.div_ceil(threads).max(1);
        let parts = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..rows)
                .step_by(span as usize)
                .map(|first| {
                    scope.spawn(move || -> Result<Vec<Vec<f64>>> {
                        let range = first..(first + span).min(rows);
                        let mut out =
                            vec![Vec::with_capacity((range.end - first) as usize); inputs.len()];
                        weights.each_row(&self.weight, range, &mut |_, values| {
                            for (products, x) in out.iter_mut().zip(inputs) {
                                products.push(values.iter().zip(x).map(|(w, v)| w * v).sum());
                            }
                        })?;
                        Ok(out)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap_or_else(|_| bail!("a head thread panicked")))
                .collect::<Result<Vec<_>>>()
        })?;
        let mut logits = vec![Vec::with_capacity(rows as usize); inputs.len()];
        for part in parts {
            for (all, slice) in logits.iter_mut().zip(part) {
                all.extend(slice);
            }
        }
        Ok(logits)
    }
}

/// One head position: the sampled token against the reference logits.
#[derive(Clone, Debug, PartialEq)]
pub struct PositionVerdict {
    pub position: u32,
    pub sampled: i32,
    /// The token with the best reference logit.
    pub best: i32,
    /// The sampled token's logit less the best other token's.
    pub margin: f64,
    /// The tie band at this position, in logits.
    pub band: f64,
    /// Not the reference best, but within the band.
    pub within_band: bool,
    /// Whether the sampler let the check judge the token; when not, the
    /// position passes on its logits alone.
    pub judged: bool,
    /// The check of the node's logits against the claimed output matrix,
    /// when the check judged them instead of computing the head.
    pub logits: Option<opened_logits::LogitsCheck>,
    /// The replay of the draw, for a sampled turn whose draw can be
    /// replayed and whose token its filters keep.
    pub draw: Option<draw::DrawVerdict>,
    /// The token the turn's reasoning budget forced here, when it forced
    /// one: the sampled token must be it, whatever the logits.
    pub forced: Option<i32>,
    /// Whether the token passed its judgement on the logits the node
    /// opened, whatever the check of those logits found. The node signed
    /// them, so a token they refuse is a defect anyone can see.
    pub token_pass: bool,
    pub pass: bool,
}

/// The head check over every head position.
#[derive(Clone, Debug, PartialEq)]
pub struct HeadVerdict {
    pub positions: Vec<PositionVerdict>,
    pub ties: usize,
    /// Draws that pick the sampled token only with the band's help. A node
    /// can steer those and no others.
    pub band_draws: usize,
    /// More ties, or more such draws, than [`allowed_ties`]: worth a second
    /// look, not a failure.
    pub flagged: bool,
    pub pass: bool,
}

/// Judges a greedy choice of `sampled` from `logits`. A sampled token the
/// vocabulary does not hold is the node's fault: the receipt names it.
pub fn judge(position: u32, logits: &[f64], sampled: i32) -> Result<PositionVerdict> {
    let Some(index) = usize::try_from(sampled).ok().filter(|&i| i < logits.len()) else {
        node_fault!("position {position}: token {sampled} is outside the vocabulary");
    };
    // `f64::max` and `total_cmp` below would hide a logit that is not a
    // number behind the others
    if logits.iter().any(|l| !l.is_finite()) {
        bail!("position {position}: a logit is not a number");
    }
    let best = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .context("the output head has no rows")?;
    let other = logits
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != index)
        .map(|(_, &v)| v)
        .fold(f64::NEG_INFINITY, f64::max);
    let band = TIE_BAND * root_mean_square(logits);
    let margin = logits[index] - other;
    let pass = margin >= -band;
    Ok(PositionVerdict {
        position,
        sampled,
        best: best as i32,
        margin,
        band,
        within_band: pass && best != index,
        judged: true,
        logits: None,
        draw: None,
        forced: None,
        token_pass: pass,
        pass,
    })
}

/// Judges the token `sampled` from `logits` at a position of a turn whose
/// reasoning budget `forcing` describes. A token the budget forced must be
/// the forced one: the sampler left it as the only candidate, so the
/// logits did not choose it. A budget the check could not replay leaves
/// the turn's tokens unjudged. Every other token is judged by `rule`
/// ([`judge_with`]).
pub(crate) fn judge_position(
    rule: &TokenRule,
    forcing: &Forcing,
    position: u32,
    logits: &[f64],
    sampled: i32,
) -> Result<PositionVerdict> {
    let mut verdict = judge_forced(rule, forcing, position, logits, sampled)?;
    verdict.token_pass = verdict.pass;
    Ok(verdict)
}

/// [`judge_position`] before it notes the token's own verdict.
fn judge_forced(
    rule: &TokenRule,
    forcing: &Forcing,
    position: u32,
    logits: &[f64],
    sampled: i32,
) -> Result<PositionVerdict> {
    if matches!(rule, TokenRule::Unjudged(_)) {
        return judge_with(rule, position, logits, sampled);
    }
    if let Some(reason) = forcing.unjudged() {
        let unjudged = TokenRule::Unjudged(reason.to_owned());
        return judge_with(&unjudged, position, logits, sampled);
    }
    let Some(forced) = forcing.forced_at(position) else {
        return judge_with(rule, position, logits, sampled);
    };
    let mut verdict = judge(position, logits, sampled)?;
    verdict.pass = sampled == forced;
    verdict.within_band = false;
    verdict.forced = Some(forced);
    Ok(verdict)
}

/// Judges the token `sampled` from `logits`, the logits the node opened, by
/// the turn's `rule`. A draw the rule replays is replayed on them: the node
/// sampled from those, so a draw that does not pick its token contradicts
/// the node's own opening, whoever checks it.
pub(crate) fn judge_with(
    rule: &TokenRule,
    position: u32,
    logits: &[f64],
    sampled: i32,
) -> Result<PositionVerdict> {
    let mut verdict = judge(position, logits, sampled)?;
    match rule {
        TokenRule::Greedy => {}
        TokenRule::Filtered {
            filters, biases, ..
        } => {
            verdict.pass =
                TokenRule::keeps(filters, biases, logits, sampled as usize, verdict.band);
            verdict.within_band = false;
        }
        TokenRule::Unjudged(_) => {
            verdict.pass = true;
            verdict.within_band = false;
            verdict.judged = false;
        }
    }
    if let Some(replayed) = rule.replayed_draw().filter(|_| verdict.pass) {
        // the token sampled from a position takes the next one
        let replay = draw::judge(
            replayed.draw,
            replayed.filters,
            &sampling::biased(logits, replayed.biases),
            sampled as usize,
            u64::from(position) + 1,
            DRAW_BAND * root_mean_square(logits),
        );
        verdict.pass = replay.pass;
        verdict.draw = Some(replay);
    }
    Ok(verdict)
}

/// The token rule of `receipt`'s sampler. An unbound receipt states none,
/// and is judged as greedy, as an audit asks for.
pub fn token_rule(receipt: &Receipt) -> TokenRule {
    receipt
        .binding
        .as_ref()
        .map_or(TokenRule::Greedy, |b| TokenRule::from_sampler(&b.sampler))
}

/// The token rule of `receipt`'s sampler for the claimed model.
///
/// The runtime takes the model's suppressed tokens out of every candidate
/// list, and for a turn that ignores its end the end-of-generation tokens
/// too ([`TokenRule::without`]). `is_end` says whether a token is one of
/// the second, as the claimed model's vocabulary does (`llama_vocab_is_eog`);
/// it is asked only for a turn that ignores its end, of every token of
/// the output head.
///
/// A model whose logits are not the matrix's product gets no token
/// judgement: see [`TIE_BAND`].
pub fn token_rule_for(
    weights: &WeightFile,
    receipt: &Receipt,
    is_end: &dyn Fn(i32) -> Result<bool>,
) -> TokenRule {
    if !logits_are_linear(weights) {
        return TokenRule::Unjudged(
            "the model soft-caps its logits, so the check cannot judge tokens on the logits \
             the node sampled from"
                .into(),
        );
    }
    let rule = token_rule(receipt);
    let mut excluded = weights.suppressed_tokens().to_vec();
    let ignores_end = receipt
        .binding
        .as_ref()
        .is_some_and(|b| sampling::ignores_end_tokens(&b.sampler));
    if ignores_end {
        match end_tokens(weights, is_end) {
            Ok(tokens) => excluded.extend(tokens),
            Err(error) => {
                let reason = format!(
                    "the claimed model's vocabulary does not name its end-of-generation \
                     tokens: {error:#}"
                );
                return rule.without(Err(&reason));
            }
        }
    }
    rule.without(Ok(&excluded))
}

/// The end-of-generation tokens among the rows of the claimed output head.
fn end_tokens(weights: &WeightFile, is_end: &dyn Fn(i32) -> Result<bool>) -> Result<Vec<i32>> {
    let rows = weights.tensor(head_weight(weights))?.rows();
    let rows = i32::try_from(rows).context("the output head has more rows than tokens")?;
    let mut tokens = Vec::new();
    for token in 0..rows {
        if is_end(token)? {
            tokens.push(token);
        }
    }
    Ok(tokens)
}

/// Checks the token sampled from each opened head position without a head
/// key, as a recheck of evidence does, by `rule` ([`token_rule_for`]). It
/// reads the whole output matrix.
///
/// It checks the logits the stage on `backend` opened against every row
/// of the matrix and judges the token on them
/// ([`every_row::check_head_on_every_row`]), which is what an audit does
/// with its key ([`key::check_head_with_key`]). For a model whose logits
/// are not the matrix's product it checks the final residuals alone
/// ([`check_computed_head`]).
///
/// `forcing` says which tokens the turn's reasoning budget forced
/// ([`reasoning_budget::replay`]).
///
/// An error that blames the node ([`crate::node_fault::NodeFault`]) is a
/// defect of the opening. Any other error is the auditor's.
pub fn check_head(
    weights: &WeightFile,
    receipt: &Receipt,
    openings: &[HeadOpening],
    backend: &str,
    rows_seed: &[u8],
    checked: (&TokenRule, &Forcing),
) -> Result<HeadVerdict> {
    if logits_are_linear(weights) {
        every_row::check_head_on_every_row(weights, receipt, openings, backend, rows_seed, checked)
    } else {
        check_computed_head(weights, receipt, openings, checked)
    }
}

/// For a model that soft-caps its logits: checks that each opened final
/// residual fits the claimed output head, and passes each token to `rule`,
/// which for such a model judges none ([`token_rule_for`]).
pub fn check_computed_head(
    weights: &WeightFile,
    receipt: &Receipt,
    openings: &[HeadOpening],
    (rule, forcing): (&TokenRule, &Forcing),
) -> Result<HeadVerdict> {
    OutputHead::prefetch(weights)?;
    let head = OutputHead::from_weights(weights)?;
    let residuals = openings
        .iter()
        .map(|opening| opening.residual_values().or_node_fault())
        .collect::<Result<Vec<_>>>()?;
    let logits = head.logits(weights, &residuals)?;
    let positions = openings
        .iter()
        .zip(&logits)
        .map(|(opening, logits)| {
            let sampled = receipt.sampled_after(opening.position).with_context(|| {
                format!("no token was sampled at position {}", opening.position)
            })?;
            judge_position(rule, forcing, opening.position, logits, sampled)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(verdict(positions))
}

fn verdict(positions: Vec<PositionVerdict>) -> HeadVerdict {
    let ties = positions.iter().filter(|p| p.within_band).count();
    let band_draws = positions
        .iter()
        .filter(|p| p.draw.is_some_and(|draw| draw.pass && !draw.exact))
        .count();
    let allowed = allowed_ties(positions.len());
    HeadVerdict {
        ties,
        band_draws,
        flagged: ties > allowed || band_draws > allowed,
        pass: !positions.is_empty() && positions.iter().all(|p| p.pass),
        positions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::test_gguf;

    #[test]
    fn the_best_token_passes_and_a_near_tie_is_counted() {
        let logits = [1.0, 5.0, 4.999, -3.0];
        let best = judge(7, &logits, 1).unwrap();
        assert!(best.pass && !best.within_band);
        assert!((best.margin - 0.001).abs() < 1e-9);
        let tie = judge(7, &logits, 2).unwrap();
        assert!(tie.pass && tie.within_band);
        assert_eq!(tie.best, 1);
        let wrong = judge(7, &logits, 0).unwrap();
        assert!(!wrong.pass);
        assert!(judge(7, &logits, 4).is_err());
        assert!(judge(7, &logits, -1).is_err());
    }

    #[test]
    fn a_residual_or_token_that_does_not_fit_the_head_is_the_nodes_fault() {
        use crate::node_fault::NodeFault;
        let weights = model(true, false);
        let head = OutputHead::from_weights(&weights).unwrap();
        for residual in [
            vec![1.0, 2.0],
            vec![1.0, f32::NAN, 2.0],
            vec![f32::INFINITY, 0.0, 2.0],
        ] {
            let error = head.logits(&weights, &[residual]).unwrap_err();
            assert!(NodeFault::find(&error).is_some(), "{error:#}");
        }
        let error = judge(7, &[1.0, 5.0], 2).unwrap_err();
        assert!(NodeFault::find(&error).is_some(), "{error:#}");
        // a logit that is not a number cannot be judged: `f64::max` would
        // skip it and pass the token
        assert!(judge(7, &[1.0, f64::NAN, 0.5], 0).is_err());
    }

    #[test]
    fn many_ties_are_flagged_and_no_position_fails() {
        let tie = |position| judge(position, &[5.0, 4.999], 1).unwrap();
        let v = verdict(vec![tie(1), judge(2, &[5.0, 1.0], 0).unwrap()]);
        assert!(v.pass && !v.flagged);
        let v = verdict(vec![tie(1), tie(2), tie(3)]);
        assert!(v.pass && v.flagged);
        assert!(!verdict(Vec::new()).pass);
    }

    /// A rule that replays the draw of `seed` over every token.
    fn drawn(seed: u32) -> TokenRule {
        TokenRule::Filtered {
            filters: Vec::new(),
            biases: Vec::new(),
            draw: sampling::DrawRule::Indexed(draw::IndexedDraw {
                seed,
                temperature: 1.0,
            }),
        }
    }

    #[test]
    fn a_sampled_token_is_the_one_its_draw_picks() {
        // weights 1 and 3: the draw picks token 0 below 0.25, else token 1
        let logits = [0.0, 3f64.ln()];
        let rule = drawn(42);
        for position in 0..40u32 {
            let u = draw::uniform(42, u64::from(position) + 1);
            // away from the boundary, where the tie band could admit both
            if (u - 0.25).abs() < 0.05 {
                continue;
            }
            let picked = i32::from(u >= 0.25);
            let right = judge_with(&rule, position, &logits, picked).unwrap();
            assert!(
                right.pass && right.draw.is_some_and(|d| d.exact),
                "{right:?}"
            );
            let wrong = judge_with(&rule, position, &logits, 1 - picked).unwrap();
            assert!(
                !wrong.pass && wrong.draw.is_some_and(|d| !d.pass),
                "{wrong:?}"
            );
        }
        let position = (0..40u32)
            .find(|&p| draw::uniform(42, u64::from(p) + 1) > 0.6)
            .unwrap();
        // a rule without a draw judges the kept token alone
        let unreplayed = TokenRule::Filtered {
            filters: Vec::new(),
            biases: Vec::new(),
            draw: sampling::DrawRule::Unjudged("no draw".into()),
        };
        let verdict = judge_with(&unreplayed, position, &logits, 0).unwrap();
        assert!(verdict.pass && verdict.draw.is_none());
    }

    #[test]
    fn a_token_the_reasoning_budget_forced_must_be_the_forced_one() {
        // the draw picks token 0 below 0.25, else token 1
        let logits = [0.0, 3f64.ln()];
        let rule = drawn(42);
        let position = (0..40u32)
            .find(|&p| draw::uniform(42, u64::from(p) + 1) > 0.6)
            .unwrap();
        // the budget forced token 0 here: the draw's token 1 is not it
        let forcing = Forcing::forced(&[(position, 0)]);
        let forced = judge_position(&rule, &forcing, position, &logits, 0).unwrap();
        assert!(forced.pass && forced.judged, "{forced:?}");
        assert_eq!((forced.forced, forced.draw), (Some(0), None));
        let drawn_token = judge_position(&rule, &forcing, position, &logits, 1).unwrap();
        assert!(!drawn_token.pass && drawn_token.forced == Some(0));
        // every other position of the turn is judged by the draw
        let free = Forcing::forced(&[(position + 1, 0)]);
        for forcing in [&free, &Forcing::Free] {
            let right = judge_position(&rule, forcing, position, &logits, 1).unwrap();
            assert!(right.pass && right.draw.is_some_and(|d| d.exact));
            assert_eq!(right.forced, None);
            let wrong = judge_position(&rule, forcing, position, &logits, 0).unwrap();
            assert!(!wrong.pass, "{wrong:?}");
        }
        // a greedy-like rule too: the forced token need not be the best
        let greedy = TokenRule::Filtered {
            filters: vec![sampling::Filter::TopK(1)],
            biases: Vec::new(),
            draw: sampling::DrawRule::Single,
        };
        assert!(
            judge_position(&greedy, &forcing, position, &logits, 0)
                .unwrap()
                .pass
        );
        assert!(
            !judge_position(&greedy, &Forcing::Free, position, &logits, 0)
                .unwrap()
                .pass
        );
        // a budget the check could not replay leaves the token unjudged
        let unjudged = Forcing::Unjudged("no tokenizer".into());
        let verdict = judge_position(&rule, &unjudged, position, &logits, 0).unwrap();
        assert!(verdict.pass && !verdict.judged && verdict.draw.is_none());
        // and a sampler that is beyond the check stays so, forced or not
        let beyond = TokenRule::Unjudged("a grammar".into());
        let verdict = judge_position(&beyond, &forcing, position, &logits, 1).unwrap();
        assert!(verdict.pass && !verdict.judged && verdict.forced.is_none());
    }

    #[test]
    fn a_draw_near_a_boundary_has_a_narrow_band() {
        // weights 1 and 3: the boundary between the tokens is at 0.25
        let logits = [0.0, 3f64.ln()];
        let rule = drawn(42);
        // a draw within 2e-3 of the boundary, on its upper side: token 1
        let position = (0..200_000u32)
            .find(|&p| {
                let u = draw::uniform(42, u64::from(p) + 1);
                u > 0.2501 && u < 0.252
            })
            .expect("a draw near the boundary");
        let judged = |token: i32| judge_with(&rule, position, &logits, token).unwrap();
        // on the node's own logits the draw picks token 1, and the narrow
        // band admits no other
        assert!(judged(1).pass);
        let steered = judged(0);
        assert!(!steered.pass, "{steered:?}");
        const { assert!(DRAW_BAND < TIE_BAND / 10.0) };
    }

    #[test]
    fn draws_that_pass_only_within_the_band_are_counted_and_flagged() {
        let position = |exact: bool| {
            let mut p = judge(1, &[5.0, 1.0], 0).unwrap();
            p.draw = Some(draw::DrawVerdict {
                uniform: 0.5,
                lower: 0.0,
                upper: 1.0,
                exact,
                pass: true,
            });
            p
        };
        let v = verdict(vec![position(true), position(false), position(true)]);
        assert_eq!((v.band_draws, v.flagged, v.pass), (1, false, true));
        let v = verdict(vec![position(false), position(false), position(true)]);
        assert_eq!((v.band_draws, v.flagged, v.pass), (2, true, true));
    }

    fn model(tied: bool, layer_norm: bool) -> WeightFile {
        let embd = vec![
            1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 1.0, //
            1.0, 1.0, -1.0,
        ];
        let mut tensors = vec![
            ("output_norm.weight", vec![3], vec![1.0, 2.0, 0.5]),
            ("token_embd.weight", vec![3, 4], embd.clone()),
        ];
        if !tied {
            let reversed: Vec<f32> = embd.iter().map(|v| -v).collect();
            tensors.push(("output.weight", vec![3, 4], reversed));
        }
        let eps_key = if layer_norm {
            "toy.attention.layer_norm_epsilon"
        } else {
            "toy.attention.layer_norm_rms_epsilon"
        };
        test_gguf::open(
            &[
                ("general.architecture", MetaValue::Str("toy".into())),
                (eps_key, MetaValue::Float(1e-6)),
            ],
            &tensors,
        )
    }

    #[test]
    fn the_head_is_the_output_norm_and_matrix() {
        let weights = model(true, false);
        let head = OutputHead::from_weights(&weights).unwrap();
        let x = vec![3.0f32, 0.0, 4.0];
        let logits = head.logits(&weights, &[x]).unwrap();
        // rms of (3, 0, 4) is sqrt(25 / 3); scaled by the norm weight
        let s = 1.0 / (25.0f64 / 3.0 + 1e-6).sqrt();
        let normed = [3.0 * s, 0.0, 4.0 * 0.5 * s];
        let expected = [
            normed[0],
            normed[1],
            normed[2],
            normed[0] + normed[1] - normed[2],
        ];
        for (got, want) in logits[0].iter().zip(expected) {
            assert!((got - want).abs() < 1e-12, "{got} {want}");
        }
        // an untied model uses its own output matrix
        let untied = model(false, false);
        let logits = OutputHead::from_weights(&untied)
            .unwrap()
            .logits(&untied, &[vec![3.0, 0.0, 4.0]])
            .unwrap();
        assert!((logits[0][0] + normed[0]).abs() < 1e-12);
    }

    #[test]
    fn a_layer_norm_subtracts_the_mean() {
        let weights = model(true, true);
        let head = OutputHead::from_weights(&weights).unwrap();
        // a constant residual normalizes to zero
        let logits = head.logits(&weights, &[vec![2.0, 2.0, 2.0]]).unwrap();
        assert!(logits[0].iter().all(|v| v.abs() < 1e-9));
    }

    #[test]
    fn many_threads_give_every_row_in_order() {
        let rows = 1000u64;
        let embd: Vec<f32> = (0..rows).flat_map(|r| [r as f32, 0.0]).collect();
        let weights = test_gguf::open(
            &[
                ("general.architecture", MetaValue::Str("toy".into())),
                (
                    "toy.attention.layer_norm_rms_epsilon",
                    MetaValue::Float(0.0),
                ),
            ],
            &[
                ("output_norm.weight", vec![2], vec![1.0, 1.0]),
                ("token_embd.weight", vec![2, rows], embd),
            ],
        );
        let head = OutputHead::from_weights(&weights).unwrap();
        let logits = head.logits(&weights, &[vec![1.0, 1.0]]).unwrap();
        assert_eq!(logits[0].len(), rows as usize);
        assert!(logits[0].windows(2).all(|w| w[0] < w[1]));
    }
}
