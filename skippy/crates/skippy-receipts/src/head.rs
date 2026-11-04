//! The output head check. The final stage opens the residual leaving the
//! model's final layer at each head position ([`crate::wire::HeadOpening`]);
//! the client applies the output norm and the whole output matrix of the
//! claimed GGUF on its CPU, and checks that the token the turn sampled from
//! that position is the greedy choice, or for a sampled turn one its
//! filters keep ([`sampling`]).
//!
//! Without this check a node could run every claimed layer and still pick
//! other tokens. The residual is committed and chained to the checked
//! layers, so the node cannot change it either.
//!
//! The node computed its logits with its own rounding, so a token whose
//! logit is within [`TIE_BAND`] of the best reference logit is accepted,
//! and counted: a node can steer near-ties inside the band, so a turn with
//! more of them than chance allows is flagged.
//!
//! Only the order of the logits matters to a greedy choice, so the check
//! skips the monotone transforms some models apply after the matrix (logit
//! scaling and soft-capping).

use anyhow::{Context, Result, bail};

use crate::weights::{MetaValue, WeightFile};
use crate::wire::{HeadOpening, Receipt};

pub mod key;
pub mod sampling;

use sampling::TokenRule;

/// The tie band, relative to the root mean square of a position's logits.
/// The worst honest matmul error measured is under 1e-3 of the column's
/// RMS per row, so the band leaves room for the largest of a vocabulary's
/// rows.
pub const TIE_BAND: f64 = 1e-2;

/// Tokens accepted within the tie band that a turn of `positions` checked
/// positions may have before it is flagged.
pub fn allowed_ties(positions: usize) -> usize {
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
    fn apply(&self, x: &[f32]) -> Result<Vec<f64>> {
        if x.len() != self.weight.len() {
            bail!(
                "the output residual has {} values, the output norm {}",
                x.len(),
                self.weight.len()
            );
        }
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
pub struct OutputHead {
    /// The output matrix: `output.weight`, or the token embedding when the
    /// model ties them.
    weight: String,
    bias: Option<Vec<f64>>,
    norm: OutputNorm,
}

/// Whether the logits a node opens are the output matrix's product up to a
/// uniform scale, so a head key can check them: no soft-capping comes
/// after it.
pub fn logits_are_linear(weights: &WeightFile) -> bool {
    !weights
        .metadata
        .keys()
        .any(|k| k.ends_with(".final_logit_softcapping"))
}

/// Whether the model scales or caps its logits after the output matrix.
pub fn transforms_logits(weights: &WeightFile) -> bool {
    weights
        .metadata
        .keys()
        .any(|k| k.ends_with(".logit_scale") || k.ends_with(".final_logit_softcapping"))
}

/// The output matrix of the claimed model: `output.weight`, or the token
/// embedding when the model ties them.
pub fn head_weight(weights: &WeightFile) -> &'static str {
    if weights.tensors.contains_key("output.weight") {
        "output.weight"
    } else {
        "token_embd.weight"
    }
}

impl OutputHead {
    pub fn from_weights(weights: &WeightFile) -> Result<Self> {
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
    /// matrix once, split over the machine's cores.
    pub fn logits(&self, weights: &WeightFile, residuals: &[Vec<f32>]) -> Result<Vec<Vec<f64>>> {
        let inputs = residuals
            .iter()
            .map(|x| self.norm.apply(x))
            .collect::<Result<Vec<_>>>()?;
        let rows = weights.tensor(&self.weight)?.rows();
        let threads = std::thread::available_parallelism()
            .map_or(1, std::num::NonZero::get)
            .min(16) as u64;
        let span = rows.div_ceil(threads).max(1);
        let parts = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..rows)
                .step_by(span as usize)
                .map(|first| {
                    let inputs = &inputs;
                    scope.spawn(move || -> Result<Vec<Vec<f64>>> {
                        let range = first..(first + span).min(rows);
                        let mut out =
                            vec![Vec::with_capacity((range.end - first) as usize); inputs.len()];
                        weights.each_row(&self.weight, range, &mut |row, values| {
                            for (logits, x) in out.iter_mut().zip(inputs) {
                                let dot: f64 = values.iter().zip(x).map(|(w, v)| w * v).sum();
                                let bias = self.bias.as_ref().map_or(0.0, |b| b[row as usize]);
                                logits.push(dot + bias);
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
    /// The check of the node's logits with the client's head key, when the
    /// check judged them instead of computing the head.
    pub logits: Option<key::LogitsCheck>,
    pub pass: bool,
}

/// The head check over every head position.
#[derive(Clone, Debug, PartialEq)]
pub struct HeadVerdict {
    pub positions: Vec<PositionVerdict>,
    pub ties: usize,
    /// More ties than [`allowed_ties`]: worth a second look, not a failure.
    pub flagged: bool,
    pub pass: bool,
}

/// Judges a greedy choice of `sampled` from `logits`.
pub fn judge(position: u32, logits: &[f64], sampled: i32) -> Result<PositionVerdict> {
    let index = usize::try_from(sampled)
        .ok()
        .filter(|&i| i < logits.len())
        .with_context(|| format!("token {sampled} is outside the vocabulary"))?;
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
    let rms = (logits.iter().map(|v| v * v).sum::<f64>() / logits.len() as f64).sqrt();
    let band = TIE_BAND * rms;
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
        pass,
    })
}

/// Judges the token `sampled` from `logits` by the turn's `rule`.
pub fn judge_with(
    rule: &TokenRule,
    position: u32,
    logits: &[f64],
    sampled: i32,
) -> Result<PositionVerdict> {
    let mut verdict = judge(position, logits, sampled)?;
    match rule {
        TokenRule::Greedy => {}
        TokenRule::Filtered { filters, biases } => {
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

/// Checks the token sampled from each opened head position.
pub fn check_head(
    weights: &WeightFile,
    receipt: &Receipt,
    openings: &[HeadOpening],
) -> Result<HeadVerdict> {
    // the head reads its whole matrix: fetch it at once, not row by row
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
    weights.prefetch(&ranges)?;
    let head = OutputHead::from_weights(weights)?;
    let residuals = openings
        .iter()
        .map(HeadOpening::residual_values)
        .collect::<Result<Vec<_>>>()?;
    let logits = head.logits(weights, &residuals)?;
    let rule = token_rule(receipt).on_untransformed_logits(transforms_logits(weights));
    let positions = openings
        .iter()
        .zip(&logits)
        .map(|(opening, logits)| {
            let sampled = receipt.sampled_after(opening.position).with_context(|| {
                format!("no token was sampled at position {}", opening.position)
            })?;
            judge_with(&rule, opening.position, logits, sampled)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(verdict(positions))
}

fn verdict(positions: Vec<PositionVerdict>) -> HeadVerdict {
    let ties = positions.iter().filter(|p| p.within_band).count();
    HeadVerdict {
        ties,
        flagged: ties > allowed_ties(positions.len()),
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
    fn many_ties_are_flagged_and_no_position_fails() {
        let tie = |position| judge(position, &[5.0, 4.999], 1).unwrap();
        let v = verdict(vec![tie(1), judge(2, &[5.0, 1.0], 0).unwrap()]);
        assert!(v.pass && !v.flagged);
        let v = verdict(vec![tie(1), tie(2), tie(3)]);
        assert!(v.pass && v.flagged);
        assert!(!verdict(Vec::new()).pass);
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
