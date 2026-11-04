use super::*;
use crate::ggml_types::{GGML_F32, GGML_Q8_0};
use crate::head::{check_computed_head, check_head};
use crate::node_fault::NodeFault;
use crate::weights::{MetaValue, test_gguf};
use crate::wire::{RECEIPT_VERSION, StageCommitment, encode_f32};

const ROWS: usize = 64;
/// One q8 block.
const WIDTH: usize = 32;
const SEED: &[u8] = b"rows";

/// The weights of the toy output matrix, as q8_0 steps of 1. Tokens 0 and
/// 1 are far ahead of the rest. Token 1's row is token 0's plus a part
/// that nearly cancels on [`residual`]: 127 on values 1 to 16, and -127 on
/// values 17 and 18.
fn steps() -> Vec<[i8; WIDTH]> {
    let mut rows: Vec<[i8; WIDTH]> = (0..ROWS)
        .map(|r| std::array::from_fn(|i| ((r * 7 + i * 3) % 5) as i8 - 2))
        .collect();
    rows[0] = [0; WIDTH];
    rows[0][0] = 127;
    rows[1] = rows[0];
    rows[1][1..=16].fill(127);
    rows[1][17] = -127;
    rows[1][18] = -127;
    rows
}

/// A toy model with the q8_0 output matrix `steps`, which a CPU node meets
/// with a q8 activation. It divides its logits by `logit_scale` when
/// given.
fn model_of(steps: &[[i8; WIDTH]], logit_scale: Option<f64>) -> WeightFile {
    model_with(steps, logit_scale, Vec::new())
}

/// [`model_of`] with more metadata.
fn model_with(
    steps: &[[i8; WIDTH]],
    logit_scale: Option<f64>,
    extra: Vec<(&str, MetaValue)>,
) -> WeightFile {
    let mut meta = vec![
        ("general.architecture", MetaValue::Str("toy".into())),
        (
            "toy.attention.layer_norm_rms_epsilon",
            MetaValue::Float(0.0),
        ),
    ];
    if let Some(scale) = logit_scale {
        meta.push(("toy.logit_scale", MetaValue::Float(scale)));
    }
    meta.extend(extra);
    let (mut bytes, _) = test_gguf::header(
        &meta,
        &[
            ("output_norm.weight", vec![WIDTH as u64], GGML_F32),
            (
                "token_embd.weight",
                vec![WIDTH as u64, ROWS as u64],
                GGML_Q8_0,
            ),
        ],
    );
    for _ in 0..WIDTH {
        bytes.extend(1.0f32.to_le_bytes());
    }
    for row in steps {
        bytes.extend(half::f16::from_f32(1.0).to_le_bytes());
        bytes.extend(row.iter().map(|&step| step as u8));
    }
    WeightFile::open(Box::new(test_gguf::Memory(bytes))).expect("a test GGUF")
}

fn model() -> WeightFile {
    model_of(&steps(), None)
}

/// A residual whose q8 activation has these steps: 127 at values 0 and
/// 17, 37 at value 18, and 10.49 at values 1 to 16, which a q8 kernel
/// rounds down to 10. On the unrounded activation token 1 leads token 0
/// by 16 × 127 × 0.49 less the 508 that the rounded one leaves token 0
/// ahead by.
fn residual() -> Vec<f32> {
    let mut x = vec![1.0f32; WIDTH];
    x[0] = 127.0;
    x[1..=16].fill(10.49);
    x[17] = 127.0;
    x[18] = 37.0;
    x
}

/// A turn of two positions whose second token was sampled from the first.
fn receipt(sampled: i32) -> Receipt {
    Receipt {
        version: RECEIPT_VERSION,
        receipt_id: "r".into(),
        model: "m".into(),
        position_start: 0,
        position_end: 2,
        tokens: vec![0, sampled],
        generation_start: 1,
        final_token: Some(1),
        stages: vec![StageCommitment {
            stage_index: 0,
            layer_start: 0,
            layer_end: 1,
            root: String::new(),
            leaf_count: 0,
            backend: "CPU".into(),
            cache_types: Vec::new(),
            prefix: None,
            ubatches: Vec::new(),
            recurrent_state: None,
        }],
        binding: None,
    }
}

/// The activation the output matrix of `weights` meets at `residual`, as
/// the client norms it and as a CPU node's q8 kernel rounds it.
fn activations(weights: &WeightFile, residual: &[f32]) -> (Vec<f32>, Vec<f64>) {
    let head = OutputHead::from_weights(weights).unwrap();
    let x: Vec<f32> = head
        .norm
        .apply(residual)
        .unwrap()
        .iter()
        .map(|&v| v as f32)
        .collect();
    let rounded = Profile::CpuQ8.apply(&x);
    (x, rounded)
}

/// The logits a node computes from `activation`.
fn logits_of(weights: &WeightFile, activation: &[f64]) -> Vec<f32> {
    let head = OutputHead::from_weights(weights).unwrap();
    head.products(weights, &[activation.to_vec()])
        .unwrap()
        .remove(0)
        .iter()
        .map(|&v| v as f32)
        .collect()
}

/// The logits an honest CPU node opens at `residual`.
fn honest(weights: &WeightFile, residual: &[f32]) -> Vec<f32> {
    logits_of(weights, &activations(weights, residual).1)
}

fn best(logits: &[f32]) -> i32 {
    (0..logits.len())
        .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
        .unwrap() as i32
}

fn opening(residual: &[f32], logits: Option<&[f32]>) -> HeadOpening {
    HeadOpening {
        position: 0,
        residual: encode_f32(residual),
        path: Vec::new(),
        logits: logits.map(encode_f32),
    }
}

fn check_of(weights: &WeightFile, residual: &[f32], logits: &[f32], sampled: i32) -> HeadVerdict {
    check_head_on_every_row(
        weights,
        &receipt(sampled),
        &[opening(residual, Some(logits))],
        "CPU",
        SEED,
        (&TokenRule::Greedy, &Forcing::Free),
    )
    .unwrap()
}

fn norm_of(logits: &[f32]) -> f32 {
    logits.iter().map(|v| v * v).sum::<f32>().sqrt()
}

#[test]
fn an_honest_token_passes_where_the_clients_own_rounding_gives_another() {
    let weights = model();
    let logits = honest(&weights, &residual());
    // the node's kernel rounds the activation, and token 0 leads
    assert_eq!(best(&logits), 0);
    let verdict = check_of(&weights, &residual(), &logits, 0);
    assert!(verdict.pass && verdict.ties == 0, "{verdict:?}");
    let position = &verdict.positions[0];
    assert_eq!((position.best, position.within_band), (0, false));
    let checked = position.logits.unwrap();
    assert!(checked.every_row && checked.pass, "{checked:?}");
    assert!(checked.error < 1e-6 && checked.row_error < 1e-6);
    assert_eq!((checked.profile, checked.rounding_flips), ("cpu_q8_0", 0));

    // logits computed without that rounding put token 1 ahead by far more
    // than the tie band, so a check that judged on them refused the honest
    // node
    let computed = check_computed_head(
        &weights,
        &receipt(0),
        &[opening(&residual(), Some(&logits))],
        (&TokenRule::Greedy, &Forcing::Free),
    )
    .unwrap();
    assert_eq!(computed.positions[0].best, 1);
    assert!(-computed.positions[0].margin > 10.0 * computed.positions[0].band);
    assert!(!computed.pass);

    // a recheck judges the node's logits
    let rechecked = check_head(
        &weights,
        &receipt(0),
        &[opening(&residual(), Some(&logits))],
        "CPU",
        SEED,
        (&TokenRule::Greedy, &Forcing::Free),
    )
    .unwrap();
    assert_eq!(rechecked, verdict);
    // and it holds the node to them: token 1 is not their best
    assert!(!check_of(&weights, &residual(), &logits, 1).pass);
}

#[test]
fn a_logit_moved_past_the_tolerance_fails() {
    let weights = model();
    let logits = honest(&weights, &residual());
    let norm = norm_of(&logits);
    // a row that the row check does not read
    let unread = (2..ROWS)
        .find(|&row| !picked_rows(SEED, ROWS as u64).contains(&(row as u64)))
        .unwrap();
    for (moved_by, passes) in [(0.1, true), (3.0, false), (-3.0, false), (1e4, false)] {
        let mut changed = logits.clone();
        changed[unread] += moved_by * EVERY_ROW_TOLERANCE as f32 * norm;
        let verdict = check_of(&weights, &residual(), &changed, 0);
        let checked = verdict.positions[0].logits.unwrap();
        assert_eq!(checked.pass, passes, "{moved_by}: {checked:?}");
        assert_eq!(verdict.pass, passes, "{moved_by}");
        if !passes {
            // only the difference over every row shows it
            assert!(checked.error > EVERY_ROW_TOLERANCE, "{checked:?}");
        }
    }
    // a logit moved to make another token the best fails as well, though
    // that token is then the best of the opened logits
    let mut steered = logits.clone();
    steered[1] = steered[0] + 1.0;
    let verdict = check_of(&weights, &residual(), &steered, 1);
    assert!(!verdict.pass && !verdict.positions[0].logits.unwrap().pass);
    // and so does every logit moved by the same small factor
    let scaled: Vec<f32> = logits.iter().map(|v| v * 1.0001).collect();
    assert!(!check_of(&weights, &residual(), &scaled, 0).pass);
}

fn fault(weights: &WeightFile, opening: HeadOpening) -> String {
    let error = check_head_on_every_row(
        weights,
        &receipt(0),
        &[opening],
        "CPU",
        SEED,
        (&TokenRule::Greedy, &Forcing::Free),
    )
    .unwrap_err();
    assert!(NodeFault::find(&error).is_some(), "{error:#}");
    format!("{error:#}")
}

#[test]
fn logits_left_out_or_malformed_are_the_nodes_fault() {
    let weights = model();
    let logits = honest(&weights, &residual());
    // an audit with a key blames the node for these too
    assert!(fault(&weights, opening(&residual(), None)).contains("opened no logits"));
    assert!(fault(&weights, opening(&residual(), Some(&logits[1..]))).contains("63 logits for 64"));
    let mut broken = logits.clone();
    broken[5] = f32::NAN;
    assert!(fault(&weights, opening(&residual(), Some(&broken))).contains("not a number"));
    assert!(fault(&weights, opening(&residual()[1..], Some(&logits))).contains("output norm"));
    // a token outside the vocabulary
    let error = check_head_on_every_row(
        &weights,
        &receipt(ROWS as i32),
        &[opening(&residual(), Some(&logits))],
        "CPU",
        SEED,
        (&TokenRule::Greedy, &Forcing::Free),
    )
    .unwrap_err();
    assert!(NodeFault::find(&error).is_some(), "{error:#}");
    // a backend without rounding profiles cannot be checked, which is not
    // the node's fault
    let error = check_head_on_every_row(
        &weights,
        &receipt(0),
        &[opening(&residual(), Some(&logits))],
        "Metal",
        SEED,
        (&TokenRule::Greedy, &Forcing::Free),
    )
    .unwrap_err();
    assert!(NodeFault::find(&error).is_none(), "{error:#}");
}

#[test]
fn logits_at_the_models_scale_pass_and_at_another_fail() {
    let scaled = model_of(&steps(), Some(4.0));
    let logits = honest(&scaled, &residual());
    // some architectures divide by the scale and some multiply
    for factor in [0.25f32, 4.0] {
        let opened: Vec<f32> = logits.iter().map(|v| v * factor).collect();
        let verdict = check_of(&scaled, &residual(), &opened, 0);
        assert!(verdict.pass, "{factor}: {verdict:?}");
    }
    let other: Vec<f32> = logits.iter().map(|v| v * 2.0).collect();
    assert!(!check_of(&scaled, &residual(), &other, 0).pass);
    // a model without a scale takes its logits as they are
    let unscaled = model();
    let opened: Vec<f32> = logits.iter().map(|v| v * 0.25).collect();
    assert!(!check_of(&unscaled, &residual(), &opened, 0).pass);
}

/// The node's activation when it rounded value `index` the other way, with
/// the residual that puts that value on a rounding boundary.
fn flipped(weights: &WeightFile, index: usize) -> (Vec<f32>, Vec<f64>) {
    let mut residual = residual();
    // halfway between steps 10 and 11
    residual[index] = 10.5;
    let (x, mut rounded) = activations(weights, &residual);
    let flip = rounding_flips::flips(Profile::CpuQ8, &x, &rounded)
        .into_iter()
        .find(|flip| flip.changes.len() == 1 && flip.changes[0].0 == index)
        .expect("the value lies on a boundary");
    rounded[index] += flip.changes[0].1;
    (residual, rounded)
}

#[test]
fn a_rounding_the_node_took_the_other_way_passes() {
    let weights = model();
    let (tied, node) = flipped(&weights, 20);
    let logits = logits_of(&weights, &node);
    let verdict = check_of(&weights, &tied, &logits, best(&logits));
    let checked = verdict.positions[0].logits.unwrap();
    assert!(verdict.pass && checked.error < 1e-6, "{checked:?}");
    assert_eq!(checked.rounding_flips, 1);
    // the same logits at a residual without that boundary are too far off
    let plain = check_of(&weights, &residual(), &logits, best(&logits));
    assert!(!plain.pass && plain.positions[0].logits.unwrap().rounding_flips == 0);
}

#[test]
fn a_flip_that_only_other_rows_show_is_found_from_them() {
    // value 20 has a weight in one row alone, which no guide row is
    let guide: Vec<u64> = picked_rows(SEED, ROWS as u64)
        .into_iter()
        .chain(drawn_rows(
            SEED,
            "rows/output-head-flips",
            GUIDE_ROWS,
            ROWS as u64,
        ))
        .collect();
    let lone = (2..ROWS)
        .find(|&row| !guide.contains(&(row as u64)))
        .expect("a row outside the guide rows");
    let mut steps = steps();
    for (row, weights) in steps.iter_mut().enumerate() {
        weights[20] = if row == lone { 100 } else { 0 };
    }
    let weights = model_of(&steps, None);
    let (tied, node) = flipped(&weights, 20);
    let logits = logits_of(&weights, &node);
    let verdict = check_of(&weights, &tied, &logits, 0);
    let checked = verdict.positions[0].logits.unwrap();
    assert!(verdict.pass, "{checked:?}");
    assert_eq!(checked.rounding_flips, 1);
}

#[test]
fn a_soft_capped_models_tokens_are_not_judged() {
    // its opened logits are not the matrix's product, and logits the
    // client computes lack the node's rounding: judged on them, the honest
    // token 0 here failed
    let capped = model_with(
        &steps(),
        None,
        vec![("toy.final_logit_softcapping", MetaValue::Float(30.0))],
    );
    let no_end = |_: i32| -> Result<bool> { Ok(false) };
    let rule = crate::head::token_rule_for(&capped, &receipt(0), &no_end);
    assert!(matches!(rule, TokenRule::Unjudged(_)), "{rule:?}");
    let logits = honest(&model(), &residual());
    let verdict = check_head(
        &capped,
        &receipt(0),
        &[opening(&residual(), Some(&logits))],
        "CPU",
        SEED,
        (&rule, &Forcing::Free),
    )
    .unwrap();
    assert!(verdict.pass, "{verdict:?}");
    assert!(verdict.positions.iter().all(|p| !p.judged));
    // a residual that does not fit the head is still the node's fault
    let error = check_head(
        &capped,
        &receipt(0),
        &[opening(&[1.0; 3], None)],
        "CPU",
        SEED,
        (&rule, &Forcing::Free),
    )
    .unwrap_err();
    assert!(NodeFault::find(&error).is_some(), "{error:#}");
}

#[test]
fn suppressed_and_ignored_end_tokens_are_taken_out_before_the_judgement() {
    let logits = honest(&model(), &residual());
    assert_eq!(best(&logits), 0);
    let no_end = |_: i32| -> Result<bool> { Ok(false) };
    // the model suppresses token 0: an honest node took the next best
    let suppressing = model_with(
        &steps(),
        None,
        vec![("tokenizer.ggml.suppress_tokens", MetaValue::Tokens(vec![0]))],
    );
    let rule = crate::head::token_rule_for(&suppressing, &receipt(1), &no_end);
    let check = |weights: &WeightFile, rule: &TokenRule, sampled: i32| {
        check_head_on_every_row(
            weights,
            &receipt(sampled),
            &[opening(&residual(), Some(&logits))],
            "CPU",
            SEED,
            (rule, &Forcing::Free),
        )
        .unwrap()
    };
    assert!(check(&suppressing, &rule, 1).pass);
    assert!(!check(&suppressing, &rule, 0).pass);
    // a turn that ignores its end loses the end tokens the same way
    let mut ignoring = receipt(1);
    ignoring.binding = Some(crate::turn::TurnBinding {
        client_key: String::new(),
        parent: None,
        runtime: String::new(),
        sampler: turn_sampler(serde_json::json!({"temperature": 0.0, "ignore_eos": true})),
        node_key: String::new(),
        turn_root: String::new(),
        signature: String::new(),
    });
    let end = |token: i32| -> Result<bool> { Ok(token == 0) };
    let rule = crate::head::token_rule_for(&model(), &ignoring, &end);
    assert!(check(&model(), &rule, 1).pass, "{rule:?}");
    assert!(!check(&model(), &rule, 0).pass);
    // a vocabulary that cannot name them leaves the tokens unjudged
    let unknown = |_: i32| -> Result<bool> { anyhow::bail!("no vocabulary") };
    let rule = crate::head::token_rule_for(&model(), &ignoring, &unknown);
    assert!(matches!(rule, TokenRule::Unjudged(_)));
}

/// A sampled statement of the default chain with `overrides`.
fn turn_sampler(overrides: serde_json::Value) -> String {
    let mut p = serde_json::json!({
        "seed": 7, "draw": null,
        "samplers": ["top_k", "top_p", "min_p", "temperature"],
        "temperature": 0.8, "top_k": 40, "top_p": 0.95, "min_p": 0.05,
        "logit_bias": [], "grammar": null,
    });
    for (k, v) in overrides.as_object().unwrap() {
        p[k] = v.clone();
    }
    crate::turn::sampled(&p)
}
