use super::*;
use crate::weights::{MetaValue, test_gguf};
use crate::wire::{RECEIPT_VERSION, StageCommitment, encode_f32};

const ROWS: u64 = 64;
const WIDTH: u64 = 8;

fn model() -> WeightFile {
    model_with(None)
}

/// The toy model, dividing its logits by `logit_scale` when given.
fn model_with(logit_scale: Option<f64>) -> WeightFile {
    let embd: Vec<f32> = (0..ROWS * WIDTH)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
        .collect();
    let mut meta = vec![
        ("general.architecture", MetaValue::Str("toy".into())),
        (
            "toy.attention.layer_norm_rms_epsilon",
            MetaValue::Float(1e-6),
        ),
    ];
    if let Some(scale) = logit_scale {
        meta.push(("toy.logit_scale", MetaValue::Float(scale)));
    }
    test_gguf::open(
        &meta,
        &[
            ("output_norm.weight", vec![WIDTH], vec![1.0; WIDTH as usize]),
            ("token_embd.weight", vec![WIDTH, ROWS], embd),
        ],
    )
}

fn residual() -> Vec<f32> {
    (0..WIDTH).map(|i| i as f32 - 3.5).collect()
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

/// The honest logits at the residual, and their best token.
fn honest(weights: &WeightFile) -> (Vec<f32>, i32) {
    let head = OutputHead::from_weights(weights).unwrap();
    let logits = head.logits(weights, &[residual()]).unwrap().remove(0);
    let best = (0..logits.len())
        .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
        .unwrap();
    (logits.iter().map(|&v| v as f32).collect(), best as i32)
}

fn opening(logits: &[f32]) -> HeadOpening {
    HeadOpening {
        position: 0,
        residual: encode_f32(&residual()),
        path: Vec::new(),
        logits: Some(encode_f32(logits)),
    }
}

fn check(weights: &WeightFile, key: &HeadKey, logits: &[f32], sampled: i32) -> HeadVerdict {
    check_head_with_key(
        weights,
        key,
        &receipt(sampled),
        &[opening(logits)],
        "CPU",
        b"rows",
        (&TokenRule::Greedy, &Forcing::Free),
    )
    .unwrap()
}

#[test]
fn honest_logits_pass_the_key_check() {
    let weights = model();
    let key = HeadKey::build(&weights, [7; 32]).unwrap();
    let (logits, best) = honest(&weights);
    let verdict = check(&weights, &key, &logits, best);
    assert!(verdict.pass, "{verdict:?}");
    let checked = verdict.positions[0].logits.unwrap();
    assert!(checked.error < 1e-6, "{checked:?}");
}

#[test]
fn one_changed_logit_fails_the_key_check() {
    let weights = model();
    let key = HeadKey::build(&weights, [7; 32]).unwrap();
    let (mut logits, best) = honest(&weights);
    // lower the honest best below another token, which the node then samples
    let (other, _) = logits
        .iter()
        .enumerate()
        .filter(|&(i, _)| i as i32 != best)
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap();
    logits[best as usize] = logits[other] - 0.5;
    let verdict = check(&weights, &key, &logits, other as i32);
    assert!(!verdict.pass);
    assert!(verdict.positions[0].logits.unwrap().error > KEY_TOLERANCE);
}

#[test]
fn a_token_that_is_not_the_best_of_the_opened_logits_fails() {
    let weights = model();
    let key = HeadKey::build(&weights, [7; 32]).unwrap();
    let (logits, best) = honest(&weights);
    let worst = (0..logits.len())
        .min_by(|&a, &b| logits[a].total_cmp(&logits[b]))
        .unwrap();
    assert_ne!(worst as i32, best);
    let verdict = check(&weights, &key, &logits, worst as i32);
    assert!(!verdict.pass);
    assert!(verdict.positions[0].logits.unwrap().pass);
}

#[test]
fn a_key_survives_a_round_trip_and_fits_only_its_matrix() {
    let weights = model();
    let key = HeadKey::build(&weights, [9; 32]).unwrap();
    let text = serde_json::to_string(&key).unwrap();
    let back: HeadKey = serde_json::from_str(&text).unwrap();
    assert_eq!(back, key);
    assert!(back.fits(&weights));
    let other = test_gguf::open(
        &[],
        &[(
            "token_embd.weight",
            vec![WIDTH, ROWS + 1],
            vec![0.0; ((ROWS + 1) * WIDTH) as usize],
        )],
    );
    assert!(!back.fits(&other));
}

#[test]
fn logits_divided_by_the_models_logit_scale_pass() {
    let weights = model_with(Some(8.0));
    let key = HeadKey::build(&weights, [7; 32]).unwrap();
    let (logits, best) = honest(&weights);
    let scaled: Vec<f32> = logits.iter().map(|v| v / 8.0).collect();
    let verdict = check(&weights, &key, &scaled, best);
    assert!(verdict.pass, "{verdict:?}");
}

#[test]
fn scaled_logits_fail_when_the_model_has_no_scale() {
    let weights = model();
    let key = HeadKey::build(&weights, [7; 32]).unwrap();
    let (logits, best) = honest(&weights);
    let scaled: Vec<f32> = logits.iter().map(|v| v / 8.0).collect();
    let verdict = check(&weights, &key, &scaled, best);
    assert!(!verdict.pass, "{verdict:?}");
}

fn toy_key() -> Key {
    Key {
        signs: vec![0b0101, 0b0011, 0b1110],
        vectors: vec![
            vec![1.0, -2.0, 0.5],
            vec![0.25, 3.0, -1.0],
            vec![-1.5, 0.75, 2.0],
            vec![2.0, 1.0, 1.0],
        ],
    }
}

#[test]
fn a_rounding_the_node_took_the_other_way_explains_the_key_residuals() {
    let key = toy_key();
    let flips = vec![
        Flip {
            changes: vec![(0, 0.01)],
        },
        Flip {
            changes: vec![(1, -0.02), (2, 0.004)],
        },
    ];
    // the node's activation had the second flip
    let residuals: Vec<f64> = key
        .vectors
        .iter()
        .map(|v| v[1] * -0.02 + v[2] * 0.004)
        .collect();
    let (taken, left) = explained(&key, &residuals, &flips);
    assert_eq!(taken, vec![1]);
    assert!(left < 1e-12);
    // both of them
    let both: Vec<f64> = key
        .vectors
        .iter()
        .zip(&residuals)
        .map(|(v, r)| r + v[0] * 0.01)
        .collect();
    assert_eq!(explained(&key, &both, &flips).0, vec![0, 1]);
}

#[test]
fn residuals_no_flip_explains_are_left_as_they_are() {
    let key = toy_key();
    let flips = vec![Flip {
        changes: vec![(0, 0.01)],
    }];
    // a changed logit moves every residual by its sign, which no
    // activation flip reproduces
    let residuals = [0.3, -0.3, 0.3, 0.3];
    let (taken, left) = explained(&key, &residuals, &flips);
    assert!(taken.len() <= 1);
    assert!(left > 0.25, "{left}");
    // a residual that is not a number is left as the worst there is
    let (_, left) = explained(&key, &[0.0, f64::NAN, 0.0, 0.0], &flips);
    assert_eq!(left, f64::INFINITY);
}

/// The node's fault that checking `opening` with a key finds.
fn fault(opening: HeadOpening) -> String {
    use crate::node_fault::NodeFault;
    let weights = model();
    let key = HeadKey::build(&weights, [7; 32]).unwrap();
    let (_, best) = honest(&weights);
    let error = check_head_with_key(
        &weights,
        &key,
        &receipt(best),
        &[opening],
        "CPU",
        b"rows",
        (&TokenRule::Greedy, &Forcing::Free),
    )
    .unwrap_err();
    assert!(
        NodeFault::find(&error).is_some(),
        "not blamed on the node: {error:#}"
    );
    format!("{error:#}")
}

#[test]
fn logits_left_out_are_the_nodes_fault() {
    // Without logits the client would read the whole output matrix and
    // compute every logit, on every core: a node may not choose that.
    let (logits, _) = honest(&model());
    let message = fault(HeadOpening {
        logits: None,
        ..opening(&logits)
    });
    assert!(message.contains("opened no logits"), "{message}");
}

#[test]
fn logits_or_a_residual_that_do_not_fit_are_the_nodes_fault() {
    let (logits, _) = honest(&model());
    let message = fault(opening(&logits[..logits.len() - 1]));
    assert!(message.contains("63 logits for 64 tokens"), "{message}");
    for bad in [f32::NAN, f32::INFINITY] {
        let mut changed = logits.clone();
        changed[5] = bad;
        let message = fault(opening(&changed));
        assert!(message.contains("not a number"), "{message}");
        let mut residual = residual();
        residual[2] = bad;
        let message = fault(HeadOpening {
            residual: encode_f32(&residual),
            ..opening(&logits)
        });
        assert!(message.contains("not a number"), "{message}");
    }
    let message = fault(HeadOpening {
        residual: encode_f32(&residual()[..7]),
        ..opening(&logits)
    });
    assert!(message.contains("7 values"), "{message}");
    fault(HeadOpening {
        logits: Some("not base64".into()),
        ..opening(&logits)
    });
}

#[test]
fn a_sampled_token_outside_the_vocabulary_is_the_nodes_fault() {
    use crate::node_fault::NodeFault;
    let weights = model();
    let key = HeadKey::build(&weights, [7; 32]).unwrap();
    let (logits, _) = honest(&weights);
    for token in [ROWS as i32, -1] {
        let error = check_head_with_key(
            &weights,
            &key,
            &receipt(token),
            &[opening(&logits)],
            "CPU",
            b"rows",
            (&TokenRule::Greedy, &Forcing::Free),
        )
        .unwrap_err();
        assert!(NodeFault::find(&error).is_some(), "{token}: {error:#}");
    }
}

#[test]
fn a_key_does_not_print_its_secrets() {
    let key = HeadKey::build(&model(), [0xab; 32]).unwrap();
    let printed = format!("{key:?} {key:#?}");
    assert!(printed.contains("token_embd.weight"), "{printed}");
    assert!(!printed.contains("abab"), "{printed}");
    assert!(!printed.contains(&key.vectors[0]), "{printed}");
    assert!(!printed.contains("seed") && !printed.contains("vectors"));
}

#[test]
fn a_key_that_parses_but_cannot_check_does_not_fit() {
    // such a key would be kept and leave the head unchecked for good
    let weights = model();
    let key = HeadKey::build(&weights, [9; 32]).unwrap();
    assert!(key.fits(&weights));
    let changed = |change: &dyn Fn(&mut serde_json::Value)| -> HeadKey {
        let mut value = serde_json::to_value(&key).unwrap();
        change(&mut value);
        serde_json::from_value(value).unwrap()
    };
    let broken = [
        changed(&|key| key["seed"] = "not hex".into()),
        changed(&|key| key["seed"] = "abcd".into()),
        changed(&|key| {
            key["vectors"].as_array_mut().unwrap().pop();
        }),
        changed(&|key| key["vectors"][1] = "AAAAAAAAAAA=".into()),
        changed(&|key| key["vectors"][2] = "@@".into()),
        // a vector that holds a value that is not a number
        changed(&|key| {
            let mut vector = vec![1.0f64; WIDTH as usize];
            vector[3] = f64::NAN;
            key["vectors"][0] = encode_f64(&vector).into();
        }),
    ];
    for (i, key) in broken.iter().enumerate() {
        assert!(!key.fits(&weights), "broken key {i} fits");
    }
}

#[test]
fn two_changes_that_cancel_in_a_few_vectors_show_in_the_rest() {
    // a node that lowers one logit and raises another by as much moves
    // `r · y` by nothing in every vector where their signs agree: with 4
    // vectors one pair of rows in 16 agreed in all
    let rows = 4096u64;
    let signs = signs(&[9; 32], rows);
    let first4 = |row: usize| signs[row] & 0xF;
    let (a, b) = (1..rows as usize)
        .find_map(|b| (first4(0) == first4(b)).then_some((0, b)))
        .expect("a pair that agrees in the first 4 vectors");
    let key = Key {
        signs,
        vectors: vec![Vec::new(); KEY_VECTORS],
    };
    let y = vec![0.5f32; rows as usize];
    let mut moved = y.clone();
    moved[a] -= 3.0;
    moved[b] += 3.0;
    let shifts: Vec<f64> = (0..KEY_VECTORS)
        .map(|k| signed_sum(&key, k, &moved) - signed_sum(&key, k, &y))
        .collect();
    assert!(shifts[..4].iter().all(|s| s.abs() < 1e-9), "{shifts:?}");
    // 16 vectors miss such a pair 1 in 65,536 times; this one shows
    assert!(shifts.iter().any(|s| s.abs() > 1.0), "{shifts:?}");
    // agreement in all 16 is as rare as that
    let agreeing = (1..rows as usize)
        .filter(|&row| key.signs[row] == key.signs[0])
        .count();
    assert!(agreeing <= 2, "{agreeing}");
}

#[test]
fn a_key_file_of_another_layout_does_not_parse() {
    // the first layout had no version and 4 vectors: it is never read as
    // a key of this one, and a client builds a new one
    let weights = model();
    let key = HeadKey::build(&weights, [7; 32]).unwrap();
    let mut json = serde_json::to_value(&key).unwrap();
    assert_eq!(json["version"], KEY_VERSION);
    assert_eq!(json["vectors"].as_array().unwrap().len(), KEY_VECTORS);
    json.as_object_mut().unwrap().remove("version");
    assert!(serde_json::from_value::<HeadKey>(json.clone()).is_err());
    json["version"] = 1.into();
    let old: HeadKey = serde_json::from_value(json).unwrap();
    assert!(!old.fits(&weights));
}

#[test]
fn a_report_shows_the_key_error_in_coarse_steps() {
    assert_eq!(key_margin(KEY_TOLERANCE / 2.0), 0);
    assert_eq!(key_margin(KEY_TOLERANCE / 20.0), 1);
    assert_eq!(key_margin(KEY_TOLERANCE / 1e9), 3);
    assert_eq!(key_margin(0.0), 3);
    assert_eq!(key_margin(KEY_TOLERANCE * 2.0), -1);
    assert_eq!(key_margin(f64::INFINITY), -3);
    assert_eq!(key_margin(f64::NAN), -3);
}
