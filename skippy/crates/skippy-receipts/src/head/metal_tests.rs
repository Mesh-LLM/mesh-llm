//! Independent Q8 arithmetic fixtures for both head verifiers. Expected
//! values use exact binary fractions, never the production rounding helpers.

use super::every_row::check_head_on_every_row;
use super::key::{HeadKey, KEY_TOLERANCE, check_head_with_key};
use super::reasoning_budget::Forcing;
use super::{HeadVerdict, OutputHead};
use crate::ggml_types::{GGML_F32, GGML_Q8_0};
use crate::weights::{MetaValue, WeightFile, test_gguf};
use crate::wire::{HeadOpening, RECEIPT_VERSION, Receipt, encode_f32};

const WIDTH: usize = 32;
const ROWS: usize = 64;
const QUANTS: [i8; 8] = [3, 5, 7, 9, 11, 13, 15, 17];
// q * (1 + 2^-10), rounded to the half grid, nearest-even.
const HALF_WEIGHTS: [f64; 8] = [
    3.00390625, 5.00390625, 7.0078125, 9.0078125, 11.0078125, 13.015625, 15.015625, 17.015625,
];

fn entry(row: usize, col: usize) -> (usize, f64) {
    (
        (row * 3 + col * 5) % 8,
        if (row + col).is_multiple_of(3) {
            -1.0
        } else {
            1.0
        },
    )
}

fn activation(col: usize) -> f32 {
    if col.is_multiple_of(2) {
        1.0009
    } else {
        -0.50045
    }
}

fn model() -> WeightFile {
    model_with_first(activation(0))
}

fn model_with_first(first: f32) -> WeightFile {
    let meta = [
        ("general.architecture", MetaValue::Str("toy".into())),
        (
            "toy.attention.layer_norm_rms_epsilon",
            MetaValue::Float(0.0),
        ),
    ];
    let (mut bytes, _) = test_gguf::header(
        &meta,
        &[
            ("output_norm.weight", vec![WIDTH as u64], GGML_F32),
            ("output.weight", vec![WIDTH as u64, ROWS as u64], GGML_Q8_0),
        ],
    );
    // An all-ones residual and zero epsilon make the activation exactly
    // these norm weights, independent of production normalization rounding.
    for col in 0..WIDTH {
        bytes.extend(if col == 0 { first } else { activation(col) }.to_le_bytes());
    }
    for row in 0..ROWS {
        bytes.extend(0x3c01u16.to_le_bytes()); // half 1 + 2^-10
        for col in 0..WIDTH {
            let (index, sign) = entry(row, col);
            bytes.push((QUANTS[index] * sign as i8) as u8);
        }
    }
    WeightFile::open(Box::new(test_gguf::Memory(bytes))).unwrap()
}

fn logits(profile: &str) -> Vec<f32> {
    logits_with_first(profile, None)
}

fn logits_with_first(profile: &str, first: Option<f64>) -> Vec<f32> {
    (0..ROWS)
        .map(|row| {
            (0..WIDTH)
                .map(|col| {
                    let (index, sign) = entry(row, col);
                    let (w, x) = match profile {
                        "raw" => (
                            f64::from(QUANTS[index]) * (1.0 + 1.0 / 1024.0),
                            f64::from(activation(col)),
                        ),
                        "f16_weights" => (
                            HALF_WEIGHTS[index],
                            if col.is_multiple_of(2) {
                                1.0009765625
                            } else {
                                -0.50048828125
                            },
                        ),
                        "f16_weights_rtz" => (
                            HALF_WEIGHTS[index],
                            if col.is_multiple_of(2) { 1.0 } else { -0.5 },
                        ),
                        _ => panic!("unknown fixture"),
                    };
                    let x = if col == 0 { first.unwrap_or(x) } else { x };
                    sign * w * x
                })
                .sum::<f64>() as f32
        })
        .collect()
}

fn receipt_and_opening(logits: &[f32]) -> (Receipt, HeadOpening) {
    let sampled = (0..ROWS)
        .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
        .unwrap() as i32;
    let receipt = Receipt {
        version: RECEIPT_VERSION,
        receipt_id: "metal-head".into(),
        model: "toy".into(),
        position_start: 0,
        position_end: 2,
        tokens: vec![0, sampled],
        generation_start: 1,
        final_token: None,
        stages: Vec::new(),
        binding: None,
    };
    let opening = HeadOpening {
        position: 0,
        residual: encode_f32(&[1.0; WIDTH]),
        path: Vec::new(),
        logits: Some(encode_f32(logits)),
    };
    (receipt, opening)
}

fn checks(weights: &WeightFile, key: &HeadKey, logits: &[f32]) -> [HeadVerdict; 2] {
    let (receipt, opening) = receipt_and_opening(logits);
    [
        check_head_with_key(
            weights,
            key,
            &receipt,
            std::slice::from_ref(&opening),
            "MTL0",
            b"metal-head-rows",
            &Forcing::Free,
        )
        .unwrap(),
        check_head_on_every_row(
            weights,
            &receipt,
            &[opening],
            "MTL0",
            b"metal-head-rows",
            &Forcing::Free,
        )
        .unwrap(),
    ]
}

#[test]
fn honest_raw_rte_and_rtz_heads_pass_both_verifiers() {
    let weights = model();
    let key = HeadKey::build(&weights, [7; 32]).unwrap();
    for profile in ["raw", "f16_weights", "f16_weights_rtz"] {
        for verdict in checks(&weights, &key, &logits(profile)) {
            let check = verdict.positions[0].logits.unwrap();
            assert!(verdict.pass, "{profile}: {verdict:?}");
            assert!(check.error < 1e-6 && check.row_error < 1e-6, "{check:?}");
            assert_eq!(check.profile, profile);
        }
    }
}

#[test]
fn rounded_weight_flips_repair_both_head_checks() {
    for (profile, boundary, other_side) in [
        ("f16_weights", 1.0 + 1.0 / 2048.0, 1.0009765625),
        ("f16_weights_rtz", 1.0, 0.99951171875),
    ] {
        let weights = model_with_first(boundary);
        let key = HeadKey::build(&weights, [7; 32]).unwrap();
        let honest = logits_with_first(profile, Some(other_side));
        for verdict in checks(&weights, &key, &honest) {
            let check = verdict.positions[0].logits.unwrap();
            assert!(verdict.pass && check.error < 1e-6, "{profile}: {verdict:?}");
            assert_eq!(check.profile, profile);
            assert_eq!(check.rounding_flips, 1);
        }
        let mut changed = honest;
        changed[19] += 1.0;
        for verdict in checks(&weights, &key, &changed) {
            assert!(!verdict.pass, "{profile}: repaired a changed logit");
        }
    }
}

#[test]
fn changed_logits_fail_both_verifiers_for_each_profile() {
    let weights = model();
    let key = HeadKey::build(&weights, [7; 32]).unwrap();
    for profile in ["raw", "f16_weights", "f16_weights_rtz"] {
        let mut changed = logits(profile);
        changed[19] += 1.0;
        for verdict in checks(&weights, &key, &changed) {
            let check = verdict.positions[0].logits.unwrap();
            assert!(!verdict.pass && !check.pass, "{profile}: {verdict:?}");
            assert!(check.error > KEY_TOLERANCE, "{check:?}");
        }
    }
}

#[test]
fn mixed_products_keep_raw_and_rounded_weights_separate() {
    let weights = model();
    let head = OutputHead::from_weights(&weights).unwrap();
    let raw: Vec<f64> = (0..WIDTH).map(|c| f64::from(activation(c))).collect();
    let rtz: Vec<f64> = (0..WIDTH)
        .map(|c| if c.is_multiple_of(2) { 1.0 } else { -0.5 })
        .collect();
    let products = head
        .products_with_rounding(&weights, &[(&raw, false), (&rtz, true)])
        .unwrap();
    for (actual, profile) in products.iter().zip(["raw", "f16_weights_rtz"]) {
        assert_eq!(
            actual.iter().map(|&v| v as f32).collect::<Vec<_>>(),
            logits(profile)
        );
    }
}

#[test]
fn legacy_q8_keys_rebuild_and_new_keys_round_trip() {
    let weights = model();
    let key = HeadKey::build(&weights, [7; 32]).unwrap();
    let json = serde_json::to_value(&key).unwrap();
    let back: HeadKey = serde_json::from_value(json.clone()).unwrap();
    assert_eq!(key, back);
    assert!(back.fits(&weights));
    let mut legacy = json;
    legacy.as_object_mut().unwrap().remove("f16_vectors");
    let legacy: HeadKey = serde_json::from_value(legacy).unwrap();
    assert!(!legacy.fits(&weights)); // load_or_build's existing rebuild predicate
    let (receipt, opening) = receipt_and_opening(&logits("f16_weights_rtz"));
    assert!(
        check_head_with_key(
            &weights,
            &legacy,
            &receipt,
            &[opening],
            "MTL0",
            b"rows",
            &Forcing::Free
        )
        .is_err()
    );
    for verdict in checks(&weights, &back, &logits("f16_weights_rtz")) {
        assert!(verdict.pass);
    }
}

#[test]
fn malformed_rounded_vectors_never_fit_the_cache() {
    let weights = model();
    let key = HeadKey::build(&weights, [7; 32]).unwrap();
    let json = serde_json::to_value(&key).unwrap();
    let bad_vectors = [
        "@@".to_owned(),
        crate::wire::encode_bytes(&[0; WIDTH * 8 - 8]),
        crate::wire::encode_bytes(&[0; WIDTH * 8 + 1]),
        crate::wire::encode_bytes(&[f64::NAN.to_le_bytes(); WIDTH].concat()),
    ];
    for bad in bad_vectors {
        let mut value = json.clone();
        value["f16_vectors"][0] = bad.into();
        let broken: HeadKey = serde_json::from_value(value).unwrap();
        assert!(!broken.fits(&weights));
    }
    let mut value = json;
    value["f16_vectors"].as_array_mut().unwrap().pop();
    let broken: HeadKey = serde_json::from_value(value).unwrap();
    assert!(!broken.fits(&weights));
    let debug = format!("{key:?}");
    assert!(!debug.contains("f16_vectors"));
}
