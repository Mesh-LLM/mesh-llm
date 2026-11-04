use super::*;

/// Shared with the node's test (`tests/test-skippy-receipt-draw.cpp` in the
/// llama.cpp patch): the seed, the index, and the uniform times 2^53.
const VECTORS: [(u32, u64, u64); 9] = [
    (0, 0, 0x1c_4415_072f_63b9),
    (0, 1, 0x12_2145_bd91_204b),
    (0, 4096, 0x1a_e753_47b2_83cf),
    (1, 0, 0x18_858b_4354_7040),
    (1, 7, 0x04_f9e2_e0f8_dc3a),
    (42, 1, 0x0d_32e6_600c_0afe),
    (42, 4_294_967_295, 0x1f_14ea_3899_a69f),
    (4_294_967_295, 4096, 0x00_ca3f_ff38_9d1d),
    (123_456_789, 7, 0x1f_6601_39cb_7f55),
];

#[test]
fn the_uniform_is_the_one_the_node_computes() {
    for (seed, index, bits) in VECTORS {
        let u = uniform(seed, index);
        assert_eq!(u, bits as f64 / (1u64 << 53) as f64, "{seed} {index}");
        assert!((0.0..1.0).contains(&u));
    }
    // only the low 32 bits of the index enter
    assert_eq!(uniform(7, 5), uniform(7, (1 << 32) + 5));
}

/// A seed and index whose uniform lies in `range`.
fn index_with_uniform(seed: u32, range: std::ops::Range<f64>) -> u64 {
    (0..10_000)
        .find(|&index| range.contains(&uniform(seed, index)))
        .expect("an index in the range")
}

const DRAW: IndexedDraw = IndexedDraw {
    seed: 42,
    temperature: 1.0,
};

/// Weights 1, 2, 1 for tokens 0, 1, 2: boundaries at 0.25 and 0.75.
fn three() -> Vec<f64> {
    vec![0.0, 2f64.ln(), 0.0]
}

#[test]
fn a_draw_picks_the_token_whose_interval_holds_the_uniform() {
    let logits = three();
    for (token, range) in [(0, 0.0..0.25), (1, 0.25..0.75), (2, 0.75..1.0)] {
        let inner = (range.start + 0.05)..(range.end - 0.05);
        let index = index_with_uniform(DRAW.seed, inner);
        let verdict = judge(DRAW, &[], &logits, token, index, 0.0);
        assert!(verdict.pass && verdict.exact, "{token}: {verdict:?}");
        assert!((verdict.lower - range.start).abs() < 1e-12);
        assert!((verdict.upper - range.end).abs() < 1e-12);
        // every other token fails at that uniform
        for other in (0..3).filter(|&other| other != token) {
            let verdict = judge(DRAW, &[], &logits, other, index, 0.0);
            assert!(!verdict.pass && !verdict.exact, "{other}: {verdict:?}");
        }
    }
}

#[test]
fn the_band_admits_only_a_draw_near_a_boundary() {
    let logits = three();
    // just above the boundary at 0.25: token 1 on these logits
    let index = index_with_uniform(DRAW.seed, 0.2501..0.2520);
    let exact = judge(DRAW, &[], &logits, 1, index, 0.0);
    assert!(exact.pass && exact.exact);
    // token 0 passes only with a band that can move the boundary that far
    assert!(!judge(DRAW, &[], &logits, 0, index, 0.0).pass);
    let banded = judge(DRAW, &[], &logits, 0, index, 0.01);
    assert!(banded.pass && !banded.exact, "{banded:?}");
    assert!(banded.upper > 0.252 && banded.upper < 0.27);
    // token 2's interval starts near 0.75, band or not
    assert!(!judge(DRAW, &[], &logits, 2, index, 0.01).pass);
    // far from a boundary the band changes nothing
    let index = index_with_uniform(DRAW.seed, 0.45..0.55);
    assert!(!judge(DRAW, &[], &logits, 0, index, 0.01).pass);
    assert!(judge(DRAW, &[], &logits, 1, index, 0.01).exact);
}

#[test]
fn the_draw_is_over_the_kept_candidates_alone() {
    // top-k 2 keeps tokens 1 and 3: weights 2 and 1, one boundary at 2/3
    let logits = vec![-5.0, 2f64.ln(), -6.0, 0.0];
    let filters = [Filter::TopK(2)];
    let low = index_with_uniform(DRAW.seed, 0.1..0.6);
    let high = index_with_uniform(DRAW.seed, 0.7..0.95);
    assert!(judge(DRAW, &filters, &logits, 1, low, 0.0).exact);
    assert!(judge(DRAW, &filters, &logits, 3, high, 0.0).exact);
    assert!(!judge(DRAW, &filters, &logits, 3, low, 0.0).pass);
    // a token the filters drop is not picked at any uniform
    for index in [low, high] {
        assert!(!judge(DRAW, &filters, &logits, 0, index, 0.0).pass);
        assert!(!judge(DRAW, &filters, &logits, 2, index, 0.0).pass);
    }
}

#[test]
fn a_candidate_the_band_may_keep_or_drop_widens_the_interval() {
    // top-k 3 over four logits, the last two a hair apart: the node may
    // have kept either with the two best
    let logits = vec![3.0, 2.0, 0.0, 0.0005];
    let filters = [Filter::TopK(3)];
    // on these logits token 3 is kept, and picked from about 0.965 up
    let index = index_with_uniform(DRAW.seed, 0.97..0.999);
    let kept = judge(DRAW, &filters, &logits, 3, index, 0.0);
    assert!(kept.exact, "{kept:?}");
    // token 2 is dropped on them, and kept within a band of 1e-3
    assert!(!judge(DRAW, &filters, &logits, 2, index, 0.0).pass);
    let other = judge(DRAW, &filters, &logits, 2, index, 1e-3);
    assert!(other.pass && !other.exact, "{other:?}");
    // the best tokens are kept either way: the best one's interval still
    // ends near e/(e+1), far below the uniform
    let best = judge(DRAW, &filters, &logits, 0, index, 1e-3);
    assert!(!best.pass && best.upper < 0.74, "{best:?}");
}

#[test]
fn the_temperature_scales_the_weights_and_the_band() {
    // at temperature 0.5 the weights are 1, 4, 1: boundaries at 1/6 and 5/6
    let draw = IndexedDraw {
        temperature: 0.5,
        ..DRAW
    };
    let logits = three();
    let index = index_with_uniform(draw.seed, 0.20..0.24);
    let verdict = judge(draw, &[], &logits, 1, index, 0.0);
    assert!(verdict.exact, "{verdict:?}");
    assert!((verdict.lower - 1.0 / 6.0).abs() < 1e-12);
    assert!((verdict.upper - 5.0 / 6.0).abs() < 1e-12);
    assert!(!judge(DRAW, &[], &logits, 1, index, 0.0).pass);
}

#[test]
fn the_chain_keeps_what_the_runtime_keeps() {
    // best first: weights 8, 4, 2, 1, 1 of 16
    let sorted: Vec<f64> = [8.0f64, 4.0, 2.0, 1.0, 1.0]
        .iter()
        .map(|w| w.ln())
        .collect();
    assert_eq!(kept(&sorted, &[]), 5);
    assert_eq!(kept(&sorted, &[Filter::TopK(0)]), 5);
    assert_eq!(kept(&sorted, &[Filter::TopK(3)]), 3);
    // top-p keeps the fewest whose mass reaches p: 8/16, then 12/16
    assert_eq!(kept(&sorted, &[Filter::TopP(0.5)]), 1);
    assert_eq!(kept(&sorted, &[Filter::TopP(0.6)]), 2);
    assert_eq!(kept(&sorted, &[Filter::TopP(1.0)]), 5);
    // after top-k 2 the mass is of the two kept: 8/12 is under 0.7
    assert_eq!(kept(&sorted, &[Filter::TopK(2), Filter::TopP(0.7)]), 2);
    assert_eq!(kept(&sorted, &[Filter::TopP(0.7)]), 2);
    // min-p keeps a logit within ln(p) of the best
    assert_eq!(kept(&sorted, &[Filter::MinP(0.3)]), 2);
    assert_eq!(kept(&sorted, &[Filter::MinP(0.0)]), 5);
    // a temperature before a filter changes what it sees: at 2.0 the
    // weights are their square roots, and 0.3 of the best keeps one more
    assert_eq!(
        kept(&sorted, &[Filter::Temperature(2.0), Filter::MinP(0.3)]),
        5
    );
}

#[test]
fn a_whole_vocabulary_is_judged() {
    // every token kept, as with no filter at a temperature of 1
    let n = 150_000usize;
    let logits: Vec<f64> = (0..n).map(|i| ((i * 7919) % 1000) as f64 / 100.0).collect();
    let index = 3;
    let u = uniform(DRAW.seed, index);
    let weights: Vec<f64> = logits.iter().map(|l| (l - 9.99).exp()).collect();
    let total: f64 = weights.iter().sum();
    let mut running = 0.0;
    let picked = weights
        .iter()
        .position(|w| {
            running += w;
            running > u * total
        })
        .expect("a picked token");
    assert!(judge(DRAW, &[], &logits, picked, index, 0.0).exact);
    assert!(!judge(DRAW, &[], &logits, picked + 1, index, 0.0).pass);
    assert!(!judge(DRAW, &[], &logits, picked - 1, index, 0.0).pass);
    // with filters and a band: the search for the kept sets ends
    let filters = [Filter::TopK(0), Filter::TopP(0.95), Filter::MinP(0.01)];
    let verdict = judge(DRAW, &filters, &logits, picked, index, 1e-2);
    assert!(verdict.lower <= verdict.upper, "{verdict:?}");
}
