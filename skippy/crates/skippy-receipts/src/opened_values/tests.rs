use super::*;
use crate::attention::opening::AttentionPartials;
use crate::binding::verify_paths;
use crate::layer_tensors::{GGML_F32, f32_bytes};
use crate::node_fault::NodeFault;
use crate::toy_turn::{HIDDEN, Turn, VOCABULARY, weights};
use crate::wire::{Challenge, Opening, encode_f32};

/// What a node changes in an honest opening.
type Change<'a> = Box<dyn FnOnce(&mut Opening) + 'a>;

fn challenged(turn: &Turn) -> Challenge {
    turn.challenge(&[(0, &[1, 3]), (1, &[0, 3])], &[2, 4])
}

/// Validates a canonical opening that `change` altered. The Merkle paths
/// no longer fit after a change, so this checks the stages directly, as
/// [`validate`] does after the paths passed.
fn check(change: impl FnOnce(&mut Opening)) -> Result<()> {
    let turn = Turn::new();
    let mut opening = turn.open(&challenged(&turn));
    change(&mut opening);
    let weights = weights();
    let shapes = ModelShapes::of(&weights)?;
    opening
        .stages
        .iter()
        .try_for_each(|stage| check_stage(stage, &weights, &shapes).or_node_fault())
}

fn fault(change: impl FnOnce(&mut Opening)) -> String {
    let error = check(change).unwrap_err();
    assert!(
        NodeFault::find(&error).is_some(),
        "not blamed on the node: {error:#}"
    );
    format!("{error:#}")
}

fn leaf(opening: &mut Opening) -> &mut LeafOpening {
    &mut opening.stages[1].layers[0].leaves[1]
}

#[test]
fn an_honest_opening_fits_the_model() {
    let turn = Turn::new();
    let challenge = challenged(&turn);
    let opening = turn.open(&challenge);
    let weights = weights();
    let shapes = ModelShapes::of(&weights).unwrap();
    assert_eq!(
        shapes,
        ModelShapes {
            hidden: HIDDEN,
            vocabulary: VOCABULARY as u64,
            tokens: Some(VOCABULARY as u64),
        }
    );
    let verified = verify_paths(&turn.receipt, &challenge, &opening).unwrap();
    let validated = validate(verified, &weights).unwrap();
    // the checks read the verified opening through it
    assert_eq!(validated.anchors().len(), 3);
    check_tokens(&turn.receipt, &shapes).unwrap();
}

#[test]
fn a_value_that_is_not_a_number_is_the_nodes_fault() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let values = [1.0, bad, 0.5, 0.25];
        let residual = encode_f32(&values);
        let changes: Vec<(&str, Change<'_>)> = vec![
            (
                "residual in",
                Box::new(|o| leaf(o).residual_in = residual.clone()),
            ),
            (
                "residual out",
                Box::new(|o| leaf(o).residual_out = Some(residual.clone())),
            ),
            (
                "matmul input",
                Box::new(|o| leaf(o).matrices[0].input = residual.clone()),
            ),
            (
                "matmul output",
                Box::new(|o| leaf(o).matrices[0].output = residual.clone()),
            ),
            (
                "mixing input",
                Box::new(|o| {
                    let bytes = f32_bytes(&values);
                    leaf(o).mixing[0] =
                        TensorOpening::new(MixingKind::Input, 0, Some((GGML_F32, &bytes)));
                }),
            ),
            (
                "mixing output",
                Box::new(|o| {
                    let bytes = f32_bytes(&values);
                    leaf(o).mixing[2] =
                        TensorOpening::new(MixingKind::Output, 0, Some((GGML_F32, &bytes)));
                }),
            ),
            (
                "stored row",
                Box::new(|o| {
                    let bytes: Vec<u8> = values
                        .iter()
                        .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
                        .collect();
                    leaf(o).mixing[1] =
                        TensorOpening::new(MixingKind::Stored, 0, Some((1, &bytes)));
                }),
            ),
            (
                "anchor",
                Box::new(|o| o.stages[0].anchors[0].residual = residual.clone()),
            ),
            (
                "head residual",
                Box::new(|o| o.stages[1].head[0].residual = residual.clone()),
            ),
            (
                "logits",
                Box::new(|o| {
                    let mut logits = vec![0.5; VOCABULARY];
                    logits[3] = bad;
                    o.stages[1].head[0].logits = Some(encode_f32(&logits));
                }),
            ),
            (
                "attention partials",
                Box::new(|o| {
                    o.stages[1].attention = vec![AttentionPartials {
                        position: 3,
                        layer: 1,
                        head: 0,
                        partials: encode_f32(&[0.0, bad, 1.0]),
                    }];
                }),
            ),
        ];
        for (what, change) in changes {
            let message = fault(change);
            assert!(
                message.contains("not a number"),
                "{what} with {bad}: {message}"
            );
        }
    }
}

#[test]
fn a_value_of_the_wrong_length_is_the_nodes_fault() {
    let short = encode_f32(&[1.0, 2.0, 3.0]);
    let long = encode_f32(&[1.0; 8]);
    let message = fault(|o| leaf(o).residual_in = short.clone());
    assert!(
        message.contains("layer 1, position 3: the residual in has 3 values"),
        "{message}"
    );
    fault(|o| leaf(o).residual_out = Some(long.clone()));
    fault(|o| o.stages[0].anchors[1].residual = short.clone());
    fault(|o| o.stages[1].head[1].residual = long.clone());
    // a column that is not the weight's columns or rows
    let message = fault(|o| leaf(o).matrices[0].input = short.clone());
    assert!(
        message.contains("do not fit the claimed weight"),
        "{message}"
    );
    fault(|o| leaf(o).matrices[0].output = long.clone());
    fault(|o| leaf(o).matrices[1].output = short.clone().repeat(2));
    // an empty column would pass every comparison
    fault(|o| {
        leaf(o).matrices[0].input = String::new();
        leaf(o).matrices[0].output = String::new();
    });
    // one output slice per expert named
    fault(|o| leaf(o).matrices[2].output = encode_f32(&[0.5; HIDDEN]));
    // logits for another vocabulary
    let message = fault(|o| o.stages[1].head[0].logits = Some(encode_f32(&[0.5; 7])));
    assert!(
        message.contains("7 logits are opened for 8 tokens"),
        "{message}"
    );
    // bytes that are not whole values
    fault(|o| leaf(o).residual_in = "AAAAAAA=".into());
    fault(|o| leaf(o).matrices[0].input = "not base64".into());
    fault(|o| {
        leaf(o).mixing[0] = TensorOpening::new(MixingKind::Input, 0, Some((GGML_F32, &[0u8; 6])));
    });
    // a stored f16 row of an odd number of bytes
    fault(|o| {
        leaf(o).mixing[1] = TensorOpening::new(MixingKind::Stored, 0, Some((1, &[0u8; 7])));
    });
}

#[test]
fn an_id_out_of_range_is_the_nodes_fault() {
    // an expert the weight does not hold
    let message = fault(|o| leaf(o).matrices[2].experts = Some(vec![0, 3]));
    assert!(message.contains("expert 3 outside 3 experts"), "{message}");
    fault(|o| leaf(o).matrices[2].experts = Some(vec![-1, 0]));
    // experts on a weight that has none
    fault(|o| leaf(o).matrices[0].experts = Some(vec![5]));
    // a selected id past the layer's experts, which the auditor's own
    // evaluation would index its expert weights with
    let message = fault(|o| leaf(o).selections[0].ids = vec![0, 1, 3]);
    assert!(message.contains("picks id 3"), "{message}");
    fault(|o| leaf(o).selections[0].ids = vec![-2, 1, 0]);
    fault(|o| leaf(o).selections[0].ids = vec![i32::MAX]);
}

#[test]
fn a_tensor_the_claimed_gguf_does_not_have_is_the_nodes_fault() {
    let message = fault(|o| leaf(o).matrices[0].weight = "blk.1.other.weight".into());
    assert!(
        message.contains("a tensor the claimed GGUF does not have"),
        "{message}"
    );
}

#[test]
fn a_stored_row_of_an_unknown_type_is_the_nodes_fault() {
    // 39 is no ggml type, and q4_1 is one no stage stores rows in: neither
    // may reach the native runtime as a cache type
    for ggml_type in [39, 3, -1, i32::MAX] {
        let message = fault(|o| leaf(o).mixing[1].ggml_type = ggml_type);
        assert!(
            message.contains("which the auditor does not decode"),
            "{ggml_type}: {message}"
        );
    }
    // a mixing input may only be f32
    fault(|o| leaf(o).mixing[0].ggml_type = 1);
}

#[test]
fn a_receipts_tokens_are_ones_the_model_embeds() {
    let shapes = ModelShapes::of(&weights()).unwrap();
    for bad in [VOCABULARY as i32, -1, i32::MAX] {
        let mut receipt = Turn::new().receipt;
        receipt.tokens[2] = bad;
        let error = check_tokens(&receipt, &shapes).unwrap_err();
        assert!(NodeFault::find(&error).is_some(), "{error:#}");
        let mut receipt = Turn::new().receipt;
        receipt.final_token = Some(bad);
        assert!(check_tokens(&receipt, &shapes).is_err(), "{bad}");
    }
}

#[test]
fn a_header_without_the_models_shapes_blames_nobody() {
    let bare =
        crate::weights::test_gguf::open(&[], &[("output.weight", vec![4, 8], vec![0.0; 32])]);
    let error = ModelShapes::of(&bare).unwrap_err();
    assert!(NodeFault::find(&error).is_none(), "{error:#}");
}
