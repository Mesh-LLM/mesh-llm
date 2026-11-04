use super::test_op::{context, op, query};
use super::*;

#[test]
fn a_head_reads_its_own_kv_head() {
    let op = op();
    assert_eq!(
        (0..4).map(|h| op.kv_head(h)).collect::<Vec<_>>(),
        vec![0, 0, 1, 1]
    );
}

#[test]
fn windows_bound_the_attended_positions() {
    assert!(Window::Causal.attends(5, 0));
    assert!(!Window::Causal.attends(5, 6));
    assert!(Window::Sliding(4).attends(5, 2));
    assert!(!Window::Sliding(4).attends(5, 1));
    assert!(Window::Chunked(4).attends(5, 4));
    assert!(!Window::Chunked(4).attends(5, 3));
}

#[test]
fn a_rounded_q_enters_the_partials_as_the_kernel_rounds_it() {
    let rounded = AttentionOp {
        head_dim_k: 32,
        q_rounding: Some(q_rounding(8).unwrap()),
        ..op()
    };
    rounded.check_shape().unwrap();
    let raw = AttentionOp {
        q_rounding: None,
        ..rounded.clone()
    };
    let rows = context(&rounded, 70);
    let q = query(&rounded);
    let as_kernel: Vec<f32> = Profile::CpuQ8.apply(&q).iter().map(|&x| x as f32).collect();
    let partials = |op: &AttentionOp, q: &[f32]| head_partials(op, 1, q, &rows).unwrap();
    assert_eq!(partials(&rounded, &q), partials(&raw, &as_kernel));
    assert_ne!(partials(&rounded, &q), partials(&raw, &q));
    assert!(q_rounding(2).is_err());
    let short = AttentionOp {
        q_rounding: Some(Profile::CpuQ8),
        ..op()
    };
    assert!(short.check_shape().is_err());
}

#[test]
fn the_context_length_counts_the_attended_positions() {
    // the length is checked against what the node opened before the list
    // of positions is made, so the two must agree
    let windows = [
        Window::Causal,
        Window::Sliding(1),
        Window::Sliding(7),
        Window::Sliding(0),
        Window::Chunked(1),
        Window::Chunked(5),
        Window::Chunked(0),
    ];
    for window in windows {
        let op = AttentionOp { window, ..op() };
        for position in 0..40u32 {
            let attended: Vec<u32> = (0..=position)
                .filter(|&key| window.attends(position, key))
                .collect();
            assert_eq!(context_positions(&op, position), attended, "{window:?}");
            assert_eq!(context_len(&op, position), attended.len() as u64);
        }
    }
    // a position as large as a receipt can claim is counted, not listed
    let far = u32::MAX - 1;
    assert_eq!(context_len(&op(), far), u64::from(far) + 1);
    let sliding = AttentionOp {
        window: Window::Sliding(64),
        ..op()
    };
    assert_eq!(context_len(&sliding, far), 64);
    assert_eq!(context_positions(&sliding, far).len(), 64);
}

#[test]
fn a_stored_row_of_another_type_length_or_not_a_number_is_the_nodes_fault() {
    use crate::node_fault::NodeFault;
    let layout = AttentionLayout {
        layer: 0,
        op: op(),
        q_input: 0,
        k_store: 0,
        v_store: 1,
        output: 0,
        output_gate: None,
    };
    let f16 = |values: &[f32]| -> Vec<u8> {
        values
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
            .collect()
    };
    let (k, v) = (f16(&[0.5; 6]), f16(&[0.25; 4]));
    let row = layout
        .context_row(9, &[(1, k.as_slice()), (1, v.as_slice())])
        .unwrap();
    assert_eq!((row.key.len(), row.value.len()), (6, 4));
    let nan = f16(&[0.5, f32::NAN, 0.5, 0.5, 0.5, 0.5]);
    let short = f16(&[0.5; 4]);
    let bad: [Vec<(i32, &[u8])>; 5] = [
        vec![(39, &k), (1, &v)],
        vec![(3, &k), (1, &v)],
        vec![(1, &nan), (1, &v)],
        vec![(1, &short), (1, &v)],
        vec![(1, &k)],
    ];
    for (i, stored) in bad.iter().enumerate() {
        let error = layout.context_row(9, stored).unwrap_err();
        assert!(NodeFault::find(&error).is_some(), "row {i}: {error:#}");
    }
}
