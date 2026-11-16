//! Stateful six-layer DFlash MIL graph for the paired Qwen3.6 checkpoint.

use anyhow::Result;
use mil_spec::{Block, DType, Feature, ModelMeta, NVT, TensorType, ValueType, encode_model};

use super::gguf::{VOCAB, WIDTH};
use super::weights::Weights;

const HEAD: i64 = 128;
const QUERY_HEADS: i64 = 32;
const KV_HEADS: i64 = 8;
const FFN: usize = 6_144;

struct Layer {
    q: String,
    k: String,
    v: String,
    output: String,
    gate: String,
    up: String,
    down: String,
    attn_norm: String,
    q_norm: String,
    k_norm: String,
    ffn_norm: String,
}

pub(super) fn build(
    weights: &mut Weights<'_>,
    mask_embedding: &[f32],
    context: usize,
    rows: usize,
) -> Result<Vec<u8>> {
    let context = context as i64;
    let rows = rows as i64;
    let mut b = Block::new();
    let fc = weights.draft_matrix(&mut b, "fc.weight", WIDTH, 8 * WIDTH)?;
    let enc_norm = weights.draft_vector(
        &mut b,
        "enc.output_norm.weight",
        WIDTH,
        &[1, WIDTH as i64, 1, 1],
    )?;
    let output_norm = weights.draft_vector(
        &mut b,
        "output_norm.weight",
        WIDTH,
        &[1, WIDTH as i64, 1, 1],
    )?;
    let mut layers = Vec::with_capacity(6);
    for index in 0..6 {
        layers.push(layer_weights(weights, &mut b, index)?);
    }
    let mut mask_rows = Vec::with_capacity(WIDTH * (rows as usize - 1));
    for &value in mask_embedding {
        mask_rows.extend(std::iter::repeat_n(value, rows as usize - 1));
    }
    let mask_embed = weights.vector(
        &mut b,
        "mask_embedding",
        &mask_rows,
        &[1, WIDTH as i64, 1, rows - 1],
    )?;
    let history = ingest(&mut b, &layers, &fc, &enc_norm, context, rows);
    let hidden = draft_block(&mut b, &layers, &history, &mask_embed, context, rows);
    let normalized = rms(
        &mut b,
        &hidden,
        &output_norm,
        &[1, WIDTH as i64, 1, rows],
        1,
        "output_norm",
    );
    let predicted = b.slice(
        &normalized,
        &[0, 0, 0, 1],
        &[1, WIDTH as i32, 1, rows as i32],
        &[1, WIDTH as i64, 1, rows - 1],
        "draft_hidden",
    );
    let head_weights = weights.head(&mut b, VOCAB, WIDTH)?;
    let mut logits = Vec::with_capacity(8);
    for (part, weight) in head_weights.iter().enumerate() {
        logits.push(conv(
            &mut b,
            &predicted,
            weight,
            (VOCAB / 8) as i64,
            rows - 1,
            &format!("head_{part}"),
        ));
    }
    let logits = b.concat(&logits, 1, &[1, VOCAB as i64, 1, rows - 1], "logits");
    let tokens = b.reduce_argmax(&logits, 1, false, &[1, 1, rows - 1], "tokens");
    b.outputs = vec![tokens];

    let inputs = vec![
        feature(
            "features",
            &[1, (8 * WIDTH) as i64, 1, rows],
            DType::Fp16,
            false,
        ),
        feature("write", &[rows, context], DType::Fp16, false),
        feature("ctx_cos", &[rows, HEAD], DType::Fp16, false),
        feature("ctx_sin", &[rows, HEAD], DType::Fp16, false),
        feature("anchor", &[1, WIDTH as i64, 1, 1], DType::Fp16, false),
        feature("q_cos", &[rows, HEAD], DType::Fp16, false),
        feature("q_sin", &[rows, HEAD], DType::Fp16, false),
        feature("mask", &[rows, context + rows], DType::Fp16, false),
    ];
    let mut states = Vec::with_capacity(12);
    for index in 0..6 {
        for kind in ["k", "v"] {
            states.push(feature(
                &format!("{kind}{index}"),
                &[KV_HEADS, context, HEAD],
                DType::Fp16,
                true,
            ));
        }
    }
    let fn_inputs = inputs
        .iter()
        .map(|feature| NVT {
            name: feature.name.clone(),
            ty: ValueType::Tensor(TensorType::f16(&feature.shape)),
        })
        .chain(states.iter().map(|feature| NVT {
            name: feature.name.clone(),
            ty: ValueType::State(TensorType::f16(&feature.shape)),
        }))
        .collect::<Vec<_>>();
    let outputs = [feature("tokens", &[1, 1, rows - 1], DType::Int32, false)];
    let meta = ModelMeta::new(9, "CoreML8")
        .creator("Skippy Rust DFlash converter")
        .description("Qwen3.6 paired DFlash draft on Apple Neural Engine")
        .user_meta("skippy.dflash.layers", "6");
    Ok(encode_model(
        &inputs, &outputs, &states, &b, &fn_inputs, &meta,
    ))
}

fn layer_weights(weights: &mut Weights<'_>, b: &mut Block, index: usize) -> Result<Layer> {
    let prefix = format!("blk.{index}.");
    Ok(Layer {
        q: weights.draft_matrix(b, &format!("{prefix}attn_q.weight"), 4_096, WIDTH)?,
        k: weights.draft_matrix(b, &format!("{prefix}attn_k.weight"), 1_024, WIDTH)?,
        v: weights.draft_matrix(b, &format!("{prefix}attn_v.weight"), 1_024, WIDTH)?,
        output: weights.draft_matrix(b, &format!("{prefix}attn_output.weight"), WIDTH, 4_096)?,
        gate: weights.draft_matrix(b, &format!("{prefix}ffn_gate.weight"), FFN, WIDTH)?,
        up: weights.draft_matrix(b, &format!("{prefix}ffn_up.weight"), FFN, WIDTH)?,
        down: weights.draft_matrix(b, &format!("{prefix}ffn_down.weight"), WIDTH, FFN)?,
        attn_norm: weights.draft_vector(
            b,
            &format!("{prefix}attn_norm.weight"),
            WIDTH,
            &[1, WIDTH as i64, 1, 1],
        )?,
        q_norm: weights.draft_vector(
            b,
            &format!("{prefix}attn_q_norm.weight"),
            HEAD as usize,
            &[HEAD],
        )?,
        k_norm: weights.draft_vector(
            b,
            &format!("{prefix}attn_k_norm.weight"),
            HEAD as usize,
            &[HEAD],
        )?,
        ffn_norm: weights.draft_vector(
            b,
            &format!("{prefix}ffn_norm.weight"),
            WIDTH,
            &[1, WIDTH as i64, 1, 1],
        )?,
    })
}

fn ingest(
    b: &mut Block,
    layers: &[Layer],
    fc: &str,
    enc_norm: &str,
    context: i64,
    rows: i64,
) -> Vec<(String, String)> {
    let fused = conv(b, "features", fc, WIDTH as i64, rows, "fused_projection");
    let fused = rms(
        b,
        &fused,
        enc_norm,
        &[1, WIDTH as i64, 1, rows],
        1,
        "fused_norm",
    );
    let write_sum = b.reduce_sum("write", &[0], true, &[1, context], "write_sum");
    let one = b.konst_f16("write_one", 1.0);
    let keep = b.sub(&one, &write_sum, &[1, context], "keep");
    let keep = b.reshape(&keep, &[1, context, 1], "keep_3d");
    let put = b.transpose("write", &[1, 0], &[context, rows], "write_transposed");
    let put = b.reshape(&put, &[1, context, rows], "write_3d");
    let mut history = Vec::with_capacity(6);
    for (index, layer) in layers.iter().enumerate() {
        let label = format!("ingest_{index}");
        let k_proj = conv(b, &fused, &layer.k, 1_024, rows, &format!("{label}_k_proj"));
        let k = heads(b, &k_proj, KV_HEADS, rows, &format!("{label}_k"));
        let k = rms(
            b,
            &k,
            &layer.k_norm,
            &[KV_HEADS, rows, HEAD],
            2,
            &format!("{label}_k_norm"),
        );
        let k = rope(
            b,
            &k,
            "ctx_cos",
            "ctx_sin",
            KV_HEADS,
            rows,
            &format!("{label}_rope"),
        );
        let v_proj = conv(b, &fused, &layer.v, 1_024, rows, &format!("{label}_v_proj"));
        let v = heads(b, &v_proj, KV_HEADS, rows, &format!("{label}_v"));
        let mut updated = Vec::with_capacity(2);
        for (kind, value) in [("k", k), ("v", v)] {
            let state = format!("{kind}{index}");
            let old = b.read_state(
                &state,
                &[KV_HEADS, context, HEAD],
                &format!("{label}_{kind}_old"),
            );
            let retained = b.mul(
                &old,
                &keep,
                &[KV_HEADS, context, HEAD],
                &format!("{label}_{kind}_retained"),
            );
            let inserted = b.matmul(
                &put,
                &value,
                false,
                &[KV_HEADS, context, HEAD],
                &format!("{label}_{kind}_inserted"),
            );
            let merged = b.add(
                &retained,
                &inserted,
                &[KV_HEADS, context, HEAD],
                &format!("{label}_{kind}_merged"),
            );
            b.write_state(&state, &merged);
            updated.push(b.read_state(
                &state,
                &[KV_HEADS, context, HEAD],
                &format!("{label}_{kind}_updated"),
            ));
        }
        history.push((updated.remove(0), updated.remove(0)));
    }
    history
}

fn draft_block(
    b: &mut Block,
    layers: &[Layer],
    history: &[(String, String)],
    mask_embed: &str,
    context: i64,
    rows: i64,
) -> String {
    let mut x = b.concat(
        &["anchor".into(), mask_embed.into()],
        3,
        &[1, WIDTH as i64, 1, rows],
        "block_input",
    );
    for (index, layer) in layers.iter().enumerate() {
        let p = format!("block_{index}");
        let projected = draft_attention(b, &x, layer, &history[index], context, rows, &p);
        x = b.add(
            &x,
            &projected,
            &[1, WIDTH as i64, 1, rows],
            &format!("{p}_attn_residual"),
        );
        let down = draft_ffn(b, &x, layer, rows, &p);
        x = b.add(
            &x,
            &down,
            &[1, WIDTH as i64, 1, rows],
            &format!("{p}_ffn_residual"),
        );
    }
    x
}

fn draft_attention(
    b: &mut Block,
    x: &str,
    layer: &Layer,
    history: &(String, String),
    context: i64,
    rows: i64,
    p: &str,
) -> String {
    let norm = rms(
        b,
        x,
        &layer.attn_norm,
        &[1, WIDTH as i64, 1, rows],
        1,
        &format!("{p}_attn_norm"),
    );
    let q_proj = conv(b, &norm, &layer.q, 4_096, rows, &format!("{p}_q_proj"));
    let k_proj = conv(b, &norm, &layer.k, 1_024, rows, &format!("{p}_k_proj"));
    let v_proj = conv(b, &norm, &layer.v, 1_024, rows, &format!("{p}_v_proj"));
    let q = heads(b, &q_proj, QUERY_HEADS, rows, &format!("{p}_q"));
    let k = heads(b, &k_proj, KV_HEADS, rows, &format!("{p}_k"));
    let v = heads(b, &v_proj, KV_HEADS, rows, &format!("{p}_v"));
    let q = rms(
        b,
        &q,
        &layer.q_norm,
        &[QUERY_HEADS, rows, HEAD],
        2,
        &format!("{p}_q_norm"),
    );
    let k = rms(
        b,
        &k,
        &layer.k_norm,
        &[KV_HEADS, rows, HEAD],
        2,
        &format!("{p}_k_norm"),
    );
    let q = rope(
        b,
        &q,
        "q_cos",
        "q_sin",
        QUERY_HEADS,
        rows,
        &format!("{p}_q_rope"),
    );
    let k = rope(
        b,
        &k,
        "q_cos",
        "q_sin",
        KV_HEADS,
        rows,
        &format!("{p}_k_rope"),
    );
    let query = b.reshape(&q, &[KV_HEADS, 4 * rows, HEAD], &format!("{p}_query"));
    let scores_history = b.matmul(
        &query,
        &history.0,
        true,
        &[KV_HEADS, 4 * rows, context],
        &format!("{p}_history_score"),
    );
    let scores_block = b.matmul(
        &query,
        &k,
        true,
        &[KV_HEADS, 4 * rows, rows],
        &format!("{p}_block_score"),
    );
    let score = b.concat(
        &[scores_history, scores_block],
        2,
        &[KV_HEADS, 4 * rows, context + rows],
        &format!("{p}_score"),
    );
    let score = b.reshape(
        &score,
        &[KV_HEADS, 4, rows, context + rows],
        &format!("{p}_score_4d"),
    );
    let scale = b.konst_f16(&format!("{p}_attention_scale"), (HEAD as f32).powf(-0.5));
    let score = b.mul(
        &score,
        &scale,
        &[KV_HEADS, 4, rows, context + rows],
        &format!("{p}_scaled_score"),
    );
    let score = b.add(
        &score,
        "mask",
        &[KV_HEADS, 4, rows, context + rows],
        &format!("{p}_masked_score"),
    );
    let prob = b.softmax(
        &score,
        -1,
        &[KV_HEADS, 4, rows, context + rows],
        &format!("{p}_prob"),
        false,
    );
    let prob = b.reshape(
        &prob,
        &[KV_HEADS, 4 * rows, context + rows],
        &format!("{p}_prob_3d"),
    );
    let ph = b.slice(
        &prob,
        &[0, 0, 0],
        &[KV_HEADS as i32, (4 * rows) as i32, context as i32],
        &[KV_HEADS, 4 * rows, context],
        &format!("{p}_prob_history"),
    );
    let pb = b.slice(
        &prob,
        &[0, 0, context as i32],
        &[KV_HEADS as i32, (4 * rows) as i32, (context + rows) as i32],
        &[KV_HEADS, 4 * rows, rows],
        &format!("{p}_prob_block"),
    );
    let ah = b.matmul(
        &ph,
        &history.1,
        false,
        &[KV_HEADS, 4 * rows, HEAD],
        &format!("{p}_attn_history"),
    );
    let ab = b.matmul(
        &pb,
        &v,
        false,
        &[KV_HEADS, 4 * rows, HEAD],
        &format!("{p}_attn_block"),
    );
    let attended = b.add(
        &ah,
        &ab,
        &[KV_HEADS, 4 * rows, HEAD],
        &format!("{p}_attended"),
    );
    let attended = b.reshape(
        &attended,
        &[QUERY_HEADS, rows, HEAD],
        &format!("{p}_attended_heads"),
    );
    let attended = b.transpose(
        &attended,
        &[0, 2, 1],
        &[QUERY_HEADS, HEAD, rows],
        &format!("{p}_attended_transposed"),
    );
    let attended = b.reshape(
        &attended,
        &[1, QUERY_HEADS * HEAD, 1, rows],
        &format!("{p}_attended_conv"),
    );
    conv(
        b,
        &attended,
        &layer.output,
        WIDTH as i64,
        rows,
        &format!("{p}_output_proj"),
    )
}

fn draft_ffn(b: &mut Block, x: &str, layer: &Layer, rows: i64, p: &str) -> String {
    let norm = rms(
        b,
        x,
        &layer.ffn_norm,
        &[1, WIDTH as i64, 1, rows],
        1,
        &format!("{p}_ffn_norm"),
    );
    let gate = conv(
        b,
        &norm,
        &layer.gate,
        FFN as i64,
        rows,
        &format!("{p}_gate"),
    );
    let sigmoid = b.sigmoid(&gate, &[1, FFN as i64, 1, rows], &format!("{p}_sigmoid"));
    let gate = b.mul(
        &gate,
        &sigmoid,
        &[1, FFN as i64, 1, rows],
        &format!("{p}_silu"),
    );
    let up = conv(b, &norm, &layer.up, FFN as i64, rows, &format!("{p}_up"));
    let hidden = b.mul(
        &gate,
        &up,
        &[1, FFN as i64, 1, rows],
        &format!("{p}_ffn_hidden"),
    );
    conv(
        b,
        &hidden,
        &layer.down,
        WIDTH as i64,
        rows,
        &format!("{p}_down"),
    )
}

fn conv(b: &mut Block, x: &str, weight: &str, output: i64, rows: i64, name: &str) -> String {
    b.conv(
        x,
        weight,
        None,
        &[1, 1],
        "valid",
        &[0, 0, 0, 0],
        &[1, 1],
        1,
        &[1, output, 1, rows],
        name,
    )
}

fn heads(b: &mut Block, x: &str, count: i64, rows: i64, name: &str) -> String {
    let reshaped = b.reshape(x, &[count, HEAD, rows], &format!("{name}_reshape"));
    b.transpose(&reshaped, &[0, 2, 1], &[count, rows, HEAD], name)
}

fn rope(b: &mut Block, x: &str, cos: &str, sin: &str, heads: i64, rows: i64, name: &str) -> String {
    let left = b.slice(
        x,
        &[0, 0, 0],
        &[heads as i32, rows as i32, 64],
        &[heads, rows, 64],
        &format!("{name}_left"),
    );
    let right = b.slice(
        x,
        &[0, 0, 64],
        &[heads as i32, rows as i32, 128],
        &[heads, rows, 64],
        &format!("{name}_right"),
    );
    let minus_one = b.konst_f16(&format!("{name}_minus_one"), -1.0);
    let negated = b.mul(
        &right,
        &minus_one,
        &[heads, rows, 64],
        &format!("{name}_negative"),
    );
    let rotated = b.concat(
        &[negated, left],
        2,
        &[heads, rows, HEAD],
        &format!("{name}_rotated"),
    );
    let real = b.mul(x, cos, &[heads, rows, HEAD], &format!("{name}_cosine"));
    let imaginary = b.mul(&rotated, sin, &[heads, rows, HEAD], &format!("{name}_sine"));
    b.add(&real, &imaginary, &[heads, rows, HEAD], name)
}

fn rms(b: &mut Block, x: &str, weight: &str, shape: &[i64], axis: i32, name: &str) -> String {
    let mut reduced = shape.to_vec();
    reduced[axis as usize] = 1;
    let absolute = b.abs(x, shape, &format!("{name}_abs"));
    let maximum = b.reduce_max(&absolute, &[axis], true, &reduced, &format!("{name}_max"));
    let inverse = b.inverse(&maximum, 1e-4, &reduced, &format!("{name}_inverse"));
    let scaled = b.mul(x, &inverse, shape, &format!("{name}_scaled"));
    let square = b.mul(&scaled, &scaled, shape, &format!("{name}_square"));
    let mean = b.reduce_mean(&square, &[axis], true, &reduced, &format!("{name}_mean"));
    let epsilon = b.konst_f16(&format!("{name}_epsilon"), 1e-6);
    let scaled_inverse = b.mul(
        &inverse,
        &epsilon,
        &reduced,
        &format!("{name}_scaled_inverse"),
    );
    let epsilon = b.mul(
        &inverse,
        &scaled_inverse,
        &reduced,
        &format!("{name}_scaled_epsilon"),
    );
    let denominator = b.add(&mean, &epsilon, &reduced, &format!("{name}_denominator"));
    let reciprocal = b.rsqrt(&denominator, 1e-7, &reduced, &format!("{name}_reciprocal"));
    let normalized = b.mul(&scaled, &reciprocal, shape, &format!("{name}_normalized"));
    b.mul(&normalized, weight, shape, name)
}

fn feature(name: &str, shape: &[i64], dtype: DType, is_state: bool) -> Feature {
    Feature {
        name: name.into(),
        shape: shape.to_vec(),
        dtype,
        is_state,
    }
}
