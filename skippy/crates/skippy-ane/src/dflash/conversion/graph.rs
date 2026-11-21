//! Stateful DFlash MIL graph generated from a paired draft profile.

use anyhow::Result;
use mil_spec::{Block, DType, Feature, ModelMeta, NVT, TensorType, ValueType, encode_model};

use super::normalization::rms;
use super::weights::Weights;
use crate::dflash::profile::{DFlashProfile, DecoderContract, HiddenActivation};

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
    attn_gate: Option<(String, usize)>,
    attn_post_norm: Option<String>,
    ffn_post_norm: Option<String>,
    output_scale: Option<String>,
}

struct Fusion<'a> {
    projection: &'a str,
    norm: &'a str,
    aux_norm: Option<&'a str>,
}

struct LayerGeometry<'a> {
    profile: &'a DFlashProfile,
    index: usize,
    context: i64,
    rows: i64,
}

struct RopeShape {
    heads: i64,
    head_width: i64,
    rotary_width: i64,
    rows: i64,
}

pub(super) fn build(
    weights: &mut Weights<'_>,
    mask_embedding: &[f32],
    context: usize,
    rows: usize,
    profile: &DFlashProfile,
) -> Result<Vec<u8>> {
    let width = profile.width;
    let vocabulary = profile.vocabulary;
    let head_width = profile.head_width as i64;
    let context = context as i64;
    let rows = rows as i64;
    let mut b = Block::new();
    let fc = weights.draft_matrix(
        &mut b,
        "fc.weight",
        width,
        profile.feature_width().expect("validated profile width"),
    )?;
    let enc_norm = weights.draft_vector(
        &mut b,
        "enc.output_norm.weight",
        width,
        &[1, width as i64, 1, 1],
    )?;
    let output_norm = weights.draft_vector(
        &mut b,
        "output_norm.weight",
        width,
        &[1, width as i64, 1, 1],
    )?;
    let aux_norm = if profile.decoder == DecoderContract::Laguna {
        Some(weights.draft_array(
            &mut b,
            "enc.aux_norm.weight",
            &[profile.target_layers.len(), width],
            &[profile.target_layers.len() as i64, width as i64, 1],
        )?)
    } else {
        None
    };
    let mut layers = Vec::with_capacity(profile.layers());
    for index in 0..profile.layers() {
        layers.push(layer_weights(weights, &mut b, index, profile)?);
    }
    let mut mask_rows = Vec::with_capacity(width * (rows as usize - 1));
    for &value in mask_embedding {
        mask_rows.extend(std::iter::repeat_n(value, rows as usize - 1));
    }
    let mask_embed = weights.vector(
        &mut b,
        "mask_embedding",
        &mask_rows,
        &[1, width as i64, 1, rows - 1],
    )?;
    let fusion = Fusion {
        projection: &fc,
        norm: &enc_norm,
        aux_norm: aux_norm.as_deref(),
    };
    let history = ingest(&mut b, &layers, &fusion, context, rows, profile);
    let hidden = draft_block(
        &mut b,
        &layers,
        &history,
        &mask_embed,
        context,
        rows,
        profile,
    );
    let normalized = rms(
        &mut b,
        &hidden,
        &output_norm,
        &[1, width as i64, 1, rows],
        1,
        "output_norm",
        profile.rms_epsilon,
    );
    let predicted = b.slice(
        &normalized,
        &[0, 0, 0, 1],
        &[1, width as i32, 1, rows as i32],
        &[1, width as i64, 1, rows - 1],
        "draft_hidden",
    );
    let head_weights = weights.head(&mut b, vocabulary, width)?;
    let mut logits = Vec::with_capacity(8);
    for (part, (weight, output)) in head_weights.iter().enumerate() {
        logits.push(conv(
            &mut b,
            &predicted,
            weight,
            *output as i64,
            rows - 1,
            &format!("head_{part}"),
        ));
    }
    let logits = b.concat(&logits, 1, &[1, vocabulary as i64, 1, rows - 1], "logits");
    let tokens = b.reduce_argmax(&logits, 1, false, &[1, 1, rows - 1], "tokens");
    b.outputs = vec![tokens];

    let inputs = vec![
        feature(
            "features",
            &[
                1,
                profile.feature_width().expect("validated profile width") as i64,
                1,
                rows,
            ],
            DType::Fp16,
            false,
        ),
        feature("write", &[rows, context], DType::Fp16, false),
        feature(
            "ctx_cos",
            &[rows, profile.rotary_width as i64],
            DType::Fp16,
            false,
        ),
        feature(
            "ctx_sin",
            &[rows, profile.rotary_width as i64],
            DType::Fp16,
            false,
        ),
        feature("anchor", &[1, width as i64, 1, 1], DType::Fp16, false),
        feature(
            "q_cos",
            &[rows, profile.rotary_width as i64],
            DType::Fp16,
            false,
        ),
        feature(
            "q_sin",
            &[rows, profile.rotary_width as i64],
            DType::Fp16,
            false,
        ),
        feature("mask", &[rows, context + rows], DType::Fp16, false),
    ];
    let mut states = Vec::with_capacity(profile.layers() * 2);
    for index in 0..profile.layers() {
        let kv_heads = profile.kv_heads[index] as i64;
        for kind in ["k", "v"] {
            states.push(feature(
                &format!("{kind}{index}"),
                &[kv_heads, context, head_width],
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
        .description("DFlash draft on Apple Neural Engine")
        .user_meta("skippy.dflash.layers", &profile.layers().to_string());
    Ok(encode_model(
        &inputs, &outputs, &states, &b, &fn_inputs, &meta,
    ))
}

fn layer_weights(
    weights: &mut Weights<'_>,
    b: &mut Block,
    index: usize,
    profile: &DFlashProfile,
) -> Result<Layer> {
    let width = profile.width;
    let head_width = profile.head_width as i64;
    let query_width = profile.query_heads[index] * profile.head_width;
    let kv_width = profile.kv_heads[index] * profile.head_width;
    let ffn_width = profile.ffn_width[index];
    let prefix = format!("blk.{index}.");
    Ok(Layer {
        q: weights.draft_matrix(b, &format!("{prefix}attn_q.weight"), query_width, width)?,
        k: weights.draft_matrix(b, &format!("{prefix}attn_k.weight"), kv_width, width)?,
        v: weights.draft_matrix(b, &format!("{prefix}attn_v.weight"), kv_width, width)?,
        output: weights.draft_matrix(
            b,
            &format!("{prefix}attn_output.weight"),
            width,
            query_width,
        )?,
        gate: weights.draft_matrix(b, &format!("{prefix}ffn_gate.weight"), ffn_width, width)?,
        up: weights.draft_matrix(b, &format!("{prefix}ffn_up.weight"), ffn_width, width)?,
        down: weights.draft_matrix(b, &format!("{prefix}ffn_down.weight"), width, ffn_width)?,
        attn_norm: weights.draft_vector(
            b,
            &format!("{prefix}attn_norm.weight"),
            width,
            &[1, width as i64, 1, 1],
        )?,
        q_norm: weights.draft_vector(
            b,
            &format!("{prefix}attn_q_norm.weight"),
            head_width as usize,
            &[head_width],
        )?,
        k_norm: weights.draft_vector(
            b,
            &format!("{prefix}attn_k_norm.weight"),
            head_width as usize,
            &[head_width],
        )?,
        ffn_norm: weights.draft_vector(
            b,
            &format!("{prefix}ffn_norm.weight"),
            width,
            &[1, width as i64, 1, 1],
        )?,
        attn_gate: if profile.decoder == DecoderContract::Laguna {
            weights.optional_draft_matrix(
                b,
                &format!("{prefix}attn_gate.weight"),
                width,
                &[profile.query_heads[index], query_width],
            )?
        } else {
            None
        },
        attn_post_norm: weights.optional_draft_vector(
            b,
            &format!("{prefix}post_attention_norm.weight"),
            width,
            &[1, width as i64, 1, 1],
        )?,
        ffn_post_norm: weights.optional_draft_vector(
            b,
            &format!("{prefix}post_ffw_norm.weight"),
            width,
            &[1, width as i64, 1, 1],
        )?,
        output_scale: weights.optional_draft_vector(
            b,
            &format!("{prefix}layer_output_scale.weight"),
            1,
            &[1],
        )?,
    })
}

fn ingest(
    b: &mut Block,
    layers: &[Layer],
    fusion: &Fusion<'_>,
    context: i64,
    rows: i64,
    profile: &DFlashProfile,
) -> Vec<(String, String)> {
    let width = profile.width;
    let head_width = profile.head_width as i64;
    let features = if let Some(aux_norm) = fusion.aux_norm {
        let taps = profile.target_layers.len() as i64;
        let shape = &[taps, width as i64, rows];
        let reshaped = b.reshape("features", shape, "aux_features");
        let normed = rms(
            b,
            &reshaped,
            aux_norm,
            shape,
            1,
            "aux_norm",
            profile.rms_epsilon,
        );
        b.reshape(&normed, &[1, taps * width as i64, 1, rows], "aux_flattened")
    } else {
        "features".to_string()
    };
    let fused = conv(
        b,
        &features,
        fusion.projection,
        width as i64,
        rows,
        "fused_projection",
    );
    let fused = rms(
        b,
        &fused,
        fusion.norm,
        &[1, width as i64, 1, rows],
        1,
        "fused_norm",
        profile.rms_epsilon,
    );
    let write_sum = b.reduce_sum("write", &[0], true, &[1, context], "write_sum");
    let one = b.konst_f16("write_one", 1.0);
    let keep = b.sub(&one, &write_sum, &[1, context], "keep");
    let keep = b.reshape(&keep, &[1, context, 1], "keep_3d");
    let put = b.transpose("write", &[1, 0], &[context, rows], "write_transposed");
    let put = b.reshape(&put, &[1, context, rows], "write_3d");
    let mut history = Vec::with_capacity(layers.len());
    for (index, layer) in layers.iter().enumerate() {
        let kv_heads = profile.kv_heads[index] as i64;
        let kv_width = kv_heads * head_width;
        let label = format!("ingest_{index}");
        let kv_input = if profile.decoder == DecoderContract::Laguna {
            rms(
                b,
                &fused,
                &layer.attn_norm,
                &[1, width as i64, 1, rows],
                1,
                &format!("{label}_attn_norm"),
                profile.rms_epsilon,
            )
        } else {
            fused.clone()
        };
        let k_proj = conv(
            b,
            &kv_input,
            &layer.k,
            kv_width,
            rows,
            &format!("{label}_k_proj"),
        );
        let k = heads(
            b,
            &k_proj,
            kv_heads,
            head_width,
            rows,
            &format!("{label}_k"),
        );
        let k = rms(
            b,
            &k,
            &layer.k_norm,
            &[kv_heads, rows, head_width],
            2,
            &format!("{label}_k_norm"),
            profile.rms_epsilon,
        );
        let k = rope(
            b,
            &k,
            "ctx_cos",
            "ctx_sin",
            RopeShape {
                heads: kv_heads,
                head_width,
                rotary_width: profile.rotary_width as i64,
                rows,
            },
            &format!("{label}_rope"),
        );
        let v_proj = conv(
            b,
            &kv_input,
            &layer.v,
            kv_width,
            rows,
            &format!("{label}_v_proj"),
        );
        let v = heads(
            b,
            &v_proj,
            kv_heads,
            head_width,
            rows,
            &format!("{label}_v"),
        );
        let mut updated = Vec::with_capacity(2);
        for (kind, value) in [("k", k), ("v", v)] {
            let state = format!("{kind}{index}");
            let old = b.read_state(
                &state,
                &[kv_heads, context, head_width],
                &format!("{label}_{kind}_old"),
            );
            let retained = b.mul(
                &old,
                &keep,
                &[kv_heads, context, head_width],
                &format!("{label}_{kind}_retained"),
            );
            let inserted = b.matmul(
                &put,
                &value,
                false,
                &[kv_heads, context, head_width],
                &format!("{label}_{kind}_inserted"),
            );
            let merged = b.add(
                &retained,
                &inserted,
                &[kv_heads, context, head_width],
                &format!("{label}_{kind}_merged"),
            );
            b.write_state(&state, &merged);
            updated.push(b.read_state(
                &state,
                &[kv_heads, context, head_width],
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
    profile: &DFlashProfile,
) -> String {
    let width = profile.width;
    let mut x = b.concat(
        &["anchor".into(), mask_embed.into()],
        3,
        &[1, width as i64, 1, rows],
        "block_input",
    );
    for (index, layer) in layers.iter().enumerate() {
        let p = format!("block_{index}");
        let geometry = LayerGeometry {
            profile,
            index,
            context,
            rows,
        };
        let projected = draft_attention(b, &x, layer, &history[index], &geometry, &p);
        let projected = if let Some(weight) = &layer.attn_post_norm {
            rms(
                b,
                &projected,
                weight,
                &[1, width as i64, 1, rows],
                1,
                &format!("{p}_attn_post_norm"),
                profile.rms_epsilon,
            )
        } else {
            projected
        };
        x = b.add(
            &x,
            &projected,
            &[1, width as i64, 1, rows],
            &format!("{p}_attn_residual"),
        );
        let down = draft_ffn(b, &x, layer, rows, &p, profile, index);
        let down = if let Some(weight) = &layer.ffn_post_norm {
            rms(
                b,
                &down,
                weight,
                &[1, width as i64, 1, rows],
                1,
                &format!("{p}_ffn_post_norm"),
                profile.rms_epsilon,
            )
        } else {
            down
        };
        x = b.add(
            &x,
            &down,
            &[1, width as i64, 1, rows],
            &format!("{p}_ffn_residual"),
        );
        if let Some(weight) = &layer.output_scale {
            x = b.mul(
                &x,
                weight,
                &[1, width as i64, 1, rows],
                &format!("{p}_output_scaled"),
            );
        }
    }
    x
}

fn draft_attention(
    b: &mut Block,
    x: &str,
    layer: &Layer,
    history: &(String, String),
    geometry: &LayerGeometry<'_>,
    p: &str,
) -> String {
    let profile = geometry.profile;
    let index = geometry.index;
    let context = geometry.context;
    let rows = geometry.rows;
    let width = profile.width;
    let head_width = profile.head_width as i64;
    let query_heads = profile.query_heads[index] as i64;
    let kv_heads = profile.kv_heads[index] as i64;
    let group = query_heads / kv_heads;
    let query_width = query_heads * head_width;
    let kv_width = kv_heads * head_width;
    let norm = rms(
        b,
        x,
        &layer.attn_norm,
        &[1, width as i64, 1, rows],
        1,
        &format!("{p}_attn_norm"),
        profile.rms_epsilon,
    );
    let q_proj = conv(
        b,
        &norm,
        &layer.q,
        query_width,
        rows,
        &format!("{p}_q_proj"),
    );
    let k_proj = conv(b, &norm, &layer.k, kv_width, rows, &format!("{p}_k_proj"));
    let v_proj = conv(b, &norm, &layer.v, kv_width, rows, &format!("{p}_v_proj"));
    let q = heads(b, &q_proj, query_heads, head_width, rows, &format!("{p}_q"));
    let k = heads(b, &k_proj, kv_heads, head_width, rows, &format!("{p}_k"));
    let v = heads(b, &v_proj, kv_heads, head_width, rows, &format!("{p}_v"));
    let q = rms(
        b,
        &q,
        &layer.q_norm,
        &[query_heads, rows, head_width],
        2,
        &format!("{p}_q_norm"),
        profile.rms_epsilon,
    );
    let k = rms(
        b,
        &k,
        &layer.k_norm,
        &[kv_heads, rows, head_width],
        2,
        &format!("{p}_k_norm"),
        profile.rms_epsilon,
    );
    let q = rope(
        b,
        &q,
        "q_cos",
        "q_sin",
        RopeShape {
            heads: query_heads,
            head_width,
            rotary_width: profile.rotary_width as i64,
            rows,
        },
        &format!("{p}_q_rope"),
    );
    let k = rope(
        b,
        &k,
        "q_cos",
        "q_sin",
        RopeShape {
            heads: kv_heads,
            head_width,
            rotary_width: profile.rotary_width as i64,
            rows,
        },
        &format!("{p}_k_rope"),
    );
    let query = b.reshape(
        &q,
        &[kv_heads, group * rows, head_width],
        &format!("{p}_query"),
    );
    let scores_history = b.matmul(
        &query,
        &history.0,
        true,
        &[kv_heads, group * rows, context],
        &format!("{p}_history_score"),
    );
    let scores_block = b.matmul(
        &query,
        &k,
        true,
        &[kv_heads, group * rows, rows],
        &format!("{p}_block_score"),
    );
    let score = b.concat(
        &[scores_history, scores_block],
        2,
        &[kv_heads, group * rows, context + rows],
        &format!("{p}_score"),
    );
    let score = b.reshape(
        &score,
        &[kv_heads, group, rows, context + rows],
        &format!("{p}_score_4d"),
    );
    let scale = b.konst_f16(&format!("{p}_attention_scale"), profile.attention_scale);
    let score = b.mul(
        &score,
        &scale,
        &[kv_heads, group, rows, context + rows],
        &format!("{p}_scaled_score"),
    );
    let score = b.add(
        &score,
        "mask",
        &[kv_heads, group, rows, context + rows],
        &format!("{p}_masked_score"),
    );
    let prob = b.softmax(
        &score,
        -1,
        &[kv_heads, group, rows, context + rows],
        &format!("{p}_prob"),
        false,
    );
    let prob = b.reshape(
        &prob,
        &[kv_heads, group * rows, context + rows],
        &format!("{p}_prob_3d"),
    );
    let ph = b.slice(
        &prob,
        &[0, 0, 0],
        &[kv_heads as i32, (group * rows) as i32, context as i32],
        &[kv_heads, group * rows, context],
        &format!("{p}_prob_history"),
    );
    let pb = b.slice(
        &prob,
        &[0, 0, context as i32],
        &[
            kv_heads as i32,
            (group * rows) as i32,
            (context + rows) as i32,
        ],
        &[kv_heads, group * rows, rows],
        &format!("{p}_prob_block"),
    );
    let ah = b.matmul(
        &ph,
        &history.1,
        false,
        &[kv_heads, group * rows, head_width],
        &format!("{p}_attn_history"),
    );
    let ab = b.matmul(
        &pb,
        &v,
        false,
        &[kv_heads, group * rows, head_width],
        &format!("{p}_attn_block"),
    );
    let attended = b.add(
        &ah,
        &ab,
        &[kv_heads, group * rows, head_width],
        &format!("{p}_attended"),
    );
    let attended = b.reshape(
        &attended,
        &[query_heads, rows, head_width],
        &format!("{p}_attended_heads"),
    );
    let attended = b.transpose(
        &attended,
        &[0, 2, 1],
        &[query_heads, head_width, rows],
        &format!("{p}_attended_transposed"),
    );
    let attended = b.reshape(
        &attended,
        &[1, query_heads * head_width, 1, rows],
        &format!("{p}_attended_conv"),
    );
    let attended = if let Some((gate_weight, gate_width)) = &layer.attn_gate {
        let gate_width = *gate_width as i64;
        let gate = conv(
            b,
            &norm,
            gate_weight,
            gate_width,
            rows,
            &format!("{p}_gate_proj"),
        );
        let gate = b.softplus(
            &gate,
            &[1, gate_width, 1, rows],
            &format!("{p}_gate_softplus"),
        );
        if gate_width == query_heads {
            let shaped = b.reshape(
                &attended,
                &[query_heads, head_width, rows],
                &format!("{p}_gated_heads"),
            );
            let gate = b.reshape(&gate, &[query_heads, 1, rows], &format!("{p}_gate_heads"));
            let gated = b.mul(
                &shaped,
                &gate,
                &[query_heads, head_width, rows],
                &format!("{p}_attn_gated"),
            );
            b.reshape(
                &gated,
                &[1, query_width, 1, rows],
                &format!("{p}_gated_conv"),
            )
        } else {
            b.mul(
                &attended,
                &gate,
                &[1, query_width, 1, rows],
                &format!("{p}_attn_gated"),
            )
        }
    } else {
        attended
    };
    let projected = conv(
        b,
        &attended,
        &layer.output,
        width as i64,
        rows,
        &format!("{p}_output_proj"),
    );
    if profile.value_scale == 1.0 {
        projected
    } else {
        let scale = b.konst_f16(&format!("{p}_value_scale"), profile.value_scale);
        b.mul(
            &projected,
            &scale,
            &[1, width as i64, 1, rows],
            &format!("{p}_value_scaled"),
        )
    }
}

fn draft_ffn(
    b: &mut Block,
    x: &str,
    layer: &Layer,
    rows: i64,
    p: &str,
    profile: &DFlashProfile,
    index: usize,
) -> String {
    let width = profile.width;
    let ffn_width = profile.ffn_width[index];
    let norm = rms(
        b,
        x,
        &layer.ffn_norm,
        &[1, width as i64, 1, rows],
        1,
        &format!("{p}_ffn_norm"),
        profile.rms_epsilon,
    );
    let gate = conv(
        b,
        &norm,
        &layer.gate,
        ffn_width as i64,
        rows,
        &format!("{p}_gate"),
    );
    let shape = &[1, ffn_width as i64, 1, rows];
    let gate = match profile.activation {
        HiddenActivation::Silu => {
            let sigmoid = b.sigmoid(&gate, shape, &format!("{p}_sigmoid"));
            b.mul(&gate, &sigmoid, shape, &format!("{p}_silu"))
        }
        HiddenActivation::Gelu => b.gelu(&gate, false, shape, &format!("{p}_gelu")),
        HiddenActivation::GeluTanh => b.gelu(&gate, true, shape, &format!("{p}_gelu_tanh")),
    };
    let up = conv(
        b,
        &norm,
        &layer.up,
        ffn_width as i64,
        rows,
        &format!("{p}_up"),
    );
    let hidden = b.mul(
        &gate,
        &up,
        &[1, ffn_width as i64, 1, rows],
        &format!("{p}_ffn_hidden"),
    );
    conv(
        b,
        &hidden,
        &layer.down,
        width as i64,
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

fn heads(b: &mut Block, x: &str, count: i64, head_width: i64, rows: i64, name: &str) -> String {
    let reshaped = b.reshape(x, &[count, head_width, rows], &format!("{name}_reshape"));
    b.transpose(&reshaped, &[0, 2, 1], &[count, rows, head_width], name)
}

fn rope(b: &mut Block, x: &str, cos: &str, sin: &str, shape: RopeShape, name: &str) -> String {
    let RopeShape {
        heads,
        head_width,
        rotary_width,
        rows,
    } = shape;
    let half = rotary_width / 2;
    let rotary = if rotary_width == head_width {
        x.to_string()
    } else {
        b.slice(
            x,
            &[0, 0, 0],
            &[heads as i32, rows as i32, rotary_width as i32],
            &[heads, rows, rotary_width],
            &format!("{name}_prefix"),
        )
    };
    let left = b.slice(
        &rotary,
        &[0, 0, 0],
        &[heads as i32, rows as i32, half as i32],
        &[heads, rows, half],
        &format!("{name}_left"),
    );
    let right = b.slice(
        &rotary,
        &[0, 0, half as i32],
        &[heads as i32, rows as i32, rotary_width as i32],
        &[heads, rows, half],
        &format!("{name}_right"),
    );
    let minus_one = b.konst_f16(&format!("{name}_minus_one"), -1.0);
    let negated = b.mul(
        &right,
        &minus_one,
        &[heads, rows, half],
        &format!("{name}_negative"),
    );
    let rotated = b.concat(
        &[negated, left],
        2,
        &[heads, rows, rotary_width],
        &format!("{name}_rotated"),
    );
    let real = b.mul(
        &rotary,
        cos,
        &[heads, rows, rotary_width],
        &format!("{name}_cosine"),
    );
    let imaginary = b.mul(
        &rotated,
        sin,
        &[heads, rows, rotary_width],
        &format!("{name}_sine"),
    );
    let value = b.add(&real, &imaginary, &[heads, rows, rotary_width], name);
    if rotary_width == head_width {
        value
    } else {
        let tail = b.slice(
            x,
            &[0, 0, rotary_width as i32],
            &[heads as i32, rows as i32, head_width as i32],
            &[heads, rows, head_width - rotary_width],
            &format!("{name}_tail"),
        );
        b.concat(
            &[value, tail],
            2,
            &[heads, rows, head_width],
            &format!("{name}_full"),
        )
    }
}

fn feature(name: &str, shape: &[i64], dtype: DType, is_state: bool) -> Feature {
    Feature {
        name: name.into(),
        shape: shape.to_vec(),
        dtype,
        is_state,
    }
}
