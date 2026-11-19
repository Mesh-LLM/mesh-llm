//! A toy model and an honest turn of it, for tests of the client's checks.
//!
//! The model has [`LAYERS`] layers of [`HIDDEN`] values, one stage per
//! layer, and a vocabulary of [`VOCABULARY`] tokens. Each layer has what
//! the checks tell apart:
//!
//! ```text
//! matmul 0 (blk.L.a.weight):      input = 2 · residual
//! matmul 1 (blk.L.router.weight): input = residual, a router
//! selection 0:                    the experts, best score first
//! matmul 2 (blk.L.exps.weight):   input = residual, for the two best experts
//! mixing input 0:                 residual
//! stored row 0:                   residual, as f16
//! mixing output 0:                a value per position
//! residual out = residual + (matmul 0 + both expert outputs) / 8 + mixing output 0
//! ```
//!
//! [`weights`] is its GGUF, [`ToyModel`] the evaluator an auditor computes
//! a layer with, and [`Turn`] a turn a node served honestly: its receipt
//! and the opening of any challenge. A test changes one thing in an opening
//! and expects the check that owns it to fail.

use anyhow::{Context, Result, bail};
use ed25519_dalek::SigningKey;

use crate::attention::opening::{BlockChallenge, stored_digests};
use crate::cache_types;
use crate::commitment::{
    Hash, MerkleTree, MixingValue, cell_digest, encode_path, mixing_leaf, position_digest,
    prefix_leaf, prefix_tokens_digest, residual_leaf, slots_per_position, stored_leaf,
};
use crate::glue::{GlueEvaluator, GlueOutput, GlueRequest, GlueTensor, GlueValues, StoredRowError};
use crate::head::OutputHead;
use crate::layer_tensors::{FLAG_ROUTER, GGML_F32, TensorKind, f32_bytes};
use crate::turn::{self, BindingRequest};
use crate::weights::{ByteSource, WeightFile};
use crate::wire::{
    AnchorOpening, BoundaryLeaf, Challenge, HeadOpening, LayerOpening, LeafOpening, MatrixOpening,
    MixingKind, Opening, Pick, PrefixCommitment, PrefixOpening, RECEIPT_VERSION, Receipt,
    SelectionOpening, StageCommitment, StageOpening, TensorOpening, encode_f32,
};

pub(crate) const HIDDEN: usize = 4;
pub(crate) const VOCABULARY: usize = 8;
pub(crate) const EXPERTS: usize = 3;
pub const LAYERS: u32 = 2;
/// Experts each position is routed to.
pub(crate) const ROUTED: usize = 2;
/// The runtime the toy node says it ran, and [`ToyModel`] runs.
pub const RUNTIME: &str = "toy-runtime 1";
/// ggml type of the stored rows: f16.
pub(crate) const CACHE_TYPE: i32 = 1;
const RECEIPT_ID: &str = "toy-turn";

/// A GGUF held in memory.
pub struct Memory(pub Vec<u8>);

impl ByteSource for Memory {
    fn read_at(&self, offset: u64, len: u64) -> Result<Vec<u8>> {
        let start = usize::try_from(offset)?;
        let end = start.saturating_add(usize::try_from(len)?);
        self.0
            .get(start..end)
            .map(<[u8]>::to_vec)
            .context("read past the end")
    }

    fn size(&self) -> u64 {
        self.0.len() as u64
    }
}

/// A weight value: a small multiple of 1/8, different for every tensor.
fn weight_value(tensor: usize, i: usize) -> f32 {
    ((tensor * 31 + i * 17 + 5) % 23) as f32 * 0.125 - 1.25
}

/// The model's tensors: name, dimensions and f32 values.
fn tensors() -> Vec<(String, Vec<u64>, Vec<f32>)> {
    let (h, e, v) = (HIDDEN as u64, EXPERTS as u64, VOCABULARY as u64);
    let mut shapes = Vec::new();
    for layer in 0..LAYERS {
        shapes.push((format!("blk.{layer}.a.weight"), vec![h, h]));
        shapes.push((format!("blk.{layer}.router.weight"), vec![h, e]));
        shapes.push((format!("blk.{layer}.exps.weight"), vec![h, h, e]));
    }
    shapes.push(("output_norm.weight".to_owned(), vec![h]));
    shapes.push(("token_embd.weight".to_owned(), vec![h, v]));
    shapes
        .into_iter()
        .enumerate()
        .map(|(tensor, (name, dims))| {
            let count = dims.iter().product::<u64>() as usize;
            let values = (0..count).map(|i| weight_value(tensor, i)).collect();
            (name, dims, values)
        })
        .collect()
}

fn gguf_string(out: &mut Vec<u8>, text: &str) {
    out.extend((text.len() as u64).to_le_bytes());
    out.extend(text.as_bytes());
}

/// The model's GGUF: its metadata and f32 tensors.
pub(crate) fn gguf_bytes() -> Vec<u8> {
    let tensors = tensors();
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend((tensors.len() as u64).to_le_bytes());
    out.extend(4u64.to_le_bytes());
    gguf_string(&mut out, "general.architecture");
    out.extend(8u32.to_le_bytes());
    gguf_string(&mut out, "toy");
    for (key, value) in [
        ("toy.embedding_length", HIDDEN as u64),
        ("toy.block_count", u64::from(LAYERS)),
    ] {
        gguf_string(&mut out, key);
        out.extend(10u32.to_le_bytes());
        out.extend(value.to_le_bytes());
    }
    gguf_string(&mut out, "toy.attention.layer_norm_rms_epsilon");
    out.extend(12u32.to_le_bytes());
    out.extend(1e-6f64.to_le_bytes());
    let mut offset = 0u64;
    for (name, dims, values) in &tensors {
        gguf_string(&mut out, name);
        out.extend((dims.len() as u32).to_le_bytes());
        for d in dims {
            out.extend(d.to_le_bytes());
        }
        out.extend(0u32.to_le_bytes());
        out.extend(offset.to_le_bytes());
        offset += (values.len() as u64 * 4).div_ceil(32) * 32;
    }
    out.resize(out.len().div_ceil(32) * 32, 0);
    for (_, _, values) in &tensors {
        out.extend(f32_bytes(values));
        out.resize(out.len().div_ceil(32) * 32, 0);
    }
    out
}

/// The model's GGUF, opened.
pub fn weights() -> WeightFile {
    WeightFile::open(Box::new(Memory(gguf_bytes()))).expect("the toy GGUF")
}

/// A GGUF whose header reads and whose weights do not, as a sparse copy
/// whose remote is out of reach.
struct HeaderOnly {
    bytes: Vec<u8>,
    header: u64,
}

impl ByteSource for HeaderOnly {
    // the header is read from the file's start; a weight starts after it
    fn read_at(&self, offset: u64, len: u64) -> Result<Vec<u8>> {
        if offset >= self.header {
            bail!("the claimed GGUF is out of reach");
        }
        Memory(self.bytes.clone()).read_at(offset, len)
    }

    fn size(&self) -> u64 {
        self.bytes.len() as u64
    }

    fn prefetch(&self, ranges: &[(u64, u64)]) -> Result<()> {
        if ranges.iter().any(|(offset, _)| *offset >= self.header) {
            bail!("the claimed GGUF is out of reach");
        }
        Ok(())
    }
}

/// The model's GGUF with its header at hand and its weights out of reach:
/// every weight read fails, which is the client's trouble and not a node's.
pub fn weights_out_of_reach() -> WeightFile {
    let header = weights().header_bytes;
    let source = HeaderOnly {
        bytes: gguf_bytes(),
        header,
    };
    WeightFile::open(Box::new(source)).expect("the toy GGUF's header")
}

/// The residual entering layer 0 for a token at a position.
pub fn entry(token: i32, position: u32) -> Vec<f32> {
    vec![token as f32 * 0.25, position as f32 * 0.5, 1.0, -0.5]
}

/// The token-mixing output of a layer at a position: what attention would
/// give.
fn mixing_output(layer: u32, position: u32) -> Vec<f32> {
    vec![
        0.125 * (position + 1) as f32,
        -0.25,
        0.5 * layer as f32,
        0.0625,
    ]
}

/// A tensor's values, by name.
fn tensor_values(name: &str) -> Vec<f32> {
    tensors()
        .into_iter()
        .find(|(tensor, _, _)| tensor == name)
        .map(|(_, _, values)| values)
        .expect("a tensor of the toy model")
}

/// `rows` rows of `HIDDEN` weights, from `first`, times `x`.
fn matmul(weight: &[f32], first: usize, rows: usize, x: &[f32]) -> Vec<f32> {
    (first..first + rows)
        .map(|row| {
            let w = &weight[row * HIDDEN..(row + 1) * HIDDEN];
            w.iter()
                .zip(x)
                .map(|(&w, &x)| f64::from(w) * f64::from(x))
                .sum::<f64>() as f32
        })
        .collect()
}

/// The experts, best score first.
fn ranked(scores: &[f32]) -> Vec<i32> {
    let mut ids: Vec<i32> = (0..scores.len() as i32).collect();
    ids.sort_by(|&a, &b| scores[b as usize].total_cmp(&scores[a as usize]));
    ids
}

/// The residual leaving a layer, from what entered it and what its matmuls
/// and token mixing gave.
fn residual_out(residual: &[f32], a_out: &[f32], experts_out: &[f32], mixing: &[f32]) -> Vec<f32> {
    (0..HIDDEN)
        .map(|i| {
            let experts: f32 = experts_out.chunks(HIDDEN).map(|chunk| chunk[i]).sum();
            residual[i] + (a_out[i] + experts) * 0.125 + mixing[i]
        })
        .collect()
}

/// One position of one layer, as an honest node computes it.
#[derive(Clone, Debug)]
struct Computed {
    residual_in: Vec<f32>,
    a_out: Vec<f32>,
    router_out: Vec<f32>,
    selection: Vec<i32>,
    experts_out: Vec<f32>,
    mixing_out: Vec<f32>,
    residual_out: Vec<f32>,
}

impl Computed {
    fn new(layer: u32, position: u32, residual_in: Vec<f32>) -> Self {
        let doubled: Vec<f32> = residual_in.iter().map(|v| v * 2.0).collect();
        let a_out = matmul(
            &tensor_values(&format!("blk.{layer}.a.weight")),
            0,
            HIDDEN,
            &doubled,
        );
        let router = tensor_values(&format!("blk.{layer}.router.weight"));
        let router_out = matmul(&router, 0, EXPERTS, &residual_in);
        let selection = ranked(&router_out);
        let experts = tensor_values(&format!("blk.{layer}.exps.weight"));
        let experts_out: Vec<f32> = selection[..ROUTED]
            .iter()
            .flat_map(|&e| matmul(&experts, e as usize * HIDDEN, HIDDEN, &residual_in))
            .collect();
        let mixing_out = mixing_output(layer, position);
        let residual_out = residual_out(&residual_in, &a_out, &experts_out, &mixing_out);
        Self {
            residual_in,
            a_out,
            router_out,
            selection,
            experts_out,
            mixing_out,
            residual_out,
        }
    }

    fn stored(&self) -> Vec<u8> {
        self.residual_in
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
            .collect()
    }

    fn mixing(&self) -> Vec<TensorOpening> {
        let f32_value = |kind, values: &[f32]| {
            TensorOpening::new(kind, 0, Some((GGML_F32, &f32_bytes(values))))
        };
        vec![
            f32_value(MixingKind::Input, &self.residual_in),
            TensorOpening::new(MixingKind::Stored, 0, Some((CACHE_TYPE, &self.stored()))),
            f32_value(MixingKind::Output, &self.mixing_out),
        ]
    }

    fn matrices(&self, layer: u32) -> Vec<MatrixOpening> {
        let doubled: Vec<f32> = self.residual_in.iter().map(|v| v * 2.0).collect();
        let matrix =
            |index, name: &str, router, experts, input: &[f32], output: &[f32]| MatrixOpening {
                index,
                weight: format!("blk.{layer}.{name}.weight"),
                router,
                experts,
                input: encode_f32(input),
                output: encode_f32(output),
            };
        vec![
            matrix(0, "a", false, None, &doubled, &self.a_out),
            matrix(1, "router", true, None, &self.residual_in, &self.router_out),
            matrix(
                2,
                "exps",
                false,
                Some(self.selection[..ROUTED].to_vec()),
                &self.residual_in,
                &self.experts_out,
            ),
        ]
    }
}

/// The auditor's evaluation of the toy model: every layer is at hand, and
/// a layer is computed on the opened values as the native evaluator
/// computes one, taking an unopened value as zeros.
pub struct ToyModel;

/// Column `col` of an opened f32 value of `len` values, or zeros.
fn opened_f32(
    request: &GlueRequest,
    kind: TensorKind,
    index: u32,
    col: usize,
    len: usize,
) -> Vec<f32> {
    let input = request
        .inputs
        .iter()
        .find(|i| i.kind == kind && i.index == index);
    match input.map(|input| (&input.values, input.len)) {
        Some((GlueValues::F32(values), width)) if width == len => values
            .get(col * len..(col + 1) * len)
            .map_or_else(|| vec![0.0; len], <[f32]>::to_vec),
        _ => vec![0.0; len],
    }
}

/// Column `col` of the opened selection, or no ids.
fn opened_selection(request: &GlueRequest, col: usize) -> Vec<i32> {
    let input = request
        .inputs
        .iter()
        .find(|i| i.kind == TensorKind::Selection && i.index == 0);
    match input.map(|input| (&input.values, input.len)) {
        Some((GlueValues::I32(ids), len)) => ids
            .get(col * len..(col + 1) * len)
            .map_or_else(Vec::new, <[i32]>::to_vec),
        _ => Vec::new(),
    }
}

/// Refuses a value the toy layer does not take, as the native evaluator
/// refuses one, in its words: a value no node takes, one of another
/// length, a selected id out of range, a mixing output left out.
fn refuse_values(request: &GlueRequest) -> Result<()> {
    let takes = |kind, index| match (kind, index) {
        (TensorKind::MatmulOutput, 0) | (TensorKind::MixingOutput, 0) => Some(HIDDEN),
        (TensorKind::MatmulOutput, 1) => Some(EXPERTS),
        (TensorKind::MatmulOutput, 2) => Some(ROUTED * HIDDEN),
        // the full order of the experts by score
        (TensorKind::Selection, 0) => Some(EXPERTS),
        _ => None,
    };
    for input in &request.inputs {
        let Some(len) = takes(input.kind, input.index) else {
            bail!(
                "an opened value (kind {}, index {}) matches nothing in the layer",
                input.kind.code(),
                input.index
            );
        };
        if input.len != len {
            bail!(
                "the opened value for {} {} has {} elements per column, the graph {len}",
                input.kind.name(),
                input.index,
                input.len
            );
        }
        if let GlueValues::I32(ids) = &input.values
            && ids
                .iter()
                .any(|&id| !usize::try_from(id).is_ok_and(|id| id < EXPERTS))
        {
            bail!("the opened selection for scores holds an id outside the {EXPERTS} rows");
        }
    }
    if !request
        .inputs
        .iter()
        .any(|input| input.kind == TensorKind::MixingOutput)
    {
        bail!("no opened value for attn_out");
    }
    Ok(())
}

impl GlueEvaluator for ToyModel {
    fn evaluate(&self, request: &GlueRequest) -> Result<GlueOutput> {
        refuse_values(request)?;
        let layer = request.layer;
        let columns = request.positions.len();
        if request.residual.len() != columns * HIDDEN {
            bail!("the residual is not {HIDDEN} values per position");
        }
        let (mut doubled, mut scores, mut routed, mut out) = (vec![], vec![], vec![], vec![]);
        for col in 0..columns {
            let residual = &request.residual[col * HIDDEN..(col + 1) * HIDDEN];
            doubled.extend(residual.iter().map(|v| v * 2.0));
            scores.extend(opened_f32(
                request,
                TensorKind::MatmulOutput,
                1,
                col,
                EXPERTS,
            ));
            let mut ids = opened_selection(request, col);
            ids.resize(ROUTED, 0);
            routed.extend(ids);
            let a_out = opened_f32(request, TensorKind::MatmulOutput, 0, col, HIDDEN);
            let experts_out =
                opened_f32(request, TensorKind::MatmulOutput, 2, col, ROUTED * HIDDEN);
            let mixing = opened_f32(request, TensorKind::MixingOutput, 0, col, HIDDEN);
            out.extend(residual_out(residual, &a_out, &experts_out, &mixing));
        }
        let residual = request.residual.clone();
        let tensor = |kind, index, flags, name: String, len, values| GlueTensor {
            layer: if kind == TensorKind::Residual {
                layer + 1
            } else {
                layer
            },
            kind,
            index,
            flags,
            name,
            len,
            values,
        };
        let weight = |name: &str| format!("blk.{layer}.{name}.weight");
        let f32s = GlueValues::F32;
        Ok(GlueOutput {
            tensors: vec![
                tensor(
                    TensorKind::MatmulInput,
                    0,
                    0,
                    weight("a"),
                    HIDDEN,
                    f32s(doubled),
                ),
                tensor(
                    TensorKind::MatmulInput,
                    1,
                    FLAG_ROUTER,
                    weight("router"),
                    HIDDEN,
                    f32s(residual.clone()),
                ),
                tensor(
                    TensorKind::MatmulInput,
                    2,
                    0,
                    weight("exps"),
                    HIDDEN,
                    f32s(residual.clone()),
                ),
                tensor(
                    TensorKind::ExpertIds,
                    2,
                    0,
                    weight("exps"),
                    ROUTED,
                    GlueValues::I32(routed),
                ),
                tensor(
                    TensorKind::MixingInput,
                    0,
                    0,
                    "q".to_owned(),
                    HIDDEN,
                    f32s(residual.clone()),
                ),
                tensor(
                    TensorKind::MixingStored,
                    0,
                    0,
                    "k".to_owned(),
                    HIDDEN,
                    f32s(residual),
                ),
                tensor(
                    TensorKind::SelectionScores,
                    0,
                    0,
                    "scores".to_owned(),
                    EXPERTS,
                    f32s(scores),
                ),
                tensor(
                    TensorKind::Residual,
                    0,
                    0,
                    "l_out".to_owned(),
                    HIDDEN,
                    f32s(out),
                ),
            ],
            unchecked_weights: Vec::new(),
            attention: Vec::new(),
        })
    }

    fn evaluate_region(&self, _: &GlueRequest) -> Result<GlueOutput> {
        bail!("the toy model keeps no recurrent state")
    }

    fn entry(&self, tokens: &[i32], positions: &[i32]) -> Result<Vec<f32>> {
        Ok(tokens
            .iter()
            .zip(positions)
            .flat_map(|(&token, &position)| entry(token, position as u32))
            .collect())
    }

    fn stored_row_error(
        &self,
        ggml_type: i32,
        stored: &[u8],
        computed: &[f32],
    ) -> Result<StoredRowError> {
        let stored = cache_types::decode_row(ggml_type, stored)?;
        if stored.len() != computed.len() {
            bail!("a stored row of another length than the computed one");
        }
        let l2 = |values: &mut dyn Iterator<Item = f64>| values.map(|v| v * v).sum::<f64>().sqrt();
        let rounded = |v: f32| f64::from(half::f16::from_f32(v).to_f32());
        Ok(StoredRowError {
            stored: l2(&mut stored.iter().zip(computed).map(|(s, &c)| s - f64::from(c))),
            rounding: l2(&mut computed.iter().map(|&c| rounded(c) - f64::from(c))),
            norm: l2(&mut computed.iter().map(|&c| f64::from(c))),
        })
    }
}

/// The best token for the residual leaving the final layer.
fn greedy_token(weights: &WeightFile, head: &OutputHead, residual: &[f32]) -> (i32, Vec<f32>) {
    let logits = head
        .logits(weights, &[residual.to_vec()])
        .expect("the toy head")
        .remove(0);
    let best = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(token, _)| token as i32);
    (best, logits.iter().map(|&l| l as f32).collect())
}

/// A turn of the toy model an honest node served.
pub struct Turn {
    pub receipt: Receipt,
    /// By layer, then by position of the turn.
    computed: Vec<Vec<Computed>>,
    /// The logits at every position of the turn.
    logits: Vec<Vec<f32>>,
    trees: Vec<MerkleTree>,
    /// The cached prefix the turn continued, which an earlier request
    /// computed.
    prefix: Prefix,
    node: SigningKey,
}

/// The cached prefix of a toy turn: what an honest node computed at every
/// position below the turn, and what each stage commits of it. The cell of
/// a position has the position's index.
struct Prefix {
    tokens: Vec<i32>,
    /// By layer, then by position.
    computed: Vec<Vec<Computed>>,
    /// By stage, then by position: the leaves the stage committed.
    leaves: Vec<Vec<[Hash; 3]>>,
    /// By stage: the tree over the prefix leaves. Empty for a turn from
    /// position 0.
    trees: Vec<MerkleTree>,
}

impl Prefix {
    /// The prefix of `length` positions: token `1 + p % 3` at position `p`.
    fn new(length: u32) -> Self {
        let tokens: Vec<i32> = (0..length as i32).map(|p| 1 + p % 3).collect();
        let mut computed: Vec<Vec<Computed>> = vec![Vec::new(); LAYERS as usize];
        for position in 0..length {
            let mut residual = entry(tokens[position as usize], position);
            for layer in 0..LAYERS {
                let values = Computed::new(layer, position, residual);
                residual = values.residual_out.clone();
                computed[layer as usize].push(values);
            }
        }
        let leaves: Vec<Vec<[Hash; 3]>> = (0..LAYERS)
            .map(|layer| {
                (0..length)
                    .map(|p| {
                        let token = (layer == 0).then(|| tokens[p as usize]);
                        position_leaves(layer, p, token, &computed[layer as usize][p as usize])
                    })
                    .collect()
            })
            .collect();
        let trees = (0..LAYERS)
            .filter(|_| length > 0)
            .map(|layer| {
                let leaves = (0..length)
                    .map(|p| {
                        let values = &computed[layer as usize][p as usize];
                        prefix_leaf(
                            p as i32,
                            p,
                            &cell_digest(vec![stored_leaf_of(layer, p, values)]).expect("a cell"),
                            &position_digest(&leaves[layer as usize][p as usize]),
                        )
                    })
                    .collect();
                MerkleTree::new(leaves).expect("a prefix has positions")
            })
            .collect();
        Self {
            tokens,
            computed,
            leaves,
            trees,
        }
    }

    fn len(&self) -> u32 {
        self.tokens.len() as u32
    }

    /// What stage `stage` commits of the prefix.
    fn commitment(&self, stage: u32) -> Option<PrefixCommitment> {
        let tree = self.trees.get(stage as usize)?;
        Some(PrefixCommitment {
            root: hex::encode(tree.root()),
            positions: self.len(),
            tokens: (stage == 0).then(|| hex::encode(prefix_tokens_digest(&self.tokens))),
        })
    }

    /// What stage `stage` opens of the prefix at `position`.
    fn open(&self, stage: u32, position: u32) -> PrefixOpening {
        let values = &self.computed[stage as usize][position as usize];
        let tree = &self.trees[stage as usize];
        PrefixOpening {
            position,
            cell: position,
            leaves: self.leaves[stage as usize][position as usize]
                .iter()
                .map(hex::encode)
                .collect(),
            stored: vec![hex::encode(stored_leaf_of(stage, position, values))],
            path: encode_path(&tree.path(u64::from(position)).expect("a path")),
        }
    }
}

/// The leaf of the row `layer` stored at `position`.
fn stored_leaf_of(layer: u32, position: u32, values: &Computed) -> Hash {
    let stored: Vec<TensorOpening> = values
        .mixing()
        .into_iter()
        .filter(|value| value.kind == MixingKind::Stored)
        .collect();
    stored_leaf(
        position,
        layer,
        &stored_digests(&stored).expect("stored rows"),
    )
}

impl Turn {
    /// A turn from position 0, not bound to a client.
    pub fn new() -> Self {
        Self::starting_at(0)
    }

    /// A turn that continues a cached prefix of `start` positions: three
    /// prompt positions, then two sampled tokens and a final one.
    pub fn starting_at(start: u32) -> Self {
        let weights = weights();
        let head = OutputHead::from_weights(&weights).expect("the toy head");
        let prompt = 3u32;
        let end = start + prompt + 2;
        let mut tokens: Vec<i32> = (0..prompt as i32).map(|i| 1 + i).collect();
        let mut computed: Vec<Vec<Computed>> = vec![Vec::new(); LAYERS as usize];
        let mut logits = Vec::new();
        let mut final_token = None;
        for position in start..end {
            let token = tokens[(position - start) as usize];
            let mut residual = entry(token, position);
            for layer in 0..LAYERS {
                let values = Computed::new(layer, position, residual);
                residual = values.residual_out.clone();
                computed[layer as usize].push(values);
            }
            let (best, at_position) = greedy_token(&weights, &head, &residual);
            logits.push(at_position);
            // the turn samples from the last prompt position on
            if position + 1 == end {
                final_token = Some(best);
            } else if position + 1 >= start + prompt {
                tokens.push(best);
            }
        }
        let trees: Vec<MerkleTree> = (0..LAYERS)
            .map(|layer| stage_tree(layer, start, &tokens, &computed[layer as usize]))
            .collect();
        let prefix = Prefix::new(start);
        let stages = trees
            .iter()
            .enumerate()
            .map(|(i, tree)| StageCommitment {
                stage_index: i as u32,
                layer_start: i as u32,
                layer_end: i as u32 + 1,
                root: hex::encode(tree.root()),
                leaf_count: tree.leaf_count(),
                backend: "CPU".to_owned(),
                cache_types: vec!["f16".to_owned()],
                prefix: prefix.commitment(i as u32),
                ubatches: Vec::new(),
                recurrent_state: None,
            })
            .collect();
        Self {
            receipt: Receipt {
                version: RECEIPT_VERSION,
                receipt_id: RECEIPT_ID.to_owned(),
                model: "toy/model@main/toy.gguf".to_owned(),
                position_start: start,
                position_end: end,
                tokens,
                generation_start: start + prompt,
                final_token,
                stages,
                binding: None,
            },
            computed,
            logits,
            trees,
            prefix,
            node: SigningKey::from_bytes(&[5; 32]),
        }
    }

    /// Binds the receipt to `client_key` with the sampler statement
    /// `sampler`, signed by the toy node.
    #[must_use]
    pub fn bound(mut self, client_key: [u8; 32], sampler: &str) -> Self {
        let request = BindingRequest {
            client_key,
            parent: None,
            runtime: RUNTIME.to_owned(),
            sampler: sampler.to_owned(),
        };
        turn::bind(&mut self.receipt, &request, &self.node);
        self
    }

    /// A challenge of `picks`, each a layer and its positions, and `head`
    /// positions.
    pub fn challenge(&self, picks: &[(u32, &[u32])], head: &[u32]) -> Challenge {
        Challenge {
            receipt_id: self.receipt.receipt_id.clone(),
            picks: picks
                .iter()
                .map(|(layer, positions)| Pick {
                    layer: *layer,
                    positions: positions.to_vec(),
                })
                .collect(),
            proof: None,
            head: head.to_vec(),
            attention: Vec::new(),
        }
    }

    fn at(&self, layer: u32, position: u32) -> &Computed {
        match position.checked_sub(self.receipt.position_start) {
            Some(index) => &self.computed[layer as usize][index as usize],
            None => &self.prefix.computed[layer as usize][position as usize],
        }
    }

    /// The token at `position` of the turn or of its prefix.
    fn token(&self, position: u32) -> i32 {
        self.receipt
            .token_at(position)
            .unwrap_or_else(|| self.prefix.tokens[position as usize])
    }

    /// The path of a leaf of stage `stage` at `position`. A leaf of a
    /// prefix position is in no tree, and has none.
    fn path(&self, stage: u32, position: u32, slot: u64) -> (Hash, Vec<String>) {
        if position < self.receipt.position_start {
            let leaves = &self.prefix.leaves[stage as usize][position as usize];
            return (leaves[slot as usize], Vec::new());
        }
        let tree = &self.trees[stage as usize];
        let index = u64::from(position - self.receipt.position_start) * SLOTS + slot;
        let leaf = tree.leaf(index).expect("a leaf of the turn");
        (leaf, encode_path(&tree.path(index).expect("a path")))
    }

    fn leaf(&self, layer: u32, position: u32) -> LeafOpening {
        let values = self.at(layer, position);
        LeafOpening {
            position,
            residual_in: encode_f32(&values.residual_in),
            residual_out: Some(encode_f32(&values.residual_out)),
            residual_in_path: self.path(layer, position, 0).1,
            mixing_path: self.path(layer, position, 1).1,
            residual_out_path: self.path(layer, position, 2).1,
            mixing: values.mixing(),
            matrices: values.matrices(layer),
            selections: vec![SelectionOpening {
                index: 0,
                ids: values.selection.clone(),
            }],
        }
    }

    fn boundary(&self, stage: u32, position: u32, slot: u64) -> BoundaryLeaf {
        let (leaf, path) = self.path(stage, position, slot);
        BoundaryLeaf {
            position,
            leaf: hex::encode(leaf),
            path,
        }
    }

    /// What an honest node opens for `challenge`.
    pub fn open(&self, challenge: &Challenge) -> Opening {
        let positions = challenge.positions();
        let stages = (0..LAYERS)
            .map(|stage| {
                let picks = challenge.picks.iter().filter(|p| p.layer == stage);
                let layers = picks
                    .map(|pick| LayerOpening {
                        layer: pick.layer,
                        leaves: pick
                            .positions
                            .iter()
                            .map(|&p| self.leaf(pick.layer, p))
                            .collect(),
                    })
                    .collect();
                let boundaries = |slot: u64| -> Vec<BoundaryLeaf> {
                    positions
                        .iter()
                        .map(|&p| self.boundary(stage, p, slot))
                        .collect()
                };
                StageOpening {
                    stage_index: stage,
                    layers,
                    anchors: positions
                        .iter()
                        .filter(|_| stage == 0)
                        .map(|&p| self.anchor(p))
                        .collect(),
                    entry: if stage > 0 { boundaries(0) } else { Vec::new() },
                    exit: boundaries(2),
                    head: challenge
                        .head
                        .iter()
                        .filter(|_| stage + 1 == LAYERS)
                        .map(|&p| self.head(p))
                        .collect(),
                    attention: Vec::new(),
                    recurrent: Vec::new(),
                    prefix: challenge
                        .prefix_positions(self.receipt.position_start)
                        .into_iter()
                        .map(|p| self.prefix.open(stage, p))
                        .collect(),
                    prefix_tokens: if stage == 0 {
                        self.prefix.tokens.clone()
                    } else {
                        Vec::new()
                    },
                }
            })
            .collect();
        Opening {
            receipt_id: challenge.receipt_id.clone(),
            stages,
        }
    }

    fn anchor(&self, position: u32) -> AnchorOpening {
        AnchorOpening {
            position,
            token: self.token(position),
            residual: encode_f32(&self.at(0, position).residual_in),
            path: self.path(0, position, 0).1,
        }
    }

    fn head(&self, position: u32) -> HeadOpening {
        let index = (position - self.receipt.position_start) as usize;
        HeadOpening {
            position,
            residual: encode_f32(&self.at(LAYERS - 1, position).residual_out),
            path: self.path(LAYERS - 1, position, 2).1,
            logits: Some(encode_f32(&self.logits[index])),
        }
    }

    /// The toy node's signature of an opening of `challenge`. The receipt
    /// must be bound.
    pub fn sign_opening(&self, challenge: &Challenge, opening_bytes: &[u8]) -> String {
        let bound = turn::verify(&self.receipt, None).expect("a bound receipt");
        turn::sign_opening(&self.node, &bound.turn_root, challenge, opening_bytes)
    }

    /// The toy node's signature of attention rows it opened for `blocks`.
    pub fn sign_rows(&self, blocks: &BlockChallenge, rows_bytes: &[u8]) -> String {
        let bound = turn::verify(&self.receipt, None).expect("a bound receipt");
        turn::sign_rows(&self.node, &bound.turn_root, blocks, rows_bytes)
    }
}

impl Default for Turn {
    fn default() -> Self {
        Self::new()
    }
}

/// Leaves per position of a stage of one layer.
const SLOTS: u64 = 3;

/// The tree of the stage holding `layer`: per position, the residual in,
/// the mixing values and the residual out.
fn stage_tree(layer: u32, start: u32, tokens: &[i32], computed: &[Computed]) -> MerkleTree {
    debug_assert_eq!(slots_per_position(layer, layer + 1), SLOTS);
    let mut leaves = Vec::new();
    for (i, values) in computed.iter().enumerate() {
        let position = start + i as u32;
        let token = (layer == 0).then(|| tokens[i]);
        leaves.extend(position_leaves(layer, position, token, values));
    }
    MerkleTree::new(leaves).expect("a turn has positions")
}

/// The leaves the stage holding `layer` commits at one position: the
/// residual in, the mixing values and the residual out.
fn position_leaves(layer: u32, position: u32, token: Option<i32>, values: &Computed) -> [Hash; 3] {
    let mixing = values.mixing();
    let decoded = crate::wire::mixing_bytes(&mixing).expect("mixing values");
    let mixing: Vec<MixingValue<'_>> = crate::wire::mixing_values(&decoded);
    [
        residual_leaf(
            position,
            layer,
            token,
            Some(&f32_bytes(&values.residual_in)),
        ),
        mixing_leaf(position, layer, &mixing),
        residual_leaf(
            position,
            layer + 1,
            None,
            Some(&f32_bytes(&values.residual_out)),
        ),
    ]
}
