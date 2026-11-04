//! Verifiable inference receipts.
//!
//! A node that serves with receipts commits, per stage and per turn, to the
//! residual stream entering every layer and to each layer's token-mixing
//! values (attention's inputs, stored KV rows and output), with a Merkle root
//! over those leaves. It keeps a small replay log instead of the values.
//! The residual entering layer 0 binds each position's token. After the
//! turn, a client challenges a few layers at a few positions each. The node
//! replays the turn, opens those leaves, and adds the layers' weight matmuls
//! and selections, the input of layer 0, and the leaves at stage
//! boundaries. The client checks the receipt against the model and request
//! and the leaves against the roots ([`binding`]), the matmuls against rows
//! of the claimed GGUF's weights ([`audit`]), and the glue between them by
//! computing each layer and the input of layer 0 itself on the opened values
//! ([`glue`]; see `mesh/docs/design/RECEIPTS.md` and `RECEIPTS_GLUE.md`),
//! and the sampled tokens against the output head on the final residual
//! ([`head`]).
//!
//! An opening is a node's free text until it is checked. So the client
//! first holds it to the one form an answer has and to its Merkle roots
//! ([`binding`]), then every value in it to the claimed model's shapes
//! ([`opened_values`]), and only then compares anything. A check that stops
//! says whose fault that is ([`node_fault`]): a defect of what the node
//! sent fails the audit, and a check the client could not run says nothing
//! about the node.
//!
//! # Layers
//!
//! The crate has three layers. Each module below is in one of them.
//!
//! - **Shared.** The format of a receipt and the arithmetic that both
//!   sides must do the same way. A shared module uses only shared modules.
//!   [`wire`], [`commitment`], [`turn`], [`vrf`], `draw`, [`challenge`]
//!   and [`layer_tensors`] are the format. [`attention`] is the layout of
//!   a layer's attention and the partial of a block of it. `ggml_types`,
//!   `dequant`, [`cache_types`] and [`activation_rounding`] decode and
//!   round values. [`node_fault`] marks an error that a defective value
//!   causes. The shared decoders set the mark, and only a client reads it.
//! - **Node.** [`recorder`] records the leaves of each ubatch, commits a
//!   turn and opens a challenge by replay. It uses shared modules only.
//! - **Client.** The checks of an opening, and the reader of the claimed
//!   GGUF: [`binding`], [`opened_values`], [`audit`], [`glue`], [`head`],
//!   [`attention_check`], [`recurrent`], [`coverage`], [`evidence`],
//!   [`sampler_request`], `error_bound`, [`weights`], [`shards`],
//!   [`sparse_copy`] and [`claimed_reads`]. A client module uses shared and
//!   client modules. It never uses [`recorder`].
//!
//! `draw`, `ggml_types`, `dequant` and `error_bound` are private to the
//! crate. No other crate uses them.
//!
//! `toy_turn`, behind the `test-support` feature, and the crate's own
//! tests use all three layers.
//!
//! `skippy-serving` uses the shared and node layers. The auditor in
//! `skippy-receipt-audit` and the Mesh host use the shared and client layers.
//!
//! `tests/layers.rs` reads the `// Layer:` comments below and every file
//! under `src/`. It fails when a module names a module of a layer that it
//! may not use. To add a module, put its `mod` line under the comment of
//! its layer.

// Layer: shared
//
// The format of a receipt, and the arithmetic that a node and a client
// must do the same way. Uses shared modules only.
pub mod activation_rounding;
pub mod attention;
pub mod cache_types;
pub mod challenge;
pub mod commitment;
pub(crate) mod dequant;
pub(crate) mod draw;
pub(crate) mod ggml_types;
pub mod layer_tensors;
pub mod node_fault;
pub mod turn;
pub mod vrf;
pub mod wire;

// Layer: node
//
// What a serving stage runs: record, commit, open by replay. Uses shared
// modules only.
pub mod recorder;

// Layer: client
//
// What an auditor runs: the checks of an opening against the receipt and
// the claimed GGUF. Uses shared and client modules, never `recorder`.
pub mod attention_check;
pub mod audit;
pub mod binding;
pub mod claimed_reads;
pub mod coverage;
pub(crate) mod error_bound;
pub mod evidence;
pub mod glue;
pub mod head;
pub mod opened_values;
pub mod recurrent;
pub mod sampler_request;
pub mod shards;
pub mod sparse_copy;
pub mod weights;

// Layer: test support
//
// A toy model and one honest turn of it, for the tests of this crate and
// of the crates that audit receipts. Uses every layer.
#[cfg(any(test, feature = "test-support"))]
pub mod toy_turn;
