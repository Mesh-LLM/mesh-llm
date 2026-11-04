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

pub mod attention;
pub mod audit;
pub mod binding;
pub mod challenge;
pub mod claimed_reads;
pub mod commitment;
pub mod coverage;
pub mod dequant;
pub mod draw;
pub mod evidence;
pub mod glue;
pub mod head;
pub mod layer_tensors;
pub mod recorder;
pub mod recurrent;
pub mod shards;
pub mod sparse_copy;
pub mod turn;
pub mod vrf;
pub mod weights;
pub mod wire;
