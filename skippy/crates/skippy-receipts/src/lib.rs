//! Verifiable inference receipts.
//!
//! A node that serves with receipts commits, per stage and per turn, to the
//! residual stream entering every layer and to each layer's attention
//! projections, with a Merkle root over those leaves. It keeps a small replay
//! log instead of the values. After the turn, a client challenges one layer
//! and a few tokens. The node replays the turn, opens those leaves, and adds
//! the layer's weight matmuls. The client checks the leaves against the root,
//! the matmuls against random rows of the claimed GGUF's weights, and the
//! glue between the matmuls and the residual stream (see
//! `mesh/docs/design/RECEIPTS.md`).

pub mod audit;
pub mod challenge;
pub mod commitment;
pub mod glue;
pub mod recorder;
pub mod weights;
pub mod wire;
