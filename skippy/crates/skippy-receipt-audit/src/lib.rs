//! The audit of an inference receipt against a claimed GGUF, on the
//! client's own machine.
//!
//! A client that received a turn and its receipt gives both to
//! [`turn_audit::audit_turn`], with the claimed model
//! ([`turn_audit::ClaimedSetup`]) and a way to send the challenge to the
//! node ([`turn_audit::OpenChallenge`]). The audit draws the challenge from
//! the VRF of the turn root, checks the node's opening against the claimed
//! GGUF, and ends one of four ways ([`outcome::Outcome`]).
//! [`recheck::recheck_evidence`] runs the same checks again on the
//! evidence an audit kept.
//!
//! The crate sends no chat request and prints nothing. `skippy-commands`
//! does both for `skippy audit-receipt` and `skippy recheck-receipt`, and
//! a Mesh node audits the turns its peers served.

pub mod attention_audit;
pub mod chat_reply;
pub mod claimed_file;
pub mod claimed_model;
mod head_key_file;
pub mod native_model;
mod node_text;
mod opening_checks;
pub mod outcome;
pub mod output;
pub mod recheck;
pub mod recurrent_audit;
pub mod report_text;
mod token_check;
pub mod turn_audit;
