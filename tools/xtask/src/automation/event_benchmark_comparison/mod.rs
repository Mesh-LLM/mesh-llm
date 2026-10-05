//! Paired event benchmark comparisons with independent certification gates.
mod callback_latency;
mod command;
mod environment_identity;
#[cfg(test)]
mod fixture;
mod gates;
mod health;
mod manifest;
mod options;
mod order_identity;
mod pairing;
mod prompt_identity;
mod report;
mod resampling;
mod retry;
mod screening;
mod statistics;

pub(crate) use command::run;
