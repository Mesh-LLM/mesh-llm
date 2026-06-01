#![forbid(unsafe_code)]

use clap::ValueEnum;

pub mod output;

pub use output::*;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
pub enum LogFormat {
    #[default]
    Pretty,
    Json,
}
