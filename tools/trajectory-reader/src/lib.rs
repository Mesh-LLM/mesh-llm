//! Portable trajectory selection contracts. Parquet and its compression codecs
//! are compiled only for the explicitly enabled optional reader.
pub mod selection;
pub mod wire;

#[cfg(feature = "parquet-input")]
pub mod parquet_input;

pub type DynResult<T> = Result<T, Box<dyn std::error::Error>>;

#[cfg(all(test, feature = "parquet-input"))]
mod parquet_tests;

#[cfg(feature = "parquet-input")]
pub mod cohorts;

#[cfg(feature = "corpus-input")]
pub mod corpus;
