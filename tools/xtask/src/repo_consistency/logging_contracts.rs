//! Authored documentation and parsed Rust ownership contracts; no runtime changes.
#[path = "logging_contracts/boundaries.rs"]
mod boundaries;
#[path = "logging_contracts/documentation.rs"]
mod documentation;

fn source(relative: &str) -> String {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    std::fs::read_to_string(root.join(relative))
        .unwrap_or_else(|error| panic!("cannot read contract source {relative}: {error}"))
}
