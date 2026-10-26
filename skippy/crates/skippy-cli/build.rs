use std::env;
use std::fs;
use std::path::Path;

// Embed the selected source revision through generated source so the value
// participates in the preprocessed crate bytes: `option_env!` reads are
// invisible to the compiler-cache key and could embed a stale revision.
fn main() {
    println!("cargo:rerun-if-env-changed=SKIPPY_SOURCE_SHA");
    let value = env::var("SKIPPY_SOURCE_SHA")
        .ok()
        .filter(|value| !value.is_empty());
    let rendered = match &value {
        Some(sha) => format!("Some({sha:?})"),
        None => "None".to_owned(),
    };
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR is set for build scripts");
    fs::write(
        Path::new(&out_dir).join("source_sha.rs"),
        format!("pub(crate) const SKIPPY_SOURCE_SHA: Option<&'static str> = {rendered};\n"),
    )
    .expect("write generated source revision");
}
