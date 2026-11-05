use std::path::PathBuf;

fn main() {
    // Cargo sets OUT_DIR to <profile>/build/<package-hash>/out for this
    // compilation, including --target, JSON targets and configured profiles.
    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("Cargo sets OUT_DIR"));
    let profile = out.ancestors().nth(3).expect("Cargo build output layout");
    println!(
        "cargo:rustc-env=XTASK_CARGO_PROFILE_DIR={}",
        profile.display()
    );
    println!("cargo:rerun-if-changed=build.rs");
}
