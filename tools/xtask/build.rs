use std::path::PathBuf;

fn main() {
    // Cargo sets OUT_DIR under <profile>/build/ for this compilation, including
    // --target, JSON targets and configured profiles: <package-hash>/out before
    // Rust 1.100 and <package>/<hash>/out since. The profile directory is the
    // parent of that `build` component in either layout.
    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("Cargo sets OUT_DIR"));
    let profile = out
        .ancestors()
        .find(|path| path.file_name().is_some_and(|name| name == "build"))
        .and_then(|build| build.parent())
        .expect("Cargo build output layout");
    println!(
        "cargo:rustc-env=XTASK_CARGO_PROFILE_DIR={}",
        profile.display()
    );
    println!("cargo:rerun-if-changed=build.rs");
}
