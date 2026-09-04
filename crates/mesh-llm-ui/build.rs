use std::fs;
use std::path::Path;

fn main() {
    // Keep Cargo's default package change detection. Watching `dist`
    // explicitly makes a missing directory look changed on every invocation,
    // so test loops rebuild this crate and every dependent crate indefinitely.
    // The default still notices when `dist` appears or its contents change.
    configure_console_dist();
}

fn configure_console_dist() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("cargo manifest dir");
    let console_dist = Path::new(&manifest_dir).join("dist");

    if console_dist.is_dir() {
        println!(
            "cargo:rustc-env=MESH_LLM_UI_DIST={}",
            console_dist.display()
        );
        return;
    }

    let fallback =
        Path::new(&std::env::var("OUT_DIR").expect("cargo out dir")).join("empty-ui-dist");
    fs::create_dir_all(&fallback).expect("create fallback UI dist dir");
    println!("cargo:rustc-env=MESH_LLM_UI_DIST={}", fallback.display());
}
