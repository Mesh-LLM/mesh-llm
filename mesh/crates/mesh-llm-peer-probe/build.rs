use std::{env, fs, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=controller.wat");
    let wasm = wat::parse_file("controller.wat").expect("valid bounded probe controller");
    let output = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    fs::write(output.join("controller.wasm"), wasm).expect("write compiled WASM controller");
}
