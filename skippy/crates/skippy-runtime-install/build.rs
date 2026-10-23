fn main() {
    // `types.rs` reads this with `option_env!`; rebuild when it changes so a
    // packager toggling it never gets a stale binary.
    println!("cargo:rerun-if-env-changed=MESH_LLM_TRUST_SHIPPED_RUNTIMES");
}
