fn main() {
    println!("cargo:rerun-if-env-changed=MESH_LLM_BUILD_VERSION");
}
