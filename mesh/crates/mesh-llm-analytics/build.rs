fn main() {
    // `option_env!("MESH_LLM_POSTHOG_KEY")` is resolved at compile time, so
    // without this the crate would not rebuild when the release pipeline sets
    // or changes the key, and a stale object would ship keyless.
    println!("cargo:rerun-if-env-changed=MESH_LLM_POSTHOG_KEY");
}
