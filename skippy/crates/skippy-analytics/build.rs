fn main() {
    // `option_env!` is resolved at compile time, so without these the crate
    // would not rebuild when the release pipeline sets or changes them, and a
    // stale object would ship keyless or with the wrong version.
    println!("cargo:rerun-if-env-changed=SKIPPY_POSTHOG_KEY");
    println!("cargo:rerun-if-env-changed=SKIPPY_BUILD_VERSION");
}
