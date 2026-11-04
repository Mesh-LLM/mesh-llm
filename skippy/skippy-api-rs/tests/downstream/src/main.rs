//! A consumer with no build script and no knowledge of native loader paths.
fn main() {
    let runtime = skippy::Runtime::new().expect("the downstream binary initializes native backends");
    assert!(runtime.plan_stages("/nonexistent/skippy-downstream-smoke.gguf", 1).is_err());
}
