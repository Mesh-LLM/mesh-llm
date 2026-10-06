use super::*;

#[test]
fn unavailable_native_runtime_rejects_local_model_before_reading_source() {
    if skippy_runtime::native_runtime_loaded() {
        return;
    }
    let missing = Path::new("/missing/issue1204-local-model.gguf");
    let options = SkippyModelLoadOptions::for_direct_gguf("fixture/model", missing);
    let error = match SkippyModelHandle::load(options) {
        Ok(_) => panic!("local load must fail without a native runtime"),
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("require a MeshLLM native runtime")
    );
    let error = infer_layer_count(missing).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("require a MeshLLM native runtime")
    );
    let error = match load_laya_model(missing, None) {
        Ok(_) => panic!("Laya load must fail without a native runtime"),
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("require a MeshLLM native runtime")
    );
}
