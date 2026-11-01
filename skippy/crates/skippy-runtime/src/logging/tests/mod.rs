mod aggregator;
mod parser_policy;

#[test]
fn native_error_line_travels_with_the_open_error() {
    super::clear_last_native_error();
    super::record_native_error_line(3, b"llama_model_loader: loaded meta data\n");
    super::record_native_error_line(
        super::NATIVE_LOG_LEVEL_ERROR,
        b"llama_model_load: error loading model: tensor data is not within the file bounds\n",
    );
    super::record_native_error_line(
        super::NATIVE_LOG_LEVEL_ERROR,
        b"llama_model_load_from_file_impl: failed to load model\n",
    );

    let error =
        super::with_last_native_error(anyhow::anyhow!("ModelError: failed to load llama model"));

    let text = format!("{error:#}");
    assert!(text.contains("failed to load llama model"), "{text}");
    assert!(text.contains("not within the file bounds"), "{text}");
    let again = super::with_last_native_error(anyhow::anyhow!("later failure"));
    assert_eq!(format!("{again:#}"), "later failure");
}
