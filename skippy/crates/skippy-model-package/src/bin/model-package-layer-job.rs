use std::io::Write as _;
fn main() {
    if skippy_model_package::layer_job::run().is_err() {
        let _ = writeln!(
            mesh_llm_events::console_err(),
            "layer job operation incomplete; inspect receipts for confirmed or uncertain mutation evidence"
        );
        std::process::exit(1);
    }
}
