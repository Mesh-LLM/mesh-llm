use std::io::Write as _;
fn main() {
    if skippy_model_package::layer_job::run(&mut std::io::stdout().lock()).is_err() {
        let _ = writeln!(
            std::io::stderr().lock(),
            "layer job operation incomplete; inspect receipts for confirmed or uncertain mutation evidence"
        );
        std::process::exit(1);
    }
}
