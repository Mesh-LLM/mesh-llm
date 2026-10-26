use std::io::Write as _;
fn main() {
    if skippy_model_package::hf_checkpoint_stitch::run(
        &std::env::args().skip(1).collect::<Vec<_>>(),
    )
    .is_err()
    {
        let _ = writeln!(
            std::io::stderr().lock(),
            "MTP checkpoint staging refused; inspect owned receipt"
        );
        std::process::exit(1);
    }
}
