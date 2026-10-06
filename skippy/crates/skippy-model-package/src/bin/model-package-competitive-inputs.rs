use std::io::Write as _;
fn main() {
    if skippy_model_package::competitive_acquisition::run(&std::env::args().skip(1).collect::<Vec<_>>())
        .is_err()
    {
        let _ = writeln!(
            mesh_llm_events::console_err(),
            "competitive input acquisition/export refused; inspect owned receipt"
        );
        std::process::exit(1);
    }
}
