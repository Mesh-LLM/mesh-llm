fn main() {
    match skippy_model_package::snapshot_promotion::local_publisher::run(
        &mut std::io::stdout().lock(),
    ) {
        Ok(true) => (),
        Ok(false) => std::process::exit(1),
        Err(_) => {
            eprintln!("publisher refused; no successful publication receipt");
            std::process::exit(1);
        }
    }
}
