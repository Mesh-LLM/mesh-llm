use std::io::Write as _;
#[cfg(unix)]
fn main() {
    match skippy_model_package::jobs::delivery::generic_cli::run() {
        Ok(true) => (),
        Ok(false) | Err(_) => {
            let _ = writeln!(
                std::io::stderr().lock(),
                "generic Jobs operation incomplete; inspect local receipts and authorized Jobs status; remote acceptance or completion may remain unconfirmed"
            );
            std::process::exit(1);
        }
    }
}
#[cfg(not(unix))]
fn main() {
    let _ = writeln!(
        std::io::stderr().lock(),
        "generic Jobs facade safe local file/credential admission requires Unix"
    );
    std::process::exit(1);
}
