//! Typed bounded HF projector transport.
mod resolver;
#[cfg(test)]
mod tests;
mod transfer;
mod url_policy;
use crate::{automation::command_interrupt::Interrupt, command::DynResult};

pub(super) fn run(args: &[String]) -> DynResult<()> {
    if args == ["--help"] {
        println!(
            "models projector-download --url HTTPS_URL --output FILE; bounded pinned HF HTTPS transfer"
        );
        return Ok(());
    }
    if let [verb, rest @ ..] = args
        && verb == "resolve-worker"
    {
        return resolver::worker(rest);
    }
    let [url_flag, url, output_flag, output] = args else {
        return Err("models projector-download --url HTTPS_URL --output FILE".into());
    };
    if url_flag != "--url" || output_flag != "--output" {
        return Err("invalid projector download options".into());
    }
    let output_path = std::path::PathBuf::from(output);
    let output_path = if output_path.is_absolute() {
        output_path
    } else {
        std::env::current_dir()?.join(output_path)
    };
    let interrupt = Interrupt::install()?;
    let result = transfer::download(url, &output_path, &interrupt.cancellation());
    interrupt.finish()?;
    println!("{}", serde_json::json!({"sha256":result?,"output":output}));
    Ok(())
}
